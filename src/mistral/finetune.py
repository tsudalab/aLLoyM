#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA fine-tuning with Unsloth (Mistral-Nemo-Instruct-2407-bnb-4bit)
Optional W&B logging.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # =========================================================
    # Dataset and sequence length configuration
    # =========================================================
    "jsonl_path": "../training/combined.jsonl", # Training data

    "max_seq_length_margin": 20,  # Additional buffer tokens added to the automatically
                                 # computed maximum sequence length. This helps avoid
                                 # truncation in slightly longer unseen examples.

    "save_max_seq_path": "max_seq_length.txt",  # Path to save the computed maximum sequence length.
                                                # This allows for reproducibility in future runs.

    # =========================================================
    # Base model and tokenizer settings
    # =========================================================
    "base_model_name": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",  # Pretrained model to fine-tune.
                                                                      # The suffix "-bnb-4bit" indicates
                                                                      # it is a 4-bit quantized model for
                                                                      # lower VRAM usage.

    "dtype": "bfloat16",   # Data type used for model weights and computations.
                           # Use "bfloat16" on GPUs such as A100 or H100 for numerical
                           # stability and better speed. Use "float16" if bf16 is unsupported.

    "load_in_4bit": True,  # Whether to load the model in 4-bit precision to reduce VRAM usage.
                           # Useful for large models and smaller GPUs.

    # =========================================================
    # Prompt formatting (how each training example is constructed)
    # =========================================================
    "prompt_template": (
        "### Instruction:\n{}\n\n"
        "### Input:\n{}\n\n"
        "### Output:\n{}"
    ),
    # Defines the textual structure of each training example:
    # - The first {} is replaced with the instruction or system message.
    # - The second {} is replaced with the user question or input.
    # - The third {} is replaced with the assistant’s output (the desired answer).
    # This consistent structure helps the model learn to respond properly.

    # =========================================================
    # LoRA (Low-Rank Adaptation) fine-tuning configuration
    # =========================================================
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    # List of transformer submodules where LoRA adapters will be applied.
    # These are typically the linear projection layers inside the attention
    # and feed-forward blocks.

    "lora_r": 16,             # Rank of the LoRA decomposition matrices.
                              # Higher values give more trainable capacity but require more memory.

    "lora_alpha": 16,         # Scaling factor that controls the update magnitude in LoRA layers.

    "lora_dropout": 0.0,      # Dropout applied within LoRA layers to improve generalization.
                              # Set to 0.0 to disable.

    "use_gradient_checkpointing": "unsloth",  # Enables Unsloth’s memory-efficient gradient checkpointing.
                                              # Saves VRAM at the cost of slightly longer training time.

    "use_rslora": False,      # Whether to use Rank-Stabilized LoRA (a variant for dynamic rank updates).
                              # Keep False unless specifically experimenting.

    "loftq_config": None,     # Optional LoftQ quantization configuration. None disables it.

    "random_state": 3407,     # Random seed for initializing LoRA layers to ensure reproducibility.

    # =========================================================
    # Training arguments (used by the Hugging Face Trainer)
    # =========================================================
    "bf16": True,  # Use bfloat16 precision for training. Improves performance and stability on modern GPUs.

    "group_by_length": True,  # Group examples of similar lengths together to improve training efficiency
                              # and reduce padding overhead.

    "per_device_train_batch_size": 16,  # Number of samples per GPU per step.
                                        # Adjust based on available VRAM.

    "gradient_accumulation_steps": 4,   # Number of steps to accumulate gradients before each optimizer update.
                                        # Effective batch size = batch_size * gradient_accumulation_steps.

    "num_train_epochs": 1,  # Total number of epochs (full passes over the dataset).

    "warmup_steps": 1500,   # Number of steps for linear learning-rate warmup from 0 to target LR.

    "learning_rate": 2e-4,  # Base learning rate. LoRA fine-tuning typically uses a higher LR
                            # than full fine-tuning since fewer parameters are being trained.

    "logging_steps": 10,    # Frequency (in steps) of logging metrics such as loss and learning rate.

    "optim": "adamw_8bit",  # Optimizer type. "adamw_8bit" uses bitsandbytes to reduce memory usage
                            # compared to standard AdamW.

    # =========================================================
    # Logging and reproducibility
    # =========================================================
    "report_to": "wandb",      # Reporting backend. Set to "wandb" to enable Weights & Biases logging.
                              # Set to "none" to disable all external logging.

    "wandb_project": "aLLoyM",     # Name of the W&B project to which logs are sent (if enabled).

    "wandb_run_name": "finetuned_model",  # Display name for the run on W&B (if enabled).

    "seed": 3407,  # Global random seed for Python, NumPy, and PyTorch to ensure deterministic results.
}

# =============================================================================
# Imports
# =============================================================================
import os, json
from typing import List, Dict, Any
import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# =============================================================================
# Environment setup and optional W&B
# =============================================================================
def setup_env_and_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_tokens_from_file(path: str = ".env.tokens") -> None:
    """
    Load tokens (HF_TOKEN, WANDB_API_KEY, etc.) from a simple key=value file
    and inject them into os.environ.
    Lines starting with '#' are ignored.
    """
    if not os.path.isfile(path):
        print(f"[warn] Token file '{path}' not found — skipping.")
        return

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"[warn] Ignoring malformed line: {line}")
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
    print(f"[info] Loaded tokens from {path}")

def env_login_if_available(enable_wandb: bool):
    """Log in to HF and optionally to W&B (only if enabled and key present)."""
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        login(hf_token)

    if enable_wandb:
        wandb_api = os.getenv("WANDB_API_KEY", "").strip()
        if wandb_api:
            import wandb
            wandb.login(key=wandb_api)
            return True
        else:
            print("[warn] WANDB_API_KEY not found — skipping W&B logging.")
    return False

# =============================================================================
# Dataset loading and formatting
# =============================================================================
def load_jsonl_messages(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and extract Instruction, Question, Answer fields."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            msgs = data.get("messages", [])
            sys_ = next((m["content"] for m in msgs if m["role"] == "system"), "")
            usr_ = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_ = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            records.append({"Instruction": sys_, "Question": usr_, "Answer": asst_})
    return records

def build_prompt(instr, inp, out, template, eos): return template.format(instr, inp, out) + eos

def compute_max_seq_length(data, tok, template, eos):
    """Tokenize all samples to determine true max length."""
    lengths = []
    for d in data:
        text = build_prompt(d["Instruction"], d["Question"], d["Answer"], template, eos)
        ids = tok(text, truncation=False, return_tensors="pt").input_ids
        lengths.append(ids.shape[1])
    return max(lengths) if lengths else 0

def dataset_from_records(records, template, eos) -> Dataset:
    """Map Instruction/Question/Answer into 'text' field."""
    ds = Dataset.from_list(records)
    def _map_fn(batch):
        return {"text": [build_prompt(i, q, a, template, eos)
                         for i, q, a in zip(batch["Instruction"], batch["Question"], batch["Answer"])]}
    return ds.map(_map_fn, batched=True)

# =============================================================================
# Model setup
# =============================================================================
def load_model_and_tokenizer(model_name, max_seq_len, dtype, load_in_4bit):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    return model, tok

def apply_lora(model, cfg, max_seq_length):
    """Attach LoRA adapters."""
    return FastLanguageModel.get_peft_model(
        model,
        target_modules=cfg["target_modules"],
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
        random_state=cfg["random_state"],
        max_seq_length=max_seq_length,
        use_rslora=cfg["use_rslora"],
        loftq_config=cfg["loftq_config"],
    )

# =============================================================================
# Training
# =============================================================================
def build_trainer(model, tok, ds, max_seq_length, cfg):
    args = TrainingArguments(
        bf16=cfg["bf16"],
        group_by_length=cfg["group_by_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        output_dir=cfg["output_dir"],
        optim=cfg["optim"],
        report_to=("wandb" if cfg["report_to"] == "wandb" else "none"),
        seed=cfg["seed"],
    )
    return SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=args,
    )

# =============================================================================
# Main
# =============================================================================
def main(cfg):
    """
    Main fine-tuning routine:
      - Seeds and environment setup
      - Loads tokens from .env.tokens
      - Logs into Hugging Face and optionally Weights & Biases
      - Loads data and computes max sequence length
      - Prepares dataset, model, LoRA adapters
      - Trains and saves results
    """
    # 1) Reproducibility setup
    setup_env_and_seed(cfg["seed"])

    # 2) Load local tokens (HF_TOKEN, WANDB_API_KEY)
    load_env_tokens(".env.tokens")

    # 3) Hugging Face login
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        login(hf_token)

    # 4) Try initializing wandb if possible, ignore all errors
    try:
        import wandb
        wandb_api = os.getenv("WANDB_API_KEY", "").strip()
        if wandb_api:
            wandb.login(key=wandb_api)
            wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"])
            print("[info] Weights & Biases logging enabled.")
        else:
            print("[info] WANDB_API_KEY not found — skipping W&B logging.")
    except Exception:
        # If wandb is not installed or login fails, silently skip
        pass

    # 5) Prepare output paths
    if not os.path.isfile(cfg["jsonl_path"]):
        raise FileNotFoundError(f"Dataset not found: {cfg['jsonl_path']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["save_dir_final"], exist_ok=True)

    # 6) Load temporary tokenizer to get EOS token and measure sequence length
    tmp_tok = AutoTokenizer.from_pretrained(cfg["base_model_name"])
    eos_token = tmp_tok.eos_token or "</s>"

    # 7) Load and parse dataset
    records = load_jsonl_messages(cfg["jsonl_path"])

    # 8) Compute max sequence length (with margin)
    auto_len = compute_max_seq_length(records, tmp_tok, cfg["prompt_template"], eos_token)
    max_seq_length = auto_len + cfg["max_seq_length_margin"]
    print(f"[info] Computed max_seq_length: {auto_len} (+{cfg['max_seq_length_margin']}) → {max_seq_length}")

    with open(cfg["save_max_seq_path"], "w") as f:
        f.write(str(max_seq_length))

    # 9) Build formatted dataset
    dataset = dataset_from_records(records, cfg["prompt_template"], eos_token)

    # 10) Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model_name"],
        max_seq_length,
        cfg["dtype"],
        cfg["load_in_4bit"],
    )

    # 11) Apply LoRA adapters
    model = apply_lora(model, cfg, max_seq_length)

    # 12) Build training configuration
    training_args = TrainingArguments(
        bf16=cfg["bf16"],
        group_by_length=cfg["group_by_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        output_dir=cfg["output_dir"],
        optim=cfg["optim"],
        report_to="wandb",  # Always try to report to wandb if available
        seed=cfg["seed"],
    )

    # 13) Trainer setup
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    # 14) Train the model
    trainer.train()

    # 15) Save final adapters and tokenizer
    model.save_pretrained(cfg["save_dir_final"])
    tokenizer.save_pretrained(cfg["save_dir_final"])

    # 16) Free CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[done] Training complete. Model saved to {cfg['save_dir_final']}")

if __name__ == "__main__":
    main(CONFIG)