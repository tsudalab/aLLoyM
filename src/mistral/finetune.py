#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA fine-tuning with Unsloth (Mistral-Nemo-Instruct-2407-bnb-4bit)
Compatible with TRL==0.9.6 (uses tokenizer= and TrainingArguments).
Relies solely on already-sourced environment variables:
  - HF_TOKEN (required to push/pull private models)
  - WANDB_API_KEY (optional)
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # ================= Dataset =================
    "jsonl_path": "../training/combined.jsonl",  
    # Path to the dataset in JSONL format (each line = one training sample with messages or QA pairs).

    "max_seq_length_margin": 20,  
    # Extra tokens added to the computed maximum sequence length to avoid truncation on slightly longer examples.

    "save_max_seq_path": "max_seq_length.txt",  
    # Path to store the calculated maximum sequence length for reproducibility across runs.

    # ================= Model ===================
    "base_model_name": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",  
    # Name of the base model to fine-tune. The suffix "-bnb-4bit" indicates it uses 4-bit quantization to reduce VRAM usage.

    "dtype": "bfloat16",  
    # Preferred tensor precision (bfloat16). Will automatically downgrade to float16 if the hardware doesn’t support bf16.

    "load_in_4bit": True,  
    # Enables loading the model in 4-bit precision mode for efficient fine-tuning on limited VRAM GPUs.

    # ================= Prompt ==================
    "prompt_template": (
        "### Instruction:\n{}\n\n"
        "### Input:\n{}\n\n"
        "### Output:\n{}"
    ),  
    # Defines how training data is formatted before being fed to the model.
    # {} placeholders correspond to instruction, input, and expected output text fields.

    # ================= LoRA ====================
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],  
    # Specifies which layers in the transformer architecture will receive LoRA adapters.
    # These correspond to attention and feed-forward projection layers.

    "lora_r": 16,  
    # Rank (low-rank dimension) of the LoRA matrices — controls how much capacity LoRA adds.

    "lora_alpha": 16,  
    # Scaling factor for LoRA updates — higher values strengthen LoRA’s effect.

    "lora_dropout": 0.0,  
    # Dropout probability applied to LoRA layers — prevents overfitting (0.0 = no dropout).

    "use_gradient_checkpointing": "unsloth",  
    # Enables Unsloth’s gradient checkpointing to reduce memory consumption during backpropagation.

    "use_rslora": False,  
    # Whether to use rank-stabilized LoRA (RS-LoRA). Useful for very low-rank setups.

    "loftq_config": None,  
    # Optional configuration for LoftQ (Low-rank fine-tuning with quantization); left None if unused.

    "random_state": 3407,  
    # Random seed for reproducibility of weight initialization and data shuffling.

    # ================= Train ===================
    "bf16": True,  
    # Whether to train in bfloat16 precision (automatically disabled if unsupported).

    "group_by_length": True,  
    # Groups training samples by length to reduce padding overhead and speed up training.

    "per_device_train_batch_size": 16,  
    # Number of samples processed per GPU per forward pass.

    "gradient_accumulation_steps": 4,  
    # Number of gradient accumulation steps before performing a backward update.
    # Effective batch size = batch_size × gradient_accumulation_steps.

    "num_train_epochs": 1,  
    # Total number of times the model will iterate over the entire training dataset.

    "warmup_steps": 1500,  
    # Number of initial steps to linearly increase the learning rate from 0 to the set value.

    "learning_rate": 2e-4,  
    # Peak learning rate used by the optimizer.

    "logging_steps": 10,  
    # Frequency (in steps) at which metrics and training progress are logged.

    "optim": "adamw_8bit",  
    # Optimizer choice: 8-bit AdamW for memory efficiency on large models.

    # ================= Logging =================
    "report_to": "wandb",  
    # Where to log training metrics: "wandb" (Weights & Biases) or "none" (disable logging).

    "wandb_project": "aLLoyM",  
    # Name of the Weights & Biases project to which training runs will be logged.

    "wandb_run_name": "finetuned_model",  
    # Custom name for this particular W&B run, helps distinguish between multiple experiments.

    # ================= Paths/Seed ==============
    "output_dir": "finetuned_model",  
    # Directory to store intermediate model checkpoints and logs during training.

    "save_dir_final": "finetuned_model/final",  
    # Directory where the final merged model (with LoRA weights applied) will be saved.

    "seed": 3407,  
    # Random seed to ensure reproducibility across runs (affects data shuffling, initialization, etc.).
}


# =============================================================================
# Imports
# =============================================================================
import os, json
from typing import List, Dict, Any
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

import os, sys, trl, transformers
print("[debug] CWD =", os.getcwd())
print("[debug] SCRIPT =", os.path.abspath(__file__))
print("[debug] TRL version =", trl.__version__)
print("[debug] Transformers version =", transformers.__version__)


# =============================================================================
# Utils
# =============================================================================
def setup_env_and_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    # Rough check: Ampere+ usually supports bf16
    major, minor = torch.cuda.get_device_capability()
    return (major >= 8)

# =============================================================================
# Dataset loading and formatting
# =============================================================================
def load_jsonl_messages(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file with OpenAI-style messages and extract fields."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            msgs = data.get("messages", [])
            sys_  = next((m["content"] for m in msgs if m["role"] == "system"), "")
            usr_  = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_ = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            records.append({"Instruction": sys_, "Question": usr_, "Answer": asst_})
    return records

def build_prompt(instr, inp, out, template, eos):
    return template.format(instr, inp, out) + eos

def compute_max_seq_length(data, tok, template, eos):
    """Tokenize all samples to determine true max length."""
    lengths = []
    for d in data:
        text = build_prompt(d["Instruction"], d["Question"], d["Answer"], template, eos)
        ids = tok(text, truncation=False, return_tensors="pt").input_ids
        lengths.append(ids.shape[1])
    return max(lengths) if lengths else 0

def dataset_from_records(records, template, eos) -> Dataset:
    """Map Instruction/Question/Answer into 'text' field for SFTTrainer."""
    ds = Dataset.from_list(records)
    def _map_fn(batch):
        return {"text": [build_prompt(i, q, a, template, eos)
                         for i, q, a in zip(batch["Instruction"], batch["Question"], batch["Answer"])]}
    return ds.map(_map_fn, batched=True)

# =============================================================================
# Model setup
# =============================================================================
def load_model_and_tokenizer(model_name, max_seq_len, dtype, load_in_4bit):
    # auto-fallback for dtype/bf16
    use_bf16 = (dtype == "bfloat16") and bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    tok.model_max_length = max_seq_len
    return model, tok

def apply_lora(model, cfg, max_seq_length):
    """Attach LoRA adapters via Unsloth."""
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
# Main
# =============================================================================
def main(cfg):
    # 1) Reproducibility
    setup_env_and_seed(cfg["seed"])

    # 2) Hugging Face login (optional; only if token is present)
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        login(hf_token)

    # 3) Optional W&B
    wandb_enabled = False
    if cfg["report_to"] == "wandb":
        try:
            import wandb
            wandb_api = os.getenv("WANDB_API_KEY", "").strip()
            if wandb_api:
                wandb.login(key=wandb_api)
                wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"])
                wandb_enabled = True
                print("[info] W&B logging enabled.")
            else:
                print("[info] WANDB_API_KEY not set — skipping W&B.")
        except Exception as e:
            print(f"[warn] W&B unavailable — continuing without it. ({e})")

    # 4) Paths
    if not os.path.isfile(cfg["jsonl_path"]):
        raise FileNotFoundError(f"Dataset not found: {cfg['jsonl_path']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["save_dir_final"], exist_ok=True)

    # 5) Temp tokenizer to compute EOS & max length
    tmp_tok = AutoTokenizer.from_pretrained(cfg["base_model_name"])
    eos_token = tmp_tok.eos_token or "</s>"

    # 6) Load dataset
    records = load_jsonl_messages(cfg["jsonl_path"])

    # 7) Compute true max length + margin
    auto_len = compute_max_seq_length(records, tmp_tok, cfg["prompt_template"], eos_token)
    max_seq_length = auto_len + cfg["max_seq_length_margin"]
    print(f"[info] Computed max_seq_length: {auto_len} (+{cfg['max_seq_length_margin']}) → {max_seq_length}")
    with open(cfg["save_max_seq_path"], "w") as f:
        f.write(str(max_seq_length))

    # 8) Build SFT dataset
    dataset = dataset_from_records(records, cfg["prompt_template"], eos_token)

    # 9) Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model_name"], max_seq_length, cfg["dtype"], cfg["load_in_4bit"]
    )

    # 10) Apply LoRA
    model = apply_lora(model, cfg, max_seq_length)

    # 11) TrainingArguments (TRL 0.9.6 style)
    args = TrainingArguments(
        bf16=(cfg["bf16"] and bf16_supported()),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        output_dir=cfg["output_dir"],
        optim=cfg["optim"],
        report_to=("wandb" if wandb_enabled else "none"),
        seed=cfg["seed"],
    )

    # 12) Trainer (tokenizer=, dataset_text_field=, max_seq_length=)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=args,
    )


    # 13) Train
    trainer.train()

    # 14) Save LoRA adapters + tokenizer
    model.save_pretrained(cfg["save_dir_final"])
    tokenizer.save_pretrained(cfg["save_dir_final"])

    # 15) Free VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[done] Training complete. Model saved to {cfg['save_dir_final']}")

if __name__ == "__main__":
    main(CONFIG)