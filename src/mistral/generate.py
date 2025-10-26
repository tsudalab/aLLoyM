#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation / generation for LoRA fine-tuned Unsloth model
- Mirrors the structure of your finetune script (CONFIG-first, env tokens, optional W&B)
- Reads max_seq_length.txt
- Logs per-sample + final accuracy to Weights & Biases (if WANDB_API_KEY is available)
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # --------- I/O ----------
    "jsonl_val_dir": "../validation",          # folder that contains {QA_type}.jsonl
    "save_results_dir": "generated",           # where to write {QA_type}.jsonl results
    "save_max_seq_path": "max_seq_length.txt", # computed at train time

    # --------- Model ----------
    "base_or_checkpoint": "finetuned_model/final",  # model or adapter dir to use for inference
    "dtype": "bfloat16",           # "bfloat16" (A100/H100) or "float16"
    "load_in_4bit": True,          # should match your training setup

    # --------- Prompt formatting ----------
    "prompt_template": (
        "### Instruction:\n{}\n\n"
        "### Input:\n{}\n\n"
        "### Output:\n{}"
    ),

    # --------- Generation ----------
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": 512,         # cap the continuation; independent from context length

    # --------- Logging ----------
    "enable_wandb": True,          # set False to hard-disable W&B even if WANDB_API_KEY exists
    "wandb_project": "aLLoyM",
    "seed": 3407,
}

# =============================================================================
# Imports
# =============================================================================
import os, json, sys
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
from huggingface_hub import login
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# =============================================================================
# Utilities
# =============================================================================
def setup_env_and_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_tokens_from_file(path: str = ".env.tokens") -> None:
    """Load lines of KEY=VALUE into os.environ; ignore comments/blank/malformed."""
    if not os.path.isfile(path):
        print(f"[warn] Token file '{path}' not found — skipping.")
        return
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ[k.strip()] = v.strip()
    print(f"[info] Loaded tokens from {path}")

def try_login_hf() -> None:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        login(token)

def maybe_init_wandb(cfg: Dict[str, Any], qa_type: str):
    """Initialize W&B if enabled and WANDB_API_KEY exists. Returns (wandb_or_None, run_name_or_None)."""
    if not cfg.get("enable_wandb", True):
        return None, None
    api = os.getenv("WANDB_API_KEY", "").strip()
    if not api:
        print("[info] WANDB_API_KEY not found — skipping W&B logging.")
        return None, None
    try:
        import wandb
        wandb.login(key=api)
        run_name = f"eval-{qa_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=cfg["wandb_project"], name=run_name, config={"QA_type": qa_type})
        print("[info] Weights & Biases logging enabled.")
        return wandb, run_name
    except Exception as e:
        print(f"[warn] Could not init W&B: {e}")
        return None, None

def read_max_seq_length(path: str) -> int:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"max_seq_length file not found: {path}")
    with open(path, "r") as f:
        return int(f.read().strip())

def load_validation_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_triplet(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    # Expect: messages[0]=system/instruction, [1]=user, [2]=assistant (gold)
    msgs = entry.get("messages", [])
    instruct = msgs[0].get("content", "") if len(msgs) > 0 else ""
    user_q   = msgs[1].get("content", "") if len(msgs) > 1 else ""
    gold     = msgs[2].get("content", "") if len(msgs) > 2 else ""
    return instruct, user_q, gold

def build_prompt(template: str, instruction: str, user_q: str, out_seed: str, eos: str = "") -> str:
    # For prompting, we normally pass an empty output seed (so model completes after "### Output:\n")
    return template.format(instruction, user_q, out_seed) + (eos or "")

def extract_after_output(decoded: str) -> str:
    """
    Extract the model continuation after the '### Output:\n' delimiter.
    Robust to different splits by using partition; fallback to whole string if delimiter not found.
    """
    head, sep, tail = decoded.partition("### Output:\n")
    return tail if sep else decoded

# =============================================================================
# Main
# =============================================================================
def main():
    # --- Args: we keep a single positional QA_type to mirror your original usage ---
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <QA_type>")
        sys.exit(1)
    QA_type = sys.argv[1]

    cfg = CONFIG
    setup_env_and_seed(cfg["seed"])

    # Env tokens + logins
    load_tokens_from_file(".env.tokens")
    try_login_hf()
    wandb, run_name = maybe_init_wandb(cfg, QA_type)

    # Max seq length
    max_seq_length = read_max_seq_length(cfg["save_max_seq_path"])
    print(f"[info] Using max_seq_length={max_seq_length}")

    # Tokenizer (for EOS & encoding)
    # Note: we use the same id space as the model/checkpoint dir
    tok = AutoTokenizer.from_pretrained(cfg["base_or_checkpoint"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
    eos_token = tok.eos_token or "</s>"

    # Model
    torch_dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_or_checkpoint"],
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=cfg["load_in_4bit"],
    )
    FastLanguageModel.for_inference(model)

    # Data
    val_path = os.path.join(cfg["jsonl_val_dir"], f"{QA_type}.jsonl")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    data = load_validation_jsonl(val_path)

    # Output
    os.makedirs(cfg["save_results_dir"], exist_ok=True)
    out_path = os.path.join(cfg["save_results_dir"], f"{QA_type}.jsonl")
    # overwrite each run for determinism
    outf = open(out_path, "w", encoding="utf-8")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total = 0
    correct = 0

    for entry in data:
        try:
            instr, user_q, gold = extract_triplet(entry)
            prompt = build_prompt(cfg["prompt_template"], instr, user_q, "", eos_token)
            inputs = tokenizer(
                [prompt],
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                use_cache=True,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
            )
            decoded_full = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            pred = extract_after_output(decoded_full).strip()

            ok = (pred == gold.strip())
            total += 1
            correct += int(ok)

            rec = {
                "user": user_q,
                "expected_answer": gold,
                "generated_answer": pred,
                "correct": ok,
            }
            json.dump(rec, outf, ensure_ascii=False)
            outf.write("\n")

            if wandb:
                wandb.log({
                    "step": total,
                    "is_correct": int(ok),
                    "running_accuracy": (correct / total) if total else 0.0,
                })

        except Exception as e:
            print(f"[warn] Error on entry: {e}")
            continue

    outf.close()
    final_acc = (correct / total) if total else 0.0
    if wandb:
        wandb.log({"final_accuracy": final_acc})
        wandb.finish()

    print(f"[done] New JSONL saved to: {out_path}")
    print(f"[done] Final Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    main()
