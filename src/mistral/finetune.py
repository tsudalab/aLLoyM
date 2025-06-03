from unsloth import FastLanguageModel
import torch
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login
#import wandb
import os

# Authenticate with Hugging Face and WandB
token = "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"
login(token)
API_KEY = "cccf5fe11c12ee940fdc38a9ed32bc2863c0fe8b"
#wandb.login(key=API_KEY)

# Load and preprocess the JSONL dataset
training_data = []
with open("../training/combined.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        messages = data.get("messages", [])
        system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        assistant_msg = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        training_data.append({"Instruction": system_msg, "Question": user_msg, "Answer": assistant_msg})

# Use tokenizer to compute max sequence length
temp_tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit")
EOS_TOKEN = temp_tokenizer.eos_token

prompt_template = """
### Instruction:
{}

### Input:
{}

### Output:
{}"""

# Compute max sequence length
def get_max_length(data, tokenizer):
    lengths = []
    for example in data:
        prompt = prompt_template.format(example["Instruction"], example["Question"], example["Answer"]) + EOS_TOKEN
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        lengths.append(tokenized.input_ids.shape[1])
    return max(lengths)

max_seq_length = get_max_length(training_data, temp_tokenizer)
print(f"Auto-calculated max_seq_length: {max_seq_length}")

# Save max_seq_length to file
with open("max_seq_length.txt", "w") as f:
    f.write(str(max_seq_length+20))

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(training_data)

# Load model and tokenizer
dtype = torch.bfloat16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Format prompts
def formatting_prompts_func(examples):
    instructions = examples["Instruction"]
    inputs = examples["Question"]
    outputs = examples["Answer"]
    texts = [prompt_template.format(instr, inp, out) + EOS_TOKEN for instr, inp, out in zip(instructions, inputs, outputs)]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None,
)

# Training configuration
training_arguments = TrainingArguments(
    bf16=True,
    group_by_length=True,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_steps=1500,
    max_steps=15000,
    learning_rate=2e-4,
    logging_steps=10,
    output_dir="model",
    optim="adamw_8bit",
    report_to=[],
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

torch.cuda.empty_cache()