from unsloth import FastLanguageModel
import torch
import json
import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login
import wandb
from datetime import datetime
import sys

QA_type = sys.argv[1]

# Authenticate with Hugging Face and WandB
token = "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"
login(token)
API_KEY = "cccf5fe11c12ee940fdc38a9ed32bc2863c0fe8b"
wandb.login(key=API_KEY)

with open("../mistral_LLM/max_seq_length.txt", "r") as f:
    max_seq_length = int(f.read().strip())
    
dtype = torch.bfloat16
load_in_4bit = True

# Load model and tokenizer
load_model, load_tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit',
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(load_model)


input_file = f"../validation/{QA_type}.jsonl"
with open(input_file, "r", encoding="utf-8") as file:
    validation_data = [json.loads(line) for line in file]

# Define prompt template
prompt_template = """
### Instruction:
{}

### Input:
{}

### Output:
{}"""

# Prepare output file
output_file = f'generation_results/{QA_type}.jsonl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'a', encoding='utf-8') as outfile:
    for entry in validation_data:
        messages = entry["messages"]
        user_message = messages[1]["content"]  # Extract user question
        expected_answer = messages[2]["content"]  # Expected model answer
        instruction = messages[0]["content"]

        # Tokenize input
        inputs = load_tokenizer(
            [prompt_template.format(instruction, user_message, "")], 
            return_tensors="pt"
        ).to("cuda")

        # Generate response
        outputs = load_model.generate(
            **inputs, 
            max_new_tokens=max_seq_length,
            use_cache=True,
            temperature=0.6,
            top_p=0.9
        )
        decoded_output = load_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("Output:\n")[1]
        
        # Compare with expected answer
        is_correct = decoded_output.strip() == expected_answer.strip()
        
        # Save results
        result = {
            "question": user_message,
            "expected_answer": expected_answer,
            "generated_answer": decoded_output,
            "correct": is_correct
        }

        # Print per line
        print(json.dumps(result, ensure_ascii=False))

        # Write per line
        json.dump(result, outfile)
        outfile.write('\n')

print(f"New JSONL file with results saved to {output_file}")