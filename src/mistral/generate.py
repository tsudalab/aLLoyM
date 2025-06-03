import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
import json
import os
import sys
from datetime import datetime
from huggingface_hub import login
from transformers import AutoTokenizer

# Second argument for the script
QA_type = sys.argv[1]

# Authenticate
login('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')

with open("max_seq_length.txt", "r") as f:
    max_seq_length = int(f.read().strip())
    
dtype = torch.bfloat16
load_in_4bit = True

# Load model and tokenizer
load_model, load_tokenizer = FastLanguageModel.from_pretrained(
    model_name='model/checkpoint-15000',
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
output_file = f'generated/{QA_type}.jsonl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'a', encoding='utf-8') as outfile:
    for entry in validation_data:
        messages = entry["messages"]
        instruction = messages[0]['content']  # Instruction
        user_message = messages[1]['content']  # Extract user question
        expected_answer = messages[2]['content']  # Expected model answer

        # Tokenize input
        inputs = load_tokenizer(
            [prompt_template.format(instruction, user_message, '')],
            max_length=max_seq_length,
            return_tensors='pt'
        ).to('cuda')

        # Generate response
        outputs = load_model.generate(
            **inputs, 
            max_new_tokens=512,
            use_cache=True,
            temperature=0.6,
            top_p=0.9
        )
        decoded_output = load_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('Output:\n')[1]

        # Save results
        result = {
            'user': user_message,
            'expected_answer': expected_answer,
            'generated_answer': decoded_output,
        }
        # Write per line
        json.dump(result, outfile)
        outfile.write('\n')

print(f'New JSONL file with results saved to {output_file}')