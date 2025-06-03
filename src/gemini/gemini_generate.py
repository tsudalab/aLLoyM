import os
import json
import time
import sys
from time import sleep
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import GEMINI_API_KEY

# Set your API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini 1.5 Flash model
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

# Set the QA type
QA_type = "phase_names"  # replace with actual QA type like "math", "bio", etc.

# File paths
input_file = f"../../dataset/multi/split_by_file/validation/{QA_type}.jsonl"
output_file = f"generated/{QA_type}.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Prompt template
prompt_template = """
### Instruction:
{}

### Input:
{}

### Output:
{}"""

# Load validation data
with open(input_file, "r", encoding="utf-8") as file:
    validation_data = [json.loads(line) for line in file]

total_tokens = 0
# Generate and save responses
with open(output_file, 'a', encoding='utf-8') as outfile:
    for entry in validation_data:
        try:
            messages = entry["messages"]
            instruction = messages[0]['content']
            user_message = messages[1]['content']
            expected_answer = messages[2]['content']

            prompt = prompt_template.format(instruction, user_message, "")

            # Generate response
            response = model.generate_content(prompt)

            # Output response
            if response.parts:
                print("\n✅ Gemini's Response:")
                print(response.text)
                generated_text = response.text.strip().lower()
                usage = response.usage_metadata.total_token_count
                total_tokens += usage
            else:
                print("\n⚠️ Gemini's Response was empty.")
                generated_text = ""

            is_correct = "true" if generated_text == expected_answer else "false"

            # Save to output file
            output_entry = {
                "instruction": instruction,
                "input": user_message,
                "expected": expected_answer,
                "generated": generated_text,
                "correct": is_correct,
            }
            outfile.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"❌ Error processing entry: {e}")

        if total_tokens > 1_000:
            sleep(60)