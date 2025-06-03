#!/usr/bin/env python3
import json
import re
import os
from typing import Tuple

def extract_phase_description(content: str) -> Tuple[str, int]:
    matches = re.findall(r'\b\d+% ([A-Z_]{2,}[A-Z0-9_]*)', content)
    return " and ".join(matches) if matches else "", len(matches)

def rewrite_jsonl_with_phase_names(input_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line)
                phase_names_message = []
                seen_first_assistant = False

                for msg in record.get("messages", []):
                    role = msg.get("role")
                    content = msg.get("content", "")

                    if role == "assistant" and not seen_first_assistant:
                        seen_first_assistant = True
                        phase_name, _ = extract_phase_description(content)
                        phase_names_message.append({"role": "assistant", "content": phase_name})
                    elif role == "user":
                        content = content + " Answer phase names only."
                        phase_names_message.append({"role": "user", "content": content})
                    else:
                        phase_names_message.append(msg)

                json.dump({"messages": phase_names_message}, outfile)
                outfile.write("\n")
            except Exception as e:
                print(f"Error processing line {line_num} in {input_path}: {e}")

def main():
    base_dir = "dataset/raw"
    splits = ["split_by_file", "split_random"]
    subsets = ["training", "validation"]

    for split in splits:
        for subset in subsets:
            input_path = os.path.join(base_dir, split, subset, "full.jsonl")
            output_path = os.path.join(base_dir, split, subset, "phase_names.jsonl")

            if os.path.exists(input_path):
                print(f"Processing {input_path} -> {output_path}")
                rewrite_jsonl_with_phase_names(input_path, output_path)
            else:
                print(f"Skipped missing file: {input_path}")

if __name__ == "__main__":
    main()
