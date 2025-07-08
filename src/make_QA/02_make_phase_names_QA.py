#!/usr/bin/env python3
import json
import re
import os
from typing import Tuple, List

def format_phase_names(phases: list) -> str:
    if not phases:
        error_message = "No phases found in the content."
        raise ValueError(error_message)
    elif len(phases) == 1:
        return phases[0]
    elif len(phases) == 2:
        return " and ".join(phases)
    else:
        return ", ".join(phases[:-1]) + " and " + phases[-1]

def extract_phase_description(content: str) -> str:
    """
    Extract phase names from the assistant's response.
    Handles two formats:
    1. "87% SOLID with composition ratio Silicon : Carbon = 1 : 1"
    2. "13% FCC_A1"
    """
    phases = []
    
    # Split by " + " to handle multiple phases
    phase_parts = content.split(' + ')
    
    for part in phase_parts:
        part = part.strip()
        
        # Pattern 1: "X% SOLID with composition ratio ..." 
        solid_match = re.search(r'\d+% SOLID with composition ratio', part)
        if solid_match:
            phases.append("SOLID")
            continue
        
        # Pattern 2: "X% PHASE_NAME" - extract the phase name after the percentage
        phase_match = re.search(r'\d+% ([A-Z_][A-Z0-9_]*)', part)
        if phase_match:
            phase_name = phase_match.group(1)
            phases.append(phase_name)
            continue
        
        # If no pattern matches, try to extract any uppercase phase name
        fallback_match = re.search(r'([A-Z_][A-Z0-9_]{2,})', part)
        if fallback_match:
            phases.append(fallback_match.group(1))
    
    # Don't remove duplicates - preserve all phases as they represent distinct phases
    # even if they have the same name (e.g., multiple SOLID phases with different compositions)
    return " + ".join(phases) + "."

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
                        phase_name = extract_phase_description(content)
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
                # Optionally, you could skip the problematic line and continue
                # continue

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
                print(f"✅ Successfully processed {input_path}")
            else:
                print(f"⚠️ Skipped missing file: {input_path}")

if __name__ == "__main__":
    main()