import os
import json
import argparse

def calculate_accuracy_from_directory(file_path):
    correct = 0
    total = 0


    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("expected_answer") == data.get("generated_answer"):
                    correct += 1
                total += 1
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
    
    if total == 0:
        print("No valid entries found.")
        return

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2f} ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate accuracy of generated answers in JSONL files.")
    parser.add_argument("directory", help="Directory containing JSONL files")
    args = parser.parse_args()
    directory_path = f"generated/{args.directory}.jsonl"

    calculate_accuracy_from_directory(directory_path)
