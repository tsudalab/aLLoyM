import os
import json
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import element_dict

def extract_element_names(text, element_names):
    """Return a set of element names found in text."""
    return {name for name in element_names if name.lower() in text.lower()}

def shuffle_assistant_as_expected_generated(input_path: str, output_path: str, seed: int = 42):
    """
    Insert original assistant messages as 'expected_answer' and globally shuffled ones as 'generated_answer'.
    For 'reverse.jsonl', verify that element names from user prompts match between original and shuffled answers.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load entries
    with open(input_path, "r", encoding="utf-8") as infile:
        entries = [json.loads(line) for line in infile]

    # Step 2: Collect assistant messages and user element sets
    assistant_contents = []
    element_names = set(element_dict.values())
    user_elements_per_entry = []

    for entry in entries:
        user_text = next((m["content"] for m in entry.get("messages", []) if m["role"] == "user"), "")
        user_elements = extract_element_names(user_text, element_names)
        user_elements_per_entry.append(user_elements)

        assistant_msg = next((m["content"] for m in entry.get("messages", []) if m["role"] == "assistant"), None)
        assistant_contents.append(assistant_msg)

    # Step 3: Shuffle valid assistant messages
    valid_assistants = [msg for msg in assistant_contents if msg is not None]
    random.shuffle(valid_assistants)

    # Step 4: Modify each entry
    shuffled_index = 0
    for idx, (entry, original_content) in enumerate(zip(entries, assistant_contents)):
        if original_content is not None:
            generated = valid_assistants[shuffled_index]

            '''
            # If reverse.jsonl → check element consistency
            if os.path.basename(input_path) == "reverse.jsonl":
                original_elements = user_elements_per_entry[idx]
                # Try to find a shuffled assistant message with the same elements
                for i, candidate in enumerate(valid_assistants):
                    candidate_elements = extract_element_names(candidate, element_names)
                    if candidate_elements == original_elements:
                        generated = candidate
                        break
            '''

            # Append expected and generated answers
            entry["messages"].append({"role": "expected_answer", "content": original_content})
            entry["messages"].append({"role": "generated_answer", "content": generated})
            shuffled_index += 1


    # Step 5: Remove assistant message and write output
    with open(output_path, "w", encoding="utf-8") as outfile:
        for entry in entries:
            entry["messages"] = [msg for msg in entry["messages"] if msg["role"] != "assistant"]
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


    print(f"Shuffled file saved to: {output_path}")

def process_split(split_name: str):
    """
    Process validation data for the given split name.

    Args:
        split_name: Either 'split_by_file' or 'split_random'
    """
    val_dir = os.path.join("dataset/raw", split_name, "validation")
    out_dir = os.path.join("dataset/raw", split_name, "shuffled_negative_control/generated")

    if not os.path.exists(val_dir):
        print(f"Validation directory {val_dir} does not exist.")
        return

    for file in os.listdir(val_dir):
        if file.endswith(".jsonl"):
            input_path = os.path.join(val_dir, file)
            output_path = os.path.join(out_dir, file)
            print(f"Processing {input_path} -> {output_path}")
            shuffle_assistant_as_expected_generated(input_path, output_path)


def main():
    for split in ["split_by_file", "split_random"]:
        process_split(split)


if __name__ == "__main__":
    main()