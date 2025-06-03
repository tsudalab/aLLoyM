import json
import os
import re

def clean_user_message(text, remove_first_n=4):
    words = text.split()
    words = words[remove_first_n:]  # Remove the first n words
    words = [w for w in words if w.lower() != "are"]  # Remove "are"
    stripped_text = " ".join(words)
    stripped_text = stripped_text.split("?")[0].strip("?")
    return stripped_text

def extract_elements(assistant_response):
    return re.findall(r"([A-Za-z]+) \([\d.]+%\)", assistant_response)

def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            messages = data["messages"]

            system_msg = messages[0]
            user_msg = next(msg for msg in messages if msg["role"] == "user")
            assistant_msg = next(msg for msg in messages if msg["role"] == "assistant")
            phases = assistant_msg["content"]

            # Clean the user message
            cleaned_user_content = clean_user_message(user_msg["content"])
            data["messages"][2]["content"] = cleaned_user_content

            cleaned_user_content_without_numbers = cleaned_user_content.translate(str.maketrans("", "", "0123456789"))
            elements = extract_elements(cleaned_user_content_without_numbers)
            element_str = " and ".join(elements)

            new_user_msg = f"Under what condition do {element_str} form {phases}?"
            data["messages"][1]["content"] = new_user_msg

            # Write the modified data to the output file
            json.dump(data, f_out, ensure_ascii=False)
            f_out.write("\n")

def main():
    base_dir = "dataset/raw"
    splits = ["split_by_file", "split_random"]
    subsets = ["training", "validation"]

    for split in splits:
        for subset in subsets:
            input_file = os.path.join(base_dir, split, subset, "phase_names.jsonl")
            output_file = os.path.join(base_dir, split, subset, "reverse.jsonl")

            if os.path.exists(input_file):
                print(f"Processing {input_file} -> {output_file}")
                process_file(input_file, output_file)
            else:
                print(f"Skipped missing file: {input_file}")

if __name__ == "__main__":
    main()
