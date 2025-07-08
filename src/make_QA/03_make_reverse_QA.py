import json
import os
import re

def clean_user_message(text, remove_first_n=4):
    """Clean user message by removing first n words and 'are', then strip question mark"""
    words = text.split()
    words = words[remove_first_n:]  # Remove the first n words
    words = [w for w in words if w.lower() != "are"]  # Remove "are"
    stripped_text = " ".join(words)
    stripped_text = stripped_text.split("?")[0].strip("?").strip(".")  # Remove trailing question mark and period
    return stripped_text + "."

def extract_elements(text):
    """Extract element names from text containing element (percentage) patterns"""
    # Match pattern like "Antimony (100%)" or "Aluminium (2%)"
    pattern = r"([A-Za-z]+)\s*\([0-9.]+%\)"
    matches = re.findall(pattern, text)
    return matches

def process_file(input_file, output_file):
    """Process each line of the input file and write modified data to output file"""
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                messages = data["messages"]

                # Find messages by role
                system_msg = None
                user_msg = None
                assistant_msg = None
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg
                    elif msg["role"] == "user":
                        user_msg = msg
                    elif msg["role"] == "assistant":
                        assistant_msg = msg

                if not all([system_msg, user_msg, assistant_msg]):
                    print(f"Warning: Missing required message types in line {line_num}")
                    continue

                phases = assistant_msg["content"].strip().rstrip('.?')  # Clean phases text
                original_user_content = user_msg["content"]

                # Clean the user message
                cleaned_user_content = clean_user_message(original_user_content)
                
                # Extract elements from the original user content (before cleaning)
                elements = extract_elements(original_user_content)
                
                if not elements:
                    print(f"Warning: No elements found in line {line_num}: {original_user_content}")
                    continue
                
                element_str = " + ".join(elements)

                # Create new user message
                new_user_msg = f"Under what condition do {element_str} form {phases}?"
                
                # Update the messages
                # Assuming the structure should be: system, new_user, cleaned_user
                modified_data = {
                    "messages": [
                        system_msg,  # Keep original system message
                        {"role": "user", "content": new_user_msg},  # New reverse question
                        {"role": "assistant", "content": cleaned_user_content}  # Cleaned original
                    ]
                }

                # Write the modified data to the output file
                json.dump(modified_data, f_out, ensure_ascii=False)
                f_out.write("\n")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

def main():
    """Main function to process all files"""
    base_dir = "dataset/raw"
    splits = ["split_by_file", "split_random"]
    subsets = ["training", "validation"]

    for split in splits:
        for subset in subsets:
            input_file = os.path.join(base_dir, split, subset, "phase_names.jsonl")
            output_file = os.path.join(base_dir, split, subset, "reverse.jsonl")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            if os.path.exists(input_file):
                print(f"Processing {input_file} -> {output_file}")
                process_file(input_file, output_file)
                print(f"✅ Successfully processed {input_file} and saved to {output_file}")
            else:
                print(f"⚠️  Skipped missing file: {input_file}")

if __name__ == "__main__":
    # Then run main processing
    print("\nRunning main processing:")
    print("=" * 50)
    main()