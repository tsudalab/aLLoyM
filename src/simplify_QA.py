import json
import re
import os

def extract_phase_description(content):
    """
    Extracts phase names from assistant content.
    For example: "94% HCP_A3 and 6% SOLID" -> ("HCP_A3 and SOLID", 2)
    """
    matches = re.findall(r'\b\d+% ([A-Z_]{2,}[A-Z0-9_]*)', content)
    return " and ".join(matches) if matches else "", len(matches)

def rewrite_jsonl_with_phase_names(input_path, phase_names_only_path, count_path):
    with open(input_path, 'r') as infile, open(phase_names_only_path, 'w') as phase_names_only_file, open(count_path, 'w') as count_file:
        for line in infile:
            try:
                record = json.loads(line)
                phase_names_message = []
                count_message = []

                seen_first_assistant = False

                for msg in record.get("messages", []):
                    role = msg.get("role")
                    content = msg.get("content", "")

                    if role == "assistant":
                        if not seen_first_assistant:
                            seen_first_assistant = True
                            phase_str, count = extract_phase_description(content)
                            phase_names_message.append({"role": "assistant", "content": phase_str})
                            count_message.append({"role": "assistant", "content": str(count)})
                        else:
                            break  # Stop processing further messages

                    elif role == "user":
                        modified = re.sub(r'^\s*What phases', 'How many phases', content, flags=re.IGNORECASE)
                        phase_names_message.append({"role": "user", "content": content})
                        count_message.append({"role": "user", "content": modified})

                    else:
                        phase_names_message.append(msg)
                        count_message.append(msg)

                json.dump({"messages": phase_names_message}, phase_names_only_file)
                phase_names_only_file.write("\n")

                json.dump({"messages": count_message}, count_file)
                count_file.write("\n")

            except Exception as e:
                print(f"Skipping line due to error: {e}")

# === Replace with your actual input/output paths ===
input_file = "/home/yoikawa/src/phase_LLM/working_data/full_info/split_random/training.jsonl"
phase_names_only_file = "/home/yoikawa/src/phase_LLM/working_data/phase_names_only/split_random/training.jsonl"
count_file = "/home/yoikawa/src/phase_LLM/working_data/phase_counts/split_random/training.jsonl"

# Create output directories if they do not exist
os.makedirs(os.path.dirname(phase_names_only_file), exist_ok=True)
os.makedirs(os.path.dirname(count_file), exist_ok=True)

rewrite_jsonl_with_phase_names(input_file, phase_names_only_file, count_file)
