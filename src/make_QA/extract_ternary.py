import json

input_file = "phase_names_ternary.jsonl"
output_file = "output_3plus.jsonl"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            user_text = data.get("user", "")
            if user_text.count("+") == 2:
                outfile.write(json.dumps(data) + "\n")
        except json.JSONDecodeError:
            continue