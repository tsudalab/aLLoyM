import json
import os

def combine_files(input_dir: str, output_file: str):
    """
    Combines all training data from JSONL files in the given directory into one JSONL file.

    Args:
        input_dir: Directory containing the training JSONL files
        output_file: Path to the output combined JSONL file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        # Iterate over all the JSONL files in the input directory
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".jsonl") and "combined" not in file:
                    input_file_path = os.path.join(root, file)
                    print(f"Combining {input_file_path} into {output_file}")
                    
                    with open(input_file_path, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            f_out.write(line)  # Write each line from the input file to the output file

def main():
    for raw_or_multi in ["raw", "multi"]:
        base_dir = f"dataset/{raw_or_multi}"
        splits = ["split_by_file", "split_random"]

        # Iterate through the splits and combine the training data
        for split in splits:
            input_dir = os.path.join(base_dir, split, "training")
            output_file = os.path.join(base_dir, split, "training/combined.jsonl")
            
            print(f"Combining training data for {split} into {output_file}")
            combine_files(input_dir, output_file)

if __name__ == "__main__":
    main()
