import os
import sys
import json
import re
from collections import defaultdict
import numpy as np

aLLoyM_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(aLLoyM_path)
from config import element_dict, phase_list

def count_elements_in_user_query(user_text, element_dict):
    """Count the number of elements mentioned in the user query using regex."""
    elements_found = set()
    
    # Create a pattern that matches any element name (case-insensitive)
    for element in element_dict.values():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(element) + r'\b'
        if re.search(pattern, user_text, re.IGNORECASE):
            elements_found.add(element)
    
    return len(elements_found)

def compare_answers(expected, generated):
    """Compare expected and generated answers for correctness."""
    if not expected or not generated:
        return False
    
    # Convert to strings and normalize whitespace
    expected_str = str(expected).strip().lower()
    generated_str = str(generated).strip().lower()
    
    # Simple exact match comparison
    return expected_str == generated_str

def extract_data_from_messages(messages):
    """Extract user query, expected answer, and generated answer from messages format."""
    user_query = ""
    expected_answer = ""
    generated_answer = ""
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'user':
            user_query = content
        elif role == 'expected_answer':
            expected_answer = content
        elif role == 'generated_answer':
            generated_answer = content
    
    return user_query, expected_answer, generated_answer

def process_jsonl_file(file_path, element_dict):
    """Process a JSONL file and group results by number of elements."""
    grouped_data = defaultdict(list)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        
                        # Handle both formats: direct fields and messages format
                        if 'messages' in data:
                            # Messages format
                            user_query, expected_answer, generated_answer = extract_data_from_messages(data['messages'])
                        else:
                            # Direct fields format
                            user_query = data.get('user', '') or data.get('question', '')
                            expected_answer = data.get('expected_answer', '')
                            generated_answer = data.get('generated_answer', '') or data.get('answer', '')
                        
                        # Skip if we don't have the required data
                        if not user_query:
                            continue
                        
                        # Compare expected vs generated answers
                        is_correct = compare_answers(expected_answer, generated_answer)
                        
                        # Count elements in the user query
                        num_elements = count_elements_in_user_query(user_query, element_dict)
                        
                        # If only 1 element, combine with 2 element group
                        if num_elements == 1:
                            num_elements = 2
                        
                        grouped_data[num_elements].append(is_correct)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
                        continue
                        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}
    
    return grouped_data

def calculate_statistics(grouped_data):
    """Calculate mean% and standard deviation for each group."""
    results = {}
    
    for num_elements, correct_values in grouped_data.items():
        if correct_values:  # Only process if there's data
            # Convert boolean values to integers (True=1, False=0)
            numeric_values = [1 if val else 0 for val in correct_values]
            
            mean_percent = np.mean(numeric_values) * 100
            std_dev = np.std(numeric_values, ddof=1) * 100 if len(numeric_values) > 1 else 0.0
            
            results[num_elements] = {
                'mean_percent': round(mean_percent, 2),
                'std_dev': round(std_dev, 2),
                'count': len(correct_values)
            }
    
    return results

def write_results_to_file(results, output_path):
    """Write results to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Statistics by Number of Elements (Expected vs Generated Answers)\n")
            f.write("=" * 60 + "\n\n")
            
            for num_elements in sorted(results.keys()):
                stats = results[num_elements]
                f.write(f"Group: {num_elements} elements\n")
                f.write(f"Mean%: {stats['mean_percent']}%\n")
                f.write(f"StdDev: {stats['std_dev']}%\n")
                f.write(f"Count: {stats['count']} samples\n")
                f.write("-" * 30 + "\n")
        
        print(f"Results written to: {output_path}")
        
    except Exception as e:
        print(f"Error writing results to file: {e}")

if __name__ == "__main__":
    pwd = os.getcwd()
    
    # go back 2 directories before
    multi_or_raw = pwd.split("/")[-3]
    if multi_or_raw == "multi":
        finetune_types = ["generation_unfinetuned", "generated"]
        QA_types = ["full", "phase_names", "reverse"]
    elif multi_or_raw == "raw":
        finetune_types = ["generated"]
        QA_types = ["full"]
    
    for finetune_type in finetune_types:
        for QA_type in QA_types:
            directory_path = f"{finetune_type}/{QA_type}.jsonl"

            output_path = f"scores/{finetune_type}/{QA_type}.txt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Processing {QA_type}...")
            
            # Process the JSONL file
            grouped_data = process_jsonl_file(directory_path, element_dict)
            
            if not grouped_data:
                print(f"No data found in {directory_path}")
                continue
            
            # Calculate statistics
            results = calculate_statistics(grouped_data)
            
            # Create output filename
            
            # Write results to file
            write_results_to_file(results, output_path)
            
            # Also print to console for immediate feedback
            print(f"\nResults for {QA_type}:")
            for num_elements in sorted(results.keys()):
                stats = results[num_elements]
                print(f"  {num_elements} elements: Mean={stats['mean_percent']}%, StdDev={stats['std_dev']}%, Count={stats['count']}")