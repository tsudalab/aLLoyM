import json
import re
import os
import time
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from tqdm import tqdm
from collections import Counter

def extract_elements_and_temp(content):
    """
    Extract elements, percentages and temperature from content string.
    Optimized version with compiled regex patterns.
    """
    # Compile regex patterns for reuse
    element_pattern = re.compile(r'([A-Z][a-z]*)[\s\(]+(\d+\.?\d*)[\s\)%]+')
    temp_pattern = re.compile(r'(\d+\.?\d*)\s*K')
    
    # Find all elements and percentages
    elements_dict = {}
    element_matches = element_pattern.findall(content)
    for element, percentage in element_matches:
        elements_dict[element.strip()] = float(percentage.strip())
    
    # Find temperature
    temp_match = temp_pattern.search(content)
    temperature = int(float(temp_match.group(1))) if temp_match else None
    
    return elements_dict, temperature

def calculate_score(llm_elements, llm_temp, valid_elements, valid_temp, max_temp=5000, min_temp=140):
    """
    Calculate score using the new formula:
    max over expected elements of (1/2N * sum of FractionAcc + 1/2N * sum of TempAcc)
    """
    # Check for empty or None inputs to prevent errors
    if not llm_elements or not valid_elements:
        return 0.0
    
    if llm_temp is None or valid_temp is None:
        return 0.0

    # Number of elements in the target system
    N = len(valid_elements)
    
    # Check if all expected elements are present in generated output
    if not all(element in llm_elements for element in valid_elements.keys()):
        return 0.0
    
    max_score = 0.0
    
    # Calculate score for this expected configuration
    fraction_acc_sum = 0.0
    temp_acc_sum = 0.0
    
    # Calculate FractionAcc for each element
    for element, expected_percent in valid_elements.items():
        if element in llm_elements:
            generated_percent = llm_elements[element]
            fraction_acc = 1 - abs(expected_percent - generated_percent) / 100.0
            fraction_acc = max(0.0, fraction_acc)  # Ensure non-negative
            fraction_acc_sum += fraction_acc
    
    # Calculate TempAcc (same for all elements in this configuration)
    temp_range = max_temp - min_temp
    temp_acc = 1 - abs(valid_temp - llm_temp) / temp_range
    temp_acc = max(0.0, temp_acc)  # Ensure non-negative
    temp_acc_sum = N * temp_acc  # Same temperature accuracy for all N elements
    
    # Calculate final score
    score = (1.0 / (2 * N)) * fraction_acc_sum + (1.0 / (2 * N)) * temp_acc_sum
    max_score = max(max_score, score)
    
    return max_score * 100  # Scale to 0-100 range

def load_training_questions(path):
    """Load questions from training.jsonl - updated for new format"""
    training_questions = set()
    if not os.path.exists(path):
        print(f"Training file {path} not found, skipping deduplication against training set")
        return training_questions
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                # Handle messages format
                if isinstance(obj, dict) and "messages" in obj:
                    for message in obj["messages"]:
                        if message.get("role") == "user":
                            question = message.get("content", "")
                            if question:
                                training_questions.add(question)
                                break
                # Handle simple format: {"role": "user", "content": "..."}
                elif isinstance(obj, dict) and obj.get("role") == "user":
                    question = obj.get("content", "")
                    if question:
                        training_questions.add(question)
            except json.JSONDecodeError:
                continue
    return training_questions

def deduplicate_llm_data(llm_data, training_path="../training/reverse.jsonl"):
    """Deduplicate LLM data based on question content and remove questions in training set"""
    print("Deduplicating LLM data...")
    start_time = time.time()

    training_questions = load_training_questions(training_path)
    seen = set()
    deduplicated = []

    for item in llm_data:
        key = None
        
        # Handle messages format
        if isinstance(item, dict) and "messages" in item:
            for message in item["messages"]:
                if message.get("role") == "user":
                    key = message.get("content", "")
                    break
        # Handle simple format
        elif isinstance(item, dict) and item.get("role") == "user":
            key = item.get("content", "")
        # Handle old format for backward compatibility
        else:
            key = item.get("user", "")
            
        if key and key not in seen and key not in training_questions:
            seen.add(key)
            deduplicated.append(item)

    elapsed = time.time() - start_time
    print(f"Deduplicated LLM data: {len(llm_data)} -> {len(deduplicated)} items in {elapsed:.2f} seconds")

    return deduplicated

def extract_elements(text):
    """Extract element names from a text string"""
    return re.findall(r"([A-Za-z]+) \([\d.]+%\)", text)

def extract_phases_with_counts(text):
    """Extract phase names preserving order and duplicates, return as Counter for comparison"""
    phase_pattern = re.compile(r"[A-Z]{3,}")
    phases = phase_pattern.findall(text)
    return Counter(phases)

def get_user_content(item):
    """Extract user content from different formats"""
    if isinstance(item, dict):
        # Handle messages format
        if "messages" in item:
            for message in item["messages"]:
                if message.get("role") == "user":
                    return message.get("content", "")
        # Handle simple format
        elif item.get("role") == "user":
            return item.get("content", "")
        # Handle old format
        else:
            return item.get("user", "")
    return ""

def get_assistant_content(item):
    """Extract assistant/generated content from different formats"""
    if isinstance(item, dict):
        # Handle messages format
        if "messages" in item:
            for message in item["messages"]:
                if message.get("role") in ["generated_answer", "assistant"]:
                    return message.get("content", "")
        # Handle simple format
        elif item.get("role") == "assistant":
            return item.get("content", "")
        # Handle old format
        else:
            return item.get("generated_answer", "")
    return ""

def get_expected_content(item):
    """Extract expected answer content from different formats"""
    if isinstance(item, dict):
        # Handle messages format
        if "messages" in item:
            for message in item["messages"]:
                if message.get("role") == "expected_answer":
                    return message.get("content", "")
        # Handle old format that might have expected answer in messages[2]
        elif "messages" in item and len(item["messages"]) > 2:
            return item["messages"][2].get("content", "")
    return ""

def preprocess_validation_data(validation_data):
    """Preprocess validation data to create indexes for faster matching"""
    
    # Create dictionaries for faster lookup
    validation_dict = {}
    
    for i, v in enumerate(validation_data):
        v_content = get_user_content(v)
        v_phases = extract_phases_with_counts(v_content)
        v_elements = frozenset(extract_elements(v_content))
        
        # Create a key using phase counts and elements
        # Convert Counter to frozenset of (phase, count) tuples for hashing
        phase_key = frozenset(v_phases.items())
        key = (phase_key, v_elements)
        
        if key not in validation_dict:
            validation_dict[key] = []
        validation_dict[key].append((i, v))
    
    return validation_dict, validation_data

def find_matching_validations(user_content, validation_dict):
    """
    Find validation items that match the user query by phase counts and elements.
    Uses preprocessed indexes for faster lookups.
    """
    user_phases = extract_phases_with_counts(user_content)
    user_elements = frozenset(extract_elements(user_content))
    
    # Create key with phase counts
    phase_key = frozenset(user_phases.items())
    key = (phase_key, user_elements)
    
    if key in validation_dict:
        return [v for _, v in validation_dict[key]]
    return []

def process_batch(batch_items, validation_dict, validation_data):
    """Process a batch of items"""
    results = []
    
    for llm_item in batch_items:
        user_content = get_user_content(llm_item)
        assistant_content = get_assistant_content(llm_item)
        
        # Extract elements and temperature from LLM output
        llm_elements, llm_temp = extract_elements_and_temp(assistant_content)
        
        max_score = 0
        best_match = None
        
        # Find matching validation items
        matching_validations = find_matching_validations(user_content, validation_dict)
        
        for valid_item in matching_validations:
            # Get expected answer content
            valid_assistant = get_expected_content(valid_item)
            if not valid_assistant:
                # Fallback to assistant content if expected_answer not found
                valid_assistant = get_assistant_content(valid_item)
                
            valid_elements, valid_temp = extract_elements_and_temp(valid_assistant)
            
            score = calculate_score(llm_elements, llm_temp, valid_elements, valid_temp)

            if score > max_score:
                max_score = score
                best_match = valid_item
        
        # Handle best match content extraction
        best_match_content = None
        if best_match:
            best_match_content = get_expected_content(best_match)
            if not best_match_content:
                best_match_content = get_assistant_content(best_match)
        
        result = {
            "user_query": user_content,
            "llm_output": assistant_content,
            "llm_extracted": {"elements": llm_elements, "temperature": llm_temp},
            "best_match": best_match_content,
            "best_match_extracted": extract_elements_and_temp(best_match_content) if best_match_content else None,
            "score": max_score
        }

        print(result)
        
        results.append(result)
    
    return results

def load_jsonl(file_path):
    """Load data from a JSONL file efficiently - updated to handle new format"""
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read all lines at once for better I/O performance
        lines = f.readlines()
    
    current_pair = None
    for line in tqdm(lines, desc=f"Parsing {os.path.basename(file_path)}"):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            
            # Handle messages format (new format with messages array)
            if isinstance(item, dict) and "messages" in item:
                data.append(item)
            # Handle simple role-based format: pair user and assistant messages
            elif isinstance(item, dict) and "role" in item:
                if item["role"] == "user":
                    current_pair = {"user": item["content"]}
                elif item["role"] == "assistant" and current_pair:
                    current_pair["generated_answer"] = item["content"]
                    data.append(current_pair)
                    current_pair = None
            # Handle old format
            else:
                data.append(item)
                
        except json.JSONDecodeError:
            continue
    
    elapsed = time.time() - start_time
    print(f"Loaded {len(data)} items from {file_path} in {elapsed:.2f} seconds")
    return data

def write_results(results, output_path):
    """Write results to output file efficiently"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

def calculate_statistics(scores):
    """Calculate comprehensive statistics for scores"""
    scores_array = np.array(scores)
    
    stats = {
        "count": len(scores),
        "mean": float(np.mean(scores_array)),
        "median": float(np.median(scores_array)),
        "std": float(np.std(scores_array))
    }
    
    return stats

def write_statistics(stats, output_path):
    """Write statistics to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def process_data_parallel(llm_data, validation_data, output_path=None, batch_size=1000):
    """Process LLM outputs against validation data in parallel"""

    # Preprocess validation data
    validation_dict, validation_data = preprocess_validation_data(validation_data)
    
    llm_data = deduplicate_llm_data(llm_data)
    
    # Split data into batches
    batches = [llm_data[i:i+batch_size] for i in range(0, len(llm_data), batch_size)]
    
    # Create a partial function with fixed parameters
    process_func = partial(process_batch, validation_dict=validation_dict, validation_data=validation_data)
    
    with Pool(processes=cpu_count()) as pool:
        batch_results = list(tqdm(pool.imap(process_func, batches), total=len(batches), desc="Processing batches"))
    
    # Flatten results
    results = [item for sublist in batch_results for item in sublist]

    # Write results if output path is provided
    if output_path:
        print(f"Writing results to {output_path}...")
        write_results(results, output_path)
    
    return results

def print_low_scores(results, threshold=70.0, output_file=None):
    """Print all results with scores lower than the specified threshold"""
    low_score_results = [r for r in results if r["score"] < threshold]
    
    if not low_score_results:
        print(f"No results found with scores below {threshold}")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(low_score_results)} results with scores below {threshold}")
    print(f"{'='*80}")
    
    output_lines = []
    
    for i, result in enumerate(low_score_results, 1):
        output_line = f"\n--- Result {i}/{len(low_score_results)} (Score: {result['score']:.2f}) ---"
        print(output_line)
        output_lines.append(output_line)
        
        # User query
        query_line = f"USER QUERY: {result['user_query']}"
        print(query_line)
        output_lines.append(query_line)
        
        # LLM output
        llm_line = f"LLM OUTPUT: {result['llm_output']}"
        print(llm_line)
        output_lines.append(llm_line)
        
        # LLM extracted data
        llm_extracted = result['llm_extracted']
        extracted_line = f"LLM EXTRACTED - Elements: {llm_extracted['elements']}, Temperature: {llm_extracted['temperature']}"
        print(extracted_line)
        output_lines.append(extracted_line)
        
        # Best match (if any)
        if result['best_match']:
            match_line = f"BEST MATCH: {result['best_match']}"
            print(match_line)
            output_lines.append(match_line)
            
            if result['best_match_extracted']:
                best_elements, best_temp = result['best_match_extracted']
                best_extracted_line = f"BEST MATCH EXTRACTED - Elements: {best_elements}, Temperature: {best_temp}"
                print(best_extracted_line)
                output_lines.append(best_extracted_line)
        else:
            no_match_line = "BEST MATCH: No matching validation data found"
            print(no_match_line)
            output_lines.append(no_match_line)
        
        print("-" * 80)
        output_lines.append("-" * 80)
    
    # Write to file if specified
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\nLow score results written to: {output_file}")
        except Exception as e:
            print(f"Error writing to file {output_file}: {e}")
    
    print(f"\nSummary: {len(low_score_results)} results below threshold {threshold}")
    if low_score_results:
        scores = [r['score'] for r in low_score_results]
        print(f"Low score statistics - Min: {min(scores):.2f}, Max: {max(scores):.2f}, Avg: {np.mean(scores):.2f}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process LLM outputs against validation data')
    parser.add_argument('--print-low-scores', action='store_true', 
                        help='Print all results with scores below the threshold')
    parser.add_argument('--score-threshold', type=float, default=70.0,
                        help='Score threshold for low score filtering (default: 70.0)')
    parser.add_argument('--low-scores-output', type=str, default=None,
                        help='Output file for low score results (optional)')
    parser.add_argument('--validation-file', type=str, default='../validation/reverse.jsonl',
                        help='Path to validation data file')
    parser.add_argument('--llm-file', type=str, default='generated/reverse.jsonl',
                        help='Path to LLM generated data file')
    parser.add_argument('--output-file', type=str, default='scores/reverse.json',
                        help='Output file for results and statistics')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    start_time = time.time()
    print(f"Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    validation_data = load_jsonl(args.validation_file)
    
    # Load LLM data
    llm_data = load_jsonl(args.llm_file)
    
    results = process_data_parallel(llm_data, validation_data, args.output_file)
    
    # Extract scores and calculate statistics
    scores = [r["score"] for r in results]
    stats = calculate_statistics(scores)
    
    # Write statistics to file
    print(f"Writing statistics to {args.output_file}...")
    write_statistics(stats, args.output_file)
    
    # Print overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total results: {stats['count']}")
    print(f"Mean score: {stats['mean']:.2f}")
    print(f"Median score: {stats['median']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    
    # Print low scores if requested
    if args.print_low_scores:
        print_low_scores(results, args.score_threshold, args.low_scores_output)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    main()