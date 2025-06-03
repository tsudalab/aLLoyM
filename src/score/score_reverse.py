import json
import re
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from tqdm import tqdm

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

def calculate_score(llm_elements, llm_temp, valid_elements, valid_temp, max_percentage=100, temp_range=4860):  # max_temp=5000K - min_temp=140K
    """
    Vectorized score calculation based on matching elements and closeness of percentages and temperature
    """
    # Check for empty or None inputs to prevent errors
    if not llm_elements or not valid_elements:
        return 0.0
    
    if llm_temp is None or valid_temp is None:
        return 0.0

    # MODIFIED: Skip if number of elements doesn't match
    if len(llm_elements) != len(valid_elements):
        return 0.0

    matched_elements = 0
    percentage_accuracies = []

    # Check elements match and their percentage closeness
    for element, percentage in valid_elements.items():
        if element in llm_elements:
            matched_elements += 1
            accuracy = 1 - min(abs(llm_elements[element] - percentage) / max_percentage, 1.0)
            percentage_accuracies.append(accuracy)

    if matched_elements == len(valid_elements):
        # Percentage accuracy score (up to 66.6 points)
        avg_accuracy = sum(percentage_accuracies) / len(percentage_accuracies) if percentage_accuracies else 0
        percentage_score = 66.6 * avg_accuracy
        
        # Temperature accuracy (up to 33.4 points if all elements matched)
        temp_accuracy = 1 - min(abs(llm_temp - valid_temp) / temp_range, 1.0)
        temp_score = 33.4 * temp_accuracy
        
        return percentage_score + temp_score

    return 0.0


def load_training_questions(path):
    """Load questions from training.jsonl as messages[1]["content"]"""
    training_questions = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                question = obj.get("messages", [None, {}])[1].get("content", "")
                if question:
                    training_questions.add(question)
            except json.JSONDecodeError:
                continue
    return training_questions

def deduplicate_llm_data(llm_data, training_path="../../../training.jsonl"):
    """Deduplicate LLM data based on question content and remove questions in training set"""
    print("Deduplicating LLM data...")
    start_time = time.time()

    training_questions = load_training_questions(training_path)
    seen = set()
    deduplicated = []

    for item in llm_data:
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

def preprocess_validation_data(validation_data):
    """Preprocess validation data to create indexes for faster matching"""
    phase_pattern = re.compile(r"[A-Z]{3,}")
    
    # Create dictionaries for faster lookup
    validation_dict = {}
    
    for i, v in enumerate(validation_data):
        v_content = v["messages"][1]["content"]
        v_phases = frozenset(phase_pattern.findall(v_content))
        v_elements = frozenset(extract_elements(v_content))
        
        # Create a key using phases and elements
        key = (v_phases, v_elements)
        
        if key not in validation_dict:
            validation_dict[key] = []
        validation_dict[key].append((i, v))
    
    return validation_dict, validation_data

def find_matching_validations(user_content, validation_dict):
    """
    Find validation items that match the user query either exactly or by phase name or element.
    Uses preprocessed indexes for faster lookups.
    """
    # Match all occurrences of at least 3 uppercase letters (e.g., BCC, FCC, LIQUID)
    phase_pattern = re.compile(r"[A-Z]{3,}")
    user_phases = frozenset(phase_pattern.findall(user_content))
    user_elements = frozenset(extract_elements(user_content))
    
    key = (user_phases, user_elements)
    
    if key in validation_dict:
        return [v for _, v in validation_dict[key]]
    return []

def process_batch(batch_items, validation_dict, validation_data):
    """Process a batch of items"""
    results = []
    
    for llm_item in batch_items:
        user_content = llm_item["user"]
        assistant_content = llm_item["generated_answer"]
        
        # Extract elements and temperature from LLM output
        llm_elements, llm_temp = extract_elements_and_temp(assistant_content)
        
        max_score = 0
        best_match = None
        
        # Find matching validation items
        matching_validations = find_matching_validations(user_content, validation_dict)
        
        for valid_item in matching_validations:
            valid_assistant = valid_item["messages"][2]["content"]
            valid_elements, valid_temp = extract_elements_and_temp(valid_assistant)
            
            score = calculate_score(llm_elements, llm_temp, valid_elements, valid_temp)
            
            if score > max_score:
                max_score = score
                best_match = valid_item
        
        result = {
            "user_query": user_content,
            "llm_output": assistant_content,
            "llm_extracted": {"elements": llm_elements, "temperature": llm_temp},
            "best_match": best_match["messages"][2]["content"] if best_match else None,
            "best_match_extracted": extract_elements_and_temp(best_match["messages"][2]["content"]) if best_match else None,
            "score": max_score
        }
        
        results.append(result)
    
    return results

def load_jsonl(file_path):
    """Load data from a JSONL file efficiently"""
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read all lines at once for better I/O performance
        lines = f.readlines()
    
    for line in tqdm(lines, desc=f"Parsing {os.path.basename(file_path)}"):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
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

def process_data_parallel(llm_data, validation_data, output_path=None, batch_size=1000):
    """Process LLM outputs against validation data in parallel"""
    print("Preprocessing validation data...")
    start_time = time.time()
    
    # Preprocess validation data
    validation_dict, validation_data = preprocess_validation_data(validation_data)
    
    elapsed = time.time() - start_time
    print(f"Preprocessed validation data in {elapsed:.2f} seconds")

    llm_data = deduplicate_llm_data(llm_data)
    
    # Split data into batches
    batches = [llm_data[i:i+batch_size] for i in range(0, len(llm_data), batch_size)]
    
    # Create a partial function with fixed parameters
    process_func = partial(process_batch, validation_dict=validation_dict, validation_data=validation_data)
    
    # Process in parallel
    print(f"Processing {len(llm_data)} items using {cpu_count()} CPU cores...")
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        batch_results = list(tqdm(pool.imap(process_func, batches), total=len(batches), desc="Processing batches"))
    
    # Flatten results
    results = [item for sublist in batch_results for item in sublist]
    
    elapsed = time.time() - start_time
    print(f"Processed {len(results)} items in {elapsed:.2f} seconds ({len(results)/elapsed:.2f} items/second)")
    
    # Write results if output path is provided
    if output_path:
        print(f"Writing results to {output_path}...")
        write_results(results, output_path)
    
    return results

def main():
    start_time = time.time()
    print(f"Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Path to generated file
    generated_dir = "generated"
    validation_file = "../../../validation.jsonl"
    
    validation_data = load_jsonl(validation_file)
    
    filename = input("Enter the filename to process (e.g., 20250507134750.jsonl): ")
    llm_file = os.path.join(generated_dir, filename)
    output_file = os.path.join(generated_dir, f"scored_{filename}")
    
    print(f"\nProcessing {llm_file}...")
    
    # Load LLM data
    llm_data = load_jsonl(llm_file)
    
    results = process_data_parallel(llm_data, validation_data, output_file)
    
    # Print summary
    avg_score = np.mean([r["score"] for r in results])
    print(f"Processed {len(results)} items with average score: {avg_score:.2f}")
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal processing completed in {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()