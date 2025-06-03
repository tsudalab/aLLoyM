#!/usr/bin/env python

import os
import re
import random
import json
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
aLLoyM_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(aLLoyM_path)
from config import element_dict, phase_list

# Function to load data from a JSONL file
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # Extract question and answer from the messages
                messages = item.get('messages', [])
                user_message = next((m['content'] for m in messages if m['role'] == 'user'), None)
                assistant_message = next((m['content'] for m in messages if m['role'] == 'assistant'), None)
                
                if user_message and assistant_message:
                    data.append({
                        'question': user_message,
                        'answer': assistant_message
                    })
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")
                continue
    return data

def extract_elements(question: str) -> List[str]:
    """Extract element names from a question"""
    # Find common elements in materials science questions
    common_elements = list(element_dict.values())
    # Create a regex pattern to find these elements in the question
    # Using word boundaries to ensure we match whole words
    pattern = r'\b(' + '|'.join(common_elements) + r')\b'
    matches = re.findall(pattern, question)
    
    return sorted(matches)  # Sort to ensure consistent ordering

def group_questions_by_elements(data: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group questions by element pairs"""
    groups = {}
    
    for item in data:
        elements = extract_elements(item["question"])
        elements_key = "-".join(elements)
        
        if elements_key not in groups:
            groups[elements_key] = []
        
        groups[elements_key].append(item)
    
    return groups

def create_element_based_distractors(correct_answer: str, same_elements: List[str], num_distractors: int = 3) -> List[str]:
    """Create synthetic distractors based on element composition changes"""
    distractors = []
    
    # Extract temperature from correct answer if present
    temp_match = re.search(r'(\d+)\s*K', correct_answer)
    temp = int(temp_match.group(1)) if temp_match else 2000
    
    # Extract percentages if present
    pct_matches = re.findall(r'(\d+\.\d+)%', correct_answer)
    pcts = [float(p) for p in pct_matches] if pct_matches else [50.0, 50.0]
    
    # Create distractors by significantly modifying the temperature and percentages
    # to ensure they're actually incorrect
    while len(distractors) < num_distractors:
        # Modify temperature significantly (by 500-1000K)
        new_temp = temp + random.choice([-1000, -750, -500, 500, 750, 1000])
        
        # Modify percentages significantly (if present)
        if len(pcts) >= 2:
            adjustment = random.uniform(10.0, 20.0)
            new_pct1 = max(5, min(95, pcts[0] + adjustment))
            new_pct2 = max(5, min(95, 100 - new_pct1))
            
            if len(same_elements) >= 2:
                new_answer = f"{same_elements[0]} ({new_pct1:.1f}%) and {same_elements[1]} ({new_pct2:.1f}%) mixed at {new_temp} K"
            else:
                new_answer = f"Elements mixed at {new_temp} K"
        else:
            # If no percentages, just modify the temperature
            new_answer = f"Elements mixed at {new_temp} K"
        
        # Make sure we don't add duplicates
        if new_answer not in distractors:
            distractors.append(new_answer)
    
    return distractors

def create_phase_name_distractors(correct_answer: str, num_distractors: int = 3, add_percentage: bool = True) -> List[str]:
    """Create synthetic distractors based on phase name variations"""
    distractors = []
    
    # Use phase names from config.py, with or without "100% " prefix
    if add_percentage:
        available_phases = [f"100% {phase}" for phase in phase_list]
    else:
        available_phases = phase_list[:]
    
    # Remove the correct answer from available phases if it matches
    available_phases = [phase for phase in available_phases if phase != correct_answer]
    
    # Randomly select distractors from available phases
    if len(available_phases) >= num_distractors:
        distractors = random.sample(available_phases, num_distractors)
    else:
        # If we don't have enough phases, use all available ones
        distractors = available_phases[:]
        
        # Fill remaining slots with variations if needed
        while len(distractors) < num_distractors:
            # Create variations by adding descriptors to existing phases
            base_phase = random.choice(phase_list)
            if add_percentage:
                variation = f"100% {base_phase}"
            else:
                variation = base_phase
            if variation not in distractors and variation != correct_answer:
                distractors.append(variation)
    
    return distractors[:num_distractors]

def create_multiple_choice_question(item: Dict[str, str], all_data: List[Dict[str, str]], file_path: str = "") -> Dict[str, Any]:
    """Create a multiple choice question with options, ensuring distractors are actually incorrect"""
    correct_answer = item["answer"]
    question = item["question"]
    
    # Find answers from other questions but with the same elements
    same_elements = extract_elements(question)
    elements_key = "-".join(sorted(same_elements))
    
    # Create a mapping of questions to their answers
    question_answer_map = {q["question"]: q["answer"] for q in all_data}
    
    # Get answers from other questions (not the current one)
    other_answers = []
    for other_item in all_data:
        if other_item["question"] == question:
            continue
            
        # Check if this answer is different from the correct one
        if other_item["answer"] != correct_answer:
            other_answers.append(other_item["answer"])
    
    # Deduplicate other answers
    other_answers = list(set(other_answers))
    
    # If we don't have enough distractors, generate more using appropriate strategy based on filename
    if len(other_answers) < 3:
        # Determine distractor strategy based on filename
        filename = os.path.basename(file_path).lower()
        
        if filename.startswith("full"):
            # Use phase-based distractors with "100%" prefix
            synthetic_distractors = create_phase_name_distractors(correct_answer, 3 - len(other_answers), add_percentage=True)
        elif filename.startswith("phase_names"):
            # Use phase-based distractors without "100%" prefix
            synthetic_distractors = create_phase_name_distractors(correct_answer, 3 - len(other_answers), add_percentage=False)
        elif filename.startswith("reverse"):
            # Use element-based distractors
            synthetic_distractors = create_element_based_distractors(correct_answer, same_elements, 3 - len(other_answers))
        else:
            # Default fallback - determine based on content
            correct_lower = correct_answer.lower()
            phase_keywords = ["phase", "alpha", "beta", "gamma", "delta", "cubic", "tetragonal", 
                             "orthorhombic", "hexagonal", "structure", "solid solution", 
                             "intermetallic", "precipitate", "eutectic", "eutectoid"]
            
            is_phase_question = any(keyword in correct_lower for keyword in phase_keywords)
            
            if is_phase_question:
                synthetic_distractors = create_phase_name_distractors(correct_answer, 3 - len(other_answers), add_percentage=True)
            else:
                synthetic_distractors = create_element_based_distractors(correct_answer, same_elements, 3 - len(other_answers))
        
        # Filter out any synthetic distractors that match existing answers
        valid_synthetic = []
        for distractor in synthetic_distractors:
            if (distractor not in other_answers and 
                distractor != correct_answer and
                distractor not in question_answer_map.values()):
                valid_synthetic.append(distractor)
        
        other_answers.extend(valid_synthetic)
        
        distractors = other_answers[:3]
    else:
        # If we have more than enough, randomly select 3
        random.shuffle(other_answers)
        distractors = other_answers[:3]
    
    # Combine correct answer and distractors, then shuffle
    choices = [correct_answer] + distractors
    random.shuffle(choices)
    
    # Find the index of the correct answer
    correct_index = choices.index(correct_answer)
    labels = ['a', 'b', 'c', 'd']
    
    return {
        "question": item["question"],
        "choices": [f"({labels[i]}) {choice}" for i, choice in enumerate(choices)],
        "correctAnswer": f"({labels[correct_index]})",
        "correctFullAnswer": f"({labels[correct_index]}) {correct_answer}"
    }

def process_element_group(elements_key_questions_filepath: tuple) -> List[Dict[str, Any]]:
    elements_key, questions, file_path = elements_key_questions_filepath

    # We need at least one question for a valid quiz item
    if len(questions) < 1:
        return []

    all_data = questions  # Pass all questions to handle distractor generation
    quiz_items = []

    for item in questions:
        mc_question = create_multiple_choice_question(item, all_data, file_path)

        formatted_question = f"{item['question']} "
        for choice in mc_question['choices']:
            formatted_question += f"{choice} "

        quiz_items.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in phase diagrams, thermodynamics, and materials science, specializing in alloy systems. Answer the multiple choice question with the letter of the correct option without brackets. Do not provide any explanations or additional information."
                },
                {
                    "role": "user",
                    "content": formatted_question.strip()
                },
                {
                    "role": "assistant",
                    "content": mc_question['correctAnswer'].strip("()")
                }
            ]
        })

    return quiz_items

def generate_quiz(data: List[Dict[str, str]], file_path: str = "") -> List[Dict[str, Any]]:
    # First, create a global mapping of questions to answers for validation
    global_question_answer_map = {item["question"]: item["answer"] for item in data}
    
    # Group items by element combinations
    element_groups = {}
    for item in data:
        elements = extract_elements(item["question"])
        if not elements:  # Skip if no elements found
            continue
        elements_key = "-".join(sorted(elements))
        if elements_key not in element_groups:
            element_groups[elements_key] = []
        element_groups[elements_key].append(item)

    quiz_questions = []

    # Use a simpler approach for small datasets
    if len(element_groups) < 10:
        for elements_key, questions in element_groups.items():
            quiz_questions.extend(process_element_group((elements_key, questions, file_path)))
    else:
        # Use parallel processing for larger datasets
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_element_group, (elements_key, questions, file_path)) for elements_key, questions in element_groups.items()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating quiz questions"):
                quiz_questions.extend(future.result())

    return quiz_questions

# Function to save a sample JSONL file for testing
def save_sample_jsonl(data, filename="sample_data.jsonl"):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Sample data saved to {filename}")

# Main execution
if __name__ == "__main__":
    # First check if we have any input files:
    for split_method in os.listdir("dataset/raw"):
        for file_type in os.listdir(f"dataset/raw/{split_method}"):
            for train_or_val in ["training", "validation"]:
                file_paths = []
                file_dir = f"dataset/raw/{split_method}/{train_or_val}"
                for file_name in os.listdir(file_dir):
                    if file_name.endswith(".jsonl") and not file_name.startswith("combined"):
                        file_paths.append(os.path.join(file_dir, file_name))

                # Process each file
                for file_path in file_paths:
                    print(f"Loading data from {file_path}...")
                    dataset = load_jsonl(file_path)
                    print(f"Successfully loaded {len(dataset)} questions from file")
                    
                    # Print example of extracted elements for verification
                    if dataset:
                        example = dataset[0]
                        print(f"\nExample question: {example['question']}")
                        elements = extract_elements(example['question'])
                        print(f"Extracted elements: {elements}")
                    
                    # Generate the quiz
                    quiz_data = generate_quiz(dataset, file_path)
                    
                    # Save to a single line JSON file
                    output_file = f"{file_path.replace('raw', 'multi')}"
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "w") as f:
                        for quiz_item in quiz_data:
                            # Write each quiz item on a separate line without indentation
                            f.write(json.dumps(quiz_item, separators=(',', ':')) + "\n")
                    
                    print(f"Quiz generated and saved to {output_file}")
                    print(f"Total questions: {len(quiz_data)}")
                    
                    # Display an example of the output format if available
                    if quiz_data:
                        print("\nExample output format:")
                        print(json.dumps(quiz_data[0], indent=2))

    import glob

    # After all processing is done, combine validation jsonl files
    print("\n>> Combining validation .jsonl files into combined.jsonl...")

    for split_method in os.listdir("dataset/multi"):
        validation_dir = f"dataset/multi/{split_method}/validation"
        combined_path = os.path.join(validation_dir, "combined.jsonl")

        # Find all .jsonl files excluding already combined ones
        jsonl_files = glob.glob(os.path.join(validation_dir, "*.jsonl"))
        jsonl_files = [f for f in jsonl_files if not os.path.basename(f).startswith("combined")]

        if not jsonl_files:
            print(f"[{split_method}] No JSONL files to combine in {validation_dir}")
            continue

        with open(combined_path, "w") as outfile:
            for file in jsonl_files:
                with open(file, "r") as infile:
                    for line in infile:
                        outfile.write(line)

        print(f"[{split_method}] Combined {len(jsonl_files)} files into {combined_path}")