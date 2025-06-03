import matplotlib.pyplot as plt
import json
import numpy as np
import re
import os
import sys
import pandas as pd
from collections import defaultdict

# Import phase list from config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import phase_list

def parse_arguments():
    """Parse command line arguments"""
    input_path = "generated/phase_name.jsonl" if len(sys.argv) < 2 else sys.argv[1]
    print(f"Input path: {input_path}")
    output_path = input_path.replace(".jsonl", "_score.csv")
    return input_path, output_path

def normalize_phase_name(phase_text):
    """
    Normalize a phase name by sorting any multi-phase descriptions.
    Example: "BCC_A2 and SOLID" and "SOLID and BCC_A2" will both return "BCC_A2 + SOLID"
    """
    if not phase_text:
        return phase_text
        
    # Common separators in phase descriptions
    separators = [' and ', ' + ', ', ', ' & ', ' with ']
    
    # Try to split the text using various separators
    phase_parts = []
    for sep in separators:
        if sep in phase_text:
            phase_parts = [part.strip() for part in phase_text.split(sep) if part.strip()]
            break
    
    # If we found multiple phases, sort and join them with a consistent separator
    if len(phase_parts) > 1:
        return " + ".join(sorted(phase_parts))
    
    # Otherwise return the original
    return phase_text

def extract_phases(answer, phase_list):
    """
    Extract phases from an answer and normalize multi-phase descriptions.
    """
    # First check for exact matches from the phase list
    matched_phases = [name for name in phase_list if re.search(r'\b' + re.escape(name) + r'\b', answer, re.IGNORECASE)]
    
    # Check for combined phases (e.g., "BCC_A2 and SOLID")
    combined_phases = []
    for sep in [' and ', ' + ', ', ', ' & ', ' with ']:
        if sep in answer:
            # This might be a multi-phase answer
            parts = [part.strip() for part in answer.split(sep) if part.strip()]
            # Check if all parts are valid phases
            if all(any(re.search(r'\b' + re.escape(name) + r'\b', part, re.IGNORECASE) for name in phase_list) for part in parts):
                combined_phases.append(normalize_phase_name(answer))
    
    # If we found combined phases, return those, otherwise return individual matches
    # If nothing was found but there's an answer, return the answer itself
    if combined_phases:
        return combined_phases
    elif matched_phases:
        return matched_phases
    elif answer:
        return [answer]
    else:
        return []

def load_jsonl_data(filepath, answer_types=["generated_answer", "expected_answer"]):
    """Load data from JSONL and process both answer types at once"""
    with open(filepath, "r") as f:
        data_lines = [json.loads(line) for line in f if line.strip()]
    
    # Store entries for both answer types
    results = {}
    
    for answer_type in answer_types:
        all_elements = set()
        entries_by_elements = defaultdict(list)
        single_element_entries = defaultdict(list)

        for i, entry in enumerate(data_lines):
            question = entry["question"]
            answer = entry.get(answer_type, "").strip()
            if not answer:
                continue

            elements = re.findall(r'([A-Z][a-z]*) \((\d+\.\d+)%\)', question)
            if not elements:
                continue
            
            temp_match = re.search(r'(\d+)\s*K', question)
            if not temp_match:
                continue
            temperature = float(temp_match.group(1))

            # Use our new function to extract phases, handling combined phases correctly
            matched_phases = extract_phases(answer, phase_list)

            element_names = [el for el, _ in elements]
            element_percentages = {el: float(pct) for el, pct in elements}

            for name in element_names:
                all_elements.add(name)

            entry_data = {
                'index': i,
                'elements': element_names,
                'percentages': element_percentages,
                'temperature': temperature,
                'phases': matched_phases,
                'raw_answer': answer
            }

            combo_key = "-".join(sorted(element_names))
            entries_by_elements[combo_key].append(entry_data)

            if len(element_names) == 1:
                single_element_entries[element_names[0]].append(entry_data)
        
        results[answer_type] = {
            'entries_by_elements': entries_by_elements,
            'single_element_entries': single_element_entries
        }
    
    return results

def load_csv_data(score_csv_path):
    """Load data from a CSV file containing scores"""
    df = pd.read_csv(score_csv_path)
    phase_list = list(df.columns[1:])  # Skip the first column (e.g., "Phase")

    scores = defaultdict(lambda: {"score_sum": 0.0, "count": 0})
    pattern = re.compile(r'([A-Z][a-z]*) \((\d+\.\d+)%\)')

    for _, row in df.iterrows():
        question = row[0]
        matched = pattern.findall(question)
        elements = sorted(el for el, _ in matched)
        if len(elements) != 2:
            continue
        key = "-".join(elements)

        score = 0
        for phase in phase_list:
            score += row[phase]
        scores[key]["score_sum"] += score
        scores[key]["count"] += 1

    return scores, phase_list

def group_binary_combinations(entries_by_elements, single_element_entries):
    """Group binary combinations with their corresponding single element entries"""
    binary_combos = {}
    for combo_key, entries in entries_by_elements.items():
        elements = combo_key.split('-')
        if len(elements) != 2:
            continue

        binary_combos[combo_key] = {
            'combo_entries': entries,
            'element1_entries': single_element_entries.get(elements[0], []),
            'element2_entries': single_element_entries.get(elements[1], [])
        }

    return binary_combos

def compute_score(entry_generated, entry_expected):
    """Compute Jaccard similarity between generated and expected phases."""
    gen_phases = set([normalize_phase_name(p) for p in entry_generated.get("phases", [])])
    exp_phases = set([normalize_phase_name(p) for p in entry_expected.get("phases", [])])
    
    if not gen_phases and not exp_phases:
        return 1.0  # Both empty, perfect match
    if not gen_phases or not exp_phases:
        return 0.0  # One is empty
    return len(gen_phases & exp_phases) / len(gen_phases | exp_phases)

def calculate_scores_from_jsonl(results):
    """Calculate average scores from JSONL processed results"""
    binary_combos_combined = {}
    
    # Group binary combos for both answer types
    for answer_type in results:
        results[answer_type]['binary_combos'] = group_binary_combinations(
            results[answer_type]['entries_by_elements'], 
            results[answer_type]['single_element_entries']
        )
    
    # Prepare combined dictionary of binary combinations
    all_combo_keys = set(results['generated_answer']['binary_combos'].keys()) | set(results['expected_answer']['binary_combos'].keys())
    
    for combo_key in all_combo_keys:
        binary_combos_combined[combo_key] = {
            'generated_answer': results['generated_answer']['binary_combos'].get(combo_key, {'combo_entries': [], 'element1_entries': [], 'element2_entries': []}),
            'expected_answer': results['expected_answer']['binary_combos'].get(combo_key, {'combo_entries': [], 'element1_entries': [], 'element2_entries': []}),
        }
    
    scores = []

    for combo_key, combo_data in binary_combos_combined.items():
        gen_entries = {e['index']: e for e in combo_data['generated_answer']['combo_entries']}
        exp_entries = {e['index']: e for e in combo_data['expected_answer']['combo_entries']}

        # Use the intersection of available indices for comparison
        common_indices = set(gen_entries.keys()) & set(exp_entries.keys())

        if not common_indices:
            continue

        score_list = [compute_score(gen_entries[i], exp_entries[i]) for i in common_indices]
        avg_score = np.mean(score_list)
        scores.append({
            'combo_key': combo_key,
            'average_score': avg_score,
            'count': len(score_list)
        })

    return scores

def calculate_average_scores_from_csv(scores_dict):
    """Calculate average scores from scores dictionary"""
    result = []
    for key, value in scores_dict.items():
        average_score = value["score_sum"] / value["count"] if value["count"] > 0 else 0.0
        result.append({
            "combo_key": key,
            "average_score": average_score,
            "count": value["count"]
        })
    return result

def calculate_element_avg_scores(scores):
    """Calculate average scores per element from binary combination scores"""
    element_scores = defaultdict(list)

    for item in scores:
        elements = item['combo_key'].split('-')
        for el in elements:
            element_scores[el].append(item['average_score'])

    element_avg_scores = {el: np.mean(score_list) for el, score_list in element_scores.items()}
    return element_avg_scores

def count_element_mentions_in_training(training_path):
    """Count how many times each element appears in the training data"""
    element_counts = defaultdict(int)
    pattern = re.compile(r'([A-Z][a-z]*) \((\d+\.\d+)%\)')

    with open(training_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            messages = entry.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    elements = pattern.findall(content)
                    for el, _ in elements:
                        element_counts[el] += 1

    return dict(element_counts)

def save_scores_to_csv(scores, out_path):
    """Save scores to CSV file"""
    df = pd.DataFrame(scores)
    df.to_csv(out_path, index=False)
    print(f"Saved combo scores to {out_path}")

def save_element_stats_csv(element_avg_scores, element_counts, out_path):
    """Save element-level statistics to CSV file"""
    all_elements = sorted(set(element_avg_scores.keys()) | set(element_counts.keys()))
    rows = []
    for el in all_elements:
        rows.append({
            "element": el,
            "average_score": element_avg_scores.get(el, np.nan),
            "count_in_training_data": element_counts.get(el, 0)
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved element stats to {out_path}")

def main():
    """Main function to orchestrate the analysis"""
    # Parse command line arguments
    input_path, output_path = parse_arguments()
    
    # Define path for additional analysis
    score_csv_path = "20250504223355_score.csv"
    training_jsonl_path = "training.jsonl"
    element_stats_path = output_path.replace("_score.csv", "_element_stats.csv")
    
    # Processing Method 1: Direct JSONL processing
    results = load_jsonl_data(input_path, answer_types=["generated_answer", "expected_answer"])
    scores = calculate_scores_from_jsonl(results)
    save_scores_to_csv(scores, output_path)
    
    # Processing Method 2: From CSV score file
    scores_dict, phase_column_list = load_csv_data(score_csv_path)
    scores = calculate_average_scores_from_csv(scores_dict)
    
    # Element-level analysis
    element_avg_scores = calculate_element_avg_scores(scores)
    element_counts = count_element_mentions_in_training(training_jsonl_path)
    save_element_stats_csv(element_avg_scores, element_counts, element_stats_path)

if __name__ == "__main__":
    main()