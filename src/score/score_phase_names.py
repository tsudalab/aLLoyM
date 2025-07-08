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
from config import phase_list, element_dict

def load_element_counts():
    """Load element training counts from summary file"""
    try:
        df = pd.read_csv("../../../summary.txt", sep='\t', usecols=['element', 'num_train_files'])
        num_train_dict = df.set_index('element')['num_train_files'].to_dict()
        print(f"Loaded training counts for {len(num_train_dict)} elements")
        return num_train_dict
    except Exception as e:
        print(f"Error loading summary file: {e}")
        return {}

def extract_phases_from_answer(answer, phase_list):
    """
    Extract valid phases from an answer string, handling different formats
    """
    if not answer:
        return []
    
    # First try to find exact matches from the phase list
    found_phases = []
    for phase in phase_list:
        if re.search(r'\b' + re.escape(phase) + r'\b', answer, re.IGNORECASE):
            found_phases.append(phase)
    
    # If we found valid phases, return them
    if found_phases:
        return found_phases
    
    # Otherwise, fall back to extracting word-like patterns
    # This handles cases where phases might not be in our predefined list
    word_phases = re.findall(r'\b[A-Z][A-Z0-9_]*\b', answer)
    return word_phases if word_phases else re.findall(r'\w+', answer)

def calculate_phase_score(expected_phases, generated_phases):
    """
    Calculate Jaccard similarity score between expected and generated phases
    """
    if not expected_phases and not generated_phases:
        return 1.0  # Both empty - perfect match
    
    if not expected_phases or not generated_phases:
        return 0.0  # One is empty
    
    # Convert to sets for comparison (case-insensitive)
    expected_set = set(phase.upper() for phase in expected_phases)
    generated_set = set(phase.upper() for phase in generated_phases)
    
    # Calculate Jaccard similarity
    intersection = len(expected_set & generated_set)
    union = len(expected_set | generated_set)
    
    return intersection / union if union > 0 else 0.0

def extract_elements_from_question_regex(question):
    """
    Extract elements from question using regex with element_dict values
    """
    found_elements = []
    
    # Create pattern for all element names (both symbols and full names)
    all_element_names = list(element_dict.keys()) + list(element_dict.values())
    # Sort by length (descending) to match longer names first
    all_element_names.sort(key=len, reverse=True)
    
    for element_name in all_element_names:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(element_name) + r'\b'
        if re.search(pattern, question, re.IGNORECASE):
            # Convert to standard element name if it's a symbol
            if element_name in element_dict:
                standard_name = element_dict[element_name]
            else:
                standard_name = element_name
            
            if standard_name not in found_elements:
                found_elements.append(standard_name)
    
    return found_elements

def extract_elements_from_question(question):
    """
    Extract elements and their percentages from the question
    Handles both formats: "Element (XX%)" and "Element (XX.XX%)"
    Falls back to regex-based extraction if no percentages found
    """
    # Pattern to match element names with percentages
    pattern = r'([A-Z][a-z]*)\s*\((\d+(?:\.\d+)?)\s*%?\)'
    matches = re.findall(pattern, question)
    
    if matches:
        return [(element, float(percentage)) for element, percentage in matches]
    
    # Fallback to regex-based extraction
    elements = extract_elements_from_question_regex(question)
    return [(element, 0) for element in elements]

def extract_data_from_entry(data):
    """
    Extract question, expected answer, and generated answer from a data entry.
    Handles both old format and new messages format.
    """
    # Check if it's the new messages format
    if 'messages' in data:
        messages = data['messages']
        
        # Find user question
        question = ""
        for msg in messages:
            if msg.get('role') == 'user':
                question = msg.get('content', '')
                break
        
        # Extract expected and generated answers
        expected_answer = data.get('expected_answer', {}).get('content', '') if isinstance(data.get('expected_answer'), dict) else data.get('expected_answer', '')
        generated_answer = data.get('generated_answer', {}).get('content', '') if isinstance(data.get('generated_answer'), dict) else data.get('generated_answer', '')
        
        # Handle case where expected/generated answers might be in messages
        for msg in messages:
            if msg.get('role') == 'expected_answer':
                expected_answer = msg.get('content', '')
            elif msg.get('role') == 'generated_answer':
                generated_answer = msg.get('content', '')
        
        return question, expected_answer, generated_answer
    
    # Handle old format
    else:
        question = data.get('user', data.get('question', ''))
        expected_answer = data.get('expected_answer', '').strip()
        generated_answer = data.get('generated_answer', '').strip()
        
        return question, expected_answer, generated_answer

def get_element_count_group_name(element_count):
    """Get a descriptive name for element count groups"""
    if element_count == 1:
        return "Single (1-element)"
    elif element_count == 2:
        return "Pair (2-element)"
    elif element_count == 3:
        return "Ternary (3-element)"
    else:
        return f"{element_count}-element"

def create_system_identifier(element_list):
    """Create a unique identifier for a system based on its elements"""
    return "-".join(sorted(element_list))

def main():
    """Main processing function"""
    # Create output directory
    output_dir = 'scores/phase_names'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load element training counts
    num_train_dict = load_element_counts()
    
    # Load generated answers from JSON Lines file
    data_list = []
    try:
        with open('generated/phase_names.jsonl', 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                        continue
        print(f"Loaded {len(data_list)} data entries")
    except FileNotFoundError:
        print("Error: generated/phase_names.jsonl not found")
        return
    except Exception as e:
        print(f"Error loading data file: {e}")
        return

    # Initialize scoring dictionaries grouped by element count
    scores_by_element_count = defaultdict(list)
    element_count_stats = defaultdict(lambda: defaultdict(list))
    
    # NEW: Initialize system-level tracking
    system_scores = defaultdict(list)  # system_id -> list of scores
    system_details = {}  # system_id -> system info (elements, element_count, etc.)
    
    processing_errors = 0
    
    # Process each data entry
    for idx, data in enumerate(data_list):
        try:
            # Extract question and answers using the new flexible function
            question, expected_answer, generated_answer = extract_data_from_entry(data)
            
            if not question or not expected_answer:
                continue
            
            # Extract elements from question using regex
            element_data = extract_elements_from_question(question)
            element_list = [element for element, _ in element_data]
            
            if not element_list:
                continue  # Skip if no elements found
            
            # Use actual element count (no combining groups)
            element_count = len(element_list)
            
            # NEW: Create system identifier
            system_id = create_system_identifier(element_list)
            
            # Extract phases from answers
            expected_phase_list = extract_phases_from_answer(expected_answer, phase_list)
            generated_phase_list = extract_phases_from_answer(generated_answer, phase_list)
            
            # Calculate score
            score = calculate_phase_score(expected_phase_list, generated_phase_list)
            
            # Store score by element count
            scores_by_element_count[element_count].append(score)
            
            # Store scores for each element within this group
            for element in element_list:
                element_count_stats[element_count][element].append(score)
            
            # NEW: Store system-level information
            system_scores[system_id].append(score)
            if system_id not in system_details:
                system_details[system_id] = {
                    'elements': element_list,
                    'element_count': element_count,
                    'element_count_group': get_element_count_group_name(element_count),
                    'system_name': system_id
                }
                        
        except Exception as e:
            processing_errors += 1
            print(f"Error processing entry {idx+1}: {data}. Error: {e}")
            continue
    
    if processing_errors > 0:
        print(f"Warning: {processing_errors} entries had processing errors")
    
    # Calculate and display summary statistics grouped by element count
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY ELEMENT COUNT")
    print("="*80)
    
    all_group_stats = []
    
    for element_count in sorted(scores_by_element_count.keys()):
        group_scores = scores_by_element_count[element_count]
        
        if not group_scores:
            continue
            
        group_name = get_element_count_group_name(element_count)
        
        print(f"\n{group_name}:")
        print("-" * len(group_name))
        
        # Overall group statistics
        group_avg = np.mean(group_scores)
        group_std = np.std(group_scores)
        group_count = len(group_scores)
        
        print(f"Overall Group Stats: Avg={group_avg:.3f} ± {group_std:.3f} (n={group_count})")
        
        # Individual element statistics within this group
        element_stats_in_group = []
        
        for element in sorted(element_count_stats[element_count].keys()):
            element_scores = element_count_stats[element_count][element]
            
            if element_scores:
                avg_score = np.mean(element_scores)
                std_score = np.std(element_scores)
                count = len(element_scores)
                num_train = num_train_dict.get(element, 0)
                
                element_stats_in_group.append({
                    'element_count': element_count,
                    'element_count_group': group_name,
                    'element': element,
                    'avg_score': avg_score,
                    'std_score': std_score,
                    'count': count,
                    'train_count': num_train
                })
                
                print(f"  {element:12s}: Avg={avg_score:.3f} ± {std_score:.3f} "
                      f"(n={count:3d}, train={num_train:3d})")
        
        # Store group-level statistics
        all_group_stats.extend(element_stats_in_group)
        
        # Save group-specific statistics
        if element_stats_in_group:
            df_group = pd.DataFrame(element_stats_in_group)
            # Use descriptive filename
            safe_group_name = group_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            group_filename = f'element_scores_{safe_group_name}.txt'
            df_group.to_csv(os.path.join(output_dir, group_filename), index=False, sep='\t')
            print(f"  → Saved to {output_dir}/{group_filename}")
    
    # NEW: Calculate and display system-level statistics
    print("\n" + "="*80)
    print("SYSTEM-LEVEL STATISTICS")
    print("="*80)
    
    # Prepare system statistics for output
    system_stats = []
    
    for system_id, scores in system_scores.items():
        if scores:
            system_info = system_details[system_id]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            count = len(scores)
            
            system_stats.append({
                'system_id': system_id,
                'elements': '-'.join(system_info['elements']),
                'element_count': system_info['element_count'],
                'element_count_group': system_info['element_count_group'],
                'avg_score': avg_score,
                'std_score': std_score,
                'count': count,
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'median_score': np.median(scores)
            })
    
    # Sort systems by average score (descending)
    system_stats.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Display system statistics by element count group
    for element_count in sorted(scores_by_element_count.keys()):
        group_name = get_element_count_group_name(element_count)
        group_systems = [s for s in system_stats if s['element_count'] == element_count]
        
        if group_systems:
            print(f"\n{group_name} Systems:")
            print("-" * (len(group_name) + 9))
            print(f"{'System':<30} {'Avg':<6} {'Std':<6} {'Count':<6} {'Min':<6} {'Max':<6} {'Median':<6}")
            print("-" * 78)
            
            for system in group_systems:
                print(f"{system['system_id']:<30} "
                      f"{system['avg_score']:<6.3f} "
                      f"{system['std_score']:<6.3f} "
                      f"{system['count']:<6d} "
                      f"{system['min_score']:<6.3f} "
                      f"{system['max_score']:<6.3f} "
                      f"{system['median_score']:<6.3f}")
    
    # Save system statistics to file
    if system_stats:
        df_systems = pd.DataFrame(system_stats)
        df_systems.to_csv(os.path.join(output_dir, 'system_scores.txt'), index=False, sep='\t')
        print(f"\nSaved system statistics to {output_dir}/system_scores.txt")
        
        # Save top performing systems
        top_systems = system_stats[:20]  # Top 20 systems
        df_top_systems = pd.DataFrame(top_systems)
        df_top_systems.to_csv(os.path.join(output_dir, 'top_system_scores.txt'), index=False, sep='\t')
        print(f"Saved top 20 system statistics to {output_dir}/top_system_scores.txt")
    
    # Save all group statistics combined
    if all_group_stats:
        df_all_groups = pd.DataFrame(all_group_stats)
        df_all_groups.to_csv(os.path.join(output_dir, 'element_scores_by_group.txt'), index=False, sep='\t')
        print(f"\nSaved combined group statistics to {output_dir}/element_scores_by_group.txt")
    
    # Overall statistics across all groups
    all_scores = [score for scores in scores_by_element_count.values() for score in scores]
    if all_scores:
        overall_stats = {
            'total_evaluations': len(all_scores),
            'total_unique_systems': len(system_scores),
            'average_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'median_score': np.median(all_scores)
        }
        
        print(f"\nOVERALL STATISTICS ACROSS ALL GROUPS:")
        print(f"Total evaluations: {overall_stats['total_evaluations']}")
        print(f"Total unique systems: {overall_stats['total_unique_systems']}")
        print(f"Average score: {overall_stats['average_score']:.3f} ± {overall_stats['std_score']:.3f}")
        print(f"Median score: {overall_stats['median_score']:.3f}")
        print(f"Score range: {overall_stats['min_score']:.3f} - {overall_stats['max_score']:.3f}")
        
        # Add group breakdown to overall stats
        group_breakdown = {}
        for element_count in sorted(scores_by_element_count.keys()):
            group_scores = scores_by_element_count[element_count]
            group_name_key = get_element_count_group_name(element_count).lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            group_breakdown[f'{group_name_key}_count'] = len(group_scores)
            group_breakdown[f'{group_name_key}_avg'] = np.mean(group_scores)
            group_breakdown[f'{group_name_key}_std'] = np.std(group_scores)
            
            # Add system counts per group
            group_systems = [s for s in system_stats if s['element_count'] == element_count]
            group_breakdown[f'{group_name_key}_unique_systems'] = len(group_systems)
        
        overall_stats.update(group_breakdown)
        
        # Save overall statistics to file
        df_overall = pd.DataFrame([overall_stats])
        df_overall.to_csv(os.path.join(output_dir, 'overall_statistics_by_group.txt'), index=False, sep='\t')
        print(f"\nSaved overall statistics to {output_dir}/overall_statistics_by_group.txt")
    
    # Summary table by element count group
    print(f"\nSUMMARY BY ELEMENT COUNT:")
    print("-" * 80)
    print(f"{'Group':<20} {'Evaluations':<12} {'Systems':<8} {'Avg Score':<12} {'Std':<8}")
    print("-" * 80)
    for element_count in sorted(scores_by_element_count.keys()):
        group_scores = scores_by_element_count[element_count]
        group_name = get_element_count_group_name(element_count)
        group_systems = [s for s in system_stats if s['element_count'] == element_count]
        print(f"{group_name:<20} {len(group_scores):<12} {len(group_systems):<8} "
              f"{np.mean(group_scores):<12.3f} {np.std(group_scores):<8.3f}")
    
    # Additional analysis for pairs and ternary systems
    print(f"\nDETAILED ANALYSIS:")
    print("-" * 60)
    
    # Analyze pair systems (2-element)
    if 2 in scores_by_element_count:
        pair_scores = scores_by_element_count[2]
        pair_systems = [s for s in system_stats if s['element_count'] == 2]
        
        print(f"Pair (Binary) Systems Analysis:")
        print(f"  Total pairs evaluated: {len(pair_scores)}")
        print(f"  Unique pair systems: {len(pair_systems)}")
        print(f"  Average performance: {np.mean(pair_scores):.3f} ± {np.std(pair_scores):.3f}")
        print(f"  Best performing pairs: {np.max(pair_scores):.3f}")
        print(f"  Worst performing pairs: {np.min(pair_scores):.3f}")
        
        # Find best and worst performing pair systems
        if pair_systems:
            best_pair = max(pair_systems, key=lambda x: x['avg_score'])
            worst_pair = min(pair_systems, key=lambda x: x['avg_score'])
            print(f"  Best pair system: {best_pair['system_id']} (avg: {best_pair['avg_score']:.3f})")
            print(f"  Worst pair system: {worst_pair['system_id']} (avg: {worst_pair['avg_score']:.3f})")
        
        # Find best and worst performing element pairs
        pair_element_performance = []
        for element in element_count_stats[2]:
            element_scores = element_count_stats[2][element]
            if element_scores:
                avg_score = np.mean(element_scores)
                pair_element_performance.append((element, avg_score, len(element_scores)))
        
        if pair_element_performance:
            pair_element_performance.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 5 elements in pairs:")
            for element, avg_score, count in pair_element_performance[:5]:
                print(f"    {element}: {avg_score:.3f} (n={count})")
    
    # Analyze ternary systems (3-element)
    if 3 in scores_by_element_count:
        ternary_scores = scores_by_element_count[3]
        ternary_systems = [s for s in system_stats if s['element_count'] == 3]
        
        print(f"\nTernary Systems Analysis:")
        print(f"  Total ternary systems evaluated: {len(ternary_scores)}")
        print(f"  Unique ternary systems: {len(ternary_systems)}")
        print(f"  Average performance: {np.mean(ternary_scores):.3f} ± {np.std(ternary_scores):.3f}")
        print(f"  Best performing ternary: {np.max(ternary_scores):.3f}")
        print(f"  Worst performing ternary: {np.min(ternary_scores):.3f}")
        
        # Find best and worst performing ternary systems
        if ternary_systems:
            best_ternary = max(ternary_systems, key=lambda x: x['avg_score'])
            worst_ternary = min(ternary_systems, key=lambda x: x['avg_score'])
            print(f"  Best ternary system: {best_ternary['system_id']} (avg: {best_ternary['avg_score']:.3f})")
            print(f"  Worst ternary system: {worst_ternary['system_id']} (avg: {worst_ternary['avg_score']:.3f})")
        
        # Find best and worst performing elements in ternary systems
        ternary_element_performance = []
        for element in element_count_stats[3]:
            element_scores = element_count_stats[3][element]
            if element_scores:
                avg_score = np.mean(element_scores)
                ternary_element_performance.append((element, avg_score, len(element_scores)))
        
        if ternary_element_performance:
            ternary_element_performance.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 5 elements in ternary systems:")
            for element, avg_score, count in ternary_element_performance[:5]:
                print(f"    {element}: {avg_score:.3f} (n={count})")
    
    # Compare performance across different system sizes
    if len(scores_by_element_count) > 1:
        print(f"\nCOMPARATIVE ANALYSIS:")
        print("-" * 60)
        system_comparison = []
        for element_count in sorted(scores_by_element_count.keys()):
            group_scores = scores_by_element_count[element_count]
            group_name = get_element_count_group_name(element_count)
            avg_score = np.mean(group_scores)
            system_comparison.append((element_count, group_name, avg_score, len(group_scores)))
        
        print("Performance by system complexity:")
        for element_count, group_name, avg_score, count in system_comparison:
            print(f"  {group_name}: {avg_score:.3f} (n={count})")
        
        # Calculate correlation between system size and performance
        if len(system_comparison) > 2:
            element_counts = [x[0] for x in system_comparison]
            avg_scores = [x[2] for x in system_comparison]
            correlation = np.corrcoef(element_counts, avg_scores)[0, 1]
            print(f"\nCorrelation between system size and performance: {correlation:.3f}")
            if abs(correlation) > 0.5:
                trend = "decreases" if correlation < 0 else "increases"
                print(f"Performance generally {trend} with system complexity.")

if __name__ == "__main__":
    main()