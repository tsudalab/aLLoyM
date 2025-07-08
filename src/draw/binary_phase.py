import matplotlib.pyplot as plt
import json
import numpy as np
import re
import os
import sys
from collections import defaultdict
# Import phase list from config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import phase_list, element_dict

phase_list.append("PLACEHOLDER_PHASE")

def clean_phase_name(phase_name):
    """
    Clean a single phase name by removing unwanted punctuation and standardizing format.
    """
    if not phase_name:
        return phase_name
    
    # Remove common unwanted characters but preserve meaningful ones
    # Keep underscores, hyphens in compound names, and numbers
    cleaned = re.sub(r'[^\w\s\-+]', '', phase_name)  # Remove punctuation except word chars, spaces, hyphens, plus
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    cleaned = cleaned.strip()
    
    return cleaned

def normalize_phase_name(phase_text):
    """
    Normalize a phase name by cleaning it and sorting any multi-phase descriptions.
    Example: "BCC_A2 and SOLID" and "SOLID and BCC_A2" will both return "BCC_A2 + SOLID"
    """
    if not phase_text:
        return phase_text
    
    # First clean the entire phase text
    phase_text = clean_phase_name(phase_text)
        
    # Common separators in phase descriptions
    separators = [' and ', ' + ', ', ', ' & ', ' with ']
    
    # Try to split the text using various separators
    phase_parts = []
    for sep in separators:
        if sep in phase_text:
            phase_parts = [clean_phase_name(part.strip()) for part in phase_text.split(sep) if part.strip()]
            break
    
    # If we found multiple phases, sort and join them with a consistent separator
    if len(phase_parts) > 1:
        # Remove any empty parts after cleaning
        phase_parts = [part for part in phase_parts if part]
        return " + ".join(sorted(phase_parts))
    
    # Otherwise return the cleaned single phase
    return phase_text


def extract_phases(answer, phase_list):
    """
    Extract phases from an answer and normalize multi-phase descriptions.
    """
    answer = answer.strip()
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
    # If nothing was found but there's an answer, return the answer itself (normalized)
    if combined_phases:
        return combined_phases
    elif matched_phases:
        return [normalize_phase_name(phase) for phase in matched_phases]
    elif answer:
        return [normalize_phase_name(answer)]
    else:
        return []


def debug_element_extraction(question, element_dict):
    """Debug function to help identify why element extraction is failing"""
    print(f"\n=== DEBUGGING ELEMENT EXTRACTION ===")
    print(f"Question: {question}")
    print(f"Element dict values: {list(element_dict.values())}")
    
    # Try different regex patterns
    element_names = list(element_dict.values())
    
    # Original pattern
    element_pattern = '|'.join(map(re.escape, element_names))
    print(f"Element pattern: {element_pattern}")
    
    # Try the original pattern
    elements_original = re.findall(rf'({element_pattern}) \((\d+\.\d+)%\)', question)
    print(f"Original pattern results: {elements_original}")
    
    # Try more flexible patterns
    # Pattern 1: Allow spaces around percentage
    elements_flexible1 = re.findall(rf'({element_pattern})\s*\(\s*(\d+(?:\.\d+)?)%\s*\)', question)
    print(f"Flexible pattern 1 (spaces): {elements_flexible1}")
    
    # Pattern 2: Allow integer percentages
    elements_flexible2 = re.findall(rf'({element_pattern}) \((\d+(?:\.\d+)?)%\)', question)
    print(f"Flexible pattern 2 (int/float): {elements_flexible2}")
    
    # Pattern 3: Case insensitive
    elements_flexible3 = re.findall(rf'({element_pattern}) \((\d+(?:\.\d+)?)%\)', question, re.IGNORECASE)
    print(f"Flexible pattern 3 (case insensitive): {elements_flexible3}")
    
    # Pattern 4: Find any element-like pattern
    general_pattern = r'([A-Z][a-z]?)\s*\(\s*(\d+(?:\.\d+)?)%\s*\)'
    elements_general = re.findall(general_pattern, question)
    print(f"General element pattern: {elements_general}")
    
    # Show what's actually in the question around parentheses
    paren_matches = re.findall(r'[A-Za-z]+\s*\([^)]*\)', question)
    print(f"Text with parentheses: {paren_matches}")
    
    return elements_flexible3 if elements_flexible3 else elements_general


def load_data(filepath, answer_types=["generated_answer", "expected_answer"], debug=False):
    """Load data and process both answer types at once"""
    with open(filepath, "r") as f:
        data_lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(data_lines)} entries from {filepath}")
    
    # Store entries for both answer types
    results = {}
    
    for answer_type in answer_types:
        all_elements = set()
        entries_by_elements = defaultdict(list)
        single_element_entries = defaultdict(list)
        
        processed_count = 0
        skipped_count = 0

        for i, entry in enumerate(data_lines):
            question = entry["user"]
            answer = entry.get(answer_type, "").strip()
            
            if debug and i < 3:  # Debug first 3 entries
                print(f"\n--- Entry {i} ({answer_type}) ---")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
            
            if not answer:
                skipped_count += 1
                continue
                
            # First, get the names (values) from element_dict
            element_names = list(element_dict.values())
            
            # Enhanced element extraction with debugging
            if debug and i < 3:
                elements = debug_element_extraction(question, element_dict)
            else:
                # Use improved regex pattern
                element_pattern = '|'.join(map(re.escape, element_names))
                # More flexible pattern that handles spaces and integer percentages
                elements = re.findall(rf'({element_pattern})\s*\(\s*(\d+(?:\.\d+)?)%\s*\)', question, re.IGNORECASE)
            
            if debug and i < 3:
                print(f"Final extracted elements: {elements}")
            
            if not elements:
                skipped_count += 1
                continue
            
            temp_match = re.search(r'(\d+)\s*K', question)
            if not temp_match:
                if debug and i < 3:
                    print(f"No temperature found in: {question}")
                skipped_count += 1
                continue
            temperature = float(temp_match.group(1))

            # Use our new function to extract phases, handling combined phases correctly
            matched_phases = extract_phases(answer, phase_list)
            
            if debug and i < 3:
                print(f"Extracted phases: {matched_phases}")

            element_names_list = [el for el, _ in elements]
            element_percentages = {el: float(pct) for el, pct in elements}

            for name in element_names_list:
                all_elements.add(name)

            entry_data = {
                'index': i,
                'elements': element_names_list,
                'percentages': element_percentages,
                'temperature': temperature,
                'phases': matched_phases,
                'raw_answer': answer
            }

            combo_key = "-".join(sorted(element_names_list))
            entries_by_elements[combo_key].append(entry_data)

            if len(element_names_list) == 1:
                single_element_entries[element_names_list[0]].append(entry_data)
            
            processed_count += 1
        
        print(f"Answer type '{answer_type}': Processed {processed_count}, Skipped {skipped_count}")
        print(f"Found elements: {sorted(all_elements)}")
        print(f"Element combinations: {list(entries_by_elements.keys())}")
        
        results[answer_type] = {
            'entries_by_elements': entries_by_elements,
            'single_element_entries': single_element_entries
        }
    
    return results


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


def create_combo_color_mapping(combo_data):
    """Create a color mapping specific to phases in this combination only"""
    # Collect all unique phases from both answer types for this specific combo
    combo_phases = set()
    
    for answer_type in ['generated_answer', 'expected_answer']:
        all_entries = (
            combo_data[answer_type]['combo_entries'] + 
            combo_data[answer_type]['element1_entries'] + 
            combo_data[answer_type]['element2_entries']
        )
        for entry in all_entries:
            # Normalize each phase name for consistency
            normalized_phases = [normalize_phase_name(phase) for phase in entry['phases']]
            combo_phases.update(normalized_phases)
    
    # Sort phases for deterministic color assignment
    combo_phases = sorted(combo_phases)
    
    # Create colormap using rainbow colors for better distinction
    if len(combo_phases) <= 1:
        phase_to_color = {list(combo_phases)[0]: plt.cm.rainbow(0.5)} if combo_phases else {}
    else:
        phase_to_color = {phase: plt.cm.rainbow(float(i) / (len(combo_phases) - 1)) 
                         for i, phase in enumerate(combo_phases)}
    
    return phase_to_color, combo_phases


def find_representative_point(points, strategy='center'):
    """
    Find a representative point for labeling from a list of (x, y) points.
    
    Args:
        points: List of (x, y) tuples
        strategy: 'center' (centroid), 'highest' (highest temperature), 'middle' (median composition)
    
    Returns:
        (x, y) tuple of the representative point
    """
    if not points:
        return None
    
    if len(points) == 1:
        return points[0]
    
    if strategy == 'center':
        # Return the centroid
        x_vals, y_vals = zip(*points)
        return (sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals))
    
    elif strategy == 'highest':
        # Return the point with highest temperature (y-value)
        return max(points, key=lambda p: p[1])
    
    elif strategy == 'middle':
        # Return the point with median composition (x-value)
        sorted_points = sorted(points, key=lambda p: p[0])
        return sorted_points[len(sorted_points) // 2]
    
    else:
        # Default to first point
        return points[0]


def add_phase_labels(ax, phase_points, phase_to_color, label_strategy='center', 
                    avoid_overlap=True, font_size=12, offset_distance=10):
    """
    Add labels to representative points for each phase.
    
    Args:
        ax: matplotlib axis object
        phase_points: dict mapping phase names to list of (x, y) points
        phase_to_color: dict mapping phase names to colors
        label_strategy: 'center', 'highest', 'middle' - how to choose representative point
        avoid_overlap: whether to try to avoid overlapping labels
        font_size: size of the label text
        offset_distance: distance to offset labels from points
    """
    labeled_positions = []  # Track label positions to avoid overlap
    
    for phase in sorted(phase_points.keys()):
        points = phase_points[phase]
        if not points:
            continue
        
        # Find representative point
        if label_strategy == 'center':
            rep_point = find_representative_point(points, 'center')
        elif label_strategy == 'highest':
            rep_point = find_representative_point(points, 'highest')
        elif label_strategy == 'middle':
            rep_point = find_representative_point(points, 'middle')
        else:
            rep_point = points[0]  # Default to first point
        
        if rep_point is None:
            continue
        
        x, y = rep_point
        
        # Calculate label position with offset
        label_x = x + offset_distance
        label_y = y + offset_distance
        
        # Simple overlap avoidance - if too close to existing label, try different positions
        if avoid_overlap and labeled_positions:
            min_distance = 30  # Minimum distance between labels
            attempts = 0
            max_attempts = 8
            
            while attempts < max_attempts:
                too_close = False
                for existing_x, existing_y in labeled_positions:
                    distance = ((label_x - existing_x) ** 2 + (label_y - existing_y) ** 2) ** 0.5
                    if distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    break
                
                # Try different offset positions
                angle = (attempts * 45) % 360  # 0, 45, 90, 135, 180, 225, 270, 315 degrees
                angle_rad = np.radians(angle)
                label_x = x + offset_distance * np.cos(angle_rad)
                label_y = y + offset_distance * np.sin(angle_rad)
                attempts += 1
        
        # Add the label
        ax.annotate(phase, 
                   xy=(x, y), 
                   xytext=(label_x, label_y),
                   fontsize=font_size,
                   fontweight='bold',
                   color='black',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='white', 
                            edgecolor=phase_to_color[phase], 
                            alpha=0.8),
                   arrowprops=dict(arrowstyle='->', 
                                 color=phase_to_color[phase], 
                                 lw=1.5))
        
        labeled_positions.append((label_x, label_y))


def plot_combined_phase_diagram(combo_key, combo_data, output_dir, add_labels=True, 
                               label_strategy='center', label_font_size=14):
    """Plot both generated and expected phase diagrams side by side with optional labels"""
    os.makedirs(output_dir, exist_ok=True)
    elements = combo_key.split('-')
    
    # Check if we have binary entries for both types
    has_generated = len(combo_data['generated_answer']['combo_entries']) > 0
    has_expected = len(combo_data['expected_answer']['combo_entries']) > 0
    
    if not (has_generated and has_expected):
        return False  # Skip if either data type is missing
    
    # Create combination-specific color mapping
    phase_to_color, combo_phases = create_combo_color_mapping(combo_data)
    
    print(f"Combination {combo_key} color mapping for {len(combo_phases)} phases:")
    for i, phase in enumerate(sorted(phase_to_color.keys())):
        print(f"  {i+1}. {phase}")
    
    # Get temperature ranges from both sets
    generated_temps = [entry['temperature'] for entry in combo_data['generated_answer']['combo_entries']]
    expected_temps = [entry['temperature'] for entry in combo_data['expected_answer']['combo_entries']]
    all_temps = generated_temps + expected_temps
    
    if not all_temps:
        return False  # Skip if no temperature data
    
    min_temp = int(min(all_temps)) - 50
    max_temp = int(max(all_temps)) + 50
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
    
    # Track all phases for a single combined legend
    all_used_phases = set()
    phase_handles = {}
    
    # Plot generated data (left subplot)
    ax1.set_title("Generated Phase Diagram", fontsize=36, pad=20)
    all_entries = (
        combo_data['generated_answer']['combo_entries'] + 
        combo_data['generated_answer']['element1_entries'] + 
        combo_data['generated_answer']['element2_entries']
    )
    
    phase_points_generated = defaultdict(list)
    for entry in all_entries:
        if len(entry['elements']) == 1:
            composition = 0.0 if entry['elements'][0] == elements[0] else 100.0
        else:
            composition = entry['percentages'].get(elements[1], 0.0)
            
        for phase in entry['phases']:
            # Normalize the phase name for consistency
            normalized_phase = normalize_phase_name(phase)
            phase_points_generated[normalized_phase].append((composition, entry['temperature']))
            all_used_phases.add(normalized_phase)
    
    # Sort phases for consistent plotting order
    for phase in sorted(phase_points_generated.keys()):
        points = phase_points_generated[phase]
        x_vals, y_vals = zip(*points) if points else ([], [])
        scatter = ax1.scatter(x_vals, y_vals, c=[phase_to_color[phase]], label=phase,
                    s=160)
        phase_handles[phase] = scatter
    
    # Add labels to generated diagram
    if add_labels:
        add_phase_labels(ax1, phase_points_generated, phase_to_color, 
                        label_strategy=label_strategy, font_size=label_font_size)
    
    # Plot expected data (right subplot)
    ax2.set_title("Expected Phase Diagram", fontsize=36, pad=20)
    all_entries = (
        combo_data['expected_answer']['combo_entries'] + 
        combo_data['expected_answer']['element1_entries'] + 
        combo_data['expected_answer']['element2_entries']
    )
    
    phase_points_expected = defaultdict(list)
    for entry in all_entries:
        if len(entry['elements']) == 1:
            composition = 0.0 if entry['elements'][0] == elements[0] else 100.0
        else:
            composition = entry['percentages'].get(elements[1], 0.0)
            
        for phase in entry['phases']:
            # Normalize the phase name for consistency
            normalized_phase = normalize_phase_name(phase)
            phase_points_expected[normalized_phase].append((composition, entry['temperature']))
            all_used_phases.add(normalized_phase)
    
    # Sort phases for consistent plotting order
    for phase in sorted(phase_points_expected.keys()):
        points = phase_points_expected[phase]
        x_vals, y_vals = zip(*points) if points else ([], [])
        scatter = ax2.scatter(x_vals, y_vals, c=[phase_to_color[phase]], label=phase,
                    s=160)
        if phase not in phase_handles:
            phase_handles[phase] = scatter
    
    # Add labels to expected diagram
    if add_labels:
        add_phase_labels(ax2, phase_points_expected, phase_to_color, 
                        label_strategy=label_strategy, font_size=label_font_size)
    
    # Set common labels and limits
    fig.text(0.5, 0.02, f"Composition {elements[0]} / {elements[1]} (%)", fontsize=36, ha='center')
    fig.text(0.04, 0.5, "Temperature (K)", fontsize=36, va='center', rotation='vertical')
    
    ax1.set_xlim(-5, 105)
    ax2.set_xlim(-5, 105)
    ax1.set_ylim(min_temp, max_temp)
    
    # Set tick parameters
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
    
    # Create a single legend for the entire figure with all phases
    # Sort phases to ensure consistent legend ordering
    sorted_phases = sorted(all_used_phases)
    handles = [phase_handles[phase] for phase in sorted_phases if phase in phase_handles]
    labels = [phase for phase in sorted_phases if phase in phase_handles]
    
    fig.legend(handles, labels, title="Phases", loc='upper center', 
               bbox_to_anchor=(0.5, 0), fontsize=28, title_fontsize=32,
               ncol=min(4, len(handles)), bbox_transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for the legend at the bottom
    plt.subplots_adjust(wspace=0.05)  # Reduce space between subplots
    
    # Save the figure
    suffix = "_labeled" if add_labels else ""
    plt.savefig(os.path.join(output_dir, f"{combo_key}_combined{suffix}.svg"),
                format='svg', bbox_inches='tight')
    plt.close()
    
    return True


def process_combined_phase_diagrams(input_file, output_dir, debug=False, add_labels=True,
                                   label_strategy='center', label_font_size=14):
    """Process combined phase diagrams for both generated and expected answers"""
    # Load data for both answer types at once
    data_results = load_data(input_file, answer_types=["generated_answer", "expected_answer"], debug=debug)
    
    # Group binary combinations for both answer types
    binary_combos = {}
    for answer_type, data in data_results.items():
        binary_combos[answer_type] = group_binary_combinations(
            data['entries_by_elements'], 
            data['single_element_entries']
        )
    
    # Collect all binary combos from both answer types
    all_combo_keys = set()
    for answer_type in binary_combos:
        all_combo_keys.update(binary_combos[answer_type].keys())
    
    print(f"Found {len(all_combo_keys)} unique binary combinations: {sorted(all_combo_keys)}")
    
    # Create a combined structure with both answer types for each combo
    binary_combos_combined = {}
    for combo_key in all_combo_keys:
        binary_combos_combined[combo_key] = {
            'generated_answer': binary_combos.get('generated_answer', {}).get(combo_key, {'combo_entries': [], 'element1_entries': [], 'element2_entries': []}),
            'expected_answer': binary_combos.get('expected_answer', {}).get(combo_key, {'combo_entries': [], 'element1_entries': [], 'element2_entries': []})
        }
    
    # Plot combined phase diagrams (each with its own color mapping)
    combined_count = 0
    
    for combo_key, combo_data in binary_combos_combined.items():
        if plot_combined_phase_diagram(combo_key, combo_data, output_dir, 
                                      add_labels=add_labels, 
                                      label_strategy=label_strategy,
                                      label_font_size=label_font_size):
            combined_count += 1
    
    print(f"Created {combined_count} combined phase diagrams")


# === Run the script ===
if __name__ == "__main__":
    input_path = "generated/phase_names.jsonl"
    output_path = "diagrams/binary_phase"
    
    # Enable debug mode to see what's happening
    # You can now control labeling options:
    # - add_labels: True/False to enable/disable labels
    # - label_strategy: 'center', 'highest', 'middle' 
    # - label_font_size: size of the label text
    process_combined_phase_diagrams(input_path, output_path, debug=True, 
                                   add_labels=True, label_strategy='center', 
                                   label_font_size=14)