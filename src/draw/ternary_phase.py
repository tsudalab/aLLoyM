import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import argparse

os.makedirs('diagrams/ternary_phase', exist_ok=True)

with open('generated/phase_names_ternary.jsonl') as f:
    jsonl_data = [json.loads(l) for l in f.readlines()]


def sort_phase_name(phase_name):
    """Sort phase components alphabetically within a phase name."""
    # Split by ' + ' to get individual phase components
    components = phase_name.split(' + ')
    # Sort the components alphabetically
    sorted_components = sorted(components)
    # Join them back with ' + '
    return ' + '.join(sorted_components)


def normalize_phase_name(phase_text):
    """
    Normalize a phase name by cleaning it and sorting any multi-phase descriptions.
    Handle different separators and formats.
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
        # Remove any empty parts after cleaning
        phase_parts = [part for part in phase_parts if part]
        return " + ".join(sorted(phase_parts))
    
    # For single phases or if no separators found, just sort if it contains '+'
    return sort_phase_name(phase_text)


def create_system_color_mapping(system_data, custom_phase_order=None):
    """Create a color mapping specific to phases in this system only"""
    system_phases = set()
    
    # Collect all unique phases from both generated_answer and expected_answer for this system
    for answer_type in ['generated_answer', 'expected_answer']:
        for entry in system_data[answer_type]:
            system_phases.add(entry['phase'])
    
    # Use custom order if provided, otherwise sort alphabetically
    if custom_phase_order:
        # Filter custom order to only include phases present in this system
        ordered_phases = [phase for phase in custom_phase_order if phase in system_phases]
        # Add any remaining phases not in custom order (sorted alphabetically)
        remaining_phases = sorted(system_phases - set(ordered_phases))
        system_phases = ordered_phases + remaining_phases
    else:
        system_phases = sorted(system_phases)
    
    # Create color mapping using rainbow colormap
    if len(system_phases) <= 1:
        phase_to_color = {list(system_phases)[0]: cm.rainbow(0.5)} if system_phases else {}
    else:
        phase_to_color = {phase: cm.rainbow(float(i) / (len(system_phases) - 1)) 
                         for i, phase in enumerate(system_phases)}
    
    return phase_to_color, system_phases


def extract_system_data(jsonl_data, system_elements):
    """Extract composition and phase data for a specific system"""
    system_data = {'generated_answer': [], 'expected_answer': []}
    
    for entry in jsonl_data:
        q_list = entry['user'].split()
        entry_system = [q_list[4], q_list[7], q_list[10]]
        
        if entry_system == system_elements:
            composition = [
                q_list[5].replace('(', '').replace(')', '').replace('%', ''), 
                q_list[8].replace('(', '').replace(')', '').replace('%', ''), 
                q_list[11].replace('(', '').replace(')', '').replace('%', '')
            ]
            
            for answer_type in ['generated_answer', 'expected_answer']:
                if answer_type in entry and entry[answer_type]:
                    original_phase = entry[answer_type].split('.')[0]
                    normalized_phase = normalize_phase_name(original_phase)
                    
                    system_data[answer_type].append({
                        'composition': composition,
                        'phase': normalized_phase
                    })
    
    return system_data


def plot_combined_ternary_diagram(system_elements, system_data, output_dir, custom_phase_order=None):
    """Plot combined ternary phase diagram with both generated and expected answers side by side"""
    
    # Check if we have data for both answer types
    has_generated = len(system_data['generated_answer']) > 0
    has_expected = len(system_data['expected_answer']) > 0
    
    if not (has_generated and has_expected):
        # Identify missing compositions
        gen_comps = {tuple(entry['composition']) for entry in system_data['generated_answer']}
        exp_comps = {tuple(entry['composition']) for entry in system_data['expected_answer']}
        
        missing_in_gen = exp_comps - gen_comps
        missing_in_exp = gen_comps - exp_comps
        
        if missing_in_gen:
            print(f"Missing in generated_answer for system {system_elements}:")
            for comp in sorted(missing_in_gen):
                print(f"  Composition: {comp} (Elements: {system_elements})")
        
        if missing_in_exp:
            print(f"Missing in expected_answer for system {system_elements}:")
            for comp in sorted(missing_in_exp):
                print(f"  Composition: {comp} (Elements: {system_elements})")
        
        return False  # Skip if either data type is missing
    
    # Create system-specific color mapping
    phase_to_color, system_phases = create_system_color_mapping(system_data, custom_phase_order)
    
    order_type = "custom order" if custom_phase_order else "alphabetical"
    print(f"System {'-'.join(system_elements)} color mapping ({order_type}) for {len(system_phases)} phases:")
    for i, phase in enumerate(system_phases):
        print(f"  {i+1}. {phase}")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    h = np.sqrt(3.0) * 0.5
    
    # Track all phases used for legend
    all_used_phases = set()
    
    # Function to setup ternary plot (matching original visual style)
    def setup_ternary_plot(ax, title):
        ax.set_aspect('equal', 'datalim')
        plt.sca(ax)  # Set current axis
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title, fontsize=16, pad=20)
        
        # Draw ternary grid (matching original)
        for i in range(1, 10):
            ax.plot([i/20.0, 1.0-i/20.0], [h*i/10.0, h*i/10.0], color='gray', lw=0.5)
            ax.plot([i/20.0, i/10.0], [h*i/10.0, 0.0], color='gray', lw=0.5)
            ax.plot([0.5+i/20.0, i/10.0], [h*(1.0-i/10.0), 0.0], color='gray', lw=0.5)
        
        # Draw ternary boundary (matching original)
        ax.plot([0.0, 1.0], [0.0, 0.0], 'k-', lw=2)
        ax.plot([0.0, 0.5], [0.0, h], 'k-', lw=2)
        ax.plot([1.0, 0.5], [0.0, h], 'k-', lw=2)
        
        # Add element labels (matching original positions and font size)
        #ax.text(0.45, h+0.02, str(system_elements[0]), fontsize=32)
        #ax.text(-0.2, -0.06, str(system_elements[1]), fontsize=32)
        #ax.text(1.03, -0.06, str(system_elements[2]), fontsize=32)
        
        # Add percentage labels (matching original)
        for i in range(1, 10):
            ax.text(0.5+(10-i)/20.0 + 0.01, h*(1.0-(10-i)/10.0), '%d0' % i, fontsize=20)
            ax.text((10-i)/20.0-0.05, h*(10-i)/10.0, '%d0' % i, fontsize=20)
            ax.text(i/10.0-0.03, -0.04, '%d0' % i, fontsize=20)
    
    # Function to plot data on ternary diagram
    def plot_ternary_data(ax, data, title):
        setup_ternary_plot(ax, title)
        
        # Group data by phase for plotting
        phase_points = defaultdict(list)
        for entry in data:
            comp = entry['composition']
            phase = entry['phase']
            
            # Convert to ternary coordinates (matching original calculation)
            x = (float(comp[0]) / 2 + float(comp[2])) / 100
            y = np.sqrt(3) * float(comp[0]) / 2 / 100
            
            phase_points[phase].append((x, y))
            all_used_phases.add(phase)
        
        # Plot points and create legend handles (matching original style)
        points = []
        labels = []
        
        for phase in sorted(phase_points.keys()):
            phase_coords = phase_points[phase]
            if phase_coords:
                x_vals, y_vals = zip(*phase_coords)
                color = phase_to_color.get(phase, 'gray')
                
                # Plot with original styling
                point = ax.scatter(x_vals, y_vals, c=[color], marker="o", s=60)
                
                if phase not in labels:
                    points.append(point)
                    labels.append(phase)
        
        return points, labels
    
    # Plot generated data (left subplot)
    points1, labels1 = plot_ternary_data(ax1, system_data['generated_answer'], "Generated Answer")
    
    # Plot expected data (right subplot)  
    points2, labels2 = plot_ternary_data(ax2, system_data['expected_answer'], "Expected Answer")
    
    # Combine points and labels for unified legend
    all_points = {}
    for i, label in enumerate(labels1):
        all_points[label] = points1[i]
    for i, label in enumerate(labels2):
        if label not in all_points:
            all_points[label] = points2[i]
    
    # Create legend with original styling
    sorted_phases = sorted(all_points.keys())
    legend_points = [all_points[phase] for phase in sorted_phases]
    
    #if legend_points:
    #    fig.legend(legend_points, sorted_phases, fontsize=5, loc='upper right', 
    #              bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save the figure
    system_name = "-".join(system_elements)
    #plt.savefig(f'{output_dir}/{system_name}_combined.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'{output_dir}/{system_name}_combined.png', format='png', dpi=600, bbox_inches='tight')

    plt.close()
    
    return True


def get_colormap_from_name(colormap_name):
    """Get matplotlib colormap from string name"""
    try:
        return getattr(cm, colormap_name)
    except AttributeError:
        print(f"Warning: Colormap '{colormap_name}' not found. Using 'rainbow' as default.")
        return cm.rainbow


def parse_phase_order(phase_order_str):
    """Parse comma-separated phase order string"""
    if not phase_order_str:
        return None
    return [phase.strip() for phase in phase_order_str.split(',')]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate ternary phase diagrams')
    parser.add_argument('--self_colormap', type=str, default=None,
                        help='Specify phase order for color mapping (comma-separated, e.g., "liquid,solid,gas")')
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Get custom phase order if specified
    custom_phase_order = parse_phase_order(args.self_colormap)
    if custom_phase_order:
        print(f"Using custom phase order: {custom_phase_order}")
    else:
        print("Using default alphabetical phase ordering")
    # Get unique systems
    system = []
    for entry in jsonl_data:
        q_list = entry['user'].split()
        system_elements = [q_list[4], q_list[7], q_list[10]]
        if system_elements not in system:
            system.append(system_elements)
    
    print(f"Found {len(system)} unique systems: {system}")
    
    # Process each system
    combined_count = 0
    for system_elements in system:
        system_data = extract_system_data(jsonl_data, system_elements)
        
        if plot_combined_ternary_diagram(system_elements, system_data, 'diagrams/ternary_phase', custom_phase_order):
            combined_count += 1
            print(f"Created combined diagram for system: {'-'.join(system_elements)}")
    
    print(f"Created {combined_count} combined ternary phase diagrams")