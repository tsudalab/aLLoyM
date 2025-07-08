import json

def generate_alloy_data(element1, element2, output_file):
    """
    Generate JSON lines for binary alloy phase diagram data.
    
    Args:
        element1: First element symbol (e.g., 'Ca')
        element2: Second element symbol (e.g., 'Sm' or 'La')
        output_file: Output filename
    """
    
    # Define the system message
    system_message = "You are an expert in phase diagrams, thermodynamics, and materials science, specializing in binary alloy systems."
    
    # Generate data
    data_lines = []
    
    # Temperature range: 140 K to 5000 K in 50 K intervals
    temperatures = range(140, 5001, 50)
    
    for temp in temperatures:
        # Fraction range: 0% to 100% in 2% increments
        for fraction in range(0, 101, 2):
            # Calculate percentages
            element1_percent = fraction
            element2_percent = 100 - fraction
            
            # Create the user message
            if element1_percent == 0:
                user_content = f"What phases form when {element2} (100%) are mixed at {temp} K? Answer phase names only."
            elif element2_percent == 0:
                user_content = f"What phases form when {element1} (100%) are mixed at {temp} K? Answer phase names only."
            else:
                user_content = f"What phases form when {element1} ({element1_percent}%) + {element2} ({element2_percent}%) are mixed at {temp} K? Answer phase names only."
            
            # Create the JSON structure
            json_line = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": "PLACEHOLDER_PHASE"  # This would be replaced with actual phase data
                    }
                ]
            }
            
            data_lines.append(json_line)
    
    # Write to file
    with open(output_file, 'w') as f:
        for line in data_lines:
            f.write(json.dumps(line) + '\n')
    
    print(f"Generated {len(data_lines)} data points for {element1}-{element2} system")
    print(f"Temperature range: 140 K to 5000 K (50 K intervals)")
    print(f"Composition range: 0% to 100% (2% increments)")
    print(f"Output saved to: {output_file}")

# Example usage:
if __name__ == "__main__":
    # Generate Ca-Sm system data
    generate_alloy_data('Calcium', 'Samarium', 'ca_sm_alloy_data.jsonl')
    generate_alloy_data('Calcium', 'Lanthanum', 'ca_la_alloy_data.jsonl')