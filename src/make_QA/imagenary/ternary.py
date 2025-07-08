import json

def generate_ternary_alloy_data(element1, element2, element3, temperature, output_file):
    """
    Generate JSON lines for ternary alloy phase diagram data at a specific temperature.
    
    Args:
        element1: First element symbol (e.g., 'Calcium')
        element2: Second element symbol (e.g., 'Samarium')
        element3: Third element symbol (e.g., 'Lanthanum')
        temperature: Temperature in Kelvin
        output_file: Output filename
    """
    
    # Define the system message
    system_message = "You are an expert in phase diagrams, thermodynamics, and materials science, specializing in ternary alloy systems."
    
    # Generate data
    data_lines = []
    
    # Generate compositions for ternary system
    # Using 5% increments to keep the dataset manageable
    # Compositions must sum to 100%
    for ca_percent in range(0, 101, 5):
        for sm_percent in range(0, 101 - ca_percent, 5):
            la_percent = 100 - ca_percent - sm_percent
            
            # Skip if lanthanum percentage is not a multiple of 5
            if la_percent % 5 != 0:
                continue
            
            # Create the user message
            if ca_percent == 100:
                user_content = f"What phases form when {element1} (100%) is heated to {temperature} K? Answer phase names only."
            elif sm_percent == 100:
                user_content = f"What phases form when {element2} (100%) is heated to {temperature} K? Answer phase names only."
            elif la_percent == 100:
                user_content = f"What phases form when {element3} (100%) is heated to {temperature} K? Answer phase names only."
            elif ca_percent == 0:
                user_content = f"What phases form when {element2} ({sm_percent}%) + {element3} ({la_percent}%) are mixed at {temperature} K? Answer phase names only."
            elif sm_percent == 0:
                user_content = f"What phases form when {element1} ({ca_percent}%) + {element3} ({la_percent}%) are mixed at {temperature} K? Answer phase names only."
            elif la_percent == 0:
                user_content = f"What phases form when {element1} ({ca_percent}%) + {element2} ({sm_percent}%) are mixed at {temperature} K? Answer phase names only."
            else:
                user_content = f"What phases form when {element1} ({ca_percent}%) + {element2} ({sm_percent}%) + {element3} ({la_percent}%) are mixed at {temperature} K? Answer phase names only."
            
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
    
    print(f"Generated {len(data_lines)} data points for {element1}-{element2}-{element3} ternary system")
    print(f"Temperature: {temperature} K")
    print(f"Composition range: 0% to 100% (5% increments)")
    print(f"Output saved to: {output_file}")

# Example usage:
if __name__ == "__main__":
    # Generate Ca-Sm-La ternary system data at 800 K
    generate_ternary_alloy_data('Calcium', 'Samarium', 'Lanthanum', 800, 'ca_sm_la_ternary_800k.jsonl')