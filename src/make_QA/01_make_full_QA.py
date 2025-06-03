import os
import re
import sys
import random
import json
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import glob
import base64
from io import BytesIO
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import element_dict, phase_list, phase_name_map

class PhaseDataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.temperature = None
        self.elements = {}
        self.phases = {}
        self.phase = {}
        self.unique_phase_names = set()  # Track all phase names
    
    def load_data(self):
        """Reads the data file, processes it, and extracts relevant information."""
        df = self._load_file()
        self.temperature = self.clean_numeric_data(df['T'])
        self._extract_elements(df)
        self._extract_phases(df)

    def _load_file(self):
        """Reads the data file and cleans it up."""
        df = pd.read_csv(self.file_path, sep='\t')
        df.columns = df.columns.str.strip()  # Clean up column names
        df = df.drop(0).reset_index(drop=True)
        return df

    def _extract_elements(self, df):
        """Extracts element compositions from the dataframe."""
        element_cols = [col for col in df.columns if col.startswith('x(') and col.endswith(')')]
        for col in element_cols:
            element = col[2:-1]  # Extract element name
            self.elements[element] = self.clean_numeric_data(df[col])
        
        # Sort elements by length (descending) to handle multi-character elements first
        # Then sort alphabetically within same length
        self.elements = dict(sorted(self.elements.items(), key=lambda x: (-len(x[0]), x[0])))
    
    def _extract_phases(self, df):
        """Extracts phase fractions and compositions from the dataframe."""
        phase_cols = [col for col in df.columns if col.startswith('f(@')]

        for col in phase_cols:
            raw_phase_name = col[3:-1].upper()  # Remove 'f(@' and ')'
            phase_name = phase_name_map.get(raw_phase_name, raw_phase_name)

            # Add to our set of all phase names
            self.unique_phase_names.add(phase_name)
            
            # Get the fraction data from the dataframe
            fraction_data = self.clean_numeric_data(df[col]).replace(np.nan, 0)*100

            # Process the phase name to extract elements and their ratios
            element_counts, ratio = self._split_elements_with_ratio(phase_name)
            
            if ratio:
                self.phases[phase_name] = {'fraction': fraction_data, 'elements': element_counts, 'ratio': ratio}
            else:
                self.phases[phase_name] = {'fraction': fraction_data}

    def _split_elements_with_ratio(self, phase_name: str) -> Tuple[Dict[str, int], Optional[str]]:
        """
        Extracts elements and their ratio from a phase name.
        
        Args:
            phase_name: The name of the phase to process.
            
        Returns:
            Tuple containing:
            - Dict mapping element names to their counts
            - String representation of the ratio, or None if no elements were found
        """
        # Check if phase is in special phase list
        if phase_name in phase_list:
            return {}, None
            
        # Get a list of all elements sorted by length (descending) to prioritize multi-character elements
        sorted_elements = sorted(self.elements.keys(), key=len, reverse=True)
        
        # Initialize dictionary to store element counts
        element_counts = defaultdict(int)
        
        # Working copy of the phase name to modify as we process it
        remaining = phase_name
        
        # Track the index in the remaining string
        i = 0
        
        while i < len(remaining):
            # Try to match an element at the current position
            matched = False
            
            for element in sorted_elements:
                # Case insensitive comparison
                if remaining[i:i+len(element)].upper() == element.upper():
                    # Found a match, move index past this element
                    i += len(element)
                    
                    # Look for a number after the element
                    count_match = re.match(r"^(\d+)", remaining[i:])
                    if count_match:
                        count = int(count_match.group(1))
                        i += len(count_match.group(1))  # Move past the number
                    else:
                        count = 1  # Default count if no number specified
                    
                    element_counts[element] += count
                    matched = True
                    break
            
            # If we couldn't match any element at this position, skip this character
            if not matched:
                i += 1
        
        # Create the ratio string if we found any elements
        if element_counts:
            # Use original sorting of elements when creating ratio
            ratio = " : ".join(f"{element_counts[e]}" for e in sorted_elements if e in element_counts)
            return dict(element_counts), ratio
        else:
            return {}, None

    def get_processed_data(self) -> List[Dict[str, Any]]:
        """Returns processed data in a structured format."""
        data_list = []
        for i in range(len(self.temperature)):
            data_point = {
                'temperature': self.temperature.iloc[i],
                'elements': {k: v.iloc[i] for k, v in self.elements.items()},
                'phases': {}
            }
            
            # Add phase data with proper structure
            for phase_name, phase_data in self.phases.items():
                phase_info = {'fraction': phase_data['fraction'].iloc[i]}
                if 'elements' in phase_data and 'ratio' in phase_data:
                    phase_info['elements'] = phase_data['elements']
                    phase_info['ratio'] = phase_data['ratio']
                data_point['phases'][phase_name] = phase_info
                
            data_list.append(data_point)
        return data_list

    def clean_numeric_data(self, data: pd.Series) -> pd.Series:
        """Cleans a pandas Series by removing non-numeric characters."""
        cleaned_data = data.replace(r'[^0-9.-]', '', regex=True)
        return pd.to_numeric(cleaned_data, errors='coerce')


def create_examples(processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create training examples for OpenAI fine-tuning in the specified format.
    
    Args:
    - processed_data (List[Dict[str, Any]]): The processed data from the PhaseDataProcessor.
    
    Returns:
    - List[Dict[str, Any]]: A list of dictionaries formatted for OpenAI fine-tuning with system, user, and assistant messages.
    """
    training_examples = []
    unique_phase_names = set()  # To collect all unique phase names
    missing_elements = set()  # To track elements not found in element_dict
    
    # Use phase_list from config.py for special phase names
    special_phase_names = phase_list
    
    for i, data_point in enumerate(processed_data):
        # Extract relevant information
        temperature = data_point['temperature']
        phases = data_point['phases']
        if not phases:
            print(f"Warning: No phases found for data point {i}")
            continue
            
        elements = data_point['elements']

        # Format element composition string
        element_composition_parts = []
        for element, composition in elements.items():
            element = element.strip().upper()  # Ensure element symbol is uppercase
            if not pd.isna(composition):
                if element not in element_dict:
                    missing_elements.add(element)
                    print(f"Error: Element '{element}' not found in element_dict")
                    continue
                element_composition_parts.append(f"{element_dict.get(element)} ({composition:.1f}%)")
      
        if not element_composition_parts:
            print(f"Warning: No valid elements found for data point {i}")
            continue
            
        element_str = " and ".join(element_composition_parts)
        
        # Format phase information
        phase_descriptions = []
        
        for phase_name, phase_info in phases.items():
            # Check if this is a special phase name that should be preserved
            is_special_phase = phase_name in special_phase_names
            
            # Split by # but preserve the full phase name for special phases
            if is_special_phase:
                base_phase_name = phase_name
            else:
                base_phase_name = phase_name.split('#')[0]  # Remove after '#' only for non-special phases
            
            fraction = phase_info['fraction'] # e.g. 87.99997
            
            # Only include phases with non-zero fractions
            if fraction > 0:
                # Check if the phase has elements and ratio
                if 'elements' in phase_info and 'ratio' in phase_info and not is_special_phase:
                    ratio = phase_info['ratio']
                    phase_elements = []
                    
                    for element, count in phase_info['elements'].items():
                        element = element.strip().upper()  # Ensure element symbol is uppercase
                        if element not in element_dict:
                            missing_elements.add(element)
                            continue
                        phase_elements.append(element_dict.get(element))
                    
                    phase_elements_str = " : ".join(phase_elements)
                    phase_descriptions.append(f"{fraction:.0f}% SOLID with composition ratio {phase_elements_str} = {ratio}")
                else:
                    # Use the original phase name with structural designation if present
                    phase_descriptions.append(f"{fraction:.0f}% {base_phase_name}")
                    unique_phase_names.add(base_phase_name)

        if not phase_descriptions:
            print(f"Warning: No valid phase descriptions for data point {i}")
            continue
            
        # Create the final phase description
        phase_str = " and ".join(phase_descriptions)
            
        # Format user and assistant messages
        user_message = f"What phases form when {element_str} are mixed at {temperature} K?"
        assistant_message = phase_str
        
        # Create example in the OpenAI fine-tuning format
        example = {
            "messages": [
                {"role": "system", "content": "You are an expert in phase diagrams, thermodynamics, and materials science, specializing in alloy systems."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        
        training_examples.append(example)
    
    # Report missing elements
    if missing_elements:
        print(f"\nError: The following {len(missing_elements)} elements were not found in element_dict:")
        for element in sorted(missing_elements):
            print(f"  - {element}")
    
    # Print list of all phase names
    print(f"\nList of all phase names found: {sorted(unique_phase_names)}")
    
    return training_examples

def save_file(examples: List[Dict[str, str]], output_file: str = "training_data.jsonl") -> str:
    """
    Save training examples to a JSONL file for OpenAI fine-tuning.
    
    Args:
        examples: List of training examples
        output_file: Path to save the JSONL file
        
    Returns:
        Path to the saved file
    """
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(examples)} training examples to {output_file}")
    return output_file

def split_files(file_list: List[str], train_ratio=0.8):
    """
    Splits the list of .dat files into training and validation sets.

    Args:
        file_list: List of .dat file names.
        train_ratio: Proportion of files to use for training (default 80%).

    Returns:
        train_files, val_files: Lists of file names for training and validation.
    """
    if not file_list:
        return [], []

    random.shuffle(file_list)  # Shuffle with fixed seed
    split_idx = int(len(file_list) * train_ratio)
    return file_list[:split_idx], file_list[split_idx:]

def process_dat_files(data_dir: str, file_list: List[str]) -> List[Dict[str, Any]]:
    """
    Processes a subset of .dat files in a directory and returns a combined list of processed data.

    Args:
        data_dir: Directory containing .dat files.
        file_list: List of file names to process.

    Returns:
        all_data: List of all processed data points.
    """
    all_data = []
    
    if not file_list:
        print("⚠️ No files to process.")
        return all_data

    for file_name in tqdm(file_list, desc="Processing .dat files"):
        file_path = os.path.join(data_dir, file_name)
        try:
            processor = PhaseDataProcessor(file_path)
            processor.load_data()
            file_data = processor.get_processed_data()

            if not file_data:
                print(f"⚠️ No data found in {file_name}, skipping...")
                continue

            all_data.extend(file_data)

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")  # Log error, don't crash

    return all_data

def process_all_dat_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Process all .dat files in the directory at once.
    
    Args:
        data_dir: Directory containing .dat files.
        
    Returns:
        all_data: List of all processed data points.
    """
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".dat")]
    return process_dat_files(data_dir, file_list)

def split_data_points(all_data: List[Dict[str, Any]], train_ratio=0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split all data points randomly into training and validation sets.
    
    Args:
        all_data: List of all processed data points.
        train_ratio: Proportion of data points to use for training.
        
    Returns:
        train_data, val_data: Lists of data points for training and validation.
    """
    random.shuffle(all_data)  # Shuffle with fixed seed
    split_idx = int(len(all_data) * train_ratio)
    return all_data[:split_idx], all_data[split_idx:]

def run_pipeline(data_directory: str, split_method="split_by_file", train_ratio=0.8, seed=42):
    """
    Run the full fine-tuning pipeline with flexible splitting options.

    Args:
        data_directory: Directory containing .dat files.
        split_method: How to split the data - "split_by_file" (split at file level) or "split_random" (split individual data points)
        train_ratio: Proportion of data to use for training (default 80%).
        seed: Random seed for reproducibility.
    """
    random.seed(seed)  # Ensure consistent shuffle across runs
    
    # Step 1: Get all .dat files
    file_list = [f for f in os.listdir(data_directory) if f.endswith(".dat")]

    if not file_list:
        print("⚠️ No .dat files found in the directory. Exiting pipeline.")
        return

    # Different splitting approaches
    if split_method.lower() == "split_by_file":
        # Original approach: Split at file level
        print(f"Using file-level splitting (train_ratio={train_ratio})...")
        train_files, val_files = split_files(file_list, train_ratio)
        
        print(f"✅ Training files: {len(train_files)}")
        print(f"✅ Validation files: {len(val_files)}")
        
        # Process files separately
        print("Processing training files...")
        train_data = process_dat_files(data_directory, train_files)
        
        print("Processing validation files...")
        val_data = process_dat_files(data_directory, val_files)
        
    elif split_method.lower() == "split_random":
        # New approach: Process all files first, then randomly split data points
        print(f"Using random data-point splitting (train_ratio={train_ratio})...")
        print("Processing all .dat files...")
        all_data = process_all_dat_files(data_directory)
        
        print(f"Total data points: {len(all_data)}")
        print("Splitting data points into training and validation sets...")
        train_data, val_data = split_data_points(all_data, train_ratio)
        
    else:
        print(f"❌ Unknown split method: {split_method}. Valid options are 'file' or 'random'.")
        return

    print(f"✅ Training data points: {len(train_data)}")
    print(f"✅ Validation data points: {len(val_data)}")

    # Step 4: Create training examples
    print("Creating training examples...")
    training_examples = create_examples(train_data)
    validation_examples = create_examples(val_data)

    # Step 5: Save training and validation data to files
    output_dir = f"{'/'.join(data_directory.split('/')[:-1])}/raw/{split_method}"
    os.makedirs(f"{output_dir}/training", exist_ok=True)
    training_file_path = save_file(training_examples, f"{output_dir}/training/full.jsonl")
    os.makedirs(f"{output_dir}/validation", exist_ok=True)
    validation_file_path = save_file(validation_examples, f"{output_dir}/validation/full.jsonl")

    print(f"✅ Training data saved to: {training_file_path}")
    print(f"✅ Validation data saved to: {validation_file_path}")

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning data pipeline")
    parser.add_argument(
        "--data_dir", 
        help="Directory containing input files", 
        default="dataset/CPDDB_data"
    )
    parser.add_argument(
        "--ratio", 
        type=float, 
        default=0.8,
        help="Train/validation split ratio (default: 0.8)"
    )
    args = parser.parse_args()

    for split in ["split_by_file", "split_random"]:
        print(f"\nRunning pipeline for split: {split}")
        run_pipeline(
            data_directory=args.data_dir,
            split_method=split,
            train_ratio=args.ratio
        )