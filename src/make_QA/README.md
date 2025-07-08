## Pipeline Overview

The QA generation pipeline processes binary and ternary alloy system data through multiple stages to create comprehensive question-answer datasets with different formats and complexities for machine learning training.
The scripts must be run in numerical order (01 → 02 → 03 → 04 → 05 → 06) as each script depends on outputs from previous stages.

## Scripts

### 1. `01_make_full_QA.py`
**Purpose**: Creates the initial short-answer comprehensive QA dataset from phase diagram data
1. Make two directories for each split method: extrapolation (`split_by_file`) or interpolatio (`random`)
2. Processes binary and ternary alloy system data files (.dat format)
3. Extracts element compositions, phase fractions, and temperature data
4. Generates question-answer pairs about phase formation at specific temperatures
- **Input**: .dat files containing phase diagram data with temperature, composition, and phase fraction columns
- **Output**: JSONL files with structured QA pairs in OpenAI fine-tuning format
- **Key Features**:
  - Preserves multiple phases with "+" separator
- **Usage**: `python3 src/make_QA/01_make_full_QA.py`

The next files will run for both the extrapolation (`split_by_file`) and interpolation (`split_random`) directories.
   
### 2. `02_make_phase_names_QA.py`
**Purpose**: Generates short-answer QA pairs focused on phase names and terminology
- Processes the full QA dataset to extract only phase names from answers
- **Input**: `full.jsonl` files from step 1 (both training and validation sets)
- **Output**: `phase_names.jsonl` files with simplified phase name answers
- **Key Features**:
  - Handles two answer formats: "X% SOLID with composition ratio..." and "X% PHASE_NAME"
  - Preserves multiple phases with "+" separator
  - Adds "Answer phase names only." to user questions to avoid LLM confusing it with the full QA
  - Processes both split_by_file and split_random datasets
- **Transformation Example**:
  - Original: "65% LIQUID + 35% SOLID with composition ratio Aluminum : Copper = 2 : 3."
  - Simplified: "LIQUID + SOLID."
- **Usage**: `python3 src/make_QA/02_make_phase_names_QA.py`

### 3. `03_make_reverse_QA.py`
**Purpose**: Creates reverse-formatted QA pairs by flipping question-answer relationships
- Transforms phase formation questions into condition-seeking questions
- Uses phase names as input to ask about formation conditions
- Extracts element compositions from original questions to create new answers
- **Input**: `phase_names.jsonl` files from step 2
- **Output**: `reverse.jsonl` files with reversed question-answer pairs
- **Key Features**:
  - Reverses the Q&A relationship: phases → conditions instead of conditions → phases
  - Extracts elements using regex pattern matching for "Element (X%)" format
  - Cleans user messages by removing first 4 words and filtering out "are"
  - Creates new questions in format: "Under what condition do [elements] form [phases]?"
- **Transformation Example**:
  - Original Q: "What phases form when Aluminum (45%) + Copper (55%) are mixed at 800 K?"
  - Original A: "LIQUID + SOLID."
  - Reverse Q: "Under what condition do Aluminum + Copper form LIQUID + SOLID?"
  - Reverse A: "Aluminum (45%) + Copper (55%) mixed at 800 K."
- **Usage**: `python3 src/make_QA/03_make_reverse_QA.py`

### 4. `04_make_multiple_choice_QA.py`
**Purpose**: Converts QA pairs into multiple choice format with intelligent distractor generation
- Transforms existing QA pairs into multiple choice questions with 4 options (a, b, c, d)
- Generates context-appropriate distractors based on file type and content
- **Input**: All `.jsonl` files from previous steps (full, phase_names, reverse)
- **Output**: Multiple choice format files in `dataset/multi/` directory structure
- **Output Format**: JSONL with system/user/assistant message structure
  - System: Expert phase diagram specialist role
  - User: Question with all 4 choices formatted as "(a) option1 (b) option2..."
  - Assistant: Single letter answer (a, b, c, or d) without brackets
- **Usage**: `python3 src/make_QA/04_make_multiple_choice_QA.py`

### 5. `05_combine_QA.py`
**Purpose**: Merges all generated QA datasets
- **Usage**: `python3 src/make_QA/05_combine_QA.py`

### 6. `06_shuffle_QA.py`
**Purpose**: Randomizes the combined QA dataset (not used in the paper because it reverse is difficult to score)
- Shuffles the order of questions to score negative control
- **Usage**: `python3 src/make_QA/06_shuffle_QA.py`

## Input Data Format

The pipeline expects `.dat` files with tab-separated columns containing:
- `T`: Temperature values
- `x(Element)`: Element composition columns (e.g., `x(Al)`, `x(Cu)`)
- `f(@Phase)`: Phase fraction columns (e.g., `f(@LIQUID)`, `f(@SOLID)`)

## Output

The pipeline generates various QA datasets including:
- **Full QA pairs**: Complete question-answer pairs about phase formation
  - Questions: "What phases form when [elements] are mixed at [temperature] K?"
  - Answers: "[fraction]% [phase_name] + [fraction]% [phase_name]..."
- Phase-specific terminology questions
- Reverse-formatted questions
- Multiple choice questions
- Combined and shuffled final dataset

### Example QA Pair
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in phase diagrams, thermodynamics, and materials science, specializing in binary alloy systems."
    },
    {
      "role": "user", 
      "content": "What phases form when Aluminum (45%) + Copper (55%) are mixed at 800 K?"
    },
    {
      "role": "assistant",
      "content": "65% LIQUID + 35% SOLID with composition ratio Aluminum : Copper = 2 : 3."
    }
  ]
}
```

### Example Multiple Choice QA Pair
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in phase diagrams, thermodynamics, and materials science, specializing in alloy systems. Answer the multiple choice question with the letter of the correct option without brackets. Do not provide any explanations or additional information."
    },
    {
      "role": "user",
      "content": "What phases form when Aluminum (45%) + Copper (55%) are mixed at 800 K? (a) 100% LIQUID (b) 65% LIQUID + 35% SOLID with composition ratio Aluminum : Copper = 2 : 3 (c) 100% SOLID (d) 80% LIQUID + 20% SOLID"
    },
    {
      "role": "assistant",
      "content": "b"
    }
  ]
}
```

## Dependencies

- pandas
- numpy
- matplotlib
- tqdm
- argparse
- concurrent.futures
- Custom modules: `config.py` (contains `element_dict`, `phase_list`, `phase_name_map`)

## Configuration

Key configuration parameters in `config.py`:
- `element_dict`: Maps element symbols to full element names
- `phase_list`: List of special phase names to preserve
- `phase_name_map`: Maps raw phase names to standardized names

---
