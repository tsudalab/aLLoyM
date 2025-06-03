#!/bin/bash
#SBATCH --job-name=mistral
#SBATCH --output=log_%j.txt
#SBATCH --error=error_log_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 #A100:1
#SBATCH --time=100:00:00
#SBATCH --mem=32G


# Check if filename is given
if [ -z "$1" ]; then
    echo "Error: No Python script specified."
    exit 1
fi

FILE_NAME=$1

# Check if the arguments are given
if [ "$2" ]; then
    ARGUMENTS=${@:2}
fi

# --- Activate your virtual environment ---
echo "Activating venv..."
source ../../../../venvs/huggingface/bin/activate

# --- Dynamically select the strongest (freest) GPU ---
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n1 | cut -d',' -f1 | xargs)
echo "Using CUDA device(s): $CUDA_VISIBLE_DEVICES"

# --- Print and run your script ---
echo "Running training script: ${FILE_NAME}"
python3 "${FILE_NAME}" ${ARGUMENTS}
