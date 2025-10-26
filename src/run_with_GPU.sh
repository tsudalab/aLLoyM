#!/bin/bash
#SBATCH --job-name=mistral
#SBATCH --output=logs/log_%j.txt
#SBATCH --error=logs/error_log_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 #A100:1
#SBATCH --time=250:00:00
#SBATCH --mem=32G

# --- Check if filename is given ---
if [ -z "$1" ]; then
    echo "Error: No Python script specified."
    exit 1
fi

SCRIPT="$1"
shift  # Now $@ contains all the arguments to the script

# --- Activate your virtual environment ---
echo "Activating venv..."
source ../../.env/bin/activate

# --- Dynamically select the strongest (freest) GPU ---
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n1 | cut -d',' -f1 | xargs)
echo "Using CUDA device(s): $CUDA_VISIBLE_DEVICES"

set -a
source ../.env.tokens
set +a

# --- Run your script with timing ---
echo "Running training script: ${SCRIPT} with args $@"
echo "Start time: $(date)"
time python3 "${SCRIPT}" "$@"
echo "End time: $(date)"