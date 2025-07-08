#!/bin/bash

# Exit on error
set -x

# Step 1: Make QA
echo ">> Creating QA datasets..."
unzip dataset/CPDDB_data.zip
python3 src/make_QA/01_make_full_QA.py
python3 src/make_QA/02_make_phase_names_QA.py
python3 src/make_QA/03_make_reverse_QA.py
python3 src/make_QA/04_make_multiple_choice_QA.py
python3 src/make_QA/05_combine_QA.py
python3 src/make_QA/06_shuffle_QA.py

# Step 2: Run mistral in parallel
echo ">> Running mistral for each split type in parallel..."

for raw_or_multi in raw multi; do
    for split_type in split_by_file split_random; do
        (
            mistral_LLM_dir="dataset/${raw_or_multi}/${split_type}/mistral_LLM"
            mkdir -p "$mistral_LLM_dir"
            cd "$mistral_LLM_dir"

            # Submit fine-tuning job
            mkdir -p logs
            JOB_ID=$(sbatch --parsable ../../../../src/run_with_GPU.sh ../../../../src/mistral/finetune.py mistral)
            echo ">> Submitted fine-tuning job $JOB_ID for $split_type"

            # Wait for fine-tuning to finish
            while squeue -j "$JOB_ID" > /dev/null 2>&1 && squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
                echo ">> Waiting for fine-tuning job $JOB_ID to finish..."
                sleep 360
            done

            echo ">> Fine-tuning completed for $split_type"

            # Back to split dir (e.g., dataset/split_by_file)
            mkdir -p generated

            QA_types="full phase_names reverse"

            for QA_type in $QA_types; do
                (
                    mkdir -p logs  # Ensure logs/ exists before sbatch
                    GEN_JOB_ID=$(sbatch --parsable ../../../../src/run_with_GPU.sh ../../../../src/mistral/generate.py "$QA_type")
                    echo ">> Submitted generation job $GEN_JOB_ID for $split_type with $QA_type"

                    while squeue -j "$GEN_JOB_ID" > /dev/null 2>&1 && squeue -j "$GEN_JOB_ID" | grep -q "$GEN_JOB_ID"; do
                        echo ">> Waiting for generation job $GEN_JOB_ID ($split_type/$QA_type)..."
                        sleep 180
                    done
                    echo ">> Generation completed for $split_type with $QA_type"

                    if [ "$raw_or_multi" == "raw" ]; then
                        if [ "$QA_type" == "full" ]; then
                            python3 ../../../../src/score/score_exact_match.py "$QA_type"
                        else
                            python3 ../../../../src/score/score_${QA_type}.py
                        fi
                    else
                        python3 ../../../../src/score/score_exact_match.py "$QA_type"
                    fi

                    mkdir -p logs  # Ensure logs/ exists again before next sbatch
                    UNFINE_JOB_ID=$(sbatch --parsable ../../../../src/run_with_GPU.sh ../../../../src/mistral/generate_unfinetuned.py "$QA_type")
                    echo ">> Submitted unfinetuned generation job $UNFINE_JOB_ID for $split_type with $QA_type"

                    # Wait for unfinetuned generation to finish
                    while squeue -j "$UNFINE_JOB_ID" > /dev/null 2>&1 && squeue -j "$UNFINE_JOB_ID" | grep -q "$UNFINE_JOB_ID"; do
                        echo ">> Waiting for unfinetuned generation job $UNFINE_JOB_ID ($split_type/$QA_type)..."
                        sleep 180
                    done
                    echo ">> Unfinetuned generation completed for $split_type with $QA_type"
                ) &
                sleep 60*60
            done
            wait
        ) &
    done
done

wait
echo ">> All fine-tuning, generation, and scoring jobs complete."

# Also score shuffled_negative_control
for split_type in split_by_file split_random; do
    for QA_type in full phase_names reverse; do
        (
            cd "dataset/raw/${split_type}/shuffled_negative_control"
            if [ "$QA_type" == "full" ]; then
                python3 ../../../../src/score/score_exact_match.py "$QA_type"
            else
                python3 ../../../../src/score/score_${QA_type}.py
            fi
        )
    done
done
echo ">> Scoring shuffled_negative_control completed"
