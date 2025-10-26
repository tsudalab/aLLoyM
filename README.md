# aLLoyM

## Paper

https://arxiv.org/abs/2507.22558

## Setup to run finetune or inference with `unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit`

1. Needs GPU with VRAM>8GB (tested for A100 80GB).
2. Run setup_llm_env.sh (ested for CUDA12.4) or manually install all necessary libraries.
3. Create your Huggingface account and paste the personal token under `.env.tokens`
4. Optional: Create your wandb account and paste the personal token under `.env.tokens` to track ...

## Finetune & inference with your own Q&As (any kind of data) or demo data

1. `cd YOUR_PROJECT_DIR_RIGHT_UNDER_HERE # demo/ for demo data`
2. Run finetune file like this: bash ../src/run_with_GPU.sh ../src/finetune.py and wait.
3. Run inference file like this: bash ../src/run_with_GPU.sh ../src/generate.py and wait.

Running run_with_GPU.sh will automatically activate .env.
To run heavy tasks, you may want to use tmux etc.


## Finetune & inference with your own phase diagram data

1. Add your .dat data under dataset/CPDDB_data. Notes: 
    a. Can only take in phase names from phase_list in config.py
    b. View demo/example.dat for the format
2. Run data processing with the single command below:

```
chmod +x run_all.sh
./run_all.sh
```