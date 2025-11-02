# aLLoyM

## Links

Paper: [aLLoyM: A Large Language Model for Alloy Phase Diagram Prediction](https://arxiv.org/abs/2507.22558)

Pretrained model: [Playingyoyo/aLLoyM](https://huggingface.co/Playingyoyo/aLLoyM/tree/main)

Dataset: [Playingyoyo/aLLoyM-dataset](https://huggingface.co/datasets/Playingyoyo/aLLoyM-dataset/tree/main)

---

## Setup for Fine-tuning or Inference

(using `unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit`)

1. **GPU Requirement**
   Use a GPU with at least **8 GB VRAM** (tested on **NVIDIA A100 80 GB**).

2. **Environment Setup**

   * Run `setup_llm_env.sh` (tested with **CUDA 12.4**), or
   * Manually install the required Python packages listed in the script.

3. **Hugging Face Authentication**

   * Create a Hugging Face account.
   * Paste your personal access token into `.env.tokens`.

4. **(Optional) Weights & Biases Tracking**

   * Create a [wandb](https://wandb.ai/) account.
   * Add your wandb token to `.env.tokens` to enable experiment tracking.

---

## Fine-tuning and Inference with Custom Q&A Data

(or demo datasets)

1. Move into your project folder (e.g., the `demo/` directory), create a new directory `mistral/`, and move into it:

   ```bash
   cd demo/
   mkdir mistral/
   cd mistral/
   ```

2. Run fine-tuning:

   ```srun
   bash ../../src/run_with_GPU.sh ../../src/mistral/finetune.py
   ```

3. Run inference:

   ```srun
   bash ../../src/run_with_GPU.sh ../../src/generate.py
   ```

`run_with_GPU.sh` automatically activates the `.env` virtual environment.
For long training or inference sessions, use `tmux` or `screen` to keep the process running in the background.

---

## Fine-tuning and Inference with Phase Diagram Data

1. **Prepare Your Data**

   * Place your `.dat` files under `dataset/CPDDB_data/`.
   * Requirements:

     * Phase names must appear in `phase_list` in `config.py`.
     * The data format should match `demo/example.dat`.

2. **Run the Complete Data Processing Pipeline**

   ```bash
   chmod +x run_all.sh
   ./run_all.sh
   ```
