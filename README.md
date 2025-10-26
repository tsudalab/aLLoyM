# aLLoyM

## Paper

**[aLLoyM: Large Language Model for Alloy Phase Diagram Reasoning](https://arxiv.org/abs/2507.22558)**

---

## Setup for Fine-tuning or Inference

(using `unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit`)

1. **GPU Requirement**
   Use a GPU with more than 8 GB of VRAM (tested on NVIDIA A100 80 GB).

2. **Environment Setup**

   * Run `setup_llm_env.sh` (tested with CUDA 12.4), or
   * Manually install the required Python libraries listed in the script.

3. **Hugging Face Authentication**

   * Create a Hugging Face account.
   * Paste your personal access token into `.env.tokens`.

4. **Optional: Weights & Biases Tracking**

   * Create a [wandb](https://wandb.ai/) account.
   * Add your wandb token to `.env.tokens` to enable experiment tracking.

---

## Fine-tuning and Inference with Custom Q&A Data

(or demo datasets)

1. Move into your project folder (for example, the `demo/` directory):

   ```bash
   cd demo/
   ```

2. Run fine-tuning:

   ```bash
   bash ../src/run_with_GPU.sh ../src/finetune.py
   ```

3. Run inference:

   ```bash
   bash ../src/run_with_GPU.sh ../src/generate.py
   ```

`run_with_GPU.sh` automatically activates the `.env` virtual environment.
For long runs, use `tmux` or `screen` to keep the session active.

---

## Fine-tuning and Inference with Phase-Diagram Data

1. **Prepare Your Data**

   * Place `.dat` files under `dataset/CPDDB_data/`.
   * Requirements:

     * Phase names must appear in `phase_list` in `config.py`.
     * Data format should follow `demo/example.dat`.

2. **Run the Complete Data Processing Pipeline**

   ```bash
   chmod +x run_all.sh
   ./run_all.sh
   ```