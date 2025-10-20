#!/bin/bash
#SBATCH --job-name=inference         # Job name
#SBATCH --output=%j.out              # Output file
#SBATCH --error=%j.err               # Error file
#SBATCH --partition=dgx              # Partition name
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=16           # CPUs per task
#SBATCH --mem=48G                    # Memory allocation
#SBATCH --time=24:00:00              # Time limit (4 hours)

# 1. Load conda to activate the environment
source ~/your_path_to_conda

# 2. Activate your conda environment
conda activate /your_path_to_your_env

# 3. Set CUDA_HOME environment variable
export CUDA_HOME="$CONDA_PREFIX"
echo $CUDA_HOME

# 4. Login to Hugging Face and Weights & Biases
huggingface-cli login --token hf_xxxx
wandb login XXXX

# 5. Set CUDA_VISIBLE_DEVICES if needed
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 6. Run the analysis script with fabric
echo "Starting inference..."

  python infer_entities.py \
  --base_model_path models/Meta-Llama-3-8B-Instruct \
  --lora_dir outputsToConsider/Meta-Llama-3-8B-Instruct-lora-optimal-exp18/checkpoint-14 \
  --output_path final_predictions_exp18_1.jsonl \
  --max_new_tokens 1491 \
  --temperature 0.30114121830086255 \
  --repetition_penalty 1.6629107730290587 \
  --num_beams 7 \
  --top_p 0.8248772432711813 \
  --do_sample \
  --no_repeat_ngram_size 6 \
  --search_method none > outputsToConsider/log_final_inference_18_1.txt 2>&1

echo "Inference finished at $(date)"