#!/bin/bash
#SBATCH --job-name=analysis          # Job name
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
echo "Starting analysis..."

python Instruction/length_analysis_full.py models/Llama-2-7b-chat-hf > log_length_llama_2_7.txt 2>&1
#python Instruction/length_analysis_full.py models/Meta-Llama-3-8B-Instruct > log_length_meta_llama_3_8.txt 2>&1
#python Instruction/length_analysis_full.py models/Llama-3.1-8B-Instruct > log_length_llama_3_1.txt 2>&1
#python Instruction/length_analysis_full.py models/llama-3.2-3b-instruct > log_length_llama_3_2.txt 2>&1

echo "Analysis finished at $(date)"
