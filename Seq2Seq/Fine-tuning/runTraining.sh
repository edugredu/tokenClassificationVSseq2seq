#!/bin/bash
#SBATCH --job-name=fineTuning        # Job name
#SBATCH --output=%j.out              # Output file
#SBATCH --error=%j.err               # Error file
#SBATCH --partition=your_partition   # Partition name
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=16           # CPUs per task
#SBATCH --mem=48G                    # Memory allocation
#SBATCH --time=48:00:00              # Time limit

echo "Starting the job in $(hostname) at $(date)"

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

# 6. Run the training script with fabric
echo "Starting training..."

python llama3_pharmacoNER.py > output/outputTrainingModel.txt 2>&1

echo "Training finished at $(date)"