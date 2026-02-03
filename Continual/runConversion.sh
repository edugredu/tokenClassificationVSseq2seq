#!/bin/bash
#SBATCH --job-name=converter         # Job name
#SBATCH --output=%j.out              # Output file
#SBATCH --error=%j.err               # Error file
#SBATCH --partition=your_partition   # Partition name
#SBATCH --gres=gpu:2                 # Request 2 GPUs
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

# 6. Run the conversion script
python convert_fabric_to_hf_models.py --config_model llama_model_3_2.json --checkpoint_path outModel/fda/meta-llama/Llama-3.2-3B/iter-165252-ckpt.pth --output_dir llama3_2/modelFDA_llama_3_2/model_iter1
python convert_fabric_to_hf_models.py --config_model llama_model_3_2.json --checkpoint_path outModel/fda/meta-llama/Llama-3.2-3B/iter-330504-ckpt.pth --output_dir llama3_2/modelFDA_llama_3_2/model_iter2

echo "Conversion finished at $(date)"