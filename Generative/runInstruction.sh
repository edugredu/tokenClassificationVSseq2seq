#!/bin/bash
#SBATCH --job-name=instruction       # Job name
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

# 6. Run the training script with fabric
echo "Starting training..."


# Remove log_training.txt if exists
if [ -f log_training.txt ]; then
    rm log_optimal_training.txt
fi

# Used for solving out of memory in GPU 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# Clear cache before training
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# 7. Run the instruction tuning script with torchrun
torchrun --nproc_per_node=2 codeInstruction.py \
       --model_path models/llama-3.2-3b-instruct \
       --train_path Instruction/instructed_prompts_train_p0.jsonl \
       --valid_path Instruction/instructed_prompts_valid_p0.jsonl \
       --template_path basePromptInstruction.txt \
       --output_dir outputs/llama-3.2-3b-instruct-lora-optimal \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 8 \
       --max_seq_len 2285 \
       --learning_rate 8e-5 \
       --num_train_epochs 6 \
       --use_lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.15 \
       --bf16 \
       --evaluation_strategy steps \
       --eval_steps 21 \
       --max_eval_samples 50 \
       --eval_max_new_tokens 1793 \
       --eval_temperature 0.1 \
       --save_strategy steps \
       --save_steps 21 \
       --load_best_model_at_end \
       --metric_for_best_model entity_f1_exact \
       --greater_is_better \
       --warmup_ratio 0.2 \
       > log_optimal_training.txt 2>&1

echo "Training finished at $(date)"
