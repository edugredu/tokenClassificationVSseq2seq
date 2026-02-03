#!/bin/bash
#SBATCH --job-name=instruction
#SBATCH --output=/path/%x-%A_%a.out
#SBATCH --error=/path/%x-%A_%a.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=140G
#SBATCH --time=30:00:00
#SBATCH --array=0-2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"

echo "Iniciando trabajo en $(hostname) a las $(date)"

# 1. Cargar conda para activar el entorno
set +u
source ~/anaconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate /path_to_conda
set -u

# Print nvidia-smi
nvidia-smi

export HF_HOME="/path"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}"

export CUDA_HOME="$CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${PATH}"
echo "$CUDA_HOME"

# 2. Iniciar sesion en huggingface
huggingface-cli login --token XXXX

# 3. Iniciar sesión en wandb
python -m wandb login XXXX

# Used for solving out of memory in GPU 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# Resolve seed from array
SEED_LIST=${SEED_LIST:-"42 1234 2024"}
IFS=' ' read -r -a SEEDS <<< "${SEED_LIST}"
SEED_OVERRIDE=${SEED_OVERRIDE:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}
if [ -z "${SEED_OVERRIDE}" ]; then
  echo "Seed not resolved (SEED_LIST=${SEED_LIST}, SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})"
  exit 1
fi

MODEL_ROOT=${MODEL_ROOT:-"${BASE_DIR}/modelsSeeded"}
OUT_DIR="${MODEL_ROOT}/llama-3.1-8B-Instruct-lora-optimal_s${SEED_OVERRIDE}"
LOG_DIR="${MODEL_ROOT}/logs"
LOG_FILE="${LOG_DIR}/log_optimal_training_llama-3.1-8b-instruct_s${SEED_OVERRIDE}.txt"
mkdir -p "${MODEL_ROOT}" "${LOG_DIR}"

# Clear cache before training
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

MASTER_PORT=${MASTER_PORT:-$((29500 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID) % 1000))}

torchrun --nproc_per_node=3 --master_port="${MASTER_PORT}" "${BASE_DIR}/Instruction/codeInstruction.py" \
       --model_path meta-llama/Llama-3.1-8B-Instruct \
       --train_path "${BASE_DIR}/Instruction/instructed_prompts_train_p0.jsonl" \
       --valid_path "${BASE_DIR}/Instruction/instructed_prompts_valid_p0.jsonl" \
       --template_path "${BASE_DIR}/Instruction/basePromptInstruction.txt" \
       --output_dir "${OUT_DIR}" \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 8 \
       --max_seq_len 2285 \
       --learning_rate 8e-5 \
       --num_train_epochs 6 \
       --save_total_limit 6 \
       --use_lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.15 \
       --bf16 \
       --evaluation_strategy steps \
       --eval_steps 14 \
       --max_eval_samples 50 \
       --eval_max_new_tokens 1793 \
       --eval_temperature 0.1 \
       --save_strategy steps \
       --save_steps 14 \
       --load_best_model_at_end \
       --metric_for_best_model entity_f1_exact \
       --greater_is_better \
       --warmup_ratio 0.2 \
       --gradient_checkpointing \
       --seed "${SEED_OVERRIDE}" \
       > "${LOG_FILE}" 2>&1

echo "Trabajo completado a las $(date)"