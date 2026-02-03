#!/bin/bash
#SBATCH --job-name=gen-inf-seeded
#SBATCH --output=/path/%x-%j.out
#SBATCH --error=/path/%x-%j.err
#SBATCH --partition=XX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"

SEED_LIST=${SEED_LIST:-"42 1234 2024"}
IFS=' ' read -r -a SEEDS <<< "${SEED_LIST}"
SEED_OVERRIDE=${SEED_OVERRIDE:-${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}}
LABEL_INDEX=${LABEL_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-0}

DEFAULT_MODEL_ROOT="${BASE_DIR}/modelsSeeded"
MODEL_ROOT=${MODEL_ROOT:-${DEFAULT_MODEL_ROOT}}
OUT_BASE=${OUT_BASE:-${MODEL_ROOT}}

BASE_MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct|llama-3.2-3b-instruct-lora-optimal"
  "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8B-Instruct-lora-optimal"
  "meta-llama/Meta-Llama-3-8B-Instruct|Meta-Llama-3-8B-Instruct-lora-optimal"
  "meta-llama/Llama-2-7b-chat-hf|llama-2-7b-chat-hf-lora-optimal"
)

IFS='|' read -r BASE_MODEL LABEL <<< "${BASE_MODELS[$LABEL_INDEX]}"
LORA_DIR="${MODEL_ROOT}/${LABEL}_s${SEED_OVERRIDE}"

OUT_JSON="${OUT_BASE}/outputPrompts/final_predictions_${LABEL}_s${SEED_OVERRIDE}.jsonl"
LOG_DIR="${OUT_BASE}/logs"
LOG_FILE="${LOG_DIR}/inference_${LABEL}_s${SEED_OVERRIDE}.txt"
mkdir -p "${OUT_BASE}/outputPrompts" "${LOG_DIR}"

# Decoding parameters from old baselines
MAX_NEW_TOKENS=860
TEMPERATURE=0.17315939349886061
REPETITION_PENALTY=1.0069760044869343
NUM_BEAMS=3
TOP_P=0.7005801448613161
DO_SAMPLE_FLAG=
NO_REPEAT_NGRAM_SIZE=6

case "${LABEL}" in
  llama-3.2-3b-instruct-lora-optimal)
    MAX_NEW_TOKENS=1124
    TEMPERATURE=0.24899144287235947
    REPETITION_PENALTY=1.327448913609007
    NUM_BEAMS=3
    TOP_P=0.8253425573316452
    ;;
  llama-3.1-8B-Instruct-lora-optimal)
    MAX_NEW_TOKENS=1071
    TEMPERATURE=0.0006183203210246985
    REPETITION_PENALTY=1.957728614827438
    NUM_BEAMS=3
    TOP_P=0.70614848048016
    DO_SAMPLE_FLAG=--do_sample
    ;;
  Meta-Llama-3-8B-Instruct-lora-optimal)
    MAX_NEW_TOKENS=1266
    TEMPERATURE=0.06011203590107267
    REPETITION_PENALTY=1.398796712930793
    NUM_BEAMS=8
    TOP_P=0.9130844873923798
    DO_SAMPLE_FLAG=--do_sample
    ;;
  llama-2-7b-chat-hf-lora-optimal)
    MAX_NEW_TOKENS=1326
    TEMPERATURE=0.3327288653275695
    REPETITION_PENALTY=1.5541629428392234
    NUM_BEAMS=4
    TOP_P=0.8021746621540992
    ;;
esac

echo "Host: $(hostname)  Start: $(date)"
echo "Seed: ${SEED_OVERRIDE}"
echo "Base model: ${BASE_MODEL}"
echo "LoRA dir: ${LORA_DIR}"
echo "Output base: ${OUT_BASE}"

set +u
source ~/anaconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate /path_to_conda
set -u

export HF_TOKEN=XXXX
export HF_HOME="/scratch/egrande/hf_cache"
mkdir -p "${HF_HOME}"
export CUDA_HOME="$CONDA_PREFIX"
export TOKENIZERS_PARALLELISM=false

python "${BASE_DIR}/Instruction/infer_entities.py" \
  --base_model_path "${BASE_MODEL}" \
  --lora_dir "${LORA_DIR}" \
  --output_path "${OUT_JSON}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --num_beams "${NUM_BEAMS}" \
  --top_p "${TOP_P}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --no_repeat_ngram_size "${NO_REPEAT_NGRAM_SIZE}" \
  ${DO_SAMPLE_FLAG} \
  --max_samples "${MAX_SAMPLES}" \
  --search_method none \
  > "${LOG_FILE}" 2>&1

BASE_PREFIX="${OUT_JSON%.*}"
GOLD_DIR="${BASE_PREFIX}_brat_gold"
PRED_RAW_DIR="${BASE_PREFIX}_brat_raw"
PRED_FIXED_DIR="${BASE_PREFIX}_brat_fixed"
{
  echo ""
  echo "Running official PharmacoNER evaluation (raw)..."
  python "${BASE_DIR}/../PharmacoNER/Evaluacion/evaluate.py" ner "${GOLD_DIR}" "${PRED_RAW_DIR}"
  echo ""
  echo "Running official PharmacoNER evaluation (fixed)..."
  python "${BASE_DIR}/../PharmacoNER/Evaluacion/evaluate.py" ner "${GOLD_DIR}" "${PRED_FIXED_DIR}"
} >> "${LOG_FILE}" 2>&1

echo "End: $(date)"
