#!/bin/bash
# Inference for continual-base instructed models (LoRA adapters).

#SBATCH --job-name=inf_cont
#SBATCH --output=/path/%x-%A_%a.out
#SBATCH --error=/path/%x-%A_%a.err
#SBATCH --partition=XX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
if [ -f "${BASE_DIR}/Instruction/infer_entities.py" ]; then
  :
elif [ -f "${BASE_DIR}/Generative/Instruction/infer_entities.py" ]; then
  BASE_DIR="${BASE_DIR}/Generative"
else
  BASE_DIR="${SCRIPT_DIR}"
fi
ROOT_DIR="$(cd "${BASE_DIR}/.." && pwd)"

SEED_LIST=${SEED_LIST:-"42 1234 2025"}
IFS=' ' read -r -a SEEDS <<< "${SEED_LIST}"
SEED_OVERRIDE=${SEED_OVERRIDE:-${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}}
MAX_SAMPLES=${MAX_SAMPLES:-0}

MODEL_ROOT=${MODEL_ROOT:-"${BASE_DIR}/modelsSeeded/continual"}
LOG_DIR="${BASE_DIR}/modelsSeeded/logs_continual"
mkdir -p "${MODEL_ROOT}/outputPrompts" "${LOG_DIR}"

set +u
source ~/anaconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate /conda_path
set -u

export HF_HOME="/path/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}"
export HF_TOKEN="${HF_TOKEN:-XXXX}"
export TOKENIZERS_PARALLELISM=false

MODELS_DEFAULT=(
  "cont_llama2_cima_e1|${ROOT_DIR}/Continual/llama2/modelCima_llama2/iter1|llama2"
  "cont_llama2_cima_e2|${ROOT_DIR}/Continual/llama2/modelCima_llama2/iter2|llama2"
  "cont_llama2_fda_e1|${ROOT_DIR}/Continual/llama2/modelFDA_llama2/iter1|llama2"
  "cont_llama2_fda_e2|${ROOT_DIR}/Continual/llama2/modelFDA_llama2/iter2|llama2"
  "cont_llama2_cimafda_e1|${ROOT_DIR}/Continual/llama2/modelCimaFDA_llama2/iter1|llama2"
  "cont_llama2_cimafda_e2|${ROOT_DIR}/Continual/llama2/modelCimaFDA_llama2/iter2|llama2"
  "cont_llama3_cima_e1|${ROOT_DIR}/Continual/llama3/modelCima_llama3/model5_hf|llama3"
  "cont_llama3_cima_e2|${ROOT_DIR}/Continual/llama3/modelCima_llama3/model6_hf|llama3"
  "cont_llama3_fda_e1|${ROOT_DIR}/Continual/llama3/modelFDA_llama3/modelFDA_iter1|llama3"
  "cont_llama3_fda_e2|${ROOT_DIR}/Continual/llama3/modelFDA_llama3/modelFDA_iter2|llama3"
  "cont_llama3_cimafda_e1|${ROOT_DIR}/Continual/llama3/modelCimaFDA_llama_3/iter1|llama3"
  "cont_llama3_cimafda_e2|${ROOT_DIR}/Continual/llama3/modelCimaFDA_llama_3/iter2|llama3"
  "cont_llama3_1_cima_e1|${ROOT_DIR}/Continual/llama3_1/modelCima_llama_3_1/model_iter1|llama3_1"
  "cont_llama3_1_cima_e2|${ROOT_DIR}/Continual/llama3_1/modelCima_llama_3_1/model_iter2|llama3_1"
  "cont_llama3_1_fda_e1|${ROOT_DIR}/Continual/llama3_1/modelFDA_llama_3_1/iter1|llama3_1"
  "cont_llama3_1_fda_e2|${ROOT_DIR}/Continual/llama3_1/modelFDA_llama_3_1/iter2|llama3_1"
  "cont_llama3_1_cimafda_e1|${ROOT_DIR}/Continual/llama3_1/modelCimaFDA_llama_3_1/iter1|llama3_1"
  "cont_llama3_1_cimafda_e2|${ROOT_DIR}/Continual/llama3_1/modelCimaFDA_llama_3_1/iter2|llama3_1"
  "cont_llama3_2_cima_e1|${ROOT_DIR}/Continual/llama3_2/modelCima_llama_3_2/model_iter1|llama3_2"
  "cont_llama3_2_cima_e2|${ROOT_DIR}/Continual/llama3_2/modelCima_llama_3_2/model_iter2|llama3_2"
  "cont_llama3_2_fda_e1|${ROOT_DIR}/Continual/llama3_2/modelFDA_llama_3_2/model_iter1|llama3_2"
  "cont_llama3_2_fda_e2|${ROOT_DIR}/Continual/llama3_2/modelFDA_llama_3_2/model_iter2|llama3_2"
  "cont_llama3_2_cimafda_e1|${ROOT_DIR}/Continual/llama3_2/modelCimaFDA_llama_3_2/model_iter1|llama3_2"
  "cont_llama3_2_cimafda_e2|${ROOT_DIR}/Continual/llama3_2/modelCimaFDA_llama_3_2/model_iter2|llama3_2"
)

if [ -n "${MODEL_INDEX:-}" ]; then
  MODELS=( "${MODELS_DEFAULT[$MODEL_INDEX]}" )
else
  MODELS=( "${MODELS_DEFAULT[@]}" )
fi

echo "Host: $(hostname)  Start: $(date)"
echo "MODEL_INDEX=${MODEL_INDEX:-all}"
echo "SEED_OVERRIDE=${SEED_OVERRIDE}"
echo "Total models selected: ${#MODELS[@]}"

for entry in "${MODELS[@]}"; do
  IFS="|" read -r label model_path family <<< "${entry}"

  LORA_DIR="${MODEL_ROOT}/${label}_s${SEED_OVERRIDE}"
  OUT_JSON="${MODEL_ROOT}/outputPrompts/final_predictions_${label}_s${SEED_OVERRIDE}.jsonl"
  LOG_FILE="${LOG_DIR}/inference_${label}_s${SEED_OVERRIDE}.txt"

  MAX_NEW_TOKENS=860
  TEMPERATURE=0.17315939349886061
  REPETITION_PENALTY=1.0069760044869343
  NUM_BEAMS=3
  TOP_P=0.7005801448613161
  DO_SAMPLE_FLAG=
  NO_REPEAT_NGRAM_SIZE=6

  case "${family}" in
    llama2)
      MAX_NEW_TOKENS=1326
      TEMPERATURE=0.3327288653275695
      REPETITION_PENALTY=1.5541629428392234
      NUM_BEAMS=4
      TOP_P=0.8021746621540992
      ;;
    llama3_2)
      MAX_NEW_TOKENS=1124
      TEMPERATURE=0.24899144287235947
      REPETITION_PENALTY=1.327448913609007
      NUM_BEAMS=3
      TOP_P=0.8253425573316452
      ;;
    llama3_1)
      MAX_NEW_TOKENS=1071
      TEMPERATURE=0.0006183203210246985
      REPETITION_PENALTY=1.957728614827438
      NUM_BEAMS=3
      TOP_P=0.70614848048016
      DO_SAMPLE_FLAG=--do_sample
      ;;
    llama3)
      MAX_NEW_TOKENS=1266
      TEMPERATURE=0.06011203590107267
      REPETITION_PENALTY=1.398796712930793
      NUM_BEAMS=8
      TOP_P=0.9130844873923798
      DO_SAMPLE_FLAG=--do_sample
      ;;
  esac

  {
    echo ""
    echo "=== INFER START $(date) ==="
    echo "label=${label}"
    echo "family=${family}"
    echo "base_model=${model_path}"
    echo "lora_dir=${LORA_DIR}"
    echo "output_json=${OUT_JSON}"
  } | tee "${LOG_FILE}"

  if python "${BASE_DIR}/Instruction/infer_entities.py" \
    --base_model_path "${model_path}" \
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
    >> "${LOG_FILE}" 2>&1; then
    echo "=== INFER DONE $(date) ===" | tee -a "${LOG_FILE}"
  else
    echo "=== INFER FAILED $(date) ===" | tee -a "${LOG_FILE}"
    exit 1
  fi

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
done

echo "End: $(date)"