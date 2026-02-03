#!/usr/bin/env bash

set -euo pipefail

# Set HF_TOKEN in your env or pass via --hf_token argument override.
HF_TOKEN_ENV="${HF_TOKEN:-}"
SEED_OVERRIDE="${SEED_OVERRIDE:-1234}"

# Optional: wait for GPU to be mostly free before starting each run.
wait_for_gpu() {
    local threshold_mb="${1:-1000}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    while :; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1{print $1}')
        if [[ -n "$used" && "$used" -lt "$threshold_mb" ]]; then
            return 0
        fi
        echo "Waiting for GPU memory < ${threshold_mb}MB (currently ${used:-?}MB)..."
        sleep 30
    done
}

# List of all discovered continued models (epoch1/epoch2) in Continual/.
# Format: label|model_id
MODELS_DEFAULT=(
    # llama3_2
    "llama3_2_cima_e1|Continual/llama3_2/modelCima_llama_3_2/model_iter1"
    "llama3_2_cima_e2|Continual/llama3_2/modelCima_llama_3_2/model_iter2"
    "llama3_2_fda_e1|Continual/llama3_2/modelFDA_llama_3_2/model_iter1"
    "llama3_2_fda_e2|Continual/llama3_2/modelFDA_llama_3_2/model_iter2"
    "llama3_2_cimafda_e1|Continual/llama3_2/modelCimaFDA_llama_3_2/model_iter1"
    "llama3_2_cimafda_e2|Continual/llama3_2/modelCimaFDA_llama_3_2/model_iter2"

    # llama3_1
    "llama3_1_cima_e1|Continual/llama3_1/modelCima_llama_3_1/model_iter1"
    "llama3_1_cima_e2|Continual/llama3_1/modelCima_llama_3_1/model_iter2"
    "llama3_1_fda_e1|Continual/llama3_1/modelFDA_llama_3_1/iter1"
    "llama3_1_fda_e2|Continual/llama3_1/modelFDA_llama_3_1/iter2"
    "llama3_1_cimafda_e1|Continual/llama3_1/modelCimaFDA_llama_3_1/iter1"
    "llama3_1_cimafda_e2|Continual/llama3_1/modelCimaFDA_llama_3_1/iter2"

    # llama3
    "llama3_cima_e1|Continual/llama3/modelCima_llama3/model5_hf"
    "llama3_cima_e2|Continual/llama3/modelCima_llama3/model6_hf"
    "llama3_fda_e1|Continual/llama3/modelFDA_llama3/modelFDA_iter1"
    "llama3_fda_e2|Continual/llama3/modelFDA_llama3/modelFDA_iter2"
    "llama3_cimafda_e1|Continual/llama3/modelCimaFDA_llama_3/iter1"
    "llama3_cimafda_e2|Continual/llama3/modelCimaFDA_llama_3/iter2"

    # llama2
    "llama2_cima_e1|Continual/llama2/modelCima_llama2/iter1"
    "llama2_cima_e2|Continual/llama2/modelCima_llama2/iter2"
    "llama2_fda_e1|Continual/llama2/modelFDA_llama2/iter1"
    "llama2_fda_e2|Continual/llama2/modelFDA_llama2/iter2"
    "llama2_cimafda_e1|Continual/llama2/modelCimaFDA_llama2/iter1"
    "llama2_cimafda_e2|Continual/llama2/modelCimaFDA_llama2/iter2"

    # Base (HF) models
    "base_llama2|meta-llama/Llama-2-7b-hf"
    "base_llama3|meta-llama/Meta-Llama-3-8B"
    "base_llama3_1|meta-llama/Meta-Llama-3.1-8B"
    "base_llama3_2|meta-llama/Llama-3.2-3B-Instruct"
)

# Allow overriding models via env (MODELS_OVERRIDE as space- or comma-separated list)
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
    IFS=', ' read -r -a MODELS <<< "${MODELS_OVERRIDE}"
else
    MODELS=("${MODELS_DEFAULT[@]}")
fi

# Allow selecting a single model by index (MODEL_INDEX or SLURM_ARRAY_TASK_ID)
if [[ -n "${MODEL_INDEX:-}" ]]; then
    MODELS=( "${MODELS[MODEL_INDEX]}" )
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    MODELS=( "${MODELS[SLURM_ARRAY_TASK_ID]}" )
fi


main() {
    local python_bin="${PYTHON_BIN:-python3}"
    local sleep_between="${SLEEP_BETWEEN:-0}" # seconds between launches
    export PYTHONUNBUFFERED=1  # stream logs immediately
    mkdir -p outputTXTs

    for idx in "${!MODELS[@]}"; do
        IFS="|" read -r label model_id <<< "${MODELS[$idx]}"
        local run_label="${label}_s${SEED_OVERRIDE}"
        local log_file="outputTXTs/${run_label}.log"

        if [[ -z "$HF_TOKEN_ENV" ]]; then
            echo "HF_TOKEN is not set and --hf_token not provided; set HF_TOKEN env to proceed." >&2
            exit 1
        fi

        # Skip wait_for_gpu when running under Slurm (GPU already allocated) unless explicitly requested
        if [[ -z "${SLURM_JOB_ID:-}" || "${FORCE_GPU_WAIT:-0}" -eq 1 ]]; then
            wait_for_gpu "${GPU_THRESHOLD_MB:-1000}"
        fi

        echo ">>> Starting ${label} (${model_id})"
        "${python_bin}" llama_pharmacoNER.py \
            --model_id "${model_id}" \
            --hf_token "${HF_TOKEN_ENV}" \
            --run_id "${run_label}" \
            --output_dir "runs/${run_label}" \
            --max_length 256 \
            --epochs 2 \
            --learning_rate 1e-4 \
            --lora_r 12 \
            --seed "${SEED_OVERRIDE}" \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 2 \
            > "${log_file}" 2>&1

        echo ">>> Running inference for ${label}"
        "${python_bin}" inference.py \
            --adapter_dir "runs/${run_label}" \
            --batch_size 8 \
            --max_length 256 \
            --save_gold >> "${log_file}" 2>&1

        echo ">>> Evaluating ${label}"
        "${python_bin}" PharmacoNER/Evaluacion/evaluate.py \
            ner "runs/${run_label}/brat_ann_gold" "runs/${run_label}/brat_ann_pred" >> "${log_file}" 2>&1

        echo ">>> Finished ${label}"
        if [[ "$sleep_between" -gt 0 ]]; then
            sleep "$sleep_between"
        fi
    done
}

main "$@"