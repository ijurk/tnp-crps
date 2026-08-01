#!/usr/bin/env bash

set -euo pipefail

cd /scratch2/ij292/tnp-crps
source setup_simpson.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_DIR="experiments/configs/models"
GENERATOR_CONFIG="experiments/configs/generators/tabular_mlp_nc128.yml"
GATE_CONFIG="experiments/configs/tabular_mlp_nc128_gate.yml"
LOG_DIR="logs/tabular_mlp_nc128_gate"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Starting fixed-Nc=128 tabular gate"
echo "Git commit: $(git rev-parse HEAD)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

run_model() {
    local label="$1"
    local model_config="$2"
    local run_id="$3"
    local log_file="${LOG_DIR}/${label}.log"

    export TNP_RUN_ID="${run_id}"

    echo
    echo "============================================================"
    echo "Starting ${label}"
    echo "Run ID: ${TNP_RUN_ID}"
    echo "Time: $(date --iso-8601=seconds)"
    echo "Log: ${log_file}"
    echo "============================================================"

    python -u experiments/lightning_train_crps.py \
        --config \
        "${model_config}" \
        "${GENERATOR_CONFIG}" \
        "${GATE_CONFIG}" \
        2>&1 | tee "${log_file}"

    echo
    echo "Completed ${label}: $(date --iso-8601=seconds)"
}

run_model \
    "gaussian-seed1-v1" \
    "${MODEL_DIR}/tnp_tabular_gaussian.yml" \
    "tabular-mlp-nc128-gaussian-seed1-v1"

run_model \
    "dropout-m4-p010-a1-seed1-v1" \
    "${MODEL_DIR}/tnp_tabular_dropout.yml" \
    "tabular-mlp-nc128-dropout-m4-p010-a1-seed1-v1"

run_model \
    "stochln-m4-z32-a1-seed1-v1" \
    "${MODEL_DIR}/tnp_tabular_stochln.yml" \
    "tabular-mlp-nc128-stochln-m4-z32-a1-seed1-v1"

echo
echo "============================================================"
echo "All three fixed-Nc=128 runs completed"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
