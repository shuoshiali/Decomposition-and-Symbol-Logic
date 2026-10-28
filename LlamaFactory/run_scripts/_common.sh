#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="${EXP_ROOT:-./tier1_runs}"
METRICS_ROOT="${EXP_ROOT}/metrics"; mkdir -p "${METRICS_ROOT}"

TRAIN_CMD="${TRAIN_CMD:-CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src python -m llamafactory.cli train config/Qwen3_lora_direrl.yaml}"
SEEDS="${SEEDS:-12}"            # SEEDS="12 22 32 42"

run_one() {
    local tag="$1"; shift
    local n_seeds
    n_seeds=$(echo ${SEEDS} | wc -w)
    for s in ${SEEDS}; do
        local rtag="${tag}"
        if [ "${n_seeds}" -gt 1 ]; then rtag="${tag}_s${s}"; fi
        echo "===== [$(date '+%H:%M:%S')] RUN ${rtag} : DIRE_SEED=${s} $* ====="
        # shellcheck disable=SC2086
        env DIRE_RUN_TAG="${rtag}" \
            DIRE_SEED="${s}" \
            DIRE_METRICS_PATH="${METRICS_ROOT}/${rtag}.jsonl" \
            "$@" ${TRAIN_CMD} 2>&1 | tee "${EXP_ROOT}/${rtag}.log"
        local rc=${PIPESTATUS[0]}
        echo "----- train exit=${rc} ${rtag} -----"
        if [ ${rc} -eq 0 ] && [ -n "${EVAL_DATASETS:-}" ]; then
            for ds in ${EVAL_DATASETS}; do
                echo "----- eval ${rtag} / ${ds} -----"
                bash "${SCRIPT_DIR}/eval_pipeline.sh" "${rtag}" "${ds}" \
                    >> "${EXP_ROOT}/${rtag}.log" 2>&1
                echo "----- eval exit=$? ${rtag}/${ds} -----"
            done
        fi
    done
}
