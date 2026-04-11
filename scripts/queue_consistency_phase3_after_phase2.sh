#!/usr/bin/env bash
# queue_consistency_phase3_after_phase2.sh
# Wait for all Phase 2 jobs + heavy eval, then launch Phase 3.

set -euo pipefail

REPO=/itf-fi-ml/home/aleksns/Thesis_repo
PHASE2_DIR=outputs/consistency_campaign/phase2
PHASE3_DIR=outputs/consistency_campaign/phase3
MODULE_CMD="source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"

PHASE2_TAGS=(
    diag_ess_n6w01
    diag_ess_n6w001
    diag_snr_n6w01
    diag_snr_n6w001
    diag_fdcolloc_n6w01
    diag_reinf_n6w01
    diag_scratch_n6w01
    diag_xfer_n6w01
)

phase2_complete() {
    for tag in "${PHASE2_TAGS[@]}"; do
        if [ ! -f "$REPO/$PHASE2_DIR/${tag}_summary.json" ]; then
            return 1
        fi
    done
    return 0
}

ensure_phase2_eval() {
    local eval_json="$REPO/$PHASE2_DIR/eval_summary.json"
    if [ -f "$eval_json" ]; then
        echo "[queue] Phase 2 eval already present: $eval_json"
        return 0
    fi

    echo "[queue] Running heavy eval for completed Phase 2 checkpoints..."
    eval "$MODULE_CMD" 2>/dev/null
    export CUDA_MANUAL_DEVICE=0
    cd "$REPO"

    local args=()
    for tag in "${PHASE2_TAGS[@]}"; do
        args+=(--checkpoint "results/arch_colloc/${tag}.pt")
    done

    python3 scripts/eval_checkpoint_matrix.py \
        "${args[@]}" \
        --n-eval 30000 \
        --output-json "$PHASE2_DIR/eval_summary.json"
}

mkdir -p "$REPO/$PHASE3_DIR"
echo "========================================================"
echo " Queueing Phase 3 after Phase 2 completion"
echo " $(date)"
echo "========================================================"

until phase2_complete; do
    echo "[queue] Phase 2 incomplete; waiting for all 8 summary files..."
    sleep 120
done

echo "[queue] Phase 2 summaries present."
ensure_phase2_eval

echo "[queue] Launching Phase 3..."
cd "$REPO"
bash scripts/launch_consistency_phase3.sh 2>&1 | tee "$PHASE3_DIR/launcher.log"