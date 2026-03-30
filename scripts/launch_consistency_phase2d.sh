#!/usr/bin/env bash
# launch_consistency_phase2d.sh
# Phase 2D — From-scratch vs. transfer at omega=0.1  (H9: transfer-basin hypothesis)
#
# GPU 6: diag_scratch_n6w01  — random init, 2000 epochs
# GPU 7: diag_xfer_n6w01    — resume bf_ctnn_vcycle (omega=1.0 original best), 800 epochs
#
# Usage (run inside consistency_p2d tmux session):
#   bash scripts/launch_consistency_phase2d.sh 2>&1 | tee outputs/consistency_campaign/phase2/launcher_2d.log

set -euo pipefail

REPO=/itf-fi-ml/home/aleksns/Thesis_repo
PHASE2_DIR=outputs/consistency_campaign/phase2
MODULE_CMD="source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
CKPT_BF_CTNN="results/arch_colloc/bf_ctnn_vcycle.pt"

LAST_PID=0
PIDS=()

mkdir -p "$REPO/$PHASE2_DIR"

launch_job() {
    local tag="$1"
    local gpu="$2"
    shift 2
    local run_args=("$@")
    (
        set +euo pipefail
        eval "$MODULE_CMD" 2>/dev/null
        set -e
        export CUDA_MANUAL_DEVICE=$gpu
        cd "$REPO"
        python3 scripts/instrumented_run.py \
            --tag "$tag" \
            --output-dir "$PHASE2_DIR" \
            -- "${run_args[@]}"
    ) > "$REPO/$PHASE2_DIR/${tag}_launcher.log" 2>&1 &
    LAST_PID=$!
}

echo "========================================================"
echo " Phase 2D: From-scratch vs transfer"
echo " $(date)"
echo " Output dir: $REPO/$PHASE2_DIR"
echo "========================================================"

echo ""
echo "=== 2D-A: scratch init, 2000 epochs (GPU 6) ==="
launch_job "diag_scratch_n6w01" 6 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 2000 --n-coll 4096 --oversample 8 --lr 5e-4 --seed 42 \
    --save-best-window 20 \
    --min-ess 0 --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_scratch_n6w01  GPU=6  PID=$LAST_PID  (no resume, 2000 epochs)"

echo ""
echo "=== 2D-B: transfer from bf_ctnn_vcycle (GPU 7) ==="
launch_job "diag_xfer_n6w01" 7 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 800 --n-coll 4096 --oversample 8 --lr 5e-4 --seed 42 \
    --save-best-window 20 \
    --min-ess 0 \
    --resume "$CKPT_BF_CTNN" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_xfer_n6w01     GPU=7  PID=$LAST_PID  (resume bf_ctnn_vcycle, 800 epochs)"

echo ""
echo "Both Phase 2D jobs submitted. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
wait "${PIDS[@]}"
echo ""
echo "Phase 2D complete: $(date)"
