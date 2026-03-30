#!/usr/bin/env bash
# launch_consistency_phase3_v2.sh
# Phase 3 (revised) — Targeted fixes, priority-ranked, 5 GPUs (1-5)
#
# Priority order (from Phase 2 findings):
#   1. 3C — patient Adam + LR warmup   (regression from best = training stability issue)
#   2. 3A — no-gate + oversample 32    (ESS-gate is primary blocker, needs more epochs)
#   3. 3B — FD-colloc + proximal       (best measured error so far; keep in matrix)
#
# GPUs:
#   GPU 1: 3C  omega=0.1   — ultra-patient Adam, 2000 epochs
#   GPU 2: 3C  omega=0.001 — ultra-patient Adam, 2000 epochs
#   GPU 3: 3A  omega=0.1   — no-gate oversample 32, 2000 epochs
#   GPU 4: 3A  omega=0.001 — no-gate oversample 32, 2000 epochs
#   GPU 5: 3B  omega=0.1   — FD-colloc + proximal, 1500 epochs
#
# All jobs resume from Phase 2 diagnostic checkpoints.
# All jobs use --save-best-window 20 to persist best rolling-mean checkpoint.
#
# Usage (run inside consistency_p3 tmux session):
#   bash scripts/launch_consistency_phase3_v2.sh 2>&1 | tee outputs/consistency_campaign/phase3/launcher.log

set -euo pipefail

REPO=/itf-fi-ml/home/aleksns/Thesis_repo
PHASE3_DIR=outputs/consistency_campaign/phase3
MODULE_CMD="source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"

# Phase 2 diagnostic checkpoints to resume from
CKPT_ESS_W01="results/arch_colloc/diag_ess_n6w01.pt"
CKPT_ESS_W001="results/arch_colloc/diag_ess_n6w001.pt"
CKPT_FDCOLLOC_W01="results/arch_colloc/diag_fdcolloc_n6w01.pt"

LAST_PID=0
PIDS=()

mkdir -p "$REPO/$PHASE3_DIR"

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
            --output-dir "$PHASE3_DIR" \
            -- "${run_args[@]}"
    ) > "$REPO/$PHASE3_DIR/${tag}_launcher.log" 2>&1 &
    LAST_PID=$!
}

echo "========================================================"
echo " Phase 3 (v2): Targeted fixes"
echo " $(date)"
echo " Output dir: $REPO/$PHASE3_DIR"
echo " GPUs 1-5  |  Priority: 3C > 3A > 3B"
echo "========================================================"

# ─── 3C: Patient Adam + LR warmup ────────────────────────────────────────────
# Hypothesis: regression from best-E is a training stability problem.
# Fix: lower LR, larger batch, linear warmup, more epochs.
# Resume from no-gate ESS checkpoints (same overlap conditions as Phase 2A).

echo ""
echo "=== 3C: Patient Adam + LR warmup ==="

launch_job "p3c_adam_n6w01" 1 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 2000 --n-coll 8192 --oversample 8 --lr 1e-4 --lr-jas 1e-5 \
    --lr-warmup-epochs 300 --lr-warmup-init-frac 0.05 --lr-min-frac 0.005 \
    --save-best-window 20 \
    --min-ess 0 \
    --resume "$CKPT_ESS_W01" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  p3c_adam_n6w01   GPU=1  PID=$LAST_PID  (omega=0.1, 2000ep, lr=1e-4, warmup=300)"

launch_job "p3c_adam_n6w001" 2 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 2000 --n-coll 8192 --oversample 8 --lr 5e-5 --lr-jas 5e-6 \
    --lr-warmup-epochs 500 --lr-warmup-init-frac 0.05 --lr-min-frac 0.005 \
    --save-best-window 20 \
    --min-ess 0 \
    --resume "$CKPT_ESS_W001" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  p3c_adam_n6w001  GPU=2  PID=$LAST_PID  (omega=0.001, 2000ep, lr=5e-5, warmup=500)"

# ─── 3A: No-gate + oversample 32, more epochs ────────────────────────────────
# Hypothesis: ESS gating was the primary blocker. Fix: remove gate entirely,
# increase oversample to maintain diverse candidates, run much longer.
# 2000 epochs vs 500 in Phase 2A.

echo ""
echo "=== 3A: No-gate + oversample 32 ==="

launch_job "p3a_nogate_n6w01" 3 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 2000 --n-coll 4096 --oversample 32 --lr 5e-4 \
    --save-best-window 20 \
    --min-ess 0 --ess-floor-ratio 0.0 \
    --resume "$CKPT_ESS_W01" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  p3a_nogate_n6w01   GPU=3  PID=$LAST_PID  (omega=0.1, 2000ep, oversample=32)"

launch_job "p3a_nogate_n6w001" 4 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 2000 --n-coll 4096 --oversample 32 --lr 5e-4 \
    --save-best-window 20 \
    --min-ess 0 --ess-floor-ratio 0.0 \
    --resume "$CKPT_ESS_W001" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  p3a_nogate_n6w001  GPU=4  PID=$LAST_PID  (omega=0.001, 2000ep, oversample=32)"

# ─── 3B: FD-colloc + proximal ────────────────────────────────────────────────
# Phase 2 best measured error was FD-colloc (+0.236%).
# This run : more epochs (1500), tighter proximal, resume from FD-colloc checkpoint.

echo ""
echo "=== 3B: FD-colloc + proximal ==="

launch_job "p3b_fdcolloc_n6w01" 5 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 1500 --n-coll 4096 --oversample 8 --lr 2e-4 \
    --loss-type fd-colloc --fd-h 0.005 --prox-mu 0.1 --fd-huber-delta 0.5 \
    --save-best-window 20 \
    --min-ess 0 \
    --resume "$CKPT_FDCOLLOC_W01" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  p3b_fdcolloc_n6w01  GPU=5  PID=$LAST_PID  (omega=0.1, 1500ep, FD-colloc)"

echo ""
echo "All 5 Phase 3 jobs submitted. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
wait "${PIDS[@]}"
echo ""
echo "Phase 3 complete: $(date)"
