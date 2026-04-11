#!/usr/bin/env bash
# launch_consistency_phase3.sh
# Phase 3 — targeted fix experiments after Phase 2 diagnostics complete.

set -euo pipefail

REPO=/itf-fi-ml/home/aleksns/Thesis_repo
PHASE3_DIR=outputs/consistency_campaign/phase3
MODULE_CMD="source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"

# Phase 3 starts from Phase 2 outputs at the same or closest already-adapted regime.
CKPT_W01_NOGATE="results/arch_colloc/diag_ess_n6w01.pt"
CKPT_W001_NOGATE="results/arch_colloc/diag_ess_n6w001.pt"
CKPT_W01_REINF="results/arch_colloc/diag_reinf_n6w01.pt"
CKPT_W01_FDCOLLOC="results/arch_colloc/diag_fdcolloc_n6w01.pt"

LAST_PID=0
PIDS=()

launch_job() {
    local tag="$1"
    local gpu="$2"
    shift 2
    local run_args=("$@")

    mkdir -p "$REPO/$PHASE3_DIR"

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

wait_for_all_gpus_free() {
    local poll_sec="${1:-60}"
    echo "[wait] Polling for GPUs 0-7 to become idle..."
    while true; do
        local busy=0
        for gpu in 0 1 2 3 4 5 6 7; do
            local mem_used
            local util
            mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
            util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu")
            if [ "$mem_used" -ge 100 ] || [ "$util" -ge 5 ]; then
                busy=1
                echo "[wait] GPU $gpu busy (mem=${mem_used} MiB util=${util}%)"
            fi
        done
        if [ "$busy" -eq 0 ]; then
            echo "[wait] All GPUs idle."
            return 0
        fi
        sleep "$poll_sec"
    done
}

mkdir -p "$REPO/$PHASE3_DIR"
wait_for_all_gpus_free 60

echo "========================================================"
echo " Phase 3: Targeted fix experiments"
echo " $(date)"
echo " Output dir: $REPO/$PHASE3_DIR"
echo "========================================================"

echo ""
echo "=== 3A: No-gate + aggressive oversample (GPUs 0-1) ==="
launch_job "fix_nogate_n6w01_s42" 0 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 1500 --n-coll 4096 --oversample 32 --lr 3e-4 --seed 42 \
    --min-ess 0 --ess-floor-ratio 0.0 --clip-el 3.0 --reward-qtrim 0.01 \
    --resume "$CKPT_W01_NOGATE" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_nogate_n6w01_s42 GPU=0 PID=$LAST_PID"

launch_job "fix_nogate_n6w001_s42" 1 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 2000 --n-coll 4096 --oversample 32 --lr 1e-4 --seed 42 \
    --min-ess 0 --ess-floor-ratio 0.0 --clip-el 3.0 --reward-qtrim 0.01 \
    --resume "$CKPT_W001_NOGATE" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_nogate_n6w001_s42 GPU=1 PID=$LAST_PID"

echo ""
echo "=== 3B: FD-colloc + proximal (GPUs 2-3) ==="
launch_job "fix_fdcol_n6w01" 2 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 1500 --n-coll 4096 --oversample 8 --lr 2e-4 --seed 42 \
    --loss-type fd-colloc --fd-h 0.005 --prox-mu 0.1 --fd-huber-delta 0.5 \
    --min-ess 0 \
    --resume "$CKPT_W01_FDCOLLOC" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_fdcol_n6w01 GPU=2 PID=$LAST_PID"

launch_job "fix_fdcol_n6w001" 3 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 2000 --n-coll 4096 --oversample 8 --lr 1e-4 --seed 42 \
    --loss-type fd-colloc --fd-h 0.002 --prox-mu 0.1 --fd-huber-delta 0.2 \
    --min-ess 0 \
    --resume "$CKPT_W001_NOGATE" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_fdcol_n6w001 GPU=3 PID=$LAST_PID"

echo ""
echo "=== 3C: Ultra-patient Adam + warmup (GPUs 4-5) ==="
launch_job "fix_patient_n6w01" 4 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 3000 --n-coll 8192 --oversample 16 --lr 1e-4 --seed 42 \
    --lr-min-frac 0.1 --lr-warmup-epochs 300 \
    --min-ess 0 \
    --resume "$CKPT_W01_REINF" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_patient_n6w01 GPU=4 PID=$LAST_PID"

launch_job "fix_patient_n6w001" 5 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 5000 --n-coll 8192 --oversample 24 --lr 5e-5 --seed 42 \
    --lr-min-frac 0.1 --lr-warmup-epochs 500 \
    --min-ess 0 \
    --resume "$CKPT_W001_NOGATE" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_patient_n6w001 GPU=5 PID=$LAST_PID"

echo ""
echo "=== 3D: Reward-normalized REINFORCE (GPUs 6-7) ==="
launch_job "fix_normrew_n6w01" 6 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 1500 --n-coll 4096 --oversample 8 --lr 5e-4 --seed 42 \
    --reward-normalize --min-ess 0 \
    --resume "$CKPT_W01_REINF" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_normrew_n6w01 GPU=6 PID=$LAST_PID"

launch_job "fix_normrew_n6w001" 7 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 2000 --n-coll 4096 --oversample 8 --lr 1e-4 --seed 42 \
    --reward-normalize --min-ess 0 \
    --resume "$CKPT_W001_NOGATE" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  fix_normrew_n6w001 GPU=7 PID=$LAST_PID"

echo ""
echo "All 8 Phase 3 jobs submitted. PID list: ${PIDS[*]}"
echo "Waiting for all to finish..."

RC=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "  PID $pid finished OK"
    else
        echo "  PID $pid exited non-zero"
        RC=1
    fi
done

echo ""
echo "========================================================"
echo " Phase 3 finished at $(date) with rc=$RC"
echo " Summaries: $REPO/$PHASE3_DIR/*_summary.json"
echo "========================================================"
exit $RC