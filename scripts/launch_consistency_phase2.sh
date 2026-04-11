#!/usr/bin/env bash
# launch_consistency_phase2.sh
# Phase 2 — Characterise failure modes at low omega
#
# Experiments:
#   2A (GPUs 0-1): ESS/overlap no-gate at omega 0.1 and 0.001
#   2B (GPUs 2-3): REINFORCE SNR at omega 0.1 and 0.001
#   2C (GPUs 4-5): FD-colloc vs REINFORCE at omega 0.1
#   2D (GPUs 6-7): From-scratch vs transfer at omega 0.1
#                  — waits for GPUs 6-7 to become free (may be delayed)
#
# Usage (run inside consistency_p2 tmux session):
#   bash scripts/launch_consistency_phase2.sh 2>&1 | tee outputs/consistency_campaign/phase2/launcher.log

set -euo pipefail

REPO=/itf-fi-ml/home/aleksns/Thesis_repo
PHASE2_DIR=outputs/consistency_campaign/phase2
MODULE_CMD="source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"

# Checkpoints from Phase 1 output + originals
CKPT_N6W1_S42="results/arch_colloc/repro_n6w1_s42.pt"    # N=6 omega=1.0
CKPT_N6W05_S42="results/arch_colloc/repro_n6w05_s42.pt"  # N=6 omega=0.5
CKPT_BF_CTNN="results/arch_colloc/bf_ctnn_vcycle.pt"     # N=6 omega=1.0 original best

LAST_PID=0
PIDS=()

# ───────────────────────────────────────────────────────────────────────────────
# launch_job TAG GPU [run_weak_form args...]
#   Runs instrumented_run.py in a background subshell, captures PID in LAST_PID.
#   Output: $PHASE2_DIR/${TAG}_launcher.log (stderr from the wrapper itself)
#   Instrumented artifacts go into $PHASE2_DIR by instrumented_run.py.
# ───────────────────────────────────────────────────────────────────────────────
launch_job() {
    local tag="$1"
    local gpu="$2"
    shift 2
    local run_args=("$@")

    mkdir -p "$REPO/$PHASE2_DIR"

    (
        set +euo pipefail  # module load returns non-zero; must not kill subshell
        eval "$MODULE_CMD" 2>/dev/null
        set -e              # re-enable for the python call itself
        export CUDA_MANUAL_DEVICE=$gpu
        cd "$REPO"
        python3 scripts/instrumented_run.py \
            --tag "$tag" \
            --output-dir "$PHASE2_DIR" \
            -- "${run_args[@]}"
    ) > "$REPO/$PHASE2_DIR/${tag}_launcher.log" 2>&1 &

    LAST_PID=$!
}

# ───────────────────────────────────────────────────────────────────────────────
# wait_for_gpu_free GPU [POLL_SECONDS]
#   Polls nvidia-smi until the GPU has <100 MiB used AND <5% utilisation.
# ───────────────────────────────────────────────────────────────────────────────
wait_for_gpu_free() {
    local gpu="$1"
    local poll_sec="${2:-120}"
    echo "[wait] GPU $gpu: polling every ${poll_sec}s for idle state..."
    while true; do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu")
        if [ "$mem_used" -lt 100 ] && [ "$util" -lt 5 ]; then
            echo "[wait] GPU $gpu is free (mem=${mem_used} MiB util=${util}%)"
            return 0
        fi
        echo "[wait] GPU $gpu busy (mem=${mem_used} MiB util=${util}%), retrying in ${poll_sec}s..."
        sleep "$poll_sec"
    done
}

mkdir -p "$REPO/$PHASE2_DIR"
echo "========================================================"
echo " Phase 2: Low-omega failure characterisation"
echo " $(date)"
echo " Output dir: $REPO/$PHASE2_DIR"
echo "========================================================"


# ─── 2A: ESS/overlap no-gate ─────────────────────────────────────────────────
# Theory: H4 — ESS deadlock prevents training at low omega.
# Method: min_ess=0 (no gating), oversample=16, observe ESS and E trajectory.

echo ""
echo "=== 2A: ESS characterisation (GPUs 0-1) ==="

launch_job "diag_ess_n6w01" 0 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 500 --n-coll 4096 --oversample 16 --lr 5e-4 --seed 42 \
    --min-ess 0 --ess-floor-ratio 0.0 \
    --resume "$CKPT_N6W1_S42" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_ess_n6w01   GPU=0  PID=$LAST_PID  (N=6 omega=0.1 no-gate oversample=16)"

launch_job "diag_ess_n6w001" 1 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 500 --n-coll 4096 --oversample 16 --lr 5e-4 --seed 42 \
    --min-ess 0 --ess-floor-ratio 0.0 \
    --resume "$CKPT_N6W05_S42" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_ess_n6w001  GPU=1  PID=$LAST_PID  (N=6 omega=0.001 no-gate oversample=16)"


# ─── 2B: REINFORCE SNR characterisation ──────────────────────────────────────
# Theory: H5 — REINFORCE gradient SNR collapses at low omega (CV >> 1).
# Method: standard training, log var_EL and E per epoch to observe CV = sqrt(var_EL)/|E|.

echo ""
echo "=== 2B: REINFORCE SNR (GPUs 2-3) ==="

launch_job "diag_snr_n6w01" 2 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 500 --n-coll 4096 --oversample 8 --lr 5e-4 --seed 11 \
    --min-ess 0 \
    --resume "$CKPT_N6W1_S42" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_snr_n6w01   GPU=2  PID=$LAST_PID  (N=6 omega=0.1  lr=5e-4)"

launch_job "diag_snr_n6w001" 3 \
    --mode bf --n-elec 6 --omega 0.001 \
    --epochs 500 --n-coll 4096 --oversample 8 --lr 1e-4 --seed 11 \
    --min-ess 0 \
    --resume "$CKPT_N6W05_S42" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_snr_n6w001  GPU=3  PID=$LAST_PID  (N=6 omega=0.001 lr=1e-4)"


# ─── 2C: FD-colloc vs REINFORCE ──────────────────────────────────────────────
# Theory: H5 alt — FD-colloc loss may be more stable at low omega because it
# does not use E_L as REINFORCE reward (avoids reward-noise problem).
# Both jobs resume from the same Phase 1 N=6 omega=0.5 checkpoint.

echo ""
echo "=== 2C: FD-colloc vs REINFORCE (GPUs 4-5) ==="

launch_job "diag_fdcolloc_n6w01" 4 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 500 --n-coll 4096 --oversample 8 --lr 2e-4 --seed 42 \
    --loss-type fd-colloc --fd-h 0.005 --prox-mu 0.1 --fd-huber-delta 0.5 \
    --min-ess 0 \
    --resume "$CKPT_N6W05_S42" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_fdcolloc_n6w01 GPU=4  PID=$LAST_PID  (fd-colloc h=0.005 prox=0.1)"

launch_job "diag_reinf_n6w01" 5 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 500 --n-coll 4096 --oversample 8 --lr 5e-4 --seed 42 \
    --loss-type reinforce \
    --min-ess 0 \
    --resume "$CKPT_N6W05_S42" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_reinf_n6w01    GPU=5  PID=$LAST_PID  (reinforce standard)"


# ─── 2D: From-scratch vs transfer — wait for GPUs 6-7 ────────────────────────
# Theory: H9 — high-omega checkpoint may put network in a bad basin for low omega.
# GPU 6: random init (scratch), GPU 7: transfer from bf_ctnn_vcycle (omega=1.0).

echo ""
echo "=== 2D: From-scratch vs transfer (GPUs 6-7 — waiting) ==="

wait_for_gpu_free 6 120
launch_job "diag_scratch_n6w01" 6 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 2000 --n-coll 4096 --oversample 8 --lr 5e-4 --seed 42 \
    --min-ess 0 --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_scratch_n6w01  GPU=6  PID=$LAST_PID  (no resume, 2000 epochs)"

wait_for_gpu_free 7 120
launch_job "diag_xfer_n6w01" 7 \
    --mode bf --n-elec 6 --omega 0.1 \
    --epochs 800 --n-coll 4096 --oversample 8 --lr 5e-4 --seed 42 \
    --min-ess 0 \
    --resume "$CKPT_BF_CTNN" --allow-missing-dmc-ref
PIDS+=("$LAST_PID")
echo "  diag_xfer_n6w01     GPU=7  PID=$LAST_PID  (resume bf_ctnn_vcycle, 800 epochs)"


# ─── Wait for all jobs ────────────────────────────────────────────────────────
echo ""
echo "All 8 Phase 2 jobs submitted. PID list: ${PIDS[*]}"
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
echo " Phase 2 finished at $(date) with rc=$RC"
echo " Summaries: $REPO/$PHASE2_DIR/*_summary.json"
echo "========================================================"
exit $RC
