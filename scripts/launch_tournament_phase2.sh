#!/usr/bin/env bash
set -euo pipefail

# Tournament Phase 2 — Main Optimizer Tournament at N=6
# 7 parallel runs across GPUs 0-6, ~12-28h wall time
#
# Phase 1 findings informed this run matrix:
#   - CG-SR + FD-colloc at ω=0.1 is most promising (oscillates but finds better regions)
#   - DiagFisher + REINFORCE at ω=0.1 maintained baseline (cheap)
#   - DiagFisher + REINFORCE at ω=0.001 maintained baseline
#   - Dropped: Woodbury (redundant w/ CG-SR), DiagFisher+FD (diagonal approx too weak)
#   - Dropped: CG-SR+REINF at ω=0.001 (ESS too low for full Fisher)
#   - Added: lower-LR variant of CG-SR+FD to address oscillation
#
# GPU allocation:
#   0: t2_cg_fd_w01      — CG-SR + FD-colloc, ω=0.1, LR=2e-4                 (2000 ep, ~9h)
#   1: t2_cg_fd_w01_lo   — CG-SR + FD-colloc, ω=0.1, LR=5e-5 (low-LR test)  (2000 ep, ~9h)
#   2: t2_df_re_w01      — DiagFisher + REINFORCE, ω=0.1                      (6000 ep, ~7h)
#   3: t2_df_re_w001     — DiagFisher + REINFORCE, ω=0.001                    (6000 ep, ~6h)
#   4: t2_adam_fd_w01     — Adam + FD-colloc, ω=0.1 (baseline control)         (8000 ep, ~22h)
#   5: t2_adam_re_w001    — Adam + REINFORCE, ω=0.001 (baseline control)       (8000 ep, ~7h)
#   6: t2_cg_scratch_w01  — CG-SR + FD-colloc, ω=0.1, FROM SCRATCH            (3000 ep, ~13h)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/tournament/phase2"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== Tournament Phase 2: Main Optimizer Tournament ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo ""

# Common params for ω=0.1 warm-start runs
W01_RESUME="results/arch_colloc/p4m1_fdmatched_n6w01.pt"
W01_FD_COMMON="--loss-type fd-colloc --fd-h 0.01 --fd-huber-delta 0.5 --prox-mu 0.1"

# Common params for ω=0.001 warm-start runs
W001_RESUME="results/arch_colloc/p4m2_nogate_n6w001.pt"

# --- GPU 0: CG-SR + FD-colloc at omega=0.1 (standard LR) ---
# Phase 1 showed this reaches training E below exact but oscillates.
# Warmup 30ep should stabilize the opening epochs. 2000ep at ~16s/ep ≈ 9h.
echo "[GPU 0] t2_cg_fd_w01: CG-SR + FD-colloc, ω=0.1, LR=2e-4 warmup=30"
CUDA_MANUAL_DEVICE=0 $PYTHON $SCRIPT \
  --tag t2_cg_fd_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume "$W01_RESUME" \
  --n-elec 6 --omega 0.1 --seed 42 \
  $W01_FD_COMMON \
  --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --epochs 2000 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.01 \
  --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 \
  --natural-grad --sr-mode cg --sr-cg-iters 50 --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 200 --vmc-n 15000 --save-best-window 30 \
  > "$OUT_DIR/t2_cg_fd_w01_launcher.log" 2>&1 &
PID0=$!

# --- GPU 1: CG-SR + FD-colloc at omega=0.1 (LOW LR to reduce oscillation) ---
# Key test: does 4× lower LR eliminate the oscillation while still converging?
echo "[GPU 1] t2_cg_fd_w01_lo: CG-SR + FD-colloc, ω=0.1, LR=5e-5 (low)"
CUDA_MANUAL_DEVICE=1 $PYTHON $SCRIPT \
  --tag t2_cg_fd_w01_lo \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume "$W01_RESUME" \
  --n-elec 6 --omega 0.1 --seed 42 \
  $W01_FD_COMMON \
  --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --epochs 2000 --lr 5e-5 --lr-jas 5e-6 --lr-min-frac 0.01 \
  --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 \
  --natural-grad --sr-mode cg --sr-cg-iters 50 --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.05 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 200 --vmc-n 15000 --save-best-window 30 \
  > "$OUT_DIR/t2_cg_fd_w01_lo_launcher.log" 2>&1 &
PID1=$!

# --- GPU 2: DiagFisher + REINFORCE at omega=0.1 ---
# Phase 1 maintained baseline (+0.079%). Very cheap (4.3s/ep). Give it 6000 epochs.
echo "[GPU 2] t2_df_re_w01: DiagFisher + REINFORCE, ω=0.1"
CUDA_MANUAL_DEVICE=2 $PYTHON $SCRIPT \
  --tag t2_df_re_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume "$W01_RESUME" \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 8192 --oversample 8 --micro-batch 1024 \
  --epochs 6000 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.01 \
  --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30 \
  > "$OUT_DIR/t2_df_re_w01_launcher.log" 2>&1 &
PID2=$!

# --- GPU 3: DiagFisher + REINFORCE at omega=0.001 ---
# Phase 1 maintained baseline (+0.123%). Cheap (3.4s/ep). Give it 6000 epochs.
echo "[GPU 3] t2_df_re_w001: DiagFisher + REINFORCE, ω=0.001"
CUDA_MANUAL_DEVICE=3 $PYTHON $SCRIPT \
  --tag t2_df_re_w001 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume "$W001_RESUME" \
  --n-elec 6 --omega 0.001 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 4096 --oversample 32 --micro-batch 1024 \
  --epochs 6000 --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.01 \
  --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30 \
  > "$OUT_DIR/t2_df_re_w001_launcher.log" 2>&1 &
PID3=$!

# --- GPU 4: Adam + FD-colloc at omega=0.1 (BASELINE CONTROL) ---
# Must compare natural grad improvements against continued Adam training.
# Resume from same p4m1 checkpoint with lower LR (0.5× original).
echo "[GPU 4] t2_adam_fd_w01: Adam + FD-colloc, ω=0.1 (baseline control)"
CUDA_MANUAL_DEVICE=4 $PYTHON $SCRIPT \
  --tag t2_adam_fd_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume "$W01_RESUME" \
  --n-elec 6 --omega 0.1 --seed 42 \
  $W01_FD_COMMON \
  --n-coll 8192 --oversample 8 --micro-batch 1024 \
  --epochs 8000 --lr 5e-5 --lr-jas 5e-6 --lr-min-frac 0.01 \
  --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30 \
  > "$OUT_DIR/t2_adam_fd_w01_launcher.log" 2>&1 &
PID4=$!

# --- GPU 5: Adam + REINFORCE at omega=0.001 (BASELINE CONTROL) ---
echo "[GPU 5] t2_adam_re_w001: Adam + REINFORCE, ω=0.001 (baseline control)"
CUDA_MANUAL_DEVICE=5 $PYTHON $SCRIPT \
  --tag t2_adam_re_w001 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume "$W001_RESUME" \
  --n-elec 6 --omega 0.001 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 4096 --oversample 32 --micro-batch 1024 \
  --epochs 8000 --lr 5e-5 --lr-jas 5e-6 --lr-min-frac 0.01 \
  --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30 \
  > "$OUT_DIR/t2_adam_re_w001_launcher.log" 2>&1 &
PID5=$!

# --- GPU 6: CG-SR + FD-colloc at omega=0.1 FROM SCRATCH ---
# THE DECISIVE TEST: If SR converges from base checkpoint to ≤0.03% at ω=0.1
# in one run (like it does at ω=1.0), it conclusively proves the optimization hypothesis.
# Higher initial LR + damping anneal scheme.
echo "[GPU 6] t2_cg_scratch_w01: CG-SR + FD-colloc, ω=0.1, FROM SCRATCH"
CUDA_MANUAL_DEVICE=6 $PYTHON $SCRIPT \
  --tag t2_cg_scratch_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume results/arch_colloc/bf_ctnn_vcycle.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  $W01_FD_COMMON \
  --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --epochs 3000 --lr 5e-4 --lr-jas 5e-5 --lr-min-frac 0.005 \
  --lr-warmup-epochs 50 --lr-warmup-init-frac 0.05 \
  --natural-grad --sr-mode cg --sr-cg-iters 50 --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 200 --vmc-n 15000 --save-best-window 30 \
  > "$OUT_DIR/t2_cg_scratch_w01_launcher.log" 2>&1 &
PID6=$!

echo ""
echo "All 7 tournament runs launched."
echo "PIDs: $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6"
echo ""
echo "Estimated completion times:"
echo "  GPU 0 (CG-SR FD, 2000ep@16s):      ~9h  → $(date -d '+9 hours' '+%H:%M' 2>/dev/null || echo 'N/A')"
echo "  GPU 1 (CG-SR FD lo, 2000ep@16s):    ~9h  → $(date -d '+9 hours' '+%H:%M' 2>/dev/null || echo 'N/A')"
echo "  GPU 2 (DiagF REINF, 6000ep@4.3s):   ~7h  → $(date -d '+7 hours' '+%H:%M' 2>/dev/null || echo 'N/A')"
echo "  GPU 3 (DiagF REINF 001, 6000ep@3.4s):~6h → $(date -d '+6 hours' '+%H:%M' 2>/dev/null || echo 'N/A')"
echo "  GPU 4 (Adam FD ctrl, 8000ep@10s):    ~22h → $(date -d '+22 hours' '+%H:%M' 2>/dev/null || echo 'N/A')"
echo "  GPU 5 (Adam REINF ctrl, 8000ep@1s):  ~7h  → $(date -d '+7 hours' '+%H:%M' 2>/dev/null || echo 'N/A')"
echo "  GPU 6 (CG-SR scratch, 3000ep@16s):   ~13h → $(date -d '+13 hours' '+%H:%M' 2>/dev/null || echo 'N/A')"
echo ""
echo "Waiting for all to complete..."

FAIL=0
for PID_TAG in "$PID0:t2_cg_fd_w01" "$PID1:t2_cg_fd_w01_lo" "$PID2:t2_df_re_w01" \
               "$PID3:t2_df_re_w001" "$PID4:t2_adam_fd_w01" "$PID5:t2_adam_re_w001" \
               "$PID6:t2_cg_scratch_w01"; do
  PID="${PID_TAG%%:*}"
  TAG="${PID_TAG##*:}"
  if wait "$PID"; then
    echo "  [OK]   $TAG (PID $PID) @ $(date)"
  else
    echo "  [FAIL] $TAG (PID $PID, rc=$?) @ $(date)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=== Phase 2 Complete: $(date) ==="
echo "Failures: $FAIL / 7"

# Quick results summary
echo ""
echo "=== Results Summary ==="
for f in "$OUT_DIR"/*_summary.json; do
  [ -f "$f" ] || continue
  TAG=$(python3 -c "import json; print(json.load(open('$f'))['tag'])")
  EPOCHS=$(python3 -c "import json; print(json.load(open('$f'))['epochs_logged'])")
  FINAL_E=$(python3 -c "import json; print(f\"{json.load(open('$f'))['final_entry']['E']:.6f}\")")
  echo "  $TAG: ${EPOCHS} epochs, final E = $FINAL_E"
done
