#!/usr/bin/env bash
set -euo pipefail

# Tournament Phase 1 — Smoke Tests: verify SR/natural-gradient at low omega post-bugfix
# 6 parallel runs, 100 epochs each, one per GPU (0-5)
# Purpose: confirm stability before committing to long Phase 2 runs

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/tournament/phase1_smoke"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== Tournament Phase 1: Smoke Tests ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo ""

# --- GPU 0: DiagFisher + FD-colloc at omega=0.1 ---
echo "[GPU 0] smoke_df_fd_w01: DiagFisher + FD-colloc, omega=0.1"
CUDA_MANUAL_DEVICE=0 $PYTHON $SCRIPT \
  --tag smoke_df_fd_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type fd-colloc --fd-h 0.01 --fd-huber-delta 0.5 --prox-mu 0.1 \
  --n-coll 8192 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20 \
  > "$OUT_DIR/smoke_df_fd_w01_launcher.log" 2>&1 &
PID0=$!

# --- GPU 1: CG-SR + FD-colloc at omega=0.1 ---
echo "[GPU 1] smoke_cg_fd_w01: CG-SR(50i) + FD-colloc, omega=0.1"
CUDA_MANUAL_DEVICE=1 $PYTHON $SCRIPT \
  --tag smoke_cg_fd_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type fd-colloc --fd-h 0.01 --fd-huber-delta 0.5 --prox-mu 0.1 \
  --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode cg --sr-cg-iters 50 --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20 \
  > "$OUT_DIR/smoke_cg_fd_w01_launcher.log" 2>&1 &
PID1=$!

# --- GPU 2: Woodbury-SR + FD-colloc at omega=0.1 ---
echo "[GPU 2] smoke_wb_fd_w01: Woodbury-SR + FD-colloc, omega=0.1"
CUDA_MANUAL_DEVICE=2 $PYTHON $SCRIPT \
  --tag smoke_wb_fd_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type fd-colloc --fd-h 0.01 --fd-huber-delta 0.5 --prox-mu 0.1 \
  --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode woodbury --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20 \
  > "$OUT_DIR/smoke_wb_fd_w01_launcher.log" 2>&1 &
PID2=$!

# --- GPU 3: DiagFisher + REINFORCE at omega=0.1 ---
echo "[GPU 3] smoke_df_re_w01: DiagFisher + REINFORCE, omega=0.1"
CUDA_MANUAL_DEVICE=3 $PYTHON $SCRIPT \
  --tag smoke_df_re_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.1 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 8192 --oversample 8 --micro-batch 1024 \
  --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20 \
  > "$OUT_DIR/smoke_df_re_w01_launcher.log" 2>&1 &
PID3=$!

# --- GPU 4: DiagFisher + REINFORCE at omega=0.001 ---
echo "[GPU 4] smoke_df_re_w001: DiagFisher + REINFORCE, omega=0.001"
CUDA_MANUAL_DEVICE=4 $PYTHON $SCRIPT \
  --tag smoke_df_re_w001 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001.pt \
  --n-elec 6 --omega 0.001 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 4096 --oversample 32 --micro-batch 1024 \
  --epochs 100 --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 \
  --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20 \
  > "$OUT_DIR/smoke_df_re_w001_launcher.log" 2>&1 &
PID4=$!

# --- GPU 5: CG-SR + REINFORCE at omega=0.001 ---
echo "[GPU 5] smoke_cg_re_w001: CG-SR(50i) + REINFORCE, omega=0.001"
CUDA_MANUAL_DEVICE=5 $PYTHON $SCRIPT \
  --tag smoke_cg_re_w001 \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001.pt \
  --n-elec 6 --omega 0.001 --seed 42 \
  --loss-type reinforce --direct-weight 0 --clip-el 5.0 \
  --n-coll 4096 --oversample 32 --micro-batch 1024 \
  --epochs 100 --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.02 \
  --natural-grad --sr-mode cg --sr-cg-iters 50 --fisher-damping 0.01 \
  --fisher-subsample 1024 --sr-max-param-change 0.1 --sr-trust-region 1.0 \
  --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20 \
  > "$OUT_DIR/smoke_cg_re_w001_launcher.log" 2>&1 &
PID5=$!

echo ""
echo "All 6 smoke tests launched."
echo "PIDs: $PID0 $PID1 $PID2 $PID3 $PID4 $PID5"
echo ""
echo "Waiting for all to complete..."

FAIL=0
for PID_TAG in "$PID0:smoke_df_fd_w01" "$PID1:smoke_cg_fd_w01" "$PID2:smoke_wb_fd_w01" \
               "$PID3:smoke_df_re_w01" "$PID4:smoke_df_re_w001" "$PID5:smoke_cg_re_w001"; do
  PID="${PID_TAG%%:*}"
  TAG="${PID_TAG##*:}"
  if wait "$PID"; then
    echo "  [OK]   $TAG (PID $PID)"
  else
    echo "  [FAIL] $TAG (PID $PID, rc=$?)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=== Phase 1 Smoke Complete: $(date) ==="
echo "Failures: $FAIL / 6"

# Quick summary extraction
echo ""
echo "=== Quick Results Summary ==="
for TAG in smoke_df_fd_w01 smoke_cg_fd_w01 smoke_wb_fd_w01 \
           smoke_df_re_w01 smoke_df_re_w001 smoke_cg_re_w001; do
  SUMMARY="$OUT_DIR/${TAG}_summary.json"
  if [ -f "$SUMMARY" ]; then
    $PYTHON -c "
import json
with open('$SUMMARY') as f:
    d = json.load(f)
fe = d.get('final_entry', {})
E = fe.get('E', 'N/A')
ess = fe.get('ess', 'N/A')
dt = fe.get('dt', 'N/A')
nep = d.get('epochs_logged', 0)
print(f'  {d[\"tag\"]:25s}  epochs={nep:3d}  E={E:12.6f}  ESS={ess:6.1f}  dt={dt:5.1f}s')
"
  else
    echo "  $TAG  [NO SUMMARY FILE]"
  fi
done

exit $FAIL
