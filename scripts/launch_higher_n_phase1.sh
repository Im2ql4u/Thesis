#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/higher_n/phase1"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== Higher-N Phase 1: N=12 smoke + N=20 diagnostics ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo

# Shared settings
N12_COMMON="--mode bf --n-elec 12 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 2048 --oversample 8 --micro-batch 256 --epochs 100 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.01 --lr-warmup-epochs 10 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20"
N12_DF="--natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9"
N20_DIAG_COMMON="--mode jastrow --n-elec 20 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 1024 --oversample 8 --micro-batch 128 --epochs 200 --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.01 --lr-warmup-epochs 10 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 50 --vmc-n 10000 --save-best-window 20"

# GPU 0: N=12 DiagFisher omega=0.1
echo "[GPU 0] smoke_n12_df_w01"
CUDA_MANUAL_DEVICE=0 $PYTHON $SCRIPT \
  --tag smoke_n12_df_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/v12b_n12w01.pt \
  --omega 0.1 \
  $N12_COMMON \
  $N12_DF \
  > "$OUT_DIR/smoke_n12_df_w01_launcher.log" 2>&1 &
PID0=$!

# GPU 1: N=12 DiagFisher omega=0.5
echo "[GPU 1] smoke_n12_df_w05"
CUDA_MANUAL_DEVICE=1 $PYTHON $SCRIPT \
  --tag smoke_n12_df_w05 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/v12b_n12w05.pt \
  --omega 0.5 \
  $N12_COMMON \
  $N12_DF \
  > "$OUT_DIR/smoke_n12_df_w05_launcher.log" 2>&1 &
PID1=$!

# GPU 2: N=12 DiagFisher omega=1.0
echo "[GPU 2] smoke_n12_df_w1"
CUDA_MANUAL_DEVICE=2 $PYTHON $SCRIPT \
  --tag smoke_n12_df_w1 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/v12b_n12w1.pt \
  --omega 1.0 \
  $N12_COMMON \
  $N12_DF \
  > "$OUT_DIR/smoke_n12_df_w1_launcher.log" 2>&1 &
PID2=$!

# GPU 3: N=12 Adam control at omega=0.1
echo "[GPU 3] smoke_n12_adam_w01"
CUDA_MANUAL_DEVICE=3 $PYTHON $SCRIPT \
  --tag smoke_n12_adam_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/v12b_n12w01.pt \
  --omega 0.1 \
  $N12_COMMON \
  > "$OUT_DIR/smoke_n12_adam_w01_launcher.log" 2>&1 &
PID3=$!

# GPU 4: N=20 diagnostic at omega=1.0
echo "[GPU 4] smoke_n20_diag_w1"
CUDA_MANUAL_DEVICE=4 $PYTHON $SCRIPT \
  --tag smoke_n20_diag_w1 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/smoke_n20_o1p0.pt \
  --omega 1.0 \
  $N20_DIAG_COMMON \
  > "$OUT_DIR/smoke_n20_diag_w1_launcher.log" 2>&1 &
PID4=$!

# GPU 5: N=20 diagnostic at omega=0.1
echo "[GPU 5] smoke_n20_diag_w01"
CUDA_MANUAL_DEVICE=5 $PYTHON $SCRIPT \
  --tag smoke_n20_diag_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/smoke_n20_o0p1.pt \
  --omega 0.1 \
  $N20_DIAG_COMMON \
  > "$OUT_DIR/smoke_n20_diag_w01_launcher.log" 2>&1 &
PID5=$!

PIDS=($PID0 $PID1 $PID2 $PID3 $PID4 $PID5)
FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    FAIL=1
  fi
done

if [[ $FAIL -ne 0 ]]; then
  echo "One or more runs failed. Check launcher logs in $OUT_DIR"
  exit 1
fi

echo
echo "All runs completed. Summaries:"
for f in "$OUT_DIR"/*_summary.json; do
  [[ -e "$f" ]] || continue
  TAG=$(python3 -c "import json; print(json.load(open('$f'))['tag'])")
  EPOCHS=$(python3 -c "import json; print(json.load(open('$f'))['epochs_logged'])")
  echo "  $TAG: $EPOCHS epochs"
done
