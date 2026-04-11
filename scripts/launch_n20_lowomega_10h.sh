#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/higher_n/phase4_n20_lowomega_escalation"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== N20 regimes + lower-omega escalation (8 GPU long run) ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo

# -----------------------------
# N=20 regime continuation runs
# -----------------------------
N20_BASE="--mode jastrow --n-elec 20 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 1024 --oversample 8 --micro-batch 128 --epochs 5600 --lr 8e-5 --lr-jas 8e-6 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"

# -----------------------------
# Lower-omega escalation runs
# -----------------------------
N12_LOW_BASE="--mode bf --n-elec 12 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 2048 --oversample 8 --micro-batch 256 --epochs 2200 --lr 1.5e-4 --lr-jas 1.5e-5 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"
N12_DF="--natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9"

N6_LOW_BASE="--mode bf --n-elec 6 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 4096 --oversample 32 --micro-batch 1024 --epochs 10000 --lr 8e-5 --lr-jas 8e-6 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"
N6_DF="--natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9"

# GPU 0-3: N20 regime continuation
for spec in \
  "0 n20x2_adam_w1 1.0 results/arch_colloc/n20_adam_w1_best.pt" \
  "1 n20x2_adam_w05 0.5 results/arch_colloc/smoke_n20_o0p5.pt" \
  "2 n20x2_adam_w028 0.28 results/arch_colloc/smoke_n20_o0p28.pt" \
  "3 n20x2_adam_w01 0.1 results/arch_colloc/n20_adam_w01_best.pt"
do
  set -- $spec
  GPU=$1; TAG=$2; OMEGA=$3; RESUME=$4
  echo "[GPU $GPU] $TAG"
  CUDA_MANUAL_DEVICE=$GPU $PYTHON $SCRIPT \
    --tag "$TAG" \
    --output-dir "$OUT_DIR" \
    -- \
    --resume "$RESUME" \
    --omega "$OMEGA" \
    $N20_BASE \
    > "$OUT_DIR/${TAG}_launcher.log" 2>&1 &
  eval "PID_${GPU}=$!"
done

# GPU 4-5: N12 omega=0.01 escalation (Adam vs DiagFisher)
echo "[GPU 4] n12x2_adam_w001"
CUDA_MANUAL_DEVICE=4 $PYTHON $SCRIPT \
  --tag n12x2_adam_w001 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/smoke_n12_o0p01.pt \
  --omega 0.01 \
  $N12_LOW_BASE \
  > "$OUT_DIR/n12x2_adam_w001_launcher.log" 2>&1 &
PID_4=$!

echo "[GPU 5] n12x2_df_w001"
CUDA_MANUAL_DEVICE=5 $PYTHON $SCRIPT \
  --tag n12x2_df_w001 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/smoke_n12_o0p01.pt \
  --omega 0.01 \
  $N12_LOW_BASE \
  $N12_DF \
  > "$OUT_DIR/n12x2_df_w001_launcher.log" 2>&1 &
PID_5=$!

# GPU 6-7: N6 omega=0.001 escalation (Adam vs DiagFisher)
echo "[GPU 6] n6x2_adam_w0001"
CUDA_MANUAL_DEVICE=6 $PYTHON $SCRIPT \
  --tag n6x2_adam_w0001 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/p4m2_nogate_n6w001_best.pt \
  --omega 0.001 \
  $N6_LOW_BASE \
  > "$OUT_DIR/n6x2_adam_w0001_launcher.log" 2>&1 &
PID_6=$!

echo "[GPU 7] n6x2_df_w0001"
CUDA_MANUAL_DEVICE=7 $PYTHON $SCRIPT \
  --tag n6x2_df_w0001 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/p4m2_nogate_n6w001_best.pt \
  --omega 0.001 \
  $N6_LOW_BASE \
  $N6_DF \
  > "$OUT_DIR/n6x2_df_w0001_launcher.log" 2>&1 &
PID_7=$!

PIDS=($PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5 $PID_6 $PID_7)
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
echo "Campaign completed. Summaries:"
for f in "$OUT_DIR"/*_summary.json; do
  [[ -e "$f" ]] || continue
  TAG=$(python3 -c "import json; print(json.load(open('$f'))['tag'])")
  EPOCHS=$(python3 -c "import json; print(json.load(open('$f'))['epochs_logged'])")
  echo "  $TAG: $EPOCHS epochs"
done
