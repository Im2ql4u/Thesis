#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/higher_n/phase2"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== Higher-N Phase 2: N=12 full campaign (DiagFisher vs Adam) ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo

N12_BASE="--mode bf --n-elec 12 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 2048 --oversample 8 --micro-batch 256 --epochs 4000 --lr 2e-4 --lr-jas 2e-5 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"
N12_DF="--natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 2048 --nat-momentum 0.9"

# DiagFisher runs
for spec in \
  "0 n12_df_w01 0.1 results/arch_colloc/v12b_n12w01.pt" \
  "1 n12_df_w05 0.5 results/arch_colloc/v12b_n12w05.pt" \
  "2 n12_df_w1 1.0 results/arch_colloc/v12b_n12w1.pt"
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
    $N12_BASE \
    $N12_DF \
    > "$OUT_DIR/${TAG}_launcher.log" 2>&1 &
  eval "PID_${GPU}=$!"
done

# Adam control runs
for spec in \
  "3 n12_adam_w01 0.1 results/arch_colloc/v12b_n12w01.pt" \
  "4 n12_adam_w05 0.5 results/arch_colloc/v12b_n12w05.pt" \
  "5 n12_adam_w1 1.0 results/arch_colloc/v12b_n12w1.pt"
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
    $N12_BASE \
    > "$OUT_DIR/${TAG}_launcher.log" 2>&1 &
  eval "PID_${GPU}=$!"
done

PIDS=($PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5)
FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    FAIL=1
  fi
done

if [[ $FAIL -ne 0 ]]; then
  echo "One or more Phase 2 runs failed. Check launcher logs in $OUT_DIR"
  exit 1
fi

echo
echo "Phase 2 completed. Summaries:"
for f in "$OUT_DIR"/*_summary.json; do
  [[ -e "$f" ]] || continue
  TAG=$(python3 -c "import json; print(json.load(open('$f'))['tag'])")
  EPOCHS=$(python3 -c "import json; print(json.load(open('$f'))['epochs_logged'])")
  echo "  $TAG: $EPOCHS epochs"
done
