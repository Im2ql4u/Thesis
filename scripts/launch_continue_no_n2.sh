#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/higher_n/phase6_continue_no_n2"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== Continuation campaign (NO N=2) ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo

N20_BASE="--mode jastrow --n-elec 20 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 1024 --oversample 8 --micro-batch 128 --epochs 5600 --lr 7e-5 --lr-jas 7e-6 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"
N12_BASE="--mode bf --n-elec 12 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 2048 --oversample 8 --micro-batch 256 --epochs 5600 --lr 1.1e-4 --lr-jas 1.1e-5 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"

# N=20 runs that still need improvement
for spec in \
  "0 n20cont_w1 1.0 results/arch_colloc/n20ovr2_w1_best.pt" \
  "1 n20cont_w05 0.5 results/arch_colloc/n20ovr2_w05_best.pt" \
  "2 n20cont_w01 0.1 results/arch_colloc/n20x2_adam_w01_best.pt"
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

# N=12 runs that still need improvement
for spec in \
  "3 n12cont_w01 0.1 results/arch_colloc/n12ovr2_w01_best.pt" \
  "4 n12cont_w001 0.01 results/arch_colloc/n12x2_adam_w001_best.pt"
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

PIDS=($PID_0 $PID_1 $PID_2 $PID_3 $PID_4)
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
