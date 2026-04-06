#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/higher_n/phase3"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== Higher-N Phase 3: N=20 post-bugfix campaign (Adam vs DiagFisher) ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo

N20_BASE="--mode jastrow --n-elec 20 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 1024 --oversample 8 --micro-batch 128 --epochs 4000 --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"
N20_DF="--natural-grad --sr-mode diagonal --fisher-damping 0.01 --fisher-ema 0.99 --fisher-probes 4 --fisher-subsample 512 --nat-momentum 0.9"

# Adam baselines
for spec in \
  "0 n20_adam_w1 1.0 results/arch_colloc/smoke_n20_o1p0.pt" \
  "1 n20_adam_w01 0.1 results/arch_colloc/smoke_n20_o0p1.pt"
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

# DiagFisher runs
for spec in \
  "2 n20_df_w1 1.0 results/arch_colloc/smoke_n20_o1p0.pt" \
  "3 n20_df_w01 0.1 results/arch_colloc/smoke_n20_o0p1.pt"
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
    $N20_DF \
    > "$OUT_DIR/${TAG}_launcher.log" 2>&1 &
  eval "PID_${GPU}=$!"
done

PIDS=($PID_0 $PID_1 $PID_2 $PID_3)
FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    FAIL=1
  fi
done

if [[ $FAIL -ne 0 ]]; then
  echo "One or more Phase 3 runs failed. Check launcher logs in $OUT_DIR"
  exit 1
fi

echo
echo "Phase 3 completed. Summaries:"
for f in "$OUT_DIR"/*_summary.json; do
  [[ -e "$f" ]] || continue
  TAG=$(python3 -c "import json; print(json.load(open('$f'))['tag'])")
  EPOCHS=$(python3 -c "import json; print(json.load(open('$f'))['epochs_logged'])")
  echo "  $TAG: $EPOCHS epochs"
done
