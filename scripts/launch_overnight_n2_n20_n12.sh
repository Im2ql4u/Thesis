#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/higher_n/phase5_overnight_n2_n20_n12"
mkdir -p "$OUT_DIR"

PYTHON=python3
SCRIPT="scripts/instrumented_run.py"

echo "=== Overnight campaign: omega set {1.0, 0.5, 0.1, 0.01, 0.001} only ==="
echo "Start: $(date)"
echo "Output: $OUT_DIR"
echo

# N=2 jastrow continuation (all omegas)
N2_BASE="--mode jastrow --n-elec 2 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 4096 --micro-batch 1024 --epochs 110000 --lr 6e-5 --lr-jas 6e-6 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 1000 --vmc-n 15000 --save-best-window 30"

# N=20 continuation (strict omega set, no 0.28)
N20_BASE="--mode jastrow --n-elec 20 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 1024 --oversample 8 --micro-batch 128 --epochs 5600 --lr 8e-5 --lr-jas 8e-6 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"

# N=12 continuation
N12_BASE="--mode bf --n-elec 12 --seed 42 --loss-type reinforce --direct-weight 0 --clip-el 5.0 --n-coll 2048 --oversample 8 --micro-batch 256 --epochs 5600 --lr 1.2e-4 --lr-jas 1.2e-5 --lr-min-frac 0.01 --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1 --patience 0 --vmc-every 500 --vmc-n 15000 --save-best-window 30"

# GPUs 0-4: N=2 requested omegas {0.001, 0.01, 0.1, 0.5, 1.0}
for spec in \
  "0 n2ovr2_w0001 0.001 32 results/arch_colloc/v16_n2w0001_transfer.pt" \
  "1 n2ovr2_w001 0.01 16 results/arch_colloc/smoke_n2_o0p01.pt" \
  "2 n2ovr2_w01 0.1 8 results/arch_colloc/smoke_n2_o0p1.pt" \
  "3 n2ovr2_w05 0.5 8 results/arch_colloc/smoke_n2_o0p5.pt" \
  "4 n2ovr2_w1 1.0 8 results/arch_colloc/smoke_n2_o1p0.pt"
do
  set -- $spec
  GPU=$1; TAG=$2; OMEGA=$3; OVERSAMPLE=$4; RESUME=$5
  echo "[GPU $GPU] $TAG"
  CUDA_MANUAL_DEVICE=$GPU $PYTHON $SCRIPT \
    --tag "$TAG" \
    --output-dir "$OUT_DIR" \
    -- \
    --resume "$RESUME" \
    --omega "$OMEGA" \
    --oversample "$OVERSAMPLE" \
    $N2_BASE \
    > "$OUT_DIR/${TAG}_launcher.log" 2>&1 &
  eval "PID_${GPU}=$!"
done

# GPU 5: N=20 omega=1.0 continuation
echo "[GPU 5] n20ovr2_w1"
CUDA_MANUAL_DEVICE=5 $PYTHON $SCRIPT \
  --tag n20ovr2_w1 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/n20x2_adam_w1_best.pt \
  --omega 1.0 \
  $N20_BASE \
  > "$OUT_DIR/n20ovr2_w1_launcher.log" 2>&1 &
PID_5=$!

# GPU 6: N=20 omega=0.5 continuation
echo "[GPU 6] n20ovr2_w05"
CUDA_MANUAL_DEVICE=6 $PYTHON $SCRIPT \
  --tag n20ovr2_w05 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/n20x2_adam_w05_best.pt \
  --omega 0.5 \
  $N20_BASE \
  > "$OUT_DIR/n20ovr2_w05_launcher.log" 2>&1 &
PID_6=$!

# GPU 7: N=12 omega=0.1 continuation
echo "[GPU 7] n12ovr2_w01"
CUDA_MANUAL_DEVICE=7 $PYTHON $SCRIPT \
  --tag n12ovr2_w01 \
  --output-dir "$OUT_DIR" \
  -- \
  --resume results/arch_colloc/n12_adam_w01_best.pt \
  --omega 0.1 \
  $N12_BASE \
  > "$OUT_DIR/n12ovr2_w01_launcher.log" 2>&1 &
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
echo "Overnight campaign completed. Summaries:"
for f in "$OUT_DIR"/*_summary.json; do
  [[ -e "$f" ]] || continue
  TAG=$(python3 -c "import json; print(json.load(open('$f'))['tag'])")
  EPOCHS=$(python3 -c "import json; print(json.load(open('$f'))['epochs_logged'])")
  echo "  $TAG: $EPOCHS epochs"
done
