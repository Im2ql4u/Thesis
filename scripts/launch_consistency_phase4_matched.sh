#!/usr/bin/env bash
set -euo pipefail

# Phase 4 matched continuations (Step 2.2)
# - 4m1: omega=0.1 continuation, matched to p4a-style FD-colloc recipe
# - 4m2: omega=0.001 continuation, matched to no-gate low-omega recipe

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

PHASE4_DIR="outputs/consistency_campaign/phase4"
mkdir -p "$PHASE4_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

TAG_W01="p4m1_fdmatched_n6w01"
TAG_W001="p4m2_nogate_n6w001"

LOG_W01="$PHASE4_DIR/${TAG_W01}_${TS}.log"
LOG_W001="$PHASE4_DIR/${TAG_W001}_${TS}.log"

# Runtime calibration from observed ~15.5 s/epoch on N=6, n_coll=8192.
# 900 epochs ~= 3.9h, leaving margin for eval and overhead.
EPOCHS_4H=900

# Launch omega=0.1 on GPU 1
CUDA_MANUAL_DEVICE=1 python3 src/run_weak_form.py \
  --mode bf \
  --resume results/arch_colloc/p4a_fdlong12h_n6w01_best_snapshot_20260331_110910.pt \
  --n-elec 6 \
  --omega 0.1 \
  --seed 42 \
  --loss-type fd-colloc \
  --n-coll 8192 \
  --oversample 8 \
  --epochs "$EPOCHS_4H" \
  --lr 1e-4 \
  --lr-jas 1e-5 \
  --lr-warmup-epochs 120 \
  --lr-warmup-init-frac 0.05 \
  --lr-min-frac 0.005 \
  --fd-h 0.005 \
  --fd-huber-delta 0.5 \
  --prox-mu 0.1 \
  --micro-batch 1024 \
  --patience 900 \
  --vmc-every 50 \
  --vmc-n 10000 \
  --save-best-window 20 \
  --tag "$TAG_W01" \
  > "$LOG_W01" 2>&1 &
PID_W01=$!

# Launch omega=0.001 on GPU 2
CUDA_MANUAL_DEVICE=2 python3 src/run_weak_form.py \
  --mode bf \
  --resume results/arch_colloc/p3a_nogate_n6w001_best.pt \
  --n-elec 6 \
  --omega 0.001 \
  --seed 42 \
  --loss-type reinforce \
  --n-coll 8192 \
  --oversample 64 \
  --epochs "$EPOCHS_4H" \
  --lr 5e-4 \
  --lr-jas 5e-5 \
  --lr-warmup-epochs 80 \
  --lr-min-frac 0.02 \
  --micro-batch 1024 \
  --patience 900 \
  --vmc-every 50 \
  --vmc-n 10000 \
  --save-best-window 20 \
  --tag "$TAG_W001" \
  > "$LOG_W001" 2>&1 &
PID_W001=$!

cat <<EOF
Launched matched Phase 4 runs:
  $TAG_W01 (GPU 1) pid=$PID_W01 log=$LOG_W01
  $TAG_W001 (GPU 2) pid=$PID_W001 log=$LOG_W001
Expected runtime per run: ~3.9h (900 epochs x ~15.5s)
EOF
