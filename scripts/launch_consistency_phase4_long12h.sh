#!/usr/bin/env bash
# launch_consistency_phase4_long12h.sh
# Phase 4 first launch: one intentionally long run (>=12h target)

set -euo pipefail

REPO=/itf-fi-ml/home/aleksns/Thesis_repo
OUT_DIR=outputs/consistency_campaign/phase4
TAG=p4a_fdlong12h_n6w01
GPU=1
MODULE_CMD="source /etc/profile.d/z00_lmod.sh && module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"

mkdir -p "$REPO/$OUT_DIR"

cd "$REPO"
eval "$MODULE_CMD"

# This run is sized for a >=12h wall-clock target:
# - 10,000 epochs
# - n-coll=8192 (slower per-epoch than Phase 3 n-coll=4096 runs)
# - patience set high to avoid early-stop truncation
export CUDA_MANUAL_DEVICE="$GPU"
python3 scripts/instrumented_run.py \
  --tag "$TAG" \
  --output-dir "$OUT_DIR" \
  -- \
  --mode bf --n-elec 6 --omega 0.1 \
  --epochs 10000 --patience 20000 \
  --n-coll 8192 --micro-batch 256 --oversample 8 \
  --lr 1e-4 --lr-jas 1e-5 \
  --lr-warmup-epochs 400 --lr-warmup-init-frac 0.05 --lr-min-frac 0.005 \
  --loss-type fd-colloc --fd-h 0.005 --prox-mu 0.1 --fd-huber-delta 0.5 \
  --save-best-window 20 \
  --min-ess 0 --ess-floor-ratio 0.0 \
  --vmc-every 250 --vmc-n 10000 \
  --resume results/arch_colloc/p3b_fdcolloc_n6w01.pt \
  --allow-missing-dmc-ref
