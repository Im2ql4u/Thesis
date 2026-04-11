#!/usr/bin/env bash
set -euo pipefail

# Phase 4 matched evaluation (Step 2.3)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/consistency_campaign/phase4"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="$OUT_DIR/eval_matched_${TS}.json"

python3 scripts/eval_checkpoint_matrix.py \
  --checkpoint results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --checkpoint results/arch_colloc/p4m1_fdmatched_n6w01_best.pt \
  --checkpoint results/arch_colloc/p4m2_nogate_n6w001.pt \
  --checkpoint results/arch_colloc/p4m2_nogate_n6w001_best.pt \
  --checkpoint results/arch_colloc/p3b_fdcolloc_n6w01.pt \
  --checkpoint results/arch_colloc/p3a_nogate_n6w001.pt \
  --n-eval 100000 \
  --output-json "$OUT_JSON"

echo "Wrote: $OUT_JSON"
