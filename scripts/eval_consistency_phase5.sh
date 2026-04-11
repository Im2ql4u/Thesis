#!/usr/bin/env bash
set -euo pipefail

# Phase 5 evaluation — evaluate final and rolling-best checkpoints for the
# 15-run N=6 multi-seed matrix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/consistency_campaign/phase5"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="$OUT_DIR/eval_phase5_${TS}.json"

CKPTS=()
for tag in \
  p5_n6w1_s42 p5_n6w1_s11 p5_n6w1_s77 \
  p5_n6w05_s42 p5_n6w05_s11 p5_n6w05_s77 \
  p5_n6w01_s42 p5_n6w01_s11 p5_n6w01_s77 \
  p5_n6w001_s42 p5_n6w001_s11 p5_n6w001_s77 \
  p5_n6w0001_s42 p5_n6w0001_s11 p5_n6w0001_s77
do
  CKPTS+=(--checkpoint "results/arch_colloc/${tag}.pt")
  CKPTS+=(--checkpoint "results/arch_colloc/${tag}_best.pt")
done

python3 scripts/eval_checkpoint_matrix.py \
  "${CKPTS[@]}" \
  --n-eval 100000 \
  --output-json "$OUT_JSON"

echo "Wrote: $OUT_JSON"