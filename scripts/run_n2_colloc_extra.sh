#!/usr/bin/env bash
# N=2 collocation (colloc_sr) at a given omega: base build -> colloc phase.
# Mirrors scripts/orchestrate_paradigm_optimizer.py (N=2 config) for the two missing omegas.
set -e
ROOT="/itf-fi-ml/home/aleksns/Thesis"
cd "$ROOT"
W="$1"; WTAG="$2"; SEED="${3:-0}"
OUT="$ROOT/results/analysis/2026-08-28_N2_colloc_extra"
BASE="$OUT/base_N2_s${SEED}_${WTAG}"
CELL="$OUT/N2_s${SEED}_${WTAG}_colloc_sr"
mkdir -p "$BASE" "$CELL"
COMMON="--N 2 --omega $W --arch pinn --backflow --backflow-arch ctnn --bf-scale-init 0.7 --bf-zero-init-last 0"

echo "[base] N=2 w=$W"
python3.11 -u scripts/run_phase_analysis.py $COMMON --build-base-only --cusp-steps 0 \
  --steps 3000 --batch 2048 --eval-samples 1024 --seed $SEED --outdir "$BASE" \
  > "$BASE/train.log" 2>&1

echo "[colloc] N=2 w=$W"
python3.11 -u scripts/run_phase_analysis.py $COMMON --paradigm colloc --optimizer sr \
  --steps 1500 --n-seg 5 --polish-steps 0 --sr-polish-steps 0 \
  --batch 2048 --eval-samples 1024 --final-samples 8192 --align-samples 2048 \
  --seed $SEED --init "$BASE/checkpoint.pt" --outdir "$CELL" \
  > "$CELL/train.log" 2>&1
echo "[done] N=2 w=$W -> $CELL"
