#!/usr/bin/env bash
# Train a well-trained N=12 DeepSet-correlator (+ CTNN backflow) so the mode-naming /
# non-physical-mass contrast can be shown at N=12, not just N=6. Cascade omega down from
# the easy (kinetic) regime to the deep-Wigner point where the DeepSet is expected to fail.
set -u
cd "$(dirname "$0")/.."
PY="python3.11"
OUT="results/analysis/2026-08-28_N12_deepset_cascade"
mkdir -p "$OUT"
RUN="scripts/run_phase_analysis.py"
COMMON="--arch deepset_big --backflow --backflow-arch ctnn --bf-scale-init 0.7 --bf-zero-init-last 0 --batch 1024 --eval-samples 512"

prev=""
for stage in "1.0:w1:5000" "0.5:w0p5:5000" "0.1:w0p1:6000" "0.01:w0p01:8000"; do
  W="${stage%%:*}"; rest="${stage#*:}"; WT="${rest%%:*}"; STEPS="${rest##*:}"
  DIR="$OUT/N12_deepset_${WT}_s0"
  if [ ! -f "$DIR/checkpoint.pt" ]; then
    INIT=""
    [ -n "$prev" ] && INIT="--init $prev/checkpoint.pt"
    $PY -u "$RUN" --N 12 --omega "$W" $COMMON --steps "$STEPS" --seed 0 \
      $INIT --outdir "$DIR" > "$OUT/train_${WT}.log" 2>&1
  fi
  echo "done N12 deepset w=$W"
  prev="$DIR"
done
echo "ALL N12 DEEPSET DONE"
