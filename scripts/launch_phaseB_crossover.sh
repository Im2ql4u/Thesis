#!/usr/bin/env bash
# Phase B1 (crossover-first): seed the omega=0.01 crossover with analysis-grade GS.
# 3 seeds x {CTNN, DeepSet}, warm-started from each arch's omega=0.1 cascade checkpoint,
# one run per GPU (1..6), so wall-clock ~ one run. Puts error bars on the d_eff crossover
# (CTNN ~5.2 vs DeepSet ~3.24) and ensures the DeepSet low-omega checkpoints are genuine GS.
#
# Seeds here vary the sampling + SR/optimisation stochasticity from a shared omega=0.1 warm-start
# (measurement error bars around the crossover), NOT independent training basins from omega=1
# (that is the full seeded cascade, deferred). Labelled as such in the writeup.
set -u
cd "$(dirname "$0")/.."
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

STAMP=2026-07-02
STEPS=400; NSEG=4; POLISH=120; SRPOLISH=200
CTNN_INIT=results/analysis/2026-06-15_N6_w01_ctnn_big_bf_casc/checkpoint.pt
DS_INIT=results/analysis/2026-06-15_N6_w01_deepset_big_bf_casc/checkpoint.pt

run() {  # $1 gpu  $2 arch  $3 short  $4 seed  $5 init
  local gpu=$1 arch=$2 short=$3 seed=$4 init=$5
  local out=results/analysis/${STAMP}_N6_w001_${short}_s${seed}
  mkdir -p "$out"
  echo "[launch] GPU$gpu $short seed$seed -> $out"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python3 -u scripts/run_phase_analysis.py \
    --N 6 --omega 0.01 --arch "$arch" --backflow --init "$init" \
    --steps $STEPS --n-seg $NSEG --polish-steps $POLISH --sr-polish-steps $SRPOLISH \
    --seed "$seed" --outdir "$out" > "$out/train.log" 2>&1 &
}

run 1 ctnn_vcycle_big ctnn    0 "$CTNN_INIT"
run 2 ctnn_vcycle_big ctnn    1 "$CTNN_INIT"
run 3 ctnn_vcycle_big ctnn    2 "$CTNN_INIT"
run 4 deepset_big     deepset 0 "$DS_INIT"
run 5 deepset_big     deepset 1 "$DS_INIT"
run 6 deepset_big     deepset 2 "$DS_INIT"

echo "[launch] 6 runs started on GPUs 1-6; waiting for all to finish..."
wait
echo "[launch] ALL DONE $(date +%H:%M:%S)"
