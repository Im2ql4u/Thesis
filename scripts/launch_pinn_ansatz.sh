#!/usr/bin/env bash
# Train the REAL thesis ansatz — Slater x PINN Jastrow x backflow — via VMC (adam + annealed SR polish)
# to DMC quality, across the omega cascade, for BOTH backflow types:
#   ctnn : message-passing CTNNBackflowNet (the thesis ansatz; message passing lives HERE)
#   conv : per-particle BackflowNet (the contrast: what does message passing in the backflow buy?)
# One chain = one omega-cascade (1 -> 0.1 -> 0.01, warm-started) on one GPU. 4 chains on GPUs 1-4.
set -u
cd "$(dirname "$0")/.."
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
STAMP=2026-07-04_pinn_ansatz
STEPS=1000; NSEG=4; POLISH=200; SRPOLISH=400
OMEGAS="1.0 0.1 0.01"

run_chain() {  # gpu bfarch seed
  local gpu=$1 bfarch=$2 seed=$3 init=""
  for w in $OMEGAS; do
    local out=results/analysis/${STAMP}/pinn_${bfarch}bf_s${seed}_w$(echo $w | sed 's/\./p/')
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/run_phase_analysis.py --N 6 --omega $w \
      --arch pinn --backflow --backflow-arch "$bfarch" --paradigm vmc --optimizer adam \
      --steps $STEPS --n-seg $NSEG --polish-steps $POLISH --sr-polish-steps $SRPOLISH \
      --seed "$seed" ${init:+--init "$init"} --outdir "$out" > "$out/train.log" 2>&1
    [ -f "$out/checkpoint.pt" ] || { echo "[chain g$gpu $bfarch s$seed] FAILED at w=$w"; break; }
    echo "[chain g$gpu $bfarch s$seed] done w=$w"
    init="$out/checkpoint.pt"
  done
}

run_chain 1 ctnn 0 &
run_chain 2 ctnn 1 &
run_chain 3 conv 0 &
run_chain 4 conv 1 &
wait
echo "=== PINN-ANSATZ CAMPAIGN DONE $(date +%H:%M:%S) ==="
