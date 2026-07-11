#!/usr/bin/env bash
# PINN-ansatz campaign v2 — RERUN after the dead-backflow fix.
#
# v1 (results/analysis/2026-07-04_pinn_ansatz) is INVALID: trained at lr=3e-3 with a single
# param group and clip=5, which drove the backflow's dx_head pre-activation into tanh
# saturation. Saturated tanh gives every particle the same +-1, and CTNNBackflowNet's
# zero-mean (COM) projection then annihilates Delta_x *exactly* -> |dx|=0, zero gradient,
# permanently dead backflow. v2 adopts the thesis optimiser (run_weak_form.py):
#   lr 5e-4 on the backflow, lr*0.1 on the Jastrow, grad_clip=1.0, bf hidden/msg = 128.
# train.py now logs |dx|/ell and tanh_sat every log step so a dead backflow is caught early.
set -u
cd "$(dirname "$0")/.."
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
STAMP=2026-07-11_pinn_ansatz_v2
STEPS=2500; NSEG=4; POLISH=500; SRPOLISH=500   # lr is 6x lower than v1 -> more steps to converge
OMEGAS="1.0 0.1 0.01"
# Only GPUs 1,2 are free (another user holds 0,3-7), so the 4 chains share them 2-up.

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
    echo "[chain g$gpu $bfarch s$seed] done w=$w  $(grep -o '|dx|/ell=[0-9.]*' "$out/train.log" | tail -1)"
    init="$out/checkpoint.pt"
  done
}

run_chain 1 ctnn 0 &
run_chain 1 conv 0 &
run_chain 2 ctnn 1 &
run_chain 2 conv 1 &
wait
echo "=== PINN-ANSATZ v2 CAMPAIGN DONE $(date +%H:%M:%S) ==="
