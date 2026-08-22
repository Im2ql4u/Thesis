#!/usr/bin/env bash
# PINN-ansatz campaign v3 — the THESIS RECIPE (run_6e_bf_extend.py 'cusp+bf+joint').
#
# v1 was invalid (CTNN backflow annihilated to |dx|=0 by tanh saturation x COM projection).
# v2 was fixed but still trained backflow+Jastrow JOINTLY FROM SCRATCH, which starves the backflow:
# the Jastrow is an easier route to the same correlation energy, wins the race, and the displacement
# stays near-trivial (|dx|/ell ~ 0.04, rank ~1). That — not physics — is where my "rank-1 collapse"
# came from. v3 uses the thesis curriculum, which forces the backflow to carry real structure:
#   1. PINN Jastrow alone (backflow detached)
#   2. cusp pre-train: fit Delta_x to the same-spin (Pauli) hole
#   3. backflow alone, Jastrow FROZEN
#   4. joint, Jastrow released at lr*0.1
# plus the thesis backflow init: bf_scale=0.7, zero_init_last=False (NOT run_weak_form's 0.05/True).
# Measured effect of staging: |dx|/ell 0.038 -> 0.47 (12x), i.e. the backflow finally does real work.
#
# The curriculum only builds the backflow once, at omega=1. The warm-started omega=0.1/0.01 legs then
# adapt it jointly — re-staging would detach the backflow and discard the structure just created.
set -u
cd "$(dirname "$0")/.."
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
STAMP=2026-07-11_pinn_ansatz_v3
STEPS=3000; NSEG=4; POLISH=500; SRPOLISH=500
BF_SCALE=0.7        # thesis PINN+backflow value (run_weak_form default 0.05 starves it)
# Only GPUs 1,2 are free (another user holds the rest), so the 4 chains share them 2-up.

run_chain() {  # gpu bfarch seed
  local gpu=$1 bfarch=$2 seed=$3 init="" staged="--staged"
  for w in 1.0 0.1 0.01; do
    local out=results/analysis/${STAMP}/pinn_${bfarch}bf_s${seed}_w$(echo $w | sed 's/\./p/')
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/run_phase_analysis.py --N 6 --omega $w \
      --arch pinn --backflow --backflow-arch "$bfarch" --paradigm vmc --optimizer adam \
      $staged --bf-scale-init $BF_SCALE --bf-zero-init-last 0 \
      --steps $STEPS --n-seg $NSEG --polish-steps $POLISH --sr-polish-steps $SRPOLISH \
      --seed "$seed" ${init:+--init "$init"} --outdir "$out" > "$out/train.log" 2>&1
    [ -f "$out/checkpoint.pt" ] || { echo "[chain g$gpu $bfarch s$seed] FAILED at w=$w"; break; }
    echo "[chain g$gpu $bfarch s$seed] done w=$w  $(grep -o '|dx|/ell=[0-9.]*' "$out/train.log" | tail -1)"
    init="$out/checkpoint.pt"
    staged=""   # curriculum only at the first omega; later legs warm-start and fine-tune jointly
  done
}

run_chain 1 ctnn 0 &
run_chain 1 conv 0 &
run_chain 2 ctnn 1 &
run_chain 2 conv 1 &
wait
echo "=== PINN-ANSATZ v3 (thesis recipe) DONE $(date +%H:%M:%S) ==="
