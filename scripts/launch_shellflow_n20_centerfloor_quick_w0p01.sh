#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-18_shellflow_n20_centerfloor_quick_w0p01_v1}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Hypothesis: keeping a nonzero centered-template mass after unlock changes the
# late proposal attractor, instead of just delaying the drift back to broad shells.

GPUS=(1 6 7)
VARIANTS=("floor010" "floor020" "floor035")
FLOORS=("0.10" "0.20" "0.35")

run_case() {
  local gpu="$1"
  local variant="$2"
  local floor="$3"
  local log_file="$4"

  local tag="shellflow_n20_${variant}_quick_w0p01_s42"
  local args=(
    --mode jastrow --n-elec 20 --omega 0.01
    --epochs 360 --lr 8e-5 --lr-jas 8e-6
    --n-coll 2048 --oversample 128 --micro-batch 256
    --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01
    --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0
    --adaptive-proposal --proposal-model shellflow
    --ess-floor-ratio 0.01 --ess-oversample-max 256 --ess-oversample-step 32 --ess-resample-tries 6
    --ess-update-floor 16
    --resample-weight-temp 0.60 --resample-logw-clip-q 0.995
    --top1-mass-update-ceiling 0.22 --top10-mass-update-ceiling 0.70
    --shell-templates "20-0,1-19" --shell-mix-logits-init "0.0,0.0"
    --shell-radii-init "0.0,1.35" --shell-sigmas "0.60,1.80"
    --shell-refit-steps 160 --shell-refit-lr 1e-3
    --shell-flow-layers 0 --shell-flow-hidden 128
    --shell-ring-warmup-steps 0 --shell-radius-anchor-weight 0.0
    --shell-curriculum-mode ess
    --shell-curriculum-unlock-ess 24
    --shell-curriculum-unlock-patience 2
    --shell-curriculum-inactive-logit-offset -12.0
    --shell-curriculum-centered-mass-floor "$floor"
    --gmm-refit-every 10 --gmm-refit-min-samples 2048
    --jas-input-radius-cap-aho 2.0
    --vmc-every 40 --vmc-n 12000 --n-eval 12000
    --seed 42 --tag "$tag"
    --no-pretrained
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} variant=${variant} centered_floor=${floor}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} variant=${variant} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

echo "[$(date '+%F %T')] Launching N=20 omega=0.01 centered-floor quick sweep: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  variant="${VARIANTS[$i]}"
  floor="${FLOORS[$i]}"
  log_file="${LOG_DIR}/n20_${variant}_quick_w0p01_s42.log"
  run_case "$gpu" "$variant" "$floor" "$log_file" &
  echo "[$(date '+%F %T')] Spawned pid=$! gpu=${gpu} variant=${variant} floor=${floor}" | tee -a "${OUT_DIR}/launcher.log"
done

wait
echo "[$(date '+%F %T')] All centered-floor quick runs finished" | tee -a "${OUT_DIR}/launcher.log"