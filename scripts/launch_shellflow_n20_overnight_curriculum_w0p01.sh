#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-17_shellflow_n20_overnight_curriculum_w0p01_v1}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Hypothesis: proposal curriculum helps only if it prevents the early collapse
# toward outer-shell templates without freezing the proposal there forever.
# These overnight runs compare three unlock rules over two seeds each.

GPUS=(1 2 3 4 5 6)
VARIANTS=("epoch2" "ess2" "radius3" "epoch2" "ess2" "radius3")
SEEDS=(42 42 42 43 43 43)
SHELL_TEMPLATES=(
  "20-0,1-19"
  "20-0,1-19"
  "20-0-0,1-19-0,1-7-12"
  "20-0,1-19"
  "20-0,1-19"
  "20-0-0,1-19-0,1-7-12"
)
SHELL_MIX_LOGITS=("0.0,0.0" "0.0,0.0" "0.0,0.0,0.0" "0.0,0.0" "0.0,0.0" "0.0,0.0,0.0")
SHELL_RADII=("0.0,1.35" "0.0,1.35" "0.0,0.75,1.50" "0.0,1.35" "0.0,1.35" "0.0,0.75,1.50")
SHELL_SIGMAS=("0.60,1.80" "0.60,1.80" "0.45,1.00,1.80" "0.60,1.80" "0.60,1.80" "0.45,1.00,1.80")
CURRICULUM_MODE=("epoch" "ess" "radius" "epoch" "ess" "radius")
CURRICULUM_UNLOCK_EPOCH=("80" "0" "0" "80" "0" "0")
CURRICULUM_UNLOCK_ESS=("0" "24" "0" "0" "24" "0")
CURRICULUM_PATIENCE=("1" "2" "2" "1" "2" "2")
CURRICULUM_RADIUS_Q=("0.85" "0.85" "0.85" "0.85" "0.85" "0.85")
CURRICULUM_RADIUS_THR=("0" "0" "1.05" "0" "0" "1.05")

run_case() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local shell_templates="$4"
  local shell_mix_logits="$5"
  local shell_radii="$6"
  local shell_sigmas="$7"
  local curriculum_mode="$8"
  local unlock_epoch="$9"
  local unlock_ess="${10}"
  local unlock_patience="${11}"
  local radius_q="${12}"
  local radius_thr="${13}"
  local log_file="${14}"

  local tag="shellflow_n20_${variant}_overnight_w0p01_s${seed}"
  local args=(
    --mode jastrow --n-elec 20 --omega 0.01
    --epochs 720 --lr 8e-5 --lr-jas 8e-6
    --n-coll 2048 --oversample 128 --micro-batch 256
    --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01
    --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0
    --adaptive-proposal --proposal-model shellflow
    --ess-floor-ratio 0.01 --ess-oversample-max 256 --ess-oversample-step 32 --ess-resample-tries 6
    --ess-update-floor 16
    --resample-weight-temp 0.60 --resample-logw-clip-q 0.995
    --top1-mass-update-ceiling 0.22 --top10-mass-update-ceiling 0.70
    --shell-templates "$shell_templates" --shell-mix-logits-init "$shell_mix_logits"
    --shell-radii-init "$shell_radii" --shell-sigmas "$shell_sigmas"
    --shell-refit-steps 160 --shell-refit-lr 1e-3
    --shell-flow-layers 0 --shell-flow-hidden 128
    --shell-ring-warmup-steps 0 --shell-radius-anchor-weight 0.0
    --shell-curriculum-mode "$curriculum_mode"
    --shell-curriculum-unlock-epoch "$unlock_epoch"
    --shell-curriculum-unlock-ess "$unlock_ess"
    --shell-curriculum-unlock-patience "$unlock_patience"
    --shell-curriculum-radius-quantile "$radius_q"
    --shell-curriculum-radius-threshold-aho "$radius_thr"
    --shell-curriculum-inactive-logit-offset -12.0
    --gmm-refit-every 10 --gmm-refit-min-samples 2048
    --jas-input-radius-cap-aho 2.0
    --vmc-every 60 --vmc-n 15000 --n-eval 20000
    --seed "$seed" --tag "$tag"
    --no-pretrained
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} variant=${variant} seed=${seed} curriculum=${curriculum_mode} unlock_epoch=${unlock_epoch} unlock_ess=${unlock_ess} radius_thr=${radius_thr}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} variant=${variant} seed=${seed} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

echo "[$(date '+%F %T')] Launching N=20 omega=0.01 overnight curriculum sweep: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  variant="${VARIANTS[$i]}"
  seed="${SEEDS[$i]}"
  shell_templates="${SHELL_TEMPLATES[$i]}"
  shell_mix_logits="${SHELL_MIX_LOGITS[$i]}"
  shell_radii="${SHELL_RADII[$i]}"
  shell_sigmas="${SHELL_SIGMAS[$i]}"
  curriculum_mode="${CURRICULUM_MODE[$i]}"
  unlock_epoch="${CURRICULUM_UNLOCK_EPOCH[$i]}"
  unlock_ess="${CURRICULUM_UNLOCK_ESS[$i]}"
  unlock_patience="${CURRICULUM_PATIENCE[$i]}"
  radius_q="${CURRICULUM_RADIUS_Q[$i]}"
  radius_thr="${CURRICULUM_RADIUS_THR[$i]}"
  log_file="${LOG_DIR}/n20_${variant}_overnight_w0p01_s${seed}.log"
  run_case "$gpu" "$variant" "$seed" "$shell_templates" "$shell_mix_logits" "$shell_radii" "$shell_sigmas" "$curriculum_mode" "$unlock_epoch" "$unlock_ess" "$unlock_patience" "$radius_q" "$radius_thr" "$log_file" &
  echo "[$(date '+%F %T')] Spawned pid=$! gpu=${gpu} variant=${variant} seed=${seed}" | tee -a "${OUT_DIR}/launcher.log"
done

wait
echo "[$(date '+%F %T')] All overnight curriculum runs finished" | tee -a "${OUT_DIR}/launcher.log"