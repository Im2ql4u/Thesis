#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-17_shellflow_n20_layer1_w0p01_v1}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Hypothesis: the low-omega overlap failure is primarily Layer 1, so a wider,
# ring-only ShellFlow proposal should raise ESS above the current update gate
# without relying on transferred Jastrow weights.

GPUS=(1 2)
VARIANTS=("2shell" "3shell")
SHELL_TEMPLATES=("20-0,1-19" "20-0-0,1-19-0,1-7-12")
SHELL_MIX_LOGITS=("0.0,0.0" "0.0,0.0,0.0")
SHELL_RADII=("0.0,1.35" "0.0,0.75,1.50")
SHELL_SIGMAS=("0.60,1.80" "0.45,1.00,1.80")

run_case() {
  local gpu="$1"
  local variant="$2"
  local shell_templates="$3"
  local shell_mix_logits="$4"
  local shell_radii="$5"
  local shell_sigmas="$6"
  local log_file="$7"

  local tag="shellflow_n20_layer1_${variant}_scratch_w0p01_s42"
  local args=(
    --mode jastrow --n-elec 20 --omega 0.01
    --epochs 600 --lr 8e-5 --lr-jas 8e-6
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
    --gmm-refit-every 10 --gmm-refit-min-samples 2048
    --jas-input-radius-cap-aho 2.0
    --vmc-every 40 --vmc-n 12000 --n-eval 30000
    --seed 42 --tag "$tag"
    --no-pretrained
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} variant=${variant} proposal=ring-only oversample=128 shell_radii=${shell_radii} shell_sigmas=${shell_sigmas}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} variant=${variant} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

echo "[$(date '+%F %T')] Launching N=20 omega=0.01 Layer 1 proposal sweep: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  variant="${VARIANTS[$i]}"
  shell_templates="${SHELL_TEMPLATES[$i]}"
  shell_mix_logits="${SHELL_MIX_LOGITS[$i]}"
  shell_radii="${SHELL_RADII[$i]}"
  shell_sigmas="${SHELL_SIGMAS[$i]}"
  log_file="${LOG_DIR}/n20_${variant}_scratch_w0p01_s42.log"
  run_case "$gpu" "$variant" "$shell_templates" "$shell_mix_logits" "$shell_radii" "$shell_sigmas" "$log_file" &
  echo "[$(date '+%F %T')] Spawned pid=$! gpu=${gpu} variant=${variant}" | tee -a "${OUT_DIR}/launcher.log"
done

wait
echo "[$(date '+%F %T')] All Layer 1 runs finished" | tee -a "${OUT_DIR}/launcher.log"