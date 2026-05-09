#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-14_shellflow_n20_lowomega}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

SEEDS=(42 137 314)
OMEGAS=(0.01 0.001)
EPOCHS=(2500 3000)
GPUS=(2 3)

SHELL_VARIANTS=("2shell" "3shell")
SHELL_TEMPLATES=("20-0,6-14" "20-0-0,2-6-12")
SHELL_MIX_LOGITS=("0.0,0.0" "0.0,0.0")
SHELL_SIGMAS=("0.30,0.90" "0.25,0.55,1.05")

run_case() {
  local gpu="$1"
  local omega="$2"
  local epochs="$3"
  local seed="$4"
  local shell_variant="$5"
  local shell_templates="$6"
  local shell_mix_logits="$7"
  local shell_sigmas="$8"
  local log_file="$9"

  local omega_tag
  omega_tag="$(echo "$omega" | sed 's/\./p/g')"
  local tag="shellflow_n20w${omega_tag}_${shell_variant}_s${seed}"

  local args=(
    --mode bf --n-elec 20 --omega "$omega"
    --epochs "$epochs" --lr 5e-4 --lr-jas 5e-5
    --n-coll 4096 --oversample 16 --micro-batch 512
    --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0
    --reward-normalize
    --rollback-jump-sigma 3 --rollback-decay 0.95
    --adaptive-proposal --proposal-model shellflow
    --ess-update-floor 16
    --shell-templates "$shell_templates" --shell-mix-logits-init "$shell_mix_logits"
    --shell-sigmas "$shell_sigmas" --shell-refit-steps 120 --shell-refit-lr 1e-3
    --shell-flow-layers 2 --shell-flow-hidden 128
    --shell-ring-warmup-steps 120 --shell-radius-anchor-weight 0.25
    --gmm-refit-every 15 --gmm-refit-min-samples 500
    --vmc-every 25 --vmc-n 50000 --n-eval 50000
    --seed "$seed" --tag "$tag"
    --no-pretrained
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} omega=${omega} seed=${seed} shell=${shell_variant} epochs=${epochs}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} omega=${omega} seed=${seed} shell=${shell_variant} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

worker() {
  local gpu="$1"
  local omega="$2"
  local epochs="$3"
  local worker_log="${LOG_DIR}/worker_gpu${gpu}_w$(echo "$omega" | sed 's/\./p/g').log"

  echo "[$(date '+%F %T')] Worker start gpu=${gpu} omega=${omega}" | tee -a "$worker_log"
  for s in "${!SHELL_VARIANTS[@]}"; do
    shell_variant="${SHELL_VARIANTS[$s]}"
    shell_templates="${SHELL_TEMPLATES[$s]}"
    shell_mix_logits="${SHELL_MIX_LOGITS[$s]}"
    shell_sigmas="${SHELL_SIGMAS[$s]}"
    for seed in "${SEEDS[@]}"; do
      log_file="${LOG_DIR}/n20w$(echo "$omega" | sed 's/\./p/g')_${shell_variant}_s${seed}.log"
      echo "[$(date '+%F %T')] Launching omega=${omega} seed=${seed} shell=${shell_variant} gpu=${gpu}" | tee -a "${OUT_DIR}/launcher.log"
      run_case "$gpu" "$omega" "$epochs" "$seed" "$shell_variant" "$shell_templates" "$shell_mix_logits" "$shell_sigmas" "$log_file"
    done
  done
  echo "[$(date '+%F %T')] Worker done gpu=${gpu} omega=${omega}" | tee -a "$worker_log"
}

echo "[$(date '+%F %T')] Launching ShellFlow N=20 low-omega campaign: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!OMEGAS[@]}"; do
  omega="${OMEGAS[$i]}"
  epochs="${EPOCHS[$i]}"
  gpu="${GPUS[$i]}"
  worker "$gpu" "$omega" "$epochs" &
  echo "[$(date '+%F %T')] Spawned worker pid=$! omega=${omega} gpu=${gpu}" | tee -a "${OUT_DIR}/launcher.log"
done
wait
echo "[$(date '+%F %T')] All workers finished" | tee -a "${OUT_DIR}/launcher.log"
