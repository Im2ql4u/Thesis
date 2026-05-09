#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-13_phase6_shellflow_12h_multitest}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

SEEDS=(42 137 314)
OMEGAS=(0.01 0.001)
EPOCHS=(1500 2000)
GPUS=(0 1)

# 5-variant sweep: ~5x the prior 2.5h campaign => roughly 12h total.
VAR_NAMES=(
  "v1_ctrl_a075_w60"
  "v2_a040_w100"
  "v3_a025_w120"
  "v4_a040_w100_mgate22"
  "v5_a025_w120_mgate18"
)
VAR_ARGS=(
  "--shell-ring-warmup-steps 60 --shell-radius-anchor-weight 0.75"
  "--shell-ring-warmup-steps 100 --shell-radius-anchor-weight 0.40"
  "--shell-ring-warmup-steps 120 --shell-radius-anchor-weight 0.25"
  "--shell-ring-warmup-steps 100 --shell-radius-anchor-weight 0.40 --top1-mass-update-ceiling 0.22 --top10-mass-update-ceiling 0.70"
  "--shell-ring-warmup-steps 120 --shell-radius-anchor-weight 0.25 --top1-mass-update-ceiling 0.18 --top10-mass-update-ceiling 0.60"
)

run_case() {
  local gpu="$1"
  local variant="$2"
  local variant_args="$3"
  local omega="$4"
  local epochs="$5"
  local seed="$6"
  local resume_ckpt="$7"
  local log_file="$8"

  local omega_tag
  omega_tag="$(echo "$omega" | sed 's/\./p/g')"
  local tag="shellflow_p6_${variant}_n6w${omega_tag}_s${seed}"

  local args=(
    --mode bf --n-elec 6 --omega "$omega"
    --epochs "$epochs" --lr 5e-4 --lr-jas 5e-5
    --n-coll 4096 --oversample 16 --micro-batch 512
    --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0
    --reward-normalize
    --rollback-jump-sigma 3 --rollback-decay 0.95
    --adaptive-proposal --proposal-model shellflow
    --ess-update-floor 16
    --shell-templates 6-0,1-5 --shell-mix-logits-init 0.0,0.0
    --shell-sigmas 0.30,0.80 --shell-refit-steps 120 --shell-refit-lr 1e-3
    --shell-flow-layers 2 --shell-flow-hidden 128
    --gmm-refit-every 15 --gmm-refit-min-samples 500
    --vmc-every 25 --vmc-n 50000 --n-eval 50000
    --seed "$seed" --tag "$tag"
    --resume "$resume_ckpt" --no-pretrained
  )

  if [[ -n "$variant_args" ]]; then
    local extra=()
    read -r -a extra <<< "$variant_args"
    args+=("${extra[@]}")
  fi

  echo "[$(date '+%F %T')] START gpu=${gpu} variant=${variant} omega=${omega} seed=${seed} epochs=${epochs}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} variant=${variant} omega=${omega} seed=${seed} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

worker_variant() {
  local gpu="$1"
  local omega="$2"
  local epochs="$3"
  local variant="$4"
  local variant_args="$5"

  local omega_tag
  omega_tag="$(echo "$omega" | sed 's/\./p/g')"
  local worker_log="${LOG_DIR}/${variant}_gpu${gpu}_w${omega_tag}.log"

  echo "[$(date '+%F %T')] Worker start variant=${variant} gpu=${gpu} omega=${omega}" | tee -a "$worker_log"

  for seed in "${SEEDS[@]}"; do
    local resume_ckpt="${ROOT}/results/arch_colloc/robust_p3cascade_n6w${omega_tag}_s${seed}.pt"
    if [[ ! -f "$resume_ckpt" ]]; then
      echo "[$(date '+%F %T')] WARN missing resume checkpoint=${resume_ckpt}; skip seed=${seed}" | tee -a "$worker_log"
      continue
    fi
    run_case "$gpu" "$variant" "$variant_args" "$omega" "$epochs" "$seed" "$resume_ckpt" "$worker_log"
    if [[ $? -ne 0 ]]; then
      echo "[$(date '+%F %T')] WARN run failed variant=${variant} omega=${omega} seed=${seed}" | tee -a "$worker_log"
    fi
  done

  echo "[$(date '+%F %T')] Worker done variant=${variant} gpu=${gpu} omega=${omega}" | tee -a "$worker_log"
}

echo "[$(date '+%F %T')] Launching ~12h ShellFlow multi-test campaign: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"

for i in "${!VAR_NAMES[@]}"; do
  variant="${VAR_NAMES[$i]}"
  variant_args="${VAR_ARGS[$i]}"
  echo "[$(date '+%F %T')] Variant start: ${variant} args='${variant_args}'" | tee -a "${OUT_DIR}/launcher.log"

  for j in "${!OMEGAS[@]}"; do
    worker_variant "${GPUS[$j]}" "${OMEGAS[$j]}" "${EPOCHS[$j]}" "$variant" "$variant_args" &
    echo "[$(date '+%F %T')] Spawned pid=$! variant=${variant} gpu=${GPUS[$j]} omega=${OMEGAS[$j]}" | tee -a "${OUT_DIR}/launcher.log"
  done

  wait
  echo "[$(date '+%F %T')] Variant done: ${variant}" | tee -a "${OUT_DIR}/launcher.log"
done

echo "[$(date '+%F %T')] All variants finished" | tee -a "${OUT_DIR}/launcher.log"
