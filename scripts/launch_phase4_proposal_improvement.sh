#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="2026-04-12_phase4_proposal_improvement"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

SEEDS=(42 137 314)
OMEGAS=(0.1 0.01 0.001)
EPOCHS=(1200 1500 2000)
GPUS=(0 1 2)

run_case() {
  local gpu="$1"
  local tag="$2"
  local omega="$3"
  local epochs="$4"
  local seed="$5"
  local resume_ckpt="$6"
  local log_file="$7"

  local args=(
    --mode bf --n-elec 6 --omega "$omega"
    --epochs "$epochs" --lr 5e-4 --lr-jas 5e-5
    --n-coll 4096 --oversample 16 --micro-batch 512
    --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0
    --reward-normalize
    --rollback-jump-sigma 3 --rollback-decay 0.95
    --adaptive-proposal --gmm-components 16 --gmm-refit-every 15
    --sigma-fs 0.6,1.0,1.5,2.5
    --vmc-every 25 --vmc-n 50000 --n-eval 50000
    --seed "$seed" --tag "$tag"
    --resume "$resume_ckpt" --no-pretrained
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} tag=${tag} omega=${omega} seed=${seed} epochs=${epochs} resume=${resume_ckpt}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} tag=${tag} omega=${omega} seed=${seed} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

worker() {
  local gpu="$1"
  local omega="$2"
  local epochs="$3"
  local omega_tag
  omega_tag="$(echo "$omega" | sed 's/\./p/g')"
  local worker_log="${LOG_DIR}/worker_gpu${gpu}_w${omega_tag}.log"

  echo "[$(date '+%F %T')] Worker start gpu=${gpu} omega=${omega}" | tee -a "$worker_log"

  for seed in "${SEEDS[@]}"; do
    local tag="robust_p4prop_n6w${omega_tag}_s${seed}"
    local resume_ckpt="${ROOT}/results/arch_colloc/robust_p3cascade_n6w${omega_tag}_s${seed}.pt"

    if [[ ! -f "$resume_ckpt" ]]; then
      echo "[$(date '+%F %T')] WARN missing resume checkpoint=${resume_ckpt}; skip tag=${tag}" | tee -a "$worker_log"
      continue
    fi

    run_case "$gpu" "$tag" "$omega" "$epochs" "$seed" "$resume_ckpt" "$worker_log"
    if [[ $? -ne 0 ]]; then
      echo "[$(date '+%F %T')] WARN run failed tag=${tag}" | tee -a "$worker_log"
    fi
  done

  echo "[$(date '+%F %T')] Worker done gpu=${gpu} omega=${omega}" | tee -a "$worker_log"
}

echo "[$(date '+%F %T')] Launching Phase-4 proposal improvement: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!OMEGAS[@]}"; do
  worker "${GPUS[$i]}" "${OMEGAS[$i]}" "${EPOCHS[$i]}" &
  echo "[$(date '+%F %T')] Spawned worker pid=$! gpu=${GPUS[$i]} omega=${OMEGAS[$i]}" | tee -a "${OUT_DIR}/launcher.log"
done

wait

echo "[$(date '+%F %T')] All workers finished" | tee -a "${OUT_DIR}/launcher.log"