#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="2026-04-11_phase1_corrected_matrix"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

SEEDS=(42 137 314)
OMEGAS=(1.0 0.5 0.1 0.01 0.001)
GPUS=(0 1 2 3 4)

run_case() {
  local gpu="$1"
  local mode_tag="$2"
  local omega="$3"
  local seed="$4"
  local log_file="$5"
  shift 5

  echo "[$(date '+%F %T')] START gpu=${gpu} tag=${mode_tag} omega=${omega} seed=${seed}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "$@" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} tag=${mode_tag} omega=${omega} seed=${seed} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

worker() {
  local gpu="$1"
  local omega="$2"
  local omega_tag
  omega_tag="$(echo "$omega" | sed 's/\./p/g')"
  local worker_log="${LOG_DIR}/worker_gpu${gpu}_w${omega_tag}.log"

  echo "[$(date '+%F %T')] Worker start gpu=${gpu} omega=${omega}" | tee -a "$worker_log"

  for seed in "${SEEDS[@]}"; do
    base_tag="baseline_p1fix_n6w${omega_tag}_s${seed}"
    robust_tag="robust_p1fix_n6w${omega_tag}_s${seed}"

    run_case "$gpu" "$base_tag" "$omega" "$seed" "$worker_log" \
      --mode bf --n-elec 6 --omega "$omega" \
      --epochs 1200 --lr 5e-4 --lr-jas 5e-5 \
      --n-coll 4096 --oversample 8 --micro-batch 512 \
      --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0 \
      --vmc-every 50 --vmc-n 20000 --n-eval 30000 \
      --seed "$seed" --tag "$base_tag"
    if [[ $? -ne 0 ]]; then
      echo "[$(date '+%F %T')] WARN baseline failed tag=${base_tag}" | tee -a "$worker_log"
    fi

    run_case "$gpu" "$robust_tag" "$omega" "$seed" "$worker_log" \
      --mode bf --n-elec 6 --omega "$omega" \
      --epochs 1200 --lr 5e-4 --lr-jas 5e-5 \
      --n-coll 4096 --oversample 16 --micro-batch 512 \
      --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0 \
      --reward-normalize \
      --rollback-jump-sigma 3 --rollback-decay 0.95 \
      --adaptive-proposal --gmm-components 8 --gmm-refit-every 30 \
      --vmc-every 50 --vmc-n 20000 --n-eval 30000 \
      --seed "$seed" --tag "$robust_tag"
    if [[ $? -ne 0 ]]; then
      echo "[$(date '+%F %T')] WARN robust failed tag=${robust_tag}" | tee -a "$worker_log"
    fi
  done

  echo "[$(date '+%F %T')] Worker done gpu=${gpu} omega=${omega}" | tee -a "$worker_log"
}

echo "[$(date '+%F %T')] Launching Phase-1 corrected matrix: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!OMEGAS[@]}"; do
  worker "${GPUS[$i]}" "${OMEGAS[$i]}" &
  echo "[$(date '+%F %T')] Spawned worker pid=$! gpu=${GPUS[$i]} omega=${OMEGAS[$i]}" | tee -a "${OUT_DIR}/launcher.log"
done

wait

echo "[$(date '+%F %T')] All workers finished" | tee -a "${OUT_DIR}/launcher.log"
