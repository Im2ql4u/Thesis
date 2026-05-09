#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-19_n20_tinybf_reentry_v1}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Reintroduce only a tiny backflow head on top of the best current N=20
# Jastrow checkpoints. This is the quickest test of whether the current N=20
# wall is still mostly ansatz-limited once the Jastrow baseline is decent.

GPUS=(1 2 3 4 5 6)
VARIANTS=("w1_h48" "w1_h32" "w05_h48" "w05_h32" "w01_h48" "w01_h32")
OMEGAS=("1.0" "1.0" "0.5" "0.5" "0.1" "0.1")
INIT_JAS=(
  "results/arch_colloc/n20ovr2_w1_best.pt"
  "results/arch_colloc/n20ovr2_w1_best.pt"
  "results/arch_colloc/n20ovr2_w05_best.pt"
  "results/arch_colloc/n20ovr2_w05_best.pt"
  "results/arch_colloc/camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt"
  "results/arch_colloc/camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt"
)
BF_HIDDEN=("48" "32" "48" "32" "48" "32")
BF_MSG_HIDDEN=("48" "32" "48" "32" "48" "32")
EPOCHS=("2400" "2400" "2800" "2800" "3600" "3600")
N_COLL=("512" "512" "512" "512" "384" "384")
OVERSAMPLE=("10" "10" "12" "12" "14" "14")
MICRO_BATCH=("32" "32" "32" "32" "24" "24")
LR=("8e-5" "8e-5" "6e-5" "6e-5" "4e-5" "4e-5")
LR_JAS=("8e-6" "8e-6" "6e-6" "6e-6" "4e-6" "4e-6")
QTRIM=("0.01" "0.01" "0.01" "0.01" "0.02" "0.02")
ROLLBACK_JUMP=("5.0" "5.0" "5.5" "5.5" "6.5" "6.5")

run_case() {
  local gpu="$1"
  local variant="$2"
  local omega="$3"
  local init_jas="$4"
  local bf_hidden="$5"
  local bf_msg_hidden="$6"
  local epochs="$7"
  local n_coll="$8"
  local oversample="$9"
  local micro_batch="${10}"
  local lr="${11}"
  local lr_jas="${12}"
  local qtrim="${13}"
  local rollback_jump="${14}"
  local log_file="${15}"

  local tag="n20_tinybf_${variant}_s42"
  local args=(
    --mode bf --n-elec 20 --omega "$omega"
    --epochs "$epochs" --n-coll "$n_coll" --oversample "$oversample" --micro-batch "$micro_batch"
    --lr "$lr" --lr-jas "$lr_jas" --lr-min-frac 0.01
    --lr-warmup-epochs 30 --lr-warmup-init-frac 0.1
    --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --reward-qtrim "$qtrim"
    --ess-floor-ratio 0.01 --ess-oversample-max 24 --ess-oversample-step 2 --ess-resample-tries 2
    --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma "$rollback_jump"
    --bf-hidden "$bf_hidden" --bf-msg-hidden "$bf_msg_hidden" --bf-layers 2
    --init-jas "$init_jas" --no-pretrained
    --vmc-every 400 --vmc-n 12000 --n-eval 20000
    --save-best-window 40
    --seed 42 --tag "$tag"
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} variant=${variant} omega=${omega} bf_hidden=${bf_hidden}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} variant=${variant} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

echo "[$(date '+%F %T')] Launching N=20 tiny-BF reentry sweep: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  variant="${VARIANTS[$i]}"
  omega="${OMEGAS[$i]}"
  init_jas="${INIT_JAS[$i]}"
  bf_hidden="${BF_HIDDEN[$i]}"
  bf_msg_hidden="${BF_MSG_HIDDEN[$i]}"
  epochs="${EPOCHS[$i]}"
  n_coll="${N_COLL[$i]}"
  oversample="${OVERSAMPLE[$i]}"
  micro_batch="${MICRO_BATCH[$i]}"
  lr="${LR[$i]}"
  lr_jas="${LR_JAS[$i]}"
  qtrim="${QTRIM[$i]}"
  rollback_jump="${ROLLBACK_JUMP[$i]}"
  log_file="${LOG_DIR}/${variant}.log"
  run_case "$gpu" "$variant" "$omega" "$init_jas" "$bf_hidden" "$bf_msg_hidden" "$epochs" "$n_coll" "$oversample" "$micro_batch" "$lr" "$lr_jas" "$qtrim" "$rollback_jump" "$log_file" &
  echo "[$(date '+%F %T')] Spawned pid=$! gpu=${gpu} variant=${variant}" | tee -a "${OUT_DIR}/launcher.log"
done

wait
echo "[$(date '+%F %T')] All tiny-BF reentry runs finished" | tee -a "${OUT_DIR}/launcher.log"
