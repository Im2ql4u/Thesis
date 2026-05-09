#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-19_n12_lowomega_hisamp_v2}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# High-sample N=12, omega=0.01 follow-up. This is the cheapest likely win:
# test whether the remaining ~1.1% gap is mostly sample-budget limited.

GPUS=(1 2 3 4)
VARIANTS=("cap2_hisamp" "cap3_hisamp" "uncap_hisamp" "cap3_ov24")
N_COLL=("8192" "8192" "8192" "6144")
OVERSAMPLE=("16" "16" "16" "24")
JAS_CAP=("2.0" "3.0" "0.0" "3.0")

run_case() {
  local gpu="$1"
  local variant="$2"
  local n_coll="$3"
  local oversample="$4"
  local jas_cap="$5"
  local log_file="$6"

  local tag="n12_lowomega_${variant}_s42"
  local args=(
    --mode bf --n-elec 12 --omega 0.01 --e-dmc 2.47363
    --epochs 2200 --n-coll "$n_coll" --oversample "$oversample" --micro-batch 512
    --sigma-fs "0.8,1.3,2.0,3.5,6.0"
    --lr 1e-4 --lr-jas 1e-5 --lr-min-frac 0.01
    --lr-warmup-epochs 40 --lr-warmup-init-frac 0.1
    --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02
    --ess-floor-ratio 0.10 --ess-oversample-max 32 --ess-oversample-step 2 --ess-resample-tries 2
    --rollback-decay 0.92 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0
    --jas-input-radius-cap-aho "$jas_cap"
    --init-jas results/arch_colloc/v12b_n12w01.pt
    --init-bf results/arch_colloc/v12b_n12w01.pt
    --no-pretrained
    --vmc-every 80 --vmc-n 20000 --n-eval 60000
    --save-best-window 40
    --seed 42 --tag "$tag"
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} variant=${variant} n_coll=${n_coll} oversample=${oversample} jas_cap=${jas_cap}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} variant=${variant} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

echo "[$(date '+%F %T')] Launching N=12 low-omega high-sample sweep: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  variant="${VARIANTS[$i]}"
  n_coll="${N_COLL[$i]}"
  oversample="${OVERSAMPLE[$i]}"
  jas_cap="${JAS_CAP[$i]}"
  log_file="${LOG_DIR}/${variant}.log"
  run_case "$gpu" "$variant" "$n_coll" "$oversample" "$jas_cap" "$log_file" &
  echo "[$(date '+%F %T')] Spawned pid=$! gpu=${gpu} variant=${variant}" | tee -a "${OUT_DIR}/launcher.log"
done

wait
echo "[$(date '+%F %T')] All N=12 low-omega high-sample runs finished" | tee -a "${OUT_DIR}/launcher.log"
