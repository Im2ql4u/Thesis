#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
OUT="${ROOT}/outputs/2026-03-21_1920_campaign_v9_long24h"
LOGDIR="${OUT}/logs"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
GPU="1"
WALL="90000s"

mkdir -p "${LOGDIR}"

run_job() {
  local tag="$1"
  shift
  local log="${LOGDIR}/${tag}.log"
  echo "# Started: $(date) GPU=${GPU} tag=${tag}" >> "${log}"
  (
    cd "${ROOT}"
    eval "${MODULE_CMD}" || true
    CUDA_VISIBLE_DEVICES="${GPU}" timeout "${WALL}" python3 src/run_weak_form.py "$@" --tag "${tag}"
  ) >> "${log}" 2>&1
  local rc=$?
  echo "# Completed: $(date) rc=${rc}" >> "${log}"
}

run_job v9_n12w1_polish_24h \
  --mode bf --n-elec 12 --omega 1.0 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 7000 --n-coll 4096 --oversample 14 --micro-batch 1024 \
  --lr 5e-5 --lr-jas 5e-5 --direct-weight 0.0 \
  --rollback-decay 0.97 --rollback-err-pct 1.0 --rollback-jump-sigma 4.0 \
  --vmc-every 100 --vmc-n 25000 --n-eval 100000 --seed 201 \
  --resume ${ROOT}/results/curated_low_error_0p1pct_2026-03-21/v7_n12w1_continue.pt --no-pretrained

run_job v9_n12w05_polish_24h \
  --mode bf --n-elec 12 --omega 0.5 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 8000 --n-coll 4096 --oversample 16 --micro-batch 1024 \
  --lr 4e-5 --lr-jas 4e-5 --direct-weight 0.0 \
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 4.5 \
  --vmc-every 100 --vmc-n 25000 --n-eval 100000 --seed 202 \
  --resume ${ROOT}/results/curated_low_error_0p1pct_2026-03-21/long_n12w05.pt --no-pretrained
