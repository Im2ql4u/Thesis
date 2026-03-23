#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
OUT="${ROOT}/outputs/2026-03-21_1920_campaign_v9_long24h"
LOGDIR="${OUT}/logs"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
GPU="0"
WALL="130000s"

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

# Stage 1: polish omega=0.01 from the best known 0.01 checkpoint.
run_job v10_n2w001_polish_36h \
  --mode bf --n-elec 2 --omega 0.01 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 20000 --n-coll 4096 --oversample 36 --micro-batch 1024 \
  --lr 5e-6 --lr-jas 5e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.002 \
  --rollback-decay 0.99 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --vmc-every 120 --vmc-n 40000 --n-eval 200000 --seed 1101 \
  --resume ${ROOT}/results/curated_low_error_0p1pct_2026-03-21/v7_n2w001_exact.pt --no-pretrained

# Stage 2: transfer to omega=0.005.
run_job v10_n2w0005_transfer_36h \
  --mode bf --n-elec 2 --omega 0.005 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 22000 --n-coll 4096 --oversample 44 --micro-batch 1024 \
  --lr 4e-6 --lr-jas 4e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.002 \
  --rollback-decay 0.992 --rollback-err-pct 0.0 --rollback-jump-sigma 6.5 \
  --vmc-every 140 --vmc-n 40000 --n-eval 200000 --seed 1102 \
  --resume ${ROOT}/results/arch_colloc/v10_n2w001_polish_36h.pt --no-pretrained

# Stage 3: transfer to omega=0.002.
run_job v10_n2w0002_transfer_36h \
  --mode bf --n-elec 2 --omega 0.002 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 26000 --n-coll 4096 --oversample 52 --micro-batch 1024 \
  --lr 3e-6 --lr-jas 3e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.0015 \
  --rollback-decay 0.994 --rollback-err-pct 0.0 --rollback-jump-sigma 7.0 \
  --vmc-every 160 --vmc-n 50000 --n-eval 240000 --seed 1103 \
  --resume ${ROOT}/results/arch_colloc/v10_n2w0005_transfer_36h.pt --no-pretrained

# Stage 4: final transfer to omega=0.001.
run_job v10_n2w0001_transfer_36h \
  --mode bf --n-elec 2 --omega 0.001 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 32000 --n-coll 4096 --oversample 60 --micro-batch 1024 \
  --lr 2e-6 --lr-jas 2e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.001 \
  --rollback-decay 0.996 --rollback-err-pct 0.0 --rollback-jump-sigma 7.5 \
  --vmc-every 180 --vmc-n 60000 --n-eval 280000 --seed 1104 \
  --resume ${ROOT}/results/arch_colloc/v10_n2w0002_transfer_36h.pt --no-pretrained
