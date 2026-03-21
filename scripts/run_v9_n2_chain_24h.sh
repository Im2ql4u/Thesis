#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
OUT="${ROOT}/outputs/2026-03-21_1920_campaign_v9_long24h"
LOGDIR="${OUT}/logs"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
GPU="0"
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

# Continue from best known N=2 checkpoints; for omega 0.001 cascade from 0.01 output.
run_job v9_n2w1_continue_24h \
  --mode bf --n-elec 2 --omega 1.0 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 7000 --n-coll 4096 --oversample 18 --micro-batch 1024 \
  --lr 2e-5 --lr-jas 2e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --rollback-decay 0.98 --rollback-err-pct 0.5 --rollback-jump-sigma 4.0 \
  --vmc-every 80 --vmc-n 25000 --n-eval 120000 --seed 101 \
  --resume ${ROOT}/results/curated_low_error_0p1pct_2026-03-21/v4_n2w1_precision.pt --no-pretrained

run_job v9_n2w05_continue_24h \
  --mode bf --n-elec 2 --omega 0.5 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 8000 --n-coll 4096 --oversample 20 --micro-batch 1024 \
  --lr 2e-5 --lr-jas 2e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --rollback-decay 0.98 --rollback-err-pct 0.5 --rollback-jump-sigma 4.0 \
  --vmc-every 80 --vmc-n 25000 --n-eval 120000 --seed 102 \
  --resume ${ROOT}/results/curated_low_error_0p1pct_2026-03-21/v4_n2w05_precision.pt --no-pretrained

run_job v9_n2w01_continue_24h \
  --mode bf --n-elec 2 --omega 0.1 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 9000 --n-coll 4096 --oversample 22 --micro-batch 1024 \
  --lr 1.5e-5 --lr-jas 1.5e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 4.5 \
  --vmc-every 80 --vmc-n 25000 --n-eval 120000 --seed 103 \
  --resume ${ROOT}/results/curated_low_error_0p1pct_2026-03-21/v7_n2w01_exact.pt --no-pretrained

run_job v9_n2w001_continue_24h \
  --mode bf --n-elec 2 --omega 0.01 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 12000 --n-coll 4096 --oversample 28 --micro-batch 1024 \
  --lr 8e-6 --lr-jas 8e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.005 \
  --rollback-decay 0.985 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 80 --vmc-n 25000 --n-eval 120000 --seed 104 \
  --resume ${ROOT}/results/curated_low_error_0p1pct_2026-03-21/v7_n2w001_exact.pt --no-pretrained

run_job v9_n2w0001_cascade_24h \
  --mode bf --n-elec 2 --omega 0.001 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 16000 --n-coll 4096 --oversample 36 --micro-batch 1024 \
  --lr 5e-6 --lr-jas 5e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.005 \
  --rollback-decay 0.99 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --vmc-every 100 --vmc-n 30000 --n-eval 160000 --seed 105 \
  --resume ${ROOT}/results/arch_colloc/v9_n2w001_continue_24h.pt --no-pretrained
