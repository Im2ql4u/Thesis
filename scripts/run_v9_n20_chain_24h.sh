#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
OUT="${ROOT}/outputs/2026-03-21_1920_campaign_v9_long24h"
LOGDIR="${OUT}/logs"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
GPU="3"
WALL="95000s"

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

# N=20 fix: lower memory footprint + disable err-threshold rollback stalls.
run_job v9_n20w1_fix_24h \
  --mode bf --n-elec 20 --omega 1.0 \
  --epochs 12000 --n-coll 768 --oversample 10 --micro-batch 64 \
  --lr 2e-4 --lr-jas 2e-4 --direct-weight 0.0 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 120 --vmc-n 20000 --n-eval 60000 --seed 401 \
  --resume ${ROOT}/results/arch_colloc/smoke_n20_o1p0.pt --no-pretrained

run_job v9_n20w05_fix_24h \
  --mode bf --n-elec 20 --omega 0.5 \
  --epochs 14000 --n-coll 768 --oversample 12 --micro-batch 64 \
  --lr 1.5e-4 --lr-jas 1.5e-4 --direct-weight 0.0 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.5 \
  --vmc-every 120 --vmc-n 20000 --n-eval 60000 --seed 402 \
  --resume ${ROOT}/results/arch_colloc/smoke_n20_o0p5.pt --no-pretrained

run_job v9_n20w01_fix_24h \
  --mode bf --n-elec 20 --omega 0.1 \
  --epochs 16000 --n-coll 768 --oversample 14 --micro-batch 64 \
  --lr 1e-4 --lr-jas 1e-4 --direct-weight 0.0 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --vmc-every 120 --vmc-n 20000 --n-eval 60000 --seed 403 \
  --resume ${ROOT}/results/arch_colloc/smoke_n20_o0p1.pt --no-pretrained
