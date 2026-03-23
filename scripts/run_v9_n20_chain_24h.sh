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

# N=20 fix: realistic 24h throughput + lower memory pressure.
run_job v9_n20w1_fix_24h \
  --mode bf --n-elec 20 --omega 1.0 \
  --epochs 3000 --n-coll 384 --oversample 10 --micro-batch 24 \
  --lr 1.0e-4 --lr-jas 1.0e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 600 --vmc-n 2000 --vmc-select-n 512 --n-eval 4000 --seed 401 \
  --resume ${ROOT}/results/arch_colloc/smoke_n20_o1p0.pt --no-pretrained

run_job v9_n20w05_fix_24h \
  --mode bf --n-elec 20 --omega 0.5 \
  --epochs 3500 --n-coll 384 --oversample 12 --micro-batch 24 \
  --lr 8e-5 --lr-jas 8e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.5 \
  --vmc-every 600 --vmc-n 2000 --vmc-select-n 512 --n-eval 4000 --seed 402 \
  --resume ${ROOT}/results/arch_colloc/smoke_n20_o0p5.pt --no-pretrained

run_job v9_n20w01_fix_24h \
  --mode bf --n-elec 20 --omega 0.1 \
  --epochs 4500 --n-coll 384 --oversample 14 --micro-batch 24 \
  --lr 5e-5 --lr-jas 5e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.02 \
  --rollback-decay 0.985 --rollback-err-pct 0.0 --rollback-jump-sigma 6.5 \
  --vmc-every 600 --vmc-n 2000 --vmc-select-n 512 --n-eval 4000 --seed 403 \
  --resume ${ROOT}/results/arch_colloc/camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt --no-pretrained
