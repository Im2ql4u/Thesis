#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
LOGDIR="${ROOT}/outputs/2026-03-21_0841_campaign_v6_n20bf_n2_fewhours/logs"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
WALL="2400s"
GPU="3"

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

run_job v6_n2w1_bf \
  --mode bf --n-elec 2 --omega 1.0 --e-dmc 3.00000 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 1200 --n-coll 4096 --oversample 12 --micro-batch 1024 \
  --lr 2e-4 --lr-jas 2e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.02 --ess-oversample-max 24 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.95 --rollback-err-pct 0.0 --rollback-jump-sigma 4.0 \
  --vmc-every 60 --vmc-n 12000 --n-eval 30000 --seed 41 \
  --no-pretrained

run_job v6_n2w05_bf \
  --mode bf --n-elec 2 --omega 0.5 --e-dmc 1.65977 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 1200 --n-coll 4096 --oversample 12 --micro-batch 1024 \
  --lr 2e-4 --lr-jas 2e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.02 --ess-oversample-max 24 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.95 --rollback-err-pct 0.0 --rollback-jump-sigma 4.0 \
  --vmc-every 60 --vmc-n 12000 --n-eval 30000 --seed 42 \
  --no-pretrained

run_job v6_n2w01_bf \
  --mode bf --n-elec 2 --omega 0.1 --e-dmc 0.44079 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 1400 --n-coll 4096 --oversample 14 --micro-batch 1024 \
  --lr 1.5e-4 --lr-jas 1.5e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.03 --ess-oversample-max 28 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.95 --rollback-err-pct 0.0 --rollback-jump-sigma 4.5 \
  --vmc-every 60 --vmc-n 12000 --n-eval 30000 --seed 43 \
  --no-pretrained

run_job v6_n2w001_bf \
  --mode bf --n-elec 2 --omega 0.01 --e-dmc 0.07384 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 1600 --n-coll 4096 --oversample 16 --micro-batch 1024 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
  --lr 1.2e-4 --lr-jas 1.2e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.04 --ess-oversample-max 32 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.95 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 60 --vmc-n 12000 --n-eval 30000 --seed 44 \
  --no-pretrained

run_job v6_n2w0001_bf \
  --mode bf --n-elec 2 --omega 0.001 --e-dmc 0.00730 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 1800 --n-coll 4096 --oversample 20 --micro-batch 1024 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0,8.0 \
  --langevin-steps 10 --langevin-step-size 0.002 \
  --lr 1e-4 --lr-jas 1e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.05 --ess-oversample-max 36 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.95 --rollback-err-pct 0.0 --rollback-jump-sigma 5.5 \
  --vmc-every 60 --vmc-n 15000 --n-eval 40000 --seed 45 \
  --no-pretrained
