#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
LOGDIR="${ROOT}/outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
WALL="14400s"
GPU="0"

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

run_job v7_n2w1_exact \
  --mode bf --n-elec 2 --omega 1.0 --e-dmc 3.00000 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 3000 --n-coll 4096 --oversample 12 --micro-batch 1024 \
  --lr 1e-4 --lr-jas 1e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.02 --ess-oversample-max 24 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.96 --rollback-err-pct 2.0 --rollback-jump-sigma 3.5 \
  --vmc-every 60 --vmc-n 15000 --n-eval 60000 --seed 51 \
  --resume ${ROOT}/results/arch_colloc/v4_n2w1_bf.pt --no-pretrained

run_job v7_n2w05_exact \
  --mode bf --n-elec 2 --omega 0.5 --e-dmc 1.65977 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 3200 --n-coll 4096 --oversample 12 --micro-batch 1024 \
  --lr 1e-4 --lr-jas 1e-4 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.02 --ess-oversample-max 24 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.96 --rollback-err-pct 2.0 --rollback-jump-sigma 3.5 \
  --vmc-every 60 --vmc-n 15000 --n-eval 60000 --seed 52 \
  --resume ${ROOT}/results/arch_colloc/v4_n2w05_bf.pt --no-pretrained

run_job v7_n2w01_exact \
  --mode bf --n-elec 2 --omega 0.1 --e-dmc 0.44079 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 3400 --n-coll 4096 --oversample 14 --micro-batch 1024 \
  --lr 8e-5 --lr-jas 8e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.03 --ess-oversample-max 28 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.96 --rollback-err-pct 0.0 --rollback-jump-sigma 4.0 \
  --vmc-every 60 --vmc-n 15000 --n-eval 60000 --seed 53 \
  --resume ${ROOT}/results/arch_colloc/v4_n2w01_bf.pt --no-pretrained

run_job v7_n2w001_exact \
  --mode bf --n-elec 2 --omega 0.01 --e-dmc 0.07384 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 3600 --n-coll 4096 --oversample 16 --micro-batch 1024 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
  --lr 8e-5 --lr-jas 8e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.04 --ess-oversample-max 32 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.96 --rollback-err-pct 0.0 --rollback-jump-sigma 4.5 \
  --vmc-every 60 --vmc-n 15000 --n-eval 60000 --seed 54 \
  --resume ${ROOT}/results/arch_colloc/v4_n2w001_bf.pt --no-pretrained

run_job v7_n2w0001_exact \
  --mode bf --n-elec 2 --omega 0.001 --e-dmc 0.00730 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 4200 --n-coll 4096 --oversample 20 --micro-batch 1024 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0,8.0 \
  --langevin-steps 12 --langevin-step-size 0.002 \
  --lr 6e-5 --lr-jas 6e-5 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.06 --ess-oversample-max 40 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
  --vmc-every 60 --vmc-n 20000 --n-eval 80000 --seed 55 \
  --resume ${ROOT}/results/arch_colloc/v4_n2w0001_bf.pt --no-pretrained
