#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-19_n20_bf_curriculum_retry_v1}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

GPU="${GPU:-7}"
SEED="${SEED:-42}"
WALL="${WALL:-90000s}"
LOG_FILE="${LOG_DIR}/gpu${GPU}_curriculum.log"
MPL_DIR="/tmp/matplotlib-n20-bfcur-gpu${GPU}"
mkdir -p "$MPL_DIR"

run_stage() {
  local tag="$1"
  local resume_ckpt="$2"
  shift 2

  echo "[$(date '+%F %T')] START gpu=${GPU} tag=${tag} resume=$(basename "$resume_ckpt")" | tee -a "$LOG_FILE"
  MPLCONFIGDIR="$MPL_DIR" CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 timeout "$WALL" \
    python3 src/run_weak_form.py "$@" --resume "$resume_ckpt" --no-pretrained --seed "$SEED" --tag "$tag" \
    2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${GPU} tag=${tag} rc=${rc}" | tee -a "$LOG_FILE"
  if [[ $rc -ne 0 ]]; then
    return "$rc"
  fi
}

stage1_resume="results/arch_colloc/n20_tinybf_w1_h48_s42_best.pt"
stage1_tag="n20_bfcur_w1_recover_s${SEED}"
run_stage "$stage1_tag" "$stage1_resume" \
  --mode bf --n-elec 20 --omega 1.0 \
  --epochs 1200 --n-coll 512 --oversample 12 --micro-batch 32 \
  --lr 2e-5 --lr-jas 2e-6 --lr-min-frac 0.02 \
  --lr-warmup-epochs 40 --lr-warmup-init-frac 0.05 \
  --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.01 --ess-oversample-max 28 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.985 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --bf-cusp-reg 1e-3 --bf-hard-cusp-gate --bf-cusp-gate-radius-aho 0.35 --bf-cusp-gate-power 2.0 \
  --vmc-every 300 --vmc-n 12000 --n-eval 20000 \
  --save-best-window 50

stage1_best="results/arch_colloc/${stage1_tag}_best.pt"
if [[ ! -f "$stage1_best" ]]; then
  echo "[$(date '+%F %T')] ERROR missing stage1 best checkpoint: ${stage1_best}" | tee -a "$LOG_FILE"
  exit 1
fi

stage2_tag="n20_bfcur_w05_fromw1_s${SEED}"
run_stage "$stage2_tag" "$stage1_best" \
  --mode bf --n-elec 20 --omega 0.5 \
  --epochs 1800 --n-coll 512 --oversample 14 --micro-batch 32 \
  --lr 1.5e-5 --lr-jas 1.5e-6 --lr-min-frac 0.02 \
  --lr-warmup-epochs 40 --lr-warmup-init-frac 0.05 \
  --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.01 \
  --ess-floor-ratio 0.01 --ess-oversample-max 28 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.985 --rollback-err-pct 0.0 --rollback-jump-sigma 6.5 \
  --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --bf-cusp-reg 2e-3 --bf-hard-cusp-gate --bf-cusp-gate-radius-aho 0.35 --bf-cusp-gate-power 2.0 \
  --vmc-every 300 --vmc-n 12000 --n-eval 20000 \
  --save-best-window 50

stage2_best="results/arch_colloc/${stage2_tag}_best.pt"
if [[ ! -f "$stage2_best" ]]; then
  echo "[$(date '+%F %T')] ERROR missing stage2 best checkpoint: ${stage2_best}" | tee -a "$LOG_FILE"
  exit 1
fi

stage3_tag="n20_bfcur_w01_fromw05_s${SEED}"
run_stage "$stage3_tag" "$stage2_best" \
  --mode bf --n-elec 20 --omega 0.1 \
  --epochs 2400 --n-coll 384 --oversample 16 --micro-batch 24 \
  --lr 8e-6 --lr-jas 8e-7 --lr-min-frac 0.02 \
  --lr-warmup-epochs 50 --lr-warmup-init-frac 0.05 \
  --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.02 \
  --ess-floor-ratio 0.01 --ess-oversample-max 32 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.99 --rollback-err-pct 0.0 --rollback-jump-sigma 8.0 \
  --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --bf-cusp-reg 5e-3 --bf-hard-cusp-gate --bf-cusp-gate-radius-aho 0.40 --bf-cusp-gate-power 2.0 \
  --vmc-every 400 --vmc-n 12000 --n-eval 20000 \
  --save-best-window 50

echo "[$(date '+%F %T')] Completed N=20 BF curriculum retry on gpu=${GPU}" | tee -a "$LOG_FILE"
