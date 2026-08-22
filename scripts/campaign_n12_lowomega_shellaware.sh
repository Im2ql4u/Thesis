#!/usr/bin/env bash
# N=12 low-ω (ω=0.01) improvement run: ShellAware Jastrow + weight tempering
#
# Use when existing hisamp checkpoints are worse than ~5% error.
# Cascades from best ω=0.1 BF checkpoint → ω=0.01 with new architecture.
#
# Key differences from previous N=12 ω=0.01 attempts (ShellFlow, uncap hisamp):
#   - CTNNShellAwareJastrow instead of vcycle (captures ring physics)
#   - resample-weight-temp 0.75 (was 1.0 in all previous runs)
#   - Warm-start from ω=0.1 checkpoint (not from scratch)
#   - jas-hidden 64 for more capacity
#
# Usage:
#   bash scripts/campaign_n12_lowomega_shellaware.sh
#   SEED=137 GPU=0 bash scripts/campaign_n12_lowomega_shellaware.sh
#
# Override with RESUME_CKPT env var to start from a specific checkpoint.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

SEED="${SEED:-42}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-2500}"

# Best ω=0.1 checkpoint — used as warm-start
W01_CKPT="${RESUME_CKPT:-results/arch_colloc/n12_adam_w01_best.pt}"

RUN_ID="2026-05-14_n12_shellaware_lowomega_s${SEED}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
CKPT_DIR="results/arch_colloc"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${OUT_DIR}/launcher.log"; }

log "=== N=12 ShellAware 2-stage cascade: ω=0.1 → 0.01, seed=${SEED}, gpu=${GPU} ==="
log "(Fresh shellaware arch — incompatible with existing vcycle checkpoints, so bootstrapping at ω=0.1 first)"

# ---- Stage 1: bootstrap at ω=0.1 (500 epochs) to get non-singular Slater det ----
TAG_W01="n12_shellaware_w01_boot_s${SEED}"
LOGFILE_W01="${LOG_DIR}/${TAG_W01}.log"
CKPT_W01="${CKPT_DIR}/${TAG_W01}_best.pt"

log "Stage 1: ω=0.1 bootstrap (500 epochs) → tag=${TAG_W01}"

CUDA_MANUAL_DEVICE="$GPU" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py \
    --mode bf --n-elec 12 --omega 0.1 \
    --no-pretrained \
    --jas-arch shellaware --jas-hidden 64 --jas-shells 3 \
    --bf-hidden 64 --bf-layers 2 \
    --epochs 600 \
    --lr 5e-5 --lr-jas 6e-6 \
    --lr-warmup-epochs 100 --lr-warmup-init-frac 0.05 \
    --n-coll 4096 --oversample 12 --micro-batch 512 \
    --loss-type reinforce --direct-weight 0.0 \
    --clip-el 4.0 --reward-qtrim 0.02 \
    --resample-weight-temp 0.80 \
    --resample-logw-clip-q 0.02 \
    --ess-floor-ratio 0.03 --ess-oversample-max 32 --ess-oversample-step 2 \
    --rollback-decay 0.95 --rollback-jump-sigma 4.0 \
    --vmc-every 50 --vmc-n 30000 \
    --n-eval 50000 \
    --seed "$SEED" \
    --tag "$TAG_W01" \
    2>&1 | tee -a "$LOGFILE_W01"

RC1=${PIPESTATUS[0]}
log "Stage 1 complete rc=${RC1}, best checkpoint: ${CKPT_W01}"

if [[ ! -f "$CKPT_W01" ]]; then
    log "ERROR: Stage 1 did not produce checkpoint: $CKPT_W01"
    exit 1
fi

# ---- Stage 2: cascade to ω=0.01, warm-start from Stage 1 ----
TAG="n12_shellaware_w001_s${SEED}"
LOGFILE="${LOG_DIR}/${TAG}.log"

log "Stage 2: ω=0.01 cascade (${EPOCHS} epochs) → tag=${TAG}"

CUDA_MANUAL_DEVICE="$GPU" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py \
    --mode bf --n-elec 12 --omega 0.01 \
    --resume "$CKPT_W01" \
    --jas-arch shellaware --jas-hidden 64 --jas-shells 3 \
    --bf-hidden 64 --bf-layers 2 \
    --epochs "$EPOCHS" \
    --lr 2e-5 --lr-jas 3e-6 \
    --lr-warmup-epochs 150 --lr-warmup-init-frac 0.05 \
    --n-coll 8192 --oversample 24 --micro-batch 512 \
    --loss-type reinforce --direct-weight 0.0 \
    --clip-el 4.0 --reward-qtrim 0.02 \
    --resample-weight-temp 0.75 \
    --resample-logw-clip-q 0.02 \
    --ess-floor-ratio 0.05 --ess-oversample-max 48 --ess-oversample-step 2 \
    --rollback-decay 0.94 --rollback-jump-sigma 3.5 \
    --vmc-every 30 --vmc-n 50000 \
    --n-eval 50000 \
    --seed "$SEED" \
    --tag "$TAG" \
    2>&1 | tee -a "$LOGFILE"

RC=${PIPESTATUS[0]}
log "Stage 2 complete rc=${RC}"
log "Best checkpoint: ${CKPT_DIR}/${TAG}_best.pt"
