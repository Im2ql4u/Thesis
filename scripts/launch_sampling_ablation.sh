#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_LIB_FIX="/itf-fi-ml/home/aleksns/.local/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/itf-fi-ml/home/aleksns/.local/lib/python3.11/site-packages/nvidia/cusparse/lib"
OUT_DIR="${OUT_DIR:-outputs/sampling_ablation}"
RUN_PREFIX="${RUN_PREFIX:-ablation}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
mkdir -p "$OUT_DIR"

# Override via environment for quick pilots:
# EPOCHS=400 N_ELEC=20 OMEGA=0.1 NCOLL=512 OVERSAMPLE=8 bash scripts/launch_sampling_ablation.sh
EPOCHS="${EPOCHS:-2000}"
N_ELEC="${N_ELEC:-20}"
OMEGA="${OMEGA:-0.1}"
NCOLL="${NCOLL:-512}"
OVERSAMPLE="${OVERSAMPLE:-8}"
SEED="${SEED:-42}"
MINSR_MIN_ESS="${MINSR_MIN_ESS:-0}"
MINSR_MAX_KHAT="${MINSR_MAX_KHAT:-0}"

GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
GPU_C="${GPU_C:-3}"
GPU_D="${GPU_D:-5}"

kill_if_exists() {
  local s="$1"
  tmux kill-session -t "$s" 2>/dev/null || true
}

start_cell() {
  local session_base="$1"
  local gpu="$2"
  shift 2
  local tag_base="$1"
  shift 1

  local session="${RUN_PREFIX}_${session_base}"
  local tag="${tag_base}${TAG_SUFFIX}"

  local log_file="$OUT_DIR/${tag}.log"
  kill_if_exists "$session"

  tmux new-session -d -s "$session" \
    "cd '$ROOT_DIR' && \
     export LD_LIBRARY_PATH='$CUDA_LIB_FIX':\$LD_LIBRARY_PATH && \
     CUDA_MANUAL_DEVICE='$gpu' PYTHONPATH=src python3.11 src/run_weak_form.py \
       --n-elec '$N_ELEC' --omega '$OMEGA' --n-coll '$NCOLL' --epochs '$EPOCHS' \
       --oversample '$OVERSAMPLE' --seed '$SEED' --tag '$tag' $* \
       > '$log_file' 2>&1"
}

# Cell A: fixed proposal + Adam (control)
start_cell "A" "$GPU_A" "ablation_fixed_adam" \
  --lr 5e-4

# Cell B: adaptive proposal + Adam
start_cell "B" "$GPU_B" "ablation_adaptive_adam" \
  --lr 5e-4 \
  --adaptive-proposal --gmm-components 8 --gmm-refit-every 30 --gmm-refit-min-samples 1024 --gmm-covariance diag

# Cell C: fixed proposal + MinSR
start_cell "C" "$GPU_C" "ablation_fixed_minsr" \
  --lr 5e-3 --natural-grad --sr-mode minsr --fisher-damping 1e-3 --fisher-subsample 512 \
  --minsr-min-ess "$MINSR_MIN_ESS" --minsr-max-khat "$MINSR_MAX_KHAT"

# Cell D: adaptive proposal + MinSR
start_cell "D" "$GPU_D" "ablation_adaptive_minsr" \
  --lr 5e-3 --natural-grad --sr-mode minsr --fisher-damping 1e-3 --fisher-subsample 512 \
  --minsr-min-ess "$MINSR_MIN_ESS" --minsr-max-khat "$MINSR_MAX_KHAT" \
  --adaptive-proposal --gmm-components 8 --gmm-refit-every 30 --gmm-refit-min-samples 1024 --gmm-covariance diag

echo "Started sessions:"
tmux ls | grep -E "^${RUN_PREFIX}_[ABCD]:" || true

echo "Logs:"
ls -1 "$OUT_DIR"/*.log
