#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-24_shellaware_n20_lowomega_v2}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"
PID_FILE="${OUT_DIR}/pids.txt"
: > "$PID_FILE"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${RUN_ID}}"
mkdir -p "$MPLCONFIGDIR"

# First ansatz-side test after adding soft radial-shell Jastrow channels.
# Keep ShellFlow/ring settings close to the strongest existing low-omega
# diagnostics, and only sweep spin ordering plus shell-aware readout.

SEED="${SEED:-42}"
GPUS=(${GPUS:-3 4})
SPIN_LAYOUTS=("block" "alternating")
OMEGA="${OMEGA:-0.01}"
EPOCHS="${EPOCHS:-800}"
N_COLL="${N_COLL:-2048}"
OVERSAMPLE="${OVERSAMPLE:-128}"
MICRO_BATCH="${MICRO_BATCH:-256}"

SHELL_TEMPLATES="${SHELL_TEMPLATES:-20-0-0,1-19-0,1-7-12}"
SHELL_MIX_LOGITS="${SHELL_MIX_LOGITS:-0.0,0.0,0.0}"
SHELL_RADII="${SHELL_RADII:-0.0,0.45,1.05}"
SHELL_SIGMAS="${SHELL_SIGMAS:-0.45,0.80,1.35}"
SHELL_FLOW_LAYERS="${SHELL_FLOW_LAYERS:-0}"
SHELL_REFIT_STEPS="${SHELL_REFIT_STEPS:-80}"
SHELL_REFIT_EVERY="${SHELL_REFIT_EVERY:-10}"
SHELL_CURRICULUM_MODE="${SHELL_CURRICULUM_MODE:-epoch}"
SHELL_CURRICULUM_UNLOCK_EPOCH="${SHELL_CURRICULUM_UNLOCK_EPOCH:-120}"
SHELL_CURRICULUM_CENTERED_MASS_FLOOR="${SHELL_CURRICULUM_CENTERED_MASS_FLOOR:-0.15}"
ESS_UPDATE_FLOOR="${ESS_UPDATE_FLOOR:-0}"
RESAMPLE_WEIGHT_TEMP="${RESAMPLE_WEIGHT_TEMP:-0.55}"
RESAMPLE_LOGW_CLIP_Q="${RESAMPLE_LOGW_CLIP_Q:-0.97}"

JAS_HIDDEN="${JAS_HIDDEN:-48}"
JAS_MP_STEPS="${JAS_MP_STEPS:-2}"
JAS_SHELLS="${JAS_SHELLS:-3}"
JAS_SHELL_RADIUS_AHO="${JAS_SHELL_RADIUS_AHO:-3.0}"
JAS_SHELL_WIDTH_AHO="${JAS_SHELL_WIDTH_AHO:-0.65}"
JAS_INPUT_RADIUS_CAP_AHO="${JAS_INPUT_RADIUS_CAP_AHO:-2.0}"

run_case() {
  local gpu="$1"
  local spin_layout="$2"
  local log_file="$3"
  local omega_tag
  omega_tag="$(echo "$OMEGA" | sed 's/\./p/g')"
  local tag="shellaware_n20_${spin_layout}_w${omega_tag}_s${SEED}"

  local args=(
    --mode jastrow --n-elec 20 --omega "$OMEGA"
    --spin-layout "$spin_layout"
    --jas-arch shellaware --jas-hidden "$JAS_HIDDEN" --jas-mp-steps "$JAS_MP_STEPS"
    --jas-shells "$JAS_SHELLS"
    --jas-shell-radius-aho "$JAS_SHELL_RADIUS_AHO"
    --jas-shell-width-aho "$JAS_SHELL_WIDTH_AHO"
    --jas-input-radius-cap-aho "$JAS_INPUT_RADIUS_CAP_AHO"
    --epochs "$EPOCHS" --lr 8e-5 --lr-jas 8e-6
    --n-coll "$N_COLL" --oversample "$OVERSAMPLE" --micro-batch "$MICRO_BATCH"
    --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01
    --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0
    --adaptive-proposal --proposal-model shellflow
    --ess-floor-ratio 0.01 --ess-oversample-max 256 --ess-oversample-step 32 --ess-resample-tries 6
    --ess-update-floor "$ESS_UPDATE_FLOOR"
    --resample-weight-temp "$RESAMPLE_WEIGHT_TEMP" --resample-logw-clip-q "$RESAMPLE_LOGW_CLIP_Q"
    --top1-mass-update-ceiling 0.35 --top10-mass-update-ceiling 0.80
    --shell-templates "$SHELL_TEMPLATES" --shell-mix-logits-init "$SHELL_MIX_LOGITS"
    --shell-radii-init "$SHELL_RADII" --shell-sigmas "$SHELL_SIGMAS"
    --shell-refit-steps "$SHELL_REFIT_STEPS" --shell-refit-lr 1e-3
    --shell-flow-layers "$SHELL_FLOW_LAYERS" --shell-flow-hidden 128
    --shell-ring-warmup-steps 0 --shell-radius-anchor-weight 0.20
    --shell-curriculum-mode "$SHELL_CURRICULUM_MODE"
    --shell-curriculum-unlock-epoch "$SHELL_CURRICULUM_UNLOCK_EPOCH"
    --shell-curriculum-centered-mass-floor "$SHELL_CURRICULUM_CENTERED_MASS_FLOOR"
    --gmm-refit-every "$SHELL_REFIT_EVERY" --gmm-refit-min-samples 500
    --vmc-every 40 --vmc-n 12000 --n-eval 30000
    --seed "$SEED" --tag "$tag"
    --no-pretrained
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} spin=${spin_layout} omega=${OMEGA} jas=shellaware" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} spin=${spin_layout} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

echo "[$(date '+%F %T')] Launching shell-aware N=20 low-omega sweep: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
for i in "${!SPIN_LAYOUTS[@]}"; do
  gpu="${GPUS[$i]:-${GPUS[0]}}"
  spin_layout="${SPIN_LAYOUTS[$i]}"
  log_file="${LOG_DIR}/n20_${spin_layout}_w$(echo "$OMEGA" | sed 's/\./p/g')_s${SEED}.log"
  run_case "$gpu" "$spin_layout" "$log_file" &
  pid=$!
  echo "${pid} ${gpu} ${spin_layout}" >> "$PID_FILE"
  echo "[$(date '+%F %T')] Spawned pid=${pid} gpu=${gpu} spin=${spin_layout}" | tee -a "${OUT_DIR}/launcher.log"
done

wait
rc=$?
echo "[$(date '+%F %T')] wait rc=${rc}" | tee -a "${OUT_DIR}/launcher.log"
echo "[$(date '+%F %T')] All shell-aware low-omega runs finished" | tee -a "${OUT_DIR}/launcher.log"
