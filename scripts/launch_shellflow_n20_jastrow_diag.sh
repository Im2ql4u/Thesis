#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

RUN_ID="${RUN_ID:-2026-04-16_shellflow_n20_jastrow_diag_v8}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

JAS_INPUT_RADIUS_CAP_AHO="${JAS_INPUT_RADIUS_CAP_AHO:-2.0}"
JAS_TAIL_GUARD_RADIUS_AHO="${JAS_TAIL_GUARD_RADIUS_AHO:-0.0}"
JAS_TAIL_GUARD_STRENGTH="${JAS_TAIL_GUARD_STRENGTH:-0.0}"
JAS_TAIL_GUARD_POWER="${JAS_TAIL_GUARD_POWER:-4.0}"
JAS_INIT_MODES_RAW="${JAS_INIT_MODES:-transfer,no_transfer}"
OMEGA_FILTER_RAW="${OMEGA_FILTER:-}"

BASE_CKPT="${ROOT}/results/arch_colloc/camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "Missing base checkpoint: $BASE_CKPT" >&2
  exit 1
fi

SEED=42
GPUS=(1 2 3 4)
OMEGAS=(0.01 0.01 0.001 0.001)
VARIANTS=("2shell" "3shell" "2shell" "3shell")
EPOCHS=(800 800 1200 1200)
SHELL_TEMPLATES=("20-0,1-19" "20-0-0,1-19-0,1-7-12" "20-0,1-19" "20-0-0,1-19-0,1-7-12")
SHELL_MIX_LOGITS=("0.0,0.0" "0.0,0.0,0.0" "0.0,0.0" "0.0,0.0,0.0")
SHELL_SIGMAS=("0.28,0.90" "0.22,0.55,1.00" "0.22,0.80" "0.18,0.45,0.85")
OVERSAMPLES=(48 48 80 80)
LRS=(8e-5 8e-5 5e-5 5e-5)
LR_JASS=(8e-6 8e-6 5e-6 5e-6)
ROLLBACK_JUMPS=(5.0 5.0 6.0 6.0)

run_case() {
  local gpu="$1"
  local omega="$2"
  local variant="$3"
  local epochs="$4"
  local shell_templates="$5"
  local shell_mix_logits="$6"
  local shell_sigmas="$7"
  local oversample="$8"
  local lr="${9}"
  local lr_jas="${10}"
  local rollback_jump="${11}"
  local init_mode="${12}"
  local log_file="${13}"

  local omega_tag
  omega_tag="$(echo "$omega" | sed 's/\./p/g')"
  local mode_tag
  local init_args=()
  case "$init_mode" in
    transfer)
      mode_tag="transfer"
      init_args=(--init-jas "$BASE_CKPT")
      ;;
    no_transfer)
      mode_tag="scratch"
      init_args=()
      ;;
    *)
      echo "Unsupported JAS_INIT_MODES entry: $init_mode" >&2
      return 2
      ;;
  esac
  local tag="shellflow_n20_jdiag_${variant}_${mode_tag}_w${omega_tag}_s${SEED}"

  local args=(
    --mode jastrow --n-elec 20 --omega "$omega"
    --epochs "$epochs" --lr "$lr" --lr-jas "$lr_jas"
    --n-coll 2048 --oversample "$oversample" --micro-batch 256
    --loss-type reinforce --direct-weight 0.0 --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01
    --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma "$rollback_jump"
    --adaptive-proposal --proposal-model shellflow
    --ess-floor-ratio 0.01 --ess-oversample-max 128 --ess-oversample-step 16 --ess-resample-tries 4
    --ess-update-floor 16
    --resample-weight-temp 0.70 --resample-logw-clip-q 0.98
    --top1-mass-update-ceiling 0.22 --top10-mass-update-ceiling 0.70
    --shell-templates "$shell_templates" --shell-mix-logits-init "$shell_mix_logits"
    --shell-sigmas "$shell_sigmas" --shell-refit-steps 120 --shell-refit-lr 1e-3
    --shell-flow-layers 2 --shell-flow-hidden 128
    --shell-ring-warmup-steps 120 --shell-radius-anchor-weight 0.35
    --gmm-refit-every 15 --gmm-refit-min-samples 500
    --jas-input-radius-cap-aho "$JAS_INPUT_RADIUS_CAP_AHO"
    --jas-tail-guard-radius-aho "$JAS_TAIL_GUARD_RADIUS_AHO"
    --jas-tail-guard-strength "$JAS_TAIL_GUARD_STRENGTH"
    --jas-tail-guard-power "$JAS_TAIL_GUARD_POWER"
    --vmc-every 40 --vmc-n 12000 --n-eval 30000
    --seed "$SEED" --tag "$tag"
    --no-pretrained
  )
  args+=("${init_args[@]}")

  echo "[$(date '+%F %T')] START gpu=${gpu} omega=${omega} variant=${variant} init_mode=${init_mode} oversample=${oversample} jas_cap=${JAS_INPUT_RADIUS_CAP_AHO} jas_tail_r=${JAS_TAIL_GUARD_RADIUS_AHO} jas_tail_s=${JAS_TAIL_GUARD_STRENGTH}" | tee -a "$log_file"
  CUDA_MANUAL_DEVICE="$gpu" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py "${args[@]}" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END   gpu=${gpu} omega=${omega} variant=${variant} init_mode=${init_mode} rc=${rc}" | tee -a "$log_file"
  return "$rc"
}

IFS=',' read -r -a JAS_INIT_MODES <<< "$JAS_INIT_MODES_RAW"
IFS=',' read -r -a OMEGA_FILTERS <<< "$OMEGA_FILTER_RAW"
echo "[$(date '+%F %T')] Launching N=20 Jastrow+ShellFlow diagnostics: ${RUN_ID} (init_modes=${JAS_INIT_MODES_RAW}, jas_cap=${JAS_INPUT_RADIUS_CAP_AHO}, jas_tail_r=${JAS_TAIL_GUARD_RADIUS_AHO}, jas_tail_s=${JAS_TAIL_GUARD_STRENGTH}, jas_tail_p=${JAS_TAIL_GUARD_POWER})" | tee -a "${OUT_DIR}/launcher.log"
for init_mode in "${JAS_INIT_MODES[@]}"; do
  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    omega="${OMEGAS[$i]}"
    if [[ -n "$OMEGA_FILTER_RAW" ]]; then
      keep_case=0
      for omega_keep in "${OMEGA_FILTERS[@]}"; do
        if [[ "$omega" == "$omega_keep" ]]; then
          keep_case=1
          break
        fi
      done
      if [[ "$keep_case" -ne 1 ]]; then
        continue
      fi
    fi
    variant="${VARIANTS[$i]}"
    epochs="${EPOCHS[$i]}"
    shell_templates="${SHELL_TEMPLATES[$i]}"
    shell_mix_logits="${SHELL_MIX_LOGITS[$i]}"
    shell_sigmas="${SHELL_SIGMAS[$i]}"
    oversample="${OVERSAMPLES[$i]}"
    lr="${LRS[$i]}"
    lr_jas="${LR_JASS[$i]}"
    rollback_jump="${ROLLBACK_JUMPS[$i]}"
    log_file="${LOG_DIR}/n20_${variant}_${init_mode}_w$(echo "$omega" | sed 's/\./p/g')_s${SEED}.log"
    run_case "$gpu" "$omega" "$variant" "$epochs" "$shell_templates" "$shell_mix_logits" "$shell_sigmas" "$oversample" "$lr" "$lr_jas" "$rollback_jump" "$init_mode" "$log_file" &
    echo "[$(date '+%F %T')] Spawned pid=$! gpu=${gpu} omega=${omega} variant=${variant} init_mode=${init_mode}" | tee -a "${OUT_DIR}/launcher.log"
  done
done

wait
echo "[$(date '+%F %T')] All diagnostics finished" | tee -a "${OUT_DIR}/launcher.log"