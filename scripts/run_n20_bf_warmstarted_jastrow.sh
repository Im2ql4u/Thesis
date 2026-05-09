#!/usr/bin/env bash
# N=20 backflow warm-started from trained Jastrow-only checkpoints.
#
# Usage:
#   ./scripts/run_n20_bf_warmstarted_jastrow.sh w1
#   ./scripts/run_n20_bf_warmstarted_jastrow.sh w05
#   ./scripts/run_n20_bf_warmstarted_jastrow.sh w01
#   ./scripts/run_n20_bf_warmstarted_jastrow.sh all    # sequential: w1 -> w05 -> w01
#
# Optional environment (override defaults):
#   PYTHON=python3
#   REPO_ROOT=/path/to/Thesis_repo          # default: parent of scripts/
#   OUT_ROOT=outputs                        # logs under OUT_ROOT/run_n20_bf_jas_<tag>/
#   SEED=42
#   BF_HIDDEN=48   BF_MSG_HIDDEN=48   BF_LAYERS=2
#   N_COLL=1024    OVERSAMPLE=8       MICRO_BATCH=128   EPOCHS=2000
#   # Per-regime Jastrow init (Jastrow-only .pt with jas_state):
#   JAS_INIT_W1=results/arch_colloc/n20x2_adam_w1_best.pt
#   JAS_INIT_W05=results/arch_colloc/n20x2_adam_w05_best.pt
#   JAS_INIT_W01=results/arch_colloc/n20x2_adam_w01_best.pt
#
# On clusters without torch in plain python, uncomment MODULE_SNIPPET below.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-outputs}"
SEED="${SEED:-42}"
BF_HIDDEN="${BF_HIDDEN:-48}"
BF_MSG_HIDDEN="${BF_MSG_HIDDEN:-48}"
BF_LAYERS="${BF_LAYERS:-2}"
N_COLL="${N_COLL:-1024}"
OVERSAMPLE="${OVERSAMPLE:-8}"
MICRO_BATCH="${MICRO_BATCH:-128}"
EPOCHS="${EPOCHS:-2000}"

JAS_INIT_W1="${JAS_INIT_W1:-results/arch_colloc/n20x2_adam_w1_best.pt}"
JAS_INIT_W05="${JAS_INIT_W05:-results/arch_colloc/n20x2_adam_w05_best.pt}"
JAS_INIT_W01="${JAS_INIT_W01:-results/arch_colloc/n20x2_adam_w01_best.pt}"

# Uncomment on systems that need Lmod PyTorch (see scripts/campaign_v6_n20bf_n2_fewhours.py):
# MODULE_SNIPPET='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
MODULE_SNIPPET="${MODULE_SNIPPET:-}"
if [[ -n "${MODULE_SNIPPET}" ]]; then
  # shellcheck disable=SC1090
  eval "$MODULE_SNIPPET"
fi

stamp="$(date +%Y-%m-%d_%H%M%S)"
run_dir="${OUT_ROOT}/run_n20_bf_jas_${stamp}"
mkdir -p "${run_dir}/logs"

run_one() {
  local name="$1"
  shift
  local log="${run_dir}/logs/${name}.log"
  echo "=== ${name} ===" | tee "${log}"
  echo "Logging to ${log}" | tee -a "${log}"
  "$PYTHON" src/run_weak_form.py "$@" 2>&1 | tee -a "${log}"
}

bf_core=(
  --mode bf
  --n-elec 20
  --bf-hidden "${BF_HIDDEN}"
  --bf-msg-hidden "${BF_MSG_HIDDEN}"
  --bf-layers "${BF_LAYERS}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --n-coll "${N_COLL}"
  --oversample "${OVERSAMPLE}"
  --micro-batch "${MICRO_BATCH}"
  --loss-type reinforce
  --direct-weight 0.0
  --clip-el 5.0
  --grad-clip 1.0
  --reward-qtrim 0.01
  --lr 6e-5
  --lr-jas 6e-6
  --lr-warmup-epochs 40
  --lr-warmup-init-frac 0.1
  --lr-min-frac 0.01
  --ess-floor-ratio 0.02
  --ess-oversample-max 32
  --ess-oversample-step 4
  --rollback-decay 0.97
  --rollback-err-pct 0.0
  --rollback-jump-sigma 5.0
  --vmc-every 200
  --vmc-n 15000
  --n-eval 30000
  --no-pretrained
)

job="${1:-}"
if [[ -z "${job}" ]]; then
  echo "Usage: $0 w1 | w05 | w01 | all"
  exit 1
fi

echo "Run directory: ${run_dir}"

run_w1() {
  [[ -f "${JAS_INIT_W1}" ]] || { echo "Missing Jastrow init: ${JAS_INIT_W1}"; exit 2; }
  run_one "n20_w1_tinybf_h${BF_HIDDEN}_s${SEED}" "${bf_core[@]}" \
    --omega 1.0 \
    --init-jas "${JAS_INIT_W1}" \
    --tag "n20_w1_tinybf_h${BF_HIDDEN}_s${SEED}_${stamp}"
}

run_w05() {
  [[ -f "${JAS_INIT_W05}" ]] || { echo "Missing Jastrow init: ${JAS_INIT_W05}"; exit 2; }
  run_one "n20_w05_tinybf_h${BF_HIDDEN}_s${SEED}" "${bf_core[@]}" \
    --omega 0.5 \
    --init-jas "${JAS_INIT_W05}" \
    --tag "n20_w05_tinybf_h${BF_HIDDEN}_s${SEED}_${stamp}"
}

run_w01() {
  [[ -f "${JAS_INIT_W01}" ]] || { echo "Missing Jastrow init: ${JAS_INIT_W01}"; exit 2; }
  run_one "n20_w01_tinybf_h${BF_HIDDEN}_s${SEED}" "${bf_core[@]}" \
    --omega 0.1 \
    --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
    --init-jas "${JAS_INIT_W01}" \
    --tag "n20_w01_tinybf_h${BF_HIDDEN}_s${SEED}_${stamp}"
}

case "${job}" in
  w1) run_w1 ;;
  w05) run_w05 ;;
  w01) run_w01 ;;
  all) run_w1 && run_w05 && run_w01 ;;
  *)
    echo "Unknown job: ${job} (use w1, w05, w01, or all)"
    exit 1
    ;;
esac

echo "Done. Artifacts under ${run_dir}"
