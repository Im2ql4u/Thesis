#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v14_lowomega_2stage_reinforce_only"
LOGDIR="${OUT}/logs"
CURATED="${ROOT}/results/curated_low_error_0p1pct_2026-03-21"
ARCH_COLLOC="${ROOT}/results/arch_colloc"
SESSION="v14_lowomega_2stage"
GPU="0"

mkdir -p "${LOGDIR}"

START_CKPT="${ARCH_COLLOC}/v12b_n2w001_stage1.pt"
if [ ! -f "${START_CKPT}" ]; then
  START_CKPT="${CURATED}/v7_n2w001_exact.pt"
fi

CHAIN_SCRIPT="${OUT}/run_chain.sh"
cat > "${CHAIN_SCRIPT}" <<'CHAIN'
#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
LOGDIR="$1"
GPU="$2"
START_CKPT="$3"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'

run_stage() {
  local tag="$1"
  shift
  local log="${LOGDIR}/${tag}.log"
  echo "# Started: $(date) GPU=${GPU} tag=${tag}" >> "${log}"
  (
    cd "${ROOT}"
    eval "${MODULE_CMD}" || true
    CUDA_VISIBLE_DEVICES="${GPU}" timeout 130000s python3 src/run_weak_form.py "$@" --tag "${tag}"
  ) >> "${log}" 2>&1
  local rc=$?
  echo "# Completed: $(date) rc=${rc}" >> "${log}"
  return ${rc}
}

# Stage 1: omega=0.01 polish (REINFORCE only)
run_stage v14_n2w001_polish_reinf \
  --mode bf --n-elec 2 --omega 0.01 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 20000 --n-coll 4096 --oversample 36 --micro-batch 1024 \
  --lr 4e-6 --lr-jas 4e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.002 \
  --rollback-decay 0.99 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess 400 \
  --vmc-every 120 --vmc-n 40000 --n-eval 200000 --seed 3101 \
  --resume "${START_CKPT}" --no-pretrained || exit 1

# Stage 2: omega=0.001 direct transfer (REINFORCE only, tighter stabilization)
run_stage v14_n2w0001_transfer_reinf \
  --mode bf --n-elec 2 --omega 0.001 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 45000 --n-coll 4096 --oversample 80 --micro-batch 1024 \
  --lr 1.2e-6 --lr-jas 1.2e-6 --direct-weight 0.0 \
  --clip-el 3.5 --reward-qtrim 0.001 \
  --rollback-decay 0.997 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess 600 \
  --vmc-every 160 --vmc-n 70000 --n-eval 300000 --seed 3102 \
  --resume "${ROOT}/results/arch_colloc/v14_n2w001_polish_reinf.pt" --no-pretrained || exit 1
CHAIN

chmod +x "${CHAIN_SCRIPT}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -c "${ROOT}" "${CHAIN_SCRIPT} '${LOGDIR}' '${GPU}' '${START_CKPT}'"

echo "Started session: ${SESSION}"
echo "Output: ${OUT}"
echo "Logs: ${LOGDIR}"
echo "Start checkpoint: ${START_CKPT}"
