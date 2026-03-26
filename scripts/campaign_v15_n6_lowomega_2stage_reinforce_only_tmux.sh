#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v15_n6_lowomega_2stage_reinforce_only"
LOGDIR="${OUT}/logs"
ARCH_COLLOC="${ROOT}/results/arch_colloc"
CURATED="${ROOT}/results/curated_low_error_0p1pct_2026-03-21"
SESSION="v15_n6_lowomega_2stage"
GPU="0"

mkdir -p "${LOGDIR}"

START_CKPT="${ARCH_COLLOC}/v12b_n6w01.pt"
if [ ! -f "${START_CKPT}" ]; then
  START_CKPT="${CURATED}/20260318_1149_n6w01_keep.pt"
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

# Stage 1a: N=6, omega=0.01 warmup for fast domain adaptation
run_stage v15_n6w001_warmup_reinf \
  --mode bf --n-elec 6 --omega 0.01 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 10000 --n-coll 4096 --oversample 56 --micro-batch 1024 \
  --lr 1.6e-5 --lr-jas 1.6e-5 --direct-weight 0.0 \
  --clip-el 3.5 --reward-qtrim 0.0015 \
  --rollback-decay 0.993 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess 420 \
  --vmc-every 120 --vmc-n 50000 --n-eval 240000 --seed 4101 \
  --resume "${START_CKPT}" --no-pretrained || exit 1

# Stage 1b: N=6, omega=0.01 polish to approach ~0.1% error band
run_stage v15_n6w001_polish_reinf \
  --mode bf --n-elec 6 --omega 0.01 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 22000 --n-coll 4096 --oversample 64 --micro-batch 1024 \
  --lr 3.5e-6 --lr-jas 3.5e-6 --direct-weight 0.0 \
  --clip-el 3.3 --reward-qtrim 0.0012 \
  --rollback-decay 0.996 --rollback-err-pct 0.0 --rollback-jump-sigma 5.8 \
  --min-ess 520 \
  --vmc-every 140 --vmc-n 60000 --n-eval 260000 --seed 4102 \
  --resume "${ROOT}/results/arch_colloc/v15_n6w001_warmup_reinf.pt" --no-pretrained || exit 1

# Stage 2: N=6, omega=0.001 direct transfer from 0.01 polish
run_stage v15_n6w0001_transfer_reinf \
  --mode bf --n-elec 6 --omega 0.001 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 42000 --n-coll 4096 --oversample 92 --micro-batch 1024 \
  --lr 1.2e-6 --lr-jas 1.2e-6 --direct-weight 0.0 \
  --clip-el 3.2 --reward-qtrim 0.001 \
  --rollback-decay 0.997 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess 700 \
  --vmc-every 180 --vmc-n 70000 --n-eval 300000 --seed 4103 \
  --resume "${ROOT}/results/arch_colloc/v15_n6w001_polish_reinf.pt" --no-pretrained || exit 1
CHAIN

chmod +x "${CHAIN_SCRIPT}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -c "${ROOT}" "${CHAIN_SCRIPT} '${LOGDIR}' '${GPU}' '${START_CKPT}'"

echo "Started session: ${SESSION}"
echo "Output: ${OUT}"
echo "Logs: ${LOGDIR}"
echo "Start checkpoint: ${START_CKPT}"
