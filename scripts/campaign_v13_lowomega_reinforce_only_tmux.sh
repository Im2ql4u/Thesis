#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v13_lowomega_reinforce_only"
LOGDIR="${OUT}/logs"
CURATED="${ROOT}/results/curated_low_error_0p1pct_2026-03-21"
ARCH_COLLOC="${ROOT}/results/arch_colloc"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
SESSION="v13_lowomega_reinf"
GPU="0"

mkdir -p "${LOGDIR}"

LOW001_CKPT="${ARCH_COLLOC}/v12b_n2w001_stage1.pt"
if [ ! -f "${LOW001_CKPT}" ]; then
  LOW001_CKPT="${CURATED}/v7_n2w001_exact.pt"
fi

cat > "${OUT}/run_chain.sh" <<'CHAIN'
#!/usr/bin/env bash
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
LOGDIR="$1"
GPU="$2"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
START_CKPT="$3"

run_stage() {
  local tag="$1"
  local resume="$2"
  shift 2
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

# Stage A: omega=0.01 polish (REINFORCE only)
run_stage v13_n2w001_polish_reinf "${START_CKPT}" \
  --mode bf --n-elec 2 --omega 0.01 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 18000 --n-coll 4096 --oversample 36 --micro-batch 1024 \
  --lr 5e-6 --lr-jas 5e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.002 \
  --rollback-decay 0.99 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --vmc-every 120 --vmc-n 40000 --n-eval 200000 --seed 2101 \
  --resume "${START_CKPT}" --no-pretrained || exit 1

# Stage B: omega=0.005 transfer (REINFORCE only)
run_stage v13_n2w0005_transfer_reinf "${ROOT}/results/arch_colloc/v13_n2w001_polish_reinf.pt" \
  --mode bf --n-elec 2 --omega 0.005 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 22000 --n-coll 4096 --oversample 44 --micro-batch 1024 \
  --lr 4e-6 --lr-jas 4e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.002 \
  --rollback-decay 0.992 --rollback-err-pct 0.0 --rollback-jump-sigma 6.5 \
  --vmc-every 140 --vmc-n 40000 --n-eval 200000 --seed 2102 \
  --resume "${ROOT}/results/arch_colloc/v13_n2w001_polish_reinf.pt" --no-pretrained || exit 1

# Stage C: omega=0.002 transfer (REINFORCE only)
run_stage v13_n2w0002_transfer_reinf "${ROOT}/results/arch_colloc/v13_n2w0005_transfer_reinf.pt" \
  --mode bf --n-elec 2 --omega 0.002 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 26000 --n-coll 4096 --oversample 52 --micro-batch 1024 \
  --lr 3e-6 --lr-jas 3e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.0015 \
  --rollback-decay 0.994 --rollback-err-pct 0.0 --rollback-jump-sigma 7.0 \
  --vmc-every 160 --vmc-n 50000 --n-eval 240000 --seed 2103 \
  --resume "${ROOT}/results/arch_colloc/v13_n2w0005_transfer_reinf.pt" --no-pretrained || exit 1

# Stage D: omega=0.001 transfer (REINFORCE only)
run_stage v13_n2w0001_transfer_reinf "${ROOT}/results/arch_colloc/v13_n2w0002_transfer_reinf.pt" \
  --mode bf --n-elec 2 --omega 0.001 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 32000 --n-coll 4096 --oversample 60 --micro-batch 1024 \
  --lr 2e-6 --lr-jas 2e-6 --direct-weight 0.0 \
  --clip-el 4.0 --reward-qtrim 0.001 \
  --rollback-decay 0.996 --rollback-err-pct 0.0 --rollback-jump-sigma 7.5 \
  --vmc-every 180 --vmc-n 60000 --n-eval 280000 --seed 2104 \
  --resume "${ROOT}/results/arch_colloc/v13_n2w0002_transfer_reinf.pt" --no-pretrained || exit 1
CHAIN

chmod +x "${OUT}/run_chain.sh"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -c "${ROOT}" "${OUT}/run_chain.sh '${LOGDIR}' '${GPU}' '${LOW001_CKPT}'"

echo "Started low-omega REINFORCE-only chain in tmux session: ${SESSION}"
echo "Output directory: ${OUT}"
echo "Logs: ${LOGDIR}"
echo "Start checkpoint: ${LOW001_CKPT}"
