#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
cd "${ROOT}"

TIMESTAMP="$(date +%Y-%m-%d_%H%M)"
RUN_ID="${RUN_ID:-${TIMESTAMP}_n20_from_n12_lowomega_transfer_v1}"
OUT_DIR="${ROOT}/outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
SESSION="${SESSION:-n20_from_n12_lowomega_transfer}"
GPU="${GPU:-1}"
WALL="${WALL:-172800s}"

MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
BASE_CKPT="${ROOT}/results/arch_colloc/n12_lowomega_uncap_long_s42_best.pt"
LOG_FILE="${LOG_DIR}/transfer.log"
MPL_DIR="/tmp/matplotlib-${SESSION}-gpu${GPU}"

mkdir -p "${LOG_DIR}" "${MPL_DIR}"

if [ ! -f "${BASE_CKPT}" ]; then
  echo "Missing base checkpoint: ${BASE_CKPT}"
  exit 1
fi

WORKER_SCRIPT="${OUT_DIR}/worker.sh"
cat > "${WORKER_SCRIPT}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$1"
GPU="$2"
LOG_FILE="$3"
MODULE_CMD="$4"
BASE_CKPT="$5"
MPL_DIR="$6"
WALL="$7"

run_stage() {
  local tag="$1"
  shift
  echo "[$(date '+%F %T')] START gpu=${GPU} tag=${tag}" >> "${LOG_FILE}"
  (
    cd "${ROOT}"
    eval "${MODULE_CMD}" || true
    MPLCONFIGDIR="${MPL_DIR}" CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 timeout "${WALL}" \
      python3 src/run_weak_form.py "$@" --seed 42 --tag "${tag}"
  ) >> "${LOG_FILE}" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   gpu=${GPU} tag=${tag} rc=${rc}" >> "${LOG_FILE}"
  return "${rc}"
}

stage1_tag="n20_xfer_from_n12uncap_w01_s42"
run_stage "${stage1_tag}" \
  --mode bf --n-elec 20 --omega 0.1 --e-dmc 29.9779 \
  --epochs 2600 --n-coll 768 --oversample 16 --micro-batch 64 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0,9.0 \
  --lr 3e-5 --lr-jas 3e-6 --lr-min-frac 0.02 \
  --lr-warmup-epochs 60 --lr-warmup-init-frac 0.05 \
  --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02 \
  --ess-floor-ratio 0.05 --ess-oversample-max 32 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --jas-input-radius-cap-aho 0.0 \
  --vmc-every 150 --vmc-n 15000 --n-eval 30000 \
  --save-best-window 80 \
  --init-jas "${BASE_CKPT}" --init-bf "${BASE_CKPT}" --no-pretrained

stage1_best="${ROOT}/results/arch_colloc/${stage1_tag}_best.pt"
if [ ! -f "${stage1_best}" ]; then
  echo "[$(date '+%F %T')] ERROR missing stage1 best checkpoint: ${stage1_best}" >> "${LOG_FILE}"
  exit 1
fi

stage2_tag="n20_xfer_from_n12uncap_w001_fromw01_s42"
run_stage "${stage2_tag}" \
  --mode bf --n-elec 20 --omega 0.01 --e-dmc 6.14645 \
  --epochs 3200 --n-coll 512 --oversample 18 --micro-batch 48 \
  --sigma-fs 0.8,1.3,2.0,3.5,6.0,9.0 \
  --lr 1.5e-5 --lr-jas 1.5e-6 --lr-min-frac 0.02 \
  --lr-warmup-epochs 80 --lr-warmup-init-frac 0.05 \
  --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02 \
  --ess-floor-ratio 0.03 --ess-oversample-max 32 --ess-oversample-step 2 --ess-resample-tries 2 \
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --jas-input-radius-cap-aho 0.0 \
  --vmc-every 200 --vmc-n 15000 --n-eval 30000 \
  --save-best-window 100 \
  --resume "${stage1_best}" --no-pretrained

echo "# Completed: $(date)" >> "${LOG_FILE}"
WORKER

chmod +x "${WORKER_SCRIPT}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "${WORKER_SCRIPT} '${ROOT}' '${GPU}' '${LOG_FILE}' '${MODULE_CMD}' '${BASE_CKPT}' '${MPL_DIR}' '${WALL}'"

echo "Started tmux session: ${SESSION}"
echo "GPU: ${GPU}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Base checkpoint: ${BASE_CKPT}"
echo "Attach with: tmux attach -t ${SESSION}"
