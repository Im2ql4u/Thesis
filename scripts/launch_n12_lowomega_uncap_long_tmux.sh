#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
cd "${ROOT}"

TIMESTAMP="$(date +%Y-%m-%d_%H%M)"
RUN_ID="${RUN_ID:-${TIMESTAMP}_n12_lowomega_uncap_long_v1}"
OUT_DIR="${ROOT}/outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
SESSION="${SESSION:-n12_lowomega_uncap_long}"
GPU="${GPU:-4}"
WALL="${WALL:-86400s}"

MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
BASE_CKPT="${ROOT}/results/arch_colloc/n12_lowomega_uncap_hisamp_cuda_s42_best.pt"
TAG="${TAG:-n12_lowomega_uncap_long_s42}"
LOG_FILE="${LOG_DIR}/${TAG}.log"
MPL_DIR="/tmp/matplotlib-${TAG}-gpu${GPU}"

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
TAG="$6"
MPL_DIR="$7"
WALL="$8"

echo "# Started: $(date) GPU=${GPU} tag=${TAG}" >> "${LOG_FILE}"
(
  cd "${ROOT}"
  eval "${MODULE_CMD}" || true
  MPLCONFIGDIR="${MPL_DIR}" CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 timeout "${WALL}" \
    python3 src/run_weak_form.py \
      --mode bf --n-elec 12 --omega 0.01 --e-dmc 2.47363 \
      --epochs 3600 --n-coll 8192 --oversample 16 --micro-batch 512 \
      --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
      --lr 5e-5 --lr-jas 5e-6 --lr-min-frac 0.02 \
      --lr-warmup-epochs 60 --lr-warmup-init-frac 0.05 \
      --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02 \
      --ess-floor-ratio 0.10 --ess-oversample-max 32 --ess-oversample-step 2 --ess-resample-tries 2 \
      --rollback-decay 0.95 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
      --jas-input-radius-cap-aho 0.0 \
      --vmc-every 120 --vmc-n 25000 --n-eval 100000 \
      --save-best-window 80 \
      --seed 42 --tag "${TAG}" \
      --resume "${BASE_CKPT}" --no-pretrained
) >> "${LOG_FILE}" 2>&1
rc=$?
echo "# Completed: $(date) rc=${rc}" >> "${LOG_FILE}"
exit "${rc}"
WORKER

chmod +x "${WORKER_SCRIPT}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "${WORKER_SCRIPT} '${ROOT}' '${GPU}' '${LOG_FILE}' '${MODULE_CMD}' '${BASE_CKPT}' '${TAG}' '${MPL_DIR}' '${WALL}'"

echo "Started tmux session: ${SESSION}"
echo "GPU: ${GPU}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Checkpoint: ${BASE_CKPT}"
echo "Attach with: tmux attach -t ${SESSION}"
