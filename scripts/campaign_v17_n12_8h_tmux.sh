#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v17_n12_8h"
LOGDIR="${OUT}/logs"
SESSION="v17_n12_8h"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'
BASE_CKPT="${ROOT}/results/arch_colloc/v12b_n12w01.pt"

mkdir -p "${LOGDIR}"

if [ ! -f "${BASE_CKPT}" ]; then
  echo "Missing base checkpoint: ${BASE_CKPT}"
  exit 1
fi

# Use currently free GPUs (explicit target for this 8h campaign)
GPU_LIST=(2 3 4 5)

WORKER_SCRIPT="${OUT}/worker.sh"
cat > "${WORKER_SCRIPT}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$1"
LOGDIR="$2"
GPU="$3"
PROFILE="$4"
MODULE_CMD="$5"
BASE_CKPT="$6"

run_stage() {
  local tag="$1"
  shift
  local log="${LOGDIR}/${tag}.log"
  echo "# Started: $(date) GPU=${GPU} tag=${tag}" >> "${log}"
  (
    cd "${ROOT}"
    eval "${MODULE_CMD}" || true
    CUDA_VISIBLE_DEVICES="${GPU}" python3 src/run_weak_form.py "$@" --tag "${tag}"
  ) >> "${log}" 2>&1
  local rc=$?
  echo "# Completed: $(date) rc=${rc}" >> "${log}"
  return ${rc}
}

# Profile-specific stabilization settings
case "${PROFILE}" in
  A)
    BR_OV=120; BR_MESS=60; BR_EPOCHS=1400; BR_LR=3.0e-6; BR_COLL=3072; BR_MB=512; BR_CLIP=3.2; BR_Q=0.0008
    TR_OV=180; TR_MESS=80; TR_EPOCHS=2600; TR_LR=9.0e-7; TR_COLL=3072; TR_MB=512; TR_CLIP=3.0; TR_Q=0.0006
    ;;
  B)
    BR_OV=140; BR_MESS=40; BR_EPOCHS=1600; BR_LR=2.5e-6; BR_COLL=3072; BR_MB=512; BR_CLIP=3.1; BR_Q=0.0008
    TR_OV=220; TR_MESS=60; TR_EPOCHS=2400; TR_LR=8.0e-7; TR_COLL=3072; TR_MB=512; TR_CLIP=2.9; TR_Q=0.0005
    ;;
  C)
    BR_OV=180; BR_MESS=20; BR_EPOCHS=1200; BR_LR=2.2e-6; BR_COLL=2048; BR_MB=512; BR_CLIP=3.0; BR_Q=0.0007
    TR_OV=260; TR_MESS=40; TR_EPOCHS=2800; TR_LR=7.0e-7; TR_COLL=2048; TR_MB=512; TR_CLIP=2.8; TR_Q=0.0005
    ;;
  D)
    BR_OV=200; BR_MESS=10; BR_EPOCHS=1000; BR_LR=2.0e-6; BR_COLL=2048; BR_MB=384; BR_CLIP=2.9; BR_Q=0.0006
    TR_OV=300; TR_MESS=20; TR_EPOCHS=3000; TR_LR=6.0e-7; TR_COLL=2048; TR_MB=384; TR_CLIP=2.7; TR_Q=0.0004
    ;;
  *)
    echo "Unknown profile ${PROFILE}"; exit 2;;
esac

BR_TAG="v17_n12w0005_bridge_${PROFILE}"
TR_TAG="v17_n12w0001_transfer_${PROFILE}"

# Hard campaign wall-clock cap: 8 hours total per worker
# timeout wraps the whole two-stage chain.
timeout 28800s bash -lc '
set -euo pipefail
run_stage_inner() {
  local tag="$1"; shift
  local log="'"${LOGDIR}"'/"${tag}".log"
  echo "# Started: $(date) GPU='"${GPU}"' tag=${tag}" >> "${log}"
  (
    cd "'"${ROOT}"'"
    eval "'"${MODULE_CMD}"'" || true
    CUDA_VISIBLE_DEVICES="'"${GPU}"'" python3 src/run_weak_form.py "$@" --tag "${tag}"
  ) >> "${log}" 2>&1
  local rc=$?
  echo "# Completed: $(date) rc=${rc}" >> "${log}"
  return ${rc}
}

run_stage_inner "'"${BR_TAG}"'" \
  --mode bf --n-elec 12 --omega 0.005 \
  --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --epochs '"${BR_EPOCHS}"' --n-coll '"${BR_COLL}"' --oversample '"${BR_OV}"' --micro-batch '"${BR_MB}"' \
  --lr '"${BR_LR}"' --lr-jas '"${BR_LR}"' --direct-weight 0.0 \
  --clip-el '"${BR_CLIP}"' --reward-qtrim '"${BR_Q}"' \
  --rollback-decay 0.997 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess '"${BR_MESS}"' \
  --vmc-every 220 --vmc-n 25000 --n-eval 100000 --seed 6001 \
  --resume '"${BASE_CKPT}"' --no-pretrained

run_stage_inner "'"${TR_TAG}"'" \
  --mode bf --n-elec 12 --omega 0.001 \
  --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 \
  --epochs '"${TR_EPOCHS}"' --n-coll '"${TR_COLL}"' --oversample '"${TR_OV}"' --micro-batch '"${TR_MB}"' \
  --lr '"${TR_LR}"' --lr-jas '"${TR_LR}"' --direct-weight 0.0 \
  --clip-el '"${TR_CLIP}"' --reward-qtrim '"${TR_Q}"' \
  --rollback-decay 0.997 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess '"${TR_MESS}"' \
  --vmc-every 260 --vmc-n 30000 --n-eval 120000 --seed 6002 \
  --resume '"${ROOT}"'/results/arch_colloc/'"${BR_TAG}"'.pt --no-pretrained
'

rc=$?
if [ $rc -eq 124 ]; then
  echo "# Worker ${PROFILE} hit 8h timeout; stopped by design." | tee -a "${LOGDIR}/v17_worker_${PROFILE}.log"
else
  echo "# Worker ${PROFILE} exited rc=${rc}" | tee -a "${LOGDIR}/v17_worker_${PROFILE}.log"
fi
WORKER

chmod +x "${WORKER_SCRIPT}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -c "${ROOT}" "echo 'v17 started at $(date)'; sleep 1"

profiles=(A B C D)
for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$i]}"
  p="${profiles[$i]}"
  tmux new-window -t "${SESSION}" -n "gpu${gpu}_${p}" -c "${ROOT}" \
    "${WORKER_SCRIPT} '${ROOT}' '${LOGDIR}' '${gpu}' '${p}' '${MODULE_CMD}' '${BASE_CKPT}'"
done

# remove placeholder window
if tmux list-windows -t "${SESSION}" | grep -q '^0:'; then
  tmux kill-window -t "${SESSION}:0" 2>/dev/null || true
fi

echo "Started session: ${SESSION}"
echo "Output: ${OUT}"
echo "Logs: ${LOGDIR}"
echo "GPUs: ${GPU_LIST[*]}"
echo "Hard stop: 8h per worker"
