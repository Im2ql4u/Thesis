#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v16_full_lowomega_adaptive"
LOGDIR="${OUT}/logs"
SESSION="v16_full_lowomega"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'

mkdir -p "${LOGDIR}"

# Pick truly available GPUs now (free memory + near-idle utilization).
mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
  | awk -F',' '{gsub(/ /, "", $0); if ($2 >= 10000 && $3 <= 10) print $1}')

if [ ${#GPU_LIST[@]} -eq 0 ]; then
  echo "No idle GPUs available (criteria: free>=10000MiB, util<=10%)."
  echo "Try again when GPUs free up."
  exit 1
fi

N_GPU=${#GPU_LIST[@]}

echo "Detected available GPUs: ${GPU_LIST[*]}"
echo "Using ${N_GPU} GPU(s)"

auto_worker_script="${OUT}/worker.sh"
cat > "${auto_worker_script}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$1"
LOGDIR="$2"
GPU="$3"
SLOT="$4"
N_GPU="$5"
MODULE_CMD="$6"

run_stage() {
  local tag="$1"
  shift
  local log="${LOGDIR}/${tag}.log"
  echo "# Started: $(date) GPU=${GPU} tag=${tag}" >> "${log}"
  (
    cd "${ROOT}"
    eval "${MODULE_CMD}" || true
    CUDA_VISIBLE_DEVICES="${GPU}" timeout 200000s python3 src/run_weak_form.py "$@" --tag "${tag}"
  ) >> "${log}" 2>&1
  local rc=$?
  echo "# Completed: $(date) rc=${rc}" >> "${log}"
  return ${rc}
}

# job format:
# tag|resume|n_elec|omega|bf_hidden|bf_msg_hidden|bf_layers|epochs|n_coll|oversample|micro_batch|lr|clip_el|qtrim|min_ess|vmc_every|vmc_n|n_eval|seed
JOBS=(
  "v16_n2w0001_transfer|results/arch_colloc/v14_n2w001_polish_reinf.pt|2|0.001|64|64|2|20000|4096|80|1024|1.2e-6|3.5|0.0010|600|160|70000|300000|5101"
  "v16_n6w0005_bridge|results/arch_colloc/v12b_n6w01.pt|6|0.005|48|48|2|3000|4096|72|1024|4.0e-6|3.4|0.0012|420|120|50000|220000|5201"
  "v16_n6w0001_transfer|results/arch_colloc/v16_n6w0005_bridge.pt|6|0.001|48|48|2|7000|4096|92|1024|1.2e-6|3.2|0.0010|500|160|70000|300000|5202"
  "v16_n12w0005_bridge|results/arch_colloc/v12b_n12w01.pt|12|0.005|48|48|2|2500|3072|88|512|3.0e-6|3.3|0.0010|350|140|50000|220000|5301"
  "v16_n12w0001_transfer|results/arch_colloc/v16_n12w0005_bridge.pt|12|0.001|48|48|2|5000|3072|110|512|8.0e-7|3.1|0.0008|420|180|70000|300000|5302"
)

for idx in "${!JOBS[@]}"; do
  # simple static sharding across available GPUs
  if [ $(( idx % N_GPU )) -ne "${SLOT}" ]; then
    continue
  fi

  IFS='|' read -r tag resume n_elec omega bf_hidden bf_msg_hidden bf_layers epochs n_coll oversample micro_batch lr clip_el qtrim min_ess vmc_every vmc_n n_eval seed <<< "${JOBS[idx]}"

  if [ ! -f "${ROOT}/${resume}" ]; then
    echo "# Skipping ${tag}: resume checkpoint missing at ${ROOT}/${resume}" | tee -a "${LOGDIR}/${tag}.log"
    continue
  fi

  run_stage "${tag}" \
    --mode bf --n-elec "${n_elec}" --omega "${omega}" \
    --bf-hidden "${bf_hidden}" --bf-msg-hidden "${bf_msg_hidden}" --bf-layers "${bf_layers}" \
    --epochs "${epochs}" --n-coll "${n_coll}" --oversample "${oversample}" --micro-batch "${micro_batch}" \
    --lr "${lr}" --lr-jas "${lr}" --direct-weight 0.0 \
    --clip-el "${clip_el}" --reward-qtrim "${qtrim}" \
    --rollback-decay 0.997 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
    --min-ess "${min_ess}" \
    --vmc-every "${vmc_every}" --vmc-n "${vmc_n}" --n-eval "${n_eval}" --seed "${seed}" \
    --resume "${ROOT}/${resume}" --no-pretrained

done
WORKER

chmod +x "${auto_worker_script}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -c "${ROOT}" "echo 'v16 coordinator started at $(date)'; sleep 1"

for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$i]}"
  wname="gpu${gpu}"
  tmux new-window -t "${SESSION}" -n "${wname}" -c "${ROOT}" \
    "${auto_worker_script} '${ROOT}' '${LOGDIR}' '${gpu}' '${i}' '${N_GPU}' '${MODULE_CMD}'"
done

# Keep only worker windows visible
if tmux list-windows -t "${SESSION}" | grep -q ':0'; then
  tmux kill-window -t "${SESSION}:0" 2>/dev/null || true
fi

echo "Started session: ${SESSION}"
echo "Output: ${OUT}"
echo "Logs: ${LOGDIR}"
echo "GPUs used: ${GPU_LIST[*]}"

echo "ETA guidance (single-GPU baseline):"
echo "  N=2 transfer: ~10h"
echo "  N=6 bridge+transfer: ~28h"
echo "  N=12 bridge+transfer: ~50h"
echo "  Total on 1 GPU: ~88h (~3.7 days)"
echo "  Total on ${N_GPU} GPUs (ideal shard): ~$(( (88 + N_GPU - 1) / N_GPU ))h"
