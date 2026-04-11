#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
SESSION="v16_full_lowomega"
OUT_DIR="${ROOT}/outputs/2026-03-26_1842_campaign_v16_full_lowomega_adaptive"
LOGDIR="${OUT_DIR}/logs"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'

mkdir -p "${LOGDIR}"

# Jobs to launch when GPUs become available.
# Format:
# job_name|resume_ckpt|n_elec|omega|bf_hidden|bf_msg_hidden|bf_layers|epochs|n_coll|oversample|micro_batch|lr|clip_el|qtrim|min_ess|vmc_every|vmc_n|n_eval|seed
JOBS=(
  "n6_chain|results/arch_colloc/v12b_n6w01.pt|6|0.005|48|48|2|3000|4096|72|1024|4.0e-6|3.4|0.0012|420|120|50000|220000|5201"
  "n12_chain|results/arch_colloc/v12b_n12w01.pt|12|0.005|48|48|2|2500|3072|88|512|3.0e-6|3.3|0.0010|350|140|50000|220000|5301"
)

job_started() {
  local job_name="$1"
  tmux list-windows -t "${SESSION}" 2>/dev/null | grep -q "${job_name}" && return 0
  return 1
}

job_done() {
  local job_name="$1"
  if [ "${job_name}" = "n6_chain" ] && [ -f "${ROOT}/results/arch_colloc/v16_n6w0001_transfer.pt" ]; then
    return 0
  fi
  if [ "${job_name}" = "n12_chain" ] && [ -f "${ROOT}/results/arch_colloc/v16_n12w0001_transfer.pt" ]; then
    return 0
  fi
  return 1
}

launch_chain_window() {
  local gpu="$1"
  local job_name="$2"
  local resume="$3"
  local n_elec="$4"
  local omega_bridge="$5"
  local bf_hidden="$6"
  local bf_msg_hidden="$7"
  local bf_layers="$8"
  local ep_bridge="$9"
  local n_coll="${10}"
  local oversample_bridge="${11}"
  local micro_batch="${12}"
  local lr_bridge="${13}"
  local clip_bridge="${14}"
  local qtrim_bridge="${15}"
  local min_ess_bridge="${16}"
  local vmc_every_bridge="${17}"
  local vmc_n_bridge="${18}"
  local n_eval_bridge="${19}"
  local seed_bridge="${20}"

  local tag_bridge tag_transfer ep_transfer oversample_transfer lr_transfer clip_transfer qtrim_transfer min_ess_transfer vmc_every_transfer vmc_n_transfer n_eval_transfer seed_transfer
  if [ "${job_name}" = "n6_chain" ]; then
    tag_bridge="v16_n6w0005_bridge"
    tag_transfer="v16_n6w0001_transfer"
    ep_transfer="7000"
    oversample_transfer="92"
    lr_transfer="1.2e-6"
    clip_transfer="3.2"
    qtrim_transfer="0.0010"
    min_ess_transfer="500"
    vmc_every_transfer="160"
    vmc_n_transfer="70000"
    n_eval_transfer="300000"
    seed_transfer="5202"
  else
    tag_bridge="v16_n12w0005_bridge"
    tag_transfer="v16_n12w0001_transfer"
    ep_transfer="5000"
    oversample_transfer="110"
    lr_transfer="8.0e-7"
    clip_transfer="3.1"
    qtrim_transfer="0.0008"
    min_ess_transfer="420"
    vmc_every_transfer="180"
    vmc_n_transfer="70000"
    n_eval_transfer="300000"
    seed_transfer="5302"
  fi

  tmux new-window -t "${SESSION}" -n "${job_name}_gpu${gpu}" -c "${ROOT}" "bash -lc '
set -euo pipefail
run_stage() {
  local tag=\"$1\"; shift
  local log=\"${LOGDIR}/\${tag}.log\"
  echo \"# Started: \$(date) GPU=${gpu} tag=\${tag}\" >> \"\${log}\"
  (
    cd \"${ROOT}\"
    eval \"${MODULE_CMD}\" || true
    CUDA_VISIBLE_DEVICES=${gpu} timeout 200000s python3 src/run_weak_form.py "$@" --tag \"\${tag}\"
  ) >> \"\${log}\" 2>&1
  echo \"# Completed: \$(date) rc=$?\" >> \"\${log}\"
}

if [ ! -f \"${ROOT}/${resume}\" ]; then
  echo \"Missing resume checkpoint: ${ROOT}/${resume}\" | tee -a \"${LOGDIR}/${job_name}.log\"
  exit 1
fi

run_stage ${tag_bridge} \
  --mode bf --n-elec ${n_elec} --omega ${omega_bridge} \
  --bf-hidden ${bf_hidden} --bf-msg-hidden ${bf_msg_hidden} --bf-layers ${bf_layers} \
  --epochs ${ep_bridge} --n-coll ${n_coll} --oversample ${oversample_bridge} --micro-batch ${micro_batch} \
  --lr ${lr_bridge} --lr-jas ${lr_bridge} --direct-weight 0.0 \
  --clip-el ${clip_bridge} --reward-qtrim ${qtrim_bridge} \
  --rollback-decay 0.997 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess ${min_ess_bridge} \
  --vmc-every ${vmc_every_bridge} --vmc-n ${vmc_n_bridge} --n-eval ${n_eval_bridge} --seed ${seed_bridge} \
  --resume ${ROOT}/${resume} --no-pretrained

run_stage ${tag_transfer} \
  --mode bf --n-elec ${n_elec} --omega 0.001 \
  --bf-hidden ${bf_hidden} --bf-msg-hidden ${bf_msg_hidden} --bf-layers ${bf_layers} \
  --epochs ${ep_transfer} --n-coll ${n_coll} --oversample ${oversample_transfer} --micro-batch ${micro_batch} \
  --lr ${lr_transfer} --lr-jas ${lr_transfer} --direct-weight 0.0 \
  --clip-el ${clip_transfer} --reward-qtrim ${qtrim_transfer} \
  --rollback-decay 0.997 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
  --min-ess ${min_ess_transfer} \
  --vmc-every ${vmc_every_transfer} --vmc-n ${vmc_n_transfer} --n-eval ${n_eval_transfer} --seed ${seed_transfer} \
  --resume ${ROOT}/results/arch_colloc/${tag_bridge}.pt --no-pretrained
'"

  echo "Launched ${job_name} on GPU ${gpu}"
}

echo "Autoscaler started: $(date)"
echo "Watching for free GPUs (free>=10000 MiB and util<=10%)"

while true; do
  mapfile -t free_gpus < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /, "", $0); if ($2 >= 10000 && $3 <= 10) print $1}')

  for row in "${JOBS[@]}"; do
    IFS='|' read -r job_name resume n_elec omega_bridge bf_hidden bf_msg_hidden bf_layers ep_bridge n_coll oversample_bridge micro_batch lr_bridge clip_bridge qtrim_bridge min_ess_bridge vmc_every_bridge vmc_n_bridge n_eval_bridge seed_bridge <<< "${row}"

    if job_done "${job_name}"; then
      continue
    fi
    if job_started "${job_name}"; then
      continue
    fi

    if [ ${#free_gpus[@]} -gt 0 ]; then
      gpu="${free_gpus[0]}"
      free_gpus=("${free_gpus[@]:1}")
      launch_chain_window "${gpu}" "${job_name}" "${resume}" "${n_elec}" "${omega_bridge}" "${bf_hidden}" "${bf_msg_hidden}" "${bf_layers}" "${ep_bridge}" "${n_coll}" "${oversample_bridge}" "${micro_batch}" "${lr_bridge}" "${clip_bridge}" "${qtrim_bridge}" "${min_ess_bridge}" "${vmc_every_bridge}" "${vmc_n_bridge}" "${n_eval_bridge}" "${seed_bridge}"
    fi
  done

  sleep 60
done
