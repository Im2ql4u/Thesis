#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_ID="${RUN_ID:-2026-04-26_grand_plan_rapid_triage_v1}"
OUT_DIR="${ROOT}/outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
CMD_DIR="${OUT_DIR}/cmds"
MPL_BASE="/tmp/matplotlib-${RUN_ID}"
WALL="${WALL:-3600s}"
GPUS_CSV="${GPUS_CSV:-1,2,3,5,6,7}"
SUITE="${SUITE:-A_B}"

N20_W001_INIT="${N20_W001_INIT:-results/arch_colloc/shellflow_n20_ess2_overnight_w0p01_s42.pt}"
N12_W001_RESUME="${N12_W001_RESUME:-results/arch_colloc/n12_lowomega_uncap_long_s42.pt}"
N20_W01_INIT="${N20_W01_INIT:-results/arch_colloc/n20x2_adam_w01_best.pt}"
N20_W1_INIT="${N20_W1_INIT:-results/arch_colloc/n20x2_adam_w1_best.pt}"
N12_W01_INIT="${N12_W01_INIT:-results/arch_colloc/n12cont_w01_best.pt}"
N16_W01_RESUME="${N16_W01_RESUME:-results/arch_colloc/rapidD1_n16_w01_from_n12.pt}"
N20_W005_RESUME="${N20_W005_RESUME:-results/arch_colloc/rapidE1_n20_w005_from_D2.pt}"

mkdir -p "$LOG_DIR" "$CMD_DIR" "$MPL_BASE"
PIDS=()

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "Need at least one GPU in GPUS_CSV" >&2
  exit 2
fi

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required checkpoint: $path" >&2
    exit 2
  fi
}

write_worker() {
  local gpu="$1"
  local tag="$2"
  local log_file="$3"
  shift 3

  local worker="${CMD_DIR}/${tag}.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    printf 'cd %q\n' "$ROOT"
    echo "source /etc/profile.d/lmod.sh 2>/dev/null || true"
    echo "module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true"
    printf 'export MPLCONFIGDIR=%q\n' "${MPL_BASE}/${tag}"
    echo 'mkdir -p "$MPLCONFIGDIR"'
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$gpu"
    echo "export PYTHONUNBUFFERED=1"
    printf 'echo "[$(date '\''+%%F %%T'\'')] START gpu=%s tag=%s" >> %q\n' "$gpu" "$tag" "$log_file"
    printf 'timeout %q python3 src/run_weak_form.py' "$WALL"
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf ' >> %q 2>&1\n' "$log_file"
    echo 'rc=$?'
    printf 'echo "[$(date '\''+%%F %%T'\'')] END   gpu=%s tag=%s rc=${rc}" >> %q\n' "$gpu" "$tag" "$log_file"
    echo 'exit "$rc"'
  } > "$worker"
  chmod +x "$worker"
  printf '%s\n' "$worker"
}

start_job() {
  local gpu="$1"
  local tag="$2"
  local log_name="$3"
  shift 3

  local log_file="${LOG_DIR}/${log_name}.log"
  local worker
  worker="$(write_worker "$gpu" "$tag" "$log_file" "$@")"
  nohup "$worker" >/dev/null 2>&1 < /dev/null &
  local pid=$!
  PIDS+=("$pid")
  echo "[$(date '+%F %T')] Spawned pid=${pid} gpu=${gpu} tag=${tag} log=${log_file}" | tee -a "${OUT_DIR}/launcher.log"
}

gpu_at() {
  local idx="$1"
  printf '%s\n' "${GPUS[$((idx % ${#GPUS[@]}))]}"
}

common_fast=(
  --seed 42
  --loss-type reinforce --direct-weight 0.0
  --grad-clip 1.0
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0
  --resample-weight-temp 0.60 --resample-logw-clip-q 0.995
  --vmc-every 20 --vmc-n 2000 --n-eval 0
  --save-best-window 10
  --print-every 5
  --max-epoch-seconds 25
  --cusp-len-mode legacy
  --no-pretrained
)

launch_ab() {
  need_file "$N20_W001_INIT"
  need_file "$N12_W001_RESUME"

  local a_epochs="${A_EPOCHS:-80}"
  local b_epochs="${B_EPOCHS:-80}"

  common_a=(
    --mode jastrow --n-elec 20 --omega 0.01 --e-dmc 6.14645
    --epochs "$a_epochs"
    --n-coll 512 --oversample 16 --micro-batch 128
    --clip-el 4.0 --reward-qtrim 0.01
    --lr 4e-5 --lr-jas 4e-6
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2
    --ess-floor-ratio 0.0
    --init-jas "$N20_W001_INIT"
    "${common_fast[@]}"
  )

  start_job "$(gpu_at 0)" "rapidA1_cap2_n20_w001" "A1_cap2" \
    "${common_a[@]}" --jas-input-radius-cap-aho 2.0 --tag rapidA1_cap2_n20_w001
  start_job "$(gpu_at 1)" "rapidA2_cap3_n20_w001" "A2_cap3" \
    "${common_a[@]}" --jas-input-radius-cap-aho 3.0 --tag rapidA2_cap3_n20_w001
  start_job "$(gpu_at 2)" "rapidA3_uncap_n20_w001" "A3_uncap" \
    "${common_a[@]}" --tag rapidA3_uncap_n20_w001

  common_b=(
    --mode bf --n-elec 12 --omega 0.01 --e-dmc 2.47363
    --epochs "$b_epochs"
    --micro-batch 256
    --clip-el 5.0 --reward-qtrim 0.02
    --lr 3e-5 --lr-jas 3e-6
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2
    --ess-floor-ratio 0.0
    --sigma-fs 0.8,1.3,2.0,3.5,6.0
    --resume "$N12_W001_RESUME"
    "${common_fast[@]}"
  )

  start_job "$(gpu_at 3)" "rapidB1_n12_w001_n1024_ov8" "B1_n1024_ov8" \
    "${common_b[@]}" --n-coll 1024 --oversample 8 --tag rapidB1_n12_w001_n1024_ov8
  start_job "$(gpu_at 4)" "rapidB2_n12_w001_n2048_ov8" "B2_n2048_ov8" \
    "${common_b[@]}" --n-coll 2048 --oversample 8 --tag rapidB2_n12_w001_n2048_ov8
  start_job "$(gpu_at 5)" "rapidB3_n12_w001_n2048_ov16" "B3_n2048_ov16" \
    "${common_b[@]}" --n-coll 2048 --oversample 16 --tag rapidB3_n12_w001_n2048_ov16
}

launch_cdf() {
  need_file "$N20_W01_INIT"
  need_file "$N20_W1_INIT"
  need_file "$N12_W01_INIT"
  need_file "$N12_W001_RESUME"

  local c_epochs="${C_EPOCHS:-80}"
  local d_epochs="${D_EPOCHS:-80}"
  local f_epochs="${F_EPOCHS:-80}"

  common_c=(
    --mode bf --n-elec 20 --omega 0.1 --e-dmc 29.9779
    --epochs "$c_epochs"
    --n-coll 512 --oversample 8 --micro-batch 128
    --clip-el 5.0 --reward-qtrim 0.01
    --lr 3e-5 --lr-jas 3e-6
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2
    --ess-floor-ratio 0.0
    --sigma-fs 0.8,1.3,2.0,3.5,6.0
    --init-jas "$N20_W01_INIT"
    "${common_fast[@]}"
  )

  start_job "$(gpu_at 0)" "rapidC1_n20_w01_tinybf32" "C1_n20_w01_tinybf32" \
    "${common_c[@]}" --bf-hidden 32 --bf-msg-hidden 32 --bf-layers 2 --tag rapidC1_n20_w01_tinybf32
  start_job "$(gpu_at 1)" "rapidC2_n20_w01_tinybf48" "C2_n20_w01_tinybf48" \
    "${common_c[@]}" --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2 --tag rapidC2_n20_w01_tinybf48

  start_job "$(gpu_at 2)" "rapidC0_n20_w1_tinybf32_sanity" "C0_n20_w1_tinybf32_sanity" \
    --mode bf --n-elec 20 --omega 1.0 --e-dmc 155.8822 \
    --epochs "$c_epochs" --n-coll 512 --oversample 8 --micro-batch 128 \
    --clip-el 5.0 --reward-qtrim 0.01 --lr 3e-5 --lr-jas 3e-6 \
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2 \
    --ess-floor-ratio 0.0 --init-jas "$N20_W1_INIT" \
    --bf-hidden 32 --bf-msg-hidden 32 --bf-layers 2 \
    "${common_fast[@]}" --tag rapidC0_n20_w1_tinybf32_sanity

  start_job "$(gpu_at 3)" "rapidD1_n16_w01_from_n12" "D1_n16_w01_from_n12" \
    --mode jastrow --n-elec 16 --omega 0.1 --allow-missing-dmc-ref \
    --epochs "$d_epochs" --n-coll 512 --oversample 8 --micro-batch 128 \
    --clip-el 5.0 --reward-qtrim 0.02 --lr 4e-5 --lr-jas 4e-6 \
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2 \
    --ess-floor-ratio 0.0 --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
    --init-jas "$N12_W01_INIT" \
    "${common_fast[@]}" --tag rapidD1_n16_w01_from_n12

  start_job "$(gpu_at 4)" "rapidF1_n12_w0001_postfix" "F1_n12_w0001_postfix" \
    --mode bf --n-elec 12 --omega 0.001 --e-dmc 0.515365 \
    --epochs "$f_epochs" --n-coll 1024 --oversample 8 --micro-batch 256 \
    --clip-el 5.0 --reward-qtrim 0.02 --lr 2e-5 --lr-jas 2e-6 \
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2 \
    --ess-floor-ratio 0.0 --resample-weight-temp 0.65 --resample-logw-clip-q 0.995 \
    --sigma-fs 0.3,0.5,0.8,1.4,2.5,4.5,8.0 \
    --resume "$N12_W001_RESUME" \
    "${common_fast[@]}" --tag rapidF1_n12_w0001_postfix

  if [[ -f "$N16_W01_RESUME" ]]; then
    start_job "$(gpu_at 5)" "rapidD2_n20_w01_from_n16" "D2_n20_w01_from_n16" \
      --mode jastrow --n-elec 20 --omega 0.1 --e-dmc 29.9779 \
      --epochs "$d_epochs" --n-coll 512 --oversample 8 --micro-batch 128 \
      --clip-el 5.0 --reward-qtrim 0.02 --lr 3e-5 --lr-jas 3e-6 \
      --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2 \
      --ess-floor-ratio 0.0 --sigma-fs 0.8,1.3,2.0,3.5,6.0 \
      --init-jas "$N16_W01_RESUME" \
      "${common_fast[@]}" --tag rapidD2_n20_w01_from_n16
  else
    echo "[$(date '+%F %T')] Skipping D2: N16_W01_RESUME not found ($N16_W01_RESUME)" | tee -a "${OUT_DIR}/launcher.log"
  fi
}

launch_e() {
  need_file "$N16_W01_RESUME"
  need_file "$N20_W005_RESUME"

  local e_epochs="${E_EPOCHS:-80}"
  start_job "$(gpu_at 0)" "rapidE2_n20_w001_from_E1" "E2_n20_w001_from_E1" \
    --mode jastrow --n-elec 20 --omega 0.01 --e-dmc 6.14645 \
    --epochs "$e_epochs" --n-coll 512 --oversample 16 --micro-batch 128 \
    --clip-el 4.0 --reward-qtrim 0.01 --lr 2e-5 --lr-jas 2e-6 \
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2 \
    --ess-floor-ratio 0.0 --resample-weight-temp 0.60 --resample-logw-clip-q 0.995 \
    --sigma-fs 0.3,0.6,1.0,1.8,3.2,6.0 \
    --resume "$N20_W005_RESUME" \
    "${common_fast[@]}" --tag rapidE2_n20_w001_from_E1
}

launch_d2_compare() {
  need_file "$N16_W01_RESUME"
  need_file "$N12_W01_INIT"

  local d_epochs="${D_EPOCHS:-60}"
  common_d2=(
    --mode jastrow --n-elec 20 --omega 0.1 --e-dmc 29.9779
    --epochs "$d_epochs" --n-coll 512 --oversample 8 --micro-batch 128
    --clip-el 5.0 --reward-qtrim 0.02 --lr 3e-5 --lr-jas 3e-6
    --lr-warmup-epochs 8 --lr-warmup-init-frac 0.2 --lr-min-frac 0.2
    --ess-floor-ratio 0.0 --sigma-fs 0.8,1.3,2.0,3.5,6.0
    "${common_fast[@]}"
  )

  start_job "$(gpu_at 0)" "rapidD2_n20_w01_from_n16" "D2_n20_w01_from_n16" \
    "${common_d2[@]}" --init-jas "$N16_W01_RESUME" --tag rapidD2_n20_w01_from_n16
  start_job "$(gpu_at 1)" "rapidD2b_n20_w01_direct_from_n12" "D2b_n20_w01_direct_from_n12" \
    "${common_d2[@]}" --init-jas "$N12_W01_INIT" --tag rapidD2b_n20_w01_direct_from_n12
}

echo "[$(date '+%F %T')] Launching rapid grand-plan suite=${SUITE} run=${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
echo "[$(date '+%F %T')] GPUs=${GPUS_CSV} WALL=${WALL}" | tee -a "${OUT_DIR}/launcher.log"

case "$SUITE" in
  A_B)
    launch_ab
    ;;
  C_D_F)
    launch_cdf
    ;;
  E)
    launch_e
    ;;
  D2_COMPARE)
    launch_d2_compare
    ;;
  ALL)
    launch_ab
    launch_cdf
    ;;
  *)
    echo "Unknown SUITE=${SUITE}; expected A_B, C_D_F, D2_COMPARE, E, or ALL" >&2
    exit 2
    ;;
esac

echo "[$(date '+%F %T')] Rapid launch complete (${#PIDS[@]} workers)" | tee -a "${OUT_DIR}/launcher.log"
echo "Output: ${OUT_DIR}"

if [[ "${WAIT_FOR_JOBS:-0}" == "1" ]]; then
  echo "[$(date '+%F %T')] Waiting for ${#PIDS[@]} rapid workers" | tee -a "${OUT_DIR}/launcher.log"
  rc=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  echo "[$(date '+%F %T')] Rapid workers finished rc=${rc}" | tee -a "${OUT_DIR}/launcher.log"
  python3 scripts/summarize_rapid_triage.py "$OUT_DIR" | tee "${OUT_DIR}/summary.txt" || true
  exit "$rc"
fi
