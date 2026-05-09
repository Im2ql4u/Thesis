#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if [[ "${ALLOW_LONG_STAGE1:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
This launcher starts completion-scale A+B jobs and has already proven too slow
for diagnosis on the current GPUs.

Use scripts/launch_2026_04_26_grand_plan_rapid_triage.sh first. Re-run this
script only after a rapid result earns promotion, with ALLOW_LONG_STAGE1=1.
EOF
  exit 2
fi

RUN_ID="${RUN_ID:-2026-04-26_grand_plan_stage1_A_B_v4}"
OUT_DIR="${ROOT}/outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
CMD_DIR="${OUT_DIR}/cmds"
MPL_BASE="/tmp/matplotlib-${RUN_ID}"
# The 3000-epoch B jobs run much longer than the plan's optimistic wall-clock
# estimate on the available RTX 2080 Ti GPUs, so default to a completion budget.
WALL="${WALL:-200000s}"
GPUS_CSV="${GPUS_CSV:-1,2,3,5,6,7}"

# The plan references v5_n12w001_bf_cascade.pt, but that checkpoint is not
# currently present in this checkout. Continue from the strongest available
# N=12, omega=0.01 checkpoint instead.
N12_RESUME="${N12_RESUME:-results/arch_colloc/n12_lowomega_uncap_long_s42_best.pt}"

mkdir -p "$LOG_DIR" "$CMD_DIR" "$MPL_BASE"
PIDS=()

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -lt 6 ]]; then
  echo "Need 6 GPUs for stage 1 A+B, got: ${GPUS_CSV}" >&2
  exit 2
fi

if [[ ! -f "$N12_RESUME" ]]; then
  echo "Missing N=12 resume checkpoint: $N12_RESUME" >&2
  exit 2
fi

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

common_a=(
  --mode jastrow --n-elec 20 --omega 0.01 --e-dmc 6.14645
  --seed 42
  --epochs 400
  --n-coll 2048 --oversample 64 --micro-batch 256
  --loss-type reinforce --direct-weight 0.0
  --clip-el 4.0 --grad-clip 1.0 --reward-qtrim 0.01
  --lr 8e-5 --lr-jas 8e-6
  --lr-warmup-epochs 20 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01
  --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0
  --ess-floor-ratio 0.01 --ess-oversample-max 256 --ess-oversample-step 32
  --resample-weight-temp 0.60 --resample-logw-clip-q 0.995
  --vmc-every 50 --vmc-n 15000 --n-eval 20000
  --save-best-window 30
  --no-pretrained
)

common_b=(
  --mode bf --n-elec 12 --omega 0.01 --e-dmc 2.47363
  --seed 42
  --epochs 3000
  --n-coll 8192 --oversample 16 --micro-batch 512
  --loss-type reinforce --direct-weight 0.0
  --clip-el 5.0 --grad-clip 1.0 --reward-qtrim 0.02
  --lr 1.2e-4 --lr-jas 1.2e-5
  --lr-warmup-epochs 60 --lr-warmup-init-frac 0.1 --lr-min-frac 0.01
  --ess-floor-ratio 0.03 --ess-oversample-max 32 --ess-oversample-step 4
  --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 4.5
  --vmc-every 100 --vmc-n 20000 --n-eval 60000
  --sigma-fs 0.8,1.3,2.0,3.5,6.0
  --resume "$N12_RESUME"
  --save-best-window 30
  --no-pretrained
)

echo "[$(date '+%F %T')] Launching stage 1 A+B: ${RUN_ID}" | tee -a "${OUT_DIR}/launcher.log"
echo "[$(date '+%F %T')] GPUs=${GPUS_CSV} N12_RESUME=${N12_RESUME} WALL=${WALL}" | tee -a "${OUT_DIR}/launcher.log"

start_job "${GPUS[0]}" "expA1_cap2_n20_w001" "A1_cap2" \
  "${common_a[@]}" \
  --jas-input-radius-cap-aho 2.0 \
  --tag expA1_cap2_n20_w001

start_job "${GPUS[1]}" "expA2_cap3_n20_w001" "A2_cap3" \
  "${common_a[@]}" \
  --jas-input-radius-cap-aho 3.0 \
  --tag expA2_cap3_n20_w001

start_job "${GPUS[2]}" "expA3_uncap_n20_w001" "A3_uncap" \
  "${common_a[@]}" \
  --tag expA3_uncap_n20_w001

start_job "${GPUS[3]}" "expB1_n12_w001_hisamp_cap16" "B1_hisamp_cap16" \
  "${common_b[@]}" \
  --tag expB1_n12_w001_hisamp_cap16

start_job "${GPUS[4]}" "expB2_n12_w001_hisamp_uncap" "B2_hisamp_uncap" \
  "${common_b[@]}" \
  --oversample 32 \
  --ess-oversample-max 64 \
  --tag expB2_n12_w001_hisamp_uncap

start_job "${GPUS[5]}" "expB3_n12_w001_hisamp_maxpressure" "B3_hisamp_maxpressure" \
  "${common_b[@]}" \
  --oversample 48 --n-coll 6144 \
  --ess-oversample-max 96 \
  --tag expB3_n12_w001_hisamp_maxpressure

echo "[$(date '+%F %T')] Stage 1 launch complete" | tee -a "${OUT_DIR}/launcher.log"
echo "Output: ${OUT_DIR}"

if [[ "${WAIT_FOR_JOBS:-0}" == "1" ]]; then
  echo "[$(date '+%F %T')] Waiting for ${#PIDS[@]} stage-one workers" | tee -a "${OUT_DIR}/launcher.log"
  rc=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  echo "[$(date '+%F %T')] Stage 1 workers finished rc=${rc}" | tee -a "${OUT_DIR}/launcher.log"
  exit "$rc"
fi
