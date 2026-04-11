#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="outputs/consistency_campaign/phase1"
mkdir -p "$OUT_DIR"

launch_job() {
  local gpu="$1"
  local tag="$2"
  shift 2

  local launcher_log="$OUT_DIR/${tag}_launcher.log"
  echo "[$(date '+%F %T')] Launching $tag on GPU $gpu" | tee -a "$launcher_log" >&2

  CUDA_MANUAL_DEVICE="$gpu" \
    python3 scripts/instrumented_run.py \
      --tag "$tag" \
      --output-dir "$OUT_DIR" \
      --run-weak-form-args "$@" \
    >> "$launcher_log" 2>&1 &

  local pid="$!"
  LAST_PID="$pid"
}

PIDS=()
LAST_PID=""

launch_job 0 repro_n6w1_s42 --mode bf --n-elec 6 --omega 1.0 --seed 42 --epochs 800 --n-coll 4096 --micro-batch 512 --lr 5e-4 --resume results/arch_colloc/bf_ctnn_vcycle.pt --vmc-every 50
PIDS+=("$LAST_PID")
launch_job 1 repro_n6w1_s11 --mode bf --n-elec 6 --omega 1.0 --seed 11 --epochs 800 --n-coll 4096 --micro-batch 512 --lr 5e-4 --resume results/arch_colloc/bf_ctnn_vcycle.pt --vmc-every 50
PIDS+=("$LAST_PID")
launch_job 2 repro_n6w1_s77 --mode bf --n-elec 6 --omega 1.0 --seed 77 --epochs 800 --n-coll 4096 --micro-batch 512 --lr 5e-4 --resume results/arch_colloc/bf_ctnn_vcycle.pt --vmc-every 50
PIDS+=("$LAST_PID")

launch_job 3 repro_n12w1_s42 --mode bf --n-elec 12 --omega 1.0 --seed 42 --epochs 800 --n-coll 4096 --micro-batch 256 --lr 5e-4 --resume results/curated_low_error_0p1pct_2026-03-21/v7_n12w1_continue.pt --vmc-every 50
PIDS+=("$LAST_PID")
launch_job 4 repro_n12w1_s11 --mode bf --n-elec 12 --omega 1.0 --seed 11 --epochs 800 --n-coll 4096 --micro-batch 256 --lr 5e-4 --resume results/curated_low_error_0p1pct_2026-03-21/w1_n12w1_xfer.pt --vmc-every 50
PIDS+=("$LAST_PID")

launch_job 5 repro_n6w05_s42 --mode bf --n-elec 6 --omega 0.5 --seed 42 --epochs 800 --n-coll 4096 --micro-batch 512 --lr 5e-4 --resume results/curated_low_error_0p1pct_2026-03-21/w1_n6w05_hisamp.pt --vmc-every 50
PIDS+=("$LAST_PID")
launch_job 6 repro_n6w05_s11 --mode bf --n-elec 6 --omega 0.5 --seed 11 --epochs 800 --n-coll 4096 --micro-batch 512 --lr 5e-4 --resume results/curated_low_error_0p1pct_2026-03-21/w1_n6w05_hisamp.pt --vmc-every 50
PIDS+=("$LAST_PID")

launch_job 7 repro_n12w05_s42 --mode bf --n-elec 12 --omega 0.5 --seed 42 --epochs 800 --n-coll 4096 --micro-batch 256 --lr 5e-4 --resume results/curated_low_error_0p1pct_2026-03-21/n12w05_cascade.pt --vmc-every 50
PIDS+=("$LAST_PID")

{
  echo "[$(date '+%F %T')] Phase 1 launched"
  for pid in "${PIDS[@]}"; do
    echo "PID=$pid"
  done
} | tee "$OUT_DIR/phase1_pids.txt"

rc=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

echo "[$(date '+%F %T')] Phase 1 completed with rc=$rc" | tee -a "$OUT_DIR/phase1_pids.txt"
exit "$rc"
