#!/usr/bin/env bash
set -euo pipefail

# Phase 5 — N=6 full grid consolidation.
# Usage:
#   LOW_OMEGA_RECIPE=nogate bash scripts/launch_consistency_phase5.sh
# or
#   LOW_OMEGA_RECIPE=fd bash scripts/launch_consistency_phase5.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

LOW_OMEGA_RECIPE="${LOW_OMEGA_RECIPE:-nogate}"
if [[ "$LOW_OMEGA_RECIPE" != "nogate" && "$LOW_OMEGA_RECIPE" != "fd" ]]; then
  echo "LOW_OMEGA_RECIPE must be 'nogate' or 'fd'" >&2
  exit 2
fi

OUT_DIR="outputs/consistency_campaign/phase5"
mkdir -p "$OUT_DIR"

launch_job() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local launcher_log="$OUT_DIR/${tag}_launcher.log"
  echo "[$(date '+%F %T')] Launching $tag on GPU $gpu" | tee -a "$launcher_log"
  CUDA_MANUAL_DEVICE="$gpu" \
    python3 scripts/instrumented_run.py \
      --tag "$tag" \
      --output-dir "$OUT_DIR" \
      --run-weak-form-args "$@" \
    >> "$launcher_log" 2>&1 &
  LAST_PID=$!
}

run_easy() {
  local gpu="$1"
  local tag="$2"
  local omega="$3"
  local seed="$4"
  local resume="$5"
  launch_job "$gpu" "$tag" \
    --mode bf --resume "$resume" --n-elec 6 --omega "$omega" --seed "$seed" \
    --epochs 800 --n-coll 4096 --oversample 8 --micro-batch 512 --lr 5e-4 \
    --patience 300 --vmc-every 50 --vmc-n 10000 --save-best-window 20
}

run_w01_fd() {
  local gpu="$1"
  local tag="$2"
  local seed="$3"
  launch_job "$gpu" "$tag" \
    --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
    --n-elec 6 --omega 0.1 --seed "$seed" \
    --loss-type fd-colloc --n-coll 8192 --oversample 8 \
    --epochs 900 --lr 1e-4 --lr-jas 1e-5 \
    --lr-warmup-epochs 120 --lr-warmup-init-frac 0.05 --lr-min-frac 0.005 \
    --fd-h 0.005 --fd-huber-delta 0.5 --prox-mu 0.1 \
    --micro-batch 1024 --patience 900 --vmc-every 50 --vmc-n 10000 \
    --save-best-window 20
}

run_low_band() {
  local gpu="$1"
  local tag="$2"
  local omega="$3"
  local seed="$4"
  if [[ "$LOW_OMEGA_RECIPE" == "nogate" ]]; then
    launch_job "$gpu" "$tag" \
      --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001.pt \
      --n-elec 6 --omega "$omega" --seed "$seed" \
      --loss-type reinforce --n-coll 8192 --oversample 64 \
      --epochs 900 --lr 5e-4 --lr-jas 5e-5 \
      --lr-warmup-epochs 80 --lr-min-frac 0.02 \
      --micro-batch 1024 --patience 900 --vmc-every 50 --vmc-n 10000 \
      --save-best-window 20
  else
    launch_job "$gpu" "$tag" \
      --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
      --n-elec 6 --omega "$omega" --seed "$seed" \
      --loss-type fd-colloc --n-coll 8192 --oversample 8 \
      --epochs 900 --lr 1e-4 --lr-jas 1e-5 \
      --lr-warmup-epochs 120 --lr-warmup-init-frac 0.05 --lr-min-frac 0.005 \
      --fd-h 0.005 --fd-huber-delta 0.5 --prox-mu 0.1 \
      --micro-batch 1024 --patience 900 --vmc-every 50 --vmc-n 10000 \
      --save-best-window 20
  fi
}

LAST_PID=0
PIDS=()

# Wave 1
run_easy 1 p5_n6w1_s42 1.0 42 results/curated_low_error_0p1pct_2026-03-21/camp_n6w1_verify.pt
PIDS+=("$LAST_PID")
run_easy 2 p5_n6w1_s11 1.0 11 results/curated_low_error_0p1pct_2026-03-21/camp_n6w1_verify.pt
PIDS+=("$LAST_PID")
run_easy 3 p5_n6w1_s77 1.0 77 results/curated_low_error_0p1pct_2026-03-21/camp_n6w1_verify.pt
PIDS+=("$LAST_PID")
run_easy 4 p5_n6w05_s42 0.5 42 results/curated_low_error_0p1pct_2026-03-21/w1_n6w05_hisamp.pt
PIDS+=("$LAST_PID")
run_easy 5 p5_n6w05_s11 0.5 11 results/curated_low_error_0p1pct_2026-03-21/w1_n6w05_hisamp.pt
PIDS+=("$LAST_PID")
run_easy 6 p5_n6w05_s77 0.5 77 results/curated_low_error_0p1pct_2026-03-21/w1_n6w05_hisamp.pt
PIDS+=("$LAST_PID")
run_w01_fd 7 p5_n6w01_s42 42
PIDS+=("$LAST_PID")

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

PIDS=()

# Wave 2
run_w01_fd 1 p5_n6w01_s11 11
PIDS+=("$LAST_PID")
run_w01_fd 2 p5_n6w01_s77 77
PIDS+=("$LAST_PID")
run_low_band 3 p5_n6w001_s42 0.01 42
PIDS+=("$LAST_PID")
run_low_band 4 p5_n6w001_s11 0.01 11
PIDS+=("$LAST_PID")
run_low_band 5 p5_n6w001_s77 0.01 77
PIDS+=("$LAST_PID")
run_low_band 6 p5_n6w0001_s42 0.001 42
PIDS+=("$LAST_PID")
run_low_band 7 p5_n6w0001_s11 0.001 11
PIDS+=("$LAST_PID")

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

run_low_band 1 p5_n6w0001_s77 0.001 77

cat <<EOF
Phase 5 launch complete under LOW_OMEGA_RECIPE=$LOW_OMEGA_RECIPE.
Final outstanding PID: $LAST_PID
Output dir: $OUT_DIR
EOF

wait "$LAST_PID"
