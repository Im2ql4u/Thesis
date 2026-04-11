#!/usr/bin/env bash
set -euo pipefail

# Phase 5 Step 3.1 bridge probe for omega=0.01.
# The plan explicitly allows a quick omega=0.01 test when the low-omega winner
# may differ between 0.001 and 0.01. This script compares the two plausible
# continuations before committing the full 15-run Phase 5 matrix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /etc/profile.d/z00_lmod.sh
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

OUT_DIR="outputs/consistency_campaign/phase5_probe"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG_FD="p5probe_fd_n6w001_s42"
TAG_NG="p5probe_ng_n6w001_s42"

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

LAST_PID=0

launch_job 1 "$TAG_FD" \
  --mode bf --resume results/arch_colloc/p4m1_fdmatched_n6w01.pt \
  --n-elec 6 --omega 0.01 --seed 42 \
  --loss-type fd-colloc --n-coll 8192 --oversample 8 \
  --epochs 300 --lr 1e-4 --lr-jas 1e-5 \
  --lr-warmup-epochs 60 --lr-warmup-init-frac 0.05 --lr-min-frac 0.005 \
  --fd-h 0.005 --fd-huber-delta 0.5 --prox-mu 0.1 \
  --micro-batch 1024 --patience 300 --vmc-every 50 --vmc-n 10000 \
  --save-best-window 20
PID_FD="$LAST_PID"

launch_job 2 "$TAG_NG" \
  --mode bf --resume results/arch_colloc/p4m2_nogate_n6w001.pt \
  --n-elec 6 --omega 0.01 --seed 42 \
  --loss-type reinforce --n-coll 8192 --oversample 64 \
  --epochs 300 --lr 5e-4 --lr-jas 5e-5 \
  --lr-warmup-epochs 60 --lr-min-frac 0.02 \
  --micro-batch 1024 --patience 300 --vmc-every 50 --vmc-n 10000 \
  --save-best-window 20
PID_NG="$LAST_PID"

cat <<EOF
Launched Phase 5 omega=0.01 bridge probe:
  $TAG_FD GPU=1 PID=$PID_FD
  $TAG_NG GPU=2 PID=$PID_NG
Expected runtime per run: ~1.3h (300 epochs x ~15.5s)
Logs: $OUT_DIR
EOF

RC=0
for pid in "$PID_FD" "$PID_NG"; do
  if ! wait "$pid"; then
    RC=1
  fi
done

echo "Phase 5 omega=0.01 bridge probe finished with rc=$RC"
exit "$RC"
