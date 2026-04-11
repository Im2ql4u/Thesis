#!/usr/bin/env bash
set -euo pipefail

# Phase 5 preparation queue (does not launch training yet).
# Produces deterministic run/eval manifest and ETA budget so we can launch
# immediately once Phase 4 gate is accepted.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="outputs/consistency_campaign/phase5"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$OUT_DIR/phase5_queue_manifest_${TS}.txt"
ETA_JSON="$OUT_DIR/phase5_eta_${TS}.json"

# Phase 5 target matrix from plan: 5 omegas x 3 seeds = 15 runs
OMEGAS=(1.0 0.5 0.1 0.01 0.001)
SEEDS=(42 11 77)

# Conservative budget based on N=6 observed throughput at n_coll=8192.
EPOCHS=900
SEC_PER_EPOCH=15.5
RUNS_TOTAL=15
GPUS_AVAILABLE=7

python3 - <<PY
import json, math
from datetime import datetime
omegas = [1.0, 0.5, 0.1, 0.01, 0.001]
seeds = [42, 11, 77]
epochs = 900
sec_per_epoch = 15.5
runs_total = len(omegas) * len(seeds)
gpus = 7
run_sec = epochs * sec_per_epoch
waves = math.ceil(runs_total / gpus)
wall_sec = waves * run_sec
payload = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "phase": "5",
    "runs_total": runs_total,
    "gpus_assumed": gpus,
    "epochs_per_run": epochs,
    "sec_per_epoch_assumed": sec_per_epoch,
    "per_run_hours": round(run_sec / 3600.0, 2),
    "waves": waves,
    "campaign_wall_hours": round(wall_sec / 3600.0, 2),
    "notes": "ETA excludes heavy-VMC post-eval and queue gaps",
}
with open("$ETA_JSON", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

{
  echo "# Phase 5 queued manifest ($TS)"
  echo "# Recipe placeholders follow Phase 4 gate decision"
  for om in "${OMEGAS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      echo "run_id=n6_w${om}_s${sd} omega=${om} seed=${sd} status=queued"
    done
  done
} > "$MANIFEST"

echo "Wrote: $MANIFEST"
echo "Wrote: $ETA_JSON"
