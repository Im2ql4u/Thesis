#!/usr/bin/env bash
set -euo pipefail

# Phase 6 preparation queue.
# Produces a deterministic manifest and ETA budget for the N=12 transfer phase.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="outputs/consistency_campaign/phase6"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="$OUT_DIR/phase6_queue_manifest_${TS}.txt"
ETA_JSON="$OUT_DIR/phase6_eta_${TS}.json"

python3 - <<PY
import json, math
from datetime import datetime
omegas = [1.0, 0.5, 0.1]
seeds = [42, 11, 77]
epochs = 1200
sec_per_epoch = 28.0
gpus = 7
runs_total = len(omegas) * len(seeds)
run_sec = epochs * sec_per_epoch
waves = math.ceil(runs_total / gpus)
wall_sec = waves * run_sec
payload = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "phase": "6",
    "runs_total": runs_total,
    "omegas": omegas,
    "seeds": seeds,
    "epochs_per_run": epochs,
    "sec_per_epoch_assumed": sec_per_epoch,
    "per_run_hours": round(run_sec / 3600.0, 2),
    "waves": waves,
    "campaign_wall_hours": round(wall_sec / 3600.0, 2),
    "notes": "ETA assumes N=12 transfer runs after Phase 5 gate acceptance",
}
with open("$ETA_JSON", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

{
  echo "# Phase 6 queued manifest ($TS)"
  echo "# Assumes Phase 5 consolidates an N=6 winner suitable for N=12 transfer"
  for om in 1.0 0.5 0.1; do
    for sd in 42 11 77; do
      echo "run_id=n12_w${om}_s${sd} omega=${om} seed=${sd} status=queued"
    done
  done
} > "$MANIFEST"

echo "Wrote: $MANIFEST"
echo "Wrote: $ETA_JSON"