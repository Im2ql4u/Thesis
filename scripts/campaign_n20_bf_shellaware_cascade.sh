#!/usr/bin/env bash
# N=20 BF + Shell-Aware Jastrow cascade: ω=1.0 → 0.5 → 0.1
#
# Key combination never previously tried:
#   - CTNNShellAwareJastrow (--jas-arch shellaware) instead of vcycle
#   - Compact BF (--bf-hidden 48 --bf-layers 2, ~15k params, fits 11 GB)
#   - Weight tempering (--resample-weight-temp), decreasing with ω
#   - Standard adaptive GMM proposal (no ShellFlow — ESS is already OK post-bugfix)
#   - Cascade: each stage warm-starts from the previous best checkpoint
#
# Usage:
#   bash scripts/campaign_n20_bf_shellaware_cascade.sh
#   SEED=137 GPU=1 bash scripts/campaign_n20_bf_shellaware_cascade.sh
#
# Environment overrides:
#   SEED   — random seed (default 42)
#   GPU    — CUDA device index (default 0)
#   EPOCHS — epochs per cascade stage (default 3000)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

SEED="${SEED:-42}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-3000}"

RUN_ID="2026-05-14_n20_bf_shellaware_cascade_s${SEED}"
OUT_DIR="outputs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
CKPT_DIR="results/arch_colloc"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${OUT_DIR}/launcher.log"; }

# --------------------------------------------------------------------------
# Shared flags
# --------------------------------------------------------------------------
SHARED_ARGS=(
    --mode bf --n-elec 20
    --jas-arch shellaware --jas-hidden 48 --jas-shells 3
    --bf-hidden 48 --bf-msg-hidden 48 --bf-layers 2
    --epochs "$EPOCHS"
    --n-coll 2048 --oversample 16 --micro-batch 256
    --loss-type reinforce --direct-weight 0.0
    --clip-el 4.5 --reward-qtrim 0.02
    --resample-logw-clip-q 0.02
    --ess-floor-ratio 0.04 --ess-oversample-max 32 --ess-oversample-step 2
    --rollback-decay 0.94 --rollback-jump-sigma 4.0
    --vmc-every 40 --vmc-n 20000
    --n-eval 50000
    --seed "$SEED"
)

# --------------------------------------------------------------------------
# Stage helper: run one omega stage, return path to best checkpoint
# --------------------------------------------------------------------------
run_stage() {
    local omega="$1"
    local lr="$2"
    local lr_jas="$3"
    local wtemp="$4"
    local resume_flag=("${@:5}")   # optional --resume <path>

    local omega_tag
    omega_tag="$(echo "$omega" | sed 's/\./p/g; s/-/m/g')"
    local tag="n20_shellaware_bf_w${omega_tag}_s${SEED}"
    local logfile="${LOG_DIR}/${tag}.log"
    local best_ckpt="${CKPT_DIR}/${tag}_best.pt"

    log "Stage START  omega=${omega} lr=${lr} lr_jas=${lr_jas} wtemp=${wtemp} seed=${SEED}"

    CUDA_MANUAL_DEVICE="$GPU" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py \
        "${SHARED_ARGS[@]}" \
        --omega "$omega" \
        --lr "$lr" --lr-jas "$lr_jas" \
        --resample-weight-temp "$wtemp" \
        --tag "$tag" \
        "${resume_flag[@]}" \
        2>&1 | tee -a "$logfile"

    local rc=${PIPESTATUS[0]}
    log "Stage END    omega=${omega} rc=${rc}"

    if [[ ! -f "$best_ckpt" ]]; then
        log "WARNING: expected best checkpoint not found: $best_ckpt"
        # Fallback: look for any matching _best.pt in CKPT_DIR
        best_ckpt="$(ls -t "${CKPT_DIR}/${tag}"*.pt 2>/dev/null | head -1)"
        log "Fallback checkpoint: $best_ckpt"
    fi

    echo "$best_ckpt"
}

# --------------------------------------------------------------------------
# Cascade
# --------------------------------------------------------------------------
log "=== N=20 BF ShellAware Cascade: seed=${SEED} gpu=${GPU} ==="

# Stage 1 — ω=1.0 — fresh start, moderate tempering
log "--- Stage 1: ω=1.0 (fresh start) ---"
ckpt_w1=$(run_stage 1.0 5e-5 6e-6 0.80)
log "Stage 1 best checkpoint: $ckpt_w1"

# Stage 2 — ω=0.5 — warm-start from ω=1.0
log "--- Stage 2: ω=0.5 (warm-start from ω=1.0) ---"
if [[ -f "$ckpt_w1" ]]; then
    ckpt_w05=$(run_stage 0.5 3e-5 4e-6 0.75 --resume "$ckpt_w1")
else
    log "WARNING: Stage 1 checkpoint missing, starting Stage 2 from scratch"
    ckpt_w05=$(run_stage 0.5 3e-5 4e-6 0.75)
fi
log "Stage 2 best checkpoint: $ckpt_w05"

# Stage 3 — ω=0.1 — warm-start from ω=0.5
log "--- Stage 3: ω=0.1 (warm-start from ω=0.5) ---"
if [[ -f "$ckpt_w05" ]]; then
    ckpt_w01=$(run_stage 0.1 2e-5 3e-6 0.70 --resume "$ckpt_w05")
else
    log "WARNING: Stage 2 checkpoint missing, starting Stage 3 from scratch"
    ckpt_w01=$(run_stage 0.1 2e-5 3e-6 0.70)
fi
log "Stage 3 best checkpoint: $ckpt_w01"

log "=== Cascade complete. Final checkpoint: $ckpt_w01 ==="

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
python3.11 - <<'PYEOF'
import json, sys, os
from pathlib import Path

ROOT = Path(os.environ.get("PWD", "."))
seed = os.environ.get("SEED", "42")
run_id = f"2026-05-14_n20_bf_shellaware_cascade_s{seed}"
log_dir = ROOT / "outputs" / run_id / "logs"

summary = {"seed": int(seed), "results": []}
for omega_tag, omega in [("1p0", 1.0), ("0p5", 0.5), ("0p1", 0.1)]:
    tag = f"n20_shellaware_bf_w{omega_tag}_s{seed}"
    logfile = log_dir / f"{tag}.log"
    if logfile.exists():
        # Extract last VMC energy line
        lines = logfile.read_text().splitlines()
        for line in reversed(lines):
            if "vmc_energy" in line.lower() or "e_vmc" in line.lower():
                summary["results"].append({"omega": omega, "log_line": line.strip()})
                break

out = ROOT / "outputs" / run_id / "cascade_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"Summary written to {out}")
PYEOF
