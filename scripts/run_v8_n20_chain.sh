#!/bin/bash
# Campaign v8: N=20 BF Training Chain

set -e

# Load PyTorch module
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true

REPO_DIR="/itf-fi-ml/home/aleksns/Thesis_repo"
cd "$REPO_DIR"

GPU=5
BUDGET_S=21600  # 6 hours total (after ω=1.0 finishes on GPU 4)

log_msg() {
    echo "[$(date '+%H:%M:%S')] $@"
}

run_n20_job() {
    local omega=$1
    local epochs=$2
    local resume_pt=${3:-""}  # Optional resume
    local lr=${4:-0.001}
    local rollback_err=${5:-1.0}
    
    # Tag mapping
    local tag_suffix=""
    case "$omega" in
        1.0) tag_suffix="1" ;;
        0.5) tag_suffix="05" ;;
        0.1) tag_suffix="01" ;;
        0.01) tag_suffix="001" ;;
        0.001) tag_suffix="0001" ;;
        *) tag_suffix="unknown" ;;
    esac
    local tag="v8_n20w${tag_suffix}_bf"
    
    if [ -z "$resume_pt" ]; then
        # Fresh training
        log_msg "Starting N=20 ω=$omega FRESH (epochs=$epochs, lr=$lr)"
        timeout $BUDGET_S python3 src/run_weak_form.py \
            --n-elec 20 \
            --omega "$omega" \
            --mode bf \
            --epochs "$epochs" \
            --lr "$lr" \
            --lr-jas "$lr" \
            --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
            --rollback-decay 0.96 \
            --rollback-err-pct "$rollback_err" \
            --rollback-jump-sigma 3.5 \
            --seed 11 \
            --tag "$tag" \
            2>&1 | tee "outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/${tag}.log" || {
            rc=$?
            log_msg "⚠ N=20 ω=$omega job exited with rc=$rc"
        }
    else
        # Resume from checkpoint
        log_msg "Starting N=20 ω=$omega RESUME from $resume_pt (epochs=$epochs, lr=$lr)"
        timeout $BUDGET_S python3 src/run_weak_form.py \
            --n-elec 20 \
            --omega "$omega" \
            --mode bf \
            --resume "$resume_pt" \
            --epochs "$epochs" \
            --lr "$lr" \
            --lr-jas "$lr" \
            --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
            --rollback-decay 0.96 \
            --rollback-err-pct "$rollback_err" \
            --rollback-jump-sigma 4.0 \
            --seed 11 \
            --tag "$tag" \
            2>&1 | tee "outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/${tag}.log" || {
            rc=$?
            log_msg "⚠ N=20 ω=$omega job exited with rc=$rc"
        }
    fi
    
    # Check for final result
    if grep -q "^  E = " "outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/${tag}.log" 2>/dev/null; then
        final_e=$(grep "^  E = " "outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/${tag}.log" | tail -1)
        log_msg "✓ N=20 ω=$omega final: $final_e"
    fi
}

log_msg "════════════════════════════════════════════════════════════"
log_msg "N=20 BF Training Chain (v8)"
log_msg "GPU=$GPU  Total budget=$BUDGET_S sec (~6h)"
log_msg "════════════════════════════════════════════════════════════"

# Wait a bit for GPU 4 (ω=1.0) to start and warm up before we begin
sleep 30

log_msg ""
log_msg "▶ ω=0.5  (fresh start)"
run_n20_job 0.5 2000 "" 0.001 0.8

log_msg ""
log_msg "▶ ω=0.1  (chain from ω=0.5 checkpoint)"
# Get the checkpoint from ω=0.5 final result
resume_ckpt="results/arch_colloc/v8_n20w05_bf.pt"
run_n20_job 0.1 2500 "$resume_ckpt" 0.0008 0.0

log_msg ""
log_msg "════════════════════════════════════════════════════════════"
log_msg "✓ N=20 chain complete"
log_msg "════════════════════════════════════════════════════════════"
