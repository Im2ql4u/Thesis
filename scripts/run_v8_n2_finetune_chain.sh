#!/bin/bash
# Campaign v8: N=2 Sub-0.01% Fine-Tuning Chain

set -e

# Load PyTorch module
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true

REPO_DIR="/itf-fi-ml/home/aleksns/Thesis_repo"
cd "$REPO_DIR"

GPU=0
BUDGET_S=25200  # 7 hours total for chain (account for slower fine-tuning)

log_msg() {
    echo "[$(date '+%H:%M:%S')] $@"
}

# Helper: run n=2 job with timeout
run_n2_job() {
    local omega=$1
    local epochs=$2
    local resume_pt=$3
    local lr=${4:-0.0005}      # Default: lower lr for fine-tuning
    local rollback_err=${5:-2.0}
    local rollback_jump=${6:-3.5}
    
    # Tag mapping: 1.0->w1, 0.5->w05, 0.1->w01, 0.01->w001, 0.001->w0001
    local tag_suffix=""
    case "$omega" in
        1.0) tag_suffix="1" ;;
        0.5) tag_suffix="05" ;;
        0.1) tag_suffix="01" ;;
        0.01) tag_suffix="001" ;;
        0.001) tag_suffix="0001" ;;
        *) tag_suffix="unknown" ;;
    esac
    local tag="v8_n2w${tag_suffix}_finetune"
    local log_file="outputs/2026-03-21_1151_campaign_v7_n2_exact_n12_continue/logs/${tag}.log"
    
    log_msg "Starting N=2 ω=$omega (epochs=$epochs, lr=$lr, resume=$resume_pt)"
    
    timeout $BUDGET_S python3 src/run_weak_form.py \
        --n-elec 2 \
        --omega "$omega" \
        --mode bf \
        --resume "$resume_pt" \
        --epochs "$epochs" \
        --lr "$lr" \
        --lr-jas "$lr" \
        --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
        --rollback-decay 0.97 \
        --rollback-err-pct "$rollback_err" \
        --rollback-jump-sigma "$rollback_jump" \
        --oversample 25 \
        --seed 42 \
        --tag "$tag" \
        2>&1 | tee "$log_file" || {
        rc=$?
        log_msg "⚠ N=2 ω=$omega job exited with rc=$rc"
        # Continue chain even if one job fails
    }
    
    # Sanity check: do final energies exist?
    if grep -q "^  E = " "$log_file" 2>/dev/null; then
        final_e=$(grep "^  E = " "$log_file" | tail -1)
        log_msg "✓ N=2 ω=$omega final: $final_e"
    else
        log_msg "⚠ No final energy found for ω=$omega"
    fi
}

log_msg "════════════════════════════════════════════════════════════"
log_msg "N=2 Sub-0.01% Fine-Tuning Chain (v8)"
log_msg "GPU=$GPU  Total budget=$BUDGET_S sec (~7h)"
log_msg "════════════════════════════════════════════════════════════"

# Chain: ω=1.0, 0.5, 0.1, 0.01, 0.001
# Each uses v7 checkpoint as warm-start, reduced lr for fine-tuning

log_msg ""
log_msg "▶ ω=1.0  (v7 result: +0.024%, target: <0.01%)"
run_n2_job 1.0 500 results/arch_colloc/v7_n2w1_exact.pt 0.0005 2.0 3.5
# Already near exact; just a kiss of fine-tuning

log_msg ""
log_msg "▶ ω=0.5  (v7 result: +0.129%, target: <0.01%)"
run_n2_job 0.5 1000 results/arch_colloc/v7_n2w05_exact.pt 0.0005 2.0 3.5
# Needs moderate fine-tuning

log_msg ""
log_msg "▶ ω=0.1  (v7 result: +0.004%, target: <0.01%)"
run_n2_job 0.1 1500 results/arch_colloc/v7_n2w01_exact.pt 0.0005 0.0 4.0
# Excellent already; extended epochs for stability at low-omega

log_msg ""
log_msg "▶ ω=0.01 (v7 result: +0.032%, target: <0.01%)"
run_n2_job 0.01 2000 results/arch_colloc/v7_n2w001_exact.pt 0.0005 0.0 4.5
# Needs significant fine-tuning; use jump-based rollback

log_msg ""
log_msg "▶ ω=0.001 (v7 result: +88.8%, CRITICAL)"
log_msg "  Strategy: Cascade from ω=0.01 final checkpoint + extended epochs"
log_msg "  Special settings: Aggressive ESS sampling, conservative updates"

# After ω=0.01 completes, cascade to 0.001
# Use final checkpoint from ω=0.01 as init
cascade_init="results/arch_colloc/v8_n2w001_finetune.pt"
run_n2_job 0.001 3000 "$cascade_init" 0.0003 0.0 5.0
# Ultra-low learning rate, jump-based only, extended epochs

log_msg ""
log_msg "════════════════════════════════════════════════════════════"
log_msg "✓ N=2 chain complete"
log_msg "════════════════════════════════════════════════════════════"
