#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Campaign v12b: PURE REINFORCE, no SR, corrected checkpoints
#  ─────────────────────────────────────────────────────────────
#  Fixed version with correct checkpoint paths
# ═══════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v12b_reinforce_only"
LOGDIR="${OUT}/logs"
CURATED="${ROOT}/results/curated_low_error_0p1pct_2026-03-21"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'

mkdir -p "${LOGDIR}"

echo "=== Campaign v12b: Pure REINFORCE with corrected checkpoints ==="
echo "Output directory: ${OUT}"
echo ""

# ───────────────────────────────────────────────────────────────
# PART A: N=2 LOW OMEGA – transfer curriculum (REINFORCE only)
# ───────────────────────────────────────────────────────────────

echo "Starting N=2 low-omega transfer curriculum (REINFORCE)..."

# Stage 1: Polish ω=0.01
tmux kill-session -t v12b_n2w001_stage1 2>/dev/null || true
tmux new-session -d -s v12b_n2w001_stage1 -c "${ROOT}"
tmux send-keys -t v12b_n2w001_stage1 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=0; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 2 --omega 0.01 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 20000 --n-coll 4096 --oversample 36 --micro-batch 1024 \
     --lr 5e-6 --lr-jas 5e-6 --direct-weight 0.0 \
     --clip-el 4.0 --reward-qtrim 0.002 \
     --rollback-decay 0.99 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
     --vmc-every 120 --vmc-n 40000 --n-eval 200000 --seed 1101 \
     --resume '${CURATED}'/v7_n2w001_exact.pt --no-pretrained \
     --tag v12b_n2w001_stage1 \
     >> '${LOGDIR}'/v12b_n2w001_stage1.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12b_n2w001_stage1.log" C-m
echo "Stage 1 (ω=0.01): GPU 0, tmux: v12b_n2w001_stage1"

# ───────────────────────────────────────────────────────────────
# PART B: N=2 HIGH OMEGAS (1.0, 0.5, 0.1) – REINFORCE
# ───────────────────────────────────────────────────────────────

echo "Starting N=2 high-omega runs (pure REINFORCE)..."

# N=2 ω=1.0
tmux kill-session -t v12b_n2w1 2>/dev/null || true
tmux new-session -d -s v12b_n2w1 -c "${ROOT}"
tmux send-keys -t v12b_n2w1 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=1; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 2 --omega 1.0 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 3000 --n-coll 4096 --oversample 8 --micro-batch 1024 \
     --lr 2e-4 --lr-jas 2e-4 \
     --clip-el 4.0 --reward-qtrim 0.001 \
     --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
     --vmc-every 30 --vmc-n 40000 --n-eval 200000 --seed 401 \
     --resume '${CURATED}'/v7_n2w1_exact.pt --no-pretrained \
     --tag v12b_n2w1 \
     >> '${LOGDIR}'/v12b_n2w1.log 2>&1" C-m
echo "N=2 ω=1.0: GPU 1, tmux: v12b_n2w1"

# N=2 ω=0.5
tmux kill-session -t v12b_n2w05 2>/dev/null || true
tmux new-session -d -s v12b_n2w05 -c "${ROOT}"
tmux send-keys -t v12b_n2w05 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=2; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 2 --omega 0.5 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 3500 --n-coll 4096 --oversample 12 --micro-batch 1024 \
     --lr 1.5e-4 --lr-jas 1.5e-4 \
     --clip-el 4.0 --reward-qtrim 0.001 \
     --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.5 \
     --vmc-every 35 --vmc-n 40000 --n-eval 200000 --seed 402 \
     --resume '${CURATED}'/v8_n2w05_finetune.pt --no-pretrained \
     --tag v12b_n2w05 \
     >> '${LOGDIR}'/v12b_n2w05.log 2>&1" C-m
echo "N=2 ω=0.5: GPU 2, tmux: v12b_n2w05"

# N=2 ω=0.1
tmux kill-session -t v12b_n2w01 2>/dev/null || true
tmux new-session -d -s v12b_n2w01 -c "${ROOT}"
tmux send-keys -t v12b_n2w01 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=3; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 2 --omega 0.1 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 4000 --n-coll 4096 --oversample 16 --micro-batch 1024 \
     --lr 1e-4 --lr-jas 1e-4 \
     --clip-el 4.0 --reward-qtrim 0.002 \
     --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
     --vmc-every 40 --vmc-n 40000 --n-eval 200000 --seed 403 \
     --resume '${CURATED}'/v7_n2w01_exact.pt --no-pretrained \
     --tag v12b_n2w01 \
     >> '${LOGDIR}'/v12b_n2w01.log 2>&1" C-m
echo "N=2 ω=0.1: GPU 3, tmux: v12b_n2w01"

# ───────────────────────────────────────────────────────────────
# PART C: N=6 HIGH OMEGAS (1.0, 0.5, 0.1) – REINFORCE
# ───────────────────────────────────────────────────────────────

echo "Starting N=6 high-omega runs (pure REINFORCE)..."

# N=6 ω=1.0
tmux kill-session -t v12b_n6w1 2>/dev/null || true
tmux new-session -d -s v12b_n6w1 -c "${ROOT}"
tmux send-keys -t v12b_n6w1 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=4; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 6 --omega 1.0 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 2500 --n-coll 4096 --oversample 8 --micro-batch 1024 \
     --lr 1.5e-4 --lr-jas 1.5e-4 \
     --clip-el 4.0 --reward-qtrim 0.001 \
     --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
     --vmc-every 25 --vmc-n 40000 --n-eval 200000 --seed 501 \
     --resume '${CURATED}'/long_n6w1.pt --no-pretrained \
     --tag v12b_n6w1 \
     >> '${LOGDIR}'/v12b_n6w1.log 2>&1" C-m
echo "N=6 ω=1.0: GPU 4, tmux: v12b_n6w1"

# N=6 ω=0.5
tmux kill-session -t v12b_n6w05 2>/dev/null || true
tmux new-session -d -s v12b_n6w05 -c "${ROOT}"
tmux send-keys -t v12b_n6w05 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=5; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 6 --omega 0.5 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 3000 --n-coll 4096 --oversample 12 --micro-batch 1024 \
     --lr 1e-4 --lr-jas 1e-4 \
     --clip-el 4.0 --reward-qtrim 0.001 \
     --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.5 \
     --vmc-every 30 --vmc-n 40000 --n-eval 200000 --seed 502 \
     --resume '${CURATED}'/long_n6w05.pt --no-pretrained \
     --tag v12b_n6w05 \
     >> '${LOGDIR}'/v12b_n6w05.log 2>&1" C-m
echo "N=6 ω=0.5: GPU 5, tmux: v12b_n6w05"

# N=6 ω=0.1 (use available checkpoint)
tmux kill-session -t v12b_n6w01 2>/dev/null || true
tmux new-session -d -s v12b_n6w01 -c "${ROOT}"
tmux send-keys -t v12b_n6w01 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=6; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 6 --omega 0.1 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 3500 --n-coll 4096 --oversample 16 --micro-batch 1024 \
     --lr 8e-5 --lr-jas 8e-5 \
     --clip-el 4.0 --reward-qtrim 0.002 \
     --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 6.0 \
     --vmc-every 35 --vmc-n 40000 --n-eval 200000 --seed 503 \
     --resume '${CURATED}'/20260318_1149_n6w01_keep.pt --no-pretrained \
     --tag v12b_n6w01 \
     >> '${LOGDIR}'/v12b_n6w01.log 2>&1" C-m
echo "N=6 ω=0.1: GPU 6, tmux: v12b_n6w01"

# ───────────────────────────────────────────────────────────────
# PART D: N=12 HIGH OMEGAS (1.0, 0.5, 0.1) – REINFORCE
# ───────────────────────────────────────────────────────────────

echo "Starting N=12 high-omega runs (pure REINFORCE)..."

# N=12 ω=1.0
tmux kill-session -t v12b_n12w1 2>/dev/null || true
tmux new-session -d -s v12b_n12w1 -c "${ROOT}"
tmux send-keys -t v12b_n12w1 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=7; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 12 --omega 1.0 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 2000 --n-coll 4096 --oversample 8 --micro-batch 1024 \
     --lr 1e-4 --lr-jas 1e-4 \
     --clip-el 4.0 --reward-qtrim 0.001 \
     --rollback-decay 0.97 --rollback-err-pct 0.0 --rollback-jump-sigma 5.0 \
     --vmc-every 20 --vmc-n 40000 --n-eval 200000 --seed 601 \
     --resume '${CURATED}'/long_n12w1.pt --no-pretrained \
     --tag v12b_n12w1 \
     >> '${LOGDIR}'/v12b_n12w1.log 2>&1" C-m
echo "N=12 ω=1.0: GPU 7, tmux: v12b_n12w1"

# N=12 ω=0.5
tmux kill-session -t v12b_n12w05 2>/dev/null || true
tmux new-session -d -s v12b_n12w05 -c "${ROOT}"
tmux send-keys -t v12b_n12w05 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=0; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 12 --omega 0.5 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 2500 --n-coll 4096 --oversample 12 --micro-batch 1024 \
     --lr 8e-5 --lr-jas 8e-5 \
     --clip-el 4.0 --reward-qtrim 0.001 \
     --rollback-decay 0.98 --rollback-err-pct 0.0 --rollback-jump-sigma 5.5 \
     --vmc-every 25 --vmc-n 40000 --n-eval 200000 --seed 602 \
     --resume '${CURATED}'/long_n12w05.pt --no-pretrained \
     --tag v12b_n12w05 \
     >> '${LOGDIR}'/v12b_n12w05.log 2>&1" C-m
echo "N=12 ω=0.5: GPU 0, tmux: v12b_n12w05"

# N=12 ω=0.1 (use available checkpoint)
tmux kill-session -t v12b_n12w01 2>/dev/null || true
tmux new-session -d -s v12b_n12w01 -c "${ROOT}"
tmux send-keys -t v12b_n12w01 \
  "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; \
   export CUDA_VISIBLE_DEVICES=1; \
   python3 src/run_weak_form.py \
     --mode bf --n-elec 12 --omega 0.1 \
     --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
     --epochs 3000 --n-coll 4096 --oversample 16 --micro-batch 1024 \
     --lr 5e-5 --lr-jas 5e-5 \
     --clip-el 4.0 --reward-qtrim 0.002 \
     --rollback-decay 0.99 --rollback-err-pct 0.0 --rollback-jump-sigma 6.5 \
     --vmc-every 30 --vmc-n 40000 --n-eval 200000 --seed 603 \
     --resume '${CURATED}'/n12w05_cascade.pt --no-pretrained \
     --tag v12b_n12w01 \
     >> '${LOGDIR}'/v12b_n12w01.log 2>&1" C-m
echo "N=12 ω=0.1: GPU 1, tmux: v12b_n12w01"

echo ""
echo "=== All jobs started ==="
echo "Monitor with: tmux list-sessions"
echo "View output with: tail -f 2026-03-24_*/logs/v12b_*.log"
