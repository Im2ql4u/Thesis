#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Campaign v12: PURE REINFORCE, no SR for low omegas
#  ─────────────────────────────────────────────────────────────
#  Revert to pure REINFORCE across all omegas. 
#  - Low omegas (0.01, 0.005, 0.002, 0.001): Use transfer curriculum
#  - High omegas (1.0, 0.5, 0.1): Use stable REINFORCE settings
#  Launch all jobs in tmux sessions.
# ═══════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v12_reinforce_only"
LOGDIR="${OUT}/logs"
CURATED="${ROOT}/results/curated_low_error_0p1pct_2026-03-21"
ARCH_COLLOC="${ROOT}/results/arch_colloc"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'

mkdir -p "${LOGDIR}"

# Function to create a tmux session and run a job
run_in_tmux() {
  local session_name="$1"
  local gpu="$2"
  local tag="$3"
  shift 3
  
  local log="${LOGDIR}/${tag}.log"
  
  # Kill existing session if it exists
  tmux kill-session -t "${session_name}" 2>/dev/null || true
  
  # Create new tmux session in detached mode
  tmux new-session -d -s "${session_name}" -c "${ROOT}" \
    "echo \"# Started: \$(date) GPU=\${gpu} tag=\${tag}\" >> \"${log}\"; \
     eval \"${MODULE_CMD}\" || true; \
     CUDA_VISIBLE_DEVICES=\"${gpu}\" timeout 25200s python3 src/run_weak_form.py \"\$@\" --tag \"${tag}\" >> \"${log}\" 2>&1; \
     echo \"# Completed: \$(date) rc=\$?\" >> \"${log}\""
  
  # Run the command with proper argument expansion
  tmux send-keys -t "${session_name}" "cd ${ROOT} && eval \"${MODULE_CMD}\" || true; CUDA_VISIBLE_DEVICES=\"${gpu}\" timeout 25200s python3 src/run_weak_form.py $@ --tag \"${tag}\" >> \"${log}\" 2>&1 && echo \"# Completed: \$(date) rc=0\" >> \"${log}\" || echo \"# Completed: \$(date) rc=1\" >> \"${log}\"" C-m
  
  echo "Started tmux session: ${session_name} (GPU ${gpu}) -> ${log}"
}

echo "=== Campaign v12: Pure REINFORCE, no SR for low omegas ==="
echo "Output directory: ${OUT}"
echo ""

# ───────────────────────────────────────────────────────────────
# PART A: N=2 LOW OMEGA — transfer curriculum (REINFORCE only)
# ───────────────────────────────────────────────────────────────

echo "Starting N=2 low-omega transfer curriculum (REINFORCE)..."

# Stage 1: Polish ω=0.01
tmux kill-session -t v12_n2w001_stage1 2>/dev/null || true
tmux new-session -d -s v12_n2w001_stage1 -c "${ROOT}"
tmux send-keys -t v12_n2w001_stage1 \
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
     --tag v12_n2w001_stage1 \
     >> '${LOGDIR}'/v12_n2w001_stage1.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n2w001_stage1.log" C-m
echo "Stage 1 (ω=0.01): GPU 0, tmux: v12_n2w001_stage1"

# ───────────────────────────────────────────────────────────────
# PART B: N=2 HIGH OMEGAS (1.0, 0.5, 0.1) — REINFORCE, no SR
# ───────────────────────────────────────────────────────────────

echo "Starting N=2 high-omega runs (pure REINFORCE)..."

# N=2 ω=1.0
tmux kill-session -t v12_n2w1 2>/dev/null || true
tmux new-session -d -s v12_n2w1 -c "${ROOT}"
tmux send-keys -t v12_n2w1 \
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
     --tag v12_n2w1 \
     >> '${LOGDIR}'/v12_n2w1.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n2w1.log" C-m
echo "N=2 ω=1.0: GPU 1, tmux: v12_n2w1"

# N=2 ω=0.5
tmux kill-session -t v12_n2w05 2>/dev/null || true
tmux new-session -d -s v12_n2w05 -c "${ROOT}"
tmux send-keys -t v12_n2w05 \
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
     --resume '${CURATED}'/v7_n2w05_exact.pt --no-pretrained \
     --tag v12_n2w05 \
     >> '${LOGDIR}'/v12_n2w05.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n2w05.log" C-m
echo "N=2 ω=0.5: GPU 2, tmux: v12_n2w05"

# N=2 ω=0.1
tmux kill-session -t v12_n2w01 2>/dev/null || true
tmux new-session -d -s v12_n2w01 -c "${ROOT}"
tmux send-keys -t v12_n2w01 \
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
     --tag v12_n2w01 \
     >> '${LOGDIR}'/v12_n2w01.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n2w01.log" C-m
echo "N=2 ω=0.1: GPU 3, tmux: v12_n2w01"

# ───────────────────────────────────────────────────────────────
# PART C: N=6 HIGH OMEGAS (1.0, 0.5, 0.1) — REINFORCE
# ───────────────────────────────────────────────────────────────

echo "Starting N=6 high-omega runs (pure REINFORCE)..."

# N=6 ω=1.0
tmux kill-session -t v12_n6w1 2>/dev/null || true
tmux new-session -d -s v12_n6w1 -c "${ROOT}"
tmux send-keys -t v12_n6w1 \
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
     --resume '${CURATED}'/v7_n6w1_exact.pt --no-pretrained \
     --tag v12_n6w1 \
     >> '${LOGDIR}'/v12_n6w1.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n6w1.log" C-m
echo "N=6 ω=1.0: GPU 4, tmux: v12_n6w1"

# N=6 ω=0.5
tmux kill-session -t v12_n6w05 2>/dev/null || true
tmux new-session -d -s v12_n6w05 -c "${ROOT}"
tmux send-keys -t v12_n6w05 \
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
     --resume '${CURATED}'/v7_n6w05_exact.pt --no-pretrained \
     --tag v12_n6w05 \
     >> '${LOGDIR}'/v12_n6w05.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n6w05.log" C-m
echo "N=6 ω=0.5: GPU 5, tmux: v12_n6w05"

# N=6 ω=0.1
tmux kill-session -t v12_n6w01 2>/dev/null || true
tmux new-session -d -s v12_n6w01 -c "${ROOT}"
tmux send-keys -t v12_n6w01 \
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
     --resume '${CURATED}'/v7_n6w01_exact.pt --no-pretrained \
     --tag v12_n6w01 \
     >> '${LOGDIR}'/v12_n6w01.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n6w01.log" C-m
echo "N=6 ω=0.1: GPU 6, tmux: v12_n6w01"

# ───────────────────────────────────────────────────────────────
# PART D: N=12 HIGH OMEGAS (1.0, 0.5, 0.1) — REINFORCE
# ───────────────────────────────────────────────────────────────

echo "Starting N=12 high-omega runs (pure REINFORCE)..."

# N=12 ω=1.0
tmux kill-session -t v12_n12w1 2>/dev/null || true
tmux new-session -d -s v12_n12w1 -c "${ROOT}"
tmux send-keys -t v12_n12w1 \
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
     --resume '${CURATED}'/v7_n12w1_exact.pt --no-pretrained \
     --tag v12_n12w1 \
     >> '${LOGDIR}'/v12_n12w1.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n12w1.log" C-m
echo "N=12 ω=1.0: GPU 7, tmux: v12_n12w1"

# N=12 ω=0.5
tmux kill-session -t v12_n12w05 2>/dev/null || true
tmux new-session -d -s v12_n12w05 -c "${ROOT}"
tmux send-keys -t v12_n12w05 \
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
     --resume '${CURATED}'/v7_n12w05_exact.pt --no-pretrained \
     --tag v12_n12w05 \
     >> '${LOGDIR}'/v12_n12w05.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n12w05.log" C-m
echo "N=12 ω=0.5: GPU 0, tmux: v12_n12w05"

# N=12 ω=0.1
tmux kill-session -t v12_n12w01 2>/dev/null || true
tmux new-session -d -s v12_n12w01 -c "${ROOT}"
tmux send-keys -t v12_n12w01 \
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
     --resume '${CURATED}'/v7_n12w01_exact.pt --no-pretrained \
     --tag v12_n12w01 \
     >> '${LOGDIR}'/v12_n12w01.log 2>&1; \
   echo \"# Completed: \$(date) rc=0\" >> '${LOGDIR}'/v12_n12w01.log" C-m
echo "N=12 ω=0.1: GPU 1, tmux: v12_n12w01"

echo ""
echo "=== All jobs started ==="
echo "High omegas (N=2,6,12 @ ω≥0.1) will complete in ~2-4 hours"
echo "After stage 1 completes (~6 hours), stages 2-4 of N=2 low-omega will auto-run"
echo ""
echo "Monitor with: tmux list-sessions"
echo "Attach with: tmux attach -t <session_name>"
