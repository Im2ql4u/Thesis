#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Launch 4 neural Pfaffian experiments in tmux (disconnect-safe)
#
#  Usage:
#    chmod +x launch_neural_pf.sh
#    ./launch_neural_pf.sh           # seed=42
#    ./launch_neural_pf.sh 123       # custom seed
#    ./launch_neural_pf.sh 42 3      # seed=42, 3 seeds (42,43,44)
#
#  Reattach:   tmux attach -t neurpf
#  Kill all:   tmux kill-session -t neurpf
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SEED_BASE=${1:-42}
N_SEEDS=${2:-1}
SESSION="neurpf"
DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_CMD="source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
SCRIPT="$DIR/run_neural_pfaffian.py"
LOG_DIR="$DIR/../results/arch_colloc/logs"
mkdir -p "$LOG_DIR"

# ── Experiment definitions ──
# Format: TAG GPU_OFFSET EXTRA_FLAGS
EXPERIMENTS=(
  "neurpf_K1:1:--K-det 1"
  "neurpf_K1_bf:3:--K-det 1 --use-backflow"
  "neurpf_K4_staged:4:--K-det 4 --use-backflow --stage-bf"
  "neurpf_K4:5:--K-det 4"
)

# Common flags for all experiments
COMMON="--epochs 1000 --n-coll 768 --lr 5e-4 --alpha-end 0.85 --patience 250 --vmc-every 50 --vmc-n 10000 --n-eval 30000 --embed-lr-ratio 0.1 --embed-warmup 50"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"

WIN=0
for s in $(seq 0 $((N_SEEDS - 1))); do
  SEED=$((SEED_BASE + s))
  for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r TAG GPU_OFF FLAGS <<< "$entry"
    GPU=$((GPU_OFF + s * 4))  # spread seeds across GPUs 0-7
    FULL_TAG="${TAG}_s${SEED}"
    LOGFILE="$LOG_DIR/${FULL_TAG}.log"

    CMD="cd $DIR; $MODULE_CMD; CUDA_MANUAL_DEVICE=$GPU python $SCRIPT $COMMON $FLAGS --seed $SEED --tag $FULL_TAG 2>&1 | tee $LOGFILE"

    if [ $WIN -eq 0 ]; then
      tmux send-keys -t "$SESSION" "$CMD" C-m
    else
      tmux new-window -t "$SESSION"
      tmux send-keys -t "$SESSION" "$CMD" C-m
    fi
    WIN=$((WIN + 1))
  done
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Launched $WIN experiments in tmux session '$SESSION'"
echo ""
echo "  Experiments:"
for s in $(seq 0 $((N_SEEDS - 1))); do
  SEED=$((SEED_BASE + s))
  for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r TAG GPU_OFF FLAGS <<< "$entry"
    GPU=$((GPU_OFF + s * 4))
    echo "    GPU $GPU: ${TAG}_s${SEED}  ($FLAGS)"
  done
done
echo ""
echo "  Monitor:    tmux attach -t $SESSION"
echo "  Kill all:   tmux kill-session -t $SESSION"
echo "  Logs:       $LOG_DIR/"
echo "═══════════════════════════════════════════════════════════"
