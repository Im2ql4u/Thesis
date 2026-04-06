#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/higher_n"
mkdir -p "$LOG_DIR"

SESSION_NAME="higherN_all"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$LOG_DIR/all_phases_${TS}.log"

echo "Starting tmux session: $SESSION_NAME"
echo "Pipeline log: $PIPE_LOG"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT_DIR' && source /etc/profile.d/z00_lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1; { echo '=== higher-N all phases started ==='; date; ./scripts/launch_higher_n_phase2.sh; echo '=== phase2 done ==='; date; ./scripts/launch_higher_n_phase3.sh; echo '=== phase3 done ==='; date; } 2>&1 | tee '$PIPE_LOG'"

echo "tmux session '$SESSION_NAME' started."
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Watch:  tail -f $PIPE_LOG"
