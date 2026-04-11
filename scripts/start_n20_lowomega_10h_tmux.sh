#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="n20_lowomega_10h"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="outputs/higher_n/phase4_n20_lowomega_escalation_tmux_${TS}.log"
mkdir -p outputs/higher_n

chmod +x scripts/launch_n20_lowomega_10h.sh

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT_DIR' && source /etc/profile.d/z00_lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1; ./scripts/launch_n20_lowomega_10h.sh 2>&1 | tee '$LOG_PATH'"

echo "Started tmux session: $SESSION_NAME"
echo "Log: $LOG_PATH"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Watch: tail -f $LOG_PATH"
