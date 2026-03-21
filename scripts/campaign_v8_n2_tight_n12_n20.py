#!/usr/bin/env python3
"""
Campaign v8: N=2 Sub-0.01% Accuracy Target + N=12 Polish + N=20 BF Training
─────────────────────────────────────────────────────────────────────────
Goals:
  • N=2: ALL omegas below 0.01% error (warm-start from v7 + fine-tuning)
  • N=2 ω=0.001: Special low-ω intervention (cascaded + conservative rollback)
  • N=12: High-ω continuation (ω=1.0, 0.1)
  • N=20: BF training from scratch (explore multi-particle regime)

Strategy:
  • N=2 ω≥0.01: Resume v7 checkpoints, reduce lr, run 1000-2000 epochs
  • N=2 ω=0.001: Cascade from ω=0.01, extended epochs (3000+), aggressive sampling
  • N=12: ω=1.0, 0.1 only (skip 0.001), parallel GPU 2-3
  • N=20: ω=1.0, 0.5, 0.1 (fresh start or good baseline), parallel GPU 4-5

Timing: ~4-5 hours wall (14400s per GPU)
"""

import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

def setup_tmux_session(session_name="v8n2n12n20", num_windows=6):
    """Create fresh tmux session with named windows."""
    # Kill existing session
    subprocess.run(["tmux", "kill-session", "-t", session_name], 
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.5)
    
    # Create new session with first window
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, 
                    "-x", "220", "-y", "55", "-n", "bash"],
                   check=True)
    
    # Create additional windows
    for i in range(1, num_windows):
        subprocess.run(["tmux", "new-window", "-t", session_name, "-n", f"gpu{i}"],
                       check=True)
    
    print(f"✓ Created tmux session '{session_name}' with {num_windows} windows")
    return session_name

def run_command_in_window(session_name, window_idx, cmd, tag=""):
    """Send command to tmux window."""
    # Load pytorch module first
    module_load = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null; "
    full_cmd = f"cd /itf-fi-ml/home/aleksns/Thesis_repo && {module_load} {cmd}"
    subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:{window_idx}",
                    full_cmd, "Enter"], check=True)
    if tag:
        print(f"  [{tag}] Sent to window {window_idx}")

def main():
    parser = argparse.ArgumentParser(description="Campaign v8 launcher")
    parser.add_argument("--skip-tmux", action="store_true", 
                       help="Skip tmux setup (use existing)")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    campaign_dir = f"outputs/{timestamp}_campaign_v8_n2_tight_n12_n20"
    Path(campaign_dir).mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(campaign_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║          CAMPAIGN V8: N=2 <0.01% + N=12 + N=20 BF             ║
║         Tight Accuracy Target with Low-ω Breakthrough          ║
║                 Starting: {timestamp}                              ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Setup tmux
    session_name = "v8n2n12n20"
    if not args.skip_tmux:
        setup_tmux_session(session_name, num_windows=6)
    
    time.sleep(1)
    
    # Window allocation:
    # 0: control/monitoring
    # 1: N=2 chain (GPU 0)
    # 2: N=12 ω=1.0 (GPU 1)
    # 3: N=12 ω=0.1 (GPU 2) + N=12 ω=0.01 cascade (GPU 3)
    # 4: N=20 ω=1.0 (GPU 4)
    # 5: N=20 ω=0.5, 0.1 (GPU 5)
    
    # N=2 CHAIN (GPU 0) - warm-start from v7, fine-tune
    n2_script = "scripts/run_v8_n2_finetune_chain.sh"
    run_command_in_window(session_name, 1, f"bash {n2_script}", "N=2 chain")
    
    # N=12 ω=1.0 (GPU 1) - 1-2 hour continuation
    run_command_in_window(session_name, 2,
        "timeout 7200 python3 src/run_weak_form.py "
        "--n-elec 12 --omega 1.0 --mode bf --resume models/2p/long_n12w1.pt "
        "--epochs 1000 --lr 0.0005 --lr-jas 0.0005 "
        "--bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 "
        "--rollback-decay 0.97 --rollback-err-pct 2.0 "
        "--seed 42 --tag v8_n12w1_finetune "
        f"2>&1 | tee {log_dir}/v8_n12w1_finetune.log",
        "N=12 ω=1.0")
    
    # N=12 ω=0.1 (GPU 2) - 1-2 hour continuation
    run_command_in_window(session_name, 3,
        "timeout 7200 python3 src/run_weak_form.py "
        "--n-elec 12 --omega 0.1 --mode bf --resume models/2p/20260318_1149_n12w01_keep.pt "
        "--epochs 1000 --lr 0.0005 --lr-jas 0.0005 "
        "--bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 "
        "--rollback-decay 0.97 --rollback-err-pct 0.0 "
        "--seed 42 --tag v8_n12w01_finetune "
        f"2>&1 | tee {log_dir}/v8_n12w01_finetune.log",
        "N=12 ω=0.1")
    
    # N=20 ω=1.0 (GPU 4) - fresh training, 2-3 hours
    run_command_in_window(session_name, 4,
        "timeout 10800 python3 src/run_weak_form.py "
        "--n-elec 20 --omega 1.0 --mode bf "
        "--epochs 2500 --lr 0.001 --lr-jas 0.001 "
        "--bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 "
        "--rollback-decay 0.96 --rollback-err-pct 1.0 "
        "--seed 11 --tag v8_n20w1_bf "
        f"2>&1 | tee {log_dir}/v8_n20w1_bf.log",
        "N=20 ω=1.0")
    
    # N=20 ω=0.5, 0.1 (GPU 5) - chain runner
    n20_script = "scripts/run_v8_n20_chain.sh"
    run_command_in_window(session_name, 5, f"bash {n20_script}", "N=20 chain")
    
    print(f"""
    ✓ Campaign launched in tmux session '{session_name}'
    
    GPU Allocation:
      GPU 0: N=2 chain (ω=1.0→0.5→0.1→0.01→0.001, fine-tune)
      GPU 1: N=12 ω=1.0 (polish)
      GPU 2: N=12 ω=0.1 (polish)
      GPU 3: [reserved for N=12 ω=0.01 or spillover]
      GPU 4: N=20 ω=1.0 (fresh)
      GPU 5: N=20 ω=0.5, 0.1 (chain)
    
    Logs: {log_dir}/
    
    Monitor:
      tmux capture-pane -t {session_name}:1 -p  # N=2 status
      tmux capture-pane -t {session_name}:4 -p  # N=20 status
      tail -f {log_dir}/v8_n2*.log               # Real-time N=2
    """)

if __name__ == "__main__":
    main()
