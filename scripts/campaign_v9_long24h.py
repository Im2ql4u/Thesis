#!/usr/bin/env python3
from pathlib import Path
import subprocess
import time
from datetime import datetime

ROOT = Path('/itf-fi-ml/home/aleksns/Thesis_repo')
OUT = ROOT / 'outputs/2026-03-21_1920_campaign_v9_long24h'
LOGS = OUT / 'logs'
SESSION = 'v9long24h'


def run(cmd):
    subprocess.run(cmd, check=True)


def setup_tmux():
    subprocess.run(['tmux', 'kill-session', '-t', SESSION], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.5)
    run(['tmux', 'new-session', '-d', '-s', SESSION, '-x', '220', '-y', '55', '-n', 'control'])
    for i in range(1, 5):
        run(['tmux', 'new-window', '-t', SESSION, '-n', f'gpu{i-1}'])


def send(window, command):
    cmd = f'cd {ROOT} && {command}'
    run(['tmux', 'send-keys', '-t', f'{SESSION}:{window}', cmd, 'Enter'])


def main():
    LOGS.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('Campaign v9 long24h')
    print(f'Start time: {datetime.now().isoformat(timespec="seconds")}')
    print(f'Output dir: {OUT}')
    print('=' * 72)

    setup_tmux()

    send(1, 'bash scripts/run_v9_n2_chain_24h.sh')
    send(2, 'bash scripts/run_v9_n12_chain_24h.sh')
    send(3, 'bash scripts/run_v9_n6_chain_24h.sh')
    send(4, 'bash scripts/run_v9_n20_chain_24h.sh')

    print(f'Launched tmux session: {SESSION}')
    print('Windows:')
    print('  1 -> N=2 continuation chain (target: all omegas)')
    print('  2 -> N=12 polish chain')
    print('  3 -> N=6 polish chain')
    print('  4 -> N=20 fix chain')
    print('\nMonitor examples:')
    print(f'  tmux capture-pane -t {SESSION}:1 -p | tail -40')
    print(f'  tail -f {LOGS}/*.log')


if __name__ == '__main__':
    main()
