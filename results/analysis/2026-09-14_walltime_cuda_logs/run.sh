#!/usr/bin/env bash
# Per-step wall time of vmc/colloc x adam/sr, thesis batch sizes, one GPU per N group.
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
cd ~/Thesis
L=outputs/2026-09-14_walltime
CUDA_VISIBLE_DEVICES=5 nohup python3 scripts/bench_walltime.py --device cuda --N 2 6 --tag N2N6 > $L/N2N6.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 scripts/bench_walltime.py --device cuda --N 12 --chunk 512 --proposal-chunk 4096 --tag N12 > $L/N12.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 scripts/bench_walltime.py --device cuda --N 20 --chunk 128 --proposal-chunk 1024 --tag N20 > $L/N20.log 2>&1 &
wait
