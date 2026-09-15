#!/usr/bin/env bash
# Collocation+Adam cells at N=12 and N=20 (the SR cells are composed from score_matrix.log).
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
cd ~/Thesis
L=outputs/2026-09-14_walltime
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
CUDA_VISIBLE_DEVICES=6 python3 scripts/bench_walltime.py --device cuda --N 12 --routes colloc-adam --chunk 512 --proposal-chunk 4096 --tag N12_colloc > $L/N12_colloc.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python3 scripts/bench_walltime.py --device cuda --N 20 --routes colloc-adam --vmc-batch 1024 --n-coll 1024 --chunk 128 --proposal-chunk 1024 --tag N20_colloc > $L/N20_colloc.log 2>&1 &
wait
