#!/usr/bin/env bash
# Production SR step at the thesis batch (~3e3 rows); 9000 rows does not fit on a shared 2080 Ti.
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
cd ~/Thesis
L=outputs/2026-09-15_production_walltime
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=4 python3 scripts/bench_production_sr.py --device cuda --N 6 --steps 2 --total-rows 3000 --tag rows3000 > $L/sr_N6_rows3000.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python3 scripts/bench_production_sr.py --device cuda --N 2 --steps 2 --total-rows 3000 --tag rows3000 > $L/sr_N2_rows3000.log 2>&1 &
wait
