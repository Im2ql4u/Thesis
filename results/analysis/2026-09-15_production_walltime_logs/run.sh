#!/usr/bin/env bash
# Production routes as the thesis ran them: SR-VMC step (example-run settings) vs weak-form
# collocation (robust campaign recipe), per-step/epoch wall time on shared RTX 2080 Ti.
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
cd ~/Thesis
L=outputs/2026-09-15_production_walltime
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 PYTHONUNBUFFERED=1
COLLOC="--mode bf --epochs 8 --lr 5e-4 --lr-jas 5e-5 --n-coll 4096 --oversample 16 --micro-batch 512 --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0 --reward-normalize --rollback-jump-sigma 3 --rollback-decay 0.95 --adaptive-proposal --gmm-components 8 --gmm-refit-every 30 --vmc-every 100000 --vmc-n 1000 --n-eval 1000 --print-every 1 --no-pretrained --seed 42"
CUDA_VISIBLE_DEVICES=4 python3 scripts/bench_production_sr.py --device cuda --N 6 --steps 2 > $L/sr_N6.log 2>&1 &
CUDA_MANUAL_DEVICE=5 python3 src/run_weak_form.py $COLLOC --n-elec 6 --omega 1.0 --tag bench_wt_N6 > $L/colloc_N6.log 2>&1 &
( CUDA_MANUAL_DEVICE=6 python3 src/run_weak_form.py $COLLOC --n-elec 2 --omega 1.0 --tag bench_wt_N2 > $L/colloc_N2.log 2>&1 ; \
  CUDA_MANUAL_DEVICE=6 python3 src/run_weak_form.py $COLLOC --n-elec 12 --omega 1.0 --tag bench_wt_N12 > $L/colloc_N12.log 2>&1 ) &
wait
