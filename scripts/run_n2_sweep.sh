#!/bin/bash
# N=2 BF sweep across all omega values on GPU 6
cd /itf-fi-ml/home/aleksns/Thesis_repo
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true
set -e
export CUDA_VISIBLE_DEVICES=6

LOGDIR="outputs/2026-03-20_1209_campaign_v4/logs"

echo "=== N=2 BF sweep starting $(date) ==="

for omega_tag in "1.0:v4_n2w1_bf" "0.5:v4_n2w05_bf" "0.1:v4_n2w01_bf" "0.01:v4_n2w001_bf"; do
    omega="${omega_tag%%:*}"
    tag="${omega_tag##*:}"
    log="${LOGDIR}/${tag}.log"
    echo "# Started: $(date) GPU=6 tag=${tag}" >> "$log"
    echo ">>> Running N=2 omega=${omega} tag=${tag}"
    python3 src/run_weak_form.py \
        --mode bf --n-elec 2 --omega "${omega}" \
        --bf-hidden 64 --bf-layers 2 \
        --epochs 2000 \
        --n-coll 2048 --oversample 8 --micro-batch 512 \
        --lr 5e-4 --lr-jas 5e-4 \
        --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.01 \
        --vmc-every 50 --vmc-n 10000 \
        --n-eval 20000 --seed 42 \
        --no-pretrained \
        --tag "${tag}" >> "$log" 2>&1
    echo "# Completed: $(date) rc=$?" >> "$log"
    echo ">>> Done N=2 omega=${omega}"
done

# Last job: omega=0.001 with wider proposal and ESS adaptive
tag="v4_n2w0001_bf"
log="${LOGDIR}/${tag}.log"
echo "# Started: $(date) GPU=6 tag=${tag}" >> "$log"
echo ">>> Running N=2 omega=0.001 tag=${tag}"
python3 src/run_weak_form.py \
    --mode bf --n-elec 2 --omega 0.001 \
    --bf-hidden 64 --bf-layers 2 \
    --epochs 2000 \
    --n-coll 2048 --oversample 8 --micro-batch 512 \
    --sigma-fs 0.8,1.3,2.0,3.5 \
    --lr 5e-4 --lr-jas 5e-4 \
    --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02 \
    --ess-floor-ratio 0.01 --ess-oversample-max 12 \
    --ess-oversample-step 2 --ess-resample-tries 2 \
    --vmc-every 50 --vmc-n 10000 \
    --n-eval 20000 --seed 42 \
    --no-pretrained \
    --tag "${tag}" >> "$log" 2>&1
echo "# Completed: $(date) rc=$?" >> "$log"
echo ">>> Done N=2 omega=0.001"

echo "=== N=2 BF sweep complete $(date) ==="
