#!/bin/bash
# N=2 precision campaign: target 5th-order agreement with DMC
# Strategy: resume from v4 BF checkpoints, CG-SR natural gradient,
#           massive collocation (8192), high oversampling (32), long runs (10000 ep)
cd /itf-fi-ml/home/aleksns/Thesis_repo
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=6

LOGDIR="outputs/2026-03-20_1209_campaign_v4/logs"
CKDIR="results/arch_colloc"

echo "=== N=2 precision campaign starting $(date) ==="

# Common polish args: CG-SR, high collocation, low LR, long training
# N=2 is only 4 coordinates — CG-SR should converge beautifully
COMMON="--mode bf --n-elec 2 --bf-hidden 64 --bf-layers 2 \
  --epochs 10000 --n-coll 8192 --oversample 32 --micro-batch 1024 \
  --lr 1e-4 --lr-jas 1e-4 \
  --natural-grad --sr-mode cg --sr-cg-iters 20 \
  --fisher-damping 1e-4 --fisher-damping-end 1e-5 --fisher-damping-anneal 5000 \
  --sr-max-param-change 0.05 --sr-trust-region 0.5 \
  --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.005 \
  --vmc-every 100 --vmc-n 50000 --n-eval 100000 --seed 42"

# ω = 1.0: already at -0.004%, polish to exactness
tag="v4_n2w1_precision"
log="${LOGDIR}/${tag}.log"
echo ">>> ω=1.0 (from ${CKDIR}/v4_n2w1_bf.pt)" | tee -a "$log"
python3 src/run_weak_form.py $COMMON \
  --omega 1.0 \
  --resume "${CKDIR}/v4_n2w1_bf.pt" \
  --tag "${tag}" >> "$log" 2>&1
echo "# Completed: $(date) rc=$?" >> "$log"

# ω = 0.5: at -0.040%, needs polishing
tag="v4_n2w05_precision"
log="${LOGDIR}/${tag}.log"
echo ">>> ω=0.5 (from ${CKDIR}/v4_n2w05_bf.pt)" | tee -a "$log"
python3 src/run_weak_form.py $COMMON \
  --omega 0.5 \
  --resume "${CKDIR}/v4_n2w05_bf.pt" \
  --tag "${tag}" >> "$log" 2>&1
echo "# Completed: $(date) rc=$?" >> "$log"

# ω = 0.1: at +0.072%, needs polishing
tag="v4_n2w01_precision"
log="${LOGDIR}/${tag}.log"
echo ">>> ω=0.1 (from ${CKDIR}/v4_n2w01_bf.pt)" | tee -a "$log"
python3 src/run_weak_form.py $COMMON \
  --omega 0.1 \
  --resume "${CKDIR}/v4_n2w01_bf.pt" \
  --tag "${tag}" >> "$log" 2>&1
echo "# Completed: $(date) rc=$?" >> "$log"

# ω = 0.01: at +0.122%, needs polishing + ESS adaptive
tag="v4_n2w001_precision"
log="${LOGDIR}/${tag}.log"
echo ">>> ω=0.01 (from ${CKDIR}/v4_n2w001_bf.pt)" | tee -a "$log"
python3 src/run_weak_form.py $COMMON \
  --omega 0.01 \
  --sigma-fs 0.8,1.3,2.0,3.5 \
  --ess-floor-ratio 0.05 --ess-oversample-max 64 \
  --ess-oversample-step 4 --ess-resample-tries 3 \
  --resume "${CKDIR}/v4_n2w001_bf.pt" \
  --tag "${tag}" >> "$log" 2>&1
echo "# Completed: $(date) rc=$?" >> "$log"

# ω = 0.001: catastrophic failure (+89%). Need completely different approach.
# Cascade from ω=0.01, ultra-wide proposal, Adam (CG-SR may struggle at this ω)
tag="v4_n2w0001_cascade"
log="${LOGDIR}/${tag}.log"
echo ">>> ω=0.001 CASCADE from ω=0.01 (from ${CKDIR}/v4_n2w001_bf.pt)" | tee -a "$log"
python3 src/run_weak_form.py \
  --mode bf --n-elec 2 --bf-hidden 64 --bf-layers 2 \
  --omega 0.001 \
  --epochs 10000 --n-coll 8192 --oversample 32 --micro-batch 1024 \
  --sigma-fs 1.0,3.0,8.0,15.0 \
  --lr 5e-5 --lr-jas 5e-5 \
  --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02 \
  --ess-floor-ratio 0.01 --ess-oversample-max 128 \
  --ess-oversample-step 8 --ess-resample-tries 5 \
  --rollback-decay 0.9 --rollback-err-pct 50 --rollback-jump-sigma 3.0 \
  --vmc-every 100 --vmc-n 50000 --n-eval 100000 --seed 42 \
  --resume "${CKDIR}/v4_n2w001_bf.pt" \
  --tag "${tag}" >> "$log" 2>&1
echo "# Completed: $(date) rc=$?" >> "$log"

# If cascade fails, try fresh start with ultra-wide proposal
tag="v4_n2w0001_fresh"
log="${LOGDIR}/${tag}.log"
echo ">>> ω=0.001 FRESH (no resume)" | tee -a "$log"
python3 src/run_weak_form.py \
  --mode bf --n-elec 2 --bf-hidden 64 --bf-layers 2 \
  --omega 0.001 \
  --epochs 10000 --n-coll 8192 --oversample 64 --micro-batch 1024 \
  --sigma-fs 2.0,5.0,12.0,25.0 \
  --lr 5e-4 --lr-jas 5e-4 \
  --direct-weight 0.0 --clip-el 5.0 --reward-qtrim 0.02 \
  --ess-floor-ratio 0.005 --ess-oversample-max 256 \
  --ess-oversample-step 16 --ess-resample-tries 5 \
  --rollback-decay 0.9 --rollback-err-pct 100 --rollback-jump-sigma 3.0 \
  --vmc-every 100 --vmc-n 50000 --n-eval 100000 --seed 42 \
  --no-pretrained \
  --tag "${tag}" >> "$log" 2>&1
echo "# Completed: $(date) rc=$?" >> "$log"

echo "=== N=2 precision campaign complete $(date) ==="
