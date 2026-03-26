#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Campaign v11: Validate importance-sampling bugfix
#  ─────────────────────────────────────────────────────────────
#  Bug: sample_mixture() used per-component log_q instead of
#  mixture density. Fixed 2026-03-24. This campaign validates
#  that low-omega + higher-N now converge with SR enabled.
#
#  8 jobs × 1 GPU each. ~3-6 hours per job.
# ═══════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT="${ROOT}/outputs/${TIMESTAMP}_campaign_v11_sampling_fix"
LOGDIR="${OUT}/logs"
CURATED="${ROOT}/results/curated_low_error_0p1pct_2026-03-21"
MODULE_CMD='source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null'

mkdir -p "${LOGDIR}"

run_job() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local log="${LOGDIR}/${tag}.log"
  echo "# Started: $(date) GPU=${gpu} tag=${tag}" >> "${log}"
  (
    cd "${ROOT}"
    eval "${MODULE_CMD}" || true
    CUDA_MANUAL_DEVICE="${gpu}" timeout 25200s python3 src/run_weak_form.py "$@" --tag "${tag}"
  ) >> "${log}" 2>&1
  local rc=$?
  echo "# Completed: $(date) rc=${rc}" >> "${log}"
}

# ─── N=2 across omega sweep, BF mode, CG-SR ──────────────────

# GPU 0: N=2 ω=1.0 (control — should match or beat previous best)
run_job 0 v11_n2w1_sr \
  --mode bf --n-elec 2 --omega 1.0 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 3000 --n-coll 4096 --oversample 8 --micro-batch 1024 \
  --lr 2e-4 --lr-jas 2e-4 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 200 \
  --fisher-subsample 1024 --nat-momentum 0.9 \
  --sr-max-param-change 0.05 --sr-trust-region 0.5 \
  --direct-weight 0.05 --clip-el 4.0 \
  --rollback-jump-sigma 5.0 --rollback-decay 0.97 \
  --vmc-every 40 --vmc-n 20000 --n-eval 60000 --seed 42 \
  --resume "${CURATED}/v7_n2w1_exact.pt" --no-pretrained &

# GPU 1: N=2 ω=0.1 (key test — was broken, should now converge with SR)
run_job 1 v11_n2w01_sr \
  --mode bf --n-elec 2 --omega 0.1 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 4000 --n-coll 4096 --oversample 16 --micro-batch 1024 \
  --lr 1e-4 --lr-jas 1e-4 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 300 \
  --fisher-subsample 1024 --nat-momentum 0.9 \
  --sr-max-param-change 0.05 --sr-trust-region 0.5 \
  --direct-weight 0.05 --clip-el 4.0 \
  --rollback-jump-sigma 5.0 --rollback-decay 0.97 \
  --vmc-every 50 --vmc-n 20000 --n-eval 80000 --seed 42 \
  --resume "${CURATED}/v4_n2w01_bf.pt" --no-pretrained &

# GPU 2: N=2 ω=0.01 (hard test — SR was disabled here before)
run_job 2 v11_n2w001_sr \
  --mode bf --n-elec 2 --omega 0.01 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 6000 --n-coll 4096 --oversample 24 --micro-batch 1024 \
  --lr 5e-5 --lr-jas 5e-5 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 400 \
  --fisher-subsample 1024 --nat-momentum 0.9 \
  --sr-max-param-change 0.03 --sr-trust-region 0.3 \
  --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.002 \
  --rollback-jump-sigma 6.0 --rollback-decay 0.98 \
  --vmc-every 60 --vmc-n 30000 --n-eval 120000 --seed 42 \
  --resume "${CURATED}/v7_n2w01_exact.pt" --no-pretrained &

# GPU 4: N=2 ω=0.001 (hardest N=2 test — most affected by the bug)
run_job 4 v11_n2w0001_sr \
  --mode bf --n-elec 2 --omega 0.001 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 8000 --n-coll 4096 --oversample 36 --micro-batch 1024 \
  --lr 2e-5 --lr-jas 2e-5 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 500 \
  --fisher-subsample 1024 --nat-momentum 0.9 \
  --sr-max-param-change 0.03 --sr-trust-region 0.3 \
  --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.002 \
  --rollback-jump-sigma 7.0 --rollback-decay 0.99 \
  --vmc-every 80 --vmc-n 40000 --n-eval 160000 --seed 42 \
  --resume "${CURATED}/v7_n2w001_exact.pt" --no-pretrained &

# ─── N=6: test higher-N improvement ──────────────────────────

# GPU 6: N=6 ω=1.0 (control)
run_job 6 v11_n6w1_sr \
  --mode bf --n-elec 6 --omega 1.0 \
  --bf-hidden 128 --bf-msg-hidden 128 --bf-layers 3 \
  --epochs 3000 --n-coll 4096 --oversample 8 --micro-batch 512 \
  --lr 1e-4 --lr-jas 5e-5 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 200 \
  --fisher-subsample 1024 --nat-momentum 0.9 \
  --sr-max-param-change 0.05 --sr-trust-region 0.5 \
  --direct-weight 0.05 --clip-el 4.0 \
  --rollback-jump-sigma 5.0 --rollback-decay 0.97 \
  --vmc-every 40 --vmc-n 20000 --n-eval 60000 --seed 42 \
  --resume "${CURATED}/long_n6w1.pt" --no-pretrained &

# GPU 7: N=6 ω=0.1 (key higher-N low-omega test)
run_job 7 v11_n6w01_sr \
  --mode bf --n-elec 6 --omega 0.1 \
  --bf-hidden 128 --bf-msg-hidden 128 --bf-layers 3 \
  --epochs 4000 --n-coll 4096 --oversample 16 --micro-batch 512 \
  --lr 5e-5 --lr-jas 2e-5 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 300 \
  --fisher-subsample 1024 --nat-momentum 0.9 \
  --sr-max-param-change 0.03 --sr-trust-region 0.3 \
  --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.002 \
  --sigma-fs "0.4,0.7,1.0,1.5,2.5,4.0" \
  --rollback-jump-sigma 6.0 --rollback-decay 0.98 \
  --vmc-every 50 --vmc-n 30000 --n-eval 100000 --seed 42 \
  --resume "${CURATED}/20260318_0858_n6w01_keep.pt" --no-pretrained &

# ─── N=12: test scaling ──────────────────────────────────────

# GPU 3: N=12 ω=1.0 (control)
run_job 3 v11_n12w1_sr \
  --mode bf --n-elec 12 --omega 1.0 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 3000 --n-coll 4096 --oversample 8 --micro-batch 256 \
  --lr 5e-5 --lr-jas 2e-5 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 200 \
  --fisher-subsample 512 --nat-momentum 0.9 \
  --sr-max-param-change 0.05 --sr-trust-region 0.5 \
  --direct-weight 0.05 --clip-el 4.0 \
  --rollback-jump-sigma 5.0 --rollback-decay 0.97 \
  --vmc-every 40 --vmc-n 20000 --n-eval 60000 --seed 42 \
  --resume "${CURATED}/v7_n12w1_continue.pt" --no-pretrained &

# GPU 5: N=12 ω=0.1 (hardest test in this campaign)
run_job 5 v11_n12w01_sr \
  --mode bf --n-elec 12 --omega 0.1 \
  --bf-hidden 64 --bf-msg-hidden 64 --bf-layers 2 \
  --epochs 4000 --n-coll 4096 --oversample 16 --micro-batch 256 \
  --lr 2e-5 --lr-jas 1e-5 \
  --natural-grad --sr-mode cg --sr-cg-iters 100 \
  --fisher-damping 1e-3 --fisher-damping-end 1e-4 --fisher-damping-anneal 300 \
  --fisher-subsample 512 --nat-momentum 0.9 \
  --sr-max-param-change 0.03 --sr-trust-region 0.3 \
  --direct-weight 0.0 --clip-el 4.0 --reward-qtrim 0.002 \
  --sigma-fs "0.4,0.7,1.0,1.5,2.5,4.0" \
  --rollback-jump-sigma 6.0 --rollback-decay 0.98 \
  --vmc-every 50 --vmc-n 30000 --n-eval 100000 --seed 42 \
  --resume "${ROOT}/results/arch_colloc/smoke_n12_o0p1.pt" --no-pretrained &

echo "All 8 jobs launched. Logs in: ${LOGDIR}"
echo "Monitor with: tail -f ${LOGDIR}/*.log"
wait
echo "All jobs completed: $(date)"
