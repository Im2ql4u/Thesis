#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null

PHASE4_RUN_ID="2026-04-12_phase4_proposal_improvement"
PHASE4_DIR="outputs/${PHASE4_RUN_ID}"
PHASE5_RUN_ID="2026-04-12_phase5_tail_from_p4_winner"
PHASE5_DIR="outputs/${PHASE5_RUN_ID}"
mkdir -p "$PHASE5_DIR"

echo "[$(date '+%F %T')] Waiting for Phase 4 completion in ${PHASE4_DIR}" | tee -a "${PHASE5_DIR}/queue.log"

while true; do
  if [[ -f "${PHASE4_DIR}/nohup.log" ]] && grep -q "All workers finished" "${PHASE4_DIR}/nohup.log"; then
    break
  fi
  sleep 60
done

echo "[$(date '+%F %T')] Phase 4 complete. Evaluating gate and selecting winner..." | tee -a "${PHASE5_DIR}/queue.log"

python3 - <<'PY' > "${PHASE5_DIR}/phase4_gate_and_winner.txt"
import re
import glob
import statistics

rows = []
for f in glob.glob('outputs/2026-04-12_phase4_proposal_improvement/logs/worker_gpu*.log'):
    txt = open(f).read()
    for m in re.finditer(r'Tag: (robust_p4prop_n6w([0-9p]+)_s(\d+))\n\s+E = ([0-9.]+) ± ([0-9.]+)\s+err = \+?([0-9.]+)%', txt):
        tag, wtag, seed, E, sig, err = m.groups()
        omega = float(wtag.replace('p', '.'))
        rows.append((omega, int(seed), tag, float(err)))

# khat averages by stage
khat_by_omega = {}
for f in glob.glob('outputs/2026-04-12_phase4_proposal_improvement/logs/worker_gpu*.log'):
    txt = open(f).read()
    starts = list(re.finditer(r'^\[.*?\] START .*?tag=(robust_p4prop_n6w([0-9p]+)_s(\d+)).*?$', txt, re.M))
    for i, m in enumerate(starts):
        wtag = m.group(2)
        omega = float(wtag.replace('p', '.'))
        seg = txt[m.end():(starts[i + 1].start() if i + 1 < len(starts) else len(txt))]
        kh = [float(x) for x in re.findall(r'khat=([0-9.]+)', seg)]
        if kh:
            khat_by_omega.setdefault(omega, []).append(sum(kh) / len(kh))

rows_001 = [r for r in rows if abs(r[0] - 0.01) < 1e-12]
mean_khat_001 = statistics.mean(khat_by_omega.get(0.01, [float('inf')]))
gate_pass = mean_khat_001 < 1.0

rows_01 = [r for r in rows if abs(r[0] - 0.1) < 1e-12]
winner = min(rows_01, key=lambda x: x[3]) if rows_01 else None

print(f'phase4_count={len(rows)}')
print(f'omega001_mean_khat={mean_khat_001:.3f}')
print(f'gate_pass={gate_pass}')
if winner:
    print(f'winner_tag={winner[2]}')
    print(f'winner_seed={winner[1]}')
    print(f'winner_err={winner[3]:.3f}')
PY

cat "${PHASE5_DIR}/phase4_gate_and_winner.txt" | tee -a "${PHASE5_DIR}/queue.log"

WINNER_TAG="$(grep '^winner_tag=' "${PHASE5_DIR}/phase4_gate_and_winner.txt" | cut -d= -f2-)"
WINNER_SEED="$(grep '^winner_seed=' "${PHASE5_DIR}/phase4_gate_and_winner.txt" | cut -d= -f2-)"
if [[ -z "${WINNER_TAG}" || -z "${WINNER_SEED}" ]]; then
  echo "[$(date '+%F %T')] ERROR: could not determine omega=0.1 Phase 4 winner" | tee -a "${PHASE5_DIR}/queue.log"
  exit 1
fi

WINNER_CKPT="${ROOT}/results/arch_colloc/${WINNER_TAG}.pt"
if [[ ! -f "${WINNER_CKPT}" ]]; then
  echo "[$(date '+%F %T')] ERROR: winner checkpoint missing: ${WINNER_CKPT}" | tee -a "${PHASE5_DIR}/queue.log"
  exit 1
fi

GPU=0
LOG_FILE="${PHASE5_DIR}/phase5_tail.log"
TAG_STAGE2="robust_p5tail_n6w0p1_s${WINNER_SEED}_stage2"
TAG_STAGE3="robust_p5tail_n6w0p1_s${WINNER_SEED}_stage3"

echo "[$(date '+%F %T')] Starting Phase 5 tail stage2 from ${WINNER_TAG}" | tee -a "$LOG_FILE"
CUDA_MANUAL_DEVICE="$GPU" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py \
  --mode bf --n-elec 6 --omega 0.1 \
  --epochs 800 --lr 2e-4 --lr-jas 2e-5 \
  --n-coll 4096 --oversample 16 --micro-batch 512 \
  --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0 \
  --reward-normalize \
  --rollback-jump-sigma 3 --rollback-decay 0.95 \
  --adaptive-proposal --gmm-components 16 --gmm-refit-every 15 \
  --sigma-fs 0.6,1.0,1.5,2.5 \
  --vmc-every 25 --vmc-n 50000 --n-eval 50000 \
  --seed "$WINNER_SEED" --tag "$TAG_STAGE2" \
  --resume "$WINNER_CKPT" --no-pretrained 2>&1 | tee -a "$LOG_FILE"
RC1=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] Stage2 finished rc=${RC1}" | tee -a "$LOG_FILE"
if [[ $RC1 -ne 0 ]]; then
  exit $RC1
fi

STAGE2_CKPT="${ROOT}/results/arch_colloc/${TAG_STAGE2}.pt"
if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "[$(date '+%F %T')] ERROR: stage2 checkpoint missing: ${STAGE2_CKPT}" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date '+%F %T')] Starting Phase 5 tail stage3 from ${TAG_STAGE2}" | tee -a "$LOG_FILE"
CUDA_MANUAL_DEVICE="$GPU" PYTHONUNBUFFERED=1 python3.11 src/run_weak_form.py \
  --mode bf --n-elec 6 --omega 0.1 \
  --epochs 500 --lr 1e-4 --lr-jas 1e-5 \
  --n-coll 4096 --oversample 16 --micro-batch 512 \
  --loss-type reinforce --direct-weight 0.0 --clip-el 5.0 --grad-clip 1.0 \
  --reward-normalize \
  --rollback-jump-sigma 3 --rollback-decay 0.95 \
  --adaptive-proposal --gmm-components 16 --gmm-refit-every 15 \
  --sigma-fs 0.6,1.0,1.5,2.5 \
  --vmc-every 25 --vmc-n 50000 --n-eval 50000 \
  --seed "$WINNER_SEED" --tag "$TAG_STAGE3" \
  --resume "$STAGE2_CKPT" --no-pretrained 2>&1 | tee -a "$LOG_FILE"
RC2=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] Stage3 finished rc=${RC2}" | tee -a "$LOG_FILE"
exit $RC2