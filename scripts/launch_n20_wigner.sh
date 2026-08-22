#!/usr/bin/env bash
# Close the Q1 scaling gap: N=20 at omega=0.01 (Wigner), warm-started from the omega=0.1 checkpoints,
# both backflows x both seeds. Uses the memory fixes (chunked exact Laplacian, no polish inflation at
# N>=20, batch 256). One run per GPU on 0-3.
set -u
cd /itf-fi-ml/home/aleksns/Thesis
source /etc/profile.d/lmod.sh 2>/dev/null
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null
STAMP=2026-07-16_scaling
run() {  # gpu arch seed
  local gpu=$1 arch=$2 seed=$3
  local init=results/analysis/${STAMP}/N20_${arch}bf_s${seed}_w0p1/checkpoint.pt
  local out=results/analysis/${STAMP}/N20_${arch}bf_s${seed}_w0p01
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -u scripts/run_phase_analysis.py --N 20 --omega 0.01 --arch pinn \
    --backflow --backflow-arch "$arch" --paradigm vmc --optimizer adam \
    --bf-scale-init 0.7 --bf-zero-init-last 0 \
    --steps 2500 --n-seg 4 --polish-steps 500 --sr-polish-steps 500 \
    --batch 256 --eval-samples 256 --final-samples 2048 --align-samples 512 \
    --seed "$seed" --init "$init" --outdir "$out" > "$out/train.log" 2>&1
  echo "[N20 $arch s$seed w0.01] done $(grep -oE 'E=[0-9.]+ *\([-+][0-9.]+%\)' "$out/train.log"|tail -1)"
}
run 0 ctnn 0 & run 1 ctnn 1 & run 2 conv 0 & run 3 conv 1 &
wait
echo "=== N=20 WIGNER DONE $(date +%H:%M:%S) ==="
