#!/usr/bin/env bash
# B4: independent blocked MCMC re-evaluation of the 30-run N=6 collocation campaign.
cd ~/Thesis_repo
source /etc/profile.d/z00_lmod.sh >/dev/null 2>&1
OUT=results/analysis/2026-09-12_colloc_blocked_eval
GPUS=(0 1 3 4 5 6)
declare -A W=( [1p0]=1.0 [0p5]=0.5 [0p1]=0.1 [0p01]=0.01 [0p001]=0.001 )
jobs_for_gpu() { :; }
i=0
for wt in 0p001 0p01 0p1 0p5 1p0; do for r in robust baseline; do for s in 42 137 314; do
  tag=${r}_p1fix_n6w${wt}_s${s}
  echo "$tag ${W[$wt]}" 
done; done; done > $OUT/queue.txt
for g in "${GPUS[@]}"; do
  ( idx=0
    while read -r tag w; do
      if (( idx % ${#GPUS[@]} == $(printf "%s\n" "${GPUS[@]}" | grep -n "^$g$" | cut -d: -f1) - 1 )); then
        CUDA_MANUAL_DEVICE=$g PYTHONUNBUFFERED=1 python3.11 scripts/eval_collocation_blocked.py \
          --ckpt results/arch_colloc/$tag.pt --omega $w --chains 4 --sweeps 200 \
          --out $OUT/$tag.json > $OUT/logs/$tag.log 2>&1
        echo "$(date +%T) gpu=$g done $tag rc=$?" >> $OUT/progress.log
      fi
      idx=$((idx+1))
    done < $OUT/queue.txt ) &
done
wait
echo "$(date +%T) ALL DONE" >> $OUT/progress.log
