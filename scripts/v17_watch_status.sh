#!/usr/bin/env bash
set -euo pipefail

ROOT="/itf-fi-ml/home/aleksns/Thesis_repo"
LOGDIR="${ROOT}/outputs/2026-03-27_1036_campaign_v17_n12_8h/logs"

while true; do
  clear
  date
  echo "==== v17 N12 8h Campaign Status ===="
  echo

  if [ ! -d "${LOGDIR}" ]; then
    echo "Log directory not found: ${LOGDIR}"
    sleep 30
    continue
  fi

  for f in "${LOGDIR}"/v17_n12w0005_bridge_*.log; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    prof=${b##*_}
    prof=${prof%.log}

    latest=$(grep -E '^  \[[[:space:]]*[0-9]+' "$f" | tail -n 1 || true)
    skips=$(grep -c 'ESS=.*SKIP' "$f" || true)
    fin=$(grep -E '\*\*\* Final:' "$f" | tail -n 1 || true)

    echo "[Bridge ${prof}]"
    echo "  latest: ${latest:-N/A}"
    echo "  skips:  ${skips}"
    if [ -n "$fin" ]; then
      echo "  final:  $fin"
    fi
  done

  echo
  for f in "${LOGDIR}"/v17_n12w0001_transfer_*.log; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    prof=${b##*_}
    prof=${prof%.log}

    latest=$(grep -E '^  \[[[:space:]]*[0-9]+' "$f" | tail -n 1 || true)
    skips=$(grep -c 'ESS=.*SKIP' "$f" || true)
    fin=$(grep -E '\*\*\* Final:' "$f" | tail -n 1 || true)

    echo "[Transfer ${prof}]"
    echo "  latest: ${latest:-N/A}"
    echo "  skips:  ${skips}"
    if [ -n "$fin" ]; then
      echo "  final:  $fin"
    fi
  done

  echo
  echo "GPU snapshot:"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

  sleep 60
done
