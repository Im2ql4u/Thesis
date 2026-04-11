from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_WEAK_FORM = ROOT / "src" / "run_weak_form.py"
RESULTS_ARCH_COLLOC = ROOT / "results" / "arch_colloc"

SKIP_RE = re.compile(r"ESS=.*-> SKIP")
ROLLBACK_RE = re.compile(r"rollback:")


def _load_hist(checkpoint_path: Path) -> list[dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hist = ckpt.get("hist", [])
    if not isinstance(hist, list):
        raise RuntimeError(f"Checkpoint hist is not a list: {checkpoint_path}")
    return hist


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run run_weak_form.py and export checkpoint hist + skip/rollback diagnostics "
            "to JSON/JSONL artifacts"
        )
    )
    parser.add_argument("--tag", required=True, help="Training tag used by run_weak_form")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/consistency_campaign/phase0",
        help="Directory for log and diagnostic artifacts",
    )
    parser.add_argument(
        "--run-weak-form-args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to run_weak_form.py",
    )
    args, unknown = parser.parse_known_args()

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_args = []
    if args.run_weak_form_args:
        run_args.extend(args.run_weak_form_args)
    if unknown:
        run_args.extend(unknown)
    if run_args and run_args[0] == "--":
        run_args = run_args[1:]

    if "--tag" in run_args:
        raise RuntimeError("Do not pass --tag in --run-weak-form-args; use --tag on this script")

    cmd = [sys.executable, str(RUN_WEAK_FORM), "--tag", args.tag, *run_args]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"{args.tag}_{ts}.log"

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT, text=True)

    checkpoint_path = RESULTS_ARCH_COLLOC / f"{args.tag}.pt"
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_weak_form failed (rc={proc.returncode}). "
            f"Check log: {log_path}"
        )
    if not checkpoint_path.exists():
        raise RuntimeError(f"Expected checkpoint missing: {checkpoint_path}")

    hist = _load_hist(checkpoint_path)

    jsonl_path = out_dir / f"{args.tag}_epochs.jsonl"
    _write_jsonl(jsonl_path, hist)

    log_text = log_path.read_text(encoding="utf-8")
    n_skip = len(SKIP_RE.findall(log_text))
    n_rollback = len(ROLLBACK_RE.findall(log_text))
    max_ep = max((int(r.get("ep", -1)) for r in hist), default=-1)

    summary = {
        "tag": args.tag,
        "timestamp": ts,
        "command": cmd,
        "log_path": str(log_path),
        "checkpoint_path": str(checkpoint_path),
        "epochs_logged": len(hist),
        "max_epoch_index": max_ep,
        "ess_skip_events": n_skip,
        "rollback_events": n_rollback,
    }

    if hist:
        summary["final_entry"] = hist[-1]

    summary_path = out_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run log: {log_path}")
    print(f"Epoch JSONL: {jsonl_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
