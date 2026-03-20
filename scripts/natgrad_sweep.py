#!/usr/bin/env python3
"""
Natural gradient sweep: 4 parallel GPU experiments.

Tests the key unknowns for diagonal Fisher preconditioning:
  GPU 0: natgrad baseline  — lr=3e-2, damping=1e-3 (the suggested default)
  GPU 4: natgrad high-lr   — lr=1e-1, damping=1e-3 (is larger step better?)
  GPU 5: natgrad low-damp  — lr=3e-2, damping=1e-4 (less regularization)
  GPU 7: adam baseline     — lr=5e-4, no Fisher    (control: current best recipe)

All start from bf_ctnn_vcycle.pt, N=6, ω=1.0, 500 epochs.
The Adam baseline uses the known-good hyperparams for fair comparison.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_natgrad_sweep"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"
RESUME = str(RESULTS / "bf_ctnn_vcycle.pt")

# Common flags for all runs
COMMON = [
    "--mode", "bf",
    "--n-elec", "6",
    "--omega", "1.0",
    "--epochs", "500",
    "--n-coll", "4096",
    "--oversample", "8",
    "--micro-batch", "512",
    "--grad-clip", "1.0",
    "--clip-el", "5.0",
    "--direct-weight", "0.0",
    "--vmc-every", "40",
    "--vmc-n", "10000",
    "--n-eval", "20000",
    "--seed", "42",
    "--resume", RESUME,
]

JOBS = [
    {
        "name": "natgrad_base",
        "gpu": "0",
        "tag": "natgrad_base_v1",
        "flags": [
            "--natural-grad",
            "--lr", "3e-2",
            "--lr-jas", "3e-3",
            "--fisher-damping", "1e-3",
            "--fisher-ema", "0.95",
            "--fisher-probes", "4",
            "--fisher-subsample", "256",
            "--nat-momentum", "0.9",
        ],
    },
    {
        "name": "natgrad_highlr",
        "gpu": "4",
        "tag": "natgrad_highlr_v1",
        "flags": [
            "--natural-grad",
            "--lr", "1e-1",
            "--lr-jas", "1e-2",
            "--fisher-damping", "1e-3",
            "--fisher-ema", "0.95",
            "--fisher-probes", "4",
            "--fisher-subsample", "256",
            "--nat-momentum", "0.9",
        ],
    },
    {
        "name": "natgrad_lowdamp",
        "gpu": "5",
        "tag": "natgrad_lowdamp_v1",
        "flags": [
            "--natural-grad",
            "--lr", "3e-2",
            "--lr-jas", "3e-3",
            "--fisher-damping", "1e-4",
            "--fisher-ema", "0.95",
            "--fisher-probes", "4",
            "--fisher-subsample", "256",
            "--nat-momentum", "0.9",
        ],
    },
    {
        "name": "adam_control",
        "gpu": "7",
        "tag": "adam_control_v1",
        "flags": [
            "--lr", "5e-4",
            "--lr-jas", "5e-5",
        ],
    },
]


def main():
    print(f"Natural gradient sweep — {len(JOBS)} jobs")
    print(f"Output: {OUTDIR}")
    print(f"Resume: {RESUME}")
    print()

    plan_path = OUTDIR / "plan.json"
    plan_path.write_text(json.dumps(JOBS, indent=2))

    procs = []
    for job in JOBS:
        tag = job["tag"]
        gpu = job["gpu"]
        logfile = LOGDIR / f"{tag}.log"

        cmd_parts = (
            [f"cd {SRC}"]
            + [MODULE_CMD]
            + [
                f"CUDA_MANUAL_DEVICE={gpu} python run_weak_form.py "
                + " ".join(COMMON + ["--tag", tag] + job["flags"])
            ]
        )
        full_cmd = "; ".join(cmd_parts)

        print(f"  [{job['name']}] GPU={gpu} tag={tag}")
        print(f"    log: {logfile}")

        with open(logfile, "w") as lf:
            lf.write(f"# {job['name']} — GPU {gpu}\n")
            lf.write(f"# {full_cmd}\n\n")

        proc = subprocess.Popen(
            ["bash", "-c", f"{full_cmd} >> {logfile} 2>&1"],
            start_new_session=True,
        )
        procs.append((job, proc))
        time.sleep(2)  # stagger GPU init

    print(f"\n  All {len(procs)} jobs launched.")
    print(f"  Monitor: tail -f {LOGDIR}/*.log")
    print(f"  PIDs: {[p.pid for _, p in procs]}")

    # Wait for all
    results = []
    for job, proc in procs:
        rc = proc.wait()
        logfile = LOGDIR / f"{job['tag']}.log"
        # Extract final energy from log
        final_E = "?"
        final_err = "?"
        try:
            lines = logfile.read_text().splitlines()
            for line in reversed(lines):
                if "*** Final:" in line:
                    final_E = line.split("E =")[1].split("±")[0].strip() if "E =" in line else "?"
                    final_err = line.split("err =")[1].strip().split("%")[0].strip() + "%" if "err =" in line else "?"
                    break
        except Exception:
            pass

        result = {
            "name": job["name"],
            "tag": job["tag"],
            "gpu": job["gpu"],
            "rc": rc,
            "final_E": final_E,
            "final_err": final_err,
        }
        results.append(result)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{job['name']}] {status}  E={final_E}  err={final_err}")

    summary_path = OUTDIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
