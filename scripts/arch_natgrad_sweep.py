#!/usr/bin/env python3
"""
Architecture sweep with natural gradients: N=6 ω=1.0 from scratch.

Tests the user's hypothesis that CTNN-style Jastrow may outperform VCycle,
and explores BF capacity. All runs use natgrad (lr=3e-2, damping=1e-3).
All from scratch (--no-pretrained) since architecture changes break checkpoint
compatibility. 800 epochs to give enough convergence time.

  GPU 0: CTNN Jastrow (64 hidden, 2 MP) + default BF (128h)
  GPU 4: Bigger VCycle Jastrow (48 hidden, 2 down/up) + default BF (128h)
  GPU 5: Default VCycle Jastrow + wider BF (256h, 4 layers)
  GPU 7: CTNN Jastrow (96 hidden, 3 MP) + wider BF (192h)
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
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_arch_natgrad_sweep"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

# Common natgrad flags for all jobs
COMMON = [
    "--mode", "bf",
    "--n-elec", "6", "--omega", "1.0",
    "--epochs", "800",
    "--n-coll", "4096",
    "--oversample", "8", "--micro-batch", "512",
    "--natural-grad",
    "--lr", "3e-2", "--lr-jas", "3e-3",
    "--fisher-damping", "1e-3",
    "--fisher-ema", "0.95",
    "--fisher-probes", "4",
    "--fisher-subsample", "256",
    "--nat-momentum", "0.9",
    "--grad-clip", "1.0",
    "--clip-el", "5.0", "--direct-weight", "0.0",
    "--vmc-every", "50", "--vmc-n", "10000",
    "--n-eval", "25000",
    "--seed", "42",
    "--no-pretrained",
]

JOBS = [
    # ── GPU 0: CTNN Jastrow (64 hidden, 2 MP) + default BF ──
    # Tests: does plain CTNN Jastrow beat VCycle?
    # CTNN has 2 message-passing rounds with 64-dim features — richer interactions
    {
        "name": "ctnn_jas64_bf128",
        "gpu": "0",
        "tag": "arch_ctnn_jas64_v1",
        "flags": [
            "--jas-arch", "ctnn",
            "--jas-hidden", "64",
            "--jas-mp-steps", "2",
        ],
    },
    # ── GPU 4: Bigger VCycle (48 hidden, 2 passes) + default BF ──
    # Tests: is the current VCycle (24 hidden, 1 pass) just too small?
    # This gives ~4x params in Jastrow while keeping VCycle structure
    {
        "name": "big_vcycle48_bf128",
        "gpu": "4",
        "tag": "arch_big_vcycle48_v1",
        "flags": [
            "--jas-arch", "vcycle",
            "--jas-hidden", "48",
            "--jas-mp-steps", "2",
        ],
    },
    # ── GPU 5: Default VCycle Jastrow + wider BF (256h, 4 layers) ──
    # Tests: does more BF capacity help? The BF shifts coordinates before
    # the Slater determinant — more capacity = more flexible nodal surface
    {
        "name": "vcycle24_bf256",
        "gpu": "5",
        "tag": "arch_wide_bf256_v1",
        "flags": [
            "--bf-hidden", "256",
            "--bf-layers", "4",
        ],
    },
    # ── GPU 7: CTNN Jastrow (96h, 3 MP) + wider BF (192h) ──
    # Tests: does scaling both help? More total capacity with natgrad
    {
        "name": "ctnn_jas96_bf192",
        "gpu": "7",
        "tag": "arch_ctnn_jas96_bf192_v1",
        "flags": [
            "--jas-arch", "ctnn",
            "--jas-hidden", "96",
            "--jas-mp-steps", "3",
            "--bf-hidden", "192",
        ],
    },
]


def main():
    print(f"Architecture + natgrad sweep — {len(JOBS)} jobs")
    print(f"Output: {OUTDIR}")
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

        with open(logfile, "w") as lf:
            lf.write(f"# {job['name']} — GPU {gpu}\n")
            lf.write(f"# {full_cmd}\n\n")

        proc = subprocess.Popen(
            ["bash", "-c", f"{full_cmd} >> {logfile} 2>&1"],
            start_new_session=True,
        )
        procs.append((job, proc))
        time.sleep(3)

    print(f"\n  All {len(procs)} jobs launched.")
    print(f"  Monitor: tail -f {LOGDIR}/*.log")
    print(f"  PIDs: {[p.pid for _, p in procs]}")

    # Wait for all
    results = []
    for job, proc in procs:
        rc = proc.wait()
        logfile = LOGDIR / f"{job['tag']}.log"
        final_E = final_err = "?"
        restored_E = restored_err = "?"
        try:
            lines = logfile.read_text().splitlines()
            for line in reversed(lines):
                if "*** Final:" in line and final_E == "?":
                    final_E = line.split("E =")[1].split("±")[0].strip() if "E =" in line else "?"
                    final_err = line.split("err =")[1].strip().split("%")[0].strip() + "%" if "err =" in line else "?"
                if "Restored VMC-best" in line and restored_E == "?":
                    restored_E = line.split("E=")[1].split()[0] if "E=" in line else "?"
                    restored_err = line.split("err=")[1].split("%")[0] + "%" if "err=" in line else "?"
        except Exception:
            pass

        result = {
            "name": job["name"],
            "tag": job["tag"],
            "gpu": job["gpu"],
            "rc": rc,
            "final_E": final_E,
            "final_err": final_err,
            "restored_E": restored_E,
            "restored_err": restored_err,
        }
        results.append(result)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{job['name']}] {status}  final={final_E} ({final_err})  best_vmc={restored_E} ({restored_err})")

    summary_path = OUTDIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
