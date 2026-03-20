#!/usr/bin/env python3
"""
Full SR sweep: Woodbury vs CG on N=6 ω=1.0.

All resume from adam_to_natgrad_v2.pt (best final E=20.167, err=+0.038%).
Budget: 2 hours max per job.

  GPU 1: Woodbury SR, lr=5e-3, damping=1e-3, 512 SR samples, 600 epochs
  GPU 2: Woodbury SR, lr=1e-2, damping=5e-3→1e-4 anneal, 512 SR samples, 600 epochs
  GPU 5: CG SR, lr=5e-3, 15 CG iters, damping=1e-3, 512 SR samples, 600 epochs
  GPU 7: CG SR, lr=1e-2, 20 CG iters, damping=5e-3→1e-4 anneal, 512 SR samples, 600 epochs
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_sr_sweep"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"
CKPT = str(RESULTS / "adam_to_natgrad_v2.pt")

# Common flags: continuation from best checkpoint, default arch, natgrad on
COMMON = [
    "--mode", "bf",
    "--n-elec", "6", "--omega", "1.0",
    "--epochs", "600",
    "--n-coll", "4096",
    "--oversample", "8", "--micro-batch", "512",
    "--natural-grad",
    "--nat-momentum", "0.9",
    "--grad-clip", "1.0",
    "--clip-el", "5.0", "--direct-weight", "0.0",
    "--vmc-every", "40", "--vmc-n", "12000",
    "--n-eval", "25000",
    "--seed", "42",
    "--resume", CKPT,
]

JOBS = [
    # ── GPU 1: Woodbury, conservative LR ──
    # Lower LR (5e-3) with full SR should be more stable.
    # Constant damping 1e-3, trust region 0.5 to be safe.
    {
        "name": "woodbury_lr5e3",
        "gpu": "1",
        "tag": "sr_woodbury_lr5e3_v1",
        "flags": [
            "--sr-mode", "woodbury",
            "--lr", "5e-3", "--lr-jas", "5e-4",
            "--fisher-damping", "1e-3",
            "--fisher-subsample", "512",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
        ],
    },
    # ── GPU 2: Woodbury, higher LR + damping anneal ──
    # Start with heavy damping (5e-3), anneal down to 1e-4 over 300 epochs.
    # This is the SR analog of "warmup then refine".
    {
        "name": "woodbury_anneal",
        "gpu": "2",
        "tag": "sr_woodbury_anneal_v1",
        "flags": [
            "--sr-mode", "woodbury",
            "--lr", "1e-2", "--lr-jas", "1e-3",
            "--fisher-damping", "5e-3",
            "--fisher-damping-end", "1e-4",
            "--fisher-damping-anneal", "300",
            "--fisher-subsample", "512",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
        ],
    },
    # ── GPU 5: CG, conservative ──
    # 15 CG iterations, lower LR. CG doesn't need to form full O matrix
    # explicitly (though our impl does for simplicity — still cheaper per
    # iteration since it's matrix-free in the CG loop).
    {
        "name": "cg_lr5e3",
        "gpu": "5",
        "tag": "sr_cg_lr5e3_v1",
        "flags": [
            "--sr-mode", "cg",
            "--lr", "5e-3", "--lr-jas", "5e-4",
            "--fisher-damping", "1e-3",
            "--fisher-subsample", "512",
            "--sr-cg-iters", "15",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
        ],
    },
    # ── GPU 7: CG, higher LR + anneal + more CG iters ──
    # 20 CG iterations for tighter solve. Damping anneal like Woodbury job.
    {
        "name": "cg_anneal",
        "gpu": "7",
        "tag": "sr_cg_anneal_v1",
        "flags": [
            "--sr-mode", "cg",
            "--lr", "1e-2", "--lr-jas", "1e-3",
            "--fisher-damping", "5e-3",
            "--fisher-damping-end", "1e-4",
            "--fisher-damping-anneal", "300",
            "--fisher-subsample", "512",
            "--sr-cg-iters", "20",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
        ],
    },
]


def main():
    print(f"Full SR sweep — {len(JOBS)} jobs")
    print(f"Resuming from: {CKPT}")
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
        try:
            lines = logfile.read_text().splitlines()
            for line in reversed(lines):
                if "*** Final:" in line and final_E == "?":
                    final_E = line.split("E =")[1].split("±")[0].strip()
                    final_err = line.split("err =")[1].strip().split("%")[0].strip() + "%"
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
        print(f"  [{job['name']}] {status}  final={final_E} ({final_err})")

    summary_path = OUTDIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
