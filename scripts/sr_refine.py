#!/usr/bin/env python3
"""
SR refinement: close the last gap to DMC from CG-SR checkpoints.

All resume from sr_cg_anneal_v1.pt (best final: 20.165, +0.029%).
Budget: ≤2 hours per job.

Refinement strategies:
  GPU 1: Low LR (2e-3), more SR samples (1024), 25 CG iters, tight damping 1e-4, 8192 coll pts
  GPU 2: Very low LR (1e-3), 1024 SR samples, 30 CG iters, damping 5e-5, 8192 coll pts
  GPU 5: From sr_cg_lr5e3_v1 — moderate LR (3e-3), anneal damping 1e-3→5e-5, 1024 SR, 20 CG
  GPU 7: Low LR (2e-3), 2048 SR samples, 15 CG iters, damping 1e-4, 8192 coll pts, clip_el=3.0
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_sr_refine"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"
CKPT_BEST = str(RESULTS / "sr_cg_anneal_v1.pt")
CKPT_ALT = str(RESULTS / "sr_cg_lr5e3_v1.pt")

JOBS = [
    # ── GPU 1: Careful refinement — low LR, many SR samples, tight CG ──
    # 1024 SR samples at 49K params → better O matrix.
    # 25 CG iters for tighter solve. Damping 1e-4 (already low).
    # 8192 collocation points for lower-noise E_L.
    {
        "name": "refine_careful",
        "gpu": "1",
        "tag": "sr_refine_careful_v1",
        "resume": CKPT_BEST,
        "flags": [
            "--epochs", "600",
            "--n-coll", "8192",
            "--oversample", "6", "--micro-batch", "512",
            "--sr-mode", "cg",
            "--lr", "2e-3", "--lr-jas", "2e-4",
            "--fisher-damping", "1e-4",
            "--fisher-subsample", "1024",
            "--sr-cg-iters", "25",
            "--sr-max-param-change", "0.03",
            "--sr-trust-region", "0.3",
            "--nat-momentum", "0.95",
            "--grad-clip", "0.5",
            "--clip-el", "5.0",
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000",
        ],
    },
    # ── GPU 2: Ultra-conservative — tiny LR, very tight damping ──
    # Almost no damping (5e-5) — trusting the CG truncation for regularization.
    # 30 CG iters for very tight solve. Very low LR (1e-3).
    {
        "name": "refine_ultra",
        "gpu": "2",
        "tag": "sr_refine_ultra_v1",
        "resume": CKPT_BEST,
        "flags": [
            "--epochs", "800",
            "--n-coll", "8192",
            "--oversample", "6", "--micro-batch", "512",
            "--sr-mode", "cg",
            "--lr", "1e-3", "--lr-jas", "1e-4",
            "--fisher-damping", "5e-5",
            "--fisher-subsample", "1024",
            "--sr-cg-iters", "30",
            "--sr-max-param-change", "0.02",
            "--sr-trust-region", "0.2",
            "--nat-momentum", "0.95",
            "--grad-clip", "0.5",
            "--clip-el", "5.0",
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000",
        ],
    },
    # ── GPU 5: From alt checkpoint — anneal into fine territory ──
    # Start from sr_cg_lr5e3_v1 (best VMC 20.160, +0.001%).
    # Anneal damping 1e-3 → 5e-5 over 200 epochs.
    {
        "name": "refine_alt_anneal",
        "gpu": "5",
        "tag": "sr_refine_alt_v1",
        "resume": CKPT_ALT,
        "flags": [
            "--epochs", "600",
            "--n-coll", "8192",
            "--oversample", "6", "--micro-batch", "512",
            "--sr-mode", "cg",
            "--lr", "3e-3", "--lr-jas", "3e-4",
            "--fisher-damping", "1e-3",
            "--fisher-damping-end", "5e-5",
            "--fisher-damping-anneal", "200",
            "--fisher-subsample", "1024",
            "--sr-cg-iters", "20",
            "--sr-max-param-change", "0.03",
            "--sr-trust-region", "0.3",
            "--nat-momentum", "0.95",
            "--grad-clip", "0.5",
            "--clip-el", "5.0",
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000",
        ],
    },
    # ── GPU 7: Heavy sampling + aggressive E_L clipping ──
    # 2048 SR samples (expensive but precise O matrix).
    # Aggressive E_L clipping (3.0 MAD) to suppress nodal noise.
    # 8192 collocation points.
    {
        "name": "refine_heavy_clip",
        "gpu": "7",
        "tag": "sr_refine_clip_v1",
        "resume": CKPT_BEST,
        "flags": [
            "--epochs", "500",
            "--n-coll", "8192",
            "--oversample", "6", "--micro-batch", "512",
            "--sr-mode", "cg",
            "--lr", "2e-3", "--lr-jas", "2e-4",
            "--fisher-damping", "1e-4",
            "--fisher-subsample", "2048",
            "--sr-cg-iters", "15",
            "--sr-max-param-change", "0.03",
            "--sr-trust-region", "0.3",
            "--nat-momentum", "0.95",
            "--grad-clip", "0.5",
            "--clip-el", "3.0",
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000",
        ],
    },
]

# Common flags shared by all
COMMON = [
    "--mode", "bf",
    "--n-elec", "6", "--omega", "1.0",
    "--natural-grad",
    "--direct-weight", "0.0",
    "--seed", "42",
]


def main():
    print(f"SR refinement — {len(JOBS)} jobs")
    print(f"Output: {OUTDIR}")
    print()

    plan_path = OUTDIR / "plan.json"
    plan_path.write_text(json.dumps(JOBS, indent=2))

    procs = []
    for job in JOBS:
        tag = job["tag"]
        gpu = job["gpu"]
        logfile = LOGDIR / f"{tag}.log"

        all_flags = COMMON + ["--tag", tag, "--resume", job["resume"]] + job["flags"]

        cmd_parts = [
            f"cd {SRC}",
            MODULE_CMD,
            f"CUDA_MANUAL_DEVICE={gpu} python run_weak_form.py " + " ".join(all_flags),
        ]
        full_cmd = "; ".join(cmd_parts)

        print(f"  [{job['name']}] GPU={gpu} resume={Path(job['resume']).name}")

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

    results = []
    for job, proc in procs:
        rc = proc.wait()
        logfile = LOGDIR / f"{job['tag']}.log"
        final_E = final_err = best_E = best_err = "?"
        try:
            lines = logfile.read_text().splitlines()
            for line in reversed(lines):
                if "*** Final:" in line and final_E == "?":
                    final_E = line.split("E =")[1].split("±")[0].strip()
                    final_err = line.split("err =")[1].strip().split("%")[0].strip() + "%"
                if "Restored VMC-best" in line and best_E == "?":
                    best_E = line.split("E=")[1].split()[0]
                    best_err = line.split("err=")[1].split("%")[0] + "%"
        except Exception:
            pass

        result = {
            "name": job["name"],
            "tag": job["tag"],
            "rc": rc,
            "final_E": final_E,
            "final_err": final_err,
            "best_E": best_E,
            "best_err": best_err,
        }
        results.append(result)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{job['name']}] {status}  final={final_E} ({final_err})  best={best_E} ({best_err})")

    summary_path = OUTDIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
