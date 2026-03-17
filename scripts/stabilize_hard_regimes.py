#!/usr/bin/env python3
"""Stabilized CG-SR rollout for hard regimes (high N / low omega).

Focuses on branches that showed instability in the cascade campaign by adding:
- tighter SR trust limits (smaller max step + trust region)
- stronger damping with slower anneal
- ESS-adaptive resampling + min-ESS gating
- tempered/clipped resampling weights
- rollback on unstable jumps
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_stabilized_hardregimes"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

JOBS = [
    {
        "name": "n6w01_stable_from_transfer",
        "gpu": "0",
        "tag": "st_n6w01_from_transfer",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "900",
            "--n-coll", "4096", "--oversample", "20", "--micro-batch", "512",
            "--natural-grad", "--sr-mode", "cg",
            "--lr", "1.5e-3", "--lr-jas", "1.5e-4",
            "--fisher-damping", "3e-3",
            "--fisher-damping-end", "2e-4",
            "--fisher-damping-anneal", "500",
            "--fisher-subsample", "1024",
            "--sr-cg-iters", "20",
            "--sr-max-param-change", "0.015",
            "--sr-trust-region", "0.15",
            "--nat-momentum", "0.8",
            "--grad-clip", "0.35", "--clip-el", "2.5",
            "--direct-weight", "0.0",
            "--ess-floor-ratio", "0.06",
            "--ess-oversample-max", "36",
            "--ess-oversample-step", "4",
            "--ess-resample-tries", "4",
            "--resample-weight-temp", "0.70",
            "--resample-logw-clip-q", "0.98",
            "--min-ess", "64",
            "--rollback-decay", "0.70",
            "--rollback-err-pct", "8.0",
            "--rollback-jump-sigma", "4.0",
            "--vmc-every", "30", "--vmc-n", "12000",
            "--n-eval", "50000",
            "--seed", "42",
            "--resume", str(RESULTS / "w1_n6w01_transfer.pt"),
        ],
    },
    {
        "name": "n12w05_stable_transfer",
        "gpu": "1",
        "tag": "st_n12w05_resume",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.5",
            "--epochs", "1000",
            "--n-coll", "6144", "--oversample", "16", "--micro-batch", "512",
            "--natural-grad", "--sr-mode", "cg",
            "--lr", "1e-3", "--lr-jas", "1e-4",
            "--fisher-damping", "5e-3",
            "--fisher-damping-end", "5e-4",
            "--fisher-damping-anneal", "600",
            "--fisher-subsample", "1024",
            "--sr-cg-iters", "20",
            "--sr-max-param-change", "0.012",
            "--sr-trust-region", "0.12",
            "--nat-momentum", "0.8",
            "--grad-clip", "0.30", "--clip-el", "2.0",
            "--direct-weight", "0.0",
            "--ess-floor-ratio", "0.08",
            "--ess-oversample-max", "32",
            "--ess-oversample-step", "4",
            "--ess-resample-tries", "4",
            "--resample-weight-temp", "0.70",
            "--resample-logw-clip-q", "0.98",
            "--min-ess", "0",
            "--rollback-decay", "0.70",
            "--rollback-err-pct", "0.0",
            "--rollback-jump-sigma", "4.0",
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
            "--resume", str(RESULTS / "camp_n12w05_smoke.pt"),
        ],
    },
    {
        "name": "n20w1_stable_resume",
        "gpu": "2",
        "tag": "st_n20w1_resume",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
            "--bf-hidden", "64", "--bf-layers", "2",
            "--epochs", "1000",
            "--n-coll", "2048", "--oversample", "16", "--micro-batch", "256",
            "--natural-grad", "--sr-mode", "cg",
            "--lr", "8e-4", "--lr-jas", "8e-5",
            "--fisher-damping", "8e-3",
            "--fisher-damping-end", "8e-4",
            "--fisher-damping-anneal", "700",
            "--fisher-subsample", "256",
            "--sr-cg-iters", "12",
            "--sr-max-param-change", "0.010",
            "--sr-trust-region", "0.10",
            "--nat-momentum", "0.75",
            "--grad-clip", "0.25", "--clip-el", "2.0",
            "--direct-weight", "0.0",
            "--ess-floor-ratio", "0.08",
            "--ess-oversample-max", "40",
            "--ess-oversample-step", "4",
            "--ess-resample-tries", "4",
            "--resample-weight-temp", "0.65",
            "--resample-logw-clip-q", "0.98",
            "--min-ess", "0",
            "--rollback-decay", "0.65",
            "--rollback-err-pct", "0.0",
            "--rollback-jump-sigma", "3.0",
            "--vmc-every", "50", "--vmc-n", "5000",
            "--n-eval", "15000",
            "--seed", "42",
            "--resume", str(RESULTS / "camp_n20w1_smoke.pt"),
        ],
    },
]


def main():
    print(f"Stabilized hard-regime rollout — {len(JOBS)} jobs")
    print(f"Output: {OUTDIR}")
    print()

    (OUTDIR / "plan.json").write_text(json.dumps(JOBS, indent=2))

    procs = []
    for job in JOBS:
        tag = job["tag"]
        gpu = job["gpu"]
        logfile = LOGDIR / f"{tag}.log"

        full_cmd = f"cd {SRC}; {MODULE_CMD}; CUDA_MANUAL_DEVICE={gpu} python run_weak_form.py " + " ".join(job["cmd"])
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

    results = []
    for job, proc in procs:
        rc = proc.wait()
        logfile = LOGDIR / f"{job['tag']}.log"
        final_E = final_err = best_E = best_err = "?"
        try:
            for line in reversed(logfile.read_text().splitlines()):
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
        print(f"  [{job['name']}] rc={rc}  final={final_E} ({final_err})")

    (OUTDIR / "summary.json").write_text(json.dumps(results, indent=2))
    print("\nSummary saved.")


if __name__ == "__main__":
    main()
