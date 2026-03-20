#!/usr/bin/env python3
"""
Natural gradient sweep v2: continuation + harder regimes.

Round 2 experiments:
  GPU 0: N=6 ω=1.0 continuation from natgrad_base_v1 — lower LR, more epochs, push to DMC
  GPU 4: N=6 ω=1.0 continuation from adam_control_v1 with natgrad — can natgrad rescue Adam?
  GPU 5: N=6 ω=0.5 natgrad from bf_ctnn_vcycle — intermediate ω, easier than 0.01
  GPU 7: N=12 ω=0.5 natgrad jastrow-only transfer — does natgrad help at higher N?

The N=6 continuations use 800 epochs with lower LR to squeeze out the basin.
The new-regime runs use 600 epochs with the base natgrad recipe.
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
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_natgrad_sweep_v2"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

JOBS = [
    # ── GPU 0: Continue natgrad_base with lower LR to push into basin ──
    # The base run's best probe was 20.1547 — continuation should drive it to 20.16x final
    {
        "name": "n6_w1_natgrad_cont",
        "gpu": "0",
        "tag": "natgrad_cont_v2",
        "flags": [
            "--mode", "bf",
            "--n-elec", "6", "--omega", "1.0",
            "--epochs", "800",
            "--n-coll", "6144",  # more collocation points for refinement
            "--oversample", "8", "--micro-batch", "512",
            "--natural-grad",
            "--lr", "1e-2", "--lr-jas", "1e-3",  # lower LR for basin sharpening
            "--fisher-damping", "5e-4",  # tighter damping
            "--fisher-ema", "0.97",  # longer memory
            "--fisher-probes", "6",  # more probes for better Fisher estimate
            "--fisher-subsample", "384",
            "--nat-momentum", "0.9",
            "--grad-clip", "1.0",
            "--clip-el", "5.0", "--direct-weight", "0.0",
            "--vmc-every", "40", "--vmc-n", "12000",
            "--n-eval", "25000",
            "--seed", "42",
            "--resume", str(RESULTS / "natgrad_base_v1.pt"),
        ],
    },
    # ── GPU 4: Take adam_control checkpoint, continue with natgrad ──
    # Tests: can natgrad improve an already-good Adam state?
    {
        "name": "n6_w1_adam_to_natgrad",
        "gpu": "4",
        "tag": "adam_to_natgrad_v2",
        "flags": [
            "--mode", "bf",
            "--n-elec", "6", "--omega", "1.0",
            "--epochs", "600",
            "--n-coll", "6144",
            "--oversample", "8", "--micro-batch", "512",
            "--natural-grad",
            "--lr", "1e-2", "--lr-jas", "1e-3",
            "--fisher-damping", "1e-3",
            "--fisher-ema", "0.95",
            "--fisher-probes", "4",
            "--fisher-subsample", "256",
            "--nat-momentum", "0.9",
            "--grad-clip", "1.0",
            "--clip-el", "5.0", "--direct-weight", "0.0",
            "--vmc-every", "40", "--vmc-n", "12000",
            "--n-eval", "25000",
            "--seed", "42",
            "--resume", str(RESULTS / "adam_control_v1.pt"),
        ],
    },
    # ── GPU 5: N=6 ω=0.5 — intermediate frequency, natgrad from bf_ctnn_vcycle ──
    # ω=0.5 is not trivially easy but not the ω=0.01 catastrophe.
    # Tests whether natgrad handles the increased sampling challenge.
    {
        "name": "n6_w05_natgrad",
        "gpu": "5",
        "tag": "n6_w05_natgrad_v2",
        "flags": [
            "--mode", "bf",
            "--n-elec", "6", "--omega", "0.5",
            "--epochs", "600",
            "--n-coll", "4096",
            "--oversample", "10",  # slightly more oversampling for lower ω
            "--micro-batch", "512",
            "--natural-grad",
            "--lr", "3e-2", "--lr-jas", "3e-3",
            "--fisher-damping", "1e-3",
            "--fisher-ema", "0.95",
            "--fisher-probes", "4",
            "--fisher-subsample", "256",
            "--nat-momentum", "0.9",
            "--grad-clip", "1.0",
            "--clip-el", "5.0", "--direct-weight", "0.0",
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "20000",
            "--seed", "42",
            # Use bf_ctnn_vcycle for Jastrow init; BF architecture will need to
            # adapt to different omega but the weights provide a warm start.
            "--init-jas", str(RESULTS / "bf_ctnn_vcycle.pt"),
            "--init-bf", str(RESULTS / "bf_ctnn_vcycle.pt"),
        ],
    },
    # ── GPU 7: N=12 ω=0.5 jastrow-only with natgrad ──
    # N=12 is where Adam historically starts to struggle.
    # Jastrow-only because BF architecture was trained for N=6.
    # DMC reference for N=12 ω=0.5: 39.15960
    {
        "name": "n12_w05_natgrad_jas",
        "gpu": "7",
        "tag": "n12_w05_natgrad_jas_v2",
        "flags": [
            "--mode", "jastrow",
            "--n-elec", "12", "--omega", "0.5",
            "--epochs", "600",
            "--n-coll", "4096",
            "--oversample", "10",
            "--micro-batch", "512",
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
            "--n-eval", "20000",
            "--seed", "42",
            "--no-pretrained",  # N=12: no N=6 pretrained checkpoint
        ],
    },
]


def main():
    print(f"Natural gradient sweep v2 — {len(JOBS)} jobs")
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
                + " ".join(["--tag", tag] + job["flags"])
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
