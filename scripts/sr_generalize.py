#!/usr/bin/env python3
"""
SR generalization: fix ω=1.0, push ω=0.5, try ω=0.1 properly.

GPU 0: N=6 ω=1.0 — replicate the CG anneal recipe that got 20.165.
        Resume from sr_cg_anneal_v1.pt, lr=1e-2, damping anneal 5e-3→1e-4.
        600 epochs, 50K final eval.

GPU 3: N=6 ω=0.5 — CG-SR from bf_ctnn_vcycle warm start.
        Fresh start with CG-SR, lr=5e-3, damping anneal, 600 epochs.

GPU 4: N=6 ω=0.5 — same but lr=1e-2, more aggressive.

GPU 7: N=6 ω=0.1 — CG-SR from scratch (no warm start).
        Needs proper initialization for this regime. 600 epochs.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_sr_gen_v2"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

JOBS = [
    # ── GPU 0: N=6 ω=1.0 — replicate best recipe ──
    # The CG anneal (lr=1e-2, damping 5e-3→1e-4, 600 ep) got 20.165.
    # Now continue with same recipe + larger final eval.
    {
        "name": "n6w1_best_recipe",
        "gpu": "0",
        "tag": "sr_n6w1_best_v2",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "1.0",
            "--tag", "sr_n6w1_best_v2",
            "--epochs", "600",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            "--natural-grad", "--sr-mode", "cg",
            "--lr", "1e-2", "--lr-jas", "1e-3",
            "--fisher-damping", "5e-3",
            "--fisher-damping-end", "1e-4",
            "--fisher-damping-anneal", "300",
            "--fisher-subsample", "512",
            "--sr-cg-iters", "15",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
            "--nat-momentum", "0.9",
            "--grad-clip", "1.0", "--clip-el", "5.0",
            "--direct-weight", "0.0",
            "--vmc-every", "40", "--vmc-n", "15000",
            "--n-eval", "50000",
            "--seed", "42",
            "--resume", str(RESULTS / "sr_cg_anneal_v1.pt"),
        ],
    },
    # ── GPU 3: N=6 ω=0.5 — CG-SR from warm start ──
    # Use bf_ctnn_vcycle.pt as init (ω=1.0 weights, arch matches).
    # Moderate LR, anneal damping.
    {
        "name": "n6w05_cgsr_v2",
        "gpu": "3",
        "tag": "sr_n6w05_v2",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--tag", "sr_n6w05_v2",
            "--epochs", "600",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            "--natural-grad", "--sr-mode", "cg",
            "--lr", "5e-3", "--lr-jas", "5e-4",
            "--fisher-damping", "1e-3",
            "--fisher-damping-end", "1e-4",
            "--fisher-damping-anneal", "200",
            "--fisher-subsample", "512",
            "--sr-cg-iters", "15",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
            "--nat-momentum", "0.9",
            "--grad-clip", "1.0", "--clip-el", "5.0",
            "--direct-weight", "0.0",
            "--vmc-every", "40", "--vmc-n", "12000",
            "--n-eval", "30000",
            "--seed", "42",
            "--init-jas", str(RESULTS / "bf_ctnn_vcycle.pt"),
            "--init-bf", str(RESULTS / "bf_ctnn_vcycle.pt"),
            "--no-pretrained",
        ],
    },
    # ── GPU 4: N=6 ω=0.5 — higher LR ──
    # Same as above but lr=1e-2, more aggressive push.
    {
        "name": "n6w05_fast_v2",
        "gpu": "4",
        "tag": "sr_n6w05_fast_v2",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--tag", "sr_n6w05_fast_v2",
            "--epochs", "600",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            "--natural-grad", "--sr-mode", "cg",
            "--lr", "1e-2", "--lr-jas", "1e-3",
            "--fisher-damping", "5e-3",
            "--fisher-damping-end", "1e-4",
            "--fisher-damping-anneal", "300",
            "--fisher-subsample", "512",
            "--sr-cg-iters", "15",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
            "--nat-momentum", "0.9",
            "--grad-clip", "1.0", "--clip-el", "5.0",
            "--direct-weight", "0.0",
            "--vmc-every", "40", "--vmc-n", "12000",
            "--n-eval", "30000",
            "--seed", "42",
            "--init-jas", str(RESULTS / "bf_ctnn_vcycle.pt"),
            "--init-bf", str(RESULTS / "bf_ctnn_vcycle.pt"),
            "--no-pretrained",
        ],
    },
    # ── GPU 7: N=6 ω=0.1 — from scratch, proper ──
    # No warm start (ω=1.0 weights are useless here).
    # Higher LR, more oversample (particles spread more at low ω).
    {
        "name": "n6w01_scratch",
        "gpu": "7",
        "tag": "sr_n6w01_v2",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--tag", "sr_n6w01_v2",
            "--epochs", "600",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            "--natural-grad", "--sr-mode", "cg",
            "--lr", "1e-2", "--lr-jas", "1e-3",
            "--fisher-damping", "5e-3",
            "--fisher-damping-end", "1e-4",
            "--fisher-damping-anneal", "300",
            "--fisher-subsample", "512",
            "--sr-cg-iters", "15",
            "--sr-max-param-change", "0.05",
            "--sr-trust-region", "0.5",
            "--nat-momentum", "0.9",
            "--grad-clip", "1.0", "--clip-el", "5.0",
            "--direct-weight", "0.0",
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000",
            "--seed", "42",
            "--no-pretrained",
        ],
    },
]


def main():
    print(f"SR gen v2 — {len(JOBS)} jobs")
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

        result = {"name": job["name"], "tag": tag, "rc": rc,
                  "final_E": final_E, "final_err": final_err,
                  "best_E": best_E, "best_err": best_err}
        results.append(result)
        print(f"  [{job['name']}] rc={rc}  final={final_E} ({final_err})")

    (OUTDIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved.")


if __name__ == "__main__":
    main()
