#!/usr/bin/env python3
"""
Adam + ESS adaptive refinement for ultra-low omega.

Targets:
  A) GPU 0: N=6 ω=0.001 resume from +0.352% (regime_low_L checkpoint)
  B) GPU 1: N=6 ω=0.001 cascade from ω=0.01 +0.205% (coal checkpoint)
  C) GPU 5: N=6 ω=0.01  polish from +0.205% → push below +0.1%

All use Adam (NOT CG-SR) + ESS adaptive sampling + instability rollback.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_adam_lowomega_refine"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

JOBS = [
    # A) GPU 0: ω=0.001 resume from +0.352%
    {
        "name": "n6_w0001_refine_A",
        "gpu": "0",
        "tag": "adam_n6w0001_refine_A",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "3000",
            "--n-coll", "8192", "--oversample", "10", "--micro-batch", "512",
            "--lr", "4e-4", "--lr-jas", "4e-5",
            "--direct-weight", "0.0",
            "--clip-el", "5.0", "--reward-qtrim", "0.02",
            "--ess-floor-ratio", "0.12",
            "--ess-oversample-max", "16",
            "--ess-oversample-step", "2",
            "--ess-resample-tries", "2",
            "--rollback-decay", "0.85",
            "--rollback-err-pct", "15.0",
            "--rollback-jump-sigma", "6.0",
            "--vmc-every", "40", "--vmc-n", "12000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", str(RESULTS / "regime_low_L_n6_o0p001_s11.pt"),
        ],
    },
    # B) GPU 1: ω=0.001 cascade from ω=0.01 +0.205%
    {
        "name": "n6_w0001_cascade_B",
        "gpu": "1",
        "tag": "adam_n6w0001_cascade_B",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            "--lr", "6e-4", "--lr-jas", "6e-5",
            "--direct-weight", "0.0",
            "--clip-el", "5.0", "--reward-qtrim", "0.02",
            "--ess-floor-ratio", "0.12",
            "--ess-oversample-max", "16",
            "--ess-oversample-step", "2",
            "--ess-resample-tries", "2",
            "--rollback-decay", "0.85",
            "--rollback-err-pct", "15.0",
            "--rollback-jump-sigma", "6.0",
            "--vmc-every", "40", "--vmc-n", "12000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", str(RESULTS / "coal_g0_r1_n6_o0p01_s11.pt"),
            "--init-bf", str(RESULTS / "coal_g0_r1_n6_o0p01_s11.pt"),
            "--no-pretrained",
        ],
    },
    # C) GPU 5: ω=0.01 polish from +0.205%
    {
        "name": "n6_w001_polish_C",
        "gpu": "5",
        "tag": "adam_n6w001_polish_C",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
            "--epochs", "2000",
            "--n-coll", "8192", "--oversample", "10", "--micro-batch", "512",
            "--lr", "3e-4", "--lr-jas", "3e-5",
            "--direct-weight", "0.0",
            "--clip-el", "5.0", "--reward-qtrim", "0.02",
            "--ess-floor-ratio", "0.12",
            "--ess-oversample-max", "14",
            "--ess-oversample-step", "2",
            "--ess-resample-tries", "2",
            "--rollback-decay", "0.85",
            "--rollback-err-pct", "15.0",
            "--rollback-jump-sigma", "6.0",
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "40000", "--seed", "42",
            "--resume", str(RESULTS / "coal_g0_r1_n6_o0p01_s11.pt"),
        ],
    },
]


def launch_job(job):
    tag = job["tag"]
    gpu = job["gpu"]
    logfile = LOGDIR / f"{tag}.log"
    cmd = list(job["cmd"])
    if "--tag" not in cmd:
        cmd.extend(["--tag", tag])
    full_cmd = (
        f"cd {SRC}; {MODULE_CMD}; "
        f"CUDA_MANUAL_DEVICE={gpu} python run_weak_form.py " + " ".join(cmd)
    )
    with open(logfile, "w") as lf:
        lf.write(f"# {job['name']} — GPU {gpu}\n")
        lf.write(f"# Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        lf.write(f"# {full_cmd}\n\n")
    return subprocess.Popen(
        ["bash", "-c", f"{full_cmd} >> {logfile} 2>&1"],
        start_new_session=True,
    )


def parse_last_epoch(logfile):
    try:
        for line in reversed(logfile.read_text().splitlines()):
            if "err=" in line and line.strip().startswith("["):
                ep = int(line.split("[")[1].split("]")[0].strip())
                err = line.split("err=")[1].split("%")[0]
                return ep, float(err.replace("+", ""))
    except Exception:
        pass
    return -1, None


def parse_final(logfile):
    try:
        for line in reversed(logfile.read_text().splitlines()):
            if "*** Final:" in line:
                E = line.split("E =")[1].split("±")[0].strip()
                err = line.split("err =")[1].strip().split("%")[0]
                return float(E), float(err.replace("+", ""))
    except Exception:
        pass
    return None, None


def main():
    started = datetime.now()
    print(f"Adam low-omega refine — {started:%Y-%m-%d %H:%M}")
    print(f"Output: {OUTDIR}")
    print()

    active = {}
    for job in JOBS:
        proc = launch_job(job)
        active[job["tag"]] = (job, proc)
        print(f"  GPU {job['gpu']}: [{job['name']}] tag={job['tag']} pid={proc.pid}")
        time.sleep(3)

    print(f"\nAll {len(active)} jobs launched.")
    print(f"Monitor: tail -f {LOGDIR}/*.log\n")
    sys.stdout.flush()

    finished = set()
    STATUS_INTERVAL = 600  # 10 min

    last_status = time.time()

    while active:
        time.sleep(30)

        for tag in list(active.keys()):
            job, proc = active[tag]
            if proc.poll() is not None:
                logfile = LOGDIR / f"{tag}.log"
                E, err = parse_final(logfile)
                elapsed = datetime.now() - started
                err_s = f"{err:+.3f}%" if err is not None else "?"
                E_s = f"{E:.5f}" if E is not None else "?"
                print(f"  [{elapsed}] DONE [{job['name']}] E={E_s} err={err_s} rc={proc.returncode}")
                with open(logfile, "a") as lf:
                    lf.write(f"\n# Completed: {datetime.now():%Y-%m-%d %H:%M:%S} rc={proc.returncode}\n")
                finished.add(tag)
                del active[tag]
                sys.stdout.flush()

        if time.time() - last_status > STATUS_INTERVAL:
            last_status = time.time()
            elapsed = datetime.now() - started
            print(f"\n  --- Status at {elapsed} ---")
            for tag, (job, _) in active.items():
                ep, err = parse_last_epoch(LOGDIR / f"{tag}.log")
                err_s = f"{err:+.3f}%" if err is not None else "?"
                print(f"    GPU {job['gpu']}: [{job['name']}] ep={ep} err={err_s}")
            print(f"  Active: {len(active)} | Done: {len(finished)}\n")
            sys.stdout.flush()

    elapsed = datetime.now() - started
    print(f"\n{'='*60}")
    print(f"  ADAM LOW-OMEGA REFINE COMPLETE — {elapsed}")
    print(f"{'='*60}\n")

    for tag in sorted(finished):
        E, err = parse_final(LOGDIR / f"{tag}.log")
        err_s = f"{err:+.3f}%" if err is not None else "?"
        E_s = f"{E:.5f}" if E is not None else "?"
        print(f"  {tag}: E={E_s} err={err_s}")


if __name__ == "__main__":
    main()
