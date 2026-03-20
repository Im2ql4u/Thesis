#!/usr/bin/env python3
"""
Long-running cascade campaign (24-30h).

All jobs launched independently. Follow-up jobs auto-launch when
prerequisites finish, reusing freed GPUs.

Key targets:
  - N=6: polish ω=1.0, 0.5 (200k eval); refine ω=0.1; push ω=0.01, ω=0.001
  - N=12: polish ω=1.0, 0.5; cascade to ω=0.1
  - Multiple seeds/recipes for ω=0.01 and ω=0.001

DMC references:
  N=6  ω=1.0: 20.15932    ω=0.5: 11.78484    ω=0.1: 3.55385
  N=6  ω=0.01: 0.69036    ω=0.001: 0.140832
  N=12 ω=1.0: 65.70010    ω=0.5: 39.15960    ω=0.1: 12.26984
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_long_campaign"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

# Best checkpoints
CKPT = {
    "n6w1":     str(RESULTS / "camp_n6w1_verify.pt"),       # +0.012%
    "n6w05":    str(RESULTS / "w1_n6w05_hisamp.pt"),        # -0.002%
    "n6w01":    str(RESULTS / "w1_n6w01_transfer.pt"),      # +0.321%
    "n6w001":   str(RESULTS / "n6w001_cascade.pt"),         # +9.46%
    "n12w1":    str(RESULTS / "w1_n12w1_xfer.pt"),          # +0.010%
    "n12w05":   str(RESULTS / "n12w05_cascade.pt"),         # +0.024%
}


def cgsr(lr="5e-3", lr_jas="5e-4", damping="1e-3", damping_end="1e-4",
         anneal="400", sub="512", cg="15", maxdp="0.03", trust="0.3",
         mom="0.95", gclip="0.5", clip_el="3.0"):
    return [
        "--natural-grad", "--sr-mode", "cg",
        "--lr", lr, "--lr-jas", lr_jas,
        "--fisher-damping", damping,
        "--fisher-damping-end", damping_end,
        "--fisher-damping-anneal", anneal,
        "--fisher-subsample", sub,
        "--sr-cg-iters", cg,
        "--sr-max-param-change", maxdp,
        "--sr-trust-region", trust,
        "--nat-momentum", mom,
        "--grad-clip", gclip, "--clip-el", clip_el,
        "--direct-weight", "0.0",
    ]


# ═══════════════════════════════════════════════════════════
# PRIMARY JOBS: launched immediately on GPUs 0-7
# ═══════════════════════════════════════════════════════════
JOBS = [
    # GPU 0: N=6 ω=0.01 — resume from +9.46%, push to <2%
    {
        "name": "n6_w001_long", "gpu": "0", "tag": "long_n6w001",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
            "--epochs", "4000",
            "--n-coll", "4096", "--oversample", "20", "--micro-batch", "512",
            *cgsr(lr="3e-3", lr_jas="3e-4", damping="5e-3",
                  damping_end="5e-5", anneal="2000",
                  sub="512", cg="15", maxdp="0.05", trust="0.5"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "40000", "--seed", "42",
            "--resume", CKPT["n6w001"],
        ],
    },
    # GPU 1: N=6 ω=0.01 — fresh from ω=0.1 init, different seed/recipe
    {
        "name": "n6_w001_v2", "gpu": "1", "tag": "long_n6w001_v2",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
            "--epochs", "4000",
            "--n-coll", "4096", "--oversample", "16", "--micro-batch", "512",
            *cgsr(lr="5e-3", lr_jas="5e-4", damping="5e-3",
                  damping_end="1e-4", anneal="1500",
                  sub="512", cg="15", maxdp="0.05", trust="0.5"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "40000", "--seed", "123",
            "--init-jas", CKPT["n6w01"], "--init-bf", CKPT["n6w01"],
            "--no-pretrained",
        ],
    },
    # GPU 2: N=6 ω=0.1 — refine from +0.321%
    {
        "name": "n6_w01_refine", "gpu": "2", "tag": "long_n6w01",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "2000",
            "--n-coll", "8192", "--oversample", "12", "--micro-batch", "512",
            *cgsr(lr="1e-3", lr_jas="1e-4", damping="1e-4",
                  damping_end="5e-5", anneal="800",
                  sub="1024", cg="20", maxdp="0.02", trust="0.2"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "50000", "--seed", "42",
            "--resume", CKPT["n6w01"],
        ],
    },
    # GPU 3: N=12 ω=0.1 — CASCADE from N=12 ω=0.5 (DMC=12.270)
    {
        "name": "n12_w01_cascade", "gpu": "3", "tag": "long_n12w01",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.1",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            *cgsr(lr="3e-3", lr_jas="3e-4", damping="5e-3",
                  damping_end="1e-4", anneal="800",
                  sub="512", cg="15", maxdp="0.05", trust="0.5"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", CKPT["n12w05"], "--init-bf", CKPT["n12w05"],
            "--no-pretrained",
        ],
    },
    # GPU 4: N=12 ω=1.0 — ultra-polish
    {
        "name": "n12_w1_polish", "gpu": "4", "tag": "long_n12w1",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "1500",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr(lr="5e-4", lr_jas="5e-5", damping="5e-5",
                  sub="1024", cg="20", maxdp="0.01", trust="0.1"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "50000", "--seed", "42",
            "--resume", CKPT["n12w1"],
        ],
    },
    # GPU 5: N=12 ω=0.5 — ultra-polish
    {
        "name": "n12_w05_polish", "gpu": "5", "tag": "long_n12w05",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.5",
            "--epochs", "1500",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr(lr="5e-4", lr_jas="5e-5", damping="5e-5",
                  sub="1024", cg="20", maxdp="0.01", trust="0.1"),
            "--vmc-every", "30", "--vmc-n", "12000",
            "--n-eval", "40000", "--seed", "42",
            "--resume", CKPT["n12w05"],
        ],
    },
    # GPU 6: N=6 ω=1.0 — final definitive number (200k eval)
    {
        "name": "n6_w1_final", "gpu": "6", "tag": "long_n6w1",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "1.0",
            "--epochs", "800",
            "--n-coll", "8192", "--oversample", "8", "--micro-batch", "512",
            *cgsr(lr="3e-4", lr_jas="3e-5", damping="5e-5",
                  sub="2048", cg="25", maxdp="0.01", trust="0.1"),
            "--vmc-every", "20", "--vmc-n", "30000",
            "--n-eval", "200000", "--seed", "42",
            "--resume", CKPT["n6w1"],
        ],
    },
    # GPU 7: N=6 ω=0.5 — final definitive number (200k eval)
    {
        "name": "n6_w05_final", "gpu": "7", "tag": "long_n6w05",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--epochs", "800",
            "--n-coll", "8192", "--oversample", "10", "--micro-batch", "512",
            *cgsr(lr="3e-4", lr_jas="3e-5", damping="5e-5",
                  sub="2048", cg="25", maxdp="0.01", trust="0.1"),
            "--vmc-every", "20", "--vmc-n", "30000",
            "--n-eval", "200000", "--seed", "42",
            "--resume", CKPT["n6w05"],
        ],
    },
]

# ═══════════════════════════════════════════════════════════
# FOLLOW-UP JOBS: auto-launch when prerequisite finishes
# ═══════════════════════════════════════════════════════════
FOLLOWUP_JOBS = [
    # After GPU 6 (N=6 ω=1.0, ~3h): cascade ω=0.001 from best ω=0.01
    {
        "name": "n6_w0001_cascade", "gpu": "6", "tag": "long_n6w0001",
        "prereq_tag": "long_n6w1",
        "init_from": ["long_n6w001", "long_n6w001_v2", "n6w001_cascade"],
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "5000",
            "--n-coll", "4096", "--oversample", "24", "--micro-batch", "512",
            *cgsr(lr="1e-3", lr_jas="1e-4", damping="5e-3",
                  damping_end="5e-5", anneal="2500",
                  sub="512", cg="15", maxdp="0.03", trust="0.3"),
            "--vmc-every", "60", "--vmc-n", "8000",
            "--n-eval", "25000", "--seed", "42",
            "--no-pretrained",
        ],
    },
    # After GPU 7 (N=6 ω=0.5, ~3h): N=6 ω=0.01 third variant
    {
        "name": "n6_w001_v3", "gpu": "7", "tag": "long_n6w001_v3",
        "prereq_tag": "long_n6w05",
        "init_from": ["long_n6w01", "w1_n6w01_transfer"],
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
            "--epochs", "4000",
            "--n-coll", "4096", "--oversample", "20", "--micro-batch", "512",
            *cgsr(lr="3e-3", lr_jas="3e-4", damping="5e-3",
                  damping_end="5e-5", anneal="2000",
                  sub="512", cg="15", maxdp="0.05", trust="0.5"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "40000", "--seed", "77",
            "--no-pretrained",
        ],
    },
    # After GPU 4 (N=12 ω=1.0, ~8h): N=12 ω=0.1 variant
    {
        "name": "n12_w01_v2", "gpu": "4", "tag": "long_n12w01_v2",
        "prereq_tag": "long_n12w1",
        "init_from": ["long_n12w05", "n12w05_cascade"],
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.1",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            *cgsr(lr="3e-3", lr_jas="3e-4", damping="5e-3",
                  damping_end="1e-4", anneal="800",
                  sub="512", cg="15", maxdp="0.05", trust="0.5"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "30000", "--seed", "123",
            "--no-pretrained",
        ],
    },
    # After GPU 5 (N=12 ω=0.5, ~6h): another N=12 ω=0.1 variant
    {
        "name": "n12_w01_v3", "gpu": "5", "tag": "long_n12w01_v3",
        "prereq_tag": "long_n12w05",
        "init_from": ["long_n12w05", "n12w05_cascade"],
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.1",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "14", "--micro-batch", "512",
            *cgsr(lr="5e-3", lr_jas="5e-4", damping="5e-3",
                  damping_end="1e-4", anneal="800",
                  sub="512", cg="15", maxdp="0.05", trust="0.5"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "30000", "--seed", "77",
            "--no-pretrained",
        ],
    },
    # After GPU 2 (N=6 ω=0.1, ~6h): ω=0.001 variant from refined ω=0.1
    {
        "name": "n6_w0001_v2", "gpu": "2", "tag": "long_n6w0001_v2",
        "prereq_tag": "long_n6w01",
        "init_from": ["long_n6w001", "long_n6w001_v2", "n6w001_cascade"],
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "5000",
            "--n-coll", "4096", "--oversample", "24", "--micro-batch", "512",
            *cgsr(lr="1e-3", lr_jas="1e-4", damping="1e-2",
                  damping_end="1e-4", anneal="2500",
                  sub="512", cg="15", maxdp="0.03", trust="0.3"),
            "--vmc-every", "60", "--vmc-n", "8000",
            "--n-eval", "25000", "--seed", "123",
            "--no-pretrained",
        ],
    },
]


def launch_job(job, logdir):
    tag = job["tag"]
    gpu = job["gpu"]
    logfile = logdir / f"{tag}.log"
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


def find_best_init(candidates):
    """Find best available checkpoint from list of tag candidates."""
    for tag in candidates:
        path = RESULTS / f"{tag}.pt"
        if path.exists():
            return str(path)
    return None


def main():
    started = datetime.now()
    print(f"Long campaign (24-30h) — {started:%Y-%m-%d %H:%M}")
    print(f"Output: {OUTDIR}")
    print(f"Primary: {len(JOBS)}, Follow-ups: {len(FOLLOWUP_JOBS)}")
    print()

    (OUTDIR / "plan.json").write_text(json.dumps(
        {"jobs": JOBS, "followups": FOLLOWUP_JOBS, "started": str(started)},
        indent=2, default=str))

    # Launch primaries
    active = {}
    for job in JOBS:
        proc = launch_job(job, LOGDIR)
        active[job["tag"]] = (job, proc)
        print(f"  GPU {job['gpu']}: [{job['name']}] tag={job['tag']} pid={proc.pid}")
        time.sleep(3)
    print(f"\nAll {len(active)} primary jobs launched.")
    print(f"Monitor: tail -f {LOGDIR}/*.log\n")
    sys.stdout.flush()

    finished = set()
    followup_launched = set()
    followup_active = {}

    STATUS_INTERVAL = 600  # 10 min
    last_status = time.time()

    while active or followup_active:
        time.sleep(30)

        # Check completions
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

        for tag in list(followup_active.keys()):
            job, proc = followup_active[tag]
            if proc.poll() is not None:
                logfile = LOGDIR / f"{tag}.log"
                E, err = parse_final(logfile)
                elapsed = datetime.now() - started
                err_s = f"{err:+.3f}%" if err is not None else "?"
                E_s = f"{E:.5f}" if E is not None else "?"
                print(f"  [{elapsed}] DONE (followup) [{job['name']}] E={E_s} err={err_s}")
                with open(logfile, "a") as lf:
                    lf.write(f"\n# Completed: {datetime.now():%Y-%m-%d %H:%M:%S} rc={proc.returncode}\n")
                finished.add(tag)
                del followup_active[tag]
                sys.stdout.flush()

        # Launch eligible follow-ups
        for fj in FOLLOWUP_JOBS:
            if fj["tag"] in followup_launched:
                continue
            if fj.get("prereq_tag") not in finished:
                continue

            init_ckpt = find_best_init(fj.get("init_from", []))
            cmd = list(fj["cmd"])
            if init_ckpt:
                has_init = any(c in ("--init-bf", "--resume") for c in cmd)
                if not has_init:
                    cmd.extend(["--init-jas", init_ckpt, "--init-bf", init_ckpt])

            fj_copy = {**fj, "cmd": cmd}
            proc = launch_job(fj_copy, LOGDIR)
            followup_active[fj["tag"]] = (fj_copy, proc)
            followup_launched.add(fj["tag"])
            elapsed = datetime.now() - started
            init_name = Path(init_ckpt).name if init_ckpt else "scratch"
            print(f"  [{elapsed}] FOLLOWUP [{fj['name']}] GPU={fj['gpu']} "
                  f"init={init_name} pid={proc.pid}")
            sys.stdout.flush()

        # Periodic status
        if time.time() - last_status > STATUS_INTERVAL:
            last_status = time.time()
            elapsed = datetime.now() - started
            print(f"\n  --- Status at {elapsed} ---")
            for tag, (job, _) in list(active.items()) + list(followup_active.items()):
                ep, err = parse_last_epoch(LOGDIR / f"{tag}.log")
                err_s = f"{err:+.3f}%" if err is not None else "?"
                print(f"    GPU {job['gpu']}: [{job['name']}] ep={ep} err={err_s}")
            print(f"  Active: {len(active)}+{len(followup_active)} | Done: {len(finished)}\n")
            sys.stdout.flush()

    # Final summary
    elapsed = datetime.now() - started
    print(f"\n{'='*60}")
    print(f"  CAMPAIGN COMPLETE — {elapsed}")
    print(f"{'='*60}\n")

    results = []
    for tag in sorted(finished):
        E, err = parse_final(LOGDIR / f"{tag}.log")
        results.append({"tag": tag, "E": E, "err": err})
        err_s = f"{err:+.3f}%" if err is not None else "?"
        E_s = f"{E:.5f}" if E is not None else "?"
        print(f"  {tag}: E={E_s} err={err_s}")

    (OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Results: {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
