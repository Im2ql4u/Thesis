#!/usr/bin/env python3
"""
N=20 + low-omega campaign with Langevin proposal refinement.

All proposals launched simultaneously on 8 GPUs:

GPU 0: Proposal 1a — N=20 ω=1.0 tiny backflow (resume bf-hidden=64)       ~10h
GPU 1: Proposal 3a — N=20 ω=0.1 Langevin + Jastrow                       ~18h
GPU 2: Proposal 3b — N=20 ω=0.1 Langevin + tiny backflow (bf-hidden=48)   ~22h
GPU 3: Proposal 2  — N=20 ω=0.1 no-ESS-floor + rollback (Jastrow)         ~16h
GPU 4: Proposal 4  — N=12 ω=0.01 Adam+ESS cascade from ω=0.1              ~7h
GPU 5: Proposal 3c — N=6 ω=0.001 Langevin test (from +0.352%)             ~4h
GPU 6: Proposal 3d — N=6 ω=0.001 Adam+ESS resume (proven recipe)          ~4h
GPU 7: Proposal 3e — N=20 ω=0.1 Langevin + Jastrow seed 2                 ~18h

Follow-ups:
  GPU 0 → N=20 ω=0.5 cascade from ω=1.0 (~10h)
  GPU 4 → N=12 ω=0.001 cascade from ω=0.01 (~10h)
  GPU 5 → N=6 ω=0.001 Langevin + backflow (~6h)
  GPU 6 → N=12 ω=0.01 Langevin test (~6h)

DMC references:
  N=6  ω=0.01: 0.69036   ω=0.001: 0.140832
  N=12 ω=0.1: 12.26984   ω=0.01: (unknown)
  N=20 ω=1.0: 155.889    ω=0.5: 93.871    ω=0.1: 29.9779
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
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_n20_langevin_campaign"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

# ═════════════════════════════════════════════════════════════
# Checkpoints
# ═════════════════════════════════════════════════════════════
CKPT = {
    "n20w1_bf64":    str(RESULTS / "w1_n20w1_cont.pt"),                              # N=20 ω=1.0, bf-hidden=64
    "n20w01_jas":    str(RESULTS / "camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt"),  # N=20 ω=0.1 Jastrow
    "n20w1_jas":     str(RESULTS / "camp_jastrow_transfer_baseline_n20_o1p0_s11.pt"),    # N=20 ω=1.0 Jastrow
    "n12w01":        str(RESULTS / "long_n12w01_v3.pt"),                             # N=12 ω=0.1, +0.155%
    "n6w0001_low":   str(RESULTS / "regime_low_L_n6_o0p001_s11.pt"),                 # N=6 ω=0.001, +0.352%
    "n6w001_coal":   str(RESULTS / "coal_g0_r1_n6_o0p01_s11.pt"),                    # N=6 ω=0.01, +0.205%
}


def adam_ess(lr="5e-4", lr_jas="5e-5", clip_el="5.0", qtrim="0.02",
            ess_floor="0.0", ess_max="0", rollback_decay="1.0",
            rollback_err="0.0", rollback_jump="0.0"):
    """Adam + ESS adaptive + rollback recipe."""
    args = [
        "--lr", lr, "--lr-jas", lr_jas,
        "--direct-weight", "0.0",
        "--clip-el", clip_el, "--reward-qtrim", qtrim,
    ]
    if float(ess_floor) > 0:
        args += ["--ess-floor-ratio", ess_floor, "--ess-oversample-max", ess_max,
                 "--ess-oversample-step", "2", "--ess-resample-tries", "2"]
    if float(rollback_decay) < 1.0:
        args += ["--rollback-decay", rollback_decay,
                 "--rollback-err-pct", rollback_err,
                 "--rollback-jump-sigma", rollback_jump]
    return args


# ═════════════════════════════════════════════════════════════
# PRIMARY JOBS
# ═════════════════════════════════════════════════════════════
JOBS = [
    # GPU 0: N=20 ω=1.0 tiny backflow resume (bf-hidden=64, existing ckpt)
    {
        "name": "n20_w1_bf64", "gpu": "0", "tag": "camp2_n20w1_bf64",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
            "--bf-hidden", "64", "--bf-layers", "3",
            "--epochs", "1500",
            "--n-coll", "2048", "--oversample", "8", "--micro-batch", "256",
            *adam_ess(lr="5e-4", lr_jas="5e-5", clip_el="3.0", qtrim="0.01",
                      rollback_decay="0.9", rollback_err="20.0", rollback_jump="6.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000", "--seed", "42",
            "--resume", CKPT["n20w1_bf64"],
        ],
    },
    # GPU 1: N=20 ω=0.1 Langevin + Jastrow (test if Langevin fixes sampling)
    {
        "name": "n20_w01_lang_jas", "gpu": "1", "tag": "camp2_n20w01_lang_jas",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "4", "--micro-batch", "512",
            "--langevin-steps", "10", "--langevin-step-size", "0.005",
            *adam_ess(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.85", rollback_err="25.0", rollback_jump="8.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000", "--seed", "42",
            "--resume", CKPT["n20w01_jas"],
        ],
    },
    # GPU 2: N=20 ω=0.1 Langevin + tiny backflow (most ambitious)
    {
        "name": "n20_w01_lang_bf", "gpu": "2", "tag": "camp2_n20w01_lang_bf",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "0.1",
            "--bf-hidden", "48", "--bf-layers", "2",
            "--epochs", "1500",
            "--n-coll", "2048", "--oversample", "4", "--micro-batch", "256",
            "--langevin-steps", "10", "--langevin-step-size", "0.005",
            *adam_ess(lr="5e-4", lr_jas="5e-5", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.85", rollback_err="25.0", rollback_jump="8.0"),
            "--vmc-every", "50", "--vmc-n", "8000",
            "--n-eval", "20000", "--seed", "42",
            "--init-jas", CKPT["n20w01_jas"],
            "--no-pretrained",
        ],
    },
    # GPU 3: N=20 ω=0.1 no-ESS-floor Jastrow with rollback only
    {
        "name": "n20_w01_nofloor", "gpu": "3", "tag": "camp2_n20w01_nofloor",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *adam_ess(lr="3e-4", lr_jas="3e-4", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.85", rollback_err="30.0", rollback_jump="8.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000", "--seed", "42",
            "--resume", CKPT["n20w01_jas"],
        ],
    },
    # GPU 4: N=12 ω=0.01 Adam+ESS cascade from ω=0.1
    {
        "name": "n12_w001_adam", "gpu": "4", "tag": "camp2_n12w001_adam",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.01",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            *adam_ess(lr="6e-4", lr_jas="6e-5", clip_el="5.0", qtrim="0.02",
                      ess_floor="0.12", ess_max="16",
                      rollback_decay="0.85", rollback_err="15.0", rollback_jump="6.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", CKPT["n12w01"], "--init-bf", CKPT["n12w01"],
            "--no-pretrained",
        ],
    },
    # GPU 5: N=6 ω=0.001 Langevin test (from +0.352% checkpoint)
    {
        "name": "n6_w0001_lang", "gpu": "5", "tag": "camp2_n6w0001_lang",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "2000",
            "--n-coll", "8192", "--oversample", "4", "--micro-batch", "512",
            "--langevin-steps", "10", "--langevin-step-size", "0.003",
            *adam_ess(lr="3e-4", lr_jas="3e-5", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.85", rollback_err="10.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n6w0001_low"],
        ],
    },
    # GPU 6: N=6 ω=0.001 Adam+ESS resume (proven recipe, push further)
    {
        "name": "n6_w0001_adam", "gpu": "6", "tag": "camp2_n6w0001_adam",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "2000",
            "--n-coll", "8192", "--oversample", "10", "--micro-batch", "512",
            *adam_ess(lr="3e-4", lr_jas="3e-5", clip_el="5.0", qtrim="0.02",
                      ess_floor="0.12", ess_max="16",
                      rollback_decay="0.85", rollback_err="10.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n6w0001_low"],
        ],
    },
    # GPU 7: N=20 ω=0.1 Langevin + Jastrow seed 2 (robustness check)
    {
        "name": "n20_w01_lang_jas2", "gpu": "7", "tag": "camp2_n20w01_lang_jas2",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "4", "--micro-batch", "512",
            "--langevin-steps", "15", "--langevin-step-size", "0.003",
            *adam_ess(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.85", rollback_err="25.0", rollback_jump="8.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000", "--seed", "77",
            "--resume", CKPT["n20w01_jas"],
        ],
    },
]

# ═════════════════════════════════════════════════════════════
# FOLLOW-UP JOBS
# ═════════════════════════════════════════════════════════════
FOLLOWUP_JOBS = [
    # After GPU 0 (N=20 ω=1.0, ~10h): cascade to ω=0.5
    {
        "name": "n20_w05_cascade", "gpu": "0", "tag": "camp2_n20w05_cascade",
        "prereq_tag": "camp2_n20w1_bf64",
        "init_from": ["camp2_n20w1_bf64"],
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "0.5",
            "--bf-hidden", "64", "--bf-layers", "3",
            "--epochs", "1500",
            "--n-coll", "2048", "--oversample", "8", "--micro-batch", "256",
            *adam_ess(lr="5e-4", lr_jas="5e-5", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.85", rollback_err="20.0", rollback_jump="8.0"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "25000", "--seed", "42",
            "--no-pretrained",
        ],
    },
    # After GPU 4 (N=12 ω=0.01, ~7h): cascade to ω=0.001
    {
        "name": "n12_w0001_cascade", "gpu": "4", "tag": "camp2_n12w0001_cascade",
        "prereq_tag": "camp2_n12w001_adam",
        "init_from": ["camp2_n12w001_adam"],
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.001",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            *adam_ess(lr="4e-4", lr_jas="4e-5", clip_el="5.0", qtrim="0.02",
                      ess_floor="0.10", ess_max="18",
                      rollback_decay="0.85", rollback_err="20.0", rollback_jump="8.0"),
            "--vmc-every", "50", "--vmc-n", "8000",
            "--n-eval", "25000", "--seed", "42",
            "--no-pretrained",
        ],
    },
    # After GPU 5 (N=6 ω=0.001 Langevin, ~4h): N=6 ω=0.001 Langevin + higher res
    {
        "name": "n6_w0001_lang_hires", "gpu": "5", "tag": "camp2_n6w0001_lang_hires",
        "prereq_tag": "camp2_n6w0001_lang",
        "init_from": ["camp2_n6w0001_lang", "camp2_n6w0001_adam"],
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "3000",
            "--n-coll", "16384", "--oversample", "3", "--micro-batch", "512",
            "--langevin-steps", "15", "--langevin-step-size", "0.002",
            *adam_ess(lr="1e-4", lr_jas="1e-5", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.9", rollback_err="5.0", rollback_jump="4.0"),
            "--vmc-every", "30", "--vmc-n", "20000",
            "--n-eval", "50000", "--seed", "42",
            "--no-pretrained",
        ],
    },
    # After GPU 6 (N=6 ω=0.001 Adam, ~4h): N=12 ω=0.01 Langevin test
    {
        "name": "n12_w001_lang", "gpu": "6", "tag": "camp2_n12w001_lang",
        "prereq_tag": "camp2_n6w0001_adam",
        "init_from": ["camp2_n12w001_adam", "long_n12w01_v3"],
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.01",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "4", "--micro-batch", "512",
            "--langevin-steps", "10", "--langevin-step-size", "0.005",
            *adam_ess(lr="5e-4", lr_jas="5e-5", clip_el="5.0", qtrim="0.02",
                      rollback_decay="0.85", rollback_err="20.0", rollback_jump="8.0"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "25000", "--seed", "42",
            "--no-pretrained",
        ],
    },
]


# ═════════════════════════════════════════════════════════════
# Infrastructure (same as long_trainer_campaign.py)
# ═════════════════════════════════════════════════════════════
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
    for tag in candidates:
        path = RESULTS / f"{tag}.pt"
        if path.exists():
            return str(path)
    return None


def main():
    started = datetime.now()
    print(f"N=20 + Langevin campaign — {started:%Y-%m-%d %H:%M}")
    print(f"Output: {OUTDIR}")
    print(f"Primary: {len(JOBS)}, Follow-ups: {len(FOLLOWUP_JOBS)}")
    print()

    (OUTDIR / "plan.json").write_text(json.dumps(
        {"jobs": JOBS, "followups": FOLLOWUP_JOBS, "started": str(started)},
        indent=2, default=str))

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
    STATUS_INTERVAL = 600
    last_status = time.time()

    while active or followup_active:
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
