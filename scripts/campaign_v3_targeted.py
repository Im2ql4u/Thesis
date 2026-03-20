#!/usr/bin/env python3
"""Campaign v3: Targeted runs with fixed Langevin + strategic polishing.

Key changes from v2:
- Langevin step size no longer divided by omega (was exploding at low ω)
- Added gradient clipping in Langevin dynamics
- Added NaN guard on importance weights
- Strategic: Polish best N=20 Jastrow checkpoints, gentle N=12 cascade,
  Langevin validation on N=6 where we have Adam baseline
"""
import subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / (time.strftime("%Y-%m-%d_%H%M") + "_campaign_v3")
LOGDIR = Path(OUTDIR) / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

# ═══════════════════════════════════════════════════════════
# Best checkpoints
# ═══════════════════════════════════════════════════════════
CKPT = {
    # N=20 Jastrow (best results)
    "n20w1_jas":  str(RESULTS / "camp_jastrow_transfer_baseline_n20_o1p0_s11.pt"),    # +2.627%
    "n20w05_jas": str(RESULTS / "camp_jastrow_transfer_stabilized_n20_o0p5_s11.pt"),  # +6.992%
    "n20w01_jas": str(RESULTS / "20260318_0858_n20w01_keep_a.pt"),                    # +5.902%
    # N=12 BF+Jastrow
    "n12w01_bf":  str(RESULTS / "20260318_1149_n12w01_keep.pt"),                      # +0.122%
    # N=6 Adam baseline
    "n6w0001":    str(RESULTS / "camp2_n6w0001_adam.pt"),                              # +0.224%
}


def adam_args(lr="5e-4", lr_jas="5e-5", clip_el="5.0", qtrim="0.02",
             ess_floor="0.0", ess_max="0",
             rollback_decay="1.0", rollback_err="0.0", rollback_jump="0.0"):
    """Adam optimizer recipe with optional ESS adaptive + rollback."""
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


# ═══════════════════════════════════════════════════════════
# JOBS
# ═══════════════════════════════════════════════════════════
JOBS = [
    # GPU 0: N=20 ω=1.0 Jastrow polish from +2.6%
    # Low LR, relaxed rollback, long run
    {
        "name": "n20w1_polish", "gpu": "0", "tag": "v3_n20w1_polish",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "1.0",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            *adam_args(lr="1e-4", lr_jas="1e-4", clip_el="3.0", qtrim="0.01",
                       rollback_decay="0.92", rollback_err="15.0", rollback_jump="5.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n20w1_jas"],
        ],
    },
    # GPU 1: N=20 ω=1.0 Langevin+Jastrow — test Langevin at N=20
    {
        "name": "n20w1_langevin", "gpu": "1", "tag": "v3_n20w1_lang",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "1.0",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            "--langevin-steps", "15", "--langevin-step-size", "0.01",
            *adam_args(lr="1e-4", lr_jas="1e-4", clip_el="3.0", qtrim="0.01",
                       rollback_decay="0.92", rollback_err="15.0", rollback_jump="5.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n20w1_jas"],
        ],
    },
    # GPU 2: N=20 ω=0.5 Jastrow polish from +7%
    {
        "name": "n20w05_polish", "gpu": "2", "tag": "v3_n20w05_polish",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.5",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            *adam_args(lr="2e-4", lr_jas="2e-4", clip_el="5.0", qtrim="0.02",
                       rollback_decay="0.90", rollback_err="20.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n20w05_jas"],
        ],
    },
    # GPU 3: N=20 ω=0.1 Jastrow polish from +5.9%
    {
        "name": "n20w01_polish", "gpu": "3", "tag": "v3_n20w01_polish",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *adam_args(lr="2e-4", lr_jas="2e-4", clip_el="5.0", qtrim="0.02",
                       ess_floor="0.001", ess_max="12",
                       rollback_decay="0.88", rollback_err="20.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n20w01_jas"],
        ],
    },
    # GPU 4: N=12 ω=0.05 gentle cascade from ω=0.1 (+0.12%)
    # Intermediate step before ω=0.01
    {
        "name": "n12w005_cascade", "gpu": "4", "tag": "v3_n12w005_cascade",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.05",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            *adam_args(lr="3e-4", lr_jas="3e-5", clip_el="5.0", qtrim="0.02",
                       ess_floor="0.05", ess_max="16",
                       rollback_decay="0.88", rollback_err="50.0", rollback_jump="8.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", CKPT["n12w01_bf"], "--init-bf", CKPT["n12w01_bf"],
            "--no-pretrained",
        ],
    },
    # GPU 5: N=12 ω=0.01 fresh Adam+ESS (no cascade)
    # Same recipe that worked for N=6 ω=0.01
    {
        "name": "n12w001_fresh", "gpu": "5", "tag": "v3_n12w001_fresh",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "12", "--omega", "0.01",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            "--sigma-fs", "0.8,1.3,2.0,3.5",
            *adam_args(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.02",
                       ess_floor="0.001", ess_max="16",
                       rollback_decay="0.85", rollback_err="15.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "11",
        ],
    },
    # GPU 6: N=6 ω=0.001 Langevin test (Adam baseline is +0.224%)
    # This validates whether Langevin actually helps vs Adam-only
    {
        "name": "n6w0001_langevin", "gpu": "6", "tag": "v3_n6w0001_lang",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            "--langevin-steps", "20", "--langevin-step-size", "0.05",
            *adam_args(lr="1e-4", lr_jas="1e-4", clip_el="5.0", qtrim="0.02",
                       rollback_decay="0.90", rollback_err="5.0", rollback_jump="4.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", CKPT["n6w0001"],
            "--no-pretrained",
        ],
    },
    # GPU 7: N=20 ω=0.1 Langevin+Jastrow
    # Langevin should help most here: 40D, mixture overlap is poor
    {
        "name": "n20w01_langevin", "gpu": "7", "tag": "v3_n20w01_lang",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            "--langevin-steps", "15", "--langevin-step-size", "0.02",
            *adam_args(lr="2e-4", lr_jas="2e-4", clip_el="5.0", qtrim="0.02",
                       rollback_decay="0.88", rollback_err="20.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n20w01_jas"],
        ],
    },
]


def launch_job(job):
    tag = job["tag"]
    gpu = job["gpu"]
    log = LOGDIR / f"{tag}.log"
    cmd_parts = [
        f"cd {ROOT}",
        MODULE_CMD,
        f"CUDA_VISIBLE_DEVICES={gpu} python3 src/run_weak_form.py "
        + " ".join(job["cmd"])
        + f" --tag {tag}"
    ]
    shell_cmd = " && ".join(cmd_parts)
    wrapper = f'echo "# Started: $(date) GPU={gpu} tag={tag}" >> {log}; '
    wrapper += f'({shell_cmd}) >> {log} 2>&1; '
    wrapper += f'echo "# Completed: $(date) rc=$?" >> {log}'
    return wrapper


def main():
    print(f"Campaign v3 — {len(JOBS)} jobs")
    print(f"Logs: {LOGDIR}")
    print()

    # Verify checkpoints exist
    for key, path in CKPT.items():
        exists = "OK" if os.path.isfile(path) else "MISSING"
        print(f"  {key}: {exists} — {path}")
    print()

    # Launch all jobs in tmux
    session = "v3camp"
    subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-x", "200", "-y", "50"])

    for i, job in enumerate(JOBS):
        wrapper = launch_job(job)
        if i == 0:
            subprocess.run(["tmux", "send-keys", "-t", f"{session}", wrapper, "Enter"])
        else:
            subprocess.run(["tmux", "new-window", "-t", session])
            subprocess.run(["tmux", "send-keys", "-t", f"{session}", wrapper, "Enter"])
        print(f"  GPU {job['gpu']}: {job['name']} → {job['tag']}")
        time.sleep(1)

    print(f"\nAll {len(JOBS)} jobs launched in tmux session '{session}'")
    print(f"Monitor: tail -f {LOGDIR}/*.log")
    print(f"Attach:  tmux attach -t {session}")


if __name__ == "__main__":
    main()
