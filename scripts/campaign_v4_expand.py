#!/usr/bin/env python3
"""Campaign v4: Expand coverage — N=20 BF, N=12 low-ω, N=2 full sweep, polish.

Goals:
1. Train backflow for N=20 (reduced hidden dim, init from best Jastrow)
2. N=12: gentle cascade to lower ω (0.08 → 0.05)
3. N=2: BF training across all ω (fill coverage gaps)
4. Polish existing N=20 Jastrow results further
"""
import subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / (time.strftime("%Y-%m-%d_%H%M") + "_campaign_v4")
LOGDIR = Path(OUTDIR) / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

# ═══════════════════════════════════════════════════════════
# Best checkpoints
# ═══════════════════════════════════════════════════════════
CKPT = {
    # N=20 Jastrow bests
    "n20w1_jas":  str(RESULTS / "v3_n20w1_ultra.pt"),       # +1.428%
    "n20w05_jas": str(RESULTS / "v3_n20w05_polish.pt"),     # +2.740%
    "n20w01_jas": str(RESULTS / "v3_n20w01_polish2.pt"),    # +5.678%
    # N=12 BF bests
    "n12w1_bf":   str(RESULTS / "long_n12w1.pt"),           # +0.018%
    "n12w01_bf":  str(RESULTS / "20260318_1149_n12w01_keep.pt"),  # +0.122%
    # N=6 BF best (for N=2 init reference)
    "n6w1_bf":    str(RESULTS / "bf_hardfocus_v1b.pt"),     # +0.009%
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
    # ──────────────────────────────────────────────────────
    # GPU 0: N=20 ω=1.0 BF (small bf_hidden=32)
    # Init Jastrow from +1.43% checkpoint, train BF from scratch.
    # Tiny lr_jas to keep Jastrow mostly frozen.
    # ──────────────────────────────────────────────────────
    {
        "name": "n20w1_bf32", "gpu": "0", "tag": "v4_n20w1_bf32",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
            "--bf-hidden", "32", "--bf-msg-hidden", "32", "--bf-layers", "2",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "8", "--micro-batch", "512",
            *adam_args(lr="2e-4", lr_jas="1e-5", clip_el="3.0", qtrim="0.01",
                       rollback_decay="0.92", rollback_err="10.0", rollback_jump="5.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", CKPT["n20w1_jas"],
            "--no-pretrained",
        ],
    },
    # ──────────────────────────────────────────────────────
    # GPU 1: N=20 ω=1.0 BF (bf_hidden=48)
    # Slightly larger BF to see if hidden=48 can still train.
    # ──────────────────────────────────────────────────────
    {
        "name": "n20w1_bf48", "gpu": "1", "tag": "v4_n20w1_bf48",
        "cmd": [
            "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
            "--bf-hidden", "48", "--bf-msg-hidden", "48", "--bf-layers", "2",
            "--epochs", "3000",
            "--n-coll", "2048", "--oversample", "10", "--micro-batch", "256",
            *adam_args(lr="2e-4", lr_jas="1e-5", clip_el="3.0", qtrim="0.01",
                       rollback_decay="0.92", rollback_err="10.0", rollback_jump="5.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", CKPT["n20w1_jas"],
            "--no-pretrained",
        ],
    },
    # ──────────────────────────────────────────────────────
    # GPU 2: N=12 ω=0.08 BF cascade from ω=0.1 (+0.12%)
    # Gentle step: 0.1→0.08 (only 20% reduction)
    # NOTE: No DMC ref for ω=0.08 → disable rollback-err, use jump-sigma only.
    # ──────────────────────────────────────────────────────
    {
        "name": "n12w008_cascade", "gpu": "2", "tag": "v4_n12w008_cascade",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.08",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *adam_args(lr="2e-4", lr_jas="2e-5", clip_el="5.0", qtrim="0.02",
                       ess_floor="0.01", ess_max="16",
                       rollback_decay="0.90", rollback_err="0.0", rollback_jump="5.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--init-jas", CKPT["n12w01_bf"], "--init-bf", CKPT["n12w01_bf"],
            "--no-pretrained",
        ],
    },
    # ──────────────────────────────────────────────────────
    # GPU 3: N=12 ω=0.05 Jastrow-only (no BF — avoids BF gradient noise)
    # Use wider proposal + ESS adaptive + aggressive rollback.
    # Jastrow-only worked for N=6 at low ω; try same strategy for N=12.
    # NOTE: No DMC ref for ω=0.05 → disable rollback-err, use jump-sigma only.
    # ──────────────────────────────────────────────────────
    {
        "name": "n12w005_jas", "gpu": "3", "tag": "v4_n12w005_jas",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "12", "--omega", "0.05",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
            "--sigma-fs", "0.8,1.3,2.0,3.5",
            *adam_args(lr="3e-4", lr_jas="3e-4", clip_el="5.0", qtrim="0.02",
                       ess_floor="0.005", ess_max="20",
                       rollback_decay="0.88", rollback_err="0.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "11",
            "--init-jas", CKPT["n12w01_bf"],
            "--no-pretrained",
        ],
    },
    # ──────────────────────────────────────────────────────
    # GPU 4: N=20 ω=0.5 ultra-polish from +2.74% checkpoint
    # Very low LR continuation.
    # ──────────────────────────────────────────────────────
    {
        "name": "n20w05_ultra", "gpu": "4", "tag": "v4_n20w05_ultra",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.5",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *adam_args(lr="5e-5", lr_jas="5e-5", clip_el="3.0", qtrim="0.01",
                       rollback_decay="0.92", rollback_err="10.0", rollback_jump="4.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n20w05_jas"],
        ],
    },
    # ──────────────────────────────────────────────────────
    # GPU 6: N=2 BF sweep: ω = 1.0, 0.5, 0.1, 0.01, 0.001
    # N=2 is tiny — run as a sequential chain on one GPU.
    # This is a shell script that runs 5 jobs one after another.
    # ──────────────────────────────────────────────────────
    {
        "name": "n2_bf_sweep", "gpu": "6", "tag": "v4_n2_sweep",
        "multi": True,  # Special flag: multi-job on one GPU
        "sub_jobs": [
            {
                "tag": "v4_n2w1_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "1.0",
                    "--bf-hidden", "64", "--bf-layers", "2",
                    "--epochs", "2000",
                    "--n-coll", "2048", "--oversample", "8", "--micro-batch", "512",
                    *adam_args(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.01"),
                    "--vmc-every", "50", "--vmc-n", "10000",
                    "--n-eval", "20000", "--seed", "42",
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v4_n2w05_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.5",
                    "--bf-hidden", "64", "--bf-layers", "2",
                    "--epochs", "2000",
                    "--n-coll", "2048", "--oversample", "8", "--micro-batch", "512",
                    *adam_args(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.01"),
                    "--vmc-every", "50", "--vmc-n", "10000",
                    "--n-eval", "20000", "--seed", "42",
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v4_n2w01_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.1",
                    "--bf-hidden", "64", "--bf-layers", "2",
                    "--epochs", "2000",
                    "--n-coll", "2048", "--oversample", "8", "--micro-batch", "512",
                    *adam_args(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.01"),
                    "--vmc-every", "50", "--vmc-n", "10000",
                    "--n-eval", "20000", "--seed", "42",
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v4_n2w001_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.01",
                    "--bf-hidden", "64", "--bf-layers", "2",
                    "--epochs", "2000",
                    "--n-coll", "2048", "--oversample", "8", "--micro-batch", "512",
                    *adam_args(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.01"),
                    "--vmc-every", "50", "--vmc-n", "10000",
                    "--n-eval", "20000", "--seed", "42",
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v4_n2w0001_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.001",
                    "--bf-hidden", "64", "--bf-layers", "2",
                    "--epochs", "2000",
                    "--n-coll", "2048", "--oversample", "8", "--micro-batch", "512",
                    "--sigma-fs", "0.8,1.3,2.0,3.5",
                    *adam_args(lr="5e-4", lr_jas="5e-4", clip_el="5.0", qtrim="0.02",
                               ess_floor="0.01", ess_max="12"),
                    "--vmc-every", "50", "--vmc-n", "10000",
                    "--n-eval", "20000", "--seed", "42",
                    "--no-pretrained",
                ],
            },
        ],
    },
    # ──────────────────────────────────────────────────────
    # GPU 7: N=20 ω=0.1 ultra-polish from +5.68% checkpoint
    # Aggressive polish: lower LR, longer run
    # ──────────────────────────────────────────────────────
    {
        "name": "n20w01_ultra", "gpu": "7", "tag": "v4_n20w01_ultra",
        "cmd": [
            "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
            "--epochs", "3000",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *adam_args(lr="5e-5", lr_jas="5e-5", clip_el="5.0", qtrim="0.02",
                       ess_floor="0.001", ess_max="16",
                       rollback_decay="0.90", rollback_err="20.0", rollback_jump="6.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "30000", "--seed", "42",
            "--resume", CKPT["n20w01_jas"],
        ],
    },
]


def launch_single_job(job_cmd, tag, gpu, root, module_cmd, logdir):
    """Build shell command for a single training job."""
    log = logdir / f"{tag}.log"
    cmd_parts = [
        f"cd {root}",
        module_cmd,
        f"CUDA_VISIBLE_DEVICES={gpu} python3 src/run_weak_form.py "
        + " ".join(job_cmd)
        + f" --tag {tag}"
    ]
    shell_cmd = " && ".join(cmd_parts)
    wrapper = f'echo "# Started: $(date) GPU={gpu} tag={tag}" >> {log}; '
    wrapper += f'({shell_cmd}) >> {log} 2>&1; '
    wrapper += f'echo "# Completed: $(date) rc=$?" >> {log}'
    return wrapper


def launch_multi_job(sub_jobs, gpu, root, module_cmd, logdir):
    """Build shell command for sequential sub-jobs on one GPU."""
    parts = []
    for sj in sub_jobs:
        tag = sj["tag"]
        log = logdir / f"{tag}.log"
        cmd_parts = [
            f"cd {root}",
            module_cmd,
            f"CUDA_VISIBLE_DEVICES={gpu} python3 src/run_weak_form.py "
            + " ".join(sj["cmd"])
            + f" --tag {tag}"
        ]
        shell_cmd = " && ".join(cmd_parts)
        part = f'echo "# Started: $(date) GPU={gpu} tag={tag}" >> {log}; '
        part += f'({shell_cmd}) >> {log} 2>&1; '
        part += f'echo "# Completed: $(date) rc=$?" >> {log}'
        parts.append(part)
    return "; ".join(parts)


def main():
    print(f"Campaign v4 — {len(JOBS)} GPU slots")
    print(f"Logs: {LOGDIR}")
    print()

    # Verify checkpoints exist
    for key, path in sorted(CKPT.items()):
        exists = "OK" if os.path.isfile(path) else "MISSING"
        print(f"  {key}: {exists} — {path}")
    print()

    # Launch all jobs in tmux
    session = "v4camp"
    subprocess.run(["tmux", "kill-session", "-t", session],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-x", "200", "-y", "50"])

    for i, job in enumerate(JOBS):
        gpu = job["gpu"]

        if job.get("multi"):
            wrapper = launch_multi_job(job["sub_jobs"], gpu, ROOT, MODULE_CMD, LOGDIR)
            tags = ", ".join(sj["tag"] for sj in job["sub_jobs"])
        else:
            wrapper = launch_single_job(job["cmd"], job["tag"], gpu, ROOT, MODULE_CMD, LOGDIR)
            tags = job["tag"]

        if i == 0:
            subprocess.run(["tmux", "send-keys", "-t", f"{session}", wrapper, "Enter"])
        else:
            subprocess.run(["tmux", "new-window", "-t", session])
            subprocess.run(["tmux", "send-keys", "-t", f"{session}", wrapper, "Enter"])

        print(f"  GPU {gpu}: {job['name']} → {tags}")
        time.sleep(1)

    print(f"\nAll {len(JOBS)} GPU slots launched in tmux session '{session}'")
    print(f"Monitor: tail -f {LOGDIR}/*.log")
    print(f"Attach:  tmux attach -t {session}")


if __name__ == "__main__":
    main()
