#!/usr/bin/env python3
"""Few-hours campaign: N=20 BF + N=2 omega sweep.

Intent:
- Run N=20 in BF mode for omega in {1.0, 0.5, 0.1}.
- Reduce low-omega rollback lock by disabling err-threshold rollback for omega<=0.1
  and keeping jump-based rollback.
- Run N=2 BF sweep for omega in {1.0, 0.5, 0.1, 0.01, 0.001} in one GPU chain.
- Keep wall time within a few hours by capping each slot with timeout.
"""

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / (time.strftime("%Y-%m-%d_%H%M") + "_campaign_v6_n20bf_n2_fewhours")
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"
MAX_WALL = os.environ.get("CAMP_V6_WALL", "12600s")

CKPT = {
    "n20w1_jas": str(RESULTS / "v3_n20w1_ultra.pt"),
    "n20w05_jas": str(RESULTS / "v3_n20w05_polish.pt"),
    "n20w01_jas": str(RESULTS / "v3_n20w01_polish2.pt"),
    "n2w1_bf": str(RESULTS / "v4_n2w1_bf.pt"),
    "n2w05_bf": str(RESULTS / "v4_n2w05_bf.pt"),
    "n2w01_bf": str(RESULTS / "v4_n2w01_bf.pt"),
    "n2w001_bf": str(RESULTS / "v4_n2w001_bf.pt"),
    "n2w0001_bf": str(RESULTS / "v4_n2w0001_bf.pt"),
}


def adam_args(
    lr: str,
    lr_jas: str,
    clip_el: str,
    qtrim: str,
    ess_floor: str,
    ess_max: str,
    rollback_decay: str,
    rollback_err: str,
    rollback_jump: str,
):
    args = [
        "--lr", lr,
        "--lr-jas", lr_jas,
        "--direct-weight", "0.0",
        "--clip-el", clip_el,
        "--reward-qtrim", qtrim,
    ]
    if float(ess_floor) > 0.0:
        args += [
            "--ess-floor-ratio", ess_floor,
            "--ess-oversample-max", ess_max,
            "--ess-oversample-step", "2",
            "--ess-resample-tries", "2",
        ]
    if float(rollback_decay) < 1.0:
        args += [
            "--rollback-decay", rollback_decay,
            "--rollback-err-pct", rollback_err,
            "--rollback-jump-sigma", rollback_jump,
        ]
    return args


SLOTS = [
    {
        "gpu": "0",
        "name": "n20_bf_w1",
        "jobs": [
            {
                "tag": "v6_n20w1_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
                    "--e-dmc", "155.88220",
                    "--bf-hidden", "48", "--bf-msg-hidden", "48", "--bf-layers", "2",
                    "--epochs", "2400",
                    "--n-coll", "4096", "--oversample", "8", "--micro-batch", "256",
                    *adam_args(lr="1e-4", lr_jas="1e-5", clip_el="3.0", qtrim="0.01",
                               ess_floor="0.01", ess_max="20",
                               rollback_decay="0.94", rollback_err="8.0", rollback_jump="4.0"),
                    "--vmc-every", "40", "--vmc-n", "15000",
                    "--n-eval", "50000", "--seed", "42",
                    "--init-jas", CKPT["n20w1_jas"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "1",
        "name": "n20_bf_w05",
        "jobs": [
            {
                "tag": "v6_n20w05_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "20", "--omega", "0.5",
                    "--e-dmc", "93.87520",
                    "--bf-hidden", "48", "--bf-msg-hidden", "48", "--bf-layers", "2",
                    "--epochs", "2600",
                    "--n-coll", "4096", "--oversample", "10", "--micro-batch", "256",
                    *adam_args(lr="8e-5", lr_jas="8e-6", clip_el="3.5", qtrim="0.01",
                               ess_floor="0.02", ess_max="24",
                               rollback_decay="0.94", rollback_err="6.0", rollback_jump="4.0"),
                    "--vmc-every", "40", "--vmc-n", "15000",
                    "--n-eval", "50000", "--seed", "43",
                    "--init-jas", CKPT["n20w05_jas"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "2",
        "name": "n20_bf_w01_lowomega_fix",
        "jobs": [
            {
                "tag": "v6_n20w01_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "20", "--omega", "0.1",
                    "--e-dmc", "29.97790",
                    "--bf-hidden", "48", "--bf-msg-hidden", "48", "--bf-layers", "2",
                    "--epochs", "3000",
                    "--n-coll", "4096", "--oversample", "12", "--micro-batch", "256",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    "--langevin-steps", "8", "--langevin-step-size", "0.003",
                    *adam_args(lr="6e-5", lr_jas="8e-6", clip_el="4.0", qtrim="0.02",
                               ess_floor="0.03", ess_max="28",
                               rollback_decay="0.94", rollback_err="0.0", rollback_jump="4.0"),
                    "--vmc-every", "40", "--vmc-n", "15000",
                    "--n-eval", "50000", "--seed", "44",
                    "--init-jas", CKPT["n20w01_jas"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "3",
        "name": "n2_all_omegas_chain",
        "jobs": [
            {
                "tag": "v6_n2w1_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "1.0", "--e-dmc", "3.00000",
                    "--bf-hidden", "64", "--bf-msg-hidden", "64", "--bf-layers", "3",
                    "--epochs", "3000",
                    "--n-coll", "4096", "--oversample", "12", "--micro-batch", "1024",
                    *adam_args(lr="2e-4", lr_jas="2e-4", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.02", ess_max="24",
                               rollback_decay="0.95", rollback_err="3.0", rollback_jump="4.0"),
                    "--vmc-every", "80", "--vmc-n", "20000", "--n-eval", "60000", "--seed", "41",
                    "--resume", CKPT["n2w1_bf"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v6_n2w05_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.5", "--e-dmc", "1.65977",
                    "--bf-hidden", "64", "--bf-msg-hidden", "64", "--bf-layers", "3",
                    "--epochs", "3000",
                    "--n-coll", "4096", "--oversample", "12", "--micro-batch", "1024",
                    *adam_args(lr="2e-4", lr_jas="2e-4", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.02", ess_max="24",
                               rollback_decay="0.95", rollback_err="3.0", rollback_jump="4.0"),
                    "--vmc-every", "80", "--vmc-n", "20000", "--n-eval", "60000", "--seed", "42",
                    "--resume", CKPT["n2w05_bf"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v6_n2w01_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.1", "--e-dmc", "0.44079",
                    "--bf-hidden", "64", "--bf-msg-hidden", "64", "--bf-layers", "3",
                    "--epochs", "3200",
                    "--n-coll", "4096", "--oversample", "14", "--micro-batch", "1024",
                    *adam_args(lr="1.5e-4", lr_jas="1.5e-4", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.03", ess_max="28",
                               rollback_decay="0.95", rollback_err="0.0", rollback_jump="4.5"),
                    "--vmc-every", "80", "--vmc-n", "20000", "--n-eval", "60000", "--seed", "43",
                    "--resume", CKPT["n2w01_bf"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v6_n2w001_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.01", "--e-dmc", "0.07384",
                    "--bf-hidden", "64", "--bf-msg-hidden", "64", "--bf-layers", "3",
                    "--epochs", "3500",
                    "--n-coll", "4096", "--oversample", "16", "--micro-batch", "1024",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    *adam_args(lr="1.2e-4", lr_jas="1.2e-4", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.04", ess_max="32",
                               rollback_decay="0.95", rollback_err="0.0", rollback_jump="5.0"),
                    "--vmc-every", "80", "--vmc-n", "20000", "--n-eval", "60000", "--seed", "44",
                    "--resume", CKPT["n2w001_bf"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v6_n2w0001_bf",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.001", "--e-dmc", "0.00730",
                    "--bf-hidden", "64", "--bf-msg-hidden", "64", "--bf-layers", "3",
                    "--epochs", "4000",
                    "--n-coll", "4096", "--oversample", "20", "--micro-batch", "1024",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0,8.0",
                    "--langevin-steps", "10", "--langevin-step-size", "0.002",
                    *adam_args(lr="1e-4", lr_jas="1e-4", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.05", ess_max="36",
                               rollback_decay="0.95", rollback_err="0.0", rollback_jump="5.5"),
                    "--vmc-every", "80", "--vmc-n", "25000", "--n-eval", "80000", "--seed", "45",
                    "--resume", CKPT["n2w0001_bf"],
                    "--no-pretrained",
                ],
            },
        ],
    },
]


def slot_command(slot):
    commands = []
    for job in slot["jobs"]:
        tag = job["tag"]
        log = LOGDIR / f"{tag}.log"
        run = (
            f"cd {ROOT}; {MODULE_CMD}; "
            f"CUDA_VISIBLE_DEVICES={slot['gpu']} timeout {MAX_WALL} "
            f"python3 src/run_weak_form.py {' '.join(job['cmd'])} --tag {tag}"
        )
        commands.append(
            f"echo \"# Started: $(date) GPU={slot['gpu']} tag={tag}\" >> {log}; "
            f"({run}) >> {log} 2>&1; "
            f"echo \"# Completed: $(date) rc=$?\" >> {log}"
        )
    return "; ".join(commands)


def main():
    missing = [k for k, p in CKPT.items() if not Path(p).is_file()]
    if missing:
        print("Missing checkpoints:")
        for k in missing:
            print(f"  - {k}: {CKPT[k]}")
        raise SystemExit(1)

    print(f"Campaign v6 (few-hours) with wall cap {MAX_WALL}")
    print(f"Logs: {LOGDIR}")

    session = "v6bfquick"
    subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "55"])

    for i, slot in enumerate(SLOTS):
        if i > 0:
            subprocess.run(["tmux", "new-window", "-t", session])
        cmd = slot_command(slot)
        subprocess.run(["tmux", "send-keys", "-t", session, cmd, "Enter"])
        print(f"  GPU {slot['gpu']}: {slot['name']} -> {[j['tag'] for j in slot['jobs']]}")
        time.sleep(1)

    print(f"Launched tmux session: {session}")
    print(f"Attach: tmux attach -t {session}")
    print(f"Monitor: tail -f {LOGDIR}/*.log")


if __name__ == "__main__":
    main()
