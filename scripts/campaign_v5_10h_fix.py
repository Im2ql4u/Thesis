#!/usr/bin/env python3
"""10-hour capped stability campaign (v5).

Focus:
- Enforce Adam-only behavior in low-omega hard regimes (no SR usage).
- Push N=2 omega=0.001 toward exact DMC with multiple robust variants.
- Push N=20 toward sub-0.1% across available omega references (1.0, 0.5, 0.1).
- Include low-omega known-reference branch at N=6 omega=0.001 and N=12 omega=0.01.

Execution model:
- One tmux window per GPU slot.
- Each slot is capped by timeout (default 10h).
- Some slots run sequential sub-jobs to test multiple ideas within the same budget.
"""

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / (time.strftime("%Y-%m-%d_%H%M") + "_campaign_v5_10h_fix")
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"
MAX_WALL = os.environ.get("CAMP_V5_WALL", "10h")

CKPT = {
    "n20w1_jas": str(RESULTS / "v3_n20w1_ultra.pt"),
    "n20w05_jas": str(RESULTS / "v3_n20w05_polish.pt"),
    "n20w01_jas": str(RESULTS / "v3_n20w01_polish2.pt"),
    "n12w01_bf": str(RESULTS / "20260318_1149_n12w01_keep.pt"),
    "n6w1_bf": str(RESULTS / "bf_hardfocus_v1b.pt"),
    "n2w001_bf": str(RESULTS / "v4_n2w001_bf.pt"),
}


def adam_args(
    lr="5e-4",
    lr_jas="5e-5",
    clip_el="5.0",
    qtrim="0.02",
    ess_floor="0.0",
    ess_max="0",
    rollback_decay="1.0",
    rollback_err="0.0",
    rollback_jump="0.0",
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


GPU_CHAINS = [
    {
        "gpu": "0",
        "name": "n2_w0001_precision_chain_a",
        "jobs": [
            {
                "tag": "v5_n2w0001_precA",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "2", "--omega", "0.001",
                    "--e-dmc", "0.00730",
                    "--epochs", "18000",
                    "--n-coll", "8192", "--oversample", "24", "--micro-batch", "1024",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    "--langevin-steps", "12", "--langevin-step-size", "0.002",
                    *adam_args(lr="1e-4", lr_jas="1e-4", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.05", ess_max="64",
                               rollback_decay="0.93", rollback_err="0.0", rollback_jump="6.0"),
                    "--vmc-every", "120", "--vmc-n", "30000",
                    "--n-eval", "120000", "--seed", "42",
                    "--resume", CKPT["n2w001_bf"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v5_n2w0001_precA_seed7",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "2", "--omega", "0.001",
                    "--e-dmc", "0.00730",
                    "--epochs", "12000",
                    "--n-coll", "8192", "--oversample", "24", "--micro-batch", "1024",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    *adam_args(lr="8e-5", lr_jas="8e-5", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.05", ess_max="64",
                               rollback_decay="0.95", rollback_err="0.0", rollback_jump="6.0"),
                    "--vmc-every", "120", "--vmc-n", "30000",
                    "--n-eval", "120000", "--seed", "7",
                    "--resume", CKPT["n2w001_bf"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "1",
        "name": "n2_w0001_precision_chain_b",
        "jobs": [
            {
                "tag": "v5_n2w0001_precB",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "2", "--omega", "0.001",
                    "--e-dmc", "0.00730",
                    "--epochs", "24000",
                    "--n-coll", "8192", "--oversample", "20", "--micro-batch", "1024",
                    "--sigma-fs", "0.6,1.0,1.6,2.6,4.5",
                    *adam_args(lr="3e-5", lr_jas="3e-5", clip_el="4.0", qtrim="0.01",
                               ess_floor="0.02", ess_max="48",
                               rollback_decay="0.95", rollback_err="0.0", rollback_jump="5.0"),
                    "--vmc-every", "120", "--vmc-n", "30000",
                    "--n-eval", "120000", "--seed", "42",
                    "--resume", CKPT["n2w001_bf"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "2",
        "name": "n6_lowomega_chain",
        "jobs": [
            {
                "tag": "v5_n6w0001_adamess",
                "cmd": [
                    "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
                    "--e-dmc", "0.140832",
                    "--epochs", "10000",
                    "--n-coll", "12288", "--oversample", "16", "--micro-batch", "512",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    *adam_args(lr="2e-4", lr_jas="2e-5", clip_el="5.0", qtrim="0.02",
                               ess_floor="0.12", ess_max="24",
                               rollback_decay="0.90", rollback_err="0.0", rollback_jump="5.0"),
                    "--vmc-every", "80", "--vmc-n", "25000",
                    "--n-eval", "80000", "--seed", "42",
                    "--resume", CKPT["n6w1_bf"],
                ],
            },
            {
                "tag": "v5_n6w001_adamess",
                "cmd": [
                    "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
                    "--e-dmc", "0.69036",
                    "--epochs", "8000",
                    "--n-coll", "8192", "--oversample", "14", "--micro-batch", "512",
                    *adam_args(lr="2e-4", lr_jas="2e-5", clip_el="5.0", qtrim="0.02",
                               ess_floor="0.10", ess_max="20",
                               rollback_decay="0.90", rollback_err="8.0", rollback_jump="4.0"),
                    "--vmc-every", "80", "--vmc-n", "25000",
                    "--n-eval", "80000", "--seed", "43",
                    "--resume", CKPT["n6w1_bf"],
                ],
            },
        ],
    },
    {
        "gpu": "3",
        "name": "n20_w1_precision_bf",
        "jobs": [
            {
                "tag": "v5_n20w1_bf48",
                "cmd": [
                    "--mode", "bf", "--n-elec", "20", "--omega", "1.0",
                    "--e-dmc", "155.88220",
                    "--bf-hidden", "48", "--bf-msg-hidden", "48", "--bf-layers", "2",
                    "--epochs", "9000",
                    "--n-coll", "4096", "--oversample", "8", "--micro-batch", "256",
                    *adam_args(lr="8e-5", lr_jas="1e-5", clip_el="3.0", qtrim="0.01",
                               ess_floor="0.01", ess_max="20",
                               rollback_decay="0.93", rollback_err="8.0", rollback_jump="4.0"),
                    "--vmc-every", "60", "--vmc-n", "20000",
                    "--n-eval", "60000", "--seed", "42",
                    "--init-jas", CKPT["n20w1_jas"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "4",
        "name": "n20_w05_precision_jas",
        "jobs": [
            {
                "tag": "v5_n20w05_jas_ultra",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "0.5",
                    "--e-dmc", "93.87520",
                    "--epochs", "12000",
                    "--n-coll", "6144", "--oversample", "14", "--micro-batch", "512",
                    *adam_args(lr="3e-5", lr_jas="3e-5", clip_el="3.0", qtrim="0.01",
                               ess_floor="0.01", ess_max="20",
                               rollback_decay="0.95", rollback_err="6.0", rollback_jump="3.5"),
                    "--vmc-every", "60", "--vmc-n", "20000",
                    "--n-eval", "60000", "--seed", "42",
                    "--resume", CKPT["n20w05_jas"],
                ],
            },
        ],
    },
    {
        "gpu": "5",
        "name": "n20_w01_precision_jas",
        "jobs": [
            {
                "tag": "v5_n20w01_jas_ultra",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
                    "--e-dmc", "29.97790",
                    "--epochs", "12000",
                    "--n-coll", "6144", "--oversample", "16", "--micro-batch", "512",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    "--langevin-steps", "8", "--langevin-step-size", "0.003",
                    *adam_args(lr="2e-5", lr_jas="2e-5", clip_el="4.0", qtrim="0.02",
                               ess_floor="0.02", ess_max="24",
                               rollback_decay="0.95", rollback_err="5.0", rollback_jump="3.0"),
                    "--vmc-every", "60", "--vmc-n", "20000",
                    "--n-eval", "60000", "--seed", "42",
                    "--resume", CKPT["n20w01_jas"],
                ],
            },
        ],
    },
    {
        "gpu": "6",
        "name": "n12_w001_lowomega_bf",
        "jobs": [
            {
                "tag": "v5_n12w001_bf_cascade",
                "cmd": [
                    "--mode", "bf", "--n-elec", "12", "--omega", "0.01",
                    "--e-dmc", "2.0",
                    "--epochs", "10000",
                    "--n-coll", "6144", "--oversample", "16", "--micro-batch", "512",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    *adam_args(lr="1.5e-4", lr_jas="1.5e-5", clip_el="5.0", qtrim="0.02",
                               ess_floor="0.10", ess_max="24",
                               rollback_decay="0.90", rollback_err="0.0", rollback_jump="5.0"),
                    "--vmc-every", "80", "--vmc-n", "20000",
                    "--n-eval", "60000", "--seed", "42",
                    "--init-jas", CKPT["n12w01_bf"],
                    "--init-bf", CKPT["n12w01_bf"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "7",
        "name": "n20_replica_seed11",
        "jobs": [
            {
                "tag": "v5_n20w1_jas_seed11",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "1.0",
                    "--e-dmc", "155.88220",
                    "--epochs", "12000",
                    "--n-coll", "6144", "--oversample", "14", "--micro-batch", "512",
                    *adam_args(lr="2e-5", lr_jas="2e-5", clip_el="3.0", qtrim="0.01",
                               ess_floor="0.01", ess_max="20",
                               rollback_decay="0.95", rollback_err="5.0", rollback_jump="3.0"),
                    "--vmc-every", "60", "--vmc-n", "20000",
                    "--n-eval", "60000", "--seed", "11",
                    "--resume", CKPT["n20w1_jas"],
                ],
            },
        ],
    },
]


def _build_slot_command(slot):
    parts = []
    for job in slot["jobs"]:
        tag = job["tag"]
        log = LOGDIR / f"{tag}.log"
        cmd = " ".join(job["cmd"] + ["--tag", tag])
        run_cmd = (
            f"cd {ROOT}; "
            f"{MODULE_CMD}; "
            f"CUDA_VISIBLE_DEVICES={slot['gpu']} timeout {MAX_WALL} "
            f"python3 src/run_weak_form.py {cmd}"
        )
        wrapper = (
            f"echo \"# Started: $(date) GPU={slot['gpu']} tag={tag}\" >> {log}; "
            f"({run_cmd}) >> {log} 2>&1; "
            f"echo \"# Completed: $(date) rc=$?\" >> {log}"
        )
        parts.append(wrapper)
    return "; ".join(parts)


def main():
    print(f"Campaign v5 (10h cap) — {len(GPU_CHAINS)} GPU slots")
    print(f"Logs: {LOGDIR}")
    print(f"Max wall time per slot: {MAX_WALL}")

    missing = [k for k, v in CKPT.items() if not Path(v).is_file()]
    if missing:
        print("Missing checkpoints:")
        for k in missing:
            print(f"  - {k}: {CKPT[k]}")
        raise SystemExit(1)

    session = "v5fix10h"
    subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "55"])

    for idx, slot in enumerate(GPU_CHAINS):
        if idx > 0:
            subprocess.run(["tmux", "new-window", "-t", session])
        cmd = _build_slot_command(slot)
        subprocess.run(["tmux", "send-keys", "-t", session, cmd, "Enter"])
        print(f"  GPU {slot['gpu']}: {slot['name']} -> {[j['tag'] for j in slot['jobs']]}")
        time.sleep(1)

    print(f"\nLaunched in tmux session '{session}'.")
    print(f"Attach: tmux attach -t {session}")
    print(f"Monitor: tail -f {LOGDIR}/*.log")


if __name__ == "__main__":
    main()
