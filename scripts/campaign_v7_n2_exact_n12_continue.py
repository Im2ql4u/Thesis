#!/usr/bin/env python3
"""v7: N=2 exactness + N=12 continuation campaign.

Goals:
- Push N=2 energies toward exact DMC across omegas (1.0, 0.5, 0.1, 0.01, 0.001).
- Keep training N=12 strong checkpoints to improve current bests.
- Use low-omega safeguards that avoid err-threshold rollback lockups.
"""

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs" / (time.strftime("%Y-%m-%d_%H%M") + "_campaign_v7_n2_exact_n12_continue")
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

RESULTS = ROOT / "results" / "arch_colloc"
MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"
MAX_WALL = os.environ.get("CAMP_V7_WALL", "14400s")

CKPT = {
    "n2w1": str(RESULTS / "v4_n2w1_bf.pt"),
    "n2w05": str(RESULTS / "v4_n2w05_bf.pt"),
    "n2w01": str(RESULTS / "v4_n2w01_bf.pt"),
    "n2w001": str(RESULTS / "v4_n2w001_bf.pt"),
    "n2w0001": str(RESULTS / "v4_n2w0001_bf.pt"),
    "n12w1": str(RESULTS / "long_n12w1.pt"),
    "n12w01": str(RESULTS / "20260318_1149_n12w01_keep.pt"),
}


def adam_args(lr, lr_jas, clip_el, qtrim, ess_floor, ess_max, rdec, rerr, rjump):
    args = [
        "--lr", str(lr),
        "--lr-jas", str(lr_jas),
        "--direct-weight", "0.0",
        "--clip-el", str(clip_el),
        "--reward-qtrim", str(qtrim),
    ]
    if float(ess_floor) > 0:
        args += [
            "--ess-floor-ratio", str(ess_floor),
            "--ess-oversample-max", str(ess_max),
            "--ess-oversample-step", "2",
            "--ess-resample-tries", "2",
        ]
    if float(rdec) < 1.0:
        args += [
            "--rollback-decay", str(rdec),
            "--rollback-err-pct", str(rerr),
            "--rollback-jump-sigma", str(rjump),
        ]
    return args


SLOTS = [
    {
        "gpu": "0",
        "name": "n2_exact_chain",
        "jobs": [
            {
                "tag": "v7_n2w1_exact",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "1.0", "--e-dmc", "3.00000",
                    "--epochs", "3000", "--n-coll", "4096", "--oversample", "12", "--micro-batch", "1024",
                    *adam_args("1e-4", "1e-4", "4.0", "0.01", "0.02", "24", "0.96", "2.0", "3.5"),
                    "--vmc-every", "60", "--vmc-n", "15000", "--n-eval", "60000", "--seed", "51",
                    "--resume", CKPT["n2w1"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v7_n2w05_exact",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.5", "--e-dmc", "1.65977",
                    "--epochs", "3200", "--n-coll", "4096", "--oversample", "12", "--micro-batch", "1024",
                    *adam_args("1e-4", "1e-4", "4.0", "0.01", "0.02", "24", "0.96", "2.0", "3.5"),
                    "--vmc-every", "60", "--vmc-n", "15000", "--n-eval", "60000", "--seed", "52",
                    "--resume", CKPT["n2w05"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v7_n2w01_exact",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.1", "--e-dmc", "0.44079",
                    "--epochs", "3400", "--n-coll", "4096", "--oversample", "14", "--micro-batch", "1024",
                    *adam_args("8e-5", "8e-5", "4.0", "0.01", "0.03", "28", "0.96", "0.0", "4.0"),
                    "--vmc-every", "60", "--vmc-n", "15000", "--n-eval", "60000", "--seed", "53",
                    "--resume", CKPT["n2w01"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v7_n2w001_exact",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.01", "--e-dmc", "0.07384",
                    "--epochs", "3600", "--n-coll", "4096", "--oversample", "16", "--micro-batch", "1024",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    *adam_args("8e-5", "8e-5", "4.0", "0.01", "0.04", "32", "0.96", "0.0", "4.5"),
                    "--vmc-every", "60", "--vmc-n", "15000", "--n-eval", "60000", "--seed", "54",
                    "--resume", CKPT["n2w001"],
                    "--no-pretrained",
                ],
            },
            {
                "tag": "v7_n2w0001_exact",
                "cmd": [
                    "--mode", "bf", "--n-elec", "2", "--omega", "0.001", "--e-dmc", "0.00730",
                    "--epochs", "4200", "--n-coll", "4096", "--oversample", "20", "--micro-batch", "1024",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0,8.0",
                    "--langevin-steps", "12", "--langevin-step-size", "0.002",
                    *adam_args("6e-5", "6e-5", "4.0", "0.01", "0.06", "40", "0.97", "0.0", "5.0"),
                    "--vmc-every", "60", "--vmc-n", "20000", "--n-eval", "80000", "--seed", "55",
                    "--resume", CKPT["n2w0001"],
                    "--no-pretrained",
                ],
            },
        ],
    },
    {
        "gpu": "1",
        "name": "n12_w1_continue",
        "jobs": [
            {
                "tag": "v7_n12w1_continue",
                "cmd": [
                    "--mode", "bf", "--n-elec", "12", "--omega", "1.0", "--e-dmc", "65.70010",
                    "--epochs", "2600", "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
                    *adam_args("1e-4", "1e-5", "3.0", "0.01", "0.01", "20", "0.95", "4.0", "3.5"),
                    "--vmc-every", "40", "--vmc-n", "12000", "--n-eval", "40000", "--seed", "61",
                    "--resume", CKPT["n12w1"],
                ],
            },
        ],
    },
    {
        "gpu": "2",
        "name": "n12_w01_continue",
        "jobs": [
            {
                "tag": "v7_n12w01_continue",
                "cmd": [
                    "--mode", "bf", "--n-elec", "12", "--omega", "0.1", "--e-dmc", "12.26984",
                    "--epochs", "3000", "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
                    *adam_args("8e-5", "8e-6", "4.0", "0.02", "0.03", "24", "0.95", "0.0", "4.0"),
                    "--vmc-every", "40", "--vmc-n", "12000", "--n-eval", "40000", "--seed", "62",
                    "--resume", CKPT["n12w01"],
                ],
            },
        ],
    },
    {
        "gpu": "3",
        "name": "n12_w001_cascade",
        "jobs": [
            {
                "tag": "v7_n12w001_cascade",
                "cmd": [
                    "--mode", "bf", "--n-elec", "12", "--omega", "0.01", "--e-dmc", "2.0",
                    "--epochs", "3600", "--n-coll", "4096", "--oversample", "16", "--micro-batch", "512",
                    "--sigma-fs", "0.8,1.3,2.0,3.5,6.0",
                    *adam_args("6e-5", "6e-6", "5.0", "0.02", "0.08", "32", "0.96", "0.0", "4.5"),
                    "--vmc-every", "50", "--vmc-n", "15000", "--n-eval", "50000", "--seed", "63",
                    "--init-jas", CKPT["n12w01"],
                    "--init-bf", CKPT["n12w01"],
                    "--no-pretrained",
                ],
            },
        ],
    },
]


def build_slot_cmd(slot):
    parts = []
    for job in slot["jobs"]:
        tag = job["tag"]
        log = LOGDIR / f"{tag}.log"
        cmd = " ".join(job["cmd"] + ["--tag", tag])
        run = (
            f"cd {ROOT}; {MODULE_CMD}; "
            f"CUDA_VISIBLE_DEVICES={slot['gpu']} timeout {MAX_WALL} "
            f"python3 src/run_weak_form.py {cmd}"
        )
        parts.append(
            f"echo \"# Started: $(date) GPU={slot['gpu']} tag={tag}\" >> {log}; "
            f"({run}) >> {log} 2>&1; "
            f"echo \"# Completed: $(date) rc=$?\" >> {log}"
        )
    return "; ".join(parts)


def main():
    missing = [k for k, p in CKPT.items() if not Path(p).is_file()]
    if missing:
        print("Missing checkpoints:")
        for k in missing:
            print(f"  - {k}: {CKPT[k]}")
        raise SystemExit(1)

    print(f"v7 campaign: N=2 exactness + N=12 continuation (wall={MAX_WALL})")
    print(f"Logs: {LOGDIR}")

    session = "v7n2n12"
    subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "55"])

    for i, slot in enumerate(SLOTS):
        if i > 0:
            subprocess.run(["tmux", "new-window", "-t", session])
        cmd = build_slot_cmd(slot)
        subprocess.run(["tmux", "send-keys", "-t", session, cmd, "Enter"])
        print(f"  GPU {slot['gpu']}: {slot['name']} -> {[j['tag'] for j in slot['jobs']]}")
        time.sleep(1)

    print(f"Launched session: {session}")
    print(f"Attach: tmux attach -t {session}")
    print(f"Monitor: tail -f {LOGDIR}/*.log")


if __name__ == "__main__":
    main()
