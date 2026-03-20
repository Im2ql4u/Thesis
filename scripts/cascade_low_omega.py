#!/usr/bin/env python3
"""
Cascade warm-start to ultra-low omega + continue all targets.

Key insight: warm-starting from higher-omega checkpoint works dramatically
better than training from scratch. ω=0.1 went from +43.7% (stuck) to
+0.321% using ω=1.0 weights.

This script cascades: ω=0.1 → ω=0.01 → ω=0.001
And continues/refines all other targets in parallel.

No wave structure — all jobs launched together. Each job is independent
and saves its own checkpoint. Run time: ~8-10 hours.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_cascade_low_omega"
LOGDIR = OUTDIR / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

MODULE_CMD = "source /etc/profile.d/lmod.sh 2>/dev/null; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"

# Best checkpoints from Wave 1
CKPT = {
    "n6w1": str(RESULTS / "camp_n6w1_verify.pt"),       # +0.012%
    "n6w05": str(RESULTS / "w1_n6w05_hisamp.pt"),       # -0.002%
    "n6w01": str(RESULTS / "w1_n6w01_transfer.pt"),     # +0.321%
    "n12w1": str(RESULTS / "w1_n12w1_xfer.pt"),         # +0.010%
    "n12w05": str(RESULTS / "camp_n12w05_smoke.pt"),    # +36.9%
}

DMC_REF = {
    (6, 1.0): 20.15932,
    (6, 0.5): 11.78484,
    (6, 0.1): 3.55385,
    (6, 0.01): 0.69036,
    (6, 0.001): 0.140832,
    (12, 1.0): 65.70010,
    (12, 0.5): 39.15960,
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


JOBS = [
    # ═══════════════════════════════════════════════════════
    # GPU 0: N=6 ω=0.01 — CASCADE from ω=0.1 checkpoint
    # DMC = 0.69036, corr_ratio = 11.5x
    # ═══════════════════════════════════════════════════════
    {
        "name": "n6_w001_cascade",
        "gpu": "0",
        "tag": "n6w001_cascade",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "20", "--micro-batch", "512",
            *cgsr(lr="3e-3", lr_jas="3e-4", damping="5e-3",
                  damping_end="1e-4", anneal="800",
                  sub="512", cg="15",
                  maxdp="0.05", trust="0.5", clip_el="3.0"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
            "--init-jas", CKPT["n6w01"],
            "--init-bf", CKPT["n6w01"],
            "--no-pretrained",
        ],
    },
    # ═══════════════════════════════════════════════════════
    # GPU 1: N=6 ω=0.001 — CASCADE from ω=0.1 checkpoint
    # DMC = 0.140832, corr_ratio = 23.5x (Wigner molecule)
    # Start from ω=0.1 (not ω=0.01) since ω=0.01 isn't trained yet
    # ═══════════════════════════════════════════════════════
    {
        "name": "n6_w0001_cascade",
        "gpu": "1",
        "tag": "n6w0001_cascade",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.001",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "24", "--micro-batch", "512",
            *cgsr(lr="1e-3", lr_jas="1e-4", damping="5e-3",
                  damping_end="1e-4", anneal="1000",
                  sub="512", cg="15",
                  maxdp="0.03", trust="0.3", clip_el="3.0"),
            "--vmc-every", "50", "--vmc-n", "8000",
            "--n-eval", "25000",
            "--seed", "42",
            "--init-jas", CKPT["n6w01"],
            "--init-bf", CKPT["n6w01"],
            "--no-pretrained",
        ],
    },
    # ═══════════════════════════════════════════════════════
    # GPU 2: N=6 ω=0.1 — REFINE from +0.321% to <0.1%
    # ═══════════════════════════════════════════════════════
    {
        "name": "n6_w01_refine",
        "gpu": "2",
        "tag": "n6w01_refine",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
            "--epochs", "1500",
            "--n-coll", "8192", "--oversample", "12", "--micro-batch", "512",
            *cgsr(lr="1e-3", lr_jas="1e-4", damping="1e-4",
                  damping_end="5e-5", anneal="600",
                  sub="1024", cg="20",
                  maxdp="0.02", trust="0.2", clip_el="3.0"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "50000",
            "--seed", "42",
            "--resume", CKPT["n6w01"],
        ],
    },
    # ═══════════════════════════════════════════════════════
    # GPU 3: N=12 ω=1.0 — already at +0.010%, ultra-polish
    # ═══════════════════════════════════════════════════════
    {
        "name": "n12_w1_polish",
        "gpu": "3",
        "tag": "n12w1_polish",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "1.0",
            "--epochs", "1000",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr(lr="5e-4", lr_jas="5e-5", damping="5e-5",
                  sub="1024", cg="20",
                  maxdp="0.01", trust="0.1"),
            "--vmc-every", "30", "--vmc-n", "15000",
            "--n-eval", "40000",
            "--seed", "42",
            "--resume", CKPT["n12w1"],
        ],
    },
    # ═══════════════════════════════════════════════════════
    # GPU 4: N=12 ω=0.5 — CASCADE from N=12 ω=1.0
    # DMC = 39.15960
    # ═══════════════════════════════════════════════════════
    {
        "name": "n12_w05_cascade",
        "gpu": "4",
        "tag": "n12w05_cascade",
        "cmd": [
            "--mode", "bf", "--n-elec", "12", "--omega", "0.5",
            "--epochs", "1500",
            "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
            *cgsr(lr="5e-3", lr_jas="5e-4", damping="5e-3",
                  damping_end="1e-4", anneal="600",
                  sub="512", cg="15",
                  maxdp="0.05", trust="0.5"),
            "--vmc-every", "40", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "42",
            "--init-jas", CKPT["n12w1"],
            "--init-bf", CKPT["n12w1"],
            "--no-pretrained",
        ],
    },
    # ═══════════════════════════════════════════════════════
    # GPU 5: N=6 ω=0.01 — variant with higher LR
    # ═══════════════════════════════════════════════════════
    {
        "name": "n6_w001_fast",
        "gpu": "5",
        "tag": "n6w001_fast",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
            "--epochs", "2000",
            "--n-coll", "4096", "--oversample", "16", "--micro-batch", "512",
            *cgsr(lr="5e-3", lr_jas="5e-4", damping="5e-3",
                  damping_end="1e-4", anneal="800",
                  sub="512", cg="15",
                  maxdp="0.05", trust="0.5", clip_el="3.0"),
            "--vmc-every", "50", "--vmc-n", "10000",
            "--n-eval", "30000",
            "--seed", "123",
            "--init-jas", CKPT["n6w01"],
            "--init-bf", CKPT["n6w01"],
            "--no-pretrained",
        ],
    },
    # ═══════════════════════════════════════════════════════
    # GPU 6: N=6 ω=0.5 — polish from -0.002%
    # Already below DMC — just do heavy eval to nail down the number
    # ═══════════════════════════════════════════════════════
    {
        "name": "n6_w05_polish",
        "gpu": "6",
        "tag": "n6w05_polish",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "0.5",
            "--epochs", "600",
            "--n-coll", "8192", "--oversample", "10", "--micro-batch", "512",
            *cgsr(lr="3e-4", lr_jas="3e-5", damping="5e-5",
                  sub="2048", cg="25",
                  maxdp="0.01", trust="0.1", clip_el="3.0"),
            "--vmc-every", "20", "--vmc-n", "25000",
            "--n-eval", "100000",
            "--seed", "42",
            "--resume", CKPT["n6w05"],
        ],
    },
    # ═══════════════════════════════════════════════════════
    # GPU 7: N=6 ω=1.0 — ultra-polish with 100k eval
    # Already at +0.012% — confirm with highest-quality eval
    # ═══════════════════════════════════════════════════════
    {
        "name": "n6_w1_ultra",
        "gpu": "7",
        "tag": "n6w1_ultra",
        "cmd": [
            "--mode", "bf", "--n-elec", "6", "--omega", "1.0",
            "--epochs", "600",
            "--n-coll", "8192", "--oversample", "8", "--micro-batch", "512",
            *cgsr(lr="3e-4", lr_jas="3e-5", damping="5e-5",
                  sub="2048", cg="25",
                  maxdp="0.01", trust="0.1", clip_el="3.0"),
            "--vmc-every", "20", "--vmc-n", "25000",
            "--n-eval", "100000",
            "--seed", "42",
            "--resume", CKPT["n6w1"],
        ],
    },
]


def main():
    started = datetime.now()
    print(f"Cascade low-omega campaign — {started:%Y-%m-%d %H:%M}")
    print(f"Output: {OUTDIR}")
    print(f"Jobs: {len(JOBS)}")
    print()
    print("DMC references:")
    for (n, w), e in sorted(DMC_REF.items()):
        print(f"  N={n} ω={w}: {e}")
    print()

    (OUTDIR / "plan.json").write_text(json.dumps(JOBS, indent=2, default=str))

    procs = []
    for job in JOBS:
        tag = job["tag"]
        gpu = job["gpu"]
        logfile = LOGDIR / f"{tag}.log"
        cmd = list(job["cmd"])

        if "--tag" not in cmd:
            cmd.extend(["--tag", tag])

        full_cmd = (
            f"cd {SRC}; {MODULE_CMD}; "
            f"CUDA_MANUAL_DEVICE={gpu} python run_weak_form.py "
            + " ".join(cmd)
        )

        with open(logfile, "w") as lf:
            lf.write(f"# {job['name']} — GPU {gpu}\n")
            lf.write(f"# {full_cmd}\n\n")

        # Identify init/resume source for logging
        init_src = None
        for i, c in enumerate(cmd):
            if c in ("--resume", "--init-bf") and i + 1 < len(cmd):
                init_src = Path(cmd[i + 1]).name
                break

        n_el = omega = "?"
        for i, c in enumerate(cmd):
            if c == "--n-elec" and i + 1 < len(cmd): n_el = cmd[i + 1]
            if c == "--omega" and i + 1 < len(cmd): omega = cmd[i + 1]

        dmc = DMC_REF.get((int(n_el), float(omega)), "?")
        print(f"  GPU {gpu}: [{job['name']}] N={n_el} ω={omega} dmc={dmc}"
              f"  init={init_src or 'scratch'}")

        proc = subprocess.Popen(
            ["bash", "-c", f"{full_cmd} >> {logfile} 2>&1"],
            start_new_session=True,
        )
        procs.append((job, proc))
        time.sleep(3)

    print(f"\nAll {len(procs)} jobs launched.")
    print(f"Monitor: tail -f {LOGDIR}/*.log")
    print(f"PIDs: {[p.pid for _, p in procs]}")
    print()

    # Wait for all to complete
    results = []
    for job, proc in procs:
        rc = proc.wait()
        logfile = LOGDIR / f"{job['tag']}.log"
        final_E = final_err = "?"
        try:
            for line in reversed(logfile.read_text().splitlines()):
                if "*** Final:" in line:
                    final_E = line.split("E =")[1].split("±")[0].strip()
                    final_err = line.split("err =")[1].strip().split("%")[0] + "%"
                    break
        except Exception:
            pass

        n_el = omega = "?"
        for i, c in enumerate(job["cmd"]):
            if c == "--n-elec" and i + 1 < len(job["cmd"]): n_el = job["cmd"][i + 1]
            if c == "--omega" and i + 1 < len(job["cmd"]): omega = job["cmd"][i + 1]

        dmc = DMC_REF.get((int(n_el), float(omega)), "?")
        result = {
            "name": job["name"], "tag": job["tag"],
            "N": n_el, "omega": omega, "dmc": dmc,
            "rc": rc, "final_E": final_E, "final_err": final_err,
        }
        results.append(result)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{job['name']}] {status}  N={n_el} ω={omega}"
              f"  final={final_E} ({final_err})  dmc={dmc}")

    elapsed = datetime.now() - started
    print(f"\nCampaign complete — {elapsed}")

    (OUTDIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Results: {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
