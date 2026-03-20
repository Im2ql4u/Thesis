#!/usr/bin/env python3
"""
Focused transfer campaign:
- Keep training omega=0.1
- Transfer from omega=0.1 to omega=0.01
- Skip omega=0.001 for now
- Try multiple configurations for N=20

Jobs are queued per GPU and run sequentially on that GPU.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "arch_colloc"
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
OUTDIR = ROOT / "outputs" / f"{datetime.now():%Y-%m-%d_%H%M}_n20_w01_to_w001_transfer"
LOGDIR = OUTDIR / "logs"

MODULE_CMD = (
    "source /etc/profile.d/lmod.sh 2>/dev/null; "
    "module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 2>/dev/null"
)


def cgsr(
    lr: str = "5e-3",
    lr_jas: str = "5e-4",
    damping: str = "1e-3",
    damping_end: str = "1e-4",
    anneal: str = "400",
    sub: str = "512",
    cg: str = "15",
    maxdp: str = "0.03",
    trust: str = "0.3",
    mom: str = "0.95",
    gclip: str = "0.5",
    clip_el: str = "3.0",
) -> List[str]:
    return [
        "--natural-grad",
        "--sr-mode",
        "cg",
        "--lr",
        lr,
        "--lr-jas",
        lr_jas,
        "--fisher-damping",
        damping,
        "--fisher-damping-end",
        damping_end,
        "--fisher-damping-anneal",
        anneal,
        "--fisher-subsample",
        sub,
        "--sr-cg-iters",
        cg,
        "--sr-max-param-change",
        maxdp,
        "--sr-trust-region",
        trust,
        "--nat-momentum",
        mom,
        "--grad-clip",
        gclip,
        "--clip-el",
        clip_el,
        "--direct-weight",
        "0.0",
    ]


def pick_existing(candidates: List[str]) -> str:
    for name in candidates:
        p = RESULTS / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No checkpoint found in candidates: {candidates}")


def parse_last_epoch(logfile: Path) -> Tuple[int, Optional[float]]:
    try:
        for line in reversed(logfile.read_text(errors="ignore").splitlines()):
            if "err=" in line and line.strip().startswith("["):
                ep = int(line.split("[")[1].split("]")[0].strip())
                err = line.split("err=")[1].split("%")[0].replace("+", "").strip()
                return ep, float(err)
    except Exception:
        pass
    return -1, None


def parse_final(logfile: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        for line in reversed(logfile.read_text(errors="ignore").splitlines()):
            if "*** Final:" in line and "E =" in line and "err =" in line:
                e_val = float(line.split("E =")[1].split("+-")[0].split("±")[0].strip())
                err = float(line.split("err =")[1].split("%")[0].replace("+", "").strip())
                return e_val, err
    except Exception:
        pass
    return None, None


def launch_job(job: dict) -> subprocess.Popen:
    logfile = LOGDIR / f"{job['tag']}.log"
    cmd = list(job["cmd"]) + ["--tag", job["tag"]]
    full_cmd = (
        f"cd {SRC}; {MODULE_CMD}; "
        f"CUDA_MANUAL_DEVICE={job['gpu']} python3 run_weak_form.py " + " ".join(cmd)
    )
    with logfile.open("w", encoding="utf-8") as f:
        f.write(f"# {job['name']} on GPU {job['gpu']}\n")
        f.write(f"# Start: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# {full_cmd}\n\n")
    return subprocess.Popen(
        ["bash", "-lc", f"{full_cmd} >> {logfile} 2>&1"],
        start_new_session=True,
    )


def build_queues() -> Tuple[Dict[str, List[dict]], str, str]:
    src_n6_w01 = pick_existing([
        "long_n6w01.pt",
        "w1_n6w01_transfer.pt",
        "camp_n6w01_hiLR.pt",
    ])
    src_n12_w01 = pick_existing([
        "long_n12w01_v3.pt",
        "long_n12w01.pt",
        "w1_n12w1_xfer.pt",
    ])
    src_n20_w01 = pick_existing([
        "camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt",
        "regime_low_L_n20_o0p1_s11.pt",
        "camp_jastrow_transfer_baseline_n20_o0p1_s11.pt",
        "smoke_n20_o0p1.pt",
    ])

    queues: Dict[str, List[dict]] = {
        "2": [
            {
                "name": "n6_w01_keep",
                "gpu": "2",
                "tag": f"{STAMP}_n6w01_keep",
                "cmd": [
                    "--mode", "bf", "--n-elec", "6", "--omega", "0.1",
                    "--epochs", "1800",
                    "--n-coll", "8192", "--oversample", "12", "--micro-batch", "512",
                    *cgsr(lr="1e-3", lr_jas="1e-4", damping="1e-4", damping_end="5e-5", anneal="800", sub="1024", cg="20", maxdp="0.02", trust="0.2", clip_el="3.0"),
                    "--vmc-every", "30", "--vmc-n", "15000",
                    "--n-eval", "50000",
                    "--seed", "42",
                    "--resume", src_n6_w01,
                ],
            },
            {
                "name": "n6_w001_xfer_from_w01",
                "gpu": "2",
                "tag": f"{STAMP}_n6w001_xfer_a",
                "cmd": [
                    "--mode", "bf", "--n-elec", "6", "--omega", "0.01",
                    "--epochs", "2600",
                    "--n-coll", "4096", "--oversample", "18", "--micro-batch", "512",
                    *cgsr(lr="3e-3", lr_jas="3e-4", damping="5e-3", damping_end="5e-5", anneal="1400", sub="512", cg="15", maxdp="0.05", trust="0.5", clip_el="3.0"),
                    "--vmc-every", "50", "--vmc-n", "12000",
                    "--n-eval", "40000",
                    "--seed", "42",
                    "--init-jas", src_n6_w01,
                    "--init-bf", src_n6_w01,
                    "--no-pretrained",
                ],
            },
        ],
        "3": [
            {
                "name": "n20_w01_keep_cfgA",
                "gpu": "3",
                "tag": f"{STAMP}_n20w01_keep_a",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
                    "--epochs", "600",
                    "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
                    "--lr", "5e-4", "--lr-jas", "5e-4",
                    "--direct-weight", "0.1", "--clip-el", "0.0", "--reward-qtrim", "0.02",
                    "--vmc-every", "40", "--vmc-n", "12000",
                    "--n-eval", "30000",
                    "--seed", "11",
                    "--resume", src_n20_w01,
                ],
            },
            {
                "name": "n20_w001_xfer_cfgA",
                "gpu": "3",
                "tag": f"{STAMP}_n20w001_xfer_a",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "0.01",
                    "--epochs", "1000",
                    "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
                    "--lr", "6e-4", "--lr-jas", "6e-4",
                    "--direct-weight", "0.05", "--clip-el", "0.0", "--reward-qtrim", "0.02",
                    "--vmc-every", "40", "--vmc-n", "10000",
                    "--n-eval", "25000",
                    "--seed", "11",
                    "--init-jas", src_n20_w01,
                    "--no-pretrained",
                ],
            },
        ],
        "7": [
            {
                "name": "n20_w01_keep_cfgB",
                "gpu": "7",
                "tag": f"{STAMP}_n20w01_keep_b",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "0.1",
                    "--epochs", "700",
                    "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
                    "--lr", "3e-4", "--lr-jas", "4e-4",
                    "--direct-weight", "0.1", "--clip-el", "0.0", "--reward-qtrim", "0.01",
                    "--vmc-every", "40", "--vmc-n", "12000",
                    "--n-eval", "30000",
                    "--seed", "22",
                    "--resume", src_n20_w01,
                ],
            },
            {
                "name": "n20_w001_xfer_cfgB",
                "gpu": "7",
                "tag": f"{STAMP}_n20w001_xfer_b",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "0.01",
                    "--epochs", "1100",
                    "--n-coll", "4096", "--oversample", "14", "--micro-batch", "512",
                    "--lr", "4e-4", "--lr-jas", "5e-4",
                    "--direct-weight", "0.1", "--clip-el", "0.0", "--reward-qtrim", "0.02",
                    "--vmc-every", "40", "--vmc-n", "10000",
                    "--n-eval", "25000",
                    "--seed", "22",
                    "--init-jas", src_n20_w01,
                    "--no-pretrained",
                ],
            },
            {
                "name": "n20_w001_xfer_cfgC",
                "gpu": "7",
                "tag": f"{STAMP}_n20w001_xfer_c",
                "cmd": [
                    "--mode", "jastrow", "--n-elec", "20", "--omega", "0.01",
                    "--epochs", "1100",
                    "--n-coll", "4096", "--oversample", "10", "--micro-batch", "512",
                    "--lr", "8e-4", "--lr-jas", "8e-4",
                    "--direct-weight", "0.0", "--clip-el", "0.0", "--reward-qtrim", "0.02",
                    "--vmc-every", "40", "--vmc-n", "10000",
                    "--n-eval", "25000",
                    "--seed", "77",
                    "--init-jas", src_n20_w01,
                    "--no-pretrained",
                ],
            },
        ],
        "4": [
            {
                "name": "n12_w01_keep",
                "gpu": "4",
                "tag": f"{STAMP}_n12w01_keep",
                "cmd": [
                    "--mode", "bf", "--n-elec", "12", "--omega", "0.1",
                    "--epochs", "1800",
                    "--n-coll", "4096", "--oversample", "12", "--micro-batch", "512",
                    *cgsr(lr="3e-3", lr_jas="3e-4", damping="5e-3", damping_end="1e-4", anneal="800", sub="512", cg="15", maxdp="0.05", trust="0.5", clip_el="3.0"),
                    "--vmc-every", "50", "--vmc-n", "10000",
                    "--n-eval", "30000",
                    "--seed", "42",
                    "--resume", src_n12_w01,
                ],
            },
            {
                "name": "n12_w001_xfer_from_w01",
                "gpu": "4",
                "tag": f"{STAMP}_n12w001_xfer_a",
                "cmd": [
                    "--mode", "bf", "--n-elec", "12", "--omega", "0.01",
                    "--epochs", "2200",
                    "--n-coll", "4096", "--oversample", "16", "--micro-batch", "512",
                    *cgsr(lr="2e-3", lr_jas="2e-4", damping="5e-3", damping_end="1e-4", anneal="1200", sub="512", cg="15", maxdp="0.05", trust="0.5", clip_el="3.0"),
                    "--vmc-every", "60", "--vmc-n", "8000",
                    "--n-eval", "25000",
                    "--seed", "42",
                    "--init-jas", src_n12_w01,
                    "--init-bf", src_n12_w01,
                    "--no-pretrained",
                ],
            },
        ],
    }
    return queues, src_n6_w01, src_n20_w01


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    queues, src_n6_w01, src_n20_w01 = build_queues()

    print(f"Campaign start: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Output dir: {OUTDIR}")
    print("Policy: train omega=0.1 and transfer to omega=0.01; omega=0.001 paused")
    print(f"Source n6,w0.1: {src_n6_w01}")
    print(f"Source n20,w0.1: {src_n20_w01}")

    (OUTDIR / "plan.json").write_text(json.dumps(queues, indent=2), encoding="utf-8")

    running: Dict[str, Tuple[dict, subprocess.Popen]] = {}
    finished: List[dict] = []

    for gpu, q in queues.items():
        if not q:
            continue
        job = q.pop(0)
        proc = launch_job(job)
        running[gpu] = (job, proc)
        print(f"  GPU {gpu}: launch {job['name']} tag={job['tag']} pid={proc.pid}")

    summary_path = OUTDIR / "results.json"

    while running:
        time.sleep(45)
        status_lines = []

        for gpu in sorted(list(running.keys()), key=int):
            job, proc = running[gpu]
            logfile = LOGDIR / f"{job['tag']}.log"

            if proc.poll() is None:
                ep, err = parse_last_epoch(logfile)
                if ep >= 0 and err is not None:
                    status_lines.append(f"GPU {gpu}: {job['tag']} ep={ep} err={err:+.3f}%")
                else:
                    status_lines.append(f"GPU {gpu}: {job['tag']} starting...")
                continue

            rc = int(proc.returncode)
            E, err = parse_final(logfile)
            finished.append({"tag": job["tag"], "gpu": gpu, "rc": rc, "E": E, "err": err})
            print(f"DONE GPU {gpu}: {job['tag']} rc={rc} E={E} err={err}")

            if queues[gpu]:
                next_job = queues[gpu].pop(0)
                next_proc = launch_job(next_job)
                running[gpu] = (next_job, next_proc)
                print(f"  GPU {gpu}: launch {next_job['name']} tag={next_job['tag']} pid={next_proc.pid}")
            else:
                del running[gpu]

            summary_path.write_text(json.dumps(finished, indent=2), encoding="utf-8")

        if status_lines:
            print("STATUS " + datetime.now().strftime("%H:%M:%S"))
            for line in status_lines:
                print("  " + line)

    summary_path.write_text(json.dumps(finished, indent=2), encoding="utf-8")
    print("Campaign complete.")
    print(f"Results: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
