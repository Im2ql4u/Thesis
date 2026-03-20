#!/usr/bin/env python3
"""One-shot launcher: N=6 low-omega BF runs from existing Jastrow checkpoints.

Targets:
  N=6 ω=0.01   init from n6_o0p01_jas_transfer_s42.pt    DMC=0.69036
  N=6 ω=0.001  init from n6_o0p001_jas_transfer_s42.pt   DMC=0.140832

Recipe: pure REINFORCE (direct-weight=0, clip-el=5) matching the winning
N=6 ω=1.0 chain (bf_joint_reinf_v3 → bf_hardfocus_v1b).

Usage:
  python scripts/launch_lowomega_bf.py [--gpus 3,6] [--epochs 500]
"""

import argparse
import json
import math
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = ROOT / "src" / "run_collocation.py"
RESULTS_DIR = ROOT / "results" / "arch_colloc"

JOBS = [
    {
        "tag":     "n6_o0p01_bf_s42",
        "n_elec":  6,
        "omega":   0.01,
        "e_dmc":   0.69036,
        "init_jas": RESULTS_DIR / "n6_o0p01_jas_transfer_s42.pt",
        "gpu_idx": 0,   # index into --gpus list
    },
    {
        "tag":     "n6_o0p001_bf_s42",
        "n_elec":  6,
        "omega":   0.001,
        "e_dmc":   0.140832,
        "init_jas": RESULTS_DIR / "n6_o0p001_jas_transfer_s42.pt",
        "gpu_idx": 1,
    },
]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_metrics(ckpt_path: Path):
    if not ckpt_path.exists():
        return None, None, None
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        E   = ckpt.get("E")
        se  = ckpt.get("se")
        err = ckpt.get("err")
        return (
            float(E)   if E   is not None and math.isfinite(float(E))   else None,
            float(se)  if se  is not None and math.isfinite(float(se))  else None,
            float(err) if err is not None and math.isfinite(float(err)) else None,
        )
    except Exception:
        return None, None, None


def run_job(job: dict, gpu: int, epochs: int, n_coll: int, out_dir: Path):
    tag = job["tag"]
    log_path = out_dir / "logs" / f"{tag}.log"
    ckpt_path = RESULTS_DIR / f"{tag}.pt"

    # Verify init checkpoint exists before launching
    if not job["init_jas"].exists():
        print(f"[ERROR] Missing init checkpoint: {job['init_jas']}")
        return

    cmd = [
        "python3", str(RUN_SCRIPT),
        "--mode",          "bf",
        "--n-elec",        str(job["n_elec"]),
        "--omega",         str(job["omega"]),
        "--e-dmc",         str(job["e_dmc"]),
        "--no-pretrained",
        "--init-jas",      str(job["init_jas"]),
        "--epochs",        str(epochs),
        "--n-coll",        str(n_coll),
        "--oversample",    "8",
        "--micro-batch",   "512",
        "--lr",            "5e-4",
        "--lr-jas",        "5e-5",
        "--grad-clip",     "1.0",
        # Pure REINFORCE + moderate clip — the recipe that worked for N=6 ω=1.0
        "--direct-weight", "0.0",
        "--clip-el",       "5.0",
        "--vmc-every",     "50",
        "--seed",          "42",
        "--tag",           tag,
    ]

    env = os.environ.copy()
    env["CUDA_MANUAL_DEVICE"] = str(gpu)

    print(f"[{now_str()}] START  {tag}  GPU={gpu}")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[{now_str()}] START {tag} on GPU {gpu}\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                              stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n[{now_str()}] END rc={proc.returncode}\n")

    elapsed = time.time() - start
    E, se, err = load_metrics(ckpt_path)
    status = "OK" if proc.returncode == 0 else f"rc={proc.returncode}"
    print(f"[{now_str()}] DONE   {tag}  {status}  E={E}  ±{se}  err={err}%  ({elapsed/60:.1f}min)")

    result = {
        "tag": tag, "gpu": gpu, "n_elec": job["n_elec"], "omega": job["omega"],
        "e_dmc": job["e_dmc"], "returncode": proc.returncode,
        "elapsed_s": elapsed, "E": E, "se": se, "err": err,
        "log": str(log_path), "ckpt": str(ckpt_path),
    }
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Launch N=6 low-omega BF jobs")
    ap.add_argument("--gpus",   default="3,6",
                    help="Comma-separated GPU indices, one per job")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--n-coll", type=int, default=4096)
    args = ap.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",")]
    if len(gpus) < len(JOBS):
        # Wrap around if fewer GPUs than jobs
        gpus = [gpus[i % len(gpus)] for i in range(len(JOBS))]

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = ROOT / "outputs" / f"{stamp}_n6_lowomega_bf"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    plan = {
        "created": now_str(),
        "gpus": gpus,
        "epochs": args.epochs,
        "n_coll": args.n_coll,
        "jobs": [j["tag"] for j in JOBS],
        "recipe": "pure REINFORCE (direct-weight=0, clip-el=5) — winning N=6 ω=1.0 recipe",
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))

    print(f"\n{'='*60}")
    print(f"  N=6 low-omega BF launcher — {now_str()}")
    print(f"  Output: {out_dir}")
    print(f"  Jobs: {[j['tag'] for j in JOBS]}")
    print(f"  GPUs: {gpus}   epochs={args.epochs}   n_coll={args.n_coll}")
    print(f"{'='*60}\n")

    threads = []
    for i, job in enumerate(JOBS):
        gpu = gpus[job["gpu_idx"]]
        t = threading.Thread(
            target=run_job,
            args=(job, gpu, args.epochs, args.n_coll, out_dir),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"\n[{now_str()}] All jobs finished. Results in {out_dir}")


if __name__ == "__main__":
    main()
