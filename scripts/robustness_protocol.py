#!/usr/bin/env python3
"""Seed-robust protocol runner for weak-form collocation.

Purpose:
- Run the same (N, omega) setup across multiple seeds
- Compare a baseline policy vs a denoised/replay policy
- Emit a scorecard focused on reproducibility, not single best energy

This script is intentionally generic and uses src/run_collocation.py as backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "src" / "run_collocation.py"
RESULTS = ROOT / "results" / "arch_colloc"


@dataclass
class RunResult:
    policy: str
    seed: int
    tag: str
    returncode: int
    ckpt_path: str
    log_path: str
    E: float | None
    se: float | None
    err: float | None
    elapsed_s: float


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_metrics(ckpt_path: Path) -> tuple[float | None, float | None, float | None]:
    if not ckpt_path.exists():
        return None, None, None
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        E = ckpt.get("E")
        se = ckpt.get("se")
        err = ckpt.get("err")
        def _ok(v):
            return float(v) if v is not None and math.isfinite(float(v)) else None
        return _ok(E), _ok(se), _ok(err)
    except Exception:
        return None, None, None


def make_policy_args(policy: str, mode: str, base_lr: float, base_lr_jas: float) -> list[str]:
    # Keep policies simple and explicit for fair ablation.
    if policy == "baseline":
        return [
            "--mode", mode,
            "--direct-weight", "0.0",
            "--clip-el", "5.0",
            "--reward-qtrim", "0.0",
            "--replay-frac", "0.0",
            "--replay-top-frac", "0.25",
            "--lr", str(base_lr),
            "--lr-jas", str(base_lr_jas),
        ]
    if policy == "stabilized":
        return [
            "--mode", mode,
            "--direct-weight", "0.0",
            "--clip-el", "5.0",
            "--reward-qtrim", "0.02",
            "--replay-frac", "0.25",
            "--replay-top-frac", "0.25",
            "--lr", str(base_lr),
            "--lr-jas", str(base_lr_jas),
        ]
    raise ValueError(f"Unknown policy: {policy}")


def run_one(
    *,
    policy: str,
    seed: int,
    gpu: int,
    mode: str,
    n_elec: int,
    omega: float,
    epochs: int,
    n_coll: int,
    oversample: int,
    micro_batch: int,
    vmc_every: int,
    vmc_n: int,
    n_eval: int,
    base_lr: float,
    base_lr_jas: float,
    resume: str | None,
    out_dir: Path,
) -> RunResult:
    tag = f"proto_{policy}_n{n_elec}_o{str(omega).replace('.', 'p')}_s{seed}"
    log_path = out_dir / "logs" / f"{tag}.log"
    ckpt_path = RESULTS / f"{tag}.pt"

    args = [
        "python3", str(RUN),
        "--tag", tag,
        "--seed", str(seed),
        "--n-elec", str(n_elec),
        "--omega", str(omega),
        "--epochs", str(epochs),
        "--n-coll", str(n_coll),
        "--oversample", str(oversample),
        "--micro-batch", str(micro_batch),
        "--vmc-every", str(vmc_every),
        "--vmc-n", str(vmc_n),
        "--n-eval", str(n_eval),
    ]
    args.extend(make_policy_args(policy, mode, base_lr, base_lr_jas))
    if resume:
        args.extend(["--resume", resume])

    env = os.environ.copy()
    env["CUDA_MANUAL_DEVICE"] = str(gpu)

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[{now()}] START {tag} on GPU {gpu}\n")
        f.write("CMD: " + " ".join(args) + "\n\n")
        f.flush()
        proc = subprocess.run(args, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n[{now()}] END rc={proc.returncode}\n")
    dt = time.time() - t0

    E, se, err = load_metrics(ckpt_path)
    return RunResult(
        policy=policy,
        seed=seed,
        tag=tag,
        returncode=int(proc.returncode),
        ckpt_path=str(ckpt_path),
        log_path=str(log_path),
        E=E,
        se=se,
        err=err,
        elapsed_s=dt,
    )


def summarize(results: list[RunResult], dmc_tol_pct: float) -> dict:
    out: dict[str, dict] = {}
    policies = sorted(set(r.policy for r in results))
    for p in policies:
        rs = [r for r in results if r.policy == p and r.returncode == 0 and r.E is not None]
        Es = [r.E for r in rs if r.E is not None]
        errs = [r.err for r in rs if r.err is not None]
        pass_count = sum(1 for e in errs if e is not None and abs(e) <= dmc_tol_pct)
        out[p] = {
            "n_ok": len(rs),
            "E_mean": mean(Es) if Es else None,
            "E_std": pstdev(Es) if len(Es) > 1 else 0.0 if len(Es) == 1 else None,
            "err_mean_pct": mean(errs) if errs else None,
            "pass_count": pass_count,
            "pass_rate": (pass_count / len(errs)) if errs else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run reproducibility protocol across seeds")
    ap.add_argument("--mode", type=str, default="bf", choices=["bf", "jastrow", "pfaffian"])
    ap.add_argument("--n-elec", type=int, default=6)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--seeds", type=str, default="11,22,42")
    ap.add_argument("--policies", type=str, default="baseline,stabilized")
    ap.add_argument("--gpus", type=str, default="3")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-coll", type=int, default=4096)
    ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=512)
    ap.add_argument("--vmc-every", type=int, default=50)
    ap.add_argument("--vmc-n", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-jas", type=float, default=2e-5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--dmc-tol-pct", type=float, default=0.2,
                    help="Absolute percent-error threshold counted as pass")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not seeds or not policies or not gpus:
        raise ValueError("seeds/policies/gpus must be non-empty")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = ROOT / "outputs" / f"{stamp}_robustness_protocol"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    plan = {
        "created": now(),
        "mode": args.mode,
        "n_elec": args.n_elec,
        "omega": args.omega,
        "seeds": seeds,
        "policies": policies,
        "gpus": gpus,
        "epochs": args.epochs,
        "n_coll": args.n_coll,
        "oversample": args.oversample,
        "micro_batch": args.micro_batch,
        "vmc_every": args.vmc_every,
        "vmc_n": args.vmc_n,
        "n_eval": args.n_eval,
        "lr": args.lr,
        "lr_jas": args.lr_jas,
        "resume": args.resume,
        "dmc_tol_pct": args.dmc_tol_pct,
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"Robustness protocol start: {now()}")
    print(f"Output: {out_dir}")
    print(f"Policies: {policies} | Seeds: {seeds} | GPUs: {gpus}")
    print("=" * 70)

    results: list[RunResult] = []
    gpu_idx = 0
    for p in policies:
        for s in seeds:
            gpu = gpus[gpu_idx % len(gpus)]
            gpu_idx += 1
            print(f"[{now()}] RUN {p} seed={s} gpu={gpu}")
            rr = run_one(
                policy=p,
                seed=s,
                gpu=gpu,
                mode=args.mode,
                n_elec=args.n_elec,
                omega=args.omega,
                epochs=args.epochs,
                n_coll=args.n_coll,
                oversample=args.oversample,
                micro_batch=args.micro_batch,
                vmc_every=args.vmc_every,
                vmc_n=args.vmc_n,
                n_eval=args.n_eval,
                base_lr=args.lr,
                base_lr_jas=args.lr_jas,
                resume=args.resume,
                out_dir=out_dir,
            )
            results.append(rr)
            with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rr.__dict__) + "\n")
            print(f"[{now()}] END {rr.tag} rc={rr.returncode} E={rr.E} err={rr.err}")

    score = summarize(results, dmc_tol_pct=args.dmc_tol_pct)
    summary = {
        "created": now(),
        "scorecard": score,
        "results": [r.__dict__ for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nScorecard:")
    for p, m in score.items():
        print(f"  {p:10s}  pass_rate={100*m['pass_rate']:.1f}%  E_mean={m['E_mean']}  E_std={m['E_std']}")
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
