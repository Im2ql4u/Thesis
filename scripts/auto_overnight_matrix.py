#!/usr/bin/env python3
"""Automated multi-GPU experiment runner — correct two-phase design.

Phase A (parallel):
  - N=6, ω=1.0: BF runs using existing pretrained bf_ctnn_vcycle.pt
    (no jastrow warmup needed — ctnn_vcycle.pt E=20.21 already done)
  - N=12 targets: proper jastrow warmup (400 epochs from scratch)

Phase B (parallel, after phase A):
  - N=12 targets: BF runs initialized from phase A jastrow checkpoints

Design rationale:
  The original run used bf_ctnn_vcycle.pt (E=20.19) as BF+jastrow init for N=6,
  then trained 500 epochs to get bf_hardfocus_v1b.pt (E=20.161, err=+0.010%).
  That recipe is replicated here. N=12 needs its own jastrow first.
"""

import argparse
import json
import math
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = ROOT / "src" / "run_weak_form.py"
RESULTS_DIR = ROOT / "results" / "arch_colloc"

# Existing pretrained checkpoints for N=6 ω=1.0
PRETRAINED_JAS = RESULTS_DIR / "ctnn_vcycle.pt"       # E=20.214, 25562 params
PRETRAINED_BF  = RESULTS_DIR / "bf_ctnn_vcycle.pt"    # E=20.188, jas+bf states

DMC_REFS = {
    (6, 1.0): 20.15932,
}


class Job:
    def __init__(self, name, gpu, args, target_key, stage, fallback=False):
        self.name = name
        self.gpu = gpu
        self.args = args
        self.target_key = target_key
        self.stage = stage
        self.fallback = fallback


class JobResult:
    def __init__(self, name, gpu, target_key, stage, fallback, returncode,
                 start_ts, end_ts, log_path, ckpt_path, E, se, err):
        self.name = name
        self.gpu = gpu
        self.target_key = target_key
        self.stage = stage
        self.fallback = fallback
        self.returncode = returncode
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.log_path = log_path
        self.ckpt_path = ckpt_path
        self.E = E
        self.se = se
        self.err = err


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_metrics(ckpt_path):
    if not ckpt_path.exists():
        return {"E": None, "se": None, "err": None}
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        E = ckpt.get("E")
        se = ckpt.get("se")
        err = ckpt.get("err")
        return {
            "E": float(E) if E is not None and math.isfinite(float(E)) else None,
            "se": float(se) if se is not None and math.isfinite(float(se)) else None,
            "err": float(err) if err is not None and math.isfinite(float(err)) else None,
        }
    except Exception:
        return {"E": None, "se": None, "err": None}


def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def pick_best(results, target_key, stage):
    candidates = [
        r for r in results
        if r.stage == stage and r.target_key == target_key
        and r.returncode == 0 and r.E is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r.E)


def run_one(job, out_dir):
    log_path = out_dir / "logs" / (job.name + ".log")
    ckpt_path = RESULTS_DIR / (job.name + ".pt")
    cmd = ["python3", str(RUN_SCRIPT)] + job.args + ["--tag", job.name]

    env = os.environ.copy()
    env["CUDA_MANUAL_DEVICE"] = str(job.gpu)

    start = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[%s] START %s on GPU %s\n" % (now_str(), job.name, job.gpu))
        f.write("CMD: %s\n\n" % " ".join(cmd))
        f.flush()
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT
        )
        f.write("\n[%s] END rc=%s\n" % (now_str(), proc.returncode))

    metrics = load_metrics(ckpt_path)
    end = time.time()
    return JobResult(
        name=job.name,
        gpu=job.gpu,
        target_key=job.target_key,
        stage=job.stage,
        fallback=job.fallback,
        returncode=proc.returncode,
        start_ts=start,
        end_ts=end,
        log_path=str(log_path),
        ckpt_path=str(ckpt_path),
        E=metrics["E"],
        se=metrics["se"],
        err=metrics["err"],
    )


def run_job_list(job_list, out_dir, deadline, max_workers, results_accum):
    """Run a list of jobs in parallel up to max_workers. No fallback logic."""
    work_q = queue.Queue()
    for job in job_list:
        work_q.put(job)

    lock = threading.Lock()

    def worker():
        while True:
            if time.time() >= deadline:
                return
            try:
                job = work_q.get_nowait()
            except queue.Empty:
                return

            result = run_one(job, out_dir)
            with lock:
                results_accum.append(result)
                append_jsonl(out_dir / "results.jsonl", dict(result.__dict__))
            work_q.task_done()

    n_workers = min(max_workers, max(1, work_q.qsize()))
    threads = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(n_workers)
    ]
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join()


# ──────────────────────────────────────────────────────────────────────────────
# Job builders
# ──────────────────────────────────────────────────────────────────────────────

def _tag(n, om):
    return "n%d_o%s" % (n, str(om).replace(".", "p"))


def make_n6_bf_jobs(gpus, bf_epochs):
    """N=6 BF runs. No --no-pretrained: loads bf_ctnn_vcycle.pt automatically.
    This replicates the recipe that produced bf_hardfocus_v1b.pt (E=20.161).
    Two seeds for statistical confidence.
    """
    jobs = []
    for seed_idx, seed in enumerate((42, 99)):
        jobs.append(Job(
            name="n6_o1p0_bf_repro_s%d" % seed,
            gpu=gpus[seed_idx % len(gpus)],
            target_key=(6, 1.0),
            stage="bf",
            args=[
                "--mode", "bf",
                "--n-elec", "6",
                "--omega", "1.0",
                # NO --no-pretrained: loads bf_ctnn_vcycle.pt (jas+bf init at E=20.19)
                "--epochs", str(bf_epochs),
                "--n-coll", "4096",
                "--oversample", "8",
                "--micro-batch", "512",
                "--lr", "5e-4",
                "--lr-jas", "5e-5",
                "--grad-clip", "1.0",
                "--clip-el", "4.0",          # "hardfocus" — the key ingredient
                "--direct-weight", "0.1",    # default hybrid weight
                "--vmc-every", "50",
                "--seed", str(seed),
            ],
        ))
    return jobs


def make_n12_jastrow_jobs(targets, gpus, jas_epochs):
    """Jastrow warmup for N=12 targets from scratch.
    400 epochs is necessary — 90 was nowhere nearly enough.
    Uses --no-pretrained because ctnn_vcycle.pt is N=6 only.
    """
    jobs = []
    gpu_idx = 2  # start after the 2 N=6 BF slots
    for target in targets:
        n, om = target["key"]
        if n == 6 and om == 1.0:
            continue  # Skip: pretrained already exists
        for seed in (11, 22):
            jobs.append(Job(
                name="%s_jas_s%d" % (_tag(n, om), seed),
                gpu=gpus[gpu_idx % len(gpus)],
                target_key=(n, om),
                stage="jastrow",
                args=[
                    "--mode", "jastrow",
                    "--n-elec", str(n),
                    "--omega", str(om),
                    "--no-pretrained",       # N=12: no pretrained exists
                    "--epochs", str(jas_epochs),
                    "--n-coll", "4096",
                    "--oversample", "8",
                    "--micro-batch", "512",
                    "--lr", "3e-4",          # slightly lower for jastrow stability
                    "--grad-clip", "1.0",
                    "--clip-el", "0.0",      # no clipping during initial jastrow
                    "--direct-weight", "0.1",
                    "--vmc-every", "50",
                    "--seed", str(seed),
                ],
            ))
            gpu_idx += 1
    return jobs


def make_n12_bf_jobs(targets, gpus, results_accum, bf_epochs):
    """BF runs for N=12 targets, initialized from the best N=12 jastrow.
    Uses --no-pretrained and --init-jas since bf_ctnn_vcycle.pt is N=6 only.
    """
    jobs = []
    gpu_idx = 2
    for target in targets:
        n, om = target["key"]
        if n == 6 and om == 1.0:
            continue
        best_jas = pick_best(results_accum, (n, om), "jastrow")
        if best_jas is None:
            print("[WARN] No jastrow result for %s — skipping BF" % _tag(n, om))
            continue
        print("[INFO] N=%d ω=%s best jastrow: E=%.4f (%s)"
              % (n, om, best_jas.E or float("nan"), best_jas.name))
        for seed_idx, seed in enumerate((11, 22)):
            jobs.append(Job(
                name="%s_bf_s%d" % (_tag(n, om), seed),
                gpu=gpus[gpu_idx % len(gpus)],
                target_key=(n, om),
                stage="bf",
                args=[
                    "--mode", "bf",
                    "--n-elec", str(n),
                    "--omega", str(om),
                    "--no-pretrained",
                    "--init-jas", best_jas.ckpt_path,
                    "--epochs", str(bf_epochs),
                    "--n-coll", "4096",
                    "--oversample", "8",
                    "--micro-batch", "512",
                    "--lr", "5e-4",
                    "--lr-jas", "5e-5",
                    "--grad-clip", "1.0",
                    "--clip-el", "4.0",
                    "--direct-weight", "0.1",
                    "--vmc-every", "50",
                    "--seed", str(seed),
                ],
            ))
            gpu_idx += 1
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Two-phase overnight runner: N=6 BF (pretrained init) + "
                    "N=12 jastrow→BF pipeline"
    )
    ap.add_argument("--hours",         type=float, default=6.0)
    ap.add_argument("--gpus",          type=str,   default="1,3,4,5,7")
    ap.add_argument("--max-workers",   type=int,   default=5)
    ap.add_argument("--n6-bf-epochs",  type=int,   default=500,
                    help="BF epochs for N=6 (replicates hardfocus recipe)")
    ap.add_argument("--n12-jas-epochs", type=int,  default=400,
                    help="Jastrow epochs for N=12 (needs many more than 90)")
    ap.add_argument("--n12-bf-epochs", type=int,   default=400,
                    help="BF epochs for N=12")
    args = ap.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")

    targets = [
        {"key": (6,  1.0)},
        {"key": (12, 1.0)},
        {"key": (12, 0.5)},
    ]

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = ROOT / "outputs" / (stamp + "_overnight_auto")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    plan = {
        "created":        now_str(),
        "hours":          args.hours,
        "gpus":           gpus,
        "n6_bf_epochs":   args.n6_bf_epochs,
        "n12_jas_epochs": args.n12_jas_epochs,
        "n12_bf_epochs":  args.n12_bf_epochs,
        "targets":        [t["key"] for t in targets],
        "design": (
            "PhaseA: N=6 BF (from bf_ctnn_vcycle.pt) + N=12 jastrow in parallel. "
            "PhaseB: N=12 BF from PhaseA jastrow results."
        ),
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("  Overnight matrix — %s" % now_str())
    print("  GPUs: %s   max_workers: %d   deadline: %.1fh" % (gpus, args.max_workers, args.hours))
    print("  Phase A: N=6 BF (%d ep) + N=12 jastrow (%d ep) in parallel"
          % (args.n6_bf_epochs, args.n12_jas_epochs))
    print("  Phase B: N=12 BF (%d ep)" % args.n12_bf_epochs)
    print("=" * 64 + "\n")

    deadline = time.time() + args.hours * 3600.0
    results_accum = []

    # ── Phase A ──
    # N=6 BF (uses pretrained bf_ctnn_vcycle.pt as init, like hardfocus)
    # N=12 jastrow from scratch (enough epochs to actually converge)
    phase_a_jobs = (
        make_n6_bf_jobs(gpus, args.n6_bf_epochs)
        + make_n12_jastrow_jobs(targets, gpus, args.n12_jas_epochs)
    )
    print("[%s] Phase A: %d jobs" % (now_str(), len(phase_a_jobs)))
    for j in phase_a_jobs:
        print("  GPU%-2s  %s" % (j.gpu, j.name))
    print()

    run_job_list(phase_a_jobs, out_dir, deadline, args.max_workers, results_accum)

    # ── Phase B ──
    if time.time() >= deadline:
        print("[%s] Deadline reached — skipping Phase B" % now_str())
    else:
        phase_b_jobs = make_n12_bf_jobs(targets, gpus, results_accum, args.n12_bf_epochs)
        print("\n[%s] Phase B: %d jobs" % (now_str(), len(phase_b_jobs)))
        for j in phase_b_jobs:
            print("  GPU%-2s  %s" % (j.gpu, j.name))
        print()
        run_job_list(phase_b_jobs, out_dir, deadline, args.max_workers, results_accum)

    # ── Summary ──
    summary = {
        "created":          now_str(),
        "hours":            args.hours,
        "deadline_reached": time.time() >= deadline,
        "total_runs":       len(results_accum),
        "completed_ok":     sum(1 for r in results_accum if r.returncode == 0),
        "completed_fail":   sum(1 for r in results_accum if r.returncode != 0),
        "best_by_target":   {},
    }
    for target in targets:
        key = target["key"]
        vals = [
            r for r in results_accum
            if r.target_key == key and r.E is not None and r.returncode == 0
        ]
        if vals:
            best = min(vals, key=lambda r: r.E)
            summary["best_by_target"]["N%d_om%s" % (key[0], key[1])] = {
                "name": best.name,
                "E":    best.E,
                "se":   best.se,
                "err":  best.err,
                "ckpt": best.ckpt_path,
            }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 64)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
