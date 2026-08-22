"""Overnight campaign orchestrator — a GPU job queue for the Grand Mechanism Program.

Each JOB is an omega-cascade chain (omega: 1 -> 0.1 -> 0.01, warm-started) for one
(paradigm, optimizer, arch, backflow, seed), run via scripts/run_phase_analysis.py. Jobs are
distributed across all GPUs by a worker-per-GPU queue; a failing job is isolated (logged, skipped)
so one crash cannot take down the night. Everything writes to results/analysis/<STAMP>/<tag>/.

Groups produced (so the post-run mechanism analysis can compare everything):
  A  architecture ablation (VMC): {ctnn,deepset} Jastrow x {backflow,none}, 2 seeds -> "what is MP worth"
  B  paradigm (collocation): ctnn+bf via weak-form collocation, 2 seeds -> internal-structure vs VMC
  C  optimizer x paradigm (Phase O) on ctnn+bf: vmc-adam, colloc-sr (+ vmc-sr from A, colloc-adam from B)
  D  scaling (bonus): N=12 via collocation (no Laplacian -> no OOM), ctnn+bf and deepset+bf

Run (detached): setsid python3 -u scripts/orchestrate_overnight.py --gpus 0-7 [--include-collocsr] [--include-n12] &
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP = "2026-07-02_overnight"
OUT = ROOT / "results/analysis" / STAMP

# recipe presets
VMC = dict(steps=500, n_seg=4, polish=120, srpolish=200)
VMC_ADAM = dict(steps=700, n_seg=4, polish=200, srpolish=0)     # pure Adam (Phase O)
COL = dict(steps=1400, n_seg=4, polish=300, srpolish=0)          # collocation needs more steps
CASCADE = ["1.0", "0.1", "0.01"]
CASCADE_N12 = ["1.0", "0.1"]  # N=12 bonus: stop at 0.1 (0.01 is the hardest)


def build_jobs(include_collocsr: bool, include_n12: bool) -> list[dict]:
    jobs = []

    def add(tag, paradigm, optimizer, arch, backflow, seed, recipe, N=6, cascade=CASCADE):
        jobs.append(dict(tag=tag, paradigm=paradigm, optimizer=optimizer, arch=arch,
                         backflow=backflow, seed=seed, recipe=recipe, N=N, cascade=cascade))

    # Group A: architecture ablation (VMC). optimizer=adam => segments train with Adam (warm), and the
    # VMC recipe's sr_polish_steps applies the annealed SR polish afterward. (optimizer="sr" would make
    # the SEGMENTS use from-scratch CG-SR, which diverges -- caught by the orchestrator smoke.)
    for arch, ashort in (("ctnn_vcycle_big", "ctnn"), ("deepset_big", "deepset")):
        for bf, bshort in ((True, "bf"), (False, "nobf")):
            for seed in (0, 1):
                add(f"A_{ashort}_{bshort}", "vmc", "adam", arch, bf, seed, VMC)
    # Group B: paradigm = collocation, ctnn+bf, 2 seeds
    for seed in (0, 1):
        add("B_ctnn_bf_colloc", "colloc", "adam", "ctnn_vcycle_big", True, seed, COL)
    # Group C: Phase O extras on ctnn+bf (vmc-adam, and colloc-sr if validated)
    add("C_ctnn_bf_vmcadam", "vmc", "adam", "ctnn_vcycle_big", True, 0, VMC_ADAM)
    if include_collocsr:
        add("C_ctnn_bf_collocsr", "colloc", "sr", "ctnn_vcycle_big", True, 0, COL)
    # Group D: N=12 scaling via collocation (no Laplacian OOM)
    if include_n12:
        for arch, ashort in (("ctnn_vcycle_big", "ctnn"), ("deepset_big", "deepset")):
            add(f"D_N12_{ashort}_bf_colloc", "colloc", "adam", arch, True, 0, COL, N=12, cascade=CASCADE_N12)
    return jobs


def run_job(job: dict, gpu: int, log) -> None:
    init = ""
    for w in job["cascade"]:
        out = OUT / f"{job['tag']}_s{job['seed']}_w{w.replace('.', 'p')}"
        out.mkdir(parents=True, exist_ok=True)
        r = job["recipe"]
        cmd = ["python3", "-u", str(ROOT / "scripts/run_phase_analysis.py"),
               "--N", str(job["N"]), "--omega", w, "--arch", job["arch"],
               "--paradigm", job["paradigm"], "--optimizer", job["optimizer"],
               "--steps", str(r["steps"]), "--n-seg", str(r["n_seg"]),
               "--polish-steps", str(r["polish"]), "--sr-polish-steps", str(r["srpolish"]),
               "--seed", str(job["seed"]), "--outdir", str(out)]
        if job["backflow"]:
            cmd.append("--backflow")
        if init:
            cmd += ["--init", init]
        # N=12: chunk the exact-Laplacian eval + smaller eval/align samples to avoid OOM (tested)
        if job["N"] >= 12:
            cmd += ["--eval-samples", "384", "--final-samples", "512", "--align-samples", "384",
                    "--batch", "512", "--micro-batch", "256"]
        env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "PATH": _PATH}
        log(f"[gpu{gpu}] START {job['tag']} s{job['seed']} w{w}")
        t0 = time.time()
        with open(out / "train.log", "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**_ENV, **env}).returncode
        log(f"[gpu{gpu}] DONE  {job['tag']} s{job['seed']} w{w} rc={rc} ({(time.time()-t0)/60:.0f} min)")
        if rc != 0 or not (out / "checkpoint.pt").exists():
            log(f"[gpu{gpu}] WARN  {job['tag']} w{w} failed/no-checkpoint -> stop this chain")
            return
        init = str(out / "checkpoint.pt")


def main():
    import os
    global _ENV, _PATH
    _ENV = dict(os.environ); _PATH = os.environ.get("PATH", "")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--include-collocsr", action="store_true")
    ap.add_argument("--include-n12", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny recipes + 2 jobs, to test the mechanics")
    a = ap.parse_args()
    if "-" in a.gpus:
        lo, hi = a.gpus.split("-"); gpus = list(range(int(lo), int(hi) + 1))
    else:
        gpus = [int(g) for g in a.gpus.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(a.include_collocsr, a.include_n12)
    if a.smoke:  # test the mechanics: 1 VMC + 1 colloc chain, 2-omega cascade (tests --init), tiny steps
        vmc_j = next(j for j in jobs if j["tag"] == "A_ctnn_bf" and j["seed"] == 0)
        col_j = next(j for j in jobs if j["tag"] == "B_ctnn_bf_colloc" and j["seed"] == 0)
        jobs = [vmc_j, col_j]
        for j in jobs:
            j["recipe"] = dict(steps=30, n_seg=2, polish=20, srpolish=20)
            j["cascade"] = ["1.0", "0.1"]
        gpus = gpus[:2]
    q: "queue.Queue[dict]" = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    logf = open(OUT / "orchestrator.log", "a")

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        with lock:
            print(line, flush=True); logf.write(line + "\n"); logf.flush()

    json.dump(jobs, open(OUT / "jobs.json", "w"), indent=2, default=str)
    log(f"=== overnight campaign: {len(jobs)} chains on GPUs {gpus} ===")

    def worker(gpu):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(job, gpu, log)
            except Exception as e:  # isolate: one bad chain can't kill the night
                log(f"[gpu{gpu}] EXC  {job['tag']}: {e!r}")
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=False) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("=== ALL CHAINS DONE ===")


if __name__ == "__main__":
    main()
