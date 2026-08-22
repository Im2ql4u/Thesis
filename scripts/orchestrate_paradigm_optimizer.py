"""Q2 (SR vs Adam) and Q3 (collocation vs VMC), together, on the thesis ansatz (PINN + CTNN backflow).

The two questions share one controlled design. For each (N, omega, seed):
  1. build a COMMON BASE with the staged curriculum through stage 3 (backflow alive, joint NOT done)
     -- run_phase_analysis --build-base-only. All cells start from this identical state.
  2. run the joint phase FOUR ways, warm-started from that base:
        vmc_adam | vmc_sr | colloc_adam | colloc_sr
     so the cells differ ONLY in optimizer x paradigm, not the starting point.

Q2 (optimizer): compare vmc_adam vs vmc_sr (and colloc_adam vs colloc_sr) -- energy gap, and whether
   SR's advantage tracks the QGT condition number kappa(S) / d_eff (the tangent-kernel prediction).
Q3 (paradigm): compare vmc_* vs colloc_* -- NOT just energy but whether they reach the SAME STATE
   (overlap^2 between the trained wavefunctions + agreement of backflow rank / d_eff). N=2 gives the
   exact-overlap ground truth; N=6 is the real test. Fixed-proposal collocation is expected to lose
   ESS at low omega -- that domain-of-validity is itself a Q3 finding (analyse_paradigm.py reports it).

Robustness mirrors the scaling orchestrator: free-GPU-only scheduling (mem AND util), OOM retry with
halved sizes, stall detection, per-chain isolation, resume (skip any stage that already has a
checkpoint). Runs unattended for days.

Run: nohup python3 -u scripts/orchestrate_paradigm_optimizer.py > results/analysis/paradigm.log 2>&1 &
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP = "2026-08-05_paradigm_optimizer"
OUTROOT = ROOT / "results/analysis" / STAMP

STALL_SECONDS = 120 * 60
POLL_SECONDS = 120
MAX_ATTEMPTS = 3

# N=2 gives an EXACT-overlap ground truth for Q3; N=6 is the real test. omega spans the three regimes
# (kinetic / crossover / Wigner). Sample/batch sizes shrink with N as usual.
N_CFG = {
    2: dict(batch=2048, eval_samples=1024, final_samples=8192, align=2048),
    6: dict(batch=2048, eval_samples=1024, final_samples=8192, align=2048),
}
OMEGAS = [1.0, 0.1, 0.01]
SEEDS = [0, 1]
BASE_STEPS = 3000          # staged build through stage 3 (~stages 1-3 get 0.3+0.3 of this)
JOINT_STEPS = 1500         # the compared phase
CELLS = ["vmc_adam", "vmc_sr", "colloc_adam", "colloc_sr"]


def wtag(w: float) -> str:
    return ("w%g" % w).replace(".", "p")


def base_dir(N, seed, w):
    return OUTROOT / f"base_N{N}_s{seed}_{wtag(w)}"


def cell_dir(N, seed, w, cell):
    return OUTROOT / f"N{N}_s{seed}_{wtag(w)}_{cell}"


def base_cmd(N, seed, w, cfg, shrink):
    f = 2 ** shrink
    return [
        "python3", "-u", str(ROOT / "scripts/run_phase_analysis.py"),
        "--N", str(N), "--omega", str(w), "--arch", "pinn",
        "--backflow", "--backflow-arch", "ctnn", "--build-base-only",
        "--bf-scale-init", "0.7", "--bf-zero-init-last", "0", "--cusp-steps", "0",
        "--steps", str(BASE_STEPS), "--batch", str(max(128, cfg["batch"] // f)),
        "--eval-samples", str(max(128, cfg["eval_samples"] // f)),
        "--seed", str(seed), "--outdir", str(base_dir(N, seed, w)),
    ]


def cell_cmd(N, seed, w, cell, init, cfg, shrink):
    f = 2 ** shrink
    paradigm = "colloc" if cell.startswith("colloc") else "vmc"
    optimizer = "sr" if cell.endswith("sr") else "adam"
    # Warm-start from the common base and run the joint phase (segment loop, NOT --staged) with this
    # cell's optimizer/paradigm. n-seg gives the kernel diagnostics along the way. SR polish is added
    # only for the adam cells (so the SR cells are pure-SR, not SR-on-SR); all get the same total steps.
    cmd = [
        "python3", "-u", str(ROOT / "scripts/run_phase_analysis.py"),
        "--N", str(N), "--omega", str(w), "--arch", "pinn",
        "--backflow", "--backflow-arch", "ctnn",
        "--paradigm", paradigm, "--optimizer", optimizer,
        "--bf-scale-init", "0.7", "--bf-zero-init-last", "0",
        "--steps", str(JOINT_STEPS), "--n-seg", "5", "--polish-steps", "0", "--sr-polish-steps", "0",
        "--batch", str(max(128, cfg["batch"] // f)),
        "--eval-samples", str(max(128, cfg["eval_samples"] // f)),
        "--final-samples", str(max(512, cfg["final_samples"] // f)),
        "--align-samples", str(max(256, cfg["align"] // f)),
        "--seed", str(seed), "--init", str(init), "--outdir", str(cell_dir(N, seed, w, cell)),
    ]
    return cmd


def run_one(cmd, out, gpu, log, what):
    ckpt = out / "checkpoint.pt"
    if ckpt.exists():
        log(f"skip (done): {out.name}")
        return ckpt
    out.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_ATTEMPTS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        logfile = out / "train.log"
        if attempt > 0 and logfile.exists():
            logfile.replace(out / f"train.attempt{attempt}.log")
        c = cmd(attempt)
        with open(logfile, "w") as fh:
            proc = subprocess.Popen(c, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            while proc.poll() is None:
                time.sleep(POLL_SECONDS)
                try:
                    idle = time.time() - logfile.stat().st_mtime
                except FileNotFoundError:
                    idle = 0
                if idle > STALL_SECONDS:
                    log(f"STALL ({idle/60:.0f}m): killing {out.name} attempt {attempt+1}")
                    proc.kill(); break
        if ckpt.exists():
            log(f"done: {out.name} (attempt {attempt+1})")
            return ckpt
        oom = "CUDA out of memory" in logfile.read_text(errors="ignore")[-20000:]
        log(f"FAILED {out.name} attempt {attempt+1}{' (OOM)' if oom else ''}")
    log(f"GIVING UP on {out.name}")
    return None


def run_group(N, seed, w, gpu, log):
    """Build the common base, then run all four cells warm-started from it."""
    cfg = N_CFG[N]
    base = run_one(lambda sh: base_cmd(N, seed, w, cfg, sh), base_dir(N, seed, w), gpu, log, "base")
    if base is None:
        log(f"group N{N} s{seed} {wtag(w)}: BASE failed, skipping its cells")
        return
    for cell in CELLS:
        run_one(lambda sh: cell_cmd(N, seed, w, cell, base, cfg, sh),
                cell_dir(N, seed, w, cell), gpu, log, cell)


def free_gpus() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    free = []
    for line in out.strip().splitlines():
        idx, used, util = (p.strip() for p in line.split(","))
        if int(used) < 1000 and int(util) < 20:
            free.append(int(idx))
    return free


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(msg):
        with lock:
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)

    groups = [(N, s, w) for N in sorted(N_CFG) for s in SEEDS for w in OMEGAS]
    gpus = free_gpus()
    if not gpus:
        log("NO free GPUs; exiting (resume skips completed groups on rerun).")
        return
    log(f"{len(groups)} groups x {len(CELLS)} cells over {len(gpus)} free GPUs {gpus}")

    work = queue.Queue()
    for g in groups:
        work.put(g)

    def worker(gpu):
        while True:
            try:
                N, seed, w = work.get_nowait()
            except queue.Empty:
                return
            try:
                log(f"GPU{gpu} start group N={N} s{seed} {wtag(w)}")
                run_group(N, seed, w, gpu, log)
            except Exception as e:
                log(f"GPU{gpu} group N={N} s{seed} {wtag(w)} EXCEPTION {e!r}")
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("=== PARADIGM/OPTIMIZER CAMPAIGN COMPLETE ===")


if __name__ == "__main__":
    main()
