"""Multi-day N-scaling campaign: explain the THESIS results (PINN Jastrow x backflow at N=6/12/20).

Answers, on the thesis ansatz and only on the thesis ansatz:
  - Does the conventional backflow's rank-1 / tangent-space collapse at Wigner persist at larger N?
    (At N=6 it is stark: BFrank 10.5 -> 1.0 and d_eff 6.4 -> 1.1 as omega goes 1 -> 0.01, while the
    CTNN backflow holds rank ~10. That collapse is the leading candidate mechanism for the thesis's
    CTNN-vs-conventional gap.)
  - Where is the knee? A finer omega grid at N=6 locates where the collapse switches on.
  - Does the gap grow, shrink, or saturate with N?

Robustness (it runs unattended for days):
  - one chain per GPU, no co-tenancy      -> the OOM that killed the last run was a co-tenant
  - OOM retry with halved batch/samples   -> large N is the memory risk
  - stall detection on log mtime          -> kill + retry a chain that stops writing
  - per-chain isolation                   -> one failure never takes down the campaign
  - resume: any stage with a checkpoint is skipped, so a restart continues where it stopped

Run: nohup python3 -u scripts/orchestrate_scaling.py > results/analysis/scaling.log 2>&1 &
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP = "2026-07-16_scaling"
OUTROOT = ROOT / "results/analysis" / STAMP

# Generous on purpose: the segment loop logs only every steps//(2*n_seg) steps (375 at N=20), and a
# large-N step is slow, so a tight threshold would kill healthy chains. This catches true hangs only.
STALL_SECONDS = 90 * 60      # no log write for this long => the chain is wedged
POLL_SECONDS = 120
MAX_ATTEMPTS = 3             # attempt 2 halves the batch, attempt 3 halves again

# Per-N settings. Batch/sample sizes shrink with N: the exact Laplacian is the memory driver.
# OMEGAS ARE THE THESIS'S OWN GRID (config.DMC_ENERGIES / thesis Table 5.2). Only these six values
# have reference energies; config._snap_omega RAISES on anything further than 50% from one of them,
# and silently SNAPS anything nearer (0.3 -> 0.28), which would score a run against the wrong
# reference. Cascades run high -> low omega, warm-starting each stage from the previous.
N_CFG = {
    6:  dict(omegas=[1.0, 0.5, 0.28, 0.1, 0.01, 0.001], steps=3000, batch=2048,
             eval_samples=1024, final_samples=8192, align=2048),
    # batch 512 (was 1024): benchmarked 6.3 s/step vs 12.4 at N=12 CTNN with the exact Laplacian, so
    # 512 halves wall-clock (~1.4 days for a CTNN seed) at 4.2 GB. Mismatched vs the completed conv@1024,
    # but the smaller batch HANDICAPS CTNN, so a CTNN win stays conservative, and BFrank (the rank-collapse
    # headline) is a property of the converged displacement, not of batch size.
    12: dict(omegas=[1.0, 0.5, 0.28, 0.1, 0.01], steps=3000, batch=512,
             eval_samples=512, final_samples=4096, align=1024),
    # batch 256 (was 512): the backflow's B x N^2 edge tensors scale ~2.8x from N=12, so keep peak low;
    # the OOM-retry halves further if needed.
    20: dict(omegas=[1.0, 0.5, 0.28, 0.1], steps=2500, batch=256,
             eval_samples=256, final_samples=2048, align=512),
}
SEEDS = [0, 1]
ARCHS = ["ctnn", "conv"]


def wtag(w: float) -> str:
    return ("w%g" % w).replace(".", "p")


def stage_dir(N: int, arch: str, seed: int, w: float) -> Path:
    return OUTROOT / f"N{N}_{arch}bf_s{seed}_{wtag(w)}"


def build_cmd(N: int, arch: str, seed: int, w: float, init: Path | None,
              staged: bool, cfg: dict, shrink: int) -> list[str]:
    """shrink halves the memory-driving sizes once per retry."""
    f = 2 ** shrink
    cmd = [
        "python3", "-u", str(ROOT / "scripts/run_phase_analysis.py"),
        "--N", str(N), "--omega", str(w), "--arch", "pinn",
        "--backflow", "--backflow-arch", arch, "--paradigm", "vmc", "--optimizer", "adam",
        "--bf-scale-init", "0.7", "--bf-zero-init-last", "0",
        "--steps", str(cfg["steps"]), "--n-seg", "4",
        "--polish-steps", "500", "--sr-polish-steps", "500",
        "--batch", str(max(128, cfg["batch"] // f)),
        "--eval-samples", str(max(128, cfg["eval_samples"] // f)),
        "--final-samples", str(max(512, cfg["final_samples"] // f)),
        "--align-samples", str(max(256, cfg["align"] // f)),
        "--seed", str(seed), "--outdir", str(stage_dir(N, arch, seed, w)),
    ]
    if staged:
        # Curriculum only builds the backflow once, at the first omega of a chain. Cusp is skipped:
        # it MSE-fits Delta_x to a target ~0.02 while a healthy backflow lives at ~0.3, which crushes
        # the displacement (measured: 0.344 -> 0.03) and the run then has to climb back out.
        cmd += ["--staged", "--cusp-steps", "0"]
    if init is not None:
        cmd += ["--init", str(init)]
    return cmd


def run_stage(N, arch, seed, w, init, staged, cfg, gpu, log) -> Path | None:
    out = stage_dir(N, arch, seed, w)
    ckpt = out / "checkpoint.pt"
    if ckpt.exists():
        log(f"skip (done): {out.name}")
        return ckpt
    out.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_ATTEMPTS):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # expandable_segments keeps fragmentation from turning a fit into an OOM on long runs
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logfile = out / "train.log"
        if attempt > 0 and logfile.exists():
            # Keep the failed log: opening "w" on a retry destroys the traceback that says WHY it
            # failed, which is the only evidence of where the memory actually went.
            logfile.replace(out / f"train.attempt{attempt}.log")
        cmd = build_cmd(N, arch, seed, w, init, staged, cfg, shrink=attempt)
        with open(logfile, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            while proc.poll() is None:
                time.sleep(POLL_SECONDS)
                try:
                    idle = time.time() - logfile.stat().st_mtime
                except FileNotFoundError:
                    idle = 0
                if idle > STALL_SECONDS:
                    log(f"STALL ({idle/60:.0f} min idle): killing {out.name} attempt {attempt+1}")
                    proc.kill()
                    break
        if ckpt.exists():
            log(f"done: {out.name} (attempt {attempt+1})")
            return ckpt
        oom = "CUDA out of memory" in logfile.read_text(errors="ignore")[-20000:]
        log(f"FAILED {out.name} attempt {attempt+1}{' (OOM -> halving sizes)' if oom else ''}")
    log(f"GIVING UP on {out.name}")
    return None


def run_chain(N, arch, seed, gpu, log):
    """One omega-cascade on one dedicated GPU. Warm-starts each omega from the previous."""
    cfg = N_CFG[N]
    init, staged = None, True
    for w in cfg["omegas"]:
        ckpt = run_stage(N, arch, seed, w, init, staged, cfg, gpu, log)
        if ckpt is None:
            # Cascade is broken; skip to the next omega from the last good checkpoint rather
            # than abandoning the whole chain.
            log(f"chain N{N} {arch} s{seed}: stage w={w} failed, continuing from previous init")
            continue
        init, staged = ckpt, False


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(msg):
        with lock:
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)

    # Chains ordered by scientific priority: N=6 fine grid locates the knee, then N=12, then N=20.
    chains = ([(6, a, s) for a in ARCHS for s in SEEDS]
              + [(12, a, s) for a in ARCHS for s in SEEDS]
              + [(20, a, s) for a in ARCHS for s in SEEDS])
    # Use only FREE GPUs. Co-tenancy is what OOM'd chains before (a neighbour's 5 GB left too little
    # for the exact Laplacian), so a busy GPU must never be scheduled onto. "Free" = < 1 GB in use.
    def free_gpus() -> list[int]:
        # Both memory AND utilisation must be low. A compute-bound neighbour (e.g. Amber MD) can hold
        # a GPU at 94% util on only ~500 MB, so a memory-only test would wrongly call it free and we
        # would co-tenant onto a saturated GPU.
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        free = []
        for line in out.strip().splitlines():
            idx, used, util = (p.strip() for p in line.split(","))
            if int(used) < 1000 and int(util) < 20:
                free.append(int(idx))
        return free
    gpus = free_gpus()
    if not gpus:
        log("NO free GPUs (all in use by other users). Exiting; rerun when GPUs free up (resume skips "
            "completed stages).")
        return
    log(f"{len(chains)} chains over {len(gpus)} FREE GPUs {gpus} (one chain per GPU; busy GPUs skipped)")

    work = queue.Queue()
    for c in chains:
        work.put(c)

    def worker(gpu):
        while True:
            try:
                N, arch, seed = work.get_nowait()
            except queue.Empty:
                return
            try:
                log(f"GPU{gpu} start N={N} {arch} s{seed}")
                run_chain(N, arch, seed, gpu, log)
            except Exception as e:                      # never let one chain kill the campaign
                log(f"GPU{gpu} chain N={N} {arch} s{seed} EXCEPTION {e!r}")
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("=== TRAINING COMPLETE — running the mechanism analysis ===")
    # Analyse automatically: the campaign runs unattended for days, so it should finish with results
    # rather than a pile of checkpoints. Errors here must not mask a successful training run.
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with open(OUTROOT / "ANALYSIS.log", "w") as fh:
            subprocess.run(["python3", "-u", str(ROOT / "scripts/analyze_pinn_ansatz.py"),
                            "--camp", str(OUTROOT)],
                           stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT), timeout=6 * 3600)
        log(f"analysis written to {OUTROOT/'ANALYSIS.log'} and master.csv")
    except Exception as e:
        log(f"analysis FAILED ({e!r}) — checkpoints are intact, rerun analyze_pinn_ansatz.py --camp {OUTROOT}")
    log("=== SCALING CAMPAIGN COMPLETE ===")


if __name__ == "__main__":
    main()
