#!/usr/bin/env python3
"""Targeted regime-aware trainer campaign.

Purpose:
- Run a small, intentional 6-job matrix derived from the latest trainer-stability discussion
- Compare regime-specific trainer policies rather than one universal policy
- Keep the runtime in the "hours, not overnight" range
- Emit structured summaries with probe/final gaps, ESS stats, rollback counts, and replay diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from statistics import mean, median

import torch


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "src" / "run_collocation.py"
RESULTS = ROOT / "results" / "arch_colloc"


@dataclass
class Job:
    gpu: int
    family: str
    policy: str
    mode: str
    n_elec: int
    omega: float
    seed: int
    epochs: int
    n_coll: int
    oversample: int
    micro_batch: int
    vmc_every: int
    vmc_n: int
    n_eval: int
    lr: float
    lr_jas: float
    direct_weight: float
    clip_el: float
    reward_qtrim: float
    no_pretrained: bool = True
    init_jas: str | None = None
    init_bf: str | None = None
    ess_floor_ratio: float = 0.0
    ess_oversample_max: int = 0
    ess_oversample_step: int = 2
    ess_resample_tries: int = 1
    rollback_decay: float = 1.0
    rollback_err_pct: float = 0.0
    rollback_jump_sigma: float = 0.0
    replay_frac: float = 0.0
    replay_top_frac: float = 0.25
    replay_stratified: bool = False
    replay_geo_bins: int = 3
    bf_cusp_reg: float = 0.0
    bf_cusp_radius_aho: float = 0.30

    @property
    def tag(self) -> str:
        om = str(self.omega).replace(".", "p")
        return f"regime_{self.family}_{self.policy}_n{self.n_elec}_o{om}_s{self.seed}"


@dataclass
class JobResult:
    tag: str
    gpu: int
    family: str
    policy: str
    mode: str
    n_elec: int
    omega: float
    seed: int
    returncode: int
    elapsed_s: float
    log_path: str
    ckpt_path: str
    final_E: float | None
    final_se: float | None
    final_err_pct: float | None
    best_probe_E: float | None
    best_probe_err_pct: float | None
    probe_final_gap_pct: float | None
    ess_min: float | None
    ess_median: float | None
    ess_target_median: float | None
    oversample_max: int | None
    rollback_count: int
    replay_bucket_entropy_mean: float | None
    replay_min_pair_mean: float | None
    last_epoch: int | None


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def finite_or_none(v):
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    return fv if math.isfinite(fv) else None


def _load_ckpt_metrics(ckpt_path: Path):
    if not ckpt_path.exists():
        return None
    try:
        return torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        return None


def _hist_probe_stats(hist):
    probes = [h for h in hist if "vmc_E" in h and "vmc_err" in h]
    if not probes:
        return None, None
    best = min(
        probes,
        key=lambda h: abs(float(h.get("vmc_err", float("inf"))))
        if finite_or_none(h.get("vmc_err")) is not None
        else float("inf"),
    )
    return finite_or_none(best.get("vmc_E")), finite_or_none(best.get("vmc_err"))


def _hist_scalar_stats(hist, key):
    vals = [finite_or_none(h.get(key)) for h in hist]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return min(vals), median(vals)


def _hist_mean(hist, key):
    vals = [finite_or_none(h.get(key)) for h in hist]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return mean(vals)


def _parse_rollbacks(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"rollback:", txt))


def policy_args(job: Job) -> list[str]:
    args = [
        "--mode", job.mode,
        "--lr", str(job.lr),
        "--lr-jas", str(job.lr_jas),
        "--direct-weight", str(job.direct_weight),
        "--clip-el", str(job.clip_el),
        "--reward-qtrim", str(job.reward_qtrim),
    ]
    if job.no_pretrained:
        args.append("--no-pretrained")
    if job.init_jas:
        args += ["--init-jas", job.init_jas]
    if job.init_bf:
        args += ["--init-bf", job.init_bf]
    if job.ess_floor_ratio > 0:
        args += [
            "--ess-floor-ratio", str(job.ess_floor_ratio),
            "--ess-oversample-max", str(job.ess_oversample_max),
            "--ess-oversample-step", str(job.ess_oversample_step),
            "--ess-resample-tries", str(job.ess_resample_tries),
        ]
    if job.rollback_decay < 1.0 or job.rollback_err_pct > 0 or job.rollback_jump_sigma > 0:
        args += [
            "--rollback-decay", str(job.rollback_decay),
            "--rollback-err-pct", str(job.rollback_err_pct),
            "--rollback-jump-sigma", str(job.rollback_jump_sigma),
        ]
    if job.replay_frac > 0:
        args += [
            "--replay-frac", str(job.replay_frac),
            "--replay-top-frac", str(job.replay_top_frac),
        ]
        if job.replay_stratified:
            args += ["--replay-stratified", "--replay-geo-bins", str(job.replay_geo_bins)]
    if job.bf_cusp_reg > 0:
        args += [
            "--bf-cusp-reg", str(job.bf_cusp_reg),
            "--bf-cusp-radius-aho", str(job.bf_cusp_radius_aho),
        ]
    return args


def make_jobs() -> list[Job]:
    bf_ckpt = str(RESULTS / "bf_ctnn_vcycle.pt")
    jobs = [
        # Policy H: moderate/high omega, full stability stack.
        Job(
            gpu=0,
            family="high",
            policy="H",
            mode="bf",
            n_elec=6,
            omega=1.0,
            seed=11,
            epochs=1200,
            n_coll=4096,
            oversample=8,
            micro_batch=512,
            vmc_every=40,
            vmc_n=10000,
            n_eval=20000,
            lr=5e-4,
            lr_jas=5e-5,
            direct_weight=0.0,
            clip_el=5.0,
            reward_qtrim=0.02,
            init_jas=bf_ckpt,
            init_bf=bf_ckpt,
            ess_floor_ratio=0.15,
            ess_oversample_max=16,
            ess_resample_tries=3,
            rollback_decay=0.8,
            rollback_err_pct=25.0,
            rollback_jump_sigma=8.0,
            replay_frac=0.25,
            replay_top_frac=0.25,
            replay_stratified=True,
            replay_geo_bins=3,
            bf_cusp_reg=1e-5,
        ),
        Job(
            gpu=5,
            family="high",
            policy="H",
            mode="bf",
            n_elec=12,
            omega=0.5,
            seed=11,
            epochs=900,
            n_coll=4096,
            oversample=8,
            micro_batch=512,
            vmc_every=40,
            vmc_n=10000,
            n_eval=20000,
            lr=5e-4,
            lr_jas=5e-5,
            direct_weight=0.0,
            clip_el=5.0,
            reward_qtrim=0.02,
            init_jas=str(RESULTS / "n12_o0p5_jas_s11.pt"),
            init_bf=bf_ckpt,
            ess_floor_ratio=0.15,
            ess_oversample_max=16,
            ess_resample_tries=3,
            rollback_decay=0.8,
            rollback_err_pct=25.0,
            rollback_jump_sigma=8.0,
            replay_frac=0.25,
            replay_top_frac=0.25,
            replay_stratified=True,
            replay_geo_bins=3,
            bf_cusp_reg=1e-5,
        ),
        Job(
            gpu=4,
            family="high",
            policy="H",
            mode="jastrow",
            n_elec=20,
            omega=1.0,
            seed=11,
            epochs=320,
            n_coll=4096,
            oversample=8,
            micro_batch=512,
            vmc_every=40,
            vmc_n=10000,
            n_eval=20000,
            lr=5e-4,
            lr_jas=3e-4,
            direct_weight=0.1,
            clip_el=0.0,
            reward_qtrim=0.02,
            ess_floor_ratio=0.10,
            ess_oversample_max=16,
            ess_resample_tries=3,
            rollback_decay=0.8,
            rollback_err_pct=30.0,
            rollback_jump_sigma=8.0,
            replay_frac=0.20,
            replay_top_frac=0.25,
            replay_stratified=True,
            replay_geo_bins=3,
        ),
        # Policy L: low-omega, faster and lighter-touch.
        Job(
            gpu=7,
            family="low",
            policy="L",
            mode="bf",
            n_elec=6,
            omega=0.01,
            seed=11,
            epochs=1200,
            n_coll=4096,
            oversample=8,
            micro_batch=512,
            vmc_every=40,
            vmc_n=10000,
            n_eval=20000,
            lr=8e-4,
            lr_jas=8e-5,
            direct_weight=0.0,
            clip_el=5.0,
            reward_qtrim=0.02,
            init_jas=str(RESULTS / "n6_o0p01_jas_transfer_s42.pt"),
            init_bf=bf_ckpt,
            ess_floor_ratio=0.12,
            ess_oversample_max=14,
            ess_resample_tries=2,
            rollback_decay=0.8,
            rollback_err_pct=20.0,
            rollback_jump_sigma=8.0,
        ),
        Job(
            gpu=7,
            family="low",
            policy="L",
            mode="bf",
            n_elec=6,
            omega=0.001,
            seed=11,
            epochs=1200,
            n_coll=4096,
            oversample=8,
            micro_batch=512,
            vmc_every=40,
            vmc_n=10000,
            n_eval=20000,
            lr=8e-4,
            lr_jas=8e-5,
            direct_weight=0.0,
            clip_el=5.0,
            reward_qtrim=0.02,
            init_jas=str(RESULTS / "n6_o0p001_jas_transfer_s42.pt"),
            init_bf=bf_ckpt,
            ess_floor_ratio=0.12,
            ess_oversample_max=14,
            ess_resample_tries=2,
            rollback_decay=0.8,
            rollback_err_pct=20.0,
            rollback_jump_sigma=8.0,
        ),
        Job(
            gpu=0,
            family="low",
            policy="L",
            mode="jastrow",
            n_elec=20,
            omega=0.1,
            seed=11,
            epochs=320,
            n_coll=4096,
            oversample=8,
            micro_batch=512,
            vmc_every=40,
            vmc_n=10000,
            n_eval=20000,
            lr=5e-4,
            lr_jas=5e-4,
            direct_weight=0.1,
            clip_el=0.0,
            reward_qtrim=0.02,
            ess_floor_ratio=0.10,
            ess_oversample_max=14,
            ess_resample_tries=2,
            rollback_decay=0.8,
            rollback_err_pct=25.0,
            rollback_jump_sigma=8.0,
        ),
    ]
    for job in jobs:
        if job.init_jas and not Path(job.init_jas).exists():
            raise FileNotFoundError(f"Missing init_jas for {job.tag}: {job.init_jas}")
        if job.init_bf and not Path(job.init_bf).exists():
            raise FileNotFoundError(f"Missing init_bf for {job.tag}: {job.init_bf}")
    return jobs


def job_command(job: Job) -> list[str]:
    cmd = [
        "python3", str(RUN),
        "--tag", job.tag,
        "--seed", str(job.seed),
        "--n-elec", str(job.n_elec),
        "--omega", str(job.omega),
        "--epochs", str(job.epochs),
        "--n-coll", str(job.n_coll),
        "--oversample", str(job.oversample),
        "--micro-batch", str(job.micro_batch),
        "--vmc-every", str(job.vmc_every),
        "--vmc-n", str(job.vmc_n),
        "--n-eval", str(job.n_eval),
    ]
    cmd.extend(policy_args(job))
    return cmd


def run_job(job: Job, out_dir: Path) -> JobResult:
    env = os.environ.copy()
    env["CUDA_MANUAL_DEVICE"] = str(job.gpu)
    log_path = out_dir / "logs" / f"{job.tag}.log"
    ckpt_path = RESULTS / f"{job.tag}.pt"
    cmd = job_command(job)

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[{now()}] START {job.tag} on GPU {job.gpu}\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n[{now()}] END rc={proc.returncode}\n")
    dt = time.time() - t0

    final_E = final_se = final_err = None
    best_probe_E = best_probe_err = None
    probe_final_gap = None
    ess_min = ess_median = ess_target_median = None
    oversample_max = None
    replay_bucket_entropy_mean = None
    replay_min_pair_mean = None
    last_epoch = None
    ckpt = _load_ckpt_metrics(ckpt_path)
    if ckpt is not None:
        final_E = finite_or_none(ckpt.get("E"))
        final_se = finite_or_none(ckpt.get("se"))
        final_err = finite_or_none(ckpt.get("err"))
        hist = ckpt.get("hist", []) or []
        if hist:
            last_epoch = int(hist[-1].get("ep")) if hist[-1].get("ep") is not None else None
            best_probe_E, best_probe_err = _hist_probe_stats(hist)
            if final_err is not None and best_probe_err is not None:
                probe_final_gap = float(final_err - 100.0 * best_probe_err)
            ess_min, ess_median = _hist_scalar_stats(hist, "ess")
            _, ess_target_median = _hist_scalar_stats(hist, "ess_target")
            overs = [int(h.get("oversample")) for h in hist if h.get("oversample") is not None]
            oversample_max = max(overs) if overs else None
            replay_bucket_entropy_mean = _hist_mean(hist, "replay_bucket_entropy")
            replay_min_pair_mean = _hist_mean(hist, "replay_min_pair_mean")

    return JobResult(
        tag=job.tag,
        gpu=job.gpu,
        family=job.family,
        policy=job.policy,
        mode=job.mode,
        n_elec=job.n_elec,
        omega=job.omega,
        seed=job.seed,
        returncode=int(proc.returncode),
        elapsed_s=dt,
        log_path=str(log_path),
        ckpt_path=str(ckpt_path),
        final_E=final_E,
        final_se=final_se,
        final_err_pct=final_err,
        best_probe_E=best_probe_E,
        best_probe_err_pct=(100.0 * best_probe_err) if best_probe_err is not None else None,
        probe_final_gap_pct=probe_final_gap,
        ess_min=ess_min,
        ess_median=ess_median,
        ess_target_median=ess_target_median,
        oversample_max=oversample_max,
        rollback_count=_parse_rollbacks(log_path),
        replay_bucket_entropy_mean=replay_bucket_entropy_mean,
        replay_min_pair_mean=replay_min_pair_mean,
        last_epoch=last_epoch,
    )


def summarize(results: list[JobResult]) -> dict:
    out: dict[str, dict] = {}
    for res in results:
        out[res.tag] = asdict(res)

    def _pick_best(rs: list[JobResult]):
        ok = [r for r in rs if r.returncode == 0 and r.final_err_pct is not None]
        if not ok:
            return None
        best = min(ok, key=lambda r: abs(float(r.final_err_pct)))
        return asdict(best)

    grouped: dict[str, list[JobResult]] = {}
    for res in results:
        grouped.setdefault(f"N={res.n_elec}|w={res.omega}", []).append(res)

    by_target = {key: _pick_best(rs) for key, rs in grouped.items()}
    return {
        "created": now(),
        "n_results": len(results),
        "by_target_best": by_target,
        "results": out,
    }


def worker(gpu: int, jobs: list[Job], out_dir: Path, results_q: Queue) -> None:
    for job in jobs:
        results_q.put(run_job(job, out_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run regime-aware trainer campaign")
    ap.add_argument("--name", type=str, default=None, help="Optional output folder suffix")
    args = ap.parse_args()

    jobs = make_jobs()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    suffix = args.name or "regime_policy_campaign"
    out_dir = ROOT / "outputs" / f"{stamp}_{suffix}"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(
        json.dumps({"created": now(), "jobs": [asdict(j) for j in jobs]}, indent=2),
        encoding="utf-8",
    )

    by_gpu: dict[int, list[Job]] = {}
    for job in jobs:
        by_gpu.setdefault(job.gpu, []).append(job)

    print("=" * 72)
    print(f"Regime campaign start: {now()}")
    print(f"Output: {out_dir}")
    for gpu, gpu_jobs in sorted(by_gpu.items()):
        print(f"  GPU {gpu}: {len(gpu_jobs)} job(s)")
        for job in gpu_jobs:
            print(f"    - {job.tag}")
    print("=" * 72)

    results_q: Queue = Queue()
    threads: list[threading.Thread] = []
    for gpu, gpu_jobs in sorted(by_gpu.items()):
        th = threading.Thread(target=worker, args=(gpu, gpu_jobs, out_dir, results_q), daemon=False)
        th.start()
        threads.append(th)

    results: list[JobResult] = []
    remaining = len(jobs)
    while remaining > 0:
        try:
            rr = results_q.get(timeout=5.0)
        except Empty:
            if any(th.is_alive() for th in threads):
                continue
            break
        results.append(rr)
        remaining -= 1
        with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rr)) + "\n")
        print(
            f"[{now()}] DONE {rr.tag} rc={rr.returncode} "
            f"E={rr.final_E} err={rr.final_err_pct} probe_gap={rr.probe_final_gap_pct}"
        )

    for th in threads:
        th.join()

    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()