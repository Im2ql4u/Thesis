#!/usr/bin/env python3
"""Stability-first campaign focused on seed robustness and tail-control ablations.

Design goals:
- Validate seed divergence after seed-path fix
- Test whether resample regularization improves final heavy VMC behavior
- Keep jobs short enough for iterative debugging while still meaningful
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
    variant: str
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
    vmc_select_n: int
    n_eval: int
    lr: float
    lr_jas: float
    direct_weight: float
    clip_el: float
    reward_qtrim: float
    init_jas: str | None = None
    init_bf: str | None = None
    no_pretrained: bool = True
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
    resample_weight_temp: float = 1.0
    resample_logw_clip_q: float = 0.0

    @property
    def tag(self) -> str:
        om = str(self.omega).replace(".", "p")
        return f"stab_{self.variant}_n{self.n_elec}_o{om}_s{self.seed}"


@dataclass
class JobResult:
    tag: str
    gpu: int
    variant: str
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
    best_sel_E: float | None
    best_sel_err_pct: float | None
    probe_final_gap_pct: float | None
    ess_min: float | None
    ess_median: float | None
    ess_raw_median: float | None
    rs_top1_mass_mean: float | None
    rs_top10_mass_mean: float | None
    rollback_count: int
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


def _hist_sel_stats(hist):
    sels = [h for h in hist if "vmc_sel_E" in h and "vmc_sel_err" in h]
    if not sels:
        return None, None
    best = min(
        sels,
        key=lambda h: abs(float(h.get("vmc_sel_err", float("inf"))))
        if finite_or_none(h.get("vmc_sel_err")) is not None
        else float("inf"),
    )
    return finite_or_none(best.get("vmc_sel_E")), finite_or_none(best.get("vmc_sel_err"))


def _parse_rollbacks(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"rollback:", txt))


def _policy_args(job: Job) -> list[str]:
    args = [
        "--mode", job.mode,
        "--lr", str(job.lr),
        "--lr-jas", str(job.lr_jas),
        "--direct-weight", str(job.direct_weight),
        "--clip-el", str(job.clip_el),
        "--reward-qtrim", str(job.reward_qtrim),
        "--resample-weight-temp", str(job.resample_weight_temp),
        "--resample-logw-clip-q", str(job.resample_logw_clip_q),
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


def _job_command(job: Job) -> list[str]:
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
        "--vmc-select-n", str(job.vmc_select_n),
        "--n-eval", str(job.n_eval),
    ]
    cmd.extend(_policy_args(job))
    return cmd


def make_jobs() -> list[Job]:
    bf_ckpt = str(RESULTS / "bf_ctnn_vcycle.pt")
    init_hi = str(RESULTS / "bf_ctnn_vcycle.pt")
    init_lo = str(RESULTS / "n6_o0p01_jas_transfer_s42.pt")

    base = dict(
        mode="bf",
        epochs=700,
        n_coll=4096,
        oversample=8,
        micro_batch=512,
        vmc_every=40,
        vmc_n=10000,
        vmc_select_n=0,
        n_eval=20000,
        lr=5e-4,
        lr_jas=5e-5,
        direct_weight=0.0,
        clip_el=5.0,
        reward_qtrim=0.02,
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
        init_bf=bf_ckpt,
    )

    def mk(**overrides) -> Job:
        cfg = dict(base)
        cfg.update(overrides)
        return Job(**cfg)

    jobs = [
        # High-omega seed robustness baseline vs regularized.
        mk(gpu=0, variant="base_hi", n_elec=6, omega=1.0, seed=11, init_jas=init_hi),
        mk(gpu=4, variant="base_hi", n_elec=6, omega=1.0, seed=22, init_jas=init_hi),
        mk(
            gpu=5,
            variant="reg_hi",
            n_elec=6,
            omega=1.0,
            seed=11,
            init_jas=init_hi,
            vmc_select_n=15000,
            resample_weight_temp=0.75,
            resample_logw_clip_q=0.995,
        ),
        mk(
            gpu=7,
            variant="reg_hi",
            n_elec=6,
            omega=1.0,
            seed=22,
            init_jas=init_hi,
            vmc_select_n=15000,
            resample_weight_temp=0.75,
            resample_logw_clip_q=0.995,
        ),
        # Low-omega stress with and without resample regularization.
        mk(
            gpu=0,
            variant="base_lo",
            n_elec=6,
            omega=0.01,
            seed=11,
            init_jas=init_lo,
            lr=8e-4,
            lr_jas=8e-5,
            ess_floor_ratio=0.12,
            ess_oversample_max=14,
            ess_resample_tries=2,
            replay_frac=0.0,
            bf_cusp_reg=0.0,
        ),
        mk(
            gpu=7,
            variant="reg_lo",
            n_elec=6,
            omega=0.01,
            seed=11,
            init_jas=init_lo,
            lr=8e-4,
            lr_jas=8e-5,
            ess_floor_ratio=0.12,
            ess_oversample_max=14,
            ess_resample_tries=2,
            replay_frac=0.0,
            bf_cusp_reg=0.0,
            vmc_select_n=15000,
            resample_weight_temp=0.70,
            resample_logw_clip_q=0.995,
        ),
    ]

    for job in jobs:
        if job.init_jas and not Path(job.init_jas).exists():
            raise FileNotFoundError(f"Missing init_jas for {job.tag}: {job.init_jas}")
        if job.init_bf and not Path(job.init_bf).exists():
            raise FileNotFoundError(f"Missing init_bf for {job.tag}: {job.init_bf}")
    return jobs


def run_job(job: Job, out_dir: Path) -> JobResult:
    env = os.environ.copy()
    env["CUDA_MANUAL_DEVICE"] = str(job.gpu)
    log_path = out_dir / "logs" / f"{job.tag}.log"
    ckpt_path = RESULTS / f"{job.tag}.pt"
    cmd = _job_command(job)

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
    best_sel_E = best_sel_err = None
    probe_final_gap = None
    ess_min = ess_median = None
    ess_raw_median = None
    rs_top1_mass_mean = None
    rs_top10_mass_mean = None
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
            best_sel_E, best_sel_err = _hist_sel_stats(hist)
            if final_err is not None and best_probe_err is not None:
                probe_final_gap = float(final_err - 100.0 * best_probe_err)
            ess_min, ess_median = _hist_scalar_stats(hist, "ess")
            _, ess_raw_median = _hist_scalar_stats(hist, "ess_raw")
            rs_top1_mass_mean = _hist_mean(hist, "rs_top1_mass")
            rs_top10_mass_mean = _hist_mean(hist, "rs_top10_mass")

    return JobResult(
        tag=job.tag,
        gpu=job.gpu,
        variant=job.variant,
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
        best_sel_E=best_sel_E,
        best_sel_err_pct=(100.0 * best_sel_err) if best_sel_err is not None else None,
        probe_final_gap_pct=probe_final_gap,
        ess_min=ess_min,
        ess_median=ess_median,
        ess_raw_median=ess_raw_median,
        rs_top1_mass_mean=rs_top1_mass_mean,
        rs_top10_mass_mean=rs_top10_mass_mean,
        rollback_count=_parse_rollbacks(log_path),
        last_epoch=last_epoch,
    )


def summarize(results: list[JobResult]) -> dict:
    out = {r.tag: asdict(r) for r in results}

    by_variant = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    variant_stats = {}
    for key, rs in by_variant.items():
        ok = [x for x in rs if x.returncode == 0 and x.final_err_pct is not None]
        variant_stats[key] = {
            "n": len(rs),
            "n_ok": len(ok),
            "mean_final_err_pct": float(mean([x.final_err_pct for x in ok])) if ok else None,
            "mean_probe_gap_pct": float(mean([x.probe_final_gap_pct for x in ok if x.probe_final_gap_pct is not None])) if ok else None,
            "mean_ess_median": float(mean([x.ess_median for x in ok if x.ess_median is not None])) if ok else None,
            "mean_rs_top1_mass": float(mean([x.rs_top1_mass_mean for x in ok if x.rs_top1_mass_mean is not None])) if ok else None,
        }

    return {
        "created": now(),
        "n_results": len(results),
        "variant_stats": variant_stats,
        "results": out,
    }


def worker(gpu: int, jobs: list[Job], out_dir: Path, results_q: Queue) -> None:
    for job in jobs:
        results_q.put(run_job(job, out_dir))


def estimate_runtime_hours(jobs: list[Job], by_gpu: dict[int, list[Job]]) -> dict:
    # Conservative rough model from recent measured N=6 BF timings in this repo.
    # 700-epoch N=6 BF ~= 30-45 min/job depending on omega and controls.
    per_job_min = []
    for j in jobs:
        base = 0.055 * j.epochs  # ~38.5 min for 700 epochs
        if j.omega <= 0.01:
            base *= 1.25
        if j.vmc_select_n > 0:
            base *= 1.20
        per_job_min.append(base)

    lane_totals = {}
    for gpu, gpu_jobs in by_gpu.items():
        lane = 0.0
        for gj in gpu_jobs:
            idx = jobs.index(gj)
            lane += per_job_min[idx]
        lane_totals[gpu] = lane

    wall_min = max(lane_totals.values()) if lane_totals else 0.0
    return {
        "estimated_wall_minutes": wall_min,
        "estimated_wall_hours": wall_min / 60.0,
        "per_gpu_minutes": lane_totals,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run stability-first seed/tail campaign")
    ap.add_argument("--name", type=str, default="stability_core_campaign")
    args = ap.parse_args()

    jobs = make_jobs()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = ROOT / "outputs" / f"{stamp}_{args.name}"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    by_gpu = {}
    for job in jobs:
        by_gpu.setdefault(job.gpu, []).append(job)

    eta = estimate_runtime_hours(jobs, by_gpu)
    plan = {
        "created": now(),
        "eta": eta,
        "jobs": [asdict(j) for j in jobs],
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Stability campaign start: {now()}")
    print(f"Output: {out_dir}")
    print(f"ETA: ~{eta['estimated_wall_hours']:.2f}h ({eta['estimated_wall_minutes']:.0f} min)")
    for gpu, gpu_jobs in sorted(by_gpu.items()):
        print(f"  GPU {gpu}: {len(gpu_jobs)} job(s), est {eta['per_gpu_minutes'][gpu]:.0f} min")
        for job in gpu_jobs:
            print(f"    - {job.tag}")
    print("=" * 72)

    results_q = Queue()
    threads = []
    for gpu, gpu_jobs in sorted(by_gpu.items()):
        th = threading.Thread(target=worker, args=(gpu, gpu_jobs, out_dir, results_q), daemon=False)
        th.start()
        threads.append(th)

    results = []
    remaining = len(jobs)
    while remaining > 0:
        try:
            rr = results_q.get(timeout=10.0)
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
            f"E={rr.final_E} err={rr.final_err_pct} "
            f"probe_gap={rr.probe_final_gap_pct} ess_med={rr.ess_median}"
        )

    for th in threads:
        th.join()

    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
