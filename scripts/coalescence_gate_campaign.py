#!/usr/bin/env python3
"""Coalescence-focused BF campaign.

2x2x3 matrix:
- hard cusp gate: off/on
- resample regularization: off/on
- seeds: 11, 22, 42
Target: N=6, omega=0.01
"""

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "src" / "run_collocation.py"
RESULTS = ROOT / "results" / "arch_colloc"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def finite_or_none(v):
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    return fv if fv == fv and abs(fv) != float("inf") else None


def _load_ckpt(path: Path):
    if not path.exists():
        return None
    try:
        return torch.load(str(path), map_location="cpu")
    except Exception:
        return None


def _hist_mean(hist, key):
    vals = [finite_or_none(h.get(key)) for h in hist]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(mean(vals))


def _hist_median(hist, key):
    vals = sorted([finite_or_none(h.get(key)) for h in hist if finite_or_none(h.get(key)) is not None])
    if not vals:
        return None
    n = len(vals)
    if n % 2 == 1:
        return float(vals[n // 2])
    return float(0.5 * (vals[n // 2 - 1] + vals[n // 2]))


def _hist_best_probe_err(hist):
    vals = [finite_or_none(h.get("vmc_err")) for h in hist]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(min(vals) * 100.0)


def _hist_best_sel_err(hist):
    vals = [finite_or_none(h.get("vmc_sel_err")) for h in hist]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(min(vals) * 100.0)


def _parse_rollbacks(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return txt.count("rollback:")


def _job_tag(job):
    g = "g1" if job["gate"] else "g0"
    r = "r1" if job["reg"] else "r0"
    om = str(job["omega"]).replace(".", "p")
    return "coal_{}_{}_n{}_o{}_s{}".format(g, r, job["n_elec"], om, job["seed"])


def make_jobs():
    seeds = [11, 22, 42]
    variants = [(False, False), (False, True), (True, False), (True, True)]
    gpus = [0, 4, 5, 7]
    jobs = []
    i = 0
    for gate, reg in variants:
        for s in seeds:
            jobs.append({
                "gpu": gpus[i % len(gpus)],
                "gate": bool(gate),
                "reg": bool(reg),
                "seed": int(s),
                "epochs": 700,
                "n_elec": 6,
                "omega": 0.01,
            })
            i += 1
    return jobs


def _cmd(job):
    init_jas = str(RESULTS / "n6_o0p01_jas_transfer_s42.pt")
    init_bf = str(RESULTS / "bf_ctnn_vcycle.pt")
    tag = _job_tag(job)

    cmd = [
        "python3", str(RUN),
        "--tag", tag,
        "--mode", "bf",
        "--seed", str(job["seed"]),
        "--n-elec", str(job["n_elec"]),
        "--omega", str(job["omega"]),
        "--epochs", str(job["epochs"]),
        "--n-coll", "4096",
        "--oversample", "8",
        "--micro-batch", "512",
        "--vmc-every", "40",
        "--vmc-n", "10000",
        "--vmc-select-n", "15000",
        "--n-eval", "20000",
        "--lr", "8e-4",
        "--lr-jas", "8e-5",
        "--direct-weight", "0.0",
        "--clip-el", "5.0",
        "--reward-qtrim", "0.02",
        "--no-pretrained",
        "--init-jas", init_jas,
        "--init-bf", init_bf,
        "--ess-floor-ratio", "0.12",
        "--ess-oversample-max", "14",
        "--ess-oversample-step", "2",
        "--ess-resample-tries", "2",
        "--rollback-decay", "0.8",
        "--rollback-err-pct", "20.0",
        "--rollback-jump-sigma", "8.0",
        "--replay-frac", "0.0",
        "--bf-cusp-reg", "0.0",
        "--bf-diag-q", "0.10",
    ]

    if job["reg"]:
        cmd += ["--resample-weight-temp", "0.70", "--resample-logw-clip-q", "0.995"]
    else:
        cmd += ["--resample-weight-temp", "1.0", "--resample-logw-clip-q", "0.0"]

    if job["gate"]:
        cmd += ["--bf-hard-cusp-gate", "--bf-cusp-gate-radius-aho", "0.30", "--bf-cusp-gate-power", "2.0"]

    return cmd


def run_job(job, out_dir):
    env = os.environ.copy()
    env["CUDA_MANUAL_DEVICE"] = str(job["gpu"])
    tag = _job_tag(job)
    log_path = out_dir / "logs" / (tag + ".log")
    ckpt_path = RESULTS / (tag + ".pt")
    cmd = _cmd(job)

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[{}] START {} gpu={}\n".format(now(), tag, job["gpu"]))
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write("\n[{}] END rc={}\n".format(now(), proc.returncode))
    dt = time.time() - t0

    final_E = final_err = None
    probe_err = sel_err = gap = None
    ess_med = ess_raw_med = top1 = None
    coal_ratio = None

    ck = _load_ckpt(ckpt_path)
    if ck is not None:
        final_E = finite_or_none(ck.get("E"))
        final_err = finite_or_none(ck.get("err"))
        hist = ck.get("hist", []) or []
        if hist:
            probe_err = _hist_best_probe_err(hist)
            sel_err = _hist_best_sel_err(hist)
            if final_err is not None and probe_err is not None:
                gap = float(final_err - probe_err)
            ess_med = _hist_median(hist, "ess")
            ess_raw_med = _hist_median(hist, "ess_raw")
            top1 = _hist_mean(hist, "rs_top1_mass")
            coal_ratio = _hist_mean(hist, "bf_coal_ratio_q")

    return {
        "tag": tag,
        "gpu": job["gpu"],
        "gate": job["gate"],
        "reg": job["reg"],
        "seed": job["seed"],
        "returncode": int(proc.returncode),
        "elapsed_s": dt,
        "log_path": str(log_path),
        "ckpt_path": str(ckpt_path),
        "final_E": final_E,
        "final_err_pct": final_err,
        "best_probe_err_pct": probe_err,
        "best_sel_err_pct": sel_err,
        "probe_final_gap_pct": gap,
        "ess_median": ess_med,
        "ess_raw_median": ess_raw_med,
        "rs_top1_mass_mean": top1,
        "rollback_count": _parse_rollbacks(log_path),
        "bf_coal_ratio_q_mean": coal_ratio,
    }


def summarize(results):
    rows = list(results)
    groups = {}
    for r in results:
        key = "gate={}|reg={}".format(int(r["gate"]), int(r["reg"]))
        groups.setdefault(key, []).append(r)

    stats = {}
    for key, rs in groups.items():
        ok = [x for x in rs if x["returncode"] == 0 and x["final_err_pct"] is not None]
        stats[key] = {
            "n": len(rs),
            "n_ok": len(ok),
            "mean_final_err_pct": float(mean([x["final_err_pct"] for x in ok])) if ok else None,
            "mean_probe_gap_pct": float(mean([x["probe_final_gap_pct"] for x in ok if x["probe_final_gap_pct"] is not None])) if ok else None,
            "mean_ess_median": float(mean([x["ess_median"] for x in ok if x["ess_median"] is not None])) if ok else None,
            "mean_ess_raw_median": float(mean([x["ess_raw_median"] for x in ok if x["ess_raw_median"] is not None])) if ok else None,
            "mean_top1_mass": float(mean([x["rs_top1_mass_mean"] for x in ok if x["rs_top1_mass_mean"] is not None])) if ok else None,
            "mean_bf_coal_ratio_q": float(mean([x["bf_coal_ratio_q_mean"] for x in ok if x["bf_coal_ratio_q_mean"] is not None])) if ok else None,
            "mean_rollbacks": float(mean([x["rollback_count"] for x in ok])) if ok else None,
        }

    return {"created": now(), "n_results": len(results), "group_stats": stats, "results": rows}


def worker(gpu, jobs, out_dir, q):
    for j in jobs:
        q.put(run_job(j, out_dir))


def main():
    ap = argparse.ArgumentParser(description="Coalescence hard-gate campaign")
    ap.add_argument("--name", type=str, default="coalescence_gate_campaign")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this campaign. Refusing CPU run.")

    jobs = make_jobs()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = ROOT / "outputs" / f"{stamp}_{args.name}"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    by_gpu = {}
    for j in jobs:
        by_gpu.setdefault(j["gpu"], []).append(j)

    plan = {"created": now(), "jobs": jobs}
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Coalescence campaign start: {now()}")
    print(f"Output: {out_dir}")
    for gpu, gj in sorted(by_gpu.items()):
        print(f"  GPU {gpu}: {len(gj)} job(s)")
        for j in gj:
            print("    - {}".format(_job_tag(j)))
    print("=" * 72)

    q = Queue()
    ths = []
    for gpu, gj in sorted(by_gpu.items()):
        th = threading.Thread(target=worker, args=(gpu, gj, out_dir, q), daemon=False)
        th.start()
        ths.append(th)

    results = []
    remaining = len(jobs)
    while remaining > 0:
        try:
            rr = q.get(timeout=8.0)
        except Empty:
            if any(t.is_alive() for t in ths):
                continue
            break
        results.append(rr)
        remaining -= 1
        with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rr) + "\n")
        print("[{}] DONE {} rc={} err={} ess={} top1={}".format(
            now(), rr["tag"], rr["returncode"], rr["final_err_pct"], rr["ess_median"], rr["rs_top1_mass_mean"]
        ))

    for t in ths:
        t.join()

    (out_dir / "summary.json").write_text(json.dumps(summarize(results), indent=2), encoding="utf-8")
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
