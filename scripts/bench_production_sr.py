"""Time the production SR-VMC step (sr_step_energy_mb) phase by phase, without editing it.

The helpers the step calls (_persistent_rw, _local_energy_multi, _score_rows) are looked up as
module globals at call time, so wrapping them in Stochastic_Reconfiguration records the time
and the work of each phase of the real step. The settings default to the thesis example run
(QuantumDot_ExampleRun.ipynb: train_model_sr_energy(sampler_steps=50, lap_mode="exact") with
micro_batch=1200, total_rows=9000). The first step carries the one-off burn-in and is reported
separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import functions.Stochastic_Reconfiguration as SRM  # noqa: E402
from analysis.system import System, _bf_kwargs  # noqa: E402
from functions.Neural_Networks import psi_fn  # noqa: E402
from functions.Physics import compute_coulomb_interaction  # noqa: E402

STATS: dict = defaultdict(lambda: dict(t=0.0, calls=0, work=0))


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _wrap(name: str, fn, device: str, work):
    def timed(*args, **kwargs):
        _sync(device)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _sync(device)
        s = STATS[name]
        s["t"] += time.perf_counter() - t0
        s["calls"] += 1
        s["work"] += work(args, kwargs)
        return out
    return timed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=2, help="timed SR steps after the burn-in step")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--micro-batch", type=int, default=1200)
    ap.add_argument("--total-rows", type=int, default=9000)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--lap-mode", default="exact")
    ap.add_argument("--small-bf", action="store_true", help="the smaller backflow widths")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    system = System(N=a.N, omega=a.omega, d=2, arch="ctnn_vcycle", use_backflow=True,
                    backflow_kwargs=_bf_kwargs(not a.small_bf), backflow_arch="ctnn",
                    device=a.device, seed=0)
    n_params = sum(p.numel() for m in system.modules() for p in m.parameters())

    # work = walker-sweeps for the sampler (Psi evaluations), rows for the other two
    SRM._persistent_rw = _wrap("sampling", SRM._persistent_rw, a.device,
                               lambda ar, kw: int(ar[1].shape[0]) * int(ar[2]))
    SRM._local_energy_multi = _wrap("local_energy", SRM._local_energy_multi, a.device,
                                    lambda ar, kw: int(ar[1].shape[0]))
    SRM._score_rows = _wrap("score_rows", SRM._score_rows, a.device,
                            lambda ar, kw: int(ar[1].shape[0]))

    kw = dict(psi_fn=psi_fn, compute_coulomb_interaction=compute_coulomb_interaction,
              backflow_net=system.backflow_net, spin=system.spin, params=system.params,
              micro_batch=a.micro_batch, total_rows=a.total_rows,
              sampler_steps=a.sampler_steps, lap_mode=a.lap_mode)
    steps = []
    for k in range(a.steps + 1):
        STATS.clear()
        _sync(a.device)
        t0 = time.perf_counter()
        SRM.sr_step_energy_mb(system.f_net, system.C_occ, **kw)
        _sync(a.device)
        total = time.perf_counter() - t0
        rec = {name: dict(s) for name, s in STATS.items()}
        rec["total"] = total
        rec["rest"] = total - sum(s["t"] for s in STATS.values())
        steps.append(rec)
        label = "burn-in step" if k == 0 else f"step {k}"
        print(f"N={a.N} {label:12s} total {total:8.1f} s | " + " | ".join(
            f"{n} {s['t']:.1f} s ({s['work']} {'walker-sweeps' if n == 'sampling' else 'rows'})"
            for n, s in rec.items() if isinstance(s, dict)) + f" | rest {rec['rest']:.1f} s",
            flush=True)

    out = Path(f"results/analysis/{date.today().isoformat()}_production_sr_"
               f"{a.device.replace(':', '')}_N{a.N}{('_' + a.tag) if a.tag else ''}")
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(args=vars(a), n_params=n_params, torch=torch.__version__, steps=steps)
    if a.device.startswith("cuda"):
        rec["gpu"] = torch.cuda.get_device_name(torch.device(a.device))
    (out / "production_sr.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
