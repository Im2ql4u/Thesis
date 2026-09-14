"""Per-step wall time of the training routes, on one ansatz, across particle numbers.

A route is a sampling paradigm times an optimiser:

  vmc     `sampler_steps` Metropolis sweeps of persistent walkers drawn from |Psi|^2.
  colloc  oversample*n_coll draws from the thesis's origin-centred Gaussian mixture
          (Eq. colloc-proposal), log|Psi| on all of them, multinomial resampling of n_coll by
          w = |Psi|^2/q. No Markov chain.

  adam    REINFORCE loss 2<(E_L - <E_L>) log|Psi|>, backward, Adam step.
  sr      the natural-gradient step (sr_natural_step: per-sample score matrix, sample-space
          Woodbury solve) on `sr_samples` of the batch. After resampling the collocation batch
          approximates |Psi|^2, so the same step applies to both paradigms.

For every (N, route) the same freshly built System (CTNN Jastrow + CTNN backflow) is used, with
the same exact-Laplacian local energy. Each step is split into sample / energy / update, the
first step of each route is discarded as warm-up, and the network is left at initialisation:
the trained state does not change per-step cost. Per-step cost is not time to convergence.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.diagnostics import local_energy  # noqa: E402
from analysis.fast_sr import sr_natural_step  # noqa: E402
from analysis.system import System, _bf_kwargs  # noqa: E402
from analysis.train import GRAD_CLIP, _heal_walkers, _param_groups  # noqa: E402
from exp_conditioning_A import _proposal_logq, _sample_proposal  # noqa: E402
from functions.Stochastic_Reconfiguration import _persistent_rw  # noqa: E402

THESIS_MIXTURE = (0.8, 1.3, 2.0)   # sigma_f of Eq. (colloc-proposal), widths in units of ell
BURN_IN_SWEEPS = 300               # train_sr's walker burn-in, excluded from the timing
ADAM_LR = 5e-4                     # collocation campaign learning rate
SR_KW = dict(damping=1e-3, lr=0.1, max_step=0.05)
ROUTES = ("vmc-adam", "vmc-sr", "colloc-adam", "colloc-sr")


def _now(device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


class VmcSampler:
    """Persistent Metropolis walkers, burned in once per System and shared by both optimisers."""

    def __init__(self, system: System, batch: int, sampler_steps: int) -> None:
        self.system, self.steps = system, sampler_steps
        self.ell = 1.0 / math.sqrt(system.omega)
        x = torch.randn(batch, system.N, system.d, device=system.device,
                        dtype=system.dtype) * self.ell
        self.x, self.sig, _, _ = _persistent_rw(system.log_psi, x, steps=BURN_IN_SWEEPS,
                                                sigma=0.4 * self.ell, adapt=True, target=0.5,
                                                adapt_lr=0.05)

    def __call__(self) -> torch.Tensor:
        self.x, self.sig, _, _ = _persistent_rw(self.system.log_psi, self.x, steps=self.steps,
                                                sigma=self.sig, adapt=True, target=0.5,
                                                adapt_lr=0.02)
        self.x = _heal_walkers(self.x, self.ell)
        return self.x


def colloc_sample(system: System, n_coll: int, oversample: int, chunk: int) -> torch.Tensor:
    with torch.no_grad():
        cand = _sample_proposal(system, oversample * n_coll, THESIS_MIXTURE)
        # Chunked: the backflow's pair tensor is O(N^2 h) per configuration, so one pass over
        # all oversample*n_coll proposals does not fit in GPU memory at N=20.
        log_psi = torch.cat([system.log_psi(c) for c in torch.split(cand, chunk)])
        logw = 2.0 * log_psi - _proposal_logq(system, cand, THESIS_MIXTURE)
        ok = torch.isfinite(logw)
        idx = torch.multinomial(torch.softmax(logw[ok], dim=0), n_coll, replacement=True)
        return cand[ok][idx]


def time_route(system: System, sample, optimiser: str, *, steps: int, sr_samples: int,
               chunk: int) -> list[dict]:
    params, groups = _param_groups(system, ADAM_LR)
    opt = torch.optim.Adam(groups, lr=ADAM_LR)
    rows = []
    for _ in range(steps + 1):
        t0 = _now(system.device)
        x = sample()
        t1 = _now(system.device)
        e_l = local_energy(system.log_psi, x, system.omega, system.params, lap_mode="exact",
                           chunk=chunk).detach()
        t2 = _now(system.device)
        ok = torch.isfinite(e_l)
        xs, es = x[ok].detach(), e_l[ok]
        if optimiser == "sr":
            sr_natural_step(system, xs[:sr_samples], es[:sr_samples], **SR_KW)
        else:
            loss = 2.0 * ((es - es.mean()) * system.log_psi(xs)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
        t3 = _now(system.device)
        rows.append(dict(sample=t1 - t0, energy=t2 - t1, update=t3 - t2, batch=int(x.shape[0])))
    return rows[1:]


def _mean(rows: list[dict]) -> dict:
    out = {k: sum(r[k] for r in rows) / len(rows) for k in ("sample", "energy", "update")}
    out["total"] = out["sample"] + out["energy"] + out["update"]
    return out


def run_N(N: int, a: argparse.Namespace) -> dict:
    torch.manual_seed(0)
    system = System(N=N, omega=a.omega, d=2, arch="ctnn_vcycle", use_backflow=True,
                    backflow_kwargs=_bf_kwargs(True), backflow_arch="ctnn",
                    device=a.device, seed=0)
    rec = dict(n_params=sum(p.numel() for m in system.modules() for p in m.parameters()),
               routes={})
    vmc = None
    for route in a.routes:
        paradigm, optimiser = route.split("-")
        if paradigm == "vmc":
            vmc = vmc or VmcSampler(system, a.vmc_batch, a.sampler_steps)
            sample = vmc
        else:
            def sample():
                return colloc_sample(system, a.n_coll, a.oversample, a.proposal_chunk)
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        rows = time_route(system, sample, optimiser, steps=a.steps, sr_samples=a.sr_samples,
                          chunk=a.chunk)
        m = _mean(rows)
        if a.device.startswith("cuda"):
            m["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec["routes"][route] = dict(steps=rows, mean=m)
        print(f"N={N:2d} {route:12s} {m['total']:8.3f} s/step  (sample {m['sample']:.3f}, "
              f"energy {m['energy']:.3f}, update {m['update']:.3f}, batch {rows[0]['batch']})",
              flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--N", type=int, nargs="+", default=[6])
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--routes", nargs="+", default=list(ROUTES), choices=ROUTES)
    ap.add_argument("--steps", type=int, default=3, help="timed steps per route (after warm-up)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--vmc-batch", type=int, default=3072, help="walkers (thesis: ~3e3)")
    ap.add_argument("--sampler-steps", type=int, default=30, help="Metropolis sweeps per step")
    ap.add_argument("--sr-samples", type=int, default=1024)
    ap.add_argument("--n-coll", type=int, default=4096)
    ap.add_argument("--oversample", type=int, default=16, help="16 robust recipe, 8 baseline")
    ap.add_argument("--chunk", type=int, default=1024, help="local-energy chunk (memory)")
    ap.add_argument("--proposal-chunk", type=int, default=8192,
                    help="configurations per log|Psi| pass over the collocation proposals")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    hw = dict(machine=platform.machine(), processor=platform.processor(), node=platform.node(),
              torch=torch.__version__, threads=torch.get_num_threads())
    if a.device.startswith("cuda"):
        hw["gpu"] = torch.cuda.get_device_name(torch.device(a.device))
    rec = dict(args=vars(a), hardware=hw, results={})
    for N in a.N:
        rec["results"][str(N)] = run_N(N, a)

    out = Path(f"results/analysis/{date.today().isoformat()}_walltime_"
               f"{a.device.replace(':', '')}{('_' + a.tag) if a.tag else ''}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "walltime.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
