"""Per-step wall time: SR-VMC against Markov-chain-free collocation, on one ansatz.

Both routes run on the same freshly built System (CTNN Jastrow + CTNN backflow), the same device
and the same exact-Laplacian local energy, so the difference is the paradigm and nothing else.
Each gradient step is split into three phases:

  sample  VMC: `sampler_steps` Metropolis sweeps of the persistent walkers.
          collocation: draw oversample*n_coll points from the thesis's origin-centred Gaussian
          mixture (Eq. colloc-proposal), evaluate log|Psi| on all of them, multinomial-resample
          n_coll by w = |Psi|^2/q.
  energy  the exact-Laplacian local energy on the batch that enters the gradient.
  update  VMC: the SR natural-gradient step (sr_natural_step) on `sr_samples` of the walkers.
          collocation: the REINFORCE loss, backward and an Adam step.

The trained state does not matter for per-step cost, so the network is left at its
initialisation. The first step of each route is discarded as warm-up. CPU timings do not
transfer to a GPU: there the collocation batch is one parallel evaluation while Metropolis
sweeps stay sequential, so treat the ratio as hardware-specific.
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


def _now(device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def time_vmc_sr(system: System, *, steps: int, batch: int, sampler_steps: int,
                sr_samples: int) -> list[dict]:
    ell = 1.0 / math.sqrt(system.omega)
    sig = 0.4 * ell
    x = torch.randn(batch, system.N, system.d, device=system.device, dtype=system.dtype) * ell
    x, sig, _, _ = _persistent_rw(system.log_psi, x, steps=BURN_IN_SWEEPS, sigma=sig,
                                  adapt=True, target=0.5, adapt_lr=0.05)
    rows = []
    for _ in range(steps + 1):
        t0 = _now(system.device)
        x, sig, _, _ = _persistent_rw(system.log_psi, x, steps=sampler_steps, sigma=sig,
                                      adapt=True, target=0.5, adapt_lr=0.02)
        x = _heal_walkers(x, ell)
        t1 = _now(system.device)
        e_l = local_energy(system.log_psi, x, system.omega, system.params, lap_mode="exact")
        t2 = _now(system.device)
        finite = torch.isfinite(e_l)
        xs, es = x[finite][:sr_samples].detach(), e_l[finite][:sr_samples].detach()
        sr_natural_step(system, xs, es, damping=1e-3, lr=0.1, max_step=0.05)
        t3 = _now(system.device)
        rows.append(dict(sample=t1 - t0, energy=t2 - t1, update=t3 - t2))
    return rows[1:]


def time_collocation(system: System, *, steps: int, n_coll: int, oversample: int) -> list[dict]:
    params, groups = _param_groups(system, ADAM_LR)
    opt = torch.optim.Adam(groups, lr=ADAM_LR)
    rows = []
    for _ in range(steps + 1):
        t0 = _now(system.device)
        with torch.no_grad():
            cand = _sample_proposal(system, oversample * n_coll, THESIS_MIXTURE)
            logw = 2.0 * system.log_psi(cand) - _proposal_logq(system, cand, THESIS_MIXTURE)
            p = torch.softmax(logw[torch.isfinite(logw)], dim=0)
            idx = torch.multinomial(p, n_coll, replacement=True)
            x = cand[torch.isfinite(logw)][idx]
        t1 = _now(system.device)
        e_l = local_energy(system.log_psi, x, system.omega, system.params,
                           lap_mode="exact").detach()
        t2 = _now(system.device)
        finite = torch.isfinite(e_l)
        reward = e_l[finite] - e_l[finite].mean()
        loss = 2.0 * (reward * system.log_psi(x[finite])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        t3 = _now(system.device)
        rows.append(dict(sample=t1 - t0, energy=t2 - t1, update=t3 - t2))
    return rows[1:]


def _summary(rows: list[dict]) -> dict:
    keys = ("sample", "energy", "update")
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    mean["total"] = sum(mean.values())
    return mean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3, help="timed steps per route (after warm-up)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--vmc-batch", type=int, default=3072, help="walkers (thesis: ~3e3)")
    ap.add_argument("--sampler-steps", type=int, default=30, help="Metropolis sweeps per step")
    ap.add_argument("--sr-samples", type=int, default=1024)
    ap.add_argument("--n-coll", type=int, default=4096)
    ap.add_argument("--oversample", type=int, default=16, help="16 robust recipe, 8 baseline")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    system = System(N=a.N, omega=a.omega, d=2, arch="ctnn_vcycle", use_backflow=True,
                    backflow_kwargs=_bf_kwargs(True), backflow_arch="ctnn",
                    device=a.device, seed=0)
    n_params = sum(p.numel() for m in system.modules() for p in m.parameters())

    vmc = time_vmc_sr(system, steps=a.steps, batch=a.vmc_batch,
                      sampler_steps=a.sampler_steps, sr_samples=a.sr_samples)
    col = time_collocation(system, steps=a.steps, n_coll=a.n_coll, oversample=a.oversample)

    out = Path(f"results/analysis/{date.today().isoformat()}_walltime_{a.device}"
               f"_N{a.N}_w{str(a.omega).replace('.', 'p')}")
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(args=vars(a), n_params=n_params, torch=torch.__version__,
               machine=platform.machine(), processor=platform.processor(),
               threads=torch.get_num_threads(),
               vmc_sr=dict(steps=vmc, mean=_summary(vmc)),
               collocation=dict(steps=col, mean=_summary(col)))
    (out / "walltime.json").write_text(json.dumps(rec, indent=2) + "\n")
    for name, key in (("SR-VMC", "vmc_sr"), ("collocation", "collocation")):
        m = rec[key]["mean"]
        print(f"{name:12s} total {m['total']:7.2f} s/step  (sample {m['sample']:.2f}, "
              f"energy {m['energy']:.2f}, update {m['update']:.2f})")
    print(f"ratio collocation/SR-VMC per step: "
          f"{rec['collocation']['mean']['total'] / rec['vmc_sr']['mean']['total']:.2f}  -> {out}")


if __name__ == "__main__":
    main()
