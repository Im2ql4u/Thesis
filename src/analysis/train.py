"""Fast VMC trainers for the analysis program.

train_vmc_adam: standard variational Monte Carlo energy minimisation with the
REINFORCE / score-function gradient and Adam. One batched backward per step
(no per-sample parameter-gradient loop), so it is fast even for over-parameterised
networks. Sampling is MCMC from |Psi|^2 with persistent, sigma-adapted walkers.

The SR / natural-gradient machinery (src/sr_preconditioner.py,
functions.Stochastic_Reconfiguration) remains available for an optional polish and,
importantly, for the *diagnostics* (the per-sample score matrix O). Training speed
should not depend on it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .diagnostics import local_energy


def train_vmc_adam(
    system,
    *,
    steps: int = 2000,
    lr: float = 3e-3,
    batch: int = 2048,
    sampler_steps: int = 20,
    sampler_sigma: float = 0.4,
    clip_mad: float = 5.0,
    lap_mode: str = "exact",
    log_every: int = 100,
    log_fn=print,
) -> dict:
    """Minimise <E> by Adam on the score-function (REINFORCE) gradient.

    Returns a history dict {step, E, var}. Walkers persist across steps.
    """
    from functions.Stochastic_Reconfiguration import _persistent_rw

    params = [p for m in system.modules() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    ell = 1.0 / math.sqrt(system.omega)
    sig = sampler_sigma * ell
    x = (
        torch.randn(batch, system.N, system.d, device=system.device, dtype=system.dtype) * ell
    )
    # burn-in
    x, sig, _, _ = _persistent_rw(system.log_psi, x, steps=200, sigma=sig,
                                  adapt=True, target=0.5, adapt_lr=0.05)

    hist = {"step": [], "E": [], "var": []}
    for t in range(steps):
        x, sig, _, _ = _persistent_rw(system.log_psi, x, steps=sampler_steps, sigma=sig,
                                      adapt=True, target=0.5, adapt_lr=0.02)
        E_L = local_energy(system.log_psi, x, system.omega, system.params, lap_mode=lap_mode)
        finite = torch.isfinite(E_L)
        xb, E_L = x[finite].detach(), E_L[finite].detach()
        med = E_L.median()
        mad = (E_L - med).abs().median() + 1e-30
        E_cl = E_L.clamp(med - clip_mad * mad, med + clip_mad * mad)
        R = E_cl.mean()

        logpsi = system.log_psi(xb)  # requires grad wrt params
        loss = 2.0 * ((E_cl - R).detach() * logpsi).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if (t % log_every == 0) or (t == steps - 1):
            e = float(R)
            v = float(E_cl.var())
            hist["step"].append(t); hist["E"].append(e); hist["var"].append(v)
            log_fn(f"[adam {t:05d}] E={e:.6f} var={v:.4e} sig={sig:.3f} acc~0.5 n={xb.shape[0]}")
    return hist
