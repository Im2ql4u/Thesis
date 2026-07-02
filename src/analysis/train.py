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


def _heal_walkers(x: torch.Tensor, ell: float) -> torch.Tensor:
    """Replace any walker that diverged to non-finite with a finite one (or re-init)."""
    bad = ~torch.isfinite(x).reshape(x.shape[0], -1).all(dim=1)
    if not bool(bad.any()):
        return x
    good = (~bad).nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        idx = good[torch.randint(good.numel(), (int(bad.sum()),), device=x.device)]
        x = x.clone()
        x[bad] = x[idx]
    else:
        x = torch.randn_like(x) * ell
    return x


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
    ckpt_every: int = 0,
    ckpt_fn=None,
) -> dict:
    """Minimise <E> by Adam on the score-function (REINFORCE) gradient.

    Returns a history dict {step, E, var}. Walkers and the optimiser persist across the call
    (a single call => no momentum resets). If ckpt_every>0 and ckpt_fn is given, calls
    ckpt_fn(step) every ckpt_every steps (for lazy-vs-rich / training-dynamics analysis).
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
        x = _heal_walkers(x, ell)  # resample any diverged (non-finite) walkers
        E_L = local_energy(system.log_psi, x, system.omega, system.params, lap_mode=lap_mode)
        finite = torch.isfinite(E_L)
        if int(finite.sum()) < 16:  # transient bad batch; skip update, keep healed walkers
            continue
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
        if ckpt_every and ckpt_fn is not None and ((t % ckpt_every == 0) or (t == steps - 1)):
            ckpt_fn(t)
    return hist


def train_collocation_weak(
    system,
    *,
    steps: int = 2000,
    lr: float = 3e-3,
    batch: int = 2048,
    sigma_q: float = 1.3,
    clip_w: float = 20.0,
    clip_mad: float = 8.0,
    log_every: int = 100,
    log_fn=print,
) -> dict:
    """Weak-form collocation training (the collocation paradigm, for the dual-track comparison).

    Minimises the importance-sampled weak (Rayleigh) energy
      E[Psi] = < 1/2 |grad logPsi|^2 + V >_{|Psi|^2}
    by sampling a FIXED Gaussian proposal q (NOT |Psi|^2-MCMC) and reweighting w = |Psi|^2/q. Uses the
    exact self-normalised importance-sampling gradient of the Rayleigh quotient,
      dE/dtheta = <d e_w/dtheta>_w + 2 <(e_w - E) d logPsi/dtheta>_w,
    (direct term + score/REINFORCE term). This is the weak-form/first-derivative collocation loss; it
    never samples |Psi|^2, so it is the paradigm contrast to train_vmc_adam. Returns {step, E, ess}.
    """
    from .diagnostics import residual_local_energy

    params = [p for m in system.modules() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    ell = 1.0 / math.sqrt(system.omega)
    sq = sigma_q * ell
    log_zq = system.N * system.d * math.log(sq * math.sqrt(2 * math.pi))
    hist = {"step": [], "E": [], "ess": []}
    for t in range(steps):
        x = torch.randn(batch, system.N, system.d, device=system.device, dtype=system.dtype) * sq
        with torch.no_grad():
            logq = -0.5 * (x ** 2).sum(dim=(1, 2)) / sq ** 2 - log_zq
            logw = 2.0 * system.log_psi(x) - logq
            logw = logw - logw.max()
            w = torch.exp(logw)
            w = torch.clamp(w, max=clip_w * (w.median() + 1e-30))  # tame extreme weights
            ess = float((w.sum() ** 2) / (w ** 2).sum()) / batch
        e_w = residual_local_energy(system, x, form="weak")
        finite = torch.isfinite(e_w) & (w > 0)
        if int(finite.sum()) < 16:
            continue
        ew, wn = e_w[finite], w[finite]
        wn = (wn / wn.sum()).detach()
        # clip the residual (Coulomb coalescence spikes) about the weighted median for stability
        med = ew.detach().median()
        mad = (ew.detach() - med).abs().median() + 1e-30
        ew_cl = ew.clamp(med - clip_mad * mad, med + clip_mad * mad)
        E = (wn * ew_cl.detach()).sum()
        logpsi = system.log_psi(x[finite])
        loss = (wn * ew_cl).sum() + 2.0 * (wn * (ew_cl.detach() - E) * logpsi).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if (t % log_every == 0) or (t == steps - 1):
            hist["step"].append(t); hist["E"].append(float(E)); hist["ess"].append(ess)
            log_fn(f"[colloc {t:05d}] E_weak={float(E):.6f} ESS={ess:.3f} n={int(finite.sum())}")
    return hist
