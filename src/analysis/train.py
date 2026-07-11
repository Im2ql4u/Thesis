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


GRAD_CLIP = 1.0      # run_weak_form.py --grad-clip default
LR_JAS_FRAC = 0.1    # run_weak_form.py: --lr-jas 5e-5 vs --lr 5e-4 (Jastrow trains 10x slower)


def _param_groups(system, lr: float) -> tuple[list, list[dict]]:
    """Split Jastrow / backflow into the two learning-rate groups the thesis pipeline uses.

    The backflow's dx_head feeds tanh and then a zero-mean (COM) projection: if its
    pre-activation saturates, every particle's tanh output pins to the same +-1 and the
    projection annihilates Delta_x *exactly*, killing the gradient permanently. A slower
    Jastrow lr plus tight clipping is what keeps that pre-activation in the linear region.
    """
    jas = list(system.f_net.parameters())
    bf = list(system.backflow_net.parameters()) if system.backflow_net is not None else []
    groups = [{"params": jas, "lr": lr * LR_JAS_FRAC}]
    if bf:
        groups.append({"params": bf, "lr": lr})
    return jas + bf, groups


@torch.no_grad()
def backflow_health(system, x: torch.Tensor) -> str:
    """|Delta_x|/ell and the tanh saturation fraction — a dead backflow reads |dx|~0, sat~1."""
    bf = system.backflow_net
    if bf is None:
        return ""
    dx = bf(x, spin=system.spin)
    mag = float(dx.norm(dim=-1).mean()) * math.sqrt(system.omega)
    sat = float("nan")
    if getattr(bf, "out_bound", "") == "tanh":
        pre = {}
        h = bf.dx_head.register_forward_hook(lambda m, i, o: pre.__setitem__("v", o.detach()))
        bf(x, spin=system.spin)
        h.remove()
        sat = float((pre["v"].abs() > 2.5).to(torch.float64).mean())  # |tanh'| < 0.03 beyond 2.5
    return f" |dx|/ell={mag:.4f} tanh_sat={sat:.2f}"


def train_vmc_adam(
    system,
    *,
    steps: int = 2000,
    lr: float = 5e-4,
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

    params, groups = _param_groups(system, lr)
    opt = torch.optim.Adam(groups, lr=lr)
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
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()

        if (t % log_every == 0) or (t == steps - 1):
            e = float(R)
            v = float(E_cl.var())
            hist["step"].append(t); hist["E"].append(e); hist["var"].append(v)
            log_fn(f"[adam {t:05d}] E={e:.6f} var={v:.4e} sig={sig:.3f} acc~0.5 n={xb.shape[0]}"
                   + backflow_health(system, xb[:256]))
        if ckpt_every and ckpt_fn is not None and ((t % ckpt_every == 0) or (t == steps - 1)):
            ckpt_fn(t)
    return hist


def train_collocation_weak(
    system,
    *,
    steps: int = 2000,
    lr: float = 5e-4,
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

    params, groups = _param_groups(system, lr)
    opt = torch.optim.Adam(groups, lr=lr)
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
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        if (t % log_every == 0) or (t == steps - 1):
            hist["step"].append(t); hist["E"].append(float(E)); hist["ess"].append(ess)
            log_fn(f"[colloc {t:05d}] E_weak={float(E):.6f} ESS={ess:.3f} n={int(finite.sum())}")
    return hist


def train_collocation_sr(
    system,
    *,
    steps: int = 400,
    batch: int = 1024,
    sigma_q: float = 1.3,
    clip_w: float = 20.0,
    clip_mad: float = 8.0,
    lr: float = 0.1,
    damping: float = 1e-3,
    max_step: float = 0.05,
    log_every: int = 50,
    log_fn=print,
) -> dict:
    """Collocation with a NATURAL-GRADIENT (SR) preconditioner -- the missing {collocation}x{SR}
    quadrant. Same weak-form importance-sampled Rayleigh gradient as train_collocation_weak, but the
    parameter gradient g is preconditioned by the (importance-weighted) quantum geometric tensor
    S = O^T diag(w) O via a sample-space Woodbury solve:  delta = (S + lambda I)^{-1} g.
    Tests whether whitening helps collocation (ill-conditioned measure) more than VMC (self-precond).

    Validated (2026-07-03): converges at N=2 (overlap^2 0.91 -> 0.97 over 240 steps, climbing). The
    weak-form E readout is too noisy to show the descent (it looks stuck ~3.4), but the wavefunction
    improves steadily -- judge by overlap/quality, not E_weak. Slower per step than VMC-SR (builds O +
    solves each step) and per-step progress is small; use enough steps."""
    from .diagnostics import residual_local_energy, build_O

    params = [p for m in system.modules() for p in m.parameters()]
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
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
            w = torch.clamp(torch.exp(logw), max=clip_w * (torch.exp(logw).median() + 1e-30))
            ess = float((w.sum() ** 2) / (w ** 2).sum()) / batch
        e_w = residual_local_energy(system, x, form="weak")
        finite = torch.isfinite(e_w) & (w > 0)
        if int(finite.sum()) < 16:
            continue
        xf = x[finite]; ew = e_w[finite]; wn = (w[finite] / w[finite].sum()).detach()
        med = ew.detach().median(); mad = (ew.detach() - med).abs().median() + 1e-30
        ew_cl = ew.clamp(med - clip_mad * mad, med + clip_mad * mad)
        E = (wn * ew_cl.detach()).sum()
        logpsi = system.log_psi(xf)
        loss = (wn * ew_cl).sum() + 2.0 * (wn * (ew_cl.detach() - E) * logpsi).sum()
        for p in params:
            p.grad = None
        loss.backward()
        g = parameters_to_vector([p.grad for p in params]).double()  # collocation gradient (P,)
        # weighted score matrix and natural-gradient (Woodbury) solve: delta = (S+lam I)^{-1} g
        O = build_O(system.log_psi, xf, system.modules(), center=True).double()  # (B,P)
        Bn = O.shape[0]
        sw = (wn.double() * Bn).sqrt().unsqueeze(1)  # row weights so O_w^T O_w = sum_k w_k O_k O_k^T
        Ow = O * sw
        G = (Ow @ Ow.t()) / Bn
        G.diagonal().add_(damping)
        Og = (Ow @ g) / math.sqrt(Bn)
        y = torch.linalg.solve(G, Og)
        delta = (g - (Ow.t() @ y) / math.sqrt(Bn)) / damping
        n = float(delta.norm())
        if n > max_step:
            delta = delta * (max_step / (n + 1e-30))
        theta = parameters_to_vector(params).double()
        vector_to_parameters((theta - lr * delta).to(theta.dtype), params)
        if (t % log_every == 0) or (t == steps - 1):
            hist["step"].append(t); hist["E"].append(float(E)); hist["ess"].append(ess)
            log_fn(f"[colloc-sr {t:05d}] E_weak={float(E):.6f} ESS={ess:.3f} |dθ|={min(n,max_step)*lr:.2e}")
    return hist
