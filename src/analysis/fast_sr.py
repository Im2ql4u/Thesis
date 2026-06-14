"""Fast per-sample score matrix and stochastic-reconfiguration (natural-gradient) training.

The repo's score builder (Stochastic_Reconfiguration._score_rows) loops over samples with
retain_graph, which is too slow for many SR steps on the analysis ansatz. Here we vectorise the
per-sample parameter gradients with torch.func.vmap(grad) and solve the SR linear system in
*sample space* (Woodbury), which is cheap when n_samples < n_params:

    O[k,i] = d log|Psi(x_k)|/dtheta_i      (vmap'd; no Python loop)
    natural-gradient step:  dtheta = O^T (G + lambda I)^{-1} u / B,
                            G = O O^T / B  (B x B),  u = E_L - <E_L>

This realises the projected imaginary-time / Fubini-Study step. SR reaches the variational minimum
(near-DMC) where Adam plateaus, and does so stably. build_O_fast is also a drop-in fast replacement
for the diagnostics' O builder.
"""

from __future__ import annotations

import math

import torch
from torch.func import functional_call, grad, vmap
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from functions.Slater_Determinant import slater_determinant_closed_shell

from .diagnostics import local_energy


def _functional_logpsi(system):
    """Return (logpsi_single, param_dicts, buffer_dicts) for torch.func.

    logpsi_single(param_dicts, x_single) -> scalar log|Psi|, differentiable wrt param_dicts.
    """
    mods = system.modules()
    bdicts = [dict(m.named_buffers()) for m in mods]
    bf = system.backflow_net is not None
    spin = system.spin
    C_occ = system.C_occ
    params = system.params

    def logpsi_single(pdicts, xi):
        X = xi.unsqueeze(0)  # (1, N, d)
        f = functional_call(system.f_net, {**pdicts[0], **bdicts[0]}, (X,), {"spin": spin})
        x_eff = X
        if bf:
            dx = functional_call(system.backflow_net, {**pdicts[1], **bdicts[1]}, (X,), {"spin": spin})
            x_eff = X + dx
        _, logabs = slater_determinant_closed_shell(
            x_config=x_eff, C_occ=C_occ, params=params, spin=spin, normalize=True
        )
        return (logabs.view(-1) + f.view(-1))[0]

    return logpsi_single, [dict(m.named_parameters()) for m in mods], bdicts


def build_O_fast(system, x: torch.Tensor, *, center: bool = True, chunk: int = 1024) -> torch.Tensor:
    """Per-sample score matrix O (B, P) via vmap(grad). Column order matches
    parameters_to_vector(system params). Chunked over samples for memory."""
    logpsi_single, pdicts, _ = _functional_logpsi(system)
    gfn = grad(logpsi_single, argnums=0)
    vg = vmap(gfn, in_dims=(None, 0))

    cols_all = []
    for s in range(0, x.shape[0], chunk):
        xb = x[s : s + chunk].detach()
        per = vg(pdicts, xb)  # list (per module) of dict name->(B, *shape)
        cols = []
        for mi, pd in enumerate(pdicts):
            for name in pd:  # named_parameters order == parameters() order
                cols.append(per[mi][name].reshape(xb.shape[0], -1))
        cols_all.append(torch.cat(cols, dim=1))
    O = torch.cat(cols_all, dim=0)
    if center:
        O = O - O.mean(dim=0, keepdim=True)
    return O


def sr_natural_step(
    system, x: torch.Tensor, E_L: torch.Tensor, *, damping: float, lr: float, max_step: float
) -> dict:
    """One natural-gradient (SR) parameter update from samples x and local energies E_L.

    dtheta = O^T (G + lambda I)^{-1} u / B  (sample-space Woodbury), with a trust-region clip.
    Returns diagnostics {step_norm, g_norm}."""
    from .diagnostics import build_O

    O = build_O(system.log_psi, x, system.modules(), center=True).double()
    B = O.shape[0]
    u = (E_L.detach().double() - E_L.detach().double().mean()).to(O.device)  # (B,)
    G = (O @ O.t()) / B  # (B,B)
    G.diagonal().add_(damping)
    y = torch.linalg.solve(G, u)  # (B,)
    dtheta = (O.t() @ y) / B  # (P,)
    g_norm = float((O.t() @ u / B).norm())
    n = float(dtheta.norm())
    if n > max_step:
        dtheta = dtheta * (max_step / (n + 1e-30))
    params = [p for m in system.modules() for p in m.parameters()]
    theta = parameters_to_vector(params).double()
    vector_to_parameters((theta - lr * dtheta).to(theta.dtype), params)
    return {"step_norm": float(lr * min(n, max_step)), "g_norm": g_norm}


def train_sr(
    system,
    *,
    steps: int = 300,
    batch: int = 1024,
    sampler_steps: int = 30,
    sampler_sigma: float = 0.4,
    lr: float = 0.3,
    damping: float = 1e-3,
    damping_final: float | None = None,
    max_step: float = 0.05,
    clip_mad: float = 8.0,
    lap_mode: str = "exact",
    log_every: int = 20,
    log_fn=print,
    ref_energy: float | None = None,
) -> dict:
    """SR (natural-gradient) VMC training. Warm-start from an Adam result for a clean approach to
    the variational minimum (~DMC). Walkers persist; damping anneals log-linearly if damping_final
    is set."""
    from functions.Stochastic_Reconfiguration import _persistent_rw

    ell = 1.0 / math.sqrt(system.omega)
    sig = sampler_sigma * ell
    xw = torch.randn(batch, system.N, system.d, device=system.device, dtype=system.dtype) * ell
    xw, sig, _, _ = _persistent_rw(system.log_psi, xw, steps=300, sigma=sig,
                                   adapt=True, target=0.5, adapt_lr=0.05)

    hist = {"step": [], "E": [], "var": []}
    for t in range(steps):
        xw, sig, _, _ = _persistent_rw(system.log_psi, xw, steps=sampler_steps, sigma=sig,
                                       adapt=True, target=0.5, adapt_lr=0.02)
        E_L = local_energy(system.log_psi, xw, system.omega, system.params, lap_mode=lap_mode)
        finite = torch.isfinite(E_L)
        xb, E_Lb = xw[finite].detach(), E_L[finite].detach()
        med = E_Lb.median()
        mad = (E_Lb - med).abs().median() + 1e-30
        E_cl = E_Lb.clamp(med - clip_mad * mad, med + clip_mad * mad)

        damp = damping
        if damping_final is not None and steps > 1:
            frac = t / (steps - 1)
            damp = math.exp(math.log(damping) + frac * (math.log(damping_final) - math.log(damping)))
        info = sr_natural_step(system, xb, E_cl, damping=damp, lr=lr, max_step=max_step)

        if (t % log_every == 0) or (t == steps - 1):
            e = float(E_cl.mean()); v = float(E_cl.var())
            hist["step"].append(t); hist["E"].append(e); hist["var"].append(v)
            err = "" if ref_energy is None else f" ({(e-ref_energy)/abs(ref_energy)*100:+.3f}%)"
            log_fn(f"[sr {t:04d}] E={e:.6f}{err} var={v:.3e} |dθ|={info['step_norm']:.2e} "
                   f"|g|={info['g_norm']:.2e} damp={damp:.1e}")
    return hist
