"""Backflow / nodal analysis (the overlooked half of Q2).

The coordinate backflow x_eff = x + Delta x(x) deforms the Slater nodes (it is what beats fixed-node
DMC). We measure:
  force_alignment   : cos(Delta x, F_trap), cos(Delta x, F_Coulomb), cos(Delta x, F_total) -> the
                      trap-restoring vs Coulomb-lattice regime switch across omega
  displacement      : |Delta x|/ell, fraction displaced toward the nearest neighbour
  ablation_energy   : energy with backflow removed (x_eff=x) -> the fixed-node contribution
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def classical_forces(x: torch.Tensor, omega: float):
    """Trap, Coulomb, and total classical force on each particle (2D Coulomb 1/r)."""
    B, N, d = x.shape
    diff = x.unsqueeze(2) - x.unsqueeze(1)                # (B,N,N,d): x_i - x_j
    r = diff.norm(dim=-1, keepdim=True)                   # (B,N,N,1)
    eye = torch.eye(N, device=x.device, dtype=torch.bool).view(1, N, N, 1)
    inv = torch.where(eye, torch.zeros_like(r), 1.0 / (r**3 + 1e-12))
    F_coul = (diff * inv).sum(dim=2)                      # (B,N,d) repulsive, outward
    F_trap = -(omega**2) * x                              # inward
    return F_trap, F_coul, F_trap + F_coul


def _cosmean(a, b):
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-30
    return float((num / den).mean())


@torch.no_grad()
def backflow_analysis(system, x: torch.Tensor) -> dict:
    if system.backflow_net is None:
        return {"available": False}
    dx = system.backflow_net(x, spin=system.spin)         # (B,N,d)
    Ft, Fc, Ftot = classical_forces(x, system.omega)
    ell = 1.0 / np.sqrt(system.omega)
    # nearest-neighbour direction
    B, N, d = x.shape
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    r = diff.norm(dim=-1)
    r = r.masked_fill(torch.eye(N, device=x.device, dtype=torch.bool), float("inf"))
    nn = r.argmin(dim=2)                                   # (B,N)
    to_nn = torch.gather(-diff, 2, nn[..., None, None].expand(B, N, 1, d)).squeeze(2)  # x_j - x_i
    frac_toward_nn = float(((dx * to_nn).sum(-1) > 0).float().mean())
    return {
        "available": True,
        "cos_trap": _cosmean(dx, Ft),
        "cos_coulomb": _cosmean(dx, Fc),
        "cos_total": _cosmean(dx, Ftot),
        "disp_aho": float((dx.norm(dim=-1).mean() / ell)),
        "frac_toward_nn": frac_toward_nn,
    }


@torch.no_grad()
def backflow_ablation_energy(system, *, n_samples: int = 1536, steps: int = 300, burn_in: int = 600) -> dict:
    """Energy with backflow removed (bare HO nodes), each sampled from its own |Psi|^2.
    dE_ablate = E(no backflow) - E(full) = the fixed-node / nodal contribution of backflow."""
    from . import diagnostics as dg

    def _eval():
        x = system.sample(n_samples, steps=steps, burn_in=burn_in)
        E = dg.local_energy(system.log_psi, x, system.omega, system.params, lap_mode="exact")
        E = E[torch.isfinite(E)]
        return float(E.mean()), float(E.var())

    e0, v0 = _eval()
    bf = system.backflow_net
    system.backflow_net = None
    try:
        ez, vz = _eval()
    finally:
        system.backflow_net = bf
    return {"E_full": e0, "var_full": v0, "E_no_bf": ez, "var_no_bf": vz,
            "dE_ablate": ez - e0, "var_ratio": vz / (v0 + 1e-30)}
