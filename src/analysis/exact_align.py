"""Exact-truth validation of the SR = projected imaginary-time claim (N=2).

Our earlier cos(SR, imag-time) was self-referential (built from the same Psi) and measured at
convergence (noise). Here we use the *exact* N=2 ground state as ground truth:

  delta_toward_exact(x) = (log|Psi_exact| - log|Psi_theta|), centred  -> the direction in
  function space that moves the variational state toward the true ground state.

We then check:
  * cos(imag-time field, delta_toward_exact)  -- does -(E_L-E) actually point toward the GS?
  * cos(SR step, delta_toward_exact) vs cos(plain step, delta_toward_exact)
  * representable fraction of delta_toward_exact (expressivity ceiling)
all on *under-converged* states (real signal), swept vs distance-from-GS.
"""

from __future__ import annotations

import numpy as np
import torch

from .diagnostics import build_O, local_energy
from .reference import TwoElectronExact


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@torch.no_grad()
def _delta_toward_exact(system, x: torch.Tensor, ex: TwoElectronExact) -> torch.Tensor:
    lognet = system.log_psi(x).double()
    logex = torch.tensor(ex.log_psi(x.detach().cpu().double().numpy()),
                         device=x.device, dtype=torch.float64)
    d = logex - lognet
    return d - d.mean()


def exact_alignment(system, x: torch.Tensor, ex: TwoElectronExact | None = None) -> dict:
    """One-shot alignment of SR vs plain gradient with the exact-GS direction (N=2)."""
    assert system.N == 2, "exact_alignment is for N=2 (Taut solution)"
    if ex is None:
        ex = TwoElectronExact(omega=system.omega)
    delta = _delta_toward_exact(system, x, ex)  # (B,) toward exact GS

    O = build_O(system.log_psi, x, system.modules(), center=True).double()
    E_L = local_energy(system.log_psi, x, system.omega, system.params, lap_mode="exact").double()
    r = E_L - E_L.mean()
    imagtime = -r  # imaginary-time descent direction in function space

    K = O @ O.t()
    mu, V = torch.linalg.eigh(K)
    mu = torch.clamp(mu, min=0.0)
    supp = mu > float(mu.max()) * 1e-10

    def proj(v):  # tangent-space (SR/whitened) projection
        c = V.t() @ v
        return V[:, supp] @ c[supp]

    dpsi_sr = proj(imagtime)
    dpsi_plain = K @ imagtime
    return {
        "cos_imagtime_toward_exact": _cos(imagtime, delta),  # does H-flow point to the GS?
        "cos_sr_toward_exact": _cos(dpsi_sr, delta),         # SR step alignment with GS direction
        "cos_plain_toward_exact": _cos(dpsi_plain, delta),   # plain-grad alignment
        "rep_fraction_exact_dir": float(proj(delta).norm() / (delta.norm() + 1e-30)),
        "residual_norm": float(r.norm() / np.sqrt(r.numel())),  # RMS residual (signal level)
        "energy_err_pct": float((E_L.mean().item() - ex.energy) / abs(ex.energy) * 100),
    }


def alignment_vs_distance(system, x: torch.Tensor, *, noise_levels=(0.0, 0.02, 0.05, 0.1, 0.2),
                          seed: int = 0) -> dict:
    """Sweep distance-from-GS by perturbing parameters; measure alignment vs residual level.

    Perturbing theta away from the trained GS creates a structured (non-noise) residual; this is the
    regime where the SR-vs-plain distinction is meaningful."""
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    ex = TwoElectronExact(omega=system.omega)
    params = [p for m in system.modules() for p in m.parameters()]
    theta0 = parameters_to_vector(params).detach().clone()
    g = torch.Generator(device=theta0.device).manual_seed(seed)
    direction = torch.randn(theta0.shape, generator=g, device=theta0.device, dtype=theta0.dtype)
    direction = direction / direction.norm()

    rows = {"noise": [], "residual": [], "cos_imagtime": [], "cos_sr": [], "cos_plain": [],
            "rep_fraction": [], "energy_err_pct": []}
    for eps in noise_levels:
        vector_to_parameters(theta0 + eps * direction * theta0.norm(), params)
        out = exact_alignment(system, x, ex)
        rows["noise"].append(float(eps))
        rows["residual"].append(out["residual_norm"])
        rows["cos_imagtime"].append(out["cos_imagtime_toward_exact"])
        rows["cos_sr"].append(out["cos_sr_toward_exact"])
        rows["cos_plain"].append(out["cos_plain_toward_exact"])
        rows["rep_fraction"].append(out["rep_fraction_exact_dir"])
        rows["energy_err_pct"].append(out["energy_err_pct"])
    vector_to_parameters(theta0, params)  # restore
    return {k: np.array(v) for k, v in rows.items()}
