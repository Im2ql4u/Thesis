"""Fair cross-architecture effective-dimension measurement (Q1).

The per-checkpoint eff-rank(S) numbers in the depth runs were each measured on samples from
that model's *own* |Psi|^2, which confounds "geometry of the ansatz" with "where it puts mass".
This module measures eff-rank(S) of the Jastrow (f_net) tangent space for every model on a
**common probe set** drawn from a fixed reference measure, with a sample-convergence sweep, so
cross-architecture comparisons are fair (same points, same estimator). Reuses build_O /
kernel_spectrum unchanged.

We measure the f_net (correlator) tangent space alone: the backflow is identical across the
CTNN/DeepSet runs (same module, ~13.3k params), so including it only dilutes the architecture
comparison with a shared offset.
"""
from __future__ import annotations

import torch

from . import diagnostics as dg


def effective_dimension(system, x: torch.Tensor, *, modules: list | None = None,
                        rel_tol: float = 1e-12, svd_on_cpu: bool = False) -> dict:
    """eff-rank(S) and condition number of the f_net tangent space at points `x`."""
    mods = modules if modules is not None else [system.f_net]
    O = dg.build_O(system.log_psi, x, mods, center=True)
    if svd_on_cpu:
        O = O.cpu()
    return dg.kernel_spectrum(O, rel_tol=rel_tol)


def pooled_probe_set(systems: list, n_per: int, **sample_kwargs) -> torch.Tensor:
    """Concatenate n_per samples from each system's own |Psi|^2 -> a shared mixture measure."""
    xs = [s.sample(n_per, **sample_kwargs) for s in systems]
    return torch.cat(xs, dim=0)


def dimension_convergence(system, x_common: torch.Tensor, n_grid: list[int], *,
                          modules: list | None = None, rel_tol: float = 1e-12,
                          seed: int = 0, svd_on_cpu: bool = False) -> list[dict]:
    """eff-rank(S) of `system` on nested random subsets of the common probe set `x_common`.

    A converged (plateaued) eff-rank vs n_samples is the acceptance criterion; a still-rising
    curve means the rank is sample-limited and the number is not trustworthy yet.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(x_common.shape[0], generator=g)
    out: list[dict] = []
    for n in n_grid:
        if n > x_common.shape[0]:
            continue
        xi = x_common[perm[:n]]
        sp = effective_dimension(system, xi, modules=modules, rel_tol=rel_tol, svd_on_cpu=svd_on_cpu)
        out.append({"n_samples": int(n), "eff_rank": float(sp["effective_rank"]),
                    "kappa": float(sp["condition_number"]), "num_rank": int(sp["numerical_rank"]),
                    "n_params": int(sp["n_params"])})
    return out
