"""Within-model message-passing ablation (the capacity-free CTNN control).

To ask "does message-passing help?" without a parameter-count confound, we zero (or randomise) the
inter-particle transport weights of a trained CTNN *in place* and re-measure. Same parameters, only
the message-passing mechanism removed -> the CTNN collapses to a per-node/per-edge (DeepSet-like)
ansatz. This is the within-model analogue of the thesis Fig F control.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

# the CTNN-VCycle ModuleLists that carry information *between* particles
_TRANSPORT = ["rho_v_to_e_down", "rho_e_to_v_down", "rho_v_to_e_up", "rho_e_to_v_up"]


def has_messages(f_net) -> bool:
    return any(getattr(f_net, n, None) is not None for n in _TRANSPORT)


@contextlib.contextmanager
def ablate_messages(f_net, mode: str = "zero", scale: float = 0.01):
    """Temporarily zero ('zero') or randomise ('random') the CTNN inter-particle transport weights.

    Restores the originals on exit. With messages removed the readout sees only the per-node and
    per-edge self-embeddings -> a pairwise/DeepSet-equivalent of the *same* network."""
    saved = []
    for name in _TRANSPORT:
        ml = getattr(f_net, name, None)
        if ml is None:
            continue
        for lin in ml:
            saved.append((lin.weight, lin.weight.detach().clone()))
            if mode == "zero":
                lin.weight.data.zero_()
            elif mode == "random":
                lin.weight.data.normal_(0.0, scale)
            else:
                raise ValueError(mode)
    try:
        yield
    finally:
        for w, w0 in saved:
            w.data.copy_(w0)


@torch.no_grad()
def message_ablation_energy(system, *, n_samples: int = 2048, steps: int = 300, burn_in: int = 600) -> dict:
    """Proper variational energy of full vs message-ablated CTNN, each sampled from ITS OWN |Psi|^2.

    Returns full/zeroed/random energies and variances. Random==zeroed confirms only learned messages
    matter; the energy/variance gap is the message-passing contribution (no capacity confound)."""
    from . import diagnostics as dg

    def _eval():
        x = system.sample(n_samples, steps=steps, burn_in=burn_in)
        E = dg.local_energy(system.log_psi, x, system.omega, system.params, lap_mode="exact")
        E = E[torch.isfinite(E)]
        return float(E.mean()), float(E.var()), x

    e0, v0, _ = _eval()
    with ablate_messages(system.f_net, "zero"):
        ez, vz, _ = _eval()
    with ablate_messages(system.f_net, "random"):
        er, vr, _ = _eval()
    return {"E_full": e0, "var_full": v0, "E_ablate": ez, "var_ablate": vz,
            "E_random": er, "var_random": vr, "dE_ablate": ez - e0,
            "var_ratio_ablate": vz / (v0 + 1e-30)}


@torch.no_grad()
def manybody_signature(system, x: torch.Tensor, *, nb: int = 30) -> dict:
    """The >pairwise content of the CTNN, isolated by the within-model ablation.

    D(x) = log|Psi_full| - log|Psi_ablated| is exactly the many-body (beyond-pairwise) contribution,
    because the ablated CTNN is a pure pairwise/DeepSet model. We report:
      * mb_fraction = std(D)/std(log|Psi|)           -- how much of the structure is many-body
      * intra-bin var of D vs nearest-pair distance  -- does the many-body term depend on the
        environment (3+body) rather than just the pair distance? (pairwise => 0)
    """
    logf = system.log_psi(x).double()
    with ablate_messages(system.f_net, "zero"):
        loga = system.log_psi(x).double()
    D = (logf - loga).cpu().numpy()
    mb_fraction = float(np.std(D) / (np.std(logf.cpu().numpy()) + 1e-30))

    N = x.shape[1]
    ii, jj = torch.triu_indices(N, N, offset=1)
    rmin = (x[:, ii, :] - x[:, jj, :]).norm(dim=-1).min(dim=1).values.cpu().numpy()
    edges = np.quantile(rmin, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(rmin, edges[1:-1]), 0, nb - 1)
    rc, intra, dmean = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() > 5:
            rc.append(rmin[m].mean()); intra.append(float(np.var(D[m]))); dmean.append(float(np.mean(D[m])))
    tot = float(np.var(D)) + 1e-30
    return {"mb_fraction": mb_fraction, "intra_bin_var_frac": float(np.mean(intra) / tot),
            "r_centers": np.array(rc), "D_intra_var": np.array(intra), "D_mean": np.array(dmean)}
