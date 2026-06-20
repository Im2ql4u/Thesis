"""Physics-extraction probes (Angle 3): read physical content out of a trained wavefunction.

  natural_orbital_occupations : 1-body reduced density matrix -> NO occupations (correlation spread)
  intrinsic_dimension         : nonlinear ID (TwoNN) of latent representations
  rotational_invariance       : is |Psi|^2 rotationally symmetric (L_z=0 ground state)?
  pair_correlation            : g(r), and learned correlation u(r) with its cusp slope
  internal_order_params       : latent observables for a network-internal (Wigner) phase diagram
"""

from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------
# A3.1 natural-orbital occupations from the 1-RDM (grid estimator, particle 0)
# ----------------------------------------------------------------------
@torch.no_grad()
def natural_orbital_occupations(system, x: torch.Tensor, *, grid_half: float | None = None,
                                n_grid: int = 22, chunk: int = 8192) -> dict:
    """Spin-up natural-orbital occupations, 1-RDM projected onto the orthonormal HO basis.

    rho_pq = N_up < phi_p(r1) * integral phi_q(r') Psi(r',rest)/Psi(r1,rest) dr' >_{|Psi|^2},
    with r1 a spin-up electron and {phi} the 2D HO Cartesian orbitals (orthonormal => eigenvalues
    of rho are occupations, sum = N_up). Spread below the HF integer pattern measures correlation.
    Validated at N=2 (one up electron -> leading occupation ~ 1)."""
    from functions.Slater_Determinant import evaluate_basis_functions_torch_batch_2d

    B, N, d = x.shape
    dev = x.device
    nx, ny = int(system.params["nx"]), int(system.params["ny"])
    ell = 1.0 / np.sqrt(system.omega)
    L = grid_half if grid_half else 4.0 * ell
    ax = torch.linspace(-L, L, n_grid, device=dev, dtype=x.dtype)
    gx, gy = torch.meshgrid(ax, ax, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (G,2)
    G = grid.shape[0]
    dA = (2 * L / (n_grid - 1)) ** 2
    n_up = N // 2

    # HO orbitals at grid and at the sampled r1 (spin-up electron 0)
    phi_grid = evaluate_basis_functions_torch_batch_2d(
        grid.view(1, G, 2), nx, ny, params=system.params)[0].double()  # (G, P)
    P = phi_grid.shape[1]
    log0 = system.log_psi(x).double()
    r1 = x[:, 0, :]
    phi_r1 = evaluate_basis_functions_torch_batch_2d(
        r1.view(1, B, 2), nx, ny, params=system.params)[0].double()  # (B, P)

    rho = torch.zeros(P, P, dtype=torch.float64, device=dev)
    for b0 in range(0, B, 32):
        xb = x[b0 : b0 + 32]; kb = xb.shape[0]
        lb0 = log0[b0 : b0 + 32]
        cfg = xb.unsqueeze(1).expand(kb, G, N, d).clone()
        cfg[:, :, 0, :] = grid.view(1, G, d).expand(kb, G, d)
        cfg = cfg.reshape(kb * G, N, d)
        logp = torch.empty(kb * G, device=dev, dtype=torch.float64)
        for s in range(0, cfg.shape[0], chunk):
            logp[s : s + chunk] = system.log_psi(cfg[s : s + chunk]).double()
        ratio = torch.exp(logp.reshape(kb, G) - lb0[:, None])  # (kb,G)
        A = (ratio @ phi_grid) * dA  # (kb, P)  = integral phi_q(r') ratio dr'
        rho += phi_r1[b0 : b0 + 32].t() @ A  # (P,P)
    rho = (n_up / B) * rho
    rho = 0.5 * (rho + rho.t())
    # correct finite-grid basis non-orthonormality: solve rho x = n S x  (S = grid overlap)
    S = (phi_grid.t() @ phi_grid) * dA
    es, Us = torch.linalg.eigh(S)
    es = torch.clamp(es, min=float(es.max()) * 1e-6)
    Sih = Us @ torch.diag(es.rsqrt()) @ Us.t()
    rho = Sih @ rho @ Sih
    rho = 0.5 * (rho + rho.t())
    evals = torch.linalg.eigvalsh(rho).cpu().double().numpy()[::-1]
    evals = np.clip(evals, 0, None)
    return {"occupations": evals[: max(4, 2 * n_up)], "n_up": n_up, "trace": float(evals.sum()),
            "leading_occ": float(evals[0]),
            "occ_entropy": float(_entropy(evals[evals > 1e-6] / max(evals.sum(), 1e-12)))}


def _entropy(p):
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


# ----------------------------------------------------------------------
# A3.2 intrinsic dimension (TwoNN)
# ----------------------------------------------------------------------
def intrinsic_dimension(feat: np.ndarray) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al.) on feature rows."""
    X = np.asarray(feat, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    n = X.shape[0]
    if n < 10:
        return float("nan")
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(np.sort(d2, axis=1)[:, :2])  # 1st, 2nd NN
    r1, r2 = dist[:, 0], dist[:, 1]
    ok = (r1 > 1e-12)
    mu = r2[ok] / r1[ok]
    mu = mu[mu > 1.0]
    # MLE: d = (n) / sum(log mu)
    return float(mu.shape[0] / np.log(mu).sum())


# ----------------------------------------------------------------------
# A3.4 rotational invariance (L_z = 0 ground state)
# ----------------------------------------------------------------------
@torch.no_grad()
def rotational_invariance(system, x: torch.Tensor, *, n_angles: int = 8) -> dict:
    """std over global rotations of (log|Psi|(R x) - log|Psi|(x)); ~0 for an L_z=0 state."""
    log0 = system.log_psi(x).double()
    diffs = []
    for k in range(1, n_angles + 1):
        th = 2 * np.pi * k / (n_angles + 1)
        c, s = np.cos(th), np.sin(th)
        Rm = torch.tensor([[c, -s], [s, c]], device=x.device, dtype=x.dtype)
        xr = x @ Rm.t()
        diffs.append((system.log_psi(xr).double() - log0))
    D = torch.stack(diffs)  # (A,B)
    return {"rot_logpsi_std": float(D.std()), "rot_logpsi_absmean": float(D.abs().mean())}


# ----------------------------------------------------------------------
# A3.5 pair correlation g(r) and learned correlation u(r) + cusp
# ----------------------------------------------------------------------
@torch.no_grad()
def pair_correlation(system, x: torch.Tensor, *, nb: int = 40) -> dict:
    """g(r): pair-distance distribution. u(r): mean (log|Psi|-log|Slater|) vs pair distance; cusp
    slope du/dr near 0."""
    B, N, _ = x.shape
    ii, jj = torch.triu_indices(N, N, offset=1)
    r = (x[:, ii, :] - x[:, jj, :]).norm(dim=-1).reshape(-1).cpu().double().numpy()
    J = (system.log_psi(x) - system.log_slater(x)).double().cpu().numpy()
    Jrep = np.repeat(J, ii.numel())
    edges = np.quantile(r, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(r, edges[1:-1]), 0, nb - 1)
    rc, gr, uc = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() > 3:
            rc.append(r[m].mean()); gr.append(int(m.sum())); uc.append(Jrep[m].mean())
    rc = np.array(rc); gr = np.array(gr, float); uc = np.array(uc)
    cusp = float((uc[1] - uc[0]) / (rc[1] - rc[0])) if rc.size > 1 else float("nan")
    return {"r": rc, "g_counts": gr, "u": uc, "cusp_slope": cusp,
            "r_mean": float(r.mean()), "r_min_mean": float(np.quantile(r, 0.02))}


# ----------------------------------------------------------------------
# A3.3 internal order parameters for a network phase diagram
# ----------------------------------------------------------------------
@torch.no_grad()
def internal_order_params(system, x: torch.Tensor) -> dict:
    """Latent observables that may track the Wigner crossover: backflow displacement magnitude,
    radial localisation (Lindemann-like), and density width."""
    out = {}
    ell = 1.0 / np.sqrt(system.omega)
    rad = x.norm(dim=-1)  # (B,N)
    out["radial_mean_aho"] = float((rad.mean() / ell))
    out["radial_relwidth"] = float(rad.std() / (rad.mean() + 1e-30))  # Lindemann-like
    if system.backflow_net is not None:
        dx = system.backflow_net(x, spin=system.spin)
        out["bf_disp_aho"] = float((dx.norm(dim=-1).mean() / ell))
    return out
