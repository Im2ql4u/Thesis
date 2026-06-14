"""Kernel-picture diagnostics for trained NQS wavefunctions.

All quantities are properties of the per-sample log-derivative matrix
    O[k, i] = d log|Psi(x_k)| / d theta_i
and the local energy E_L(x_k). Implements:

  build_O                : assemble O (reuses Stochastic_Reconfiguration._score_rows)
  kernel_spectrum        : eigenvalues / effective rank / condition number of S=O^T O /B
  local_energy           : E_L (reuses _local_energy_multi)
  gs_quality             : energy vs reference, var(E_L), error
  sr_vs_plain_alignment  : how SR (natural-gradient) vs plain-gradient align with the
                           imaginary-time / Hamiltonian flow; the NTK-whitening picture
  two_body_correlation   : learned correlation log|Psi| - log|Slater| vs pair distance
  overlap_with_exact     : |<Psi_net|Psi_exact>|^2 via importance sampling

Nothing here is specialised to a particular N or omega.
"""

from __future__ import annotations

import numpy as np
import torch

from functions.Physics import compute_coulomb_interaction
from functions.Stochastic_Reconfiguration import _local_energy_multi, _score_rows


# ----------------------------------------------------------------------
# O matrix and its spectrum
# ----------------------------------------------------------------------
def build_O(
    log_psi_fn,
    x: torch.Tensor,
    modules: list,
    *,
    center: bool = True,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Per-sample score matrix O (B, P). If center, subtract the |Psi|^2-mean per param
    (the quantum geometric tensor / Fisher metric uses centred scores)."""
    O, _ = _score_rows(log_psi_fn, x, modules, chunk_size=chunk_size)
    if center:
        O = O - O.mean(dim=0, keepdim=True)
    return O


def kernel_spectrum(O: torch.Tensor, *, rel_tol: float = 1e-12) -> dict:
    """Spectrum of the quantum geometric tensor S = O^T O / B (== NTK / B spectrum).

    Returns eigenvalues (descending), effective rank (participation ratio),
    numerical rank, and condition number over the supported spectrum.
    """
    B = O.shape[0]
    s = torch.linalg.svdvals(O.double())  # singular values of O
    lam = (s**2 / B).cpu().numpy()  # eigenvalues of S
    lam = np.sort(lam)[::-1]
    lam_max = float(lam[0]) if lam.size else 0.0
    supp = lam[lam > lam_max * rel_tol]
    eff_rank = float((supp.sum() ** 2) / (supp**2).sum()) if supp.size else 0.0
    cond = float(supp[0] / supp[-1]) if supp.size else float("inf")
    return {
        "eigenvalues": lam,
        "lam_max": lam_max,
        "lam_min_supported": float(supp[-1]) if supp.size else 0.0,
        "effective_rank": eff_rank,
        "numerical_rank": int(supp.size),
        "condition_number": cond,
        "n_params": int(O.shape[1]),
        "n_samples": B,
    }


# ----------------------------------------------------------------------
# Local energy and ground-state quality
# ----------------------------------------------------------------------
def local_energy(
    log_psi_fn,
    x: torch.Tensor,
    omega: float,
    params: dict,
    *,
    lap_mode: str = "exact",
    lap_probes: int = 16,
) -> torch.Tensor:
    def coul(xx):
        return compute_coulomb_interaction(xx, params=params)

    E_L, _ = _local_energy_multi(
        log_psi_fn, x, coul, omega, lap_mode=lap_mode, lap_probes=lap_probes
    )
    return E_L


def _blocking_stderr(v: np.ndarray, n_blocks: int = 32) -> float:
    """Standard error from block means (mitigates MCMC autocorrelation)."""
    n = (v.size // n_blocks) * n_blocks
    if n < n_blocks:
        return float(v.std(ddof=1) / max(1, np.sqrt(v.size)))
    means = v[:n].reshape(n_blocks, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_blocks))


def gs_quality(
    E_L: torch.Tensor,
    *,
    ref_energy: float | None = None,
    clip_mad: float = 8.0,
) -> dict:
    """Energy, variance, blocked stderr, and error vs a reference energy."""
    e = E_L.detach().cpu().double().numpy()
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-30
    e_cl = np.clip(e, med - clip_mad * mad, med + clip_mad * mad)
    E_mean = float(e_cl.mean())
    var = float(e_cl.var(ddof=1))
    se = _blocking_stderr(e_cl)
    out = {
        "E_mean": E_mean,            # clipped (stable) mean
        "E_mean_raw": float(e.mean()),  # unclipped mean (unbiased but higher variance)
        "E_stderr": se,
        "var_EL": var,
        "n_samples": int(e.size),
        "frac_clipped": float(np.mean(e != e_cl)),
    }
    if ref_energy is not None and np.isfinite(ref_energy):
        out["ref_energy"] = float(ref_energy)
        out["error_pct"] = float((E_mean - ref_energy) / abs(ref_energy) * 100.0)
        out["error_sigma"] = float((E_mean - ref_energy) / se) if se > 0 else float("nan")
    return out


# ----------------------------------------------------------------------
# SR vs plain gradient: alignment with the imaginary-time (Hamiltonian) flow
# ----------------------------------------------------------------------
def sr_vs_plain_alignment(O: torch.Tensor, E_L: torch.Tensor, *, rel_tol: float = 1e-10) -> dict:
    """Compare natural-gradient (SR) and plain-gradient directions to the
    imaginary-time target in *function* space.

    Imaginary-time evolution moves the wavefunction by delta log|Psi|(x) ~ -(E_L(x) - <E_L>).
    Both optimisers produce a function-space move that is a linear combination of the
    tangent functions O[:, i]:
      * plain gradient  ->  apply the NTK operator K = O O^T to the residual r
      * SR / nat. grad  ->  orthogonally PROJECT r onto the tangent space (NTK-whitened)
    The SR move equals the best tangent-space approximation of the imaginary-time flow.

    Returns alignment cosines, the representable fraction, and per-mode weight
    profiles that expose the NTK-whitening (SR weights all supported modes equally;
    plain gradient reweights mode a by its eigenvalue mu_a).
    """
    Od = O.double()
    r = (E_L.detach().double() - E_L.detach().double().mean()).to(Od.device)  # (B,)
    K = Od @ Od.t()  # (B,B) NTK Gram (parameter sum)
    mu, V = torch.linalg.eigh(K)  # ascending
    mu = torch.clamp(mu, min=0.0)
    mu_max = float(mu[-1]) if mu.numel() else 0.0
    supp = mu > mu_max * rel_tol  # supported (tangent) modes

    c = V.t() @ r  # residual coefficients in NTK eigenbasis (B,)
    r_norm = float(r.norm()) + 1e-30

    # SR move = projection of r onto supported eigenvectors
    psi_sr = (V[:, supp] @ c[supp]) if supp.any() else torch.zeros_like(r)
    rep_fraction = float(psi_sr.norm() / r_norm)  # cos(delta_SR, r) = ||P r||/||r||

    # plain move = K r
    psi_plain = K @ r
    cos_plain = float((r @ psi_plain) / (r_norm * (psi_plain.norm() + 1e-30)))

    mu_np = mu.flip(0).cpu().numpy()  # descending
    c2 = (c.flip(0).cpu().numpy()) ** 2  # residual power per mode (descending mu)
    supp_n = int(supp.sum().item())
    eff_rank = float((mu_np[mu_np > mu_max * rel_tol].sum() ** 2)
                     / (mu_np[mu_np > mu_max * rel_tol] ** 2).sum()) if supp_n else 0.0
    return {
        "cos_sr": rep_fraction,          # SR alignment with imaginary-time flow
        "cos_plain": cos_plain,          # plain-gradient alignment
        "rep_fraction": rep_fraction,    # fraction of imag-time direction representable
        "ntk_condition": float(mu_np[0] / mu_np[supp_n - 1]) if supp_n else float("inf"),
        "ntk_eff_rank": eff_rank,
        "ntk_numerical_rank": supp_n,
        "mu_desc": mu_np,                # NTK eigenvalues (descending)
        "residual_power_desc": c2,       # |<v_a, r>|^2 per mode (descending mu)
        "sr_mode_weight": np.where(mu_np > mu_max * rel_tol, 1.0, 0.0),
        "plain_mode_weight": mu_np / (mu_max + 1e-30),
    }


# ----------------------------------------------------------------------
# What the network learned: two-body correlation and exact overlap
# ----------------------------------------------------------------------
@torch.no_grad()
def two_body_correlation(log_psi_fn, log_slater_fn, x: torch.Tensor) -> dict:
    """Learned correlation factor J(x) = log|Psi| - log|Slater_core| and pair distances.

    For N=2 the correlation depends only on the single pair distance, so (r12, J) is
    an exact read-out of the learned Jastrow. For N>2, J is the total correlation
    (binning by individual pair distance is only approximate)."""
    logpsi = log_psi_fn(x).detach()
    logsd = log_slater_fn(x).detach()
    J = (logpsi - logsd).cpu().double().numpy()
    B, N, _ = x.shape
    ii, jj = torch.triu_indices(N, N, offset=1)
    r12 = (x[:, ii, :] - x[:, jj, :]).norm(dim=-1).cpu().double().numpy()  # (B, n_pairs)
    return {"r12": r12, "J": J, "n_pairs": int(ii.numel())}


def zero_variance_extrapolation(E: np.ndarray, var: np.ndarray, *, max_points: int = 6) -> dict:
    """Zero-variance extrapolation E(var->0).

    Near a variational optimum the energy estimate and var(E_L) are linearly
    related; the intercept of E vs var is the bias-reduced energy estimate. Fits a
    line through the lowest-variance trajectory points. Returns the intercept and slope.
    """
    E = np.asarray(E, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    m = np.isfinite(E) & np.isfinite(var)
    E, var = E[m], var[m]
    if E.size < 2:
        return {"E_zv": float(E[-1]) if E.size else float("nan"), "slope": 0.0, "n_points": int(E.size)}
    order = np.argsort(var)
    k = min(max_points, E.size)
    vv, ee = var[order][:k], E[order][:k]
    A = np.vstack([vv, np.ones_like(vv)]).T
    slope, intercept = np.linalg.lstsq(A, ee, rcond=None)[0]
    return {"E_zv": float(intercept), "slope": float(slope), "n_points": int(k)}


@torch.no_grad()
def overlap_with_exact(log_psi_fn, exact_log_psi_np, x: torch.Tensor) -> dict:
    """|<Psi_net|Psi_exact>|^2 estimated from samples x ~ |Psi_net|^2.

    With ratio = Psi_exact/Psi_net, overlap^2 = <ratio>^2 / <ratio^2>. Valid for the
    nodeless N=2 singlet (both wavefunctions positive)."""
    lognet = log_psi_fn(x).detach().cpu().double().numpy()
    logex = exact_log_psi_np(x.detach().cpu().double().numpy())
    d = logex - lognet
    d = d - d.max()  # stabilise
    ratio = np.exp(d)
    num = ratio.mean() ** 2
    den = (ratio**2).mean()
    return {"overlap_sq": float(num / den), "n_samples": int(x.shape[0])}
