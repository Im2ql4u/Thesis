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


def kernel_spectrum(O: torch.Tensor, *, rel_tol: float = 1e-12, gram_chunk: int = 8192) -> dict:
    """Spectrum of the quantum geometric tensor S = O^T O / B (== NTK / B spectrum).

    Returns eigenvalues (descending), effective rank (participation ratio),
    numerical rank, and condition number over the supported spectrum.
    """
    B, P = O.shape
    if B <= P:
        # Gram trick: O^T O and O O^T share their non-zero spectrum, and O O^T is only B x B.
        # svdvals(O) needs a workspace on the order of O itself (B x P) and OOM'd at N=6 already;
        # accumulating the Gram in parameter blocks keeps the peak allocation tiny.
        G = torch.zeros(B, B, dtype=torch.float64, device=O.device)
        for j in range(0, P, gram_chunk):
            blk = O[:, j : j + gram_chunk].double()
            G += blk @ blk.T
        lam = torch.linalg.eigvalsh(G).clamp_min(0.0).cpu().numpy() / B
    else:
        lam = (torch.linalg.svdvals(O.double()) ** 2 / B).cpu().numpy()
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
# Collocation conditioning: the strong/weak residual Gauss-Newton operator A = J^T J
# ----------------------------------------------------------------------
def _grad_logpsi(log_psi_fn, x: torch.Tensor):
    """grad_x log|Psi| (B, N, d), kept differentiable wrt parameters."""
    x = x.detach().requires_grad_(True)
    lp = log_psi_fn(x)  # (B,)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True)[0]
    return x, g


def _laplacian_logpsi(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Exact Laplacian sum_a d^2 log|Psi| / dx_a^2 (B,), keeping the parameter graph."""
    B = x.shape[0]
    flat = g.reshape(B, -1)
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for j in range(flat.shape[1]):
        gj = torch.autograd.grad(flat[:, j].sum(), x, create_graph=True, retain_graph=True)[0]
        lap = lap + gj.reshape(B, -1)[:, j]
    return lap


def residual_local_energy(system, x: torch.Tensor, *, form: str = "strong") -> torch.Tensor:
    """Per-sample collocation residual, differentiable wrt parameters.

      strong : R = E_L = -1/2 (lap logPsi + |grad logPsi|^2) + V   (De Ryck A=J^T J, the k^4 operator)
      weak   : R = 1/2 |grad logPsi|^2 + V                         (Rayleigh; first derivatives only)

    The least-squares loss E[(R - E)^2] is what the collocation trainer minimises; its Gauss-Newton
    operator is A = J^T J with J[k,i] = dR(x_k)/dtheta_i (see residual_jacobian)."""
    B = x.shape[0]
    x, g = _grad_logpsi(system.log_psi, x)
    trap = 0.5 * system.omega**2 * (x**2).sum(dim=(1, 2)).reshape(B)
    V = trap + compute_coulomb_interaction(x, params=system.params).reshape(B)
    g2 = (g**2).sum(dim=(1, 2)).reshape(B)
    if form == "weak":
        return 0.5 * g2 + V
    if form == "strong":
        lap = _laplacian_logpsi(x, g).reshape(B)
        return -0.5 * (lap + g2) + V
    raise ValueError(f"form must be 'strong' or 'weak', got {form!r}")


def residual_jacobian(system, x: torch.Tensor, *, form: str = "strong",
                      center: bool = False, chunk: int = 32) -> torch.Tensor:
    """J (B, P) with J[k,i] = dR(x_k)/dtheta_i for the strong/weak collocation residual.

    A = J^T J / B is the Gauss-Newton operator whose spectrum the conditioning theory predicts
    (kappa(A) ~ k^4 strong, ~ k^2 weak). Pass J to kernel_spectrum to get its spectrum/kappa.
    center=False keeps the raw Gauss-Newton operator (the learning operator); center=True gives the
    covariance form. Per-sample autograd loop (the residual already carries up to 2nd x-derivatives)."""
    params = [p for m in system.modules() for p in m.parameters()]
    rows = []
    for s in range(0, x.shape[0], chunk):
        R = residual_local_energy(system, x[s : s + chunk], form=form)  # (b,)
        b = R.shape[0]
        for k in range(b):
            g = torch.autograd.grad(R[k], params, retain_graph=(k < b - 1),
                                    create_graph=False, allow_unused=True)
            rows.append(torch.cat([
                (gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                for gi, p in zip(g, params)
            ]).detach())
    J = torch.stack(rows, dim=0)
    if center:
        J = J - J.mean(dim=0, keepdim=True)
    return J


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
    chunk: int = 1024,
) -> torch.Tensor:
    """E_L over x, chunked over the batch to bound memory (the exact Laplacian builds N*d graphs)."""
    def coul(xx):
        return compute_coulomb_interaction(xx, params=params)

    outs = []
    for s in range(0, x.shape[0], chunk):
        E_L, _ = _local_energy_multi(
            log_psi_fn, x[s : s + chunk], coul, omega, lap_mode=lap_mode, lap_probes=lap_probes
        )
        outs.append(E_L)
    return torch.cat(outs, dim=0)


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


def gradient_snr(O: torch.Tensor, E_L: torch.Tensor) -> dict:
    """Signal-to-noise of the VMC energy gradient g = 2<(E_L-E) O>.

    Adam preconditions by the *noise* (~Var of the per-sample gradient, which carries var(E_L));
    SR preconditions by the *geometry* S. A low gradient SNR is exactly what floors Adam. Returns
    ||g||, the gradient SNR, and var(E_L). Higher SNR / lower var(E_L) => easier under Adam.
    """
    Od = O.double()
    r = (E_L.detach().double() - E_L.detach().double().mean()).to(Od.device)  # (B,)
    B = Od.shape[0]
    per = Od * r[:, None]                # (B,P) per-sample gradient contributions
    g = per.mean(0)                      # (P,)
    noise = per.std(0) / np.sqrt(B)      # (P,) stderr per component
    snr = float(g.norm() / (noise.norm() + 1e-30))
    return {"g_norm": float(g.norm()), "grad_snr": snr, "var_EL": float(r.var())}


def bootstrap_ci(values: np.ndarray, stat=np.mean, *, n_boot: int = 400, ci: float = 0.95) -> dict:
    """Bootstrap CI of a statistic over a sample of per-config values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4:
        return {"mean": float(stat(v)) if n else float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(0)
    boot = np.array([stat(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(stat(v)), "lo": float(lo), "hi": float(hi), "se": float(boot.std())}


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
