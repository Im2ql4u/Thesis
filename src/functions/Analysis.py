# analysis_clean.py — EXACT-only, self-contained, backflow-aware analysis
# Paste this whole file. It defines every helper it uses (no NameErrors).
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm


# ---------------------------
# Wrap your psi_fn into log|Ψ|
# ---------------------------
def make_psi_log_fn(psi_fn, f_net, C_occ, *, backflow_net=None, spin=None, params=None):
    """
    Returns psi_log_fn(x)->(B,). Always ensures x.requires_grad_(True) to satisfy psi_fn's assert.
    """

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi.view(-1)

    return psi_log_fn


# ---------------------------
# Initializer (HO Gaussian)
# ---------------------------
@torch.no_grad()
def init_positions_gaussian(
    B: int, N: int, d: int, omega: float, *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """
    HO ground-state |ψ0|^2 ∝ exp(-ω r^2) => per-coordinate N(0, σ^2), σ^2=1/(2ω).
    """
    sigma = (2.0 * float(omega)) ** -0.5
    return torch.randn(B, N, d, device=device, dtype=dtype) * sigma


# ---------------------------------------
# Random-Walk Metropolis kernel for |Ψ|²
# ---------------------------------------
@torch.no_grad()
def mh_rw_step(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    step_sigma: float,
) -> tuple[torch.Tensor, float]:
    """
    One RW-MH step for all walkers (vectorized).
    x: (B,N,d), returns (x_next, accept_rate_this_step)
    """
    B = x.shape[0]
    # Current log-density: log p(x) = 2 * log|Ψ(x)|
    lp = psi_log_fn(x) * 2.0  # (B,)
    lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e30))

    # Propose
    prop = x + step_sigma * torch.randn_like(x)
    lp_prop = psi_log_fn(prop) * 2.0
    lp_prop = torch.where(torch.isfinite(lp_prop), lp_prop, torch.full_like(lp_prop, -1e30))

    # Accept / reject in log space
    logu = torch.log(torch.rand_like(lp))
    accept = (logu < (lp_prop - lp)).view(B, 1, 1)
    x_next = torch.where(accept, prop, x)

    acc_rate = float(accept.float().mean().item())
    return x_next, acc_rate


# ---------------------------------------
# MALA (Langevin) kernel for |Ψ|² (opt.)
# ---------------------------------------
def _grad_logp(psi_log_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    ∇_x log p(x) = ∇_x (2 log|Ψ(x)|)
    Returns (B,N,d)
    """
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        lp = psi_log_fn(x) * 2.0
        (gx,) = torch.autograd.grad(lp.sum(), x, create_graph=False)
    gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
    return gx


@torch.no_grad()
def mh_mala_step(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    step_sigma: float,
) -> tuple[torch.Tensor, float]:
    """
    One MALA step with preconditioner = I. Proposal:
      x' = x + (step_sigma^2 / 2) * ∇ log p(x) + step_sigma * ξ
    Includes the MALA acceptance correction.
    """
    # Drift at current x
    g = _grad_logp(psi_log_fn, x)  # (B,N,d)
    drift = 0.5 * (step_sigma**2) * g
    noise = step_sigma * torch.randn_like(x)
    prop = x + drift + noise

    # Drift at proposal (for reverse density)
    gp = _grad_logp(psi_log_fn, prop)
    drift_p = 0.5 * (step_sigma**2) * gp

    # MH ratio with Gaussian proposals
    lp = psi_log_fn(x) * 2.0
    lp_prop = psi_log_fn(prop) * 2.0
    lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e30))
    lp_prop = torch.where(torch.isfinite(lp_prop), lp_prop, torch.full_like(lp_prop, -1e30))

    def gaussian_log_q(x_from, x_to, drift_vec):
        # q(x_to | x_from) = N(x_from + drift_vec, step_sigma^2 I)
        diff = x_to - (x_from + drift_vec)
        return -0.5 * (diff.square().sum(dim=(1, 2)) / (step_sigma**2))

    log_q_prop_given_x = gaussian_log_q(x, prop, drift)  # (B,)
    log_q_x_given_prop = gaussian_log_q(prop, x, drift_p)  # (B,)

    log_alpha = (lp_prop + log_q_x_given_prop) - (lp + log_q_prop_given_x)
    logu = torch.log(torch.rand_like(log_alpha))
    accept = (logu < log_alpha).view(-1, 1, 1)

    x_next = torch.where(accept, prop, x)
    acc_rate = float(accept.float().mean().item())
    return x_next, acc_rate


# ---------------------------------------------------------
# Step-size autotuning (short pilot) toward target accept
# ---------------------------------------------------------
@torch.no_grad()
def autotune_step_sigma(
    kernel: Callable[
        [Callable[[torch.Tensor], torch.Tensor], torch.Tensor, float], tuple[torch.Tensor, float]
    ],
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    step_sigma: float,
    *,
    target_accept: float,
    iters: int = 25,
    k_p: float = 0.5,
    min_sigma: float = 1e-4,
    max_sigma: float = 5.0,
) -> tuple[torch.Tensor, float]:
    """
    Simple proportional controller to hit target_accept.
    Returns (x, tuned_step_sigma).
    """
    x = x0
    s = step_sigma
    for _ in range(iters):
        x, acc = kernel(psi_log_fn, x, s)
        # multiplicative update
        s = s * (1.0 + k_p * (acc - target_accept))
        s = float(max(min_sigma, min(max_sigma, s)))
    return x, s


# ---------------------------------------------------------
# One-shot helper: get a full batch X ~ |Ψ|²
# ---------------------------------------------------------
@torch.no_grad()
def sample_psi2_batch(
    psi_log_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    B: int,
    N: int,
    d: int,
    omega: float,
    device: torch.device | str,
    dtype: torch.dtype,
    method: str = "rw",  # "rw" | "mala"
    step_sigma: float = 0.2,
    burn_in: int = 50,
    mix_steps: int = 30,
    autotune: bool = True,
    target_accept: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns:
      X         : (B,N,d) sampled approx. from |Ψ|^2
      acc_burn  : mean accept rate during burn-in/autotune
      acc_mix   : mean accept rate during final mixing
    Notes:
      - Uses B independent walkers; returns their final states.
      - Set method="rw" for Random-Walk Metropolis (robust), or "mala" for Langevin.
      - For RW in moderate dims, a target accept ~0.4–0.6 is fine (default 0.5).
        For MALA, ~0.57 is classic; use 0.5–0.6 in practice.
    """
    # choose kernel + default target
    if method == "rw":
        kernel = mh_rw_step
        if target_accept is None:
            target_accept = 0.5
    elif method == "mala":
        kernel = mh_mala_step
        if target_accept is None:
            target_accept = 0.57
    else:
        raise ValueError("method must be 'rw' or 'mala'")

    # init walkers
    X = init_positions_gaussian(B, N, d, omega, device=device, dtype=dtype)

    # optional autotune (short)
    acc_hist = []
    if autotune and burn_in > 0:
        X, step_sigma = autotune_step_sigma(
            kernel, psi_log_fn, X, step_sigma, target_accept=target_accept, iters=min(burn_in, 50)
        )
        acc_hist.append(target_accept)  # approximate
        acc_burn = target_accept
        # Finish remaining burn-in steps (if any) at tuned step size
        extra = max(0, burn_in - 50)
        accepts = []
        for _ in range(extra):
            X, acc = kernel(psi_log_fn, X, step_sigma)
            accepts.append(acc)
        if accepts:
            acc_burn = float(sum(accepts) / len(accepts))
    else:
        # plain burn-in
        accepts = []
        for _ in range(burn_in):
            X, acc = kernel(psi_log_fn, X, step_sigma)
            accepts.append(acc)
        acc_burn = float(sum(accepts) / max(1, len(accepts)))

    # short mixing to decorrelate
    accepts = []
    for _ in tqdm(range(mix_steps)):
        X, acc = kernel(psi_log_fn, X, step_sigma)
        accepts.append(acc)
    acc_mix = float(sum(accepts) / max(1, len(accepts)))

    # Return with requires_grad=True (useful for downstream autograd)
    X = X.detach().requires_grad_(True)
    return X, acc_burn, acc_mix


# --- define compute_local_energy_batch (with SR fallbacks) ---
import torch


def _hutch_lap_logpsi(psi_log_fn, x, probes=8, fd_eps=1e-4):
    x = x.detach().requires_grad_(True)
    logpsi = psi_log_fn(x)
    g = torch.autograd.grad(logpsi.sum(), x, create_graph=True)[0]  # (B,N,d)
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        try:
            hv = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=True, create_graph=False)[0]
        except RuntimeError:
            # FD fallback
            xp = (x + fd_eps * v).detach().requires_grad_(True)
            xm = (x - fd_eps * v).detach().requires_grad_(True)
            gp = torch.autograd.grad(psi_log_fn(xp).sum(), xp, create_graph=False)[0]
            gm = torch.autograd.grad(psi_log_fn(xm).sum(), xm, create_graph=False)[0]
            hv = (gp - gm) / (2.0 * fd_eps)
        hv = torch.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0)
        acc += (v * hv).sum(dim=(1, 2))
    lap_log = acc / max(1, probes)  # (B,)
    return g, lap_log


def compute_local_energy_batch(
    mode: str,  # "hvp" | "fd" | "exact"
    psi_log_fn,
    psi_fn,
    f_net: nn.Module,
    C_occ: torch.Tensor,
    x: torch.Tensor,
    compute_coulomb_interaction,
    omega: float,
    *,
    backflow_net=None,
    spin=None,
    params=None,
    hvp_probes: int = 10,
    fd_eps: float = 1e-4,
    fd_probes: int = 6,
):
    # Try to use your training_sr implementations first
    try:
        from training_sr import _local_energy_exact as _lee
        from training_sr import _local_energy_fd as _lef
        from training_sr import _local_energy_hvp as _leh

        if mode == "hvp":
            E, _ = _leh(
                psi_log_fn, x, compute_coulomb_interaction, omega, probes=hvp_probes, fd_eps=fd_eps
            )
            return E
        if mode == "fd":
            E, _ = _lef(
                psi_log_fn, x, compute_coulomb_interaction, omega, probes=fd_probes, eps=fd_eps
            )
            return E
        if mode == "exact":
            E, _ = _lee(
                psi_fn,
                f_net,
                C_occ,
                x,
                compute_coulomb_interaction,
                omega,
                backflow_net=backflow_net,
                spin=spin,
                params=params,
            )
            return E
    except Exception:
        pass  # fall through to generic versions below

    # --- Generic fallbacks (no training_sr needed) ---
    x = x.detach().requires_grad_(True)

    if mode in ("hvp", "fd"):
        g, lap_log = _hutch_lap_logpsi(psi_log_fn, x, probes=hvp_probes, fd_eps=fd_eps)
        g2 = (g**2).sum(dim=(1, 2))
        V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))
        V_int = compute_coulomb_interaction(x).view(-1)
        return -0.5 * (lap_log + g2) + (V_harm + V_int)

    if mode == "exact":
        # Δψ/ψ with nested autograd (expensive)
        logpsi, psi = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        psi = psi.view(-1, 1)
        grads = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        B, N, d = x.shape
        lap = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(N):
            for j in range(d):
                gij = grads[:, i, j]
                second = torch.autograd.grad(gij.sum(), x, retain_graph=True)[0]
                lap += second[:, i, j]
        psi_safe = psi.clamp_min(1e-30)
        delta_over_psi = (lap / psi_safe.view(-1)).view(-1)
        V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))
        V_int = compute_coulomb_interaction(x).view(-1)
        return -0.5 * delta_over_psi + (V_harm + V_int)

    raise ValueError("mode must be 'hvp' | 'fd' | 'exact'")


# =============================================================================
# 1) CORE TAPS (unchanged API; relies on f_net internals)
# =============================================================================


def forward_taps(
    f_net: nn.Module,
    x: torch.Tensor,
    spin: torch.Tensor | None = None,
    *,
    track_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Mirrors PINN.forward but exposes intermediates for inspection.
    If track_grad=True, builds a graph so outputs are differentiable w.r.t. x.
    """
    B, N, d = x.shape
    assert N == f_net.n_particles and d == f_net.d

    with torch.set_grad_enabled(track_grad):
        # ===== coords =====
        x_scaled = x * (f_net.omega**0.5)

        diff = x_scaled.unsqueeze(2) - x_scaled.unsqueeze(1)
        diff_pairs = diff[:, f_net.idx_i, f_net.idx_j, :]  # (B,P,d)
        r2 = (diff_pairs * diff_pairs).sum(dim=-1, keepdim=True)  # (B,P,1)
        r = torch.sqrt(r2 + torch.finfo(x.dtype).eps)  # (B,P,1)
        P = r.shape[1]

        # ===== φ branch =====
        phi_flat = x_scaled.reshape(B * N, d)
        phi_out = f_net.phi(phi_flat).reshape(B, N, f_net.dL)
        phi_mean = phi_out.mean(dim=1)  # (B,dL)

        # ===== ψ branch =====
        psi_in, s1_mean = f_net._safe_pair_features(r)  # (B,P,psi_in_dim), (B,1)
        psi_out = f_net.psi(psi_in.reshape(-1, f_net.psi_in_dim)).reshape(B, P, f_net.dL)

        # short-range gate

        gate = f_net._short_range_gate(r)  # (B,P,1)
        psi_out = psi_out * gate

        # pooling
        if getattr(f_net, "use_pair_attn", False):
            attn_rc = torch.as_tensor(
                getattr(f_net, "attn_rc", 1.0), dtype=x.dtype, device=x.device
            ).clamp_min(1e-4)
            attn_p = torch.as_tensor(
                getattr(f_net, "attn_p", 2.0), dtype=x.dtype, device=x.device
            ).clamp_min(1.0)
            w = torch.exp(-((r / attn_rc) ** attn_p)).squeeze(-1)  # (B,P)

            i_idx = f_net.idx_i
            den = torch.zeros(B, N, dtype=x.dtype, device=x.device)
            den.scatter_add_(1, i_idx.unsqueeze(0).expand(B, -1), w)
            den = den.index_select(1, i_idx).unsqueeze(-1) + 1e-12
            w = (w / den.squeeze(-1)).unsqueeze(-1)  # (B,P,1)

            g = torch.zeros(B, N, f_net.dL, dtype=psi_out.dtype, device=x.device)
            g.index_add_(1, i_idx, w * psi_out)  # (B,N,dL)
            psi_mean = g.mean(dim=1)  # (B,dL)
        else:
            psi_mean = psi_out.mean(dim=1)  # (B,dL)

        # ===== extras (size 2) =====
        r2_mean = (x_scaled**2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # (B,1)
        extras = torch.cat([r2_mean, s1_mean], dim=1)  # (B,2)

        # ===== readout =====
        rho_in = torch.cat([phi_mean, psi_mean, extras], dim=1)  # (B,2*dL+2)
        out_base = f_net.rho(rho_in)  # (B,1)

        # ===== cusp term =====
        if spin is None:
            up = N // 2
            spin = (
                torch.cat(
                    [
                        torch.zeros(up, dtype=torch.long, device=x.device),
                        torch.ones(N - up, dtype=torch.long, device=x.device),
                    ],
                    dim=0,
                )
                .unsqueeze(0)
                .expand(B, -1)
            )  # (B,N)
        else:
            if spin.dim() == 1:
                spin = spin.to(x.device).long().unsqueeze(0).expand(B, -1)
            elif spin.dim() == 2:
                spin = spin.to(x.device).long()
                if spin.shape != (B, N):
                    raise ValueError(f"spin shape {tuple(spin.shape)} != (B,N)=({B},{N})")
            else:
                raise ValueError("spin must be (N,) or (B,N)")

        si = spin[:, f_net.idx_i]
        sj = spin[:, f_net.idx_j]
        same_spin = (si == sj).to(x.dtype).unsqueeze(-1)  # (B,P,1)

        gamma_para = torch.as_tensor(f_net.gamma_para, dtype=x.dtype, device=x.device).view(1, 1, 1)
        gamma_apara = torch.as_tensor(f_net.gamma_apara, dtype=x.dtype, device=x.device).view(
            1, 1, 1
        )
        gamma = same_spin * gamma_para + (1.0 - same_spin) * gamma_apara  # (B,P,1)

        ell = torch.as_tensor(f_net.cusp_len, dtype=x.dtype, device=x.device).view(1, 1, 1)
        pair_u = gamma * r * torch.exp(-r)  # (B,P,1)
        cusp_sum = pair_u.sum(dim=1)  # (B,1)

        out = out_base + cusp_sum

    return dict(
        out=out,
        out_wo_cusp=out_base,
        cusp=cusp_sum,
        phi_mean=phi_mean,
        psi_mean=psi_mean,
        extras=extras,
        rho_in=rho_in,
        psi_in=psi_in,
        r=r,
        same_spin=same_spin.squeeze(-1),
    )


# =============================================================================
# 2) LIGHTWEIGHT DIAGNOSTICS
# =============================================================================


def analyze_rho_weights(
    f_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """
    Local (batch-averaged) gradient importances for a non-linear ρ:
      w_eff_k = ⟨ |∂ρ/∂Z_k| ⟩_batch
      importance_k = w_eff_k * std(Z_k)
    where Z = rho_in.
    """
    f_net.eval()
    taps = forward_taps(f_net, X, spin, track_grad=False)
    Z = taps["rho_in"].detach()  # (B, F)
    Z_leaf = Z.requires_grad_(True)
    y = f_net.rho(Z_leaf)  # (B, 1)
    (gZ,) = torch.autograd.grad(y.sum(), Z_leaf)  # (B, F)
    w_eff = gZ.abs().mean(dim=0)  # (F,)
    std = Z.std(dim=0) + 1e-12  # (F,)
    z_imp = std * w_eff
    return dict(std=std.detach(), weight=w_eff.detach(), std_weight_importance=z_imp.detach())


@torch.no_grad()
def feature_svd(
    f_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """
    PCA/SVD of concatenated features [φ_mean | ψ_mean | extras].
    Returns singular values, explained variance, and entropy effective rank.
    """
    taps = forward_taps(f_net, X, spin)
    Fmat = torch.cat([taps["phi_mean"], taps["psi_mean"], taps["extras"]], dim=1)  # (B, 2*dL+2)
    Fc = Fmat - Fmat.mean(dim=0, keepdim=True)
    C = (Fc.T @ Fc) / (Fc.shape[0] - 1) + 1e-12 * torch.eye(
        Fc.shape[1], device=Fmat.device, dtype=Fmat.dtype
    )
    evals, _ = torch.linalg.eigh(C)  # ascending
    evals = torch.flip(evals, dims=[0])
    svals = evals.sqrt()
    expvar = evals / (evals.sum() + 1e-12)
    eff_rank = torch.exp(-(expvar * (expvar.clamp_min(1e-12)).log()).sum())
    return dict(singular_values=svals, explained_variance=expvar, effective_rank=eff_rank)


@torch.no_grad()
def cusp_vs_residual_means(
    f_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None
) -> dict[str, float]:
    taps = forward_taps(f_net, X, spin)
    out, base, cusp = taps["out"], taps["out_wo_cusp"], taps["cusp"]
    return dict(
        mean_out=float(out.mean().item()),
        mean_cusp=float(cusp.mean().item()),
        mean_base=float(base.mean().item()),
    )


def pearson_corr(y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """y: (B,), X: (B,F) -> (F,)"""
    y = y.view(-1)
    y_c = y - y.mean()
    X_c = X - X.mean(dim=0, keepdim=True)
    num = (X_c * y_c.view(-1, 1)).sum(dim=0)
    den = X_c.square().sum(dim=0).sqrt() * y_c.square().sum().sqrt() + 1e-12
    return num / den


def energy_feature_correlations(
    taps: dict[str, torch.Tensor], E_L: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Correlate local energy with [φ_mean | ψ_mean | extras] and with cusp & residual.
    """
    Fmat = torch.cat([taps["phi_mean"], taps["psi_mean"], taps["extras"]], dim=1)  # (B, 2*dL+2)
    corr_F = pearson_corr(E_L, Fmat)
    base = taps["out_wo_cusp"].view(-1)
    cusp = taps["cusp"].view(-1)
    c_base = float(torch.corrcoef(torch.stack([E_L.view(-1), base]))[0, 1])
    c_cusp = float(torch.corrcoef(torch.stack([E_L.view(-1), cusp]))[0, 1])
    return {
        "corr_features": corr_F,
        "corr_out_base": torch.tensor(c_base),
        "corr_cusp": torch.tensor(c_cusp),
    }


# =============================================================================
# 3) PCA ON ρ-IN, PC ABLATION, PROBES
# =============================================================================


@torch.no_grad()
def pca_on_rho_in(
    f_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """PCA of Z = rho_in: returns PCs U, eigenvalues, explained variance, mean, eff_rank."""
    taps = forward_taps(f_net, X, spin)
    Z = taps["rho_in"]  # (B, F)
    mu = Z.mean(dim=0, keepdim=True)
    Zc = Z - mu
    C = (Zc.T @ Zc) / (Zc.shape[0] - 1) + 1e-12 * torch.eye(
        Zc.shape[1], device=Z.device, dtype=Z.dtype
    )
    evals, U = torch.linalg.eigh(C)  # ascending
    evals = torch.flip(evals, dims=[0])  # descending
    U = torch.flip(U, dims=[1])
    expvar = evals / (evals.sum() + 1e-12)
    eff_rank = torch.exp(-(expvar * (expvar.clamp_min(1e-12)).log()).sum())
    return dict(U=U, evals=evals, expvar=expvar, mu=mu, eff_rank=eff_rank)


@torch.no_grad()
def pc_projection_ablation(
    f_net: nn.Module,
    X: torch.Tensor,
    spin: torch.Tensor | None,
    ks: Iterable[int] = (1, 2, 3, 4, 6, 8, 12),
    n_random: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Keep only top-k PCs of Z=rho_in; compare MAE vs random k-subspaces.
    Works for non-linear ρ by recomputing ρ(Z_proj).
    """
    taps = forward_taps(f_net, X, spin)
    Z = taps["rho_in"]
    base = f_net.rho(Z).reshape(-1, 1)

    pca = pca_on_rho_in(f_net, X, spin)
    U, mu = pca["U"], pca["mu"]
    Zc = Z - mu
    Fdim = Z.shape[1]
    out_pc = []
    out_rand = {k: [] for k in ks}

    for k in ks:
        Uk = U[:, :k]
        Z_proj = (Zc @ Uk) @ Uk.T + mu
        yk = f_net.rho(Z_proj).reshape(-1, 1)
        out_pc.append((yk - base).abs().mean().item())

        for _ in range(n_random):
            G = torch.randn(Fdim, k, device=Z.device, dtype=Z.dtype)
            Q, _ = torch.linalg.qr(G, mode="reduced")
            Zr = (Zc @ Q) @ Q.T + mu
            yr = f_net.rho(Zr).reshape(-1, 1)
            out_rand[k].append((yr - base).abs().mean().item())

    rand_mae = torch.tensor(
        [float(torch.tensor(out_rand[k]).mean()) for k in ks], device=Z.device, dtype=Z.dtype
    )
    pc_mae = torch.tensor(out_pc, device=Z.device, dtype=Z.dtype)
    return dict(
        k=torch.tensor(list(ks), device=Z.device),
        mae_pc=pc_mae,
        mae_rand=rand_mae,
        pca_eff_rank=pca["eff_rank"],
    )


@torch.no_grad()
def compute_physical_summaries(
    f_net: nn.Module,
    X: torch.Tensor,
    spin: torch.Tensor | None = None,
    r0: float = 0.25,
    nbins: int = 24,
) -> dict[str, torch.Tensor]:
    """
    Returns per-config summaries (using r from taps):
      - mean pair distance r̄
      - var of pair distance
      - fraction of 'close' pairs Pr(r < r0)
      - 'shell contrast' from radial histogram of single-particle radii (scaled coords)
    """
    taps = forward_taps(f_net, X, spin)
    r = taps["r"].squeeze(-1)  # (B, P)

    r_mean = r.mean(dim=1, keepdim=True)
    r_var = r.var(dim=1, unbiased=False, keepdim=True)
    frac_close = (r < r0).float().mean(dim=1, keepdim=True)

    Xsc = X * (f_net.omega**0.5)
    r_part = Xsc.norm(dim=-1)  # (B, N)
    rmin, rmax = float(r_part.min().item()), float(r_part.max().item()) + 1e-6
    edges = torch.linspace(rmin, rmax, steps=nbins + 1, device=X.device, dtype=X.dtype)

    H = []
    for b in range(X.shape[0]):
        try:
            h = torch.histogram(r_part[b], bins=edges)[0]
            h = (h / (h.sum() + 1e-12)).to(X.dtype)
        except Exception:
            h = torch.histc(r_part[b].float(), bins=nbins, min=rmin, max=rmax).to(X.dtype)
            h = h / (h.sum() + 1e-12)
        H.append(h)
    H = torch.stack(H, dim=0)  # (B, nbins)
    shell_contrast = H.std(dim=1, keepdim=True)

    feats = torch.cat([r_mean, r_var, frac_close, shell_contrast], dim=1)  # (B, 4)
    feat_names = ["r_mean", "r_var", f"Pr(r<{r0:.2f})", "shell_contrast"]
    return dict(features=feats, names=feat_names)


@torch.no_grad()
def linear_probe_pcs(
    f_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None, top_k: int = 3
) -> dict[str, torch.Tensor]:
    """
    Project Z onto top-k PCs, then fit linear probes from PC scores -> physical summaries.
    Return R^2 per summary.
    """
    taps = forward_taps(f_net, X, spin)
    Z = taps["rho_in"]
    pca = pca_on_rho_in(f_net, X, spin)
    U, mu = pca["U"], pca["mu"]
    Zc = Z - mu
    Uk = U[:, :top_k]  # (F, k)
    S = Zc @ Uk  # (B, k)

    phys = compute_physical_summaries(f_net, X, spin)
    Y = phys["features"]  # (B, 4)
    Sb = torch.cat(
        [S, torch.ones(S.shape[0], 1, device=S.device, dtype=S.dtype)], dim=1
    )  # (B, k+1)

    R2 = []
    for t in range(Y.shape[1]):
        y = Y[:, t : t + 1]
        W, *_ = torch.linalg.lstsq(Sb, y)  # (k+1, 1)
        yhat = Sb @ W
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
        R2.append(1.0 - (ss_res / ss_tot).item())
    return dict(
        R2=torch.tensor(R2, device=Z.device, dtype=Z.dtype), target_names=phys["names"], U_topk=Uk
    )


# =============================================================================
# 4) NEAR-FIELD (residual gradient share by small-r_min)
# =============================================================================


def near_field_grad_share_by_quantile(
    f_net: nn.Module,
    X: torch.Tensor,
    spin: torch.Tensor | None = None,
    qs: Iterable[float] = (0.01, 0.05, 0.10),
    min_count: int = 32,
) -> dict[str, list]:
    """
    For each quantile q of the per-config min pair distance r_min,
    report: share = <||∇ f_res||^2 | r_min <= q-quantile> / <||∇ f_res||^2>, with a minimum count.
    """
    X = X.detach().requires_grad_(True)
    taps = forward_taps(f_net, X, spin, track_grad=True)

    # residual gradients (learned part only)
    y = taps["out_wo_cusp"].sum()
    (gx,) = torch.autograd.grad(y, X, create_graph=False)
    g2 = (gx**2).sum(dim=(1, 2)) + 1e-18  # (B,)

    # --- make rmin 1D ---
    r = taps["r"].squeeze(-1)  # (B, P)
    rmin = r.amin(dim=1)  # (B,)

    B = rmin.numel()
    g2_tot = float(g2.mean())

    rows = []
    for q in qs:
        cutoff = torch.quantile(rmin, q)
        mask = rmin <= cutoff  # (B,)
        cnt = int(mask.sum().item())

        if cnt < min_count:
            k = min(min_count, B)
            idx = torch.topk(-rmin, k).indices  # smallest rmin
            mask = torch.zeros_like(rmin, dtype=torch.bool)
            mask[idx] = True
            cnt = k

        share = float((g2[mask].mean().item()) / g2_tot)
        rows.append(dict(q=float(q), share=share, count=cnt, frac=float(cnt / B)))

    return dict(rows=rows)


# =============================================================================
# 5) SAVE HELPERS
# =============================================================================


def save_metrics(out: dict, save_json: str | Path, base: str | Path = "..") -> Path:
    base = Path(base)
    json_dir = base / "results" / "tables"
    json_dir.mkdir(parents=True, exist_ok=True)

    save_json = Path(save_json)
    if save_json.suffix.lower() != ".json":
        save_json = save_json.with_suffix(".json")

    file_path = json_dir / save_json.name
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {file_path}")
    return file_path


# =============================================================================
# 6) BRANCH ABLATION (evaluates non-linear ρ directly)
# =============================================================================


@torch.no_grad()
def branch_ablation_drop(
    f_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None
) -> dict[str, float]:
    """
    Zero φ_mean / ψ_mean / extras in ρ-in and measure |Δ head| using the (non-linear) ρ.
    """
    taps = forward_taps(f_net, X, spin)
    Z = taps["rho_in"].clone()
    B, _ = Z.shape
    base = f_net.rho(Z).reshape(B, 1)

    dL = f_net.dL
    ranges = {"phi": (0, dL), "psi": (dL, 2 * dL), "extras": (2 * dL, 2 * dL + 2)}
    out = {}
    for name, (a, bnd) in ranges.items():
        Z_ = Z.clone()
        Z_[:, a:bnd] = 0.0
        y = f_net.rho(Z_).reshape(B, 1)
        out[name] = float((y - base).abs().mean().item())
    return out


# =============================================================================
# 7) COMPACT ANALYSIS RUNNER — EXACT ONLY + BACKFLOW SUPPORT
# =============================================================================
def run_compact_analysis(
    f_net: nn.Module,
    psi_fn: nn.Module,
    C_occ,
    params: dict,
    std: float,
    *,
    make_psi_log_fn=make_psi_log_fn,
    sample_psi2_batch=sample_psi2_batch,
    compute_local_energy_batch=None,  # unused; kept for API compatibility
    compute_coulomb_interaction=None,  # must be provided (vectorized)
    spin: torch.Tensor | None = None,
    backflow_net: nn.Module | None = None,
    B_smallN: int = 16384,
    B_bigN: int = 8096,
    save_json: str | None = None,
    near_qs: Iterable[float] = (0.01, 0.05, 0.10),
    min_qcount: int = 32,
    winsor_pct: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Unified analysis:
      • PINN correlator analysis (same as before).
      • BackflowNet analysis (NEW): analyzes the BF module itself, not f_net.

    Energy uses exact logΨ autodiff:
      E_L = V_trap + V_int - 1/2 (Δ logΨ + ||∇ logΨ||^2)

    Required external helpers already in your codebase (PINN side):
      forward_taps, feature_svd, pc_projection_ablation, linear_probe_pcs,
      pca_on_rho_in, cusp_vs_residual_means, near_field_grad_share_by_quantile,
      energy_feature_correlations
    """

    # ---------------- small utilities ----------------
    import json
    from pathlib import Path

    def _save_metrics_local(out: dict, save_json: str | Path, base: str | Path = "..") -> Path:
        base = Path(base)
        json_dir = base / "results" / "tables"
        json_dir.mkdir(parents=True, exist_ok=True)
        save_json = Path(save_json)
        if save_json.suffix.lower() != ".json":
            save_json = save_json.with_suffix(".json")
        file_path = json_dir / save_json.name
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {file_path}")
        return file_path

    def _device_dtype_of(m: nn.Module):
        p = next(m.parameters())
        return p.device, p.dtype

    def _finite_stats(x: torch.Tensor) -> tuple[float, float, float, int, int]:
        mask = torch.isfinite(x)
        n_tot = int(mask.numel())
        n_fin = int(mask.sum().item())
        if n_fin == 0:
            return float("nan"), float("nan"), float("nan"), 0, n_tot
        xf = x[mask]
        if winsor_pct > 0.0:
            lo = torch.quantile(xf, winsor_pct)
            hi = torch.quantile(xf, 1 - winsor_pct)
            xf = xf.clamp(min=float(lo), max=float(hi))
        mu = float(xf.mean().item())
        sd = float(xf.std(unbiased=False).item())
        se = sd / max(1.0, math.sqrt(n_fin))
        return mu, sd, se, n_fin, n_tot

    # -------- exact local energy via autograd on log Ψ --------
    assert compute_coulomb_interaction is not None, "compute_coulomb_interaction must be provided."

    def _exact_local_energy(
        psi_log_fn,
        x: torch.Tensor,
        *,
        chunk_B: int = 96,  # tune: 64–192 for float64 on 24GB; 256–512 on float32
    ) -> torch.Tensor:
        """
        Exact Δ log|Ψ| via reverse-over-reverse, but memory-safe:
        - Split walkers into chunks along batch dimension.
        - For each chunk: build graph once, loop diagonals, then free.
        """
        device, dtype = x.device, x.dtype
        B_tot, N, d = x.shape
        D = N * d
        out_chunks = []

        # Process walkers in micro-batches
        for xb in x.split(chunk_B, dim=0):
            xb = xb.detach().clone().requires_grad_(True)

            # 1) ∇ log|Ψ|
            logpsi = psi_log_fn(xb).view(-1, 1)
            g = torch.autograd.grad(logpsi.sum(), xb, create_graph=True, retain_graph=True)[
                0
            ]  # (b,N,d)

            # 2) ||∇ log|Ψ||^2
            g2 = (g * g).sum(dim=(1, 2))  # (b,)

            # 3) Δ log|Ψ| = sum_j ∂g_j/∂x_j (exact, diagonal of Hessian)
            b = xb.shape[0]
            lap = torch.zeros(b, device=device, dtype=dtype)
            g_flat = g.reshape(b, D)

            # loop over coordinates; retain_graph=True so we can take multiple second-grads
            for j in range(D):
                sj = g_flat[:, j].sum()  # scalar
                hj = torch.autograd.grad(
                    sj, xb, retain_graph=True, create_graph=False, allow_unused=False
                )[0]
                lap.add_(hj.reshape(b, D)[:, j])

            # 4) potentials
            omega = float(f_net.omega)
            V_trap = 0.5 * (omega**2) * (xb * xb).sum(dim=(1, 2))
            V_int = compute_coulomb_interaction(xb, params=params).view(-1)

            # 5) local energy for this chunk
            E = V_trap + V_int - 0.5 * (lap + g2)  # (b,)
            out_chunks.append(E.detach().view(-1, 1))

            # free ASAP
            del logpsi, g, g_flat, hj, lap, V_trap, V_int, E, xb
            torch.cuda.empty_cache()

        return torch.cat(out_chunks, dim=0)  # (B_tot, 1)

    # ====================== BF-ONLY HELPERS (NEW) ======================
    @torch.no_grad()
    def _bf_forward_taps(
        backflow_net: nn.Module,
        X: torch.Tensor,
        spin: torch.Tensor | None = None,
        *,
        track_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        with torch.set_grad_enabled(track_grad):
            B, N, d = X.shape
            r = X.unsqueeze(2) - X.unsqueeze(1)
            r2 = (r**2).sum(dim=-1, keepdim=True)
            r1 = torch.sqrt(r2 + 1e-12)
            xi = X.unsqueeze(2).expand(B, N, N, d)
            xj = X.unsqueeze(1).expand(B, N, N, d)
            msg_in = torch.cat([xi, xj, r, r1, r2], dim=-1)  # (B,N,N,3d+2)

            m_ij = backflow_net.phi(msg_in)
            # spin/self masks
            if getattr(backflow_net, "use_spin", False) and (spin is not None):
                if spin.ndim == 1:
                    s_i = spin.view(1, N, 1, 1).to(X.dtype).expand(B, N, N, 1)
                    s_j = spin.view(1, 1, N, 1).to(X.dtype).expand(B, N, N, 1)
                else:
                    s_i = spin.view(B, N, 1, 1).to(X.dtype).expand(B, N, N, 1)
                    s_j = spin.view(B, 1, N, 1).to(X.dtype).expand(B, N, N, 1)
                same = (s_i == s_j).to(X.dtype)
                weight = (
                    same
                    if getattr(backflow_net, "same_spin_only", False)
                    else torch.ones_like(same)
                )
            else:
                weight = torch.ones_like(m_ij[..., :1])

            eye = torch.eye(N, device=X.device, dtype=X.dtype).view(1, N, N, 1)
            weight = weight * (1.0 - eye)
            m_ij = m_ij * weight

            agg = getattr(backflow_net, "aggregation", "sum")
            if agg == "sum":
                m_i = m_ij.sum(dim=2)
            elif agg == "mean":
                m_i = m_ij.mean(dim=2)
            elif agg == "max":
                m_i = m_ij.max(dim=2).values
            else:
                m_i = m_ij.sum(dim=2)

            upd = torch.cat([X, m_i], dim=-1)
            dx_pre = backflow_net.psi(upd)

            out_bound = getattr(backflow_net, "out_bound", "tanh")
            if out_bound == "tanh":
                dx_pre = torch.tanh(dx_pre)

            if hasattr(backflow_net, "bf_scale_raw"):
                bf_scale = torch.nn.functional.softplus(backflow_net.bf_scale_raw)
            else:
                bf_scale = torch.tensor(1.0, device=X.device, dtype=X.dtype)

            dx = dx_pre * bf_scale

        return dict(
            msg_in=msg_in, m_ij=m_ij, m_i=m_i, upd=upd, dx_pre=dx_pre, dx=dx, weights=weight
        )

    def _cov_eigs(M: torch.Tensor) -> torch.Tensor:
        Mc = M - M.mean(dim=0, keepdim=True)
        B = Mc.shape[0]
        F = Mc.shape[1]
        C = (Mc.T @ Mc) / max(1, B - 1) + 1e-12 * torch.eye(F, device=Mc.device, dtype=Mc.dtype)
        return torch.linalg.eigvalsh(C).clamp_min(1e-18)

    def _shannon_rank(evals: torch.Tensor) -> float:
        p = evals / (evals.sum() + 1e-12)
        return float(torch.exp(-(p * p.log()).sum()).item())

    def _pr_rank(evals: torch.Tensor) -> float:
        return float((evals.sum() ** 2 / ((evals**2).sum() + 1e-18)).item())

    @torch.no_grad()
    def _bf_effective_ranks(
        backflow_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None
    ) -> dict[str, float]:
        taps = _bf_forward_taps(backflow_net, X, spin, track_grad=False)
        disp = taps["dx"].reshape(X.shape[0], -1)
        msgs = taps["m_i"].reshape(X.shape[0], -1)
        eval_disp = _cov_eigs(disp)
        eval_msgs = _cov_eigs(msgs)
        return dict(
            disp_eff_rank=_shannon_rank(eval_disp),
            disp_pr_rank=_pr_rank(eval_disp),
            msg_eff_rank=_shannon_rank(eval_msgs),
            msg_pr_rank=_pr_rank(eval_msgs),
        )

    @torch.no_grad()
    def _bf_pca_on_disp(backflow_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None):
        taps = _bf_forward_taps(backflow_net, X, spin, track_grad=False)
        D = taps["dx"].reshape(X.shape[0], -1)
        mu = D.mean(dim=0, keepdim=True)
        Dc = D - mu
        evals, U = torch.linalg.eigh(
            (Dc.T @ Dc) / max(1, D.shape[0] - 1)
            + 1e-12 * torch.eye(D.shape[1], device=D.device, dtype=D.dtype)
        )
        evals = torch.flip(evals, dims=[0])
        U = torch.flip(U, dims=[1])
        expvar = evals / (evals.sum() + 1e-12)
        eff_rank = torch.exp(-(expvar.clamp_min(1e-18) * expvar.clamp_min(1e-18).log()).sum())
        return dict(U=U, evals=evals, expvar=expvar, mu=mu, eff_rank=float(eff_rank))

    @torch.no_grad()
    def _bf_pc_ablation_on_disp(
        backflow_net: nn.Module,
        X: torch.Tensor,
        spin: torch.Tensor | None = None,
        ks: Iterable[int] = (1, 2, 3, 4, 6, 8, 12),
        n_random: int = 12,
    ):
        taps = _bf_forward_taps(backflow_net, X, spin, track_grad=False)
        D = taps["dx"].reshape(X.shape[0], -1)
        pca = _bf_pca_on_disp(backflow_net, X, spin)
        U, mu = pca["U"], pca["mu"]
        Dc = D - mu
        base_norm = float(torch.norm(Dc)) + 1e-12
        rel_err_pc, rel_err_rand = [], []
        F = D.shape[1]
        for k in ks:
            Uk = U[:, :k]
            Dk = (Dc @ Uk) @ Uk.T + mu
            err_pc = float(torch.norm(D - Dk) / base_norm)
            rel_err_pc.append(err_pc)
            errs = []
            for _ in range(n_random):
                G = torch.randn(F, k, device=D.device, dtype=D.dtype)
                Q, _ = torch.linalg.qr(G, mode="reduced")
                Dr = (Dc @ Q) @ Q.T + mu
                errs.append(float(torch.norm(D - Dr) / base_norm))
            rel_err_rand.append(float(torch.tensor(errs).mean()))
        return dict(
            k=torch.tensor(list(ks), device=D.device),
            rel_err_pc=torch.tensor(rel_err_pc, device=D.device),
            rel_err_rand=torch.tensor(rel_err_rand, device=D.device),
            pca_eff_rank=pca["eff_rank"],
        )

    @torch.no_grad()
    def _bf_channel_ablation(
        backflow_net: nn.Module, X: torch.Tensor, spin: torch.Tensor | None = None
    ) -> dict[str, float]:
        taps0 = _bf_forward_taps(backflow_net, X, spin, track_grad=False)
        base = taps0["dx"]
        B, N, d = X.shape
        r = X.unsqueeze(2) - X.unsqueeze(1)
        r2 = (r**2).sum(dim=-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)
        xi = X.unsqueeze(2).expand(B, N, N, d)
        xj = X.unsqueeze(1).expand(B, N, N, d)

        def _flow_from_msg_in(msg_in):
            m_ij = backflow_net.phi(msg_in)
            if getattr(backflow_net, "use_spin", False) and (spin is not None):
                if spin.ndim == 1:
                    s_i = spin.view(1, N, 1, 1).to(X.dtype).expand(B, N, N, 1)
                    s_j = spin.view(1, 1, N, 1).to(X.dtype).expand(B, N, N, 1)
                else:
                    s_i = spin.view(B, N, 1, 1).to(X.dtype).expand(B, N, N, 1)
                    s_j = spin.view(B, 1, N, 1).to(X.dtype).expand(B, N, N, 1)
                same = (s_i == s_j).to(X.dtype)
                weight = (
                    same
                    if getattr(backflow_net, "same_spin_only", False)
                    else torch.ones_like(same)
                )
            else:
                weight = torch.ones_like(m_ij[..., :1])
            eye = torch.eye(N, device=X.device, dtype=X.dtype).view(1, N, N, 1)
            weight = weight * (1.0 - eye)
            m_ij = m_ij * weight

            agg = getattr(backflow_net, "aggregation", "sum")
            if agg == "sum":
                m_i = m_ij.sum(dim=2)
            elif agg == "mean":
                m_i = m_ij.mean(dim=2)
            elif agg == "max":
                m_i = m_ij.max(dim=2).values
            else:
                m_i = m_ij.sum(dim=2)

            upd = torch.cat([X, m_i], dim=-1)
            dx_pre = backflow_net.psi(upd)
            if getattr(backflow_net, "out_bound", "tanh") == "tanh":
                dx_pre = torch.tanh(dx_pre)
            bf_scale = (
                torch.nn.functional.softplus(backflow_net.bf_scale_raw)
                if hasattr(backflow_net, "bf_scale_raw")
                else torch.tensor(1.0, device=X.device, dtype=X.dtype)
            )
            return dx_pre * bf_scale

        base_msg = torch.cat([xi, xj, r, r1, r2], dim=-1)
        abls = {}
        # xi
        msg = base_msg.clone()
        msg[..., 0:d] = 0.0
        abls["xi"] = float(((_flow_from_msg_in(msg) - base).abs().mean()).item())
        # xj
        msg = base_msg.clone()
        msg[..., d : 2 * d] = 0.0
        abls["xj"] = float(((_flow_from_msg_in(msg) - base).abs().mean()).item())
        # r_vec
        msg = base_msg.clone()
        msg[..., 2 * d : 3 * d] = 0.0
        abls["r_vec"] = float(((_flow_from_msg_in(msg) - base).abs().mean()).item())
        # r_norm
        off = 3 * d
        msg = base_msg.clone()
        msg[..., off : off + 1] = 0.0
        abls["r_norm"] = float(((_flow_from_msg_in(msg) - base).abs().mean()).item())
        # r_sq
        msg = base_msg.clone()
        msg[..., off + 1 : off + 2] = 0.0
        abls["r_sq"] = float(((_flow_from_msg_in(msg) - base).abs().mean()).item())
        return abls

    @torch.no_grad()
    def _bf_near_field_concentration(
        backflow_net: nn.Module,
        X: torch.Tensor,
        omega: float,
        spin: torch.Tensor | None = None,
        qs: Iterable[float] = (0.01, 0.05, 0.10),
    ) -> dict[str, list]:
        taps = _bf_forward_taps(backflow_net, X, spin, track_grad=False)
        DX = taps["dx"]
        mag = (DX**2).sum(dim=(1, 2)) + 1e-18
        a = math.sqrt(omega)
        diff = (X * a).unsqueeze(2) - (X * a).unsqueeze(1)
        r = torch.sqrt((diff**2).sum(dim=-1) + 1e-12)
        iu = torch.triu_indices(X.shape[1], X.shape[1], offset=1, device=X.device)
        rmin = r[:, iu[0], iu[1]].amin(dim=1)
        base = float(mag.mean().item())
        rows = []
        B = rmin.numel()
        for q in qs:
            cutoff = torch.quantile(rmin, q)
            mask = rmin <= cutoff
            cnt = int(mask.sum().item())
            share = float((mag[mask].mean().item()) / base) if cnt > 0 else float("nan")
            rows.append(dict(q=float(q), share=share, count=cnt, frac=float(cnt / B)))
        return dict(rows=rows)

    @torch.no_grad()
    def _bf_linear_probes_on_disp(
        backflow_net: nn.Module,
        X: torch.Tensor,
        omega: float,
        spin: torch.Tensor | None = None,
        top_k: int = 3,
    ):
        # reuse basic physical summaries from coords (local, minimal)
        def _phys_summ(X, omega, r0=0.25, nbins=24):
            B, N, d = X.shape
            a = math.sqrt(omega)
            Xs = X * a
            diff = Xs.unsqueeze(2) - Xs.unsqueeze(1)
            r = torch.sqrt((diff**2).sum(dim=-1) + 1e-12)
            iu = torch.triu_indices(N, N, offset=1, device=X.device)
            rp = r[:, iu[0], iu[1]]
            r_mean = rp.mean(dim=1, keepdim=True)
            r_var = rp.var(dim=1, unbiased=False, keepdim=True)
            frac_close = (rp < r0).float().mean(dim=1, keepdim=True)
            radii = torch.sqrt((Xs**2).sum(dim=-1) + 1e-12)
            rmin, rmax = float(radii.min().item()), float(radii.max().item()) + 1e-6
            edges = torch.linspace(rmin, rmax, steps=nbins + 1, device=X.device, dtype=X.dtype)
            H = []
            for b in range(B):
                try:
                    h = torch.histogram(radii[b], bins=edges)[0]
                    h = (h / (h.sum() + 1e-12)).to(X.dtype)
                except Exception:
                    h = torch.histc(radii[b].float(), bins=nbins, min=rmin, max=rmax).to(X.dtype)
                    h = h / (h.sum() + 1e-12)
                H.append(h)
            H = torch.stack(H, dim=0)
            shell_contrast = H.std(dim=1, keepdim=True)
            Y = torch.cat([r_mean, r_var, frac_close, shell_contrast], dim=1)
            names = ["r_mean", "r_var", f"Pr(r<{r0:.2f})", "shell_contrast"]
            return dict(features=Y, names=names)

        taps = _bf_forward_taps(backflow_net, X, spin, track_grad=False)
        D = taps["dx"].reshape(X.shape[0], -1)
        pca = _bf_pca_on_disp(backflow_net, X, spin)
        U, mu = pca["U"], pca["mu"]
        Dc = D - mu
        Uk = U[:, :top_k]
        S = Dc @ Uk
        phys = _phys_summ(X, omega)
        Y = phys["features"]
        Sb = torch.cat([S, torch.ones(S.shape[0], 1, device=S.device, dtype=S.dtype)], dim=1)
        R2 = []
        for t in range(Y.shape[1]):
            y = Y[:, t : t + 1]
            W, *_ = torch.linalg.lstsq(Sb, y)
            yhat = Sb @ W
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
            R2.append(1.0 - (ss_res / ss_tot).item())
        return dict(
            R2=torch.tensor(R2, device=D.device, dtype=D.dtype),
            target_names=phys["names"],
            U_topk=Uk,
        )

    # =========================== SETUP & SAMPLING ===========================
    f_net.eval()
    device, dtype = _device_dtype_of(f_net)
    B = B_smallN if f_net.n_particles <= 2 else B_bigN

    psi_log_fn_nb = make_psi_log_fn(
        psi_fn, f_net, C_occ, backflow_net=None, spin=None, params=params
    )
    psi_log_fn_bf = (
        make_psi_log_fn(psi_fn, f_net, C_occ, backflow_net=backflow_net, spin=None, params=params)
        if backflow_net is not None
        else None
    )

    X_nb, acc_burn_nb, acc_mix_nb = sample_psi2_batch(
        psi_log_fn_nb,
        B=B,
        N=f_net.n_particles,
        d=f_net.d,
        omega=f_net.omega,
        device=device,
        dtype=dtype,
        method="rw",
        step_sigma=0.20 * std,
        burn_in=100,
        mix_steps=30,
        autotune=True,
        target_accept=0.60,
    )
    print(f"[MCMC | no  BF] accept ~ burn {acc_burn_nb:.2f}, mix {acc_mix_nb:.2f}  (B={B})")

    if backflow_net is not None:
        X_bf, acc_burn_bf, acc_mix_bf = sample_psi2_batch(
            psi_log_fn_bf,
            B=B,
            N=f_net.n_particles,
            d=f_net.d,
            omega=f_net.omega,
            device=device,
            dtype=dtype,
            method="rw",
            step_sigma=0.20 * std,
            burn_in=100,
            mix_steps=30,
            autotune=True,
            target_accept=0.60,
        )
        print(f"[MCMC | with BF] accept ~ burn {acc_burn_bf:.2f}, mix {acc_mix_bf:.2f}  (B={B})")
    else:
        X_bf = None
    torch.cuda.empty_cache()

    # =========================== ENERGIES ===========================

    E_L_nb = _exact_local_energy(psi_log_fn_nb, X_nb)  # (B,1)
    mu_nb, sd_nb, se_nb, nfin_nb, ntot_nb = _finite_stats(E_L_nb.view(-1))
    if nfin_nb < ntot_nb:
        print(f"[WARN noBF] non-finite E_L: {ntot_nb - nfin_nb}/{ntot_nb}")
    print(f"[no BF]  E[L] ≈ {mu_nb:.6f}  (std {sd_nb:.6f}, se {se_nb:.6f})")

    if backflow_net is not None:
        E_L_bf = _exact_local_energy(psi_log_fn_bf, X_bf)
        mu_bf, sd_bf, se_bf, nfin_bf, ntot_bf = _finite_stats(E_L_bf.view(-1))
        if nfin_bf < ntot_bf:
            print(f"[WARN  BF] non-finite E_L: {ntot_bf - nfin_bf}/{ntot_bf}")
        print(f"[with BF] E[L] ≈ {mu_bf:.6f}  (std {sd_bf:.6f}, se {se_bf:.6f})")
    else:
        E_L_bf = None
        mu_bf = sd_bf = se_bf = None

    # =========================== PINN DIAGNOSTICS (existing) ===========================
    taps_nb = forward_taps(f_net, X_nb, None, track_grad=False)
    try:
        ablation_nb = branch_ablation_drop(f_net, X_nb, None)
    except Exception:
        # local fallback identical to your earlier code
        Z = forward_taps(f_net, X_nb, None)["rho_in"].clone()
        BZ, _ = Z.shape
        base = f_net.rho(Z).reshape(BZ, 1)
        dL = f_net.dL
        ranges = {"phi": (0, dL), "psi": (dL, 2 * dL), "extras": (2 * dL, 2 * dL + 2)}
        ablation_nb = {}
        for name, (a, b) in ranges.items():
            Z_ = Z.clone()
            Z_[:, a:b] = 0.0
            y = f_net.rho(Z_).reshape(BZ, 1)
            ablation_nb[name] = float((y - base).abs().mean().item())

    cusp_means_nb = cusp_vs_residual_means(f_net, X_nb, None)
    svd_nb = feature_svd(f_net, X_nb, None)

    res_pc_nb = pc_projection_ablation(f_net, X_nb, None, ks=(1, 2, 3, 4, 6, 8, 12), n_random=12)
    probe_nb = linear_probe_pcs(f_net, X_nb, None, top_k=3)

    Z_nb = taps_nb["rho_in"]
    y_nb = f_net.rho(Z_nb).reshape(-1)
    y_nb_std = float(y_nb.std(unbiased=False)) + 1e-12
    rel_pc_nb = (res_pc_nb["mae_pc"] / y_nb_std).detach().cpu().tolist()
    rel_rand_nb = (res_pc_nb["mae_rand"] / y_nb_std).detach().cpu().tolist()

    pca_nb = pca_on_rho_in(f_net, X_nb, None)
    Zc_nb = Z_nb - pca_nb["mu"]
    Uk1_nb = pca_nb["U"][:, :1]
    Z1_nb = (Zc_nb @ Uk1_nb) @ Uk1_nb.T + pca_nb["mu"]
    y1_nb = f_net.rho(Z1_nb).reshape(-1)
    corr_full_pc1_nb = float(torch.corrcoef(torch.stack([y_nb, y1_nb]))[0, 1])
    dL = f_net.dL
    u1 = pca_nb["U"][:, 0]
    pc1_block_power_nb = {
        "phi": float((u1[0:dL] ** 2).sum() / (u1**2).sum()),
        "psi": float((u1[dL : 2 * dL] ** 2).sum() / (u1**2).sum()),
        "extras": float((u1[2 * dL : 2 * dL + 2] ** 2).sum() / (u1**2).sum()),
    }

    nf_nb = near_field_grad_share_by_quantile(f_net, X_nb, None, qs=near_qs, min_count=min_qcount)

    # =========================== BACKFLOW: PINN vs BF & BF-ONLY ===========================
    backflow_report = None
    if backflow_net is not None:
        # --- PINN-on-BF comparisons (for completeness)
        taps_bf = forward_taps(f_net, X_bf, None, track_grad=False)
        svd_bf = feature_svd(f_net, X_bf, None)
        res_pc_bf = pc_projection_ablation(
            f_net, X_bf, None, ks=(1, 2, 3, 4, 6, 8, 12), n_random=12
        )

        # ΔZ effective rank (using covariance entropy rank)
        Bm = min(Z_nb.shape[0], taps_bf["rho_in"].shape[0])
        Z_bf = taps_bf["rho_in"][:Bm]
        Z_nb_m = Z_nb[:Bm].detach()
        dZ = Z_bf - Z_nb_m
        dZc = dZ - dZ.mean(dim=0, keepdim=True)
        C_dZ = (dZc.T @ dZc) / (max(1, dZc.shape[0] - 1)) + 1e-12 * torch.eye(
            dZc.shape[1], device=dZ.device, dtype=dZ.dtype
        )
        evals_dZ = torch.linalg.eigvalsh(C_dZ).clamp_min(1e-18)
        expvar_dZ = evals_dZ / (evals_dZ.sum() + 1e-12)
        eff_rank_dZ = float(torch.exp(-(expvar_dZ * expvar_dZ.log()).sum()).item())

        # corr(head, head@PC1) on BF
        pca_bf = pca_on_rho_in(f_net, X_bf, None)
        Z_bf_rho = taps_bf["rho_in"]
        Zc_bf = Z_bf_rho - pca_bf["mu"]
        Uk1_bf = pca_bf["U"][:, :1]
        y_bf = f_net.rho(Z_bf_rho).reshape(-1)
        y1_bf = f_net.rho((Zc_bf @ Uk1_bf) @ Uk1_bf.T + pca_bf["mu"]).reshape(-1)
        corr_full_pc1_bf = float(torch.corrcoef(torch.stack([y_bf, y1_bf]))[0, 1])
        # cosine alignment of PC1
        cos_pc1 = float(
            (
                (Uk1_nb.squeeze() @ Uk1_bf.squeeze())
                / (Uk1_nb.squeeze().norm() * Uk1_bf.squeeze().norm() + 1e-12)
            ).item()
        )

        # energy deltas ON BF samples
        E_L_nb_on_bf = _exact_local_energy(psi_log_fn_nb, X_bf)
        delta_E_on_bf = (E_L_bf.view(-1) - E_L_nb_on_bf.view(-1)).detach()
        dmu, dsd, dse, dfin, dtot = _finite_stats(delta_E_on_bf)

        # synergy with PINN residual vs cusp on BF set
        base_bf = taps_bf["out_wo_cusp"].view(-1)
        cusp_bf = taps_bf["cusp"].view(-1)
        mask_fin = torch.isfinite(delta_E_on_bf) & torch.isfinite(base_bf) & torch.isfinite(cusp_bf)
        if mask_fin.any():
            corr_dE_base = float(
                torch.corrcoef(torch.stack([delta_E_on_bf[mask_fin], base_bf[mask_fin]]))[0, 1]
            )
            corr_dE_cusp = float(
                torch.corrcoef(torch.stack([delta_E_on_bf[mask_fin], cusp_bf[mask_fin]]))[0, 1]
            )
        else:
            corr_dE_base = float("nan")
            corr_dE_cusp = float("nan")

        # Near-field concentration of |ΔE|
        def _near_field_delta_share(
            f_net: nn.Module,
            X: torch.Tensor,
            delta: torch.Tensor,
            qs: Iterable[float],
            min_count: int,
        ) -> dict[str, list]:
            taps = forward_taps(f_net, X, None)
            rmin = taps["r"].squeeze(-1).amin(dim=1)
            B = rmin.numel()
            rows = []
            base = float(torch.nanmean(delta.abs()).item() + 1e-18)
            for q in qs:
                cutoff = torch.quantile(rmin, q)
                mask = rmin <= cutoff
                cnt = int(mask.sum().item())
                if cnt < min_count:
                    k = min(min_count, B)
                    idx = torch.topk(-rmin, k).indices
                    mask = torch.zeros_like(rmin, dtype=torch.bool)
                    mask[idx] = True
                    cnt = k
                share = float((torch.nanmean(delta[mask].abs()).item()) / base)
                rows.append(dict(q=float(q), share=share, count=cnt, frac=float(cnt / B)))
            return dict(rows=rows)

        nf_delta = _near_field_delta_share(
            f_net, X_bf, delta_E_on_bf, qs=near_qs, min_count=min_qcount
        )

        # conditioning proxies on Z
        def _feature_cov_cond(f_net: nn.Module, X: torch.Tensor) -> float:
            taps = forward_taps(f_net, X, None)
            Z = taps["rho_in"]
            Zc = Z - Z.mean(dim=0, keepdim=True)
            C = (Zc.T @ Zc) / (max(1, Zc.shape[0] - 1)) + 1e-10 * torch.eye(
                Zc.shape[1], device=Z.device, dtype=Z.dtype
            )
            evals = torch.linalg.eigvalsh(C)
            return float((evals.max() / (evals.min() + 1e-16)).item())

        cond_feat_nb = _feature_cov_cond(f_net, X_nb)
        cond_feat_bf = _feature_cov_cond(f_net, X_bf)

        # ----------- BF NETWORK ANALYSIS (THIS IS THE NEW IMPORTANT PART) -----------
        bf_net = backflow_net
        bf_ranks = _bf_effective_ranks(bf_net, X_bf, spin=None)
        bf_pca = _bf_pca_on_disp(bf_net, X_bf, spin=None)
        bf_pcab = _bf_pc_ablation_on_disp(
            bf_net, X_bf, spin=None, ks=(1, 2, 3, 4, 6, 8, 12), n_random=12
        )
        bf_chan = _bf_channel_ablation(bf_net, X_bf, spin=None)
        bf_probe = _bf_linear_probes_on_disp(
            bf_net, X_bf, omega=float(f_net.omega), spin=None, top_k=3
        )
        bf_nf_dx2 = _bf_near_field_concentration(
            bf_net, X_bf, omega=float(f_net.omega), spin=None, qs=(0.01, 0.05, 0.10)
        )

        backflow_report = dict(
            energies=dict(
                mean_noBF=mu_nb,
                std_noBF=sd_nb,
                se_noBF=se_nb,
                mean_BF=mu_bf,
                std_BF=sd_bf,
                se_BF=se_bf,
                mean_delta=dmu,
                std_delta=dsd,
                se_delta=dse,
                nfinite_delta=dfin,
                ntotal_delta=dtot,
            ),
            # PINN geometry comparisons (context)
            feature_eff_rank=dict(
                noBF=float(svd_nb["effective_rank"]), BF=float(svd_bf["effective_rank"])
            ),
            pca_noBF=dict(
                eff_rank=float(res_pc_nb["pca_eff_rank"]),
                expvar_top8=pca_nb["expvar"][:8].detach().cpu().tolist(),
                head_corr_pc1=float(corr_full_pc1_nb),
            ),
            pca_BF=dict(
                eff_rank=float(res_pc_bf["pca_eff_rank"]),
                expvar_top8=pca_bf["expvar"][:8].detach().cpu().tolist(),
                head_corr_pc1=float(corr_full_pc1_bf),
            ),
            pc1_alignment=dict(cosine=cos_pc1),
            pc_ablation=dict(
                noBF=dict(
                    k=res_pc_nb["k"].tolist(),
                    rel_mae_pc=(
                        res_pc_nb["mae_pc"]
                        / (float(f_net.rho(taps_nb["rho_in"]).std(unbiased=False)) + 1e-12)
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    rel_mae_rand=(
                        res_pc_nb["mae_rand"]
                        / (float(f_net.rho(taps_nb["rho_in"]).std(unbiased=False)) + 1e-12)
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    pca_eff_rank=float(res_pc_nb["pca_eff_rank"]),
                ),
                BF=dict(
                    k=res_pc_bf["k"].tolist(),
                    rel_mae_pc=(
                        res_pc_bf["mae_pc"]
                        / (float(f_net.rho(taps_bf["rho_in"]).std(unbiased=False)) + 1e-12)
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    rel_mae_rand=(
                        res_pc_bf["mae_rand"]
                        / (float(f_net.rho(taps_bf["rho_in"]).std(unbiased=False)) + 1e-12)
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    pca_eff_rank=float(res_pc_bf["pca_eff_rank"]),
                ),
            ),
            energy_feature_corr=dict(
                noBF=dict(
                    corr_features=energy_feature_correlations(taps_nb, E_L_nb.view(-1))[
                        "corr_features"
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                    corr_out_base=float(
                        energy_feature_correlations(taps_nb, E_L_nb.view(-1))["corr_out_base"]
                    ),
                    corr_cusp=float(
                        energy_feature_correlations(taps_nb, E_L_nb.view(-1))["corr_cusp"]
                    ),
                ),
                BF=dict(
                    corr_features=energy_feature_correlations(taps_bf, E_L_bf.view(-1))[
                        "corr_features"
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                    corr_out_base=float(
                        energy_feature_correlations(taps_bf, E_L_bf.view(-1))["corr_out_base"]
                    ),
                    corr_cusp=float(
                        energy_feature_correlations(taps_bf, E_L_bf.view(-1))["corr_cusp"]
                    ),
                ),
            ),
            conditioning=dict(covZ_cond_noBF=cond_feat_nb, covZ_cond_BF=cond_feat_bf),
            deltaZ_eff_rank=eff_rank_dZ,
            near_field_delta=nf_delta["rows"],
            # >>> BF network proper:
            bf_network=dict(
                ranks=bf_ranks,
                pca_disp=dict(
                    eff_rank=bf_pca["eff_rank"],
                    expvar_top8=bf_pca["expvar"][:8].detach().cpu().tolist(),
                ),
                pc_ablation=dict(
                    k=bf_pcab["k"].tolist(),
                    rel_err_pc=bf_pcab["rel_err_pc"].detach().cpu().tolist(),
                    rel_err_rand=bf_pcab["rel_err_rand"].detach().cpu().tolist(),
                    pca_eff_rank=float(bf_pcab["pca_eff_rank"]),
                ),
                probes=dict(
                    names=[str(n) for n in bf_probe["target_names"]],
                    R2=[float(x) for x in bf_probe["R2"].tolist()],
                ),
                channel_ablation={k: float(v) for k, v in bf_chan.items()},
                near_field_dx2=bf_nf_dx2["rows"],
            ),
        )

    # =========================== PRINT COMPACT REPORT ===========================
    print("\n=== Compact report ===")
    print(f"System: N={f_net.n_particles}, d={f_net.d}, ω={float(f_net.omega)}")
    print(f"[no BF]  Energy: E≈{mu_nb:.6f}  (±{se_nb:.6f} se)  std={sd_nb:.6f}")
    if backflow_net is not None:
        print(f"[with BF] Energy: E≈{mu_bf:.6f}  (±{se_bf:.6f} se)  std={sd_bf:.6f}")
    print(
        f"Branch Δout (no-BF): ψ={ablation_nb['psi']:.3f}, φ={ablation_nb['phi']:.3f}, extras={ablation_nb['extras']:.3f}"
    )
    print(f"Means (out, cusp, base | no-BF): {cusp_means_nb}")
    print(
        f"Eff. rank (features | no-BF) = {float(svd_nb['effective_rank']):.3f} | Eff. rank (ρ_in | no-BF) = {float(res_pc_nb['pca_eff_rank']):.3f}"
    )
    print("PC ablation rel-MAE (PC vs random | no-BF):")
    for k, r_pc, r_rd in zip(res_pc_nb["k"].tolist(), rel_pc_nb, rel_rand_nb, strict=False):
        print(f"  k={k:>2}  PC={r_pc:.4e}  rand={r_rd:.4e}")
    print(
        "PC probes R^2 (no-BF):",
        {
            n: float(r)
            for n, r in zip(probe_nb["target_names"], probe_nb["R2"].tolist(), strict=False)
        },
    )
    print(f"corr(head, head@PC1 | no-BF) = {corr_full_pc1_nb:.6f}")
    print(
        "PC1 block power shares (no-BF):", {k: round(v, 3) for k, v in pc1_block_power_nb.items()}
    )
    for row in nf_nb["rows"]:
        print(
            f"[Near-field q | no-BF] q={row['q']:.0%}  share={row['share']:.3f}  count={row['count']}  frac={row['frac']:.5f}"
        )

    if backflow_net is not None:
        print("\n--- Backflow vs No-Backflow (PINN context) ---")
        print(
            f"ΔE (mean±se) on BF samples: {backflow_report['energies']['mean_delta']:.6f} ± {backflow_report['energies']['se_delta']:.6f} "
            f"(std {backflow_report['energies']['std_delta']:.6f}, finite {backflow_report['energies']['nfinite_delta']}/{backflow_report['energies']['ntotal_delta']})"
        )
        print(f"PC1 cosine(noBF,BF) = {backflow_report['pc1_alignment']['cosine']:.6f}")
        print(f"ΔZ effective rank ≈ {backflow_report['deltaZ_eff_rank']:.3f}")
        for row in backflow_report["near_field_delta"]:
            print(
                f"[Near-field ΔE | BF] q={row['q']:.0%}  share={row['share']:.3f}  count={row['count']}  frac={row['frac']:.5f}"
            )

        # --- BF network proper ---
        bf = backflow_report["bf_network"]
        print("\n--- BackflowNet (Δx/messages) ---")
        print(
            f"Eff-rank(Δx)={bf['ranks']['disp_eff_rank']:.3f}  PR(Δx)={bf['ranks']['disp_pr_rank']:.3f} | "
            f"Eff-rank(msg)={bf['ranks']['msg_eff_rank']:.3f}  PR(msg)={bf['ranks']['msg_pr_rank']:.3f}"
        )
        print(
            f"PCA(Δx) eff-rank={backflow_report['bf_network']['pca_disp']['eff_rank']:.3f} "
            f"| expvar top-4={ [round(float(x),6) for x in backflow_report['bf_network']['pca_disp']['expvar_top8'][:4]] }"
        )
        print("PC ablation rel-||·|| (Δx):")
        for k, r_pc, r_rd in zip(
            bf["pc_ablation"]["k"],
            bf["pc_ablation"]["rel_err_pc"],
            bf["pc_ablation"]["rel_err_rand"],
            strict=False,
        ):
            print(f"  k={int(k):>2}  PC={r_pc:.4e}  rand={r_rd:.4e}")
        print(
            "Linear probes R^2 (Δx PCs → phys):",
            {n: float(r) for n, r in zip(bf["probes"]["names"], bf["probes"]["R2"], strict=False)},
        )
        print(
            "Input-channel ablation Δx-change:",
            {k: round(float(v), 4) for k, v in bf["channel_ablation"].items()},
        )
        for row in bf["near_field_dx2"]:
            print(
                f"[Near-field ||Δx||²] q={row['q']:.0%}  share={row['share']:.3f}  count={row['count']}  frac={row['frac']:.5f}"
            )

    # =========================== BUILD OUTPUT JSON ===========================
    out = dict(
        system=dict(N=f_net.n_particles, d=f_net.d, omega=float(f_net.omega)),
        energy=dict(noBF=dict(mean=mu_nb, std=sd_nb, se=se_nb, nfinite=nfin_nb, ntotal=ntot_nb)),
        ablation_noBF=ablation_nb,
        cusp_means_noBF=cusp_means_nb,
        feature_eff_rank_noBF=float(svd_nb["effective_rank"]),
        pca_noBF=dict(
            eff_rank=float(res_pc_nb["pca_eff_rank"]),
            k=res_pc_nb["k"].tolist(),
            rel_mae_pc=rel_pc_nb,
            rel_mae_rand=rel_rand_nb,
            expvar_top8=pca_nb["expvar"][:8].detach().cpu().tolist(),
            head_corr_pc1=corr_full_pc1_nb,
            pc1_block_power=pc1_block_power_nb,
        ),
        probes_noBF=dict(
            names=probe_nb["target_names"], R2=[float(x) for x in probe_nb["R2"].tolist()]
        ),
        near_field_quantiles_noBF=nf_nb["rows"],
        with_backflow=backflow_net is not None,
        winsor_pct=winsor_pct,
    )

    if backflow_net is not None:
        out["energy"]["withBF"] = dict(
            mean=mu_bf, std=sd_bf, se=se_bf, nfinite=nfin_bf, ntotal=ntot_bf
        )
        out["backflow_report"] = backflow_report

    if save_json is not None:
        _save_metrics_local(out, save_json=save_json)

    # Prefer BF set in return (if available)
    ret_X = X_bf if (backflow_net is not None) else X_nb
    ret_E = E_L_bf if (backflow_net is not None) else E_L_nb
    return dict(
        X=ret_X,
        E_L=ret_E,
        report=out,
        E_L_noBF=E_L_nb,
        E_L_BF=E_L_bf,
    )
