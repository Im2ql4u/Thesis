# stable_training.py — drop-in stabilized ψ, Laplacians, and trainer
# - C² soft-core Coulomb (vectorized)
# - Stable psi_fn (slogdet guard + centered logψ before exp)
# - HVP Hutchinson with vJP and FD fallback
# - FD Hutchinson with adaptive epsilon
# - Residual trainer with per-row NaN masking and ψ in-graph normalization

from typing import Literal

import torch
from torch import nn

# if you already have this in your project, keep using it
from utils import inject_params

from .Physics import compute_coulomb_interaction

# keep your Slater import (assumed to use slogdet internally)
from .Slater_Determinant import slater_determinant_closed_shell


# ---------------------------------------------------------------------
# 1) Stable ψ(x): slogdet guard + center logψ before exp
# ---------------------------------------------------------------------
@inject_params
def psi_fn(
    f_net: nn.Module,
    x_batch: torch.Tensor,
    C_occ: torch.Tensor,
    *,
    backflow_net: nn.Module | None = None,
    spin: torch.Tensor | None = None,
    params=None,
):
    """
    ψ(x) = det(Slater(x+Δx; C_occ)) * exp(f_net(x+Δx)).

    Stabilizations:
      - slater_determinant_closed_shell is assumed to use slogdet; we
        additionally guard non-finite rows.
      - center logψ by subtracting max(logψ) before exp to avoid overflow.
        This constant cancels in ψ-normalized training objectives.
    """
    x_batch = x_batch.contiguous()
    C_occ = C_occ.to(device=x_batch.device, dtype=x_batch.dtype).contiguous()

    # optional backflow
    if backflow_net is not None:
        dx = backflow_net(x_batch, spin=spin)
        x_eff = x_batch + dx
    else:
        x_eff = x_batch

    assert x_eff.requires_grad, "x_eff must have requires_grad=True for autograd"

    # Slater (log-space)
    sign, logabs = slater_determinant_closed_shell(
        x_config=x_eff,
        C_occ=C_occ,
        params=params,
        normalize=True,
    )  # (B,1)

    # guard slater pathologies without breaking autograd graph
    with torch.no_grad():
        bad = (~torch.isfinite(logabs)) | (sign == 0)
        if bad.any():
            logabs[bad] = torch.as_tensor(-1e6, dtype=logabs.dtype, device=logabs.device)
            sign[bad] = torch.as_tensor(1.0, dtype=sign.dtype, device=sign.device)

    # Jastrow/log factor
    f = f_net(x_eff).squeeze(-1)  # (B,)

    # logψ and centered ψ
    logpsi = logabs.view(-1) + f  # (B,)
    c = torch.amax(logpsi.detach())  # scalar, not in the graph
    logpsi_centered = logpsi - c
    psi = sign.view(-1) * torch.exp(logpsi_centered)  # (B,)

    return logpsi, psi


# ---------------------------------------------------------------------
# 2) HVP Hutchinson for Δ logψ (vJP + FD fallback)
# ---------------------------------------------------------------------
def grad_and_laplace_logpsi(psi_log_fn, x, probes: int = 4, fd_eps: float = 1e-4):
    """
    psi_log_fn(x)->(B,)  → returns:
      grad_logpsi: (B,N,d)
      lap_logpsi : (B,1)
    Uses vJP for HVP; falls back to FD along v if hv contains non-finite.
    """
    x = x.requires_grad_(True)
    logpsi = psi_log_fn(x)  # (B,)
    g = torch.autograd.grad(logpsi.sum(), x, create_graph=True, retain_graph=True)[0]

    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)

    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)  # Rademacher ±1
        # H·v via vJP
        hv = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=True, create_graph=True)[0]

        # FD fallback if needed
        if not torch.isfinite(hv).all():
            xp = (x + fd_eps * v).requires_grad_(True)
            xm = (x - fd_eps * v).requires_grad_(True)
            gp = torch.autograd.grad(
                psi_log_fn(xp).sum(), xp, retain_graph=True, create_graph=True
            )[0]
            gm = torch.autograd.grad(
                psi_log_fn(xm).sum(), xm, retain_graph=True, create_graph=True
            )[0]
            hv = (gp - gm) / (2.0 * fd_eps)
            hv = torch.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0)

        acc += (v * hv).sum(dim=(1, 2))

    lap = (acc / max(1, probes)).unsqueeze(1)  # (B,1)
    return g, lap


# ---------------------------------------------------------------------
# 3) Exact Δψ via nested autograd (helper for lap_mode="exact")
# ---------------------------------------------------------------------
def compute_laplacian_fast(psi_only, f_net, x, C_occ, **psi_kwargs):
    """
    Exact Laplacian of ψ via nested autograd.
    Args:
      psi_only: callable(f_net, x, C_occ, **kw) -> ψ (B,)
    Returns:
      Psi: (B,1), Laplacian: (B,1)
    """
    x = x.requires_grad_(True)
    B, N, d = x.shape

    Psi = psi_only(f_net, x, C_occ, **psi_kwargs)  # (B,)
    grads = torch.autograd.grad(Psi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)

    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            g_ij = grads[:, i, j]  # (B,)
            gsum = g_ij.sum()
            second = torch.autograd.grad(gsum, x, create_graph=True, retain_graph=True)[0]
            lap = lap + second[:, i, j]

    return Psi.unsqueeze(1), lap.unsqueeze(1)


# ---------------------------------------------------------------------
# 4) FD Hutchinson for Δ logψ (first-order only, adaptive eps)
# ---------------------------------------------------------------------
def _laplacian_logpsi_fd(psi_log_fn, x_eff, eps: float, probes: int = 2):
    """
    Returns:
      grad_logpsi : (B,N,d)
      g2          : (B,1)   = ||∇ log ψ||^2
      lap_logpsi  : (B,1)   = Δ log ψ (FD-Hutch estimate)
    """
    logpsi = psi_log_fn(x_eff)  # (B,)
    grad_logpsi = torch.autograd.grad(logpsi.sum(), x_eff, create_graph=True)[0]  # (B,N,d)
    g2 = (grad_logpsi**2).sum(dim=(1, 2), keepdim=True)  # (B,1)

    B = x_eff.shape[0]
    acc = torch.zeros(B, device=x_eff.device, dtype=x_eff.dtype)

    for _ in range(probes):
        v = torch.empty_like(x_eff).bernoulli_(0.5).mul_(2).add_(-1)
        x_plus = (x_eff + eps * v).requires_grad_(True)
        lp_plus = psi_log_fn(x_plus)
        g_plus = torch.autograd.grad(lp_plus.sum(), x_plus, create_graph=True)[0]

        x_minus = (x_eff - eps * v).requires_grad_(True)
        lp_minus = psi_log_fn(x_minus)
        g_minus = torch.autograd.grad(lp_minus.sum(), x_minus, create_graph=True)[0]

        acc += ((g_plus * v).sum(dim=(1, 2)) - (g_minus * v).sum(dim=(1, 2))) / (2.0 * eps)

    lap_logpsi = (acc / max(1, probes)).unsqueeze(1)  # (B,1)
    return grad_logpsi, g2, lap_logpsi


# ---------------------------------------------------------------------
# 5) Residual trainer (ψ-normalized, with row masking)
# ---------------------------------------------------------------------
@inject_params
def train_model(
    f_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    C_occ: torch.Tensor,
    mapper=None,  # kept for interface parity
    *,
    backflow_net: nn.Module | None = None,
    spin: torch.Tensor | None = None,
    params=None,
    std: float = 2.5,
    norm_penalty: float = 0.0,  # unused; kept for compat
    probes: int = 2,
    eps_scale: float = 1e-3,
    print_e: int = 50,
    lap_mode: Literal["fd-hutch", "hvp-hutch", "exact"] = "fd-hutch",
):
    """
    Residual training with in-graph ψ normalization:
      minimize ||H ψ_n - E ψ_n||^2, where ψ_n = ψ / ||ψ||_2.

    Stabilizations:
      - per-row finite-mask before building residual (drops catastrophic samples)
      - centered exp in psi_fn prevents overflow
      - soft-core Coulomb avoids second-derivative singularities
    """
    device = params["device"]
    w = float(params["omega"])
    n_particles = int(params["n_particles"])
    n_epochs = int(params["n_epochs"])
    E_target = float(params["E"])
    N_collocation = int(params["N_collocation"])
    d = int(params["d"])
    dtype = params.get("torch_dtype", None)
    QHO_const = 0.5 * (w**2)

    f_net.to(device)
    if backflow_net is not None:
        backflow_net.to(device)

    if spin is None:
        up = n_particles // 2
        down = n_particles - up
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(down, dtype=torch.long)]
        ).to(device)
    else:
        spin = spin.to(device)

    # ψ-only wrapper for exact Laplacian helper
    def psi_only(_f, _x, _C, **kw):
        _logpsi, _psi = psi_fn(_f, _x, _C, **kw)
        return _psi.view(-1)

    # log ψ closure for Hutchinson variants
    def psi_log_closure(y):
        logpsi_y, _psi_y = psi_fn(
            f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params
        )
        return logpsi_y  # (B,)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Sample collocation points
        x_kwargs = dict(device=device)
        if dtype is not None:
            x_kwargs["dtype"] = dtype
        x = (
            torch.normal(0, std, size=(N_collocation, n_particles, d), **x_kwargs)
            .clamp(min=-9, max=9)
            .requires_grad_(True)
        )

        # ---- Laplacian paths ----
        if lap_mode == "exact":
            # Exact Δψ via nested autograd on ψ
            Psi, Delta_psi = compute_laplacian_fast(
                psi_only, f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )  # (B,1), (B,1)
            psi = Psi  # (B,1)

        elif lap_mode == "hvp-hutch":
            # Hutchinson on Δ log ψ using HVPs
            logpsi, psi_raw = psi_fn(
                f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )  # logpsi:(B,), ψ:(B,)
            psi = psi_raw.view(-1, 1)  # (B,1)
            grad_logpsi, lap_logpsi = grad_and_laplace_logpsi(
                psi_log_closure, x, probes=probes, fd_eps=1e-4
            )  # grad:(B,N,d), lap_logpsi:(B,1)
            g2 = (grad_logpsi**2).sum(dim=(1, 2), keepdim=True)  # (B,1)
            Delta_psi = psi * (lap_logpsi + g2)  # (B,1)

        elif lap_mode == "fd-hutch":
            # Finite-difference Hutchinson on Δ log ψ (first-order only)
            logpsi, psi_raw = psi_fn(
                f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )
            psi = psi_raw.view(-1, 1)
            eps = eps_scale * float(std)
            grad_logpsi, g2, lap_logpsi = _laplacian_logpsi_fd(
                psi_log_closure, x, eps=eps, probes=probes
            )  # grad:(B,N,d), g2:(B,1), lap_logpsi:(B,1)
            Delta_psi = psi * (lap_logpsi + g2)  # (B,1)

        else:
            raise ValueError(f"Unknown lap_mode={lap_mode!r}")

        # ---- Guard and normalize ψ & Δψ ----
        row_ok = torch.isfinite(psi.view(-1)) & torch.isfinite(Delta_psi.view(-1))
        if row_ok.sum() < psi.shape[0]:
            x = x[row_ok]
            psi = psi[row_ok]
            Delta_psi = Delta_psi[row_ok]
        if psi.numel() == 0:
            continue  # resample next epoch

        # in-graph normalization (remove scale mode)
        psi_norm = torch.linalg.vector_norm(psi) + 1e-30
        psi_n = psi / psi_norm
        Delta_psi_n = Delta_psi / psi_norm

        # ---- Potentials ----
        V_harmonic = QHO_const * (x**2).sum(dim=(1, 2), keepdim=True)  # (B,1)
        V_int = compute_coulomb_interaction(x)  # (B,1)
        if V_int.dim() != 2:
            V_int = V_int.view(-1, 1)
        V_total = V_harmonic + V_int  # (B,1)

        # ---- Residual & loss ----
        H_psi_n = -0.5 * Delta_psi_n + V_total * psi_n
        residual = H_psi_n - E_target * psi_n
        loss_pde = (residual**2).mean()
        loss_pde.backward()
        optimizer.step()

        if epoch % print_e == 0:
            with torch.no_grad():
                E_L = (H_psi_n / psi_n).clamp(min=-1e6, max=1e6)
                var_E = torch.var(E_L)
                raw_norm = float(torch.linalg.vector_norm(psi).detach().cpu())
            bf_scale_str = ""
            if backflow_net is not None and hasattr(backflow_net, "bf_scale"):
                try:
                    bf_val = backflow_net.bf_scale
                    bf_val = bf_val.item() if torch.is_tensor(bf_val) else float(bf_val)
                    bf_scale_str = f"  bf_scale={bf_val:.3e}"
                except Exception:
                    pass
            print(
                f"Epoch {epoch:05d}: Resid={loss_pde.item():.3e}  "
                f"||psi||={raw_norm:.3e}  Var(EL)={var_E.item():.3e}  "
                f"[lap={lap_mode.replace('fd-hutch','fd').replace('hvp-hutch','hvp')}]"
                + bf_scale_str
            )

        # free big tensors
        del (
            loss_pde,
            residual,
            H_psi_n,
            V_total,
            V_int,
            V_harmonic,
            psi,
            Delta_psi,
            psi_n,
            Delta_psi_n,
            psi_norm,
        )
