# energy.py
import math

import torch
from tqdm import tqdm

from utils import inject_params


# -----------------------
# Potentials & utilities
# -----------------------
def _coulomb_energy_batched(
    X: torch.Tensor, inv_kappa_scale: float = 0.0, eps: float = 1e-8
) -> torch.Tensor:
    """
    X: (B, N, D)
    Returns Coulomb sum per batch: (B,)
    If inv_kappa_scale==0.0 -> disabled (returns zeros).
    Uses the user's convention: V_ij = 1 / (kappa * r_ij); here we pass inv_kappa_scale = 1/kappa.
    """
    if inv_kappa_scale == 0.0:
        return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

    B, N, D = X.shape
    diffs = X[:, :, None, :] - X[:, None, :, :]  # (B,N,N,D)
    d = torch.linalg.norm(diffs, dim=-1).clamp_min(eps)  # (B,N,N)
    i, j = torch.tril_indices(N, N, offset=-1, device=X.device)
    return inv_kappa_scale * (1.0 / d[:, i, j]).sum(dim=1)  # (B,)


@inject_params
def potential_qdot_2d(X: torch.Tensor, *, params=None) -> torch.Tensor:
    """
    X: (B,N,2)
    V_trap = 1/2 * ω^2 * sum_i |r_i|^2
    V_coul (user's convention) = sum_{i<j} 1/(κ * |r_i - r_j|); set kappa=0 to disable.
    Returns: (B,)
    """
    omega = float(params.get("omega", 1.0))
    kappa = float(params.get("kappa", 0.0))

    # trap
    r2_sum = (X**2).sum(dim=-1).sum(dim=1)  # (B,)
    V_trap = 0.5 * (omega**2) * r2_sum  # (B,)

    # coulomb with user's scaling: 1/(kappa * r)
    inv_kappa_scale = 0.0 if (kappa == 0.0) else (1.0 / kappa)
    V_coul = _coulomb_energy_batched(X, inv_kappa_scale=inv_kappa_scale)  # (B,)

    return V_trap + V_coul


# -----------------------
# Local energy via log|ψ|
# -----------------------
@inject_params
def local_energy_autograd(
    psi_fn, f_net, X: torch.Tensor, C_occ: torch.Tensor, *, params=None
) -> torch.Tensor:
    """
    E_L = -1/2 * [ Δ log|ψ| + ||∇ log|ψ||^2 ] + V(X)
    X: (B,N,D) requires_grad=True
    Returns: (B,)
    """
    # ψ
    psi = psi_fn(f_net, X, C_occ, params)  # (B,)
    if torch.is_complex(psi):
        amp = torch.sqrt(psi.real**2 + psi.imag**2 + 1e-30)
    else:
        amp = psi.abs() + 1e-30

    logpsi = amp.log()  # (B,)

    # score = ∇_X log|ψ|
    (score,) = torch.autograd.grad(logpsi.sum(), X, create_graph=True)  # (B,N,D)

    # Δ log|ψ| = div(score) = sum of diagonal entries of Jacobian(score)
    B, N, D = X.shape
    lap_logpsi = torch.zeros(B, device=X.device, dtype=X.dtype)
    score_flat = score.reshape(B, N * D)
    for k in range(N * D):
        gk = score_flat[:, k]  # (B,)
        (dgk_dx,) = torch.autograd.grad(
            gk.sum(), X, retain_graph=True, create_graph=False, allow_unused=False
        )  # (B,N,D)
        lap_logpsi = lap_logpsi + dgk_dx.reshape(B, N * D)[:, k]

    kinetic = -0.5 * (lap_logpsi + (score**2).sum(dim=[1, 2]))

    # Potential
    V = potential_qdot_2d(X, params=params)  # (B,)

    return kinetic + V


# -----------------------
# VMC estimator (streaming)
# -----------------------
@inject_params
def estimate_energy_vmc(
    psi_fn,
    f_net,
    C_occ: torch.Tensor,
    sample_fn,
    *,
    batches: int = 200,
    batch_size: int = 1024,
    chunk: int = 256,
    params=None,
):
    """
    Streaming mean / stderr of E_L using samples ~ |ψ|^2.
    'chunk' trades memory for speed when doing second derivatives.

    Returns: (mean, stderr)
    """
    # Choose device/dtype from C_occ or model params
    if hasattr(C_occ, "device"):
        device = C_occ.device
    else:
        device = next(f_net.parameters()).device
    if hasattr(C_occ, "dtype"):
        dtype = C_occ.dtype
    else:
        dtype = next(f_net.parameters()).dtype

    mean, m2, n = 0.0, 0.0, 0

    for _ in tqdm(range(batches), desc="VMC"):
        Xb = sample_fn(batch_size).to(device=device, dtype=dtype)  # (B,N,D)

        # Process in chunks to keep autograd graphs small
        for i0 in range(0, Xb.shape[0], chunk):
            Xi = Xb[i0 : i0 + chunk].clone().requires_grad_(True)
            Ei = local_energy_autograd(psi_fn, f_net, Xi, C_occ, params=params).detach()  # (Ci,)

            # Welford online stats for numerical stability
            for e in Ei.tolist():
                n += 1
                delta = e - mean
                mean += delta / n
                m2 += delta * (e - mean)

    var = m2 / max(n - 1, 1)
    stderr = math.sqrt(var / max(n, 1))
    return mean, stderr


# -----------------------
# Example usage (comment)
# -----------------------
# from your_code import psi_fn, f_net, C_occ, make_mala_sample_fn, config
# cfg = config.get()
# sample_fn = make_mala_sample_fn(
#     psi_fn, f_net, C_occ, params=cfg,
#     step_size=0.02, n_steps=40, burn_in=80, thinning=2,
#     init_std=1.0,
#     device=cfg.torch_device, dtype=cfg.torch_dtype,
# )
# f_net = f_net.to(cfg.torch_device, cfg.torch_dtype).eval()
# C_occ = C_occ.to(cfg.torch_device, cfg.torch_dtype)
# E, dE = estimate_energy_vmc(psi_fn, f_net, C_occ, sample_fn,
#                             batches=150, batch_size=1024, chunk=256, params=cfg)
# print(f"VMC energy: {E:.6f} ± {dE:.6f} (1σ)")
