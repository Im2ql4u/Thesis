# functions/DoubleWell.py
# ---------------------------------------------------------------
# Double Well System: Two particles in two separate harmonic wells
# ---------------------------------------------------------------
"""
Two-well quantum system for studying electron localization and tunneling.

Physics:
- Each well is a 2D harmonic oscillator centered at position ±(d/2, 0)
- When wells are far apart: E ≈ 2.0 (two independent ground states)
- When wells overlap (d=0): E ≈ 3.0 (two interacting electrons in one well)

The potential for particle i centered at well position w_i:
    V_i(r) = 0.5 * ω² * |r - w_i|²

For a symmetric double-well with separation d along x-axis:
    - Well 1 at (-d/2, 0)
    - Well 2 at (+d/2, 0)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from utils import inject_params

# ===============================================================
# Double-Well Potential Functions
# ===============================================================


@inject_params
def double_well_potential(
    x: torch.Tensor,
    well_separation: float,
    *,
    params=None,
    assignment: Literal["nearest", "fixed"] = "fixed",
) -> torch.Tensor:
    """
    Compute the double-well harmonic potential energy.

    Parameters
    ----------
    x : torch.Tensor, shape (B, N, d)
        Particle positions. For N=2, particle 0 is in left well, particle 1 in right.
    well_separation : float
        Distance between well centers (in units of 1/√ω).
    params : dict
        Must contain 'omega' (trap frequency).
    assignment : str
        'fixed': particle 0 → left well, particle 1 → right well
        'nearest': each particle assigned to nearest well (symmetric)

    Returns
    -------
    torch.Tensor, shape (B, 1)
        Total harmonic potential energy for all particles.
    """
    omega = float(params["omega"])
    B, N, d = x.shape
    device, dtype = x.device, x.dtype

    # Well centers: left at (-sep/2, 0, ...), right at (+sep/2, 0, ...)
    # Scale separation by characteristic length 1/√ω
    ell = 1.0 / math.sqrt(max(omega, 1e-12))
    sep_phys = well_separation * ell

    left_center = torch.zeros(d, device=device, dtype=dtype)
    left_center[0] = -sep_phys / 2.0

    right_center = torch.zeros(d, device=device, dtype=dtype)
    right_center[0] = +sep_phys / 2.0

    well_centers = torch.stack([left_center, right_center], dim=0)  # (2, d)

    if assignment == "fixed" and N == 2:
        # Particle 0 → left well, particle 1 → right well
        # Displacement from assigned well
        disp = x.clone()  # (B, N, d)
        disp[:, 0, :] = x[:, 0, :] - left_center
        disp[:, 1, :] = x[:, 1, :] - right_center
    elif assignment == "nearest":
        # Each particle goes to nearest well
        # Compute distance to each well center for each particle
        dist_to_left = ((x - left_center.view(1, 1, d)) ** 2).sum(dim=-1)  # (B, N)
        dist_to_right = ((x - right_center.view(1, 1, d)) ** 2).sum(dim=-1)  # (B, N)

        # Assign to nearer well
        use_left = (dist_to_left < dist_to_right).unsqueeze(-1).float()  # (B, N, 1)
        disp = use_left * (x - left_center) + (1.0 - use_left) * (x - right_center)
    else:
        raise ValueError(f"Unknown assignment mode: {assignment}")

    # V = 0.5 * ω² * Σ_i |r_i - w_i|²
    V_trap = 0.5 * (omega**2) * (disp**2).sum(dim=(1, 2), keepdim=True)  # (B, 1)

    return V_trap


@inject_params
def double_well_local_potential(
    x: torch.Tensor,
    well_separation: float,
    *,
    params=None,
) -> torch.Tensor:
    """
    Per-particle potential in double well (for diagnostics/visualization).

    Returns
    -------
    torch.Tensor, shape (B, N)
        Potential energy contribution from each particle.
    """
    omega = float(params["omega"])
    B, N, d = x.shape
    device, dtype = x.device, x.dtype

    ell = 1.0 / math.sqrt(max(omega, 1e-12))
    sep_phys = well_separation * ell

    left_center = torch.zeros(d, device=device, dtype=dtype)
    left_center[0] = -sep_phys / 2.0

    right_center = torch.zeros(d, device=device, dtype=dtype)
    right_center[0] = +sep_phys / 2.0

    # For N=2: fixed assignment
    V_i = torch.zeros(B, N, device=device, dtype=dtype)

    if N >= 1:
        disp0 = x[:, 0, :] - left_center  # (B, d)
        V_i[:, 0] = 0.5 * (omega**2) * (disp0**2).sum(dim=-1)

    if N >= 2:
        disp1 = x[:, 1, :] - right_center  # (B, d)
        V_i[:, 1] = 0.5 * (omega**2) * (disp1**2).sum(dim=-1)

    return V_i


@inject_params
def double_well_coulomb_interaction(
    x: torch.Tensor,
    *,
    params=None,
    eps_rel: float = 1e-19,
    cap: float = 1e8,
) -> torch.Tensor:
    """
    Coulomb interaction between particles (same as standard, but kept for completeness).

    For double-well, the main change is that when wells are far apart,
    inter-particle distance is large and Coulomb repulsion is small.
    """
    kappa = 1.0  # Coulomb constant (atomic units)
    omega = float(params["omega"])
    B, N, d = x.shape
    dev = x.device

    if N < 2:
        return torch.zeros(B, 1, device=dev, dtype=x.dtype)

    # Pairwise distances
    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, N, N, d)
    ii, jj = torch.triu_indices(N, N, 1, device=dev)
    r2 = (diff[:, ii, jj, :] ** 2).sum(-1)  # (B, P)

    # Soft-core regularization
    l0 = 1.0 / math.sqrt(max(omega, 1e-12))
    eps2 = (eps_rel * l0) ** 2

    r_soft = torch.sqrt(r2 + eps2)
    Vij = kappa / r_soft

    Vij = torch.clamp(Vij, max=cap)
    Vij = torch.nan_to_num(Vij, nan=cap, posinf=cap, neginf=-cap)

    V = Vij.sum(dim=1, keepdim=True)

    return V


@inject_params
def double_well_total_potential(
    x: torch.Tensor,
    well_separation: float,
    *,
    params=None,
) -> torch.Tensor:
    """
    Total potential energy: harmonic confinement + Coulomb interaction.
    """
    V_trap = double_well_potential(x, well_separation, params=params)
    V_coul = double_well_coulomb_interaction(x, params=params)
    return V_trap + V_coul


# ===============================================================
# Two-Well Basis Functions
# ===============================================================


@inject_params
def two_well_basis_1d_torch(
    x: torch.Tensor,
    n_basis: int,
    well_center: float,
    *,
    params=None,
) -> torch.Tensor:
    """
    1D harmonic oscillator basis centered at well_center.

    Parameters
    ----------
    x : torch.Tensor, shape (..., 1) or (...)
        Coordinate values.
    n_basis : int
        Number of basis functions.
    well_center : float
        Center of the harmonic well.

    Returns
    -------
    torch.Tensor
        Basis function values.
    """
    omega = float(params["omega"])
    dtype = x.dtype
    sqrt_omega = math.sqrt(omega)

    # Shift coordinate to well center
    xi = (x - well_center) * sqrt_omega
    gauss = torch.exp(-0.5 * omega * (x - well_center) ** 2)

    norm0 = (omega / math.pi) ** 0.25
    cols = [norm0 * gauss]

    if n_basis > 1:
        norm1 = norm0 / math.sqrt(2.0)
        cols.append(norm1 * (2.0 * xi) * gauss)

    if n_basis > 2:
        H_nm1 = 2.0 * xi
        H_nm2 = torch.ones_like(xi)
        for n in range(1, n_basis - 1):
            H_n = 2.0 * xi * H_nm1 - 2.0 * n * H_nm2
            norm = norm0 / math.sqrt((2.0 ** (n + 1)) * math.factorial(n + 1))
            cols.append(norm * H_n * gauss)
            H_nm2, H_nm1 = H_nm1, H_n

    return torch.stack(cols, dim=-1)


@inject_params
def two_well_basis_2d_torch(
    x: torch.Tensor,
    n_basis_x: int,
    n_basis_y: int,
    well_center: torch.Tensor,
    *,
    params=None,
) -> torch.Tensor:
    """
    2D separable harmonic oscillator basis centered at well_center.

    Parameters
    ----------
    x : torch.Tensor, shape (B, N, 2) or (N, 2)
        Particle positions.
    well_center : torch.Tensor, shape (2,)
        Center of the well [x_c, y_c].

    Returns
    -------
    torch.Tensor, shape (..., n_basis_x * n_basis_y)
    """
    x_coord = x[..., 0]
    y_coord = x[..., 1]

    phi_x = two_well_basis_1d_torch(x_coord, n_basis_x, well_center[0].item(), params=params)
    phi_y = two_well_basis_1d_torch(y_coord, n_basis_y, well_center[1].item(), params=params)

    # Outer product
    prod = phi_x.unsqueeze(-1) * phi_y.unsqueeze(-2)
    return prod.reshape(*x.shape[:-1], n_basis_x * n_basis_y)


# ===============================================================
# Double-Well Sampler
# ===============================================================


@inject_params
def sample_double_well(
    B: int,
    N: int,
    d: int,
    well_separation: float,
    *,
    params=None,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """
    Sample initial configurations for double-well system.

    Each particle is initialized near its assigned well center with
    Gaussian spread σ = σ_scale / √ω.

    Parameters
    ----------
    B : int
        Batch size.
    N : int
        Number of particles (typically 2).
    d : int
        Spatial dimensions (typically 2).
    well_separation : float
        Separation between wells in units of 1/√ω.
    sigma_scale : float
        Scale factor for initial spread.

    Returns
    -------
    torch.Tensor, shape (B, N, d)
    """
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])

    ell = 1.0 / math.sqrt(max(omega, 1e-12))
    sep_phys = well_separation * ell
    sigma = sigma_scale * ell

    x = torch.randn(B, N, d, device=device, dtype=dtype) * sigma

    # Shift particle 0 to left well, particle 1 to right well
    if N >= 1:
        x[:, 0, 0] += -sep_phys / 2.0
    if N >= 2:
        x[:, 1, 0] += +sep_phys / 2.0

    return x


# ===============================================================
# Energy Evaluation for Double Well
# ===============================================================


def _laplacian_logpsi_double_well(psi_log_fn, x: torch.Tensor):
    """
    Exact Laplacian of log(ψ) for double-well system.
    """
    x = x.requires_grad_(True)
    lp = psi_log_fn(x)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True, retain_graph=True)[0]

    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)

    for i in range(N):
        for j in range(d):
            gij = g[:, i, j]
            sec = torch.autograd.grad(gij.sum(), x, create_graph=True, retain_graph=True)[0]
            lap += sec[:, i, j]

    g2 = (g**2).sum(dim=(1, 2), keepdim=True)
    return g, g2, lap.view(B, 1)


@inject_params
def local_energy_double_well(
    psi_log_fn,
    x: torch.Tensor,
    well_separation: float,
    *,
    params=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute local energy for double-well system.

    E_L = T + V_trap + V_int

    where T = -0.5 * (Δlogψ + |∇logψ|²)

    Returns
    -------
    E_L : torch.Tensor, shape (B,)
        Local energy.
    T : torch.Tensor, shape (B,)
        Kinetic energy.
    V_trap : torch.Tensor, shape (B,)
        Trap potential.
    V_int : torch.Tensor, shape (B,)
        Coulomb interaction.
    """
    omega = float(params["omega"])

    # Laplacian and gradient
    g, g2, lap_log = _laplacian_logpsi_double_well(psi_log_fn, x)

    # Kinetic energy
    T = -0.5 * (lap_log.squeeze(-1) + g2.squeeze(-1))

    # Potentials
    V_trap = double_well_potential(x, well_separation, params=params).squeeze(-1)
    V_int = double_well_coulomb_interaction(x, params=params).squeeze(-1)

    E_L = T + V_trap + V_int

    return E_L, T, V_trap, V_int


@inject_params
def evaluate_double_well_energy(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    well_separation: float,
    *,
    psi_fn,
    backflow_net: nn.Module = None,
    spin: torch.Tensor = None,
    params=None,
    n_samples: int = 50000,
    batch_size: int = 1024,
    mcmc_steps: int = 40,
    mcmc_sigma: float = 0.15,
    progress: bool = True,
) -> dict:
    """
    Evaluate ground state energy for double-well system using VMC.

    Parameters
    ----------
    well_separation : float
        Distance between well centers (in units of 1/√ω).

    Returns
    -------
    dict
        Energy statistics: E_mean, E_std, E_stderr, decomposition (T, V_trap, V_int).
    """
    from tqdm.auto import tqdm

    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])

    if spin is None:
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)

    f_net.to(device).to(dtype).eval()
    if backflow_net is not None:
        backflow_net.to(device).to(dtype).eval()

    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi

    # MCMC sampling
    ell = 1.0 / math.sqrt(max(omega, 1e-12))
    sigma_phys = mcmc_sigma * ell

    # Accumulators
    total = 0
    sum_E, sum_E2 = 0.0, 0.0
    sum_T, sum_T2 = 0.0, 0.0
    sum_Vt, sum_Vt2 = 0.0, 0.0
    sum_Vi, sum_Vi2 = 0.0, 0.0

    pbar = (
        tqdm(total=n_samples, desc=f"Double-well (d={well_separation:.2f})", leave=True)
        if progress
        else None
    )

    while total < n_samples:
        bsz = min(batch_size, n_samples - total)

        # Initialize samples near wells
        x = sample_double_well(bsz, N, d, well_separation, params=params)

        # MCMC burn-in
        with torch.no_grad():
            lp = psi_log_fn(x) * 2.0
            for _ in range(mcmc_steps):
                prop = x + torch.randn_like(x) * sigma_phys
                lp_prop = psi_log_fn(prop) * 2.0
                accept = (torch.rand_like(lp_prop).log() < (lp_prop - lp)).view(-1, 1, 1).float()
                x = accept * prop + (1.0 - accept) * x
                lp = accept.view(-1) * lp_prop + (1.0 - accept.view(-1)) * lp

        # Compute local energy
        with torch.set_grad_enabled(True):
            E_L, T, V_trap, V_int = local_energy_double_well(
                psi_log_fn, x, well_separation, params=params
            )

        # Accumulate statistics
        E = E_L.detach().cpu()
        T_cpu = T.detach().cpu()
        Vt_cpu = V_trap.detach().cpu()
        Vi_cpu = V_int.detach().cpu()

        sum_E += float(E.sum().item())
        sum_E2 += float((E**2).sum().item())
        sum_T += float(T_cpu.sum().item())
        sum_T2 += float((T_cpu**2).sum().item())
        sum_Vt += float(Vt_cpu.sum().item())
        sum_Vt2 += float((Vt_cpu**2).sum().item())
        sum_Vi += float(Vi_cpu.sum().item())
        sum_Vi2 += float((Vi_cpu**2).sum().item())
        total += bsz

        if pbar is not None:
            E_mean = sum_E / total
            E_var = max(sum_E2 / total - E_mean**2, 0.0)
            pbar.update(bsz)
            pbar.set_postfix_str(f"E≈{E_mean:.4f}, σ≈{math.sqrt(E_var):.4f}")

    if pbar is not None:
        pbar.close()

    # Finalize statistics
    def finish(s1, s2, n):
        mean = s1 / n
        var = max(s2 / n - mean**2, 0.0)
        std = math.sqrt(var)
        stderr = std / math.sqrt(n)
        return mean, std, stderr

    E_mean, E_std, E_stderr = finish(sum_E, sum_E2, total)
    T_mean, T_std, T_stderr = finish(sum_T, sum_T2, total)
    Vt_mean, Vt_std, Vt_stderr = finish(sum_Vt, sum_Vt2, total)
    Vi_mean, Vi_std, Vi_stderr = finish(sum_Vi, sum_Vi2, total)

    return {
        "well_separation": well_separation,
        "E_mean": E_mean,
        "E_std": E_std,
        "E_stderr": E_stderr,
        "T_mean": T_mean,
        "T_std": T_std,
        "T_stderr": T_stderr,
        "V_trap_mean": Vt_mean,
        "V_trap_std": Vt_std,
        "V_trap_stderr": Vt_stderr,
        "V_int_mean": Vi_mean,
        "V_int_std": Vi_std,
        "V_int_stderr": Vi_stderr,
        "n_samples": total,
    }


# ===============================================================
# Training for Double Well
# ===============================================================


@inject_params
def train_double_well(
    f_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    C_occ: torch.Tensor,
    well_separation: float,
    *,
    psi_fn,
    backflow_net: nn.Module = None,
    spin: torch.Tensor = None,
    params=None,
    n_epochs: int = 1000,
    N_collocation: int = 256,
    micro_batch: int = 64,
    grad_clip: float = 0.5,
    print_every: int = 50,
    E_target: float = None,
    use_huber: bool = True,
    huber_delta: float = 1.0,
) -> list:
    """
    Train the wavefunction ansatz for double-well system.

    Parameters
    ----------
    well_separation : float
        Distance between wells (in units of 1/√ω).
    E_target : float, optional
        Target energy for variance minimization. If None, uses running mean.

    Returns
    -------
    list
        Training history with loss and energy per epoch.
    """
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])

    f_net.to(device).to(dtype)
    if backflow_net is not None:
        backflow_net.to(device).to(dtype)

    if spin is None:
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)

    def psi_log_fn(x):
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi

    def huber_loss(resid, delta):
        abs_r = resid.abs()
        quad = 0.5 * (abs_r.clamp(max=delta) ** 2)
        lin = delta * (abs_r - delta).clamp(min=0.0)
        return quad + lin

    history = []
    ell = 1.0 / math.sqrt(max(omega, 1e-12))

    for epoch in range(n_epochs):
        f_net.train()
        if backflow_net is not None:
            backflow_net.train()

        # Sample configurations
        X = sample_double_well(N_collocation, N, d, well_separation, params=params)

        loss_acc = 0.0
        total_rows = 0
        E_all = []

        for s in range(0, N_collocation, micro_batch):
            e = min(s + micro_batch, N_collocation)
            x = X[s:e].requires_grad_(True)

            # Local energy
            E_L, T, V_trap, V_int = local_energy_double_well(
                psi_log_fn, x, well_separation, params=params
            )

            # Target for variance minimization
            mu = E_L.mean().detach()
            E_eff = mu if E_target is None else E_target

            resid = E_L - E_eff
            if use_huber:
                loss = huber_loss(resid, huber_delta).mean()
            else:
                loss = (resid**2).mean()

            loss.backward()
            loss_acc += float(loss.detach())
            total_rows += E_L.numel()
            E_all.append(E_L.detach())

        # Gradient step
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(f_net.parameters(), grad_clip)
            if backflow_net is not None:
                torch.nn.utils.clip_grad_norm_(backflow_net.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Log
        E_cat = torch.cat(E_all)
        E_mean = float(E_cat.mean().item())
        E_var = float(E_cat.var().item())

        history.append(
            {
                "epoch": epoch,
                "loss": loss_acc / max(1, total_rows),
                "E_mean": E_mean,
                "E_var": E_var,
            }
        )

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch:4d} | E={E_mean:.5f} ± {math.sqrt(E_var):.5f} | loss={loss_acc:.4e}"
            )

    return history


# ===============================================================
# Time Evolution for Double Well
# ===============================================================


@inject_params
def time_evolve_double_well(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    well_separation: float,
    *,
    psi_fn,
    backflow_net: nn.Module = None,
    spin: torch.Tensor = None,
    params=None,
    dt: float = 0.01,
    n_steps: int = 100,
    n_samples: int = 1000,
    observable_fn=None,
) -> dict:
    """
    Real-time evolution using split-operator / imaginary time propagation.

    This implements a simple Euler-based imaginary time evolution for
    studying dynamics in the double-well system.

    Parameters
    ----------
    dt : float
        Time step.
    n_steps : int
        Number of time steps.
    observable_fn : callable, optional
        Function to compute observables at each time step.
        Signature: observable_fn(x, t) -> dict

    Returns
    -------
    dict
        Time evolution results including observables history.
    """
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])

    f_net.to(device).to(dtype).eval()
    if backflow_net is not None:
        backflow_net.to(device).to(dtype).eval()

    if spin is None:
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)

    def psi_log_fn(x):
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi

    ell = 1.0 / math.sqrt(max(omega, 1e-12))

    # Initialize samples
    x = sample_double_well(n_samples, N, d, well_separation, params=params, sigma_scale=0.5)

    # Time evolution using quantum drift-diffusion
    history = {
        "times": [],
        "positions_x": [],  # Mean x positions
        "positions_y": [],  # Mean y positions
        "energies": [],
        "observables": [],
    }

    sigma_drift = 0.1 * ell

    for step in range(n_steps):
        t = step * dt

        # Compute quantum drift (gradient of log|ψ|)
        x_grad = x.detach().requires_grad_(True)
        logpsi = psi_log_fn(x_grad)
        grad_logpsi = torch.autograd.grad(logpsi.sum(), x_grad, create_graph=False)[0]

        # Drift-diffusion step (simplified Langevin dynamics)
        drift = grad_logpsi * dt
        diffusion = math.sqrt(dt) * torch.randn_like(x) * sigma_drift

        x = x + drift + diffusion
        x = x.detach()

        # Record observables
        with torch.no_grad():
            mean_x = x[..., 0].mean(dim=0).cpu().numpy()  # (N,)
            mean_y = x[..., 1].mean(dim=0).cpu().numpy() if d > 1 else np.zeros(N)

            # Estimate energy
            E_L, _, _, _ = local_energy_double_well(
                psi_log_fn, x[:100], well_separation, params=params
            )
            E_mean = float(E_L.mean().item())

        history["times"].append(t)
        history["positions_x"].append(mean_x.tolist())
        history["positions_y"].append(mean_y.tolist())
        history["energies"].append(E_mean)

        if observable_fn is not None:
            obs = observable_fn(x, t)
            history["observables"].append(obs)

    return history


# ===============================================================
# Scan Energy vs Well Separation
# ===============================================================


@inject_params
def scan_energy_vs_separation(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    separations: list,
    *,
    psi_fn,
    backflow_net: nn.Module = None,
    spin: torch.Tensor = None,
    params=None,
    n_samples: int = 10000,
    batch_size: int = 512,
    mcmc_steps: int = 30,
) -> list:
    """
    Compute energy as a function of well separation.

    Expected behavior:
    - d → ∞: E ≈ 2.0 (two independent ground states)
    - d → 0: E ≈ 3.0 (two interacting electrons in single well)

    Parameters
    ----------
    separations : list of float
        Well separations to scan (in units of 1/√ω).

    Returns
    -------
    list of dict
        Energy results for each separation.
    """
    results = []

    for sep in separations:
        print(f"\n=== Well separation d = {sep:.2f} ===")
        result = evaluate_double_well_energy(
            f_net,
            C_occ,
            sep,
            psi_fn=psi_fn,
            backflow_net=backflow_net,
            spin=spin,
            params=params,
            n_samples=n_samples,
            batch_size=batch_size,
            mcmc_steps=mcmc_steps,
            progress=True,
        )
        results.append(result)
        print(f"E = {result['E_mean']:.4f} ± {result['E_stderr']:.4f}")

    return results


def plot_energy_vs_separation(results: list, save_path: str = None):
    """
    Plot energy as a function of well separation.
    """
    import matplotlib.pyplot as plt

    seps = [r["well_separation"] for r in results]
    E_means = [r["E_mean"] for r in results]
    E_errs = [r["E_stderr"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        seps,
        E_means,
        yerr=E_errs,
        fmt="o-",
        capsize=4,
        markersize=8,
        linewidth=2,
        color="blue",
        label="VMC Energy",
    )

    # Reference lines
    ax.axhline(y=2.0, color="green", linestyle="--", linewidth=1.5, label="E=2.0 (separated wells)")
    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1.5, label="E=3.0 (overlapping wells)")

    ax.set_xlabel("Well Separation d (units of $1/\\sqrt{\\omega}$)", fontsize=12)
    ax.set_ylabel("Ground State Energy (a.u.)", fontsize=12)
    ax.set_title("Double-Well System: Energy vs Well Separation", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limits
    ax.set_ylim(1.5, 3.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig, ax
