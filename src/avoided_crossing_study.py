#!/usr/bin/env python
"""
Avoided Crossing Study - Comparison with Jonny's Thesis

This script implements a sweep over a configuration parameter λ to study:
1. Avoided crossings in the energy spectrum (E0, E1, E2 vs λ)
2. State mixing in the "logical" excited subspace (|10⟩ vs |01⟩)
3. Entanglement behavior near the avoided crossing
4. Transverse leakage in 2D (deviation from effective 1D behavior)

The system: Two electrons in a double-well potential with tunable asymmetry.

Key insight: λ controls the relative depth/frequency of left vs right wells,
causing the single-particle excitation energies to become degenerate at some λ*,
leading to an avoided crossing in the two-particle spectrum.

Author: Comparison study for thesis
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Set GPU before importing torch
os.environ["CUDA_MANUAL_DEVICE"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Neural_Networks import psi_fn
from PINN import PINN, CTNNBackflowNet

torch.set_num_threads(4)

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
OMEGA_BASE = 1.0  # Base trap frequency
N_PARTICLES = 2
D = 2  # 2D system (can compare leakage vs 1D)

# Results directory
RESULTS_DIR = Path("/Users/aleksandersekkelsten/thesis/results/avoided_crossing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Asymmetric Double-Well Potential with λ knob
# ============================================================


@dataclass
class DoubleWellConfig:
    """Configuration for asymmetric double-well potential.

    The potential is:
        V(x) = V_left(x) for particle 0 + V_right(x) for particle 1 + Coulomb

    where:
        V_left(r) = 0.5 * ω_left² * |r - r_left|²
        V_right(r) = 0.5 * ω_right² * |r - r_right|²

    The asymmetry parameter λ controls:
        ω_left = ω_base * (1 + λ * asymmetry_strength)
        ω_right = ω_base * (1 - λ * asymmetry_strength)

    So λ=0 gives symmetric wells, λ>0 makes left well tighter (higher ground state),
    and λ<0 makes right well tighter.
    """

    well_separation: float = 4.0  # Distance between wells in units of l_ho
    omega_base: float = 1.0  # Base trap frequency
    asymmetry_strength: float = 0.3  # How much λ affects relative frequencies
    lam: float = 0.0  # The sweep parameter λ ∈ [-1, 1]
    softening: float = 1e-6  # Coulomb softening to prevent divergence

    @property
    def omega_left(self) -> float:
        """Frequency of left well."""
        return self.omega_base * (1.0 + self.lam * self.asymmetry_strength)

    @property
    def omega_right(self) -> float:
        """Frequency of right well."""
        return self.omega_base * (1.0 - self.lam * self.asymmetry_strength)

    @property
    def ell_base(self) -> float:
        """Characteristic length scale (base harmonic oscillator length)."""
        return 1.0 / math.sqrt(self.omega_base)

    @property
    def sep_physical(self) -> float:
        """Physical well separation."""
        return self.well_separation * self.ell_base

    def single_particle_ground_energy(self, which: str) -> float:
        """Ground state energy of single particle in one well (2D HO)."""
        omega = self.omega_left if which == "left" else self.omega_right
        return omega * D / 2  # E = (d/2) * ℏω in 2D

    def expected_E_asymptotic(self) -> tuple[float, float, float]:
        """Expected energies in the non-interacting limit (large separation).

        Returns (E_00, E_10, E_01) where:
        - E_00: both particles in ground state of their wells
        - E_10: left particle excited, right in ground
        - E_01: left in ground, right particle excited
        """
        E_L0 = self.single_particle_ground_energy("left")
        E_R0 = self.single_particle_ground_energy("right")
        E_L1 = E_L0 + self.omega_left  # First excited in left
        E_R1 = E_R0 + self.omega_right  # First excited in right

        E_00 = E_L0 + E_R0
        E_10 = E_L1 + E_R0
        E_01 = E_L0 + E_R1

        return E_00, E_10, E_01


def asymmetric_double_well_potential(x: torch.Tensor, cfg: DoubleWellConfig) -> torch.Tensor:
    """
    Compute asymmetric double-well potential energy.

    Particle 0 → left well at (-d/2, 0) with frequency ω_left
    Particle 1 → right well at (+d/2, 0) with frequency ω_right

    Args:
        x: Positions (B, N, d) where N=2, d=2
        cfg: DoubleWellConfig with asymmetry settings

    Returns:
        V_trap: (B,) trap potential energy
    """
    B, N, d = x.shape
    sep = cfg.sep_physical

    # Displacements from well centers
    r0 = x[:, 0, :].clone()
    r0[:, 0] = r0[:, 0] + sep / 2  # Particle 0 from left well center

    r1 = x[:, 1, :].clone()
    r1[:, 0] = r1[:, 0] - sep / 2  # Particle 1 from right well center

    # Harmonic potential for each particle in its well
    V_left = 0.5 * (cfg.omega_left**2) * (r0**2).sum(dim=-1)
    V_right = 0.5 * (cfg.omega_right**2) * (r1**2).sum(dim=-1)

    return V_left + V_right  # (B,)


def softened_coulomb(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Softened Coulomb repulsion between electrons.

    V = 1 / sqrt(r² + ε²)

    This prevents the 1/r divergence while preserving long-range behavior.
    """
    B, N, d = x.shape
    if N < 2:
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    r2 = ((x[:, 0, :] - x[:, 1, :]) ** 2).sum(dim=-1)
    r_soft = torch.sqrt(r2 + eps**2)
    return 1.0 / r_soft  # (B,)


def total_potential(x: torch.Tensor, cfg: DoubleWellConfig) -> torch.Tensor:
    """Total potential energy = trap + Coulomb."""
    V_trap = asymmetric_double_well_potential(x, cfg)
    V_coul = softened_coulomb(x, cfg.softening)
    return V_trap + V_coul


# ============================================================
# Sampling for Asymmetric Wells
# ============================================================


def sample_asymmetric_positions(
    B: int, cfg: DoubleWellConfig, device=DEVICE, dtype=DTYPE
) -> torch.Tensor:
    """Sample initial positions adapted to asymmetric wells."""
    sep = cfg.sep_physical

    # Widths scaled by local frequency
    sigma_left = 0.5 / math.sqrt(cfg.omega_left)
    sigma_right = 0.5 / math.sqrt(cfg.omega_right)

    x = torch.zeros(B, N_PARTICLES, D, device=device, dtype=dtype)

    # Particle 0 in left well
    x[:, 0, :] = torch.randn(B, D, device=device, dtype=dtype) * sigma_left
    x[:, 0, 0] -= sep / 2

    # Particle 1 in right well
    x[:, 1, :] = torch.randn(B, D, device=device, dtype=dtype) * sigma_right
    x[:, 1, 0] += sep / 2

    return x


# ============================================================
# Model Building (reuse existing architecture)
# ============================================================


def make_cartesian_C_occ(nx, ny, n_occ, device=DEVICE, dtype=DTYPE):
    """Create occupation matrix for Cartesian HO basis."""
    pairs = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    pairs.sort(key=lambda t: (t[0] + t[1], t[0]))
    sel = pairs[:n_occ]
    cols = [ix * ny + iy for (ix, iy) in sel]
    C = torch.zeros(nx * ny, n_occ, dtype=dtype, device=device)
    for j, c in enumerate(cols):
        C[c, j] = 1.0
    return C


def build_model(omega=OMEGA_BASE):
    """Build the CTNN + PINN model."""
    f_net = PINN(
        n_particles=N_PARTICLES,
        d=D,
        omega=omega,
        dL=5,
        hidden_dim=128,
        n_layers=2,
        act="gelu",
        init="xavier",
        use_gate=True,
    ).to(DEVICE, DTYPE)

    backflow_net = CTNNBackflowNet(
        d=D,
        msg_hidden=128,
        msg_layers=2,
        hidden=128,
        layers=3,
        act="gelu",
        aggregation="mean",
        use_spin=True,
        same_spin_only=False,
        out_bound="tanh",
        bf_scale_init=0.3,
        zero_init_last=True,
        omega=omega,
    ).to(DEVICE, DTYPE)

    return f_net, backflow_net


def make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, cfg: DoubleWellConfig):
    """Create log(ψ) closure with coordinate transform for asymmetric double-well."""
    sep = cfg.sep_physical

    def psi_log_fn(x):
        # Transform: shift each particle to its well center
        x_shifted = x.clone()
        x_shifted[:, 0, 0] = x[:, 0, 0] + sep / 2
        x_shifted[:, 1, 0] = x[:, 1, 0] - sep / 2

        # Scale by effective length (helps the Slater basis)
        # x_scaled = x_shifted * math.sqrt(omega_eff)  # Optional scaling

        logpsi, _ = psi_fn(
            f_net, x_shifted, C_occ, backflow_net=backflow_net, spin=spin, params=params
        )
        return logpsi

    return psi_log_fn


# ============================================================
# Energy Computation
# ============================================================


def compute_laplacian_logpsi(psi_log_fn, x):
    """Compute Laplacian of log(psi) exactly."""
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

    g2 = (g**2).sum(dim=(1, 2))
    return lap, g2, g


def local_energy(psi_log_fn, x, cfg: DoubleWellConfig):
    """Compute local energy E_L = T + V for asymmetric double-well."""
    lap_logpsi, grad2_logpsi, _ = compute_laplacian_logpsi(psi_log_fn, x)

    # Kinetic energy: T = -0.5 * (Δlog ψ + |∇log ψ|²)
    T = -0.5 * (lap_logpsi + grad2_logpsi)

    # Potential energies
    V_trap = asymmetric_double_well_potential(x, cfg)
    V_coul = softened_coulomb(x, cfg.softening)

    E_L = T + V_trap + V_coul

    return E_L, T, V_trap, V_coul


# ============================================================
# Ground State Training (standard VMC)
# ============================================================


def train_ground_state(
    f_net,
    backflow_net,
    C_occ,
    cfg: DoubleWellConfig,
    params,
    n_epochs: int = 300,
    n_collocation: int = 256,
    lr: float = 5e-4,
    print_every: int = 100,
):
    """Train wavefunction for ground state using VMC."""
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)

    optimizer = optim.Adam(
        list(f_net.parameters()) + list(backflow_net.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 10)

    psi_log_fn = make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, cfg)

    mcmc_sigma = 0.15 * cfg.ell_base

    history = []
    best_energy = float("inf")
    best_state = None

    # Initialize MCMC chain
    x = sample_asymmetric_positions(n_collocation, cfg)

    for epoch in range(n_epochs):
        f_net.train()
        backflow_net.train()

        # MCMC sampling
        with torch.no_grad():
            logp = 2.0 * psi_log_fn(x)
            for _ in range(20):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(n_collocation, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

        # Compute loss
        optimizer.zero_grad()
        x_batch = x.detach().requires_grad_(True)
        E_L, T, V_trap, V_coul = local_energy(psi_log_fn, x_batch, cfg)

        E_mean = E_L.mean().detach()
        loss = ((E_L - E_mean) ** 2).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(f_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(backflow_net.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        E_mean_val = float(E_mean.item())
        E_std = float(E_L.std().item())

        history.append({"epoch": epoch, "E_mean": E_mean_val, "E_std": E_std})

        if E_mean_val < best_energy:
            best_energy = E_mean_val
            best_state = {
                "f_net": {k: v.clone() for k, v in f_net.state_dict().items()},
                "backflow_net": {k: v.clone() for k, v in backflow_net.state_dict().items()},
            }

        if epoch % print_every == 0:
            print(f"  Epoch {epoch:4d} | E = {E_mean_val:.5f} ± {E_std:.4f}")

    # Restore best
    if best_state:
        f_net.load_state_dict(best_state["f_net"])
        backflow_net.load_state_dict(best_state["backflow_net"])

    return history, best_energy


# ============================================================
# Excited State Training (with orthogonality penalty)
# ============================================================


def compute_overlap(psi1_log_fn, psi2_log_fn, x: torch.Tensor) -> torch.Tensor:
    """Estimate |⟨ψ₁|ψ₂⟩|² using importance sampling.

    Using the identity:
        ⟨ψ₁|ψ₂⟩ = E_{|ψ₁|²}[ψ₂/ψ₁]

    For monitoring (no grad needed).
    """
    with torch.no_grad():
        log1 = psi1_log_fn(x)
        log2 = psi2_log_fn(x)

        # Stabilized ratio computation
        log_ratio = log2 - log1
        # Clip to prevent overflow
        log_ratio = torch.clamp(log_ratio, -20, 20)
        ratio = torch.exp(log_ratio)
        overlap = ratio.mean()

    return overlap


def compute_overlap_penalty_differentiable(
    psi_log_fn_exc,
    psi_log_fn_lower,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable overlap penalty for excited state training.

    Uses the "penalty method": minimize ⟨ψ_exc|P_lower|ψ_exc⟩ where P_lower
    projects onto the lower state subspace.

    Approximation: |⟨ψ_exc|ψ_lower⟩|² ≈ E_{|ψ_exc|²}[(ψ_lower/ψ_exc)²]

    This is computed with gradients flowing through ψ_exc.
    """
    # Get log values
    log_exc = psi_log_fn_exc(x)  # With gradients
    with torch.no_grad():
        log_lower = psi_log_fn_lower(x)  # Frozen lower state

    # Compute log(ψ_lower²/ψ_exc²) = 2*(log_lower - log_exc)
    log_ratio_sq = 2.0 * (log_lower - log_exc)
    # Stabilize
    log_ratio_sq = torch.clamp(log_ratio_sq, -40, 40)

    # ⟨ψ_lower|ψ_exc⟩² ≈ E_{|ψ_exc|²}[|ψ_lower/ψ_exc|²]
    overlap_sq = torch.exp(log_ratio_sq).mean()

    return overlap_sq


def train_excited_state(
    f_net_excited,
    backflow_net_excited,
    C_occ,
    cfg: DoubleWellConfig,
    params,
    lower_states: list,  # List of (f_net, backflow_net, psi_log_fn) for orthogonality
    n_epochs: int = 400,
    n_collocation: int = 256,
    lr: float = 3e-4,
    orthog_penalty: float = 50.0,  # Increased penalty
    print_every: int = 100,
):
    """Train excited state with orthogonality penalty to lower states.

    Uses "penalty method" for excited states:
        Loss = Var(E_L) + λ * Σ_i |⟨ψ_i|ψ_exc⟩|²

    The penalty forces orthogonality while variance minimization finds
    the lowest energy state orthogonal to lower states.
    """
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)

    optimizer = optim.Adam(
        list(f_net_excited.parameters()) + list(backflow_net_excited.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 10)

    psi_log_fn_exc = make_psi_log_fn(f_net_excited, C_occ, backflow_net_excited, spin, params, cfg)

    mcmc_sigma = 0.15 * cfg.ell_base

    history = []
    best_energy = float("inf")
    best_state = None

    x = sample_asymmetric_positions(n_collocation, cfg)

    # Also sample from lower states for better overlap estimation
    x_lower = sample_asymmetric_positions(n_collocation, cfg)

    for epoch in range(n_epochs):
        f_net_excited.train()
        backflow_net_excited.train()

        # MCMC sampling from |ψ_exc|²
        with torch.no_grad():
            logp = 2.0 * psi_log_fn_exc(x)
            for _ in range(20):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn_exc(x_prop)
                accept = torch.rand(n_collocation, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

        # Also update x_lower by sampling from lower states
        if lower_states:
            with torch.no_grad():
                _, _, psi_log_lower_0 = lower_states[0]
                logp_l = 2.0 * psi_log_lower_0(x_lower)
                for _ in range(10):
                    x_prop_l = x_lower + torch.randn_like(x_lower) * mcmc_sigma
                    logp_prop_l = 2.0 * psi_log_lower_0(x_prop_l)
                    accept_l = torch.rand(n_collocation, device=DEVICE).log() < (
                        logp_prop_l - logp_l
                    )
                    x_lower = torch.where(accept_l.view(-1, 1, 1), x_prop_l, x_lower)
                    logp_l = torch.where(accept_l, logp_prop_l, logp_l)

        optimizer.zero_grad()
        x_batch = x.detach().requires_grad_(True)

        # Energy term (variance minimization)
        E_L, _, _, _ = local_energy(psi_log_fn_exc, x_batch, cfg)
        E_mean = E_L.mean()
        loss = ((E_L - E_mean.detach()) ** 2).mean()

        # Orthogonality penalty (differentiable)
        orthog_loss = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        for _, _, psi_log_fn_lower in lower_states:
            # Sample from excited state distribution
            overlap_sq = compute_overlap_penalty_differentiable(
                psi_log_fn_exc, psi_log_fn_lower, x.detach()
            )
            orthog_loss = orthog_loss + overlap_sq

        total_loss = loss + orthog_penalty * orthog_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(f_net_excited.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(backflow_net_excited.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        E_mean_val = float(E_mean.detach().item())
        E_std = float(E_L.std().item())

        history.append({"epoch": epoch, "E_mean": E_mean_val, "E_std": E_std})

        if E_mean_val < best_energy:
            best_energy = E_mean_val
            best_state = {
                "f_net": {k: v.clone() for k, v in f_net_excited.state_dict().items()},
                "backflow_net": {
                    k: v.clone() for k, v in backflow_net_excited.state_dict().items()
                },
            }

        if epoch % print_every == 0:
            # Monitor overlaps (non-differentiable, just for display)
            overlaps_str = ", ".join(
                [
                    f"{float(compute_overlap(psi_log_fn_exc, psi, x.detach()).item()):.3f}"
                    for _, _, psi in lower_states
                ]
            )
            print(f"  Epoch {epoch:4d} | E = {E_mean_val:.5f} | [{overlaps_str}]")

    # Restore best
    if best_state:
        f_net_excited.load_state_dict(best_state["f_net"])
        backflow_net_excited.load_state_dict(best_state["backflow_net"])

    return history, best_energy


# ============================================================
# Diagnostics
# ============================================================


def evaluate_energy_precise(psi_log_fn, cfg: DoubleWellConfig, n_samples: int = 50000):
    """Precise energy evaluation with error bars."""
    mcmc_sigma = 0.12 * cfg.ell_base
    batch_size = 1024

    sum_E, sum_E2 = 0.0, 0.0
    total = 0

    x = sample_asymmetric_positions(batch_size, cfg)

    # Burn-in
    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(200):
            x_prop = x + torch.randn_like(x) * mcmc_sigma
            logp_prop = 2.0 * psi_log_fn(x_prop)
            accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
            x = torch.where(accept.view(-1, 1, 1), x_prop, x)
            logp = torch.where(accept, logp_prop, logp)

    # Sampling
    while total < n_samples:
        with torch.no_grad():
            for _ in range(10):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

        with torch.set_grad_enabled(True):
            x_eval = x.detach().requires_grad_(True)
            E_L, _, _, _ = local_energy(psi_log_fn, x_eval, cfg)

        E = E_L.detach()
        sum_E += float(E.sum().item())
        sum_E2 += float((E**2).sum().item())
        total += batch_size

    E_mean = sum_E / total
    E_var = max(sum_E2 / total - E_mean**2, 0.0)
    E_std = math.sqrt(E_var)
    E_stderr = E_std / math.sqrt(total)

    return E_mean, E_stderr


def compute_mixing_weights(
    psi_log_fn,
    cfg: DoubleWellConfig,
    n_samples: int = 10000,
) -> dict:
    """Compute weights in the "logical" excited subspace.

    We define reference configurations:
    - |10⟩: Left particle excited (further from well center), right in ground
    - |01⟩: Left in ground, right particle excited

    We estimate this by looking at the density in different regions:
    - P_L_excited: probability that left particle is far from its well center
    - P_R_excited: probability that right particle is far from its well center

    This is a proxy for the mixing coefficients w_10 and w_01.
    """
    mcmc_sigma = 0.12 * cfg.ell_base
    batch_size = 1024
    sep = cfg.sep_physical

    # Thresholds for "excited" (further than 1 sigma from well center)
    sigma_left = 1.0 / math.sqrt(cfg.omega_left)
    sigma_right = 1.0 / math.sqrt(cfg.omega_right)

    x = sample_asymmetric_positions(batch_size, cfg)

    # Burn-in
    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(100):
            x_prop = x + torch.randn_like(x) * mcmc_sigma
            logp_prop = 2.0 * psi_log_fn(x_prop)
            accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
            x = torch.where(accept.view(-1, 1, 1), x_prop, x)
            logp = torch.where(accept, logp_prop, logp)

    # Collect statistics
    left_excited_count = 0
    right_excited_count = 0
    both_ground_count = 0
    both_excited_count = 0
    total = 0

    while total < n_samples:
        with torch.no_grad():
            for _ in range(5):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

            # Distance from well centers
            r_left = torch.sqrt((x[:, 0, 0] + sep / 2) ** 2 + x[:, 0, 1] ** 2)
            r_right = torch.sqrt((x[:, 1, 0] - sep / 2) ** 2 + x[:, 1, 1] ** 2)

            # Check if "excited" (beyond threshold)
            threshold = 1.5  # in units of local sigma
            L_exc = r_left > threshold * sigma_left
            R_exc = r_right > threshold * sigma_right

            left_excited_count += int((L_exc & ~R_exc).sum().item())
            right_excited_count += int((~L_exc & R_exc).sum().item())
            both_ground_count += int((~L_exc & ~R_exc).sum().item())
            both_excited_count += int((L_exc & R_exc).sum().item())
            total += batch_size

    # Normalize
    total_exc = left_excited_count + right_excited_count
    if total_exc > 0:
        w_10 = left_excited_count / total_exc
        w_01 = right_excited_count / total_exc
    else:
        w_10 = w_01 = 0.5

    return {
        "w_10": w_10,  # Weight on |10⟩ (left excited)
        "w_01": w_01,  # Weight on |01⟩ (right excited)
        "p_ground": both_ground_count / total,
        "p_L_exc": left_excited_count / total,
        "p_R_exc": right_excited_count / total,
        "p_both_exc": both_excited_count / total,
    }


def compute_entanglement_proxy(
    psi_log_fn,
    cfg: DoubleWellConfig,
    n_samples: int = 10000,
) -> float:
    """Compute a proxy for entanglement: correlation between left and right excitations.

    For separable states: ⟨n_L n_R⟩ = ⟨n_L⟩⟨n_R⟩
    For entangled states: ⟨n_L n_R⟩ ≠ ⟨n_L⟩⟨n_R⟩

    We compute the covariance and normalize it.
    """
    mcmc_sigma = 0.12 * cfg.ell_base
    batch_size = 1024
    sep = cfg.sep_physical

    sigma_left = 1.0 / math.sqrt(cfg.omega_left)
    sigma_right = 1.0 / math.sqrt(cfg.omega_right)

    x = sample_asymmetric_positions(batch_size, cfg)

    # Burn-in
    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(100):
            x_prop = x + torch.randn_like(x) * mcmc_sigma
            logp_prop = 2.0 * psi_log_fn(x_prop)
            accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
            x = torch.where(accept.view(-1, 1, 1), x_prop, x)
            logp = torch.where(accept, logp_prop, logp)

    # Collect samples
    n_L_list: list[torch.Tensor] = []
    n_R_list: list[torch.Tensor] = []

    while len(n_L_list) * batch_size < n_samples:
        with torch.no_grad():
            for _ in range(5):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

            # "Excitation" proxies (normalized distance from center)
            r_left = torch.sqrt((x[:, 0, 0] + sep / 2) ** 2 + x[:, 0, 1] ** 2) / sigma_left
            r_right = torch.sqrt((x[:, 1, 0] - sep / 2) ** 2 + x[:, 1, 1] ** 2) / sigma_right

            n_L_list.append(r_left.cpu())
            n_R_list.append(r_right.cpu())

    n_L = torch.cat(n_L_list)
    n_R = torch.cat(n_R_list)

    # Compute correlation
    cov = float(((n_L - n_L.mean()) * (n_R - n_R.mean())).mean().item())
    std_L = float(n_L.std().item())
    std_R = float(n_R.std().item())

    if std_L > 0 and std_R > 0:
        correlation = cov / (std_L * std_R)
    else:
        correlation = 0.0

    # Convert to entanglement-like measure (0 for uncorrelated, higher for correlated)
    entanglement_proxy = abs(correlation)

    return entanglement_proxy


def compute_transverse_leakage(
    psi_log_fn,
    cfg: DoubleWellConfig,
    n_samples: int = 10000,
) -> float:
    """Compute transverse leakage: how much the state spills into y-modes.

    In the 1D limit, particles should stay on the x-axis.
    Leakage L = ⟨y²⟩ / ⟨x² + y²⟩

    For 1D-like behavior: L → 0
    For 2D spreading: L → 0.5 (isotropic)
    """
    mcmc_sigma = 0.12 * cfg.ell_base
    batch_size = 1024
    sep = cfg.sep_physical

    x = sample_asymmetric_positions(batch_size, cfg)

    # Burn-in
    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(100):
            x_prop = x + torch.randn_like(x) * mcmc_sigma
            logp_prop = 2.0 * psi_log_fn(x_prop)
            accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
            x = torch.where(accept.view(-1, 1, 1), x_prop, x)
            logp = torch.where(accept, logp_prop, logp)

    sum_y2 = 0.0
    sum_total = 0.0
    total = 0

    while total < n_samples:
        with torch.no_grad():
            for _ in range(5):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

            # Compute y² contribution (relative to well centers)
            y_left = x[:, 0, 1]
            y_right = x[:, 1, 1]

            # Distance from well centers
            x_left_disp = x[:, 0, 0] + sep / 2
            x_right_disp = x[:, 1, 0] - sep / 2

            y2 = y_left**2 + y_right**2
            total_disp2 = x_left_disp**2 + y_left**2 + x_right_disp**2 + y_right**2

            sum_y2 += float(y2.sum().item())
            sum_total += float(total_disp2.sum().item())
            total += batch_size

    if sum_total > 0:
        leakage = sum_y2 / sum_total
    else:
        leakage = 0.0

    return leakage


# ============================================================
# Main Sweep Function
# ============================================================


def run_lambda_sweep(
    lambda_values: list,
    well_separation: float = 4.0,
    n_epochs_ground: int = 300,
    n_epochs_excited: int = 400,
    n_eval_samples: int = 30000,
):
    """Run the full λ sweep and compute all diagnostics.

    For each λ:
    1. Train ground state (E0)
    2. Train first excited state (E1) orthogonal to E0
    3. Train second excited state (E2) orthogonal to E0, E1
    4. Compute diagnostics: gap, mixing, entanglement, leakage
    """
    print("=" * 70)
    print("AVOIDED CROSSING STUDY - λ SWEEP")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Well separation: {well_separation}")
    print(f"λ values: {lambda_values}")
    print()

    # Setup config
    config.update(
        device=DEVICE,
        omega=OMEGA_BASE,
        n_particles=N_PARTICLES,
        d=D,
        basis="cart",
        nx=2,
        ny=2,
    )
    cfg_base = config.get()
    params = cfg_base.as_dict()

    C_occ = make_cartesian_C_occ(2, 2, 1, device=DEVICE, dtype=DTYPE)

    results = []

    for lam in lambda_values:
        print(f"\n{'='*70}")
        print(f"λ = {lam:.3f}")
        print(f"{'='*70}")

        # Create config for this λ
        cfg = DoubleWellConfig(
            well_separation=well_separation,
            omega_base=OMEGA_BASE,
            asymmetry_strength=0.3,
            lam=lam,
        )

        print(f"ω_left = {cfg.omega_left:.3f}, ω_right = {cfg.omega_right:.3f}")
        E00, E10, E01 = cfg.expected_E_asymptotic()
        print(f"Expected asymptotic: E_00 = {E00:.3f}, E_10 = {E10:.3f}, E_01 = {E01:.3f}")

        spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)

        # --- Train ground state ---
        print("\n[Ground State]")
        f_net_0, bf_net_0 = build_model(OMEGA_BASE)
        _, E0_train = train_ground_state(
            f_net_0, bf_net_0, C_occ, cfg, params, n_epochs=n_epochs_ground, print_every=100
        )

        psi_log_0 = make_psi_log_fn(f_net_0, C_occ, bf_net_0, spin, params, cfg)
        E0, E0_err = evaluate_energy_precise(psi_log_0, cfg, n_eval_samples)
        print(f"  E0 = {E0:.5f} ± {E0_err:.5f}")

        # --- Train first excited state ---
        print("\n[First Excited State]")
        f_net_1, bf_net_1 = build_model(OMEGA_BASE)
        lower_states_1 = [(f_net_0, bf_net_0, psi_log_0)]

        _, E1_train = train_excited_state(
            f_net_1,
            bf_net_1,
            C_occ,
            cfg,
            params,
            lower_states=lower_states_1,
            n_epochs=n_epochs_excited,
            print_every=100,
        )

        psi_log_1 = make_psi_log_fn(f_net_1, C_occ, bf_net_1, spin, params, cfg)
        E1, E1_err = evaluate_energy_precise(psi_log_1, cfg, n_eval_samples)
        print(f"  E1 = {E1:.5f} ± {E1_err:.5f}")

        # --- Train second excited state ---
        print("\n[Second Excited State]")
        f_net_2, bf_net_2 = build_model(OMEGA_BASE)
        lower_states_2 = [
            (f_net_0, bf_net_0, psi_log_0),
            (f_net_1, bf_net_1, psi_log_1),
        ]

        _, E2_train = train_excited_state(
            f_net_2,
            bf_net_2,
            C_occ,
            cfg,
            params,
            lower_states=lower_states_2,
            n_epochs=n_epochs_excited,
            print_every=100,
        )

        psi_log_2 = make_psi_log_fn(f_net_2, C_occ, bf_net_2, spin, params, cfg)
        E2, E2_err = evaluate_energy_precise(psi_log_2, cfg, n_eval_samples)
        print(f"  E2 = {E2:.5f} ± {E2_err:.5f}")

        # --- Compute diagnostics ---
        print("\n[Diagnostics]")

        # Gap
        gap = E2 - E1
        print(f"  Gap Δ = E2 - E1 = {gap:.5f}")

        # Mixing weights for E1 and E2
        mix_1 = compute_mixing_weights(psi_log_1, cfg, n_samples=10000)
        mix_2 = compute_mixing_weights(psi_log_2, cfg, n_samples=10000)
        print(f"  E1 mixing: w_10={mix_1['w_10']:.3f}, w_01={mix_1['w_01']:.3f}")
        print(f"  E2 mixing: w_10={mix_2['w_10']:.3f}, w_01={mix_2['w_01']:.3f}")

        # Entanglement proxy for E1 and E2
        ent_1 = compute_entanglement_proxy(psi_log_1, cfg, n_samples=10000)
        ent_2 = compute_entanglement_proxy(psi_log_2, cfg, n_samples=10000)
        print(f"  Entanglement proxy: E1={ent_1:.3f}, E2={ent_2:.3f}")

        # Transverse leakage
        leak_0 = compute_transverse_leakage(psi_log_0, cfg, n_samples=10000)
        leak_1 = compute_transverse_leakage(psi_log_1, cfg, n_samples=10000)
        leak_2 = compute_transverse_leakage(psi_log_2, cfg, n_samples=10000)
        print(f"  Transverse leakage: E0={leak_0:.3f}, E1={leak_1:.3f}, E2={leak_2:.3f}")

        # Store results
        results.append(
            {
                "lambda": lam,
                "omega_left": cfg.omega_left,
                "omega_right": cfg.omega_right,
                "E0": E0,
                "E0_err": E0_err,
                "E1": E1,
                "E1_err": E1_err,
                "E2": E2,
                "E2_err": E2_err,
                "gap": gap,
                "E1_w10": mix_1["w_10"],
                "E1_w01": mix_1["w_01"],
                "E2_w10": mix_2["w_10"],
                "E2_w01": mix_2["w_01"],
                "E1_entanglement": ent_1,
                "E2_entanglement": ent_2,
                "E0_leakage": leak_0,
                "E1_leakage": leak_1,
                "E2_leakage": leak_2,
            }
        )

        # Save intermediate results
        with open(RESULTS_DIR / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


# ============================================================
# Plotting Functions
# ============================================================


def create_comparison_plots(results: list, save_dir: Path = RESULTS_DIR):
    """Create the 5 comparison plots for Jonny's thesis comparison."""

    lambdas = [r["lambda"] for r in results]
    E0 = [r["E0"] for r in results]
    E1 = [r["E1"] for r in results]
    E2 = [r["E2"] for r in results]
    gaps = [r["gap"] for r in results]

    E1_w10 = [r["E1_w10"] for r in results]
    E1_w01 = [r["E1_w01"] for r in results]
    E2_w10 = [r["E2_w10"] for r in results]
    E2_w01 = [r["E2_w01"] for r in results]

    E1_ent = [r["E1_entanglement"] for r in results]
    E2_ent = [r["E2_entanglement"] for r in results]

    leak_0 = [r["E0_leakage"] for r in results]
    leak_1 = [r["E1_leakage"] for r in results]
    leak_2 = [r["E2_leakage"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Energy spectrum (avoided crossing)
    ax = axes[0, 0]
    ax.plot(lambdas, E0, "b-o", label=r"$E_0$ (ground)", markersize=4)
    ax.plot(lambdas, E1, "r-s", label=r"$E_1$", markersize=4)
    ax.plot(lambdas, E2, "g-^", label=r"$E_2$", markersize=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Energy")
    ax.set_title(r"(a) Energy Spectrum $E_n(\lambda)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gap (avoided crossing signature)
    ax = axes[0, 1]
    ax.plot(lambdas, gaps, "k-o", markersize=5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta = E_2 - E_1$")
    ax.set_title(r"(b) Energy Gap $\Delta(\lambda)$")
    ax.grid(True, alpha=0.3)

    # Find minimum gap
    min_gap_idx = np.argmin(gaps)
    ax.axvline(
        lambdas[min_gap_idx],
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"$\lambda^* = {lambdas[min_gap_idx]:.2f}$",
    )
    ax.legend()

    # Plot 3: Mixing weights for E1
    ax = axes[0, 2]
    ax.plot(lambdas, E1_w10, "b-o", label=r"$w_{10}$ (left exc.)", markersize=4)
    ax.plot(lambdas, E1_w01, "r-s", label=r"$w_{01}$ (right exc.)", markersize=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Mixing weight")
    ax.set_title(r"(c) State $E_1$ Mixing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 4: Mixing weights for E2
    ax = axes[1, 0]
    ax.plot(lambdas, E2_w10, "b-o", label=r"$w_{10}$ (left exc.)", markersize=4)
    ax.plot(lambdas, E2_w01, "r-s", label=r"$w_{01}$ (right exc.)", markersize=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Mixing weight")
    ax.set_title(r"(d) State $E_2$ Mixing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 5: Entanglement
    ax = axes[1, 1]
    ax.plot(lambdas, E1_ent, "r-s", label=r"$E_1$", markersize=4)
    ax.plot(lambdas, E2_ent, "g-^", label=r"$E_2$", markersize=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Entanglement proxy")
    ax.set_title(r"(e) Entanglement vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(lambdas[min_gap_idx], color="gray", linestyle="--", alpha=0.5)

    # Plot 6: Transverse leakage (2D-only)
    ax = axes[1, 2]
    ax.plot(lambdas, leak_0, "b-o", label=r"$E_0$", markersize=4)
    ax.plot(lambdas, leak_1, "r-s", label=r"$E_1$", markersize=4)
    ax.plot(lambdas, leak_2, "g-^", label=r"$E_2$", markersize=4)
    ax.axhline(0.5, color="gray", linestyle=":", label="2D isotropic")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Leakage $L = \langle y^2 \rangle / \langle r^2 \rangle$")
    ax.set_title(r"(f) Transverse Leakage (2D$\rightarrow$1D)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)

    plt.tight_layout()
    plt.savefig(save_dir / "avoided_crossing_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(save_dir / "avoided_crossing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to {save_dir}")


# ============================================================
# Quick Test Mode
# ============================================================


def quick_test():
    """Run a quick test with just a few λ values to verify everything works."""
    print("=" * 70)
    print("QUICK TEST MODE")
    print("=" * 70)

    # Just 3 λ values for testing
    lambda_values = [-0.5, 0.0, 0.5]

    results = run_lambda_sweep(
        lambda_values,
        well_separation=4.0,
        n_epochs_ground=100,  # Reduced for testing
        n_epochs_excited=150,
        n_eval_samples=10000,
    )

    create_comparison_plots(results)

    print("\n" + "=" * 70)
    print("QUICK TEST COMPLETE")
    print("=" * 70)

    return results


# ============================================================
# Main Entry Point
# ============================================================


def main():
    """Full production run."""
    # Sweep λ from -1 to +1 with enough points to see the avoided crossing
    lambda_values = np.linspace(-0.8, 0.8, 9).tolist()

    results = run_lambda_sweep(
        lambda_values,
        well_separation=4.0,
        n_epochs_ground=300,
        n_epochs_excited=400,
        n_eval_samples=30000,
    )

    create_comparison_plots(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    gaps = [r["gap"] for r in results]
    min_idx = np.argmin(gaps)

    print(f"Minimum gap at λ* = {results[min_idx]['lambda']:.3f}")
    print(f"  Gap Δ = {results[min_idx]['gap']:.5f}")
    print(
        f"  E1 mixing: w_10={results[min_idx]['E1_w10']:.3f}, w_01={results[min_idx]['E1_w01']:.3f}"
    )
    print(
        f"  E2 mixing: w_10={results[min_idx]['E2_w10']:.3f}, w_01={results[min_idx]['E2_w01']:.3f}"
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avoided Crossing Study")
    parser.add_argument("--quick", action="store_true", help="Run quick test mode")
    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        main()
