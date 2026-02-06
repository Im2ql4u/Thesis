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
        for _ in range(50):
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


def compute_mixing_weights_proper(
    psi_log_fn,
    cfg: DoubleWellConfig,
    n_samples: int = 5000,
) -> dict:
    """Compute proper 2-level subspace projection for mixing weights.

    We define the logical basis states:
    - |10⟩: Left particle in first excited state (n=1), right in ground (n=0)
    - |01⟩: Left in ground, right in first excited state

    For a 2D harmonic oscillator, excited means higher radial quantum number.
    We estimate this by computing:
    1. W_sub = total weight in the {|10⟩, |01⟩} subspace
    2. Within that subspace: Ψ_∥ = a|10⟩ + b|01⟩, report |a|², |b|²

    The key insight: For proper 2-level physics, we need W_sub to be large
    and |a|² + |b|² ≈ 1 within the subspace (after normalization).
    """
    mcmc_sigma = 0.12 * cfg.ell_base
    batch_size = 1024
    sep = cfg.sep_physical

    # Length scales for each well
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

    # Collect detailed statistics
    # We classify each sample into configurations based on radial excitation
    # Ground state: r < threshold, Excited: r > threshold
    threshold_ground = 1.2  # in units of local sigma (conservative)
    threshold_excited_min = 1.2
    threshold_excited_max = 3.0  # Not too far out

    counts = {
        "00": 0,  # Both ground
        "10": 0,  # Left excited only
        "01": 0,  # Right excited only
        "11": 0,  # Both excited
        "other": 0,  # Ambiguous or outside clean classification
    }
    total = 0

    # Also track continuous "excitation" measures for better estimates
    sum_nL = 0.0  # Mean excitation of left particle
    sum_nR = 0.0  # Mean excitation of right particle
    sum_nL_sq = 0.0
    sum_nR_sq = 0.0

    while total < n_samples:
        with torch.no_grad():
            for _ in range(5):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

            # Radial distance from well centers (normalized by sigma)
            r_L = torch.sqrt((x[:, 0, 0] + sep / 2) ** 2 + x[:, 0, 1] ** 2) / sigma_left
            r_R = torch.sqrt((x[:, 1, 0] - sep / 2) ** 2 + x[:, 1, 1] ** 2) / sigma_right

            # Classification based on radial position
            L_ground = r_L < threshold_ground
            L_excited = (r_L >= threshold_excited_min) & (r_L < threshold_excited_max)
            R_ground = r_R < threshold_ground
            R_excited = (r_R >= threshold_excited_min) & (r_R < threshold_excited_max)

            # Count configurations
            counts["00"] += int((L_ground & R_ground).sum().item())
            counts["10"] += int((L_excited & R_ground).sum().item())
            counts["01"] += int((L_ground & R_excited).sum().item())
            counts["11"] += int((L_excited & R_excited).sum().item())

            # "Other" = doesn't fit clean classification
            clean = (L_ground | L_excited) & (R_ground | R_excited)
            counts["other"] += int((~clean).sum().item())

            # Continuous excitation measures (proxy for n_L, n_R)
            # Use r²/(2σ²) as proxy for oscillator quantum number
            n_L_proxy = (r_L**2) / 2.0
            n_R_proxy = (r_R**2) / 2.0

            sum_nL += float(n_L_proxy.sum().item())
            sum_nR += float(n_R_proxy.sum().item())
            sum_nL_sq += float((n_L_proxy**2).sum().item())
            sum_nR_sq += float((n_R_proxy**2).sum().item())

            total += batch_size

    # Compute subspace weight W_sub = P(|10⟩) + P(|01⟩)
    total_classified = sum(counts.values()) - counts["other"]
    if total_classified > 0:
        p_00 = counts["00"] / total_classified
        p_10 = counts["10"] / total_classified
        p_01 = counts["01"] / total_classified
        p_11 = counts["11"] / total_classified
    else:
        p_00 = p_10 = p_01 = p_11 = 0.25

    W_sub = p_10 + p_01  # Total weight in the 2-level subspace

    # Normalized mixing within the subspace
    if W_sub > 1e-6:
        a_sq = p_10 / W_sub  # |a|² = P(|10⟩) / W_sub
        b_sq = p_01 / W_sub  # |b|² = P(|01⟩) / W_sub
    else:
        a_sq = b_sq = 0.5

    # Continuous measures
    mean_nL = sum_nL / total
    mean_nR = sum_nR / total

    return {
        # Proper 2-level subspace analysis
        "W_sub": W_sub,  # Total weight in {|10⟩, |01⟩} subspace
        "a_sq": a_sq,  # |⟨10|Ψ⟩|² / W_sub (normalized left excitation)
        "b_sq": b_sq,  # |⟨01|Ψ⟩|² / W_sub (normalized right excitation)
        # Raw probabilities
        "p_00": p_00,
        "p_10": p_10,
        "p_01": p_01,
        "p_11": p_11,
        "p_other": counts["other"] / total,
        # Continuous excitation measures
        "mean_nL": mean_nL,
        "mean_nR": mean_nR,
    }


def compute_entanglement_proxy(
    psi_log_fn,
    cfg: DoubleWellConfig,
    n_samples: int = 5000,
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
    n_samples: int = 5000,
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
    n_diag_samples: int = 5000,
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

        # --- Sort energies for consistent gap definition ---
        # E_lower = min(E1, E2), E_upper = max(E1, E2)
        if E1 <= E2:
            E_lower, E_lower_err = E1, E1_err
            E_upper, E_upper_err = E2, E2_err
            psi_lower, psi_upper = psi_log_1, psi_log_2
        else:
            E_lower, E_lower_err = E2, E2_err
            E_upper, E_upper_err = E1, E1_err
            psi_lower, psi_upper = psi_log_2, psi_log_1

        # --- Compute diagnostics ---
        print("\n[Diagnostics]")

        # Proper gap (always positive)
        gap = E_upper - E_lower
        print(f"  Gap Δ = E_upper - E_lower = {gap:.5f}")
        print(f"    E_lower = {E_lower:.5f}, E_upper = {E_upper:.5f}")

        # Proper mixing weights using 2-level subspace projection
        mix_lower = compute_mixing_weights_proper(psi_lower, cfg, n_samples=n_diag_samples)
        mix_upper = compute_mixing_weights_proper(psi_upper, cfg, n_samples=n_diag_samples)

        print(f"  E_lower: W_sub={mix_lower['W_sub']:.3f}, "
              f"a²={mix_lower['a_sq']:.3f}, b²={mix_lower['b_sq']:.3f}")
        print(f"  E_upper: W_sub={mix_upper['W_sub']:.3f}, "
              f"a²={mix_upper['a_sq']:.3f}, b²={mix_upper['b_sq']:.3f}")

        # Entanglement proxy for both excited states
        ent_lower = compute_entanglement_proxy(psi_lower, cfg, n_samples=n_diag_samples)
        ent_upper = compute_entanglement_proxy(psi_upper, cfg, n_samples=n_diag_samples)
        print(f"  Entanglement proxy: lower={ent_lower:.3f}, upper={ent_upper:.3f}")

        # Transverse leakage
        leak_0 = compute_transverse_leakage(psi_log_0, cfg, n_samples=n_diag_samples)
        leak_lower = compute_transverse_leakage(psi_lower, cfg, n_samples=n_diag_samples)
        leak_upper = compute_transverse_leakage(psi_upper, cfg, n_samples=n_diag_samples)
        print(f"  Transverse leakage: E0={leak_0:.3f}, "
              f"lower={leak_lower:.3f}, upper={leak_upper:.3f}")

        # Store results with both raw and sorted data
        results.append(
            {
                "lambda": lam,
                "omega_left": cfg.omega_left,
                "omega_right": cfg.omega_right,
                # Raw energies (as trained)
                "E0": E0, "E0_err": E0_err,
                "E1_raw": E1, "E1_raw_err": E1_err,
                "E2_raw": E2, "E2_raw_err": E2_err,
                # Sorted energies (for consistent gap)
                "E_lower": E_lower, "E_lower_err": E_lower_err,
                "E_upper": E_upper, "E_upper_err": E_upper_err,
                "gap": gap,  # Always positive
                # Proper 2-level subspace mixing (for sorted states)
                "lower_W_sub": mix_lower["W_sub"],
                "lower_a_sq": mix_lower["a_sq"],
                "lower_b_sq": mix_lower["b_sq"],
                "upper_W_sub": mix_upper["W_sub"],
                "upper_a_sq": mix_upper["a_sq"],
                "upper_b_sq": mix_upper["b_sq"],
                # Entanglement
                "ent_lower": ent_lower,
                "ent_upper": ent_upper,
                # Leakage
                "E0_leakage": leak_0,
                "lower_leakage": leak_lower,
                "upper_leakage": leak_upper,
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
    """Create comparison plots with proper energy ordering and 2-level analysis."""

    lambdas = [r["lambda"] for r in results]
    E0 = [r["E0"] for r in results]

    # Use sorted energies for consistent plotting
    E_lower = [r["E_lower"] for r in results]
    E_upper = [r["E_upper"] for r in results]
    gaps = [r["gap"] for r in results]  # Now always positive

    # 2-level subspace analysis
    lower_a_sq = [r["lower_a_sq"] for r in results]
    lower_b_sq = [r["lower_b_sq"] for r in results]
    upper_a_sq = [r["upper_a_sq"] for r in results]
    upper_b_sq = [r["upper_b_sq"] for r in results]
    lower_W_sub = [r["lower_W_sub"] for r in results]
    upper_W_sub = [r["upper_W_sub"] for r in results]

    ent_lower = [r["ent_lower"] for r in results]
    ent_upper = [r["ent_upper"] for r in results]

    leak_0 = [r["E0_leakage"] for r in results]
    leak_lower = [r["lower_leakage"] for r in results]
    leak_upper = [r["upper_leakage"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Energy spectrum (with sorted excited states)
    ax = axes[0, 0]
    ax.plot(lambdas, E0, "b-o", label=r"$E_0$ (ground)", markersize=5)
    ax.plot(lambdas, E_lower, "r-s", label=r"$E_{lower}$", markersize=5)
    ax.plot(lambdas, E_upper, "g-^", label=r"$E_{upper}$", markersize=5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Energy")
    ax.set_title(r"(a) Energy Spectrum (sorted)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gap (now always positive, proper avoided crossing signature)
    ax = axes[0, 1]
    ax.plot(lambdas, gaps, "k-o", markersize=5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta = E_{upper} - E_{lower} \geq 0$")
    ax.set_title(r"(b) Energy Gap $\Delta(\lambda)$")
    ax.grid(True, alpha=0.3)

    # Find minimum gap
    min_gap_idx = np.argmin(gaps)
    ax.axvline(
        lambdas[min_gap_idx],
        color="r",
        linestyle="--",
        alpha=0.5,
        label=rf"$\lambda^* = {lambdas[min_gap_idx]:.2f}$, $\Delta^* = {gaps[min_gap_idx]:.3f}$",
    )
    ax.legend()
    ax.set_ylim(bottom=0)

    # Plot 3: Normalized mixing in 2-level subspace for E_lower
    ax = axes[0, 2]
    ax.plot(lambdas, lower_a_sq, "b-o", label=r"$|a|^2$ (left)", markersize=4)
    ax.plot(lambdas, lower_b_sq, "r-s", label=r"$|b|^2$ (right)", markersize=4)
    ax.plot(lambdas, lower_W_sub, "k--", label=r"$W_{sub}$", markersize=3, alpha=0.7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Weight")
    ax.set_title(r"(c) $E_{lower}$: 2-level mixing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    # Plot 4: Normalized mixing in 2-level subspace for E_upper
    ax = axes[1, 0]
    ax.plot(lambdas, upper_a_sq, "b-o", label=r"$|a|^2$ (left)", markersize=4)
    ax.plot(lambdas, upper_b_sq, "r-s", label=r"$|b|^2$ (right)", markersize=4)
    ax.plot(lambdas, upper_W_sub, "k--", label=r"$W_{sub}$", markersize=3, alpha=0.7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Weight")
    ax.set_title(r"(d) $E_{upper}$: 2-level mixing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    # Plot 5: Entanglement (sorted by energy)
    ax = axes[1, 1]
    ax.plot(lambdas, ent_lower, "r-s", label=r"$E_{lower}$", markersize=4)
    ax.plot(lambdas, ent_upper, "g-^", label=r"$E_{upper}$", markersize=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Entanglement proxy")
    ax.set_title(r"(e) Entanglement vs $\lambda$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(lambdas[min_gap_idx], color="gray", linestyle="--", alpha=0.5)

    # Plot 6: Transverse leakage (sorted by energy)
    ax = axes[1, 2]
    ax.plot(lambdas, leak_0, "b-o", label=r"$E_0$", markersize=4)
    ax.plot(lambdas, leak_lower, "r-s", label=r"$E_{lower}$", markersize=4)
    ax.plot(lambdas, leak_upper, "g-^", label=r"$E_{upper}$", markersize=4)
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


def tiny_test():
    """Smoke test: 3 λ values, minimal epochs. ~5 min on CPU."""
    print("=" * 70)
    print("TINY TEST MODE (smoke test)")
    print("=" * 70)

    lambda_values = [-0.5, 0.0, 0.5]

    results = run_lambda_sweep(
        lambda_values,
        well_separation=4.0,
        n_epochs_ground=50,
        n_epochs_excited=80,
        n_eval_samples=3000,
        n_diag_samples=2000,
    )

    create_comparison_plots(results)
    print("\nTINY TEST COMPLETE")
    return results


def quick_test():
    """Quick test: 7 λ values, moderate epochs. ~30 min on CPU."""
    print("=" * 70)
    print("QUICK TEST MODE")
    print("=" * 70)

    lambda_values = [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]

    results = run_lambda_sweep(
        lambda_values,
        well_separation=4.0,
        n_epochs_ground=150,
        n_epochs_excited=200,
        n_eval_samples=8000,
        n_diag_samples=4000,
    )

    create_comparison_plots(results)
    print("\nQUICK TEST COMPLETE")
    return results


# ============================================================
# Main Entry Point
# ============================================================


def main():
    """Full production run with more λ points near the crossing."""
    # Cluster more points near λ=0 where the avoided crossing occurs
    # Coarse grid + fine grid near center
    coarse = [-0.8, -0.6, -0.4, 0.4, 0.6, 0.8]
    fine = np.linspace(-0.3, 0.3, 9).tolist()  # Dense sampling near crossing
    lambda_values = sorted(set(coarse + fine))

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

    r = results[min_idx]
    print(f"Minimum gap at λ* = {r['lambda']:.3f}")
    print(f"  Gap Δ = {r['gap']:.5f}")
    print(f"  E_lower: a²={r['lower_a_sq']:.3f}, "
          f"b²={r['lower_b_sq']:.3f}, W={r['lower_W_sub']:.3f}")
    print(f"  E_upper: a²={r['upper_a_sq']:.3f}, "
          f"b²={r['upper_b_sq']:.3f}, W={r['upper_W_sub']:.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avoided Crossing Study")
    parser.add_argument("--tiny", action="store_true",
                        help="Smoke test (~5 min)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (~30 min)")
    args = parser.parse_args()

    if args.tiny:
        tiny_test()
    elif args.quick:
        quick_test()
    else:
        main()
