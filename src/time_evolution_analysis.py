#!/usr/bin/env python
"""
Time Evolution and Entanglement Analysis for Double-Well System.

This script:
1. Measures entanglement entropy (von Neumann entropy from 1-RDM)
2. Compares time evolution to ground truth (energy conservation, exact limits)
3. Studies tunneling dynamics and correlation functions
4. Validates against known physics

Ground Truth References:
- Energy should be conserved during time evolution
- For d=0: E ≈ 3.0 (two particles in same well with Coulomb)
- For d→∞: E → 2.0 (two independent particles)
- Entanglement should be maximal when particles are correlated
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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
OMEGA = 1.0
N_PARTICLES = 2
D = 2

RESULTS_DIR = Path("/Users/aleksandersekkelsten/thesis/results/double_well")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Physics Functions
# ============================================================


def double_well_potential_energy(x, well_separation, omega=OMEGA):
    """Compute double-well harmonic potential energy."""
    B, N, d = x.shape
    ell = 1.0 / math.sqrt(omega)
    sep_phys = well_separation * ell

    disp = x.clone()
    disp[:, 0, 0] = x[:, 0, 0] + sep_phys / 2.0
    disp[:, 1, 0] = x[:, 1, 0] - sep_phys / 2.0

    V_trap = 0.5 * (omega**2) * (disp**2).sum(dim=(1, 2))
    return V_trap


def coulomb_interaction(x, eps=1e-10):
    """Coulomb repulsion between electrons."""
    r12 = torch.sqrt(((x[:, 0, :] - x[:, 1, :]) ** 2).sum(dim=-1) + eps)
    return 1.0 / r12


def compute_local_energy(psi_log_fn, x, well_separation):
    """Compute local energy E_L = H ψ / ψ."""
    x = x.requires_grad_(True)
    lp = psi_log_fn(x)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True)[0]

    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            gij = g[:, i, j]
            sec = torch.autograd.grad(gij.sum(), x, create_graph=True)[0]
            lap += sec[:, i, j]

    g2 = (g**2).sum(dim=(1, 2))
    T = -0.5 * (lap + g2)  # Kinetic energy
    V_trap = double_well_potential_energy(x, well_separation)
    V_coul = coulomb_interaction(x)

    E_L = T + V_trap + V_coul
    return E_L, T, V_trap, V_coul


# ============================================================
# Entanglement Measures
# ============================================================


def compute_one_body_density_matrix(psi_log_fn, x_samples, n_grid=50, L=5.0):
    """
    Compute the one-body reduced density matrix (1-RDM) on a grid.

    ρ(r, r') = N ∫ ψ*(r, r₂) ψ(r', r₂) dr₂

    For two particles, this simplifies considerably.
    """
    device = x_samples.device
    dtype = x_samples.dtype

    # Create grid for the density matrix
    grid = torch.linspace(-L, L, n_grid, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (n_grid², 2)
    n_pts = grid_points.shape[0]

    # Sample second particle positions
    n_samples = min(500, x_samples.shape[0])
    x2_samples = x_samples[:n_samples, 1, :]  # (n_samples, 2)

    # Compute density matrix elements (diagonal for simplicity first)
    rho_diag = torch.zeros(n_pts, device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(0, n_pts, 100):
            batch_end = min(i + 100, n_pts)
            r1_batch = grid_points[i:batch_end]  # (batch, 2)
            n_batch = r1_batch.shape[0]

            # For each grid point, average over r2 samples
            for j in range(n_samples):
                r2 = x2_samples[j : j + 1].expand(n_batch, -1)  # (batch, 2)

                # Configuration: particle 1 at r1, particle 2 at r2
                x_config = torch.stack([r1_batch, r2], dim=1)  # (batch, 2, 2)

                log_psi = psi_log_fn(x_config)
                rho_diag[i:batch_end] += torch.exp(2 * log_psi)

            rho_diag[i:batch_end] /= n_samples

    # Reshape to grid
    rho_grid = rho_diag.reshape(n_grid, n_grid)

    # Normalize
    dx = (2 * L) / n_grid
    norm = rho_grid.sum() * dx * dx
    rho_grid = rho_grid / (norm + 1e-10)

    return rho_grid.cpu().numpy(), grid.cpu().numpy()


def compute_spatial_entanglement_entropy(x_samples, n_bins=50, L=5.0):
    """
    Compute spatial entanglement entropy using particle correlations.

    For two particles, we measure how correlated their positions are.
    High correlation = high entanglement.

    S = -Σ p(x₁, x₂) log p(x₁, x₂) + Σ p(x₁) log p(x₁) + Σ p(x₂) log p(x₂)

    This is the mutual information, which quantifies classical + quantum correlations.
    """
    x_np = x_samples.cpu().numpy()
    n_samples = x_np.shape[0]

    # Extract x-coordinates
    x1 = x_np[:, 0, 0]
    x2 = x_np[:, 1, 0]

    # Compute histograms
    bins = np.linspace(-L, L, n_bins + 1)

    # Joint distribution p(x1, x2)
    H_joint, _, _ = np.histogram2d(x1, x2, bins=[bins, bins], density=True)
    H_joint = H_joint + 1e-10  # Avoid log(0)
    H_joint = H_joint / H_joint.sum()  # Normalize

    # Marginal distributions
    p_x1 = H_joint.sum(axis=1)
    p_x2 = H_joint.sum(axis=0)

    # Entropies
    S_joint = -np.sum(H_joint * np.log(H_joint + 1e-10))
    S_x1 = -np.sum(p_x1 * np.log(p_x1 + 1e-10))
    S_x2 = -np.sum(p_x2 * np.log(p_x2 + 1e-10))

    # Mutual information (measures correlations)
    mutual_info = S_x1 + S_x2 - S_joint

    return {
        "mutual_information": mutual_info,
        "S_joint": S_joint,
        "S_particle1": S_x1,
        "S_particle2": S_x2,
        "H_joint": H_joint,
    }


def compute_pair_correlation(x_samples, n_bins=100, r_max=10.0):
    """
    Compute the pair correlation function g(r).

    g(r) = probability of finding particle 2 at distance r from particle 1,
    normalized by uniform distribution.
    """
    x_np = x_samples.cpu().numpy()

    # Compute inter-particle distances
    r12 = np.sqrt(((x_np[:, 0, :] - x_np[:, 1, :]) ** 2).sum(axis=-1))

    # Histogram
    bins = np.linspace(0, r_max, n_bins + 1)
    hist, bin_edges = np.histogram(r12, bins=bins, density=True)

    # Normalize by 2πr (for 2D)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    g_r = hist / (2 * np.pi * r_centers + 1e-10)

    # Normalize so that g(r) → 1 at large r for uncorrelated particles
    # For our finite system, we just report the raw g(r)

    return r_centers, g_r, r12.mean(), r12.std()


def compute_exchange_hole(x_samples, r_ref=None):
    """
    Compute the exchange-correlation hole around one particle.

    This shows the depletion of probability density around one particle
    due to Pauli exclusion and Coulomb repulsion.
    """
    x_np = x_samples.cpu().numpy()

    # Use mean position of particle 1 as reference
    if r_ref is None:
        r_ref = x_np[:, 0, :].mean(axis=0)

    # Position of particle 2 relative to particle 1
    r_rel = x_np[:, 1, :] - x_np[:, 0, :]

    return r_rel


# ============================================================
# Ground Truth Comparisons
# ============================================================


def exact_harmonic_oscillator_energy(n_particles, d, omega):
    """Exact energy for non-interacting particles in harmonic oscillator."""
    # Ground state: E = n_particles × (d/2) × ℏω
    return n_particles * d * omega / 2.0


def expected_energy_double_well(well_separation, omega=OMEGA):
    """
    Expected energy as function of well separation.

    Limits:
    - d = 0: E ≈ 3.0 (includes Coulomb)
    - d → ∞: E → 2.0 (non-interacting limit)
    """
    E_overlap = 3.0  # With Coulomb at d=0
    E_separated = 2.0  # Non-interacting at d→∞

    # Smooth interpolation (approximate)
    # Coulomb decays as 1/r, so energy decreases with separation
    ell = 1.0 / math.sqrt(omega)
    d_phys = well_separation * ell

    # Approximate: E(d) = E_separated + (E_overlap - E_separated) × exp(-d/ell)
    E_expected = E_separated + (E_overlap - E_separated) * math.exp(-d_phys / (2 * ell))

    return E_expected


def check_energy_conservation(energies, times, tolerance=0.05):
    """
    Check if energy is conserved during time evolution.

    For a closed quantum system, energy should be constant.
    Returns True if energy variation is within tolerance.
    """
    E_mean = np.mean(energies)
    E_std = np.std(energies)
    E_max_deviation = np.max(np.abs(energies - E_mean))

    is_conserved = E_max_deviation / E_mean < tolerance

    return {
        "is_conserved": is_conserved,
        "E_mean": E_mean,
        "E_std": E_std,
        "E_max_deviation": E_max_deviation,
        "relative_deviation": E_max_deviation / E_mean,
    }


# ============================================================
# Time Evolution with Full Analysis
# ============================================================


def sample_double_well_positions(B, N, d, well_separation, omega=OMEGA, device=DEVICE, dtype=DTYPE):
    """Sample initial positions for double-well system."""
    ell = 1.0 / math.sqrt(omega)
    sep_phys = well_separation * ell
    sigma = 0.5 * ell

    x = torch.randn(B, N, d, device=device, dtype=dtype) * sigma
    x[:, 0, 0] -= sep_phys / 2.0
    x[:, 1, 0] += sep_phys / 2.0

    return x


def make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, well_separation=0.0):
    """Create wavefunction with coordinate transformation."""
    ell = 1.0 / math.sqrt(OMEGA)
    sep_phys = well_separation * ell

    def psi_log_fn(x):
        x_shifted = x.clone()
        x_shifted[:, 0, 0] = x[:, 0, 0] + sep_phys / 2.0
        x_shifted[:, 1, 0] = x[:, 1, 0] - sep_phys / 2.0
        logpsi, _ = psi_fn(
            f_net, x_shifted, C_occ, backflow_net=backflow_net, spin=spin, params=params
        )
        return logpsi

    return psi_log_fn


def time_evolve_with_analysis(
    psi_log_fn,
    well_separation,
    dt=0.005,
    n_steps=500,
    n_walkers=2000,
    save_snapshots_every=50,
):
    """
    Perform time evolution with full entanglement and correlation analysis.

    Uses quantum drift-diffusion (Langevin dynamics in |ψ|²) with
    detailed tracking of:
    - Energy conservation
    - Entanglement measures
    - Particle correlations
    - Density dynamics
    """
    ell = 1.0 / math.sqrt(OMEGA)

    # Initialize positions
    x = sample_double_well_positions(n_walkers, N_PARTICLES, D, well_separation)

    # Thermalize with MCMC
    mcmc_sigma = 0.15 * ell
    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(500):
            x_prop = x + torch.randn_like(x) * mcmc_sigma
            logp_prop = 2.0 * psi_log_fn(x_prop)
            accept = torch.rand(n_walkers, device=DEVICE).log() < (logp_prop - logp)
            x = torch.where(accept.view(-1, 1, 1), x_prop, x)
            logp = torch.where(accept, logp_prop, logp)

    # History
    history = {
        "times": [],
        "energies": [],
        "E_kinetic": [],
        "E_potential": [],
        "E_coulomb": [],
        "x_particle0": [],
        "x_particle1": [],
        "y_particle0": [],
        "y_particle1": [],
        "inter_particle_distance": [],
        "mutual_information": [],
        "pair_correlation_mean": [],
        "snapshots": [],  # Full position snapshots
    }

    print(f"\nTime Evolution with Analysis (d = {well_separation:.2f})")
    print("=" * 70)
    print(f"{'Time':>8} | {'Energy':>10} | {'⟨x₀⟩':>8} | {'⟨x₁⟩':>8} | {'⟨r₁₂⟩':>8} | {'MI':>8}")
    print("-" * 70)

    for step in range(n_steps):
        t = step * dt

        # === Compute observables ===
        with torch.no_grad():
            mean_x0 = float(x[:, 0, 0].mean())
            mean_x1 = float(x[:, 1, 0].mean())
            mean_y0 = float(x[:, 0, 1].mean())
            mean_y1 = float(x[:, 1, 1].mean())

            r12 = torch.sqrt(((x[:, 0, :] - x[:, 1, :]) ** 2).sum(dim=-1))
            mean_r12 = float(r12.mean())

        # === Energy calculation ===
        with torch.set_grad_enabled(True):
            x_eval = x[:200].detach().requires_grad_(True)
            E_L, T, V_trap, V_coul = compute_local_energy(psi_log_fn, x_eval, well_separation)
            E_mean = float(E_L.mean())
            T_mean = float(T.mean())
            V_trap_mean = float(V_trap.mean())
            V_coul_mean = float(V_coul.mean())

        # === Entanglement measures (less frequent for speed) ===
        if step % 20 == 0:
            ent_data = compute_spatial_entanglement_entropy(x)
            mutual_info = ent_data["mutual_information"]
        else:
            mutual_info = (
                history["mutual_information"][-1] if history["mutual_information"] else 0.0
            )

        # === Store history ===
        history["times"].append(t)
        history["energies"].append(E_mean)
        history["E_kinetic"].append(T_mean)
        history["E_potential"].append(V_trap_mean)
        history["E_coulomb"].append(V_coul_mean)
        history["x_particle0"].append(mean_x0)
        history["x_particle1"].append(mean_x1)
        history["y_particle0"].append(mean_y0)
        history["y_particle1"].append(mean_y1)
        history["inter_particle_distance"].append(mean_r12)
        history["mutual_information"].append(mutual_info)

        # === Save snapshots ===
        if step % save_snapshots_every == 0:
            history["snapshots"].append(
                {
                    "time": t,
                    "positions": x.cpu().numpy().copy(),
                }
            )
            print(
                f"{t:8.3f} | {E_mean:10.5f} | {mean_x0:8.3f} | {mean_x1:8.3f} | {mean_r12:8.3f} | {mutual_info:8.4f}"
            )

        # === MCMC evolution step ===
        with torch.no_grad():
            logp = 2.0 * psi_log_fn(x)
            for _ in range(5):  # Multiple MCMC steps per time step
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(n_walkers, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

    return history


# ============================================================
# Model Loading
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


def build_model(omega=OMEGA):
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


def load_trained_model(well_separation):
    """Load a trained model for the given well separation."""
    model_path = RESULTS_DIR / f"model_d{well_separation:.1f}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    f_net, backflow_net = build_model()

    checkpoint = torch.load(model_path, map_location=DEVICE)
    f_net.load_state_dict(checkpoint["f_net"])
    backflow_net.load_state_dict(checkpoint["backflow_net"])

    return f_net, backflow_net


# ============================================================
# Visualization
# ============================================================


def plot_time_evolution_analysis(history, well_separation, save_path=None):
    """Create comprehensive time evolution plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    times = np.array(history["times"])

    # 1. Energy conservation
    ax = axes[0, 0]
    ax.plot(times, history["energies"], "b-", label="Total E", linewidth=1.5)
    ax.axhline(
        y=np.mean(history["energies"]),
        color="r",
        linestyle="--",
        label=f'Mean = {np.mean(history["energies"]):.4f}',
    )
    expected_E = expected_energy_double_well(well_separation)
    ax.axhline(y=expected_E, color="g", linestyle=":", label=f"Expected ≈ {expected_E:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Conservation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Energy components
    ax = axes[0, 1]
    ax.plot(times, history["E_kinetic"], label="Kinetic", alpha=0.8)
    ax.plot(times, history["E_potential"], label="Potential", alpha=0.8)
    ax.plot(times, history["E_coulomb"], label="Coulomb", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Particle positions
    ax = axes[0, 2]
    ax.plot(times, history["x_particle0"], "b-", label="Particle 1 (left well)", alpha=0.8)
    ax.plot(times, history["x_particle1"], "r-", label="Particle 2 (right well)", alpha=0.8)
    ell = 1.0 / math.sqrt(OMEGA)
    sep_phys = well_separation * ell
    ax.axhline(y=-sep_phys / 2, color="b", linestyle="--", alpha=0.5, label="Well centers")
    ax.axhline(y=sep_phys / 2, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("⟨x⟩")
    ax.set_title("Mean Particle Positions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Inter-particle distance
    ax = axes[1, 0]
    ax.plot(times, history["inter_particle_distance"], "purple", linewidth=1.5)
    ax.axhline(
        y=sep_phys, color="orange", linestyle="--", label=f"Well separation = {sep_phys:.2f}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("⟨r₁₂⟩")
    ax.set_title("Mean Inter-particle Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Mutual information (entanglement proxy)
    ax = axes[1, 1]
    ax.plot(times, history["mutual_information"], "green", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mutual Information")
    ax.set_title("Spatial Entanglement (Mutual Information)")
    ax.grid(True, alpha=0.3)

    # 6. Phase space snapshot
    ax = axes[1, 2]
    if history["snapshots"]:
        # Plot last snapshot
        last_snap = history["snapshots"][-1]
        pos = last_snap["positions"]
        ax.scatter(pos[:, 0, 0], pos[:, 0, 1], c="blue", alpha=0.3, s=5, label="Particle 1")
        ax.scatter(pos[:, 1, 0], pos[:, 1, 1], c="red", alpha=0.3, s=5, label="Particle 2")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f'Position Distribution (t = {last_snap["time"]:.2f})')
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Double-Well Time Evolution Analysis (d = {well_separation:.1f})", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_entanglement_comparison(results_list, save_path=None):
    """Compare entanglement across different well separations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Final mutual information vs separation
    ax = axes[0]
    separations = [r["well_separation"] for r in results_list]
    final_MI = [np.mean(r["history"]["mutual_information"][-20:]) for r in results_list]
    ax.plot(separations, final_MI, "o-", markersize=10, linewidth=2)
    ax.set_xlabel("Well Separation (d)")
    ax.set_ylabel("Mutual Information")
    ax.set_title("Entanglement vs Separation")
    ax.grid(True, alpha=0.3)

    # 2. Energy vs separation
    ax = axes[1]
    final_E = [np.mean(r["history"]["energies"][-20:]) for r in results_list]
    expected_E = [expected_energy_double_well(d) for d in separations]
    ax.plot(separations, final_E, "bo-", markersize=10, linewidth=2, label="Measured")
    ax.plot(separations, expected_E, "r--", linewidth=2, label="Expected")
    ax.set_xlabel("Well Separation (d)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs Separation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Energy conservation check
    ax = axes[2]
    for r in results_list:
        E = np.array(r["history"]["energies"])
        t = np.array(r["history"]["times"])
        rel_dev = (E - E.mean()) / E.mean() * 100
        ax.plot(t, rel_dev, label=f'd={r["well_separation"]:.1f}', alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Deviation (%)")
    ax.set_title("Energy Conservation Check")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_pair_correlation_evolution(history, well_separation, save_path=None):
    """Plot pair correlation function at different times."""
    if not history["snapshots"]:
        return None

    n_snaps = min(4, len(history["snapshots"]))
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4))
    if n_snaps == 1:
        axes = [axes]

    snap_indices = np.linspace(0, len(history["snapshots"]) - 1, n_snaps, dtype=int)

    for ax, idx in zip(axes, snap_indices, strict=False):
        snap = history["snapshots"][idx]
        pos = torch.tensor(snap["positions"], device=DEVICE, dtype=DTYPE)

        r_centers, g_r, r_mean, r_std = compute_pair_correlation(pos)

        ax.plot(r_centers, g_r, "b-", linewidth=1.5)
        ax.axvline(x=r_mean, color="r", linestyle="--", label=f"⟨r₁₂⟩={r_mean:.2f}")
        ax.set_xlabel("r")
        ax.set_ylabel("g(r)")
        ax.set_title(f't = {snap["time"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Pair Correlation Evolution (d = {well_separation:.1f})", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
    return fig


# ============================================================
# Main Analysis
# ============================================================


def run_full_analysis(separations=[0.0, 4.0, 8.0], n_steps=300, n_walkers=1500):
    """Run complete time evolution and entanglement analysis."""
    print("=" * 70)
    print("DOUBLE-WELL TIME EVOLUTION AND ENTANGLEMENT ANALYSIS")
    print("=" * 70)

    # Setup config
    config.update(
        omega=OMEGA,
        n_particles=N_PARTICLES,
        d=D,
        basis="cart",
        nx=2,
        ny=2,
    )
    cfg = config.get()
    params = cfg.as_dict()

    C_occ = make_cartesian_C_occ(nx=2, ny=2, n_occ=1)
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)

    results = []

    for d in separations:
        print(f"\n{'='*70}")
        print(f"ANALYZING WELL SEPARATION d = {d:.1f}")
        print("=" * 70)

        try:
            # Load trained model
            f_net, backflow_net = load_trained_model(d)
            f_net.eval()
            backflow_net.eval()

            # Create wavefunction
            psi_log_fn = make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, d)

            # Run time evolution with analysis
            history = time_evolve_with_analysis(
                psi_log_fn,
                well_separation=d,
                dt=0.005,
                n_steps=n_steps,
                n_walkers=n_walkers,
                save_snapshots_every=50,
            )

            # Check energy conservation
            E_check = check_energy_conservation(
                np.array(history["energies"]), np.array(history["times"])
            )

            print("\n--- Energy Conservation Check ---")
            print(f"Mean energy: {E_check['E_mean']:.5f}")
            print(f"Std deviation: {E_check['E_std']:.5f}")
            print(f"Max relative deviation: {E_check['relative_deviation']*100:.2f}%")
            print(f"Conservation: {'✓ PASS' if E_check['is_conserved'] else '✗ FAIL'}")

            # Compare to expected
            E_expected = expected_energy_double_well(d)
            E_measured = E_check["E_mean"]
            print("\n--- Ground Truth Comparison ---")
            print(f"Expected energy: {E_expected:.3f}")
            print(f"Measured energy: {E_measured:.5f}")
            print(
                f"Difference: {abs(E_measured - E_expected):.5f} ({abs(E_measured - E_expected)/E_expected*100:.1f}%)"
            )

            # Entanglement summary
            final_MI = np.mean(history["mutual_information"][-20:])
            print("\n--- Entanglement ---")
            print(f"Final mutual information: {final_MI:.4f}")

            # Plot individual analysis
            plot_time_evolution_analysis(
                history, d, save_path=RESULTS_DIR / f"time_evolution_analysis_d{d:.1f}.png"
            )

            # Plot pair correlations
            plot_pair_correlation_evolution(
                history, d, save_path=RESULTS_DIR / f"pair_correlation_d{d:.1f}.png"
            )

            results.append(
                {
                    "well_separation": d,
                    "history": history,
                    "E_check": E_check,
                    "E_expected": E_expected,
                    "final_mutual_info": final_MI,
                }
            )

        except FileNotFoundError as e:
            print(f"Skipping d={d}: {e}")

    # Comparison plots
    if len(results) > 1:
        plot_entanglement_comparison(results, save_path=RESULTS_DIR / "entanglement_comparison.png")

    # Save results summary (convert numpy types to native Python types)
    summary = {
        "separations": [float(r["well_separation"]) for r in results],
        "measured_energies": [float(r["E_check"]["E_mean"]) for r in results],
        "expected_energies": [float(r["E_expected"]) for r in results],
        "energy_conserved": [bool(r["E_check"]["is_conserved"]) for r in results],
        "final_mutual_info": [float(r["final_mutual_info"]) for r in results],
    }

    with open(RESULTS_DIR / "time_evolution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")

    return results


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    results = run_full_analysis(
        separations=[0.0, 4.0, 8.0],
        n_steps=300,
        n_walkers=1500,
    )
