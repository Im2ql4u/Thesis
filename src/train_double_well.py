#!/usr/bin/env python
"""
Train and validate double-well system wavefunctions.

This script:
1. Trains the CTNN+PINN wavefunction for different well separations
2. Validates that E ≈ 3.0 for overlapping wells (d=0) and E ≈ 2.0 for separated wells
3. Runs time evolution at a reasonable separation with the trained model
"""

import json
import math
import os
import sys
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

# MPS doesn't support float64, so use CPU for physics simulations
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float64
OMEGA = 1.0  # Trap frequency (ω=1 gives nice reference energies)
N_PARTICLES = 2
D = 2

# Training hyperparameters - reduced for faster iteration
N_EPOCHS = 200  # Reduced for faster convergence check
N_COLLOCATION = 256
MICRO_BATCH = 64
LEARNING_RATE = 5e-4
GRAD_CLIP = 1.0

# Results directory
RESULTS_DIR = Path("/Users/aleksandersekkelsten/thesis/results/double_well")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Double-Well Physics
# ============================================================


def double_well_potential_energy(x, well_separation, omega=OMEGA):
    """
    Compute double-well harmonic potential energy.

    Particle 0 -> left well at (-d/2, 0)
    Particle 1 -> right well at (+d/2, 0)

    V = 0.5 * ω² * Σ_i |r_i - w_i|²
    """
    B, N, d = x.shape
    ell = 1.0 / math.sqrt(omega)
    sep_phys = well_separation * ell

    # Displacement from assigned well centers
    disp = x.clone()
    disp[:, 0, 0] = x[:, 0, 0] + sep_phys / 2.0  # particle 0 from left well
    disp[:, 1, 0] = x[:, 1, 0] - sep_phys / 2.0  # particle 1 from right well

    V_trap = 0.5 * (omega**2) * (disp**2).sum(dim=(1, 2))
    return V_trap  # (B,)


def coulomb_interaction(x, eps=1e-10):
    """Coulomb repulsion between electrons."""
    B, N, d = x.shape
    if N < 2:
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    # Distance between particles
    r12 = torch.sqrt(((x[:, 0, :] - x[:, 1, :]) ** 2).sum(dim=-1) + eps)
    V_coul = 1.0 / r12
    return V_coul  # (B,)


def sample_double_well_positions(B, N, d, well_separation, omega=OMEGA, device=DEVICE, dtype=DTYPE):
    """Sample initial positions for double-well system."""
    ell = 1.0 / math.sqrt(omega)
    sep_phys = well_separation * ell
    sigma = 0.5 * ell

    x = torch.randn(B, N, d, device=device, dtype=dtype) * sigma

    # Shift particles to their wells
    x[:, 0, 0] -= sep_phys / 2.0  # particle 0 to left well
    x[:, 1, 0] += sep_phys / 2.0  # particle 1 to right well

    return x


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


def local_energy(psi_log_fn, x, well_separation, omega=OMEGA):
    """
    Compute local energy E_L = T + V_trap + V_coul

    T = -0.5 * (Δlogψ + |∇logψ|²)
    """
    lap_logpsi, grad2_logpsi, _ = compute_laplacian_logpsi(psi_log_fn, x)

    # Kinetic energy
    T = -0.5 * (lap_logpsi + grad2_logpsi)

    # Potential energies
    V_trap = double_well_potential_energy(x, well_separation, omega)
    V_coul = coulomb_interaction(x)

    E_L = T + V_trap + V_coul

    return E_L, T, V_trap, V_coul


# ============================================================
# Model Building
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


def make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, well_separation=0.0):
    """
    Create a closure for log(psi) with coordinate transformation for double-well.

    The key insight: shift coordinates so each particle sees its local well as the origin.
    This allows the standard HO-based ansatz to work for separated wells.
    """
    ell = 1.0 / math.sqrt(OMEGA)
    sep_phys = well_separation * ell

    def psi_log_fn(x):
        # Transform coordinates: shift each particle to its well center
        x_shifted = x.clone()
        x_shifted[:, 0, 0] = (
            x[:, 0, 0] + sep_phys / 2.0
        )  # Shift particle 0 from left well to origin
        x_shifted[:, 1, 0] = (
            x[:, 1, 0] - sep_phys / 2.0
        )  # Shift particle 1 from right well to origin

        logpsi, _ = psi_fn(
            f_net, x_shifted, C_occ, backflow_net=backflow_net, spin=spin, params=params
        )
        return logpsi

    return psi_log_fn


# ============================================================
# Training
# ============================================================


def train_double_well_vmc(
    f_net,
    backflow_net,
    C_occ,
    well_separation,
    params,
    n_epochs=N_EPOCHS,
    n_collocation=N_COLLOCATION,
    lr=LEARNING_RATE,
    print_every=100,
):
    """
    Train wavefunction for double-well system using VMC energy minimization.
    """
    # Setup
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)  # One up, one down

    optimizer = optim.Adam(
        list(f_net.parameters()) + list(backflow_net.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 10)

    psi_log_fn = make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, well_separation)

    ell = 1.0 / math.sqrt(OMEGA)
    mcmc_sigma = 0.15 * ell

    history = []
    best_energy = float("inf")
    best_state = None

    # Initialize MCMC chain
    x = sample_double_well_positions(n_collocation, N_PARTICLES, D, well_separation)

    print(f"\nTraining for well separation d = {well_separation:.2f}")
    print("=" * 60)

    for epoch in range(n_epochs):
        f_net.train()
        backflow_net.train()

        # MCMC sampling (Metropolis-Hastings)
        with torch.no_grad():
            logp = 2.0 * psi_log_fn(x)
            for _ in range(20):  # MCMC steps
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(n_collocation, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

        # Compute local energy in mini-batches
        optimizer.zero_grad()

        E_all = []
        for start in range(0, n_collocation, MICRO_BATCH):
            end = min(start + MICRO_BATCH, n_collocation)
            x_batch = x[start:end].detach().requires_grad_(True)

            E_L, T, V_trap, V_coul = local_energy(psi_log_fn, x_batch, well_separation)
            E_all.append(E_L.detach())

            # Variance minimization loss
            E_mean = E_L.mean().detach()
            loss = ((E_L - E_mean) ** 2).mean()
            loss.backward()

        # Gradient step
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(f_net.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(backflow_net.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        # Statistics
        E_cat = torch.cat(E_all)
        E_mean = float(E_cat.mean().item())
        E_std = float(E_cat.std().item())

        history.append(
            {
                "epoch": epoch,
                "E_mean": E_mean,
                "E_std": E_std,
            }
        )

        # Track best
        if E_mean < best_energy:
            best_energy = E_mean
            best_state = {
                "f_net": {k: v.clone() for k, v in f_net.state_dict().items()},
                "backflow_net": {k: v.clone() for k, v in backflow_net.state_dict().items()},
            }

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch:4d} | E = {E_mean:.5f} ± {E_std:.5f} | lr = {scheduler.get_last_lr()[0]:.2e}"
            )

    # Restore best model
    if best_state is not None:
        f_net.load_state_dict(best_state["f_net"])
        backflow_net.load_state_dict(best_state["backflow_net"])

    print(f"Training complete. Best E = {best_energy:.5f}")

    return history, best_energy


def evaluate_energy(f_net, backflow_net, C_occ, well_separation, params, n_samples=50000):
    """Evaluate energy with trained model using more samples."""
    f_net.eval()
    backflow_net.eval()

    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    psi_log_fn = make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, well_separation)

    ell = 1.0 / math.sqrt(OMEGA)
    mcmc_sigma = 0.12 * ell

    batch_size = 1024
    sum_E, sum_E2 = 0.0, 0.0
    sum_T, sum_V_trap, sum_V_coul = 0.0, 0.0, 0.0
    total = 0

    # Initialize chain
    x = sample_double_well_positions(batch_size, N_PARTICLES, D, well_separation)

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
        # MCMC steps
        with torch.no_grad():
            for _ in range(10):
                x_prop = x + torch.randn_like(x) * mcmc_sigma
                logp_prop = 2.0 * psi_log_fn(x_prop)
                accept = torch.rand(batch_size, device=DEVICE).log() < (logp_prop - logp)
                x = torch.where(accept.view(-1, 1, 1), x_prop, x)
                logp = torch.where(accept, logp_prop, logp)

        # Compute local energy
        with torch.set_grad_enabled(True):
            x_eval = x.detach().requires_grad_(True)
            E_L, T, V_trap, V_coul = local_energy(psi_log_fn, x_eval, well_separation)

        E = E_L.detach()
        sum_E += float(E.sum().item())
        sum_E2 += float((E**2).sum().item())
        sum_T += float(T.detach().sum().item())
        sum_V_trap += float(V_trap.detach().sum().item())
        sum_V_coul += float(V_coul.detach().sum().item())
        total += batch_size

    E_mean = sum_E / total
    E_var = max(sum_E2 / total - E_mean**2, 0.0)
    E_std = math.sqrt(E_var)
    E_stderr = E_std / math.sqrt(total)

    return {
        "E_mean": E_mean,
        "E_std": E_std,
        "E_stderr": E_stderr,
        "T_mean": sum_T / total,
        "V_trap_mean": sum_V_trap / total,
        "V_coul_mean": sum_V_coul / total,
        "n_samples": total,
    }


# ============================================================
# Time Evolution
# ============================================================


def time_evolve_trained(
    f_net,
    backflow_net,
    C_occ,
    well_separation,
    params,
    dt=0.005,
    n_steps=400,
    n_walkers=1000,
    n_particles=N_PARTICLES,
    dim=D,
):
    """
    Time evolution using quantum drift-diffusion (Langevin dynamics in |ψ|²).

    This simulates the time-dependent behavior of the system where particles
    are initially localized in their respective wells.
    """
    f_net.eval()
    backflow_net.eval()

    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    psi_log_fn = make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, well_separation)

    ell = 1.0 / math.sqrt(OMEGA)

    # Initialize walkers in the double-well ground state
    x = sample_double_well_positions(n_walkers, n_particles, dim, well_separation)

    # Thermalize first
    mcmc_sigma = 0.1 * ell
    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(300):
            x_prop = x + torch.randn_like(x) * mcmc_sigma
            logp_prop = 2.0 * psi_log_fn(x_prop)
            accept = torch.rand(n_walkers, device=DEVICE).log() < (logp_prop - logp)
            x = torch.where(accept.view(-1, 1, 1), x_prop, x)
            logp = torch.where(accept, logp_prop, logp)

    history = {
        "times": [],
        "x_particle0": [],
        "x_particle1": [],
        "y_particle0": [],
        "y_particle1": [],
        "separation": [],
        "energies": [],
    }

    print(f"\nTime evolution at d = {well_separation:.2f}")
    print("=" * 60)

    for step in range(n_steps):
        t = step * dt

        # Record observables
        with torch.no_grad():
            mean_x0 = float(x[:, 0, 0].mean().item())
            mean_x1 = float(x[:, 1, 0].mean().item())
            mean_y0 = float(x[:, 0, 1].mean().item())
            mean_y1 = float(x[:, 1, 1].mean().item())

            sep = float(torch.sqrt(((x[:, 0, :] - x[:, 1, :]) ** 2).sum(dim=-1)).mean().item())

        # Estimate energy periodically
        if step % 20 == 0:
            with torch.set_grad_enabled(True):
                x_eval = x[:100].detach().requires_grad_(True)
                E_L, _, _, _ = local_energy(psi_log_fn, x_eval, well_separation)
                E_mean = float(E_L.mean().item())

            if step % 100 == 0:
                print(
                    f"t = {t:.3f} | ⟨x₀⟩ = {mean_x0:.3f}, ⟨x₁⟩ = {mean_x1:.3f} | E = {E_mean:.4f}"
                )
        else:
            E_mean = history["energies"][-1] if history["energies"] else 0.0

        history["times"].append(t)
        history["x_particle0"].append(mean_x0)
        history["x_particle1"].append(mean_x1)
        history["y_particle0"].append(mean_y0)
        history["y_particle1"].append(mean_y1)
        history["separation"].append(sep)
        history["energies"].append(E_mean)

        # Quantum drift-diffusion step
        # The drift is ∇log|ψ|² = 2∇logψ
        x_grad = x.detach().requires_grad_(True)
        logpsi = psi_log_fn(x_grad)
        grad_logpsi = torch.autograd.grad(logpsi.sum(), x_grad, create_graph=False)[0]

        # Langevin dynamics: dx = D*∇log|ψ|² dt + √(2D) dW
        D = 0.5  # Diffusion constant (in atomic units, D = ℏ/2m = 0.5)
        drift = 2.0 * D * grad_logpsi * dt
        diffusion = math.sqrt(2.0 * D * dt) * torch.randn_like(x)

        x = x + drift.detach() + diffusion

    return history


# ============================================================
# Main Execution
# ============================================================


def main():
    print("=" * 70)
    print("DOUBLE-WELL QUANTUM SYSTEM - TRAINING AND VALIDATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"ω = {OMEGA}, N = {N_PARTICLES}, d = {D}")
    print(f"Training epochs: {N_EPOCHS}")
    print()

    # Setup config
    config.update(
        device=DEVICE,
        omega=OMEGA,
        n_particles=N_PARTICLES,
        d=D,
        basis="cart",
        nx=2,
        ny=2,
    )
    cfg = config.get()
    params = cfg.as_dict()

    # Build C_occ matrix
    C_occ = make_cartesian_C_occ(2, 2, 1, device=DEVICE, dtype=DTYPE)

    # Test separations
    # d=0: overlapping wells -> expect E ≈ 3.0 (two interacting electrons)
    # d=6-8: separated wells -> expect E ≈ 2.0 (two independent ground states)
    separations = [0.0, 4.0, 8.0]

    all_results = []

    for sep in separations:
        print(f"\n{'='*70}")
        print(f"WELL SEPARATION d = {sep:.1f}")
        print(f"{'='*70}")

        # Build fresh model for each separation
        f_net, backflow_net = build_model()

        # Train
        history, best_E = train_double_well_vmc(
            f_net,
            backflow_net,
            C_occ,
            sep,
            params,
            n_epochs=N_EPOCHS,
            print_every=100,
        )

        # Evaluate with more samples
        print("\nFinal evaluation...")
        result = evaluate_energy(f_net, backflow_net, C_occ, sep, params, n_samples=100000)
        result["well_separation"] = sep
        result["training_history"] = history
        all_results.append(result)

        print(f"\nResult for d = {sep:.1f}:")
        print(f"  E = {result['E_mean']:.5f} ± {result['E_stderr']:.5f}")
        print(f"  T = {result['T_mean']:.5f}")
        print(f"  V_trap = {result['V_trap_mean']:.5f}")
        print(f"  V_coul = {result['V_coul_mean']:.5f}")

        # Save model
        model_path = RESULTS_DIR / f"model_d{sep:.1f}.pt"
        torch.save(
            {
                "f_net": f_net.state_dict(),
                "backflow_net": backflow_net.state_dict(),
                "well_separation": sep,
                "energy": result["E_mean"],
            },
            model_path,
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ENERGY VS WELL SEPARATION")
    print("=" * 70)
    print(f"{'d':<8} {'E':<12} {'Error':<10} {'Expected':<10}")
    print("-" * 45)

    for r in all_results:
        d = r["well_separation"]
        expected = "~3.0" if d < 1.0 else ("~2.0" if d > 5.0 else "")
        print(f"{d:<8.1f} {r['E_mean']:<12.5f} {r['E_stderr']:<10.5f} {expected:<10}")

    # Validation
    E_overlap = all_results[0]["E_mean"]
    E_separated = all_results[-1]["E_mean"]

    print("\nPhysics Validation:")
    print(
        f"  E(d=0) = {E_overlap:.4f}  {'✓ PASS' if 2.8 < E_overlap < 3.2 else '✗ FAIL'} (expected ~3.0)"
    )
    print(
        f"  E(d=8) = {E_separated:.4f}  {'✓ PASS' if 1.8 < E_separated < 2.2 else '✗ FAIL'} (expected ~2.0)"
    )

    # Save all results
    results_json = []
    for r in all_results:
        rj = {k: v for k, v in r.items() if k != "training_history"}
        results_json.append(rj)

    with open(RESULTS_DIR / "energy_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Plot energy vs separation
    fig, ax = plt.subplots(figsize=(10, 6))

    seps = [r["well_separation"] for r in all_results]
    E_means = [r["E_mean"] for r in all_results]
    E_errs = [r["E_stderr"] for r in all_results]

    ax.errorbar(
        seps,
        E_means,
        yerr=E_errs,
        fmt="o-",
        capsize=5,
        markersize=10,
        linewidth=2,
        color="blue",
        label="VMC Energy",
    )
    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1.5, label="E=3.0 (overlapping)")
    ax.axhline(y=2.0, color="green", linestyle="--", linewidth=1.5, label="E=2.0 (separated)")

    ax.set_xlabel("Well Separation d (units of $1/\\sqrt{\\omega}$)", fontsize=12)
    ax.set_ylabel("Ground State Energy (a.u.)", fontsize=12)
    ax.set_title("Double-Well System: Energy vs Well Separation (Trained)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.5, 3.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "energy_vs_separation_trained.png", dpi=150)
    print(f"\nPlot saved to {RESULTS_DIR / 'energy_vs_separation_trained.png'}")

    # Time evolution at reasonable separation (d=4 or d=6)
    print("\n" + "=" * 70)
    print("TIME EVOLUTION")
    print("=" * 70)

    # Use d=4 for time evolution (wells are separated but not too far)
    sep_evolve = 4.0

    # Load or use the trained model for this separation
    model_path = RESULTS_DIR / f"model_d{sep_evolve:.1f}.pt"
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=DEVICE)
        f_net, backflow_net = build_model()
        f_net.load_state_dict(ckpt["f_net"])
        backflow_net.load_state_dict(ckpt["backflow_net"])

    time_history = time_evolve_trained(
        f_net,
        backflow_net,
        C_occ,
        sep_evolve,
        params,
        dt=0.005,
        n_steps=400,
        n_walkers=2000,
    )

    # Plot time evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    times = np.array(time_history["times"])
    x0 = np.array(time_history["x_particle0"])
    x1 = np.array(time_history["x_particle1"])
    y0 = np.array(time_history["y_particle0"])
    y1 = np.array(time_history["y_particle1"])
    sep_hist = np.array(time_history["separation"])
    energies = np.array(time_history["energies"])

    # Energy
    axes[0, 0].plot(times, energies, "b-", linewidth=1.5)
    axes[0, 0].set_xlabel("Time (a.u.)")
    axes[0, 0].set_ylabel("Energy (a.u.)")
    axes[0, 0].set_title("Energy vs Time")
    axes[0, 0].grid(True, alpha=0.3)

    # X positions
    axes[0, 1].plot(times, x0, "r-", linewidth=1.5, label="Particle 0 (left well)")
    axes[0, 1].plot(times, x1, "b-", linewidth=1.5, label="Particle 1 (right well)")
    axes[0, 1].set_xlabel("Time (a.u.)")
    axes[0, 1].set_ylabel("⟨x⟩ (a.u.)")
    axes[0, 1].set_title("Mean X Position vs Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Y positions
    axes[1, 0].plot(times, y0, "r-", linewidth=1.5, label="Particle 0")
    axes[1, 0].plot(times, y1, "b-", linewidth=1.5, label="Particle 1")
    axes[1, 0].set_xlabel("Time (a.u.)")
    axes[1, 0].set_ylabel("⟨y⟩ (a.u.)")
    axes[1, 0].set_title("Mean Y Position vs Time")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Inter-particle separation
    axes[1, 1].plot(times, sep_hist, "purple", linewidth=1.5)
    axes[1, 1].set_xlabel("Time (a.u.)")
    axes[1, 1].set_ylabel("⟨|r₁ - r₂|⟩ (a.u.)")
    axes[1, 1].set_title("Mean Inter-particle Distance vs Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Time Evolution at d = {sep_evolve}", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "time_evolution_trained.png", dpi=150)
    print(f"Time evolution plot saved to {RESULTS_DIR / 'time_evolution_trained.png'}")

    # Save time evolution data
    with open(RESULTS_DIR / "time_evolution.json", "w") as f:
        json.dump(
            {
                "well_separation": sep_evolve,
                "dt": 0.005,
                "n_steps": 400,
                "times": time_history["times"],
                "x_particle0": time_history["x_particle0"],
                "x_particle1": time_history["x_particle1"],
                "energies": time_history["energies"],
                "separation": time_history["separation"],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
