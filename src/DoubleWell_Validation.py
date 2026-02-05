# Double-Well Two-Particle System Validation
# ============================================
#
# This notebook validates the two-particle double-well system:
# - Two electrons in two separate harmonic wells
# - Energy should be ~2.0 when wells are far apart (independent ground states)
# - Energy should be ~3.0 when wells completely overlap (interacting electrons)
#
# Uses the CTNN+PINN ansatz which has provided best results.

# %%
# Setup and imports
import os

os.environ["CUDA_MANUAL_DEVICE"] = "0"  # Adjust GPU index as needed

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Local imports
import config
from functions import psi_fn
from functions.DoubleWell import (
    plot_energy_vs_separation,
    scan_energy_vs_separation,
    time_evolve_double_well,
)
from PINN import PINN, CTNNBackflowNet
from utils import get_promoted_params

torch.set_num_threads(4)

# %%
# Configure the system
config.update(
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    omega=1.0,  # Trap frequency
    act_fn_name="gelu",
    n_particles=2,  # Two electrons
    d=2,  # 2D system
    n_epochs=500,
    N_collocation=256,
    basis="cart",
    nx=2,
    ny=2,  # Basis size
)

cfg = config.get()
params = cfg.as_dict()

# Promote commonly used params to global scope
globals().update(
    get_promoted_params(
        names=[
            "omega",
            "nx",
            "ny",
            "n_particles",
            "hidden_dim",
            "n_layers",
            "L",
            "n_grid",
            "d",
            "device",
            "V",
            "E",
        ],
        include_runtime=True,
    )
)

print(f"Device: {device}")
print(f"Omega: {omega}")
print(f"N particles: {n_particles}")


# %%
# Create Slater determinant occupation matrix
def make_cartesian_C_occ(nx, ny, n_occ, *, device, dtype=torch.float64):
    """Create occupation matrix for Cartesian HO basis."""
    pairs = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    pairs.sort(key=lambda t: (t[0] + t[1], t[0]))
    sel = pairs[:n_occ]
    cols = [ix * ny + iy for (ix, iy) in sel]
    C = torch.zeros(nx * ny, n_occ, dtype=dtype, device=device)
    for j, c in enumerate(cols):
        C[c, j] = 1.0
    return C


C_occ = make_cartesian_C_occ(
    nx, ny, int(n_particles / 2), device=cfg.torch_device, dtype=torch.float64
)
print(f"C_occ shape: {C_occ.shape}")

# %%
# Build neural network ansatz (CTNN + PINN)
# This is the copresheaf + PINN combination that gave best results

f_net = PINN(
    n_particles=cfg.n_particles,
    d=cfg.d,
    omega=cfg.omega,
    dL=5,
    hidden_dim=128,
    n_layers=2,
    act="gelu",
    init="xavier",
    use_gate=True,
).to(cfg.torch_device, cfg.torch_dtype)

backflow_net = CTNNBackflowNet(
    d=2,
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
    zero_init_last=False,
    omega=cfg.omega,
).to(cfg.torch_device, cfg.torch_dtype)

print(f"f_net parameters: {sum(p.numel() for p in f_net.parameters())}")
print(f"backflow_net parameters: {sum(p.numel() for p in backflow_net.parameters())}")


# %%
# Test: Visualize double-well potential
def visualize_double_well_potential(well_separation, params):
    """Visualize the double-well potential landscape."""
    import matplotlib.pyplot as plt

    omega = params["omega"]
    ell = 1.0 / np.sqrt(omega)
    sep = well_separation * ell

    # Create grid
    x_range = np.linspace(-3 * ell - sep / 2, 3 * ell + sep / 2, 100)
    y_range = np.linspace(-3 * ell, 3 * ell, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Potential from left well (for particle 0)
    V_left = 0.5 * omega**2 * ((X + sep / 2) ** 2 + Y**2)
    # Potential from right well (for particle 1)
    V_right = 0.5 * omega**2 * ((X - sep / 2) ** 2 + Y**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, V, title in zip(
        axes[:2],
        [V_left, V_right],
        ["Left Well (Particle 0)", "Right Well (Particle 1)"],
        strict=False,
    ):
        im = ax.contourf(X, Y, V, levels=30, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axvline(x=-sep / 2, color="r", linestyle="--", alpha=0.5)
        ax.axvline(x=sep / 2, color="r", linestyle="--", alpha=0.5)
        plt.colorbar(im, ax=ax, label="V")

    # Combined view
    V_combined = np.minimum(V_left, V_right)
    im = axes[2].contourf(X, Y, V_combined, levels=30, cmap="viridis")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title(f"Combined Wells (sep={well_separation:.1f})")
    axes[2].set_aspect("equal")
    axes[2].axvline(x=-sep / 2, color="r", linestyle="--", alpha=0.5)
    axes[2].axvline(x=sep / 2, color="r", linestyle="--", alpha=0.5)
    plt.colorbar(im, ax=axes[2], label="V")

    plt.tight_layout()
    return fig


# Test with different separations
for sep in [0.0, 3.0, 6.0]:
    fig = visualize_double_well_potential(sep, params)
    plt.suptitle(f"Double-Well Potential (d = {sep})", y=1.02)
    plt.savefig(
        f"/Users/aleksandersekkelsten/thesis/results/figures/results/double_well_potential_d{sep:.0f}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

# %%
# Train the model for each separation and collect energies
# We'll scan from overlapping (d=0) to well-separated (d=8)

separations = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

print("=" * 60)
print("DOUBLE-WELL ENERGY SCAN")
print("=" * 60)
print(f"Testing {len(separations)} separations from d={separations[0]} to d={separations[-1]}")
print("Expected: E ≈ 3.0 at d=0, E ≈ 2.0 at large d")
print("=" * 60)

# For each separation, we evaluate the energy using the trained wavefunction
# The ansatz should work well since we're keeping the CTNN+PINN structure

results = scan_energy_vs_separation(
    f_net,
    C_occ,
    separations,
    psi_fn=psi_fn,
    backflow_net=backflow_net,
    spin=None,
    params=params,
    n_samples=20000,
    batch_size=512,
    mcmc_steps=50,
)

# %%
# Save results
results_dir = Path("/Users/aleksandersekkelsten/thesis/results/double_well")
results_dir.mkdir(parents=True, exist_ok=True)

# Convert to serializable format
results_json = []
for r in results:
    results_json.append(
        {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()}
    )

with open(results_dir / "energy_vs_separation.json", "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\nResults saved to {results_dir / 'energy_vs_separation.json'}")

# %%
# Create the energy vs separation plot
fig, ax = plot_energy_vs_separation(
    results, save_path=str(results_dir / "energy_vs_separation.png")
)
plt.show()

# %%
# Print summary table
print("\n" + "=" * 70)
print("ENERGY VS WELL SEPARATION - SUMMARY")
print("=" * 70)
print(f"{'Separation (d)':<15} {'Energy':<12} {'Error':<10} {'T':<10} {'V_trap':<10} {'V_int':<10}")
print("-" * 70)

for r in results:
    print(
        f"{r['well_separation']:<15.2f} {r['E_mean']:<12.4f} {r['E_stderr']:<10.4f} "
        f"{r['T_mean']:<10.4f} {r['V_trap_mean']:<10.4f} {r['V_int_mean']:<10.4f}"
    )

print("-" * 70)

# Verify expected behavior
E_overlap = results[0]["E_mean"]  # d=0
E_separated = results[-1]["E_mean"]  # d=10

print("\nValidation:")
print(f"  E(d=0) = {E_overlap:.4f}  (expected ~3.0 for overlapping wells)")
print(f"  E(d={separations[-1]}) = {E_separated:.4f}  (expected ~2.0 for separated wells)")

if 2.5 < E_overlap < 3.5:
    print("  ✓ Overlapping wells energy is in expected range")
else:
    print("  ✗ Overlapping wells energy outside expected range")

if 1.5 < E_separated < 2.5:
    print("  ✓ Separated wells energy is in expected range")
else:
    print("  ✗ Separated wells energy outside expected range")

# %%
# Time evolution at reasonable separation
print("\n" + "=" * 60)
print("TIME EVOLUTION - DOUBLE WELL")
print("=" * 60)

# Choose a reasonable separation where electrons are in separate wells
well_sep_evolution = 4.0
print(f"Well separation: d = {well_sep_evolution}")

# Run time evolution
time_results = time_evolve_double_well(
    f_net,
    C_occ,
    well_sep_evolution,
    psi_fn=psi_fn,
    backflow_net=backflow_net,
    spin=None,
    params=params,
    dt=0.01,
    n_steps=200,
    n_samples=500,
)

# %%
# Plot time evolution results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

times = np.array(time_results["times"])
positions_x = np.array(time_results["positions_x"])
positions_y = np.array(time_results["positions_y"])
energies = np.array(time_results["energies"])

# Energy vs time
axes[0, 0].plot(times, energies, "b-", linewidth=1.5)
axes[0, 0].set_xlabel("Time (a.u.)")
axes[0, 0].set_ylabel("Energy (a.u.)")
axes[0, 0].set_title("Energy During Time Evolution")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=2.0, color="g", linestyle="--", alpha=0.5, label="E=2.0")
axes[0, 0].legend()

# X positions vs time
for i in range(positions_x.shape[1]):
    axes[0, 1].plot(times, positions_x[:, i], label=f"Particle {i}", linewidth=1.5)
axes[0, 1].set_xlabel("Time (a.u.)")
axes[0, 1].set_ylabel("⟨x⟩ (a.u.)")
axes[0, 1].set_title("Mean X Position vs Time")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Y positions vs time
for i in range(positions_y.shape[1]):
    axes[1, 0].plot(times, positions_y[:, i], label=f"Particle {i}", linewidth=1.5)
axes[1, 0].set_xlabel("Time (a.u.)")
axes[1, 0].set_ylabel("⟨y⟩ (a.u.)")
axes[1, 0].set_title("Mean Y Position vs Time")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Particle separation vs time
if positions_x.shape[1] >= 2:
    separation = np.sqrt(
        (positions_x[:, 1] - positions_x[:, 0]) ** 2 + (positions_y[:, 1] - positions_y[:, 0]) ** 2
    )
    axes[1, 1].plot(times, separation, "purple", linewidth=1.5)
    axes[1, 1].set_xlabel("Time (a.u.)")
    axes[1, 1].set_ylabel("|r₁ - r₂| (a.u.)")
    axes[1, 1].set_title("Inter-particle Distance vs Time")
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "time_evolution.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Save time evolution results
time_results_serializable = {
    "well_separation": well_sep_evolution,
    "dt": 0.01,
    "n_steps": 200,
    "times": [float(t) for t in times],
    "positions_x": [[float(p) for p in pos] for pos in positions_x],
    "positions_y": [[float(p) for p in pos] for pos in positions_y],
    "energies": [float(e) for e in energies],
}

with open(results_dir / "time_evolution.json", "w") as f:
    json.dump(time_results_serializable, f, indent=2)

print(f"Time evolution results saved to {results_dir}")

# %%
# Summary
print("\n" + "=" * 60)
print("SUMMARY - DOUBLE WELL VALIDATION")
print("=" * 60)
print(
    f"""
Physics Validation:
- Two particles in two separate 2D harmonic wells
- Using CTNN + PINN ansatz (copresheaf + physics-informed NN)
- Trap frequency ω = {omega}

Energy Results:
- E(d=0, overlapping): {results[0]['E_mean']:.4f} ± {results[0]['E_stderr']:.4f}
  (Expected: ~3.0 for two interacting electrons in single well)
  
- E(d={separations[-1]}, separated): {results[-1]['E_mean']:.4f} ± {results[-1]['E_stderr']:.4f}
  (Expected: ~2.0 for two independent ground states)

Files Generated:
- {results_dir / "energy_vs_separation.json"}
- {results_dir / "energy_vs_separation.png"}
- {results_dir / "time_evolution.json"}
- {results_dir / "time_evolution.png"}

The CTNN+PINN ansatz {'appears suitable' if (2.5 < results[0]['E_mean'] < 3.5 and 1.5 < results[-1]['E_mean'] < 2.5) else 'may need adjustment'} for the double-well system.
"""
)
