"""
Simple 2e run: non-interacting SD + PINN Jastrow + original train_model.

Uses the EXISTING train_model from Neural_Networks.py which already works.
No HF — just identity C_occ for the lowest orbitals.
"""

import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import psi_fn, train_model
from functions.Physics import compute_coulomb_interaction
from PINN import PINN


def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
    """Non-interacting C_occ: occupy lowest Cartesian HO orbitals."""
    n_occ = N // 2
    nx = {2: 2, 6: 3, 12: 4, 20: 5}.get(N, 4)
    ny = nx
    n_basis = nx * ny
    L = max(8.0, 3.0 / math.sqrt(omega))

    config.update(
        omega=omega,
        n_particles=N,
        d=d,
        L=L,
        n_grid=80,
        nx=nx,
        ny=ny,
        basis="cart",
        device=str(device),
        dtype="float64",
    )

    # Energy ordering for 2D Cartesian HO: E = omega*(nx + ny + 1)
    energies = []
    for ix in range(nx):
        for iy in range(ny):
            energies.append((omega * (ix + iy + 1), ix, iy))
    energies.sort(key=lambda t: t[0])

    C_occ_np = np.zeros((n_basis, n_occ), dtype=np.float64)
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C_occ_np[ix * ny + iy, k] = 1.0

    C_occ = torch.tensor(C_occ_np, dtype=dtype, device=device)
    E_ni = sum(energies[k][0] for k in range(n_occ)) * 2
    E_DMC = config.DMC_ENERGIES.get(N, {}).get(config._snap_omega(omega), None)

    params = config.get().as_dict()
    params["device"] = device
    params["torch_dtype"] = dtype
    params["E"] = E_DMC

    occ_str = ", ".join(f"({energies[k][1]},{energies[k][2]})" for k in range(n_occ))
    print(f"N={N}, ω={omega}, basis={nx}×{ny}={n_basis}")
    print(f"  Occupied: {occ_str}   E_ni={E_ni:.2f}   E_DMC={E_DMC}")
    return C_occ, params


def run_2e():
    device = "cpu"
    dtype = torch.float64

    C_occ, params = setup_noninteracting(2, 1.0, device=device, dtype=dtype)

    # Training params expected by train_model
    params["n_epochs"] = 200
    params["N_collocation"] = 1024

    f_net = (
        PINN(
            n_particles=2,
            d=2,
            omega=1.0,
            dL=5,
            hidden_dim=64,
            n_layers=2,
            act="gelu",
            init="xavier",
            use_gate=True,
            use_pair_attn=False,
        )
        .to(device)
        .to(dtype)
    )

    n_p = sum(p.numel() for p in f_net.parameters())
    print(f"PINN params: {n_p:,}")

    optimizer = torch.optim.Adam(f_net.parameters(), lr=3e-4)

    f_net, _, optimizer, hist = train_model(
        f_net,
        optimizer,
        C_occ,
        psi_fn=psi_fn,
        lap_mode="exact",
        objective="energy_var",
        micro_batch=128,
        grad_clip=0.3,
        print_every=10,
        quantile_trim=0.03,
        use_huber=True,
        huber_delta=1.0,
        params=params,
    )

    # VMC evaluation
    print("\n── VMC evaluation ──")
    result = evaluate_energy_vmc(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=None,
        params=params,
        n_samples=15_000,
        batch_size=512,
        sampler_steps=50,
        sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True,
        sampler_burn_in=300,
        sampler_thin=3,
        progress=True,
    )
    E = result["E_mean"]
    E_std = result["E_stderr"]
    print(f"  VMC E = {E:.6f} ± {E_std:.6f}  (target 3.0, err {abs(E-3.0)/3.0*100:.2f}%)")
    return f_net, result, hist


if __name__ == "__main__":
    run_2e()
