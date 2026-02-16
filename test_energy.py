#!/usr/bin/env python3
"""Quick sanity test: load one CTNN pair and evaluate energy."""
import gc
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import psi_fn
from functions.Physics import compute_coulomb_interaction
from PINN import PINN, CTNNBackflowNet

N, OMEGA, d = 6, 0.1, 2
nx = ny = 3
n_occ = 3

L = max(8.0, 3.0 / math.sqrt(OMEGA))
config.update(
    omega=OMEGA,
    n_particles=N,
    d=d,
    L=L,
    n_grid=80,
    nx=nx,
    ny=ny,
    basis="cart",
    device="cpu",
    dtype="float64",
)
params = config.get().as_dict()
params["torch_dtype"] = torch.float64
params["device"] = torch.device("cpu")

energies = [(OMEGA * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)]
energies.sort(key=lambda t: t[0])
C = np.zeros((nx * ny, n_occ), dtype=np.float64)
for k in range(n_occ):
    _, ix, iy = energies[k]
    C[ix * ny + iy, k] = 1.0
C_occ = torch.tensor(C, dtype=torch.float64)


def load_sd(p):
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


bf_sd = load_sd("results/models 2/6p/w_01/backflowCTNN_temp2.pt")
fn_sd = load_sd("results/models 2/6p/w_01/f_netCTNN_temp2.pt")

bf = CTNNBackflowNet(
    d=2, msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", use_spin=True, omega=OMEGA
)
bf.load_state_dict(bf_sd)
bf.to(torch.float64).eval()

f_net = PINN(N, 2, OMEGA, dL=5, hidden_dim=128, n_layers=2, act="gelu")
f_net.load_state_dict(fn_sd)
f_net.to(torch.float64).eval()

gc.collect()
metrics = evaluate_energy_vmc(
    f_net,
    C_occ,
    psi_fn=psi_fn,
    backflow_net=bf,
    compute_coulomb_interaction=compute_coulomb_interaction,
    params=params,
    persistent=True,
    n_samples=20_000,
    batch_size=1024,
    sampler_step_sigma=0.2,
    sampler_target_accept=0.5,
    sampler_adapt_lr=0.05,
    sampler_burn_in=200,
    sampler_thin=20,
    lap_mode="exact",
    lap_probes=32,
    fd_eps=3e-3,
    assume_omega_invariant=True,
    progress=True,
)
print(f"\nE = {metrics['E_mean']:.6f} +/- {metrics['E_stderr']:.6f}  target=3.55388")
print(f"accept = {metrics['accept_rate_avg']:.3f}")
