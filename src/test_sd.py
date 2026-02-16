"""Quick test of the SD-only setup."""

import sys

import torch

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

from functions.Neural_Networks import psi_fn
from PINN import ZeroJastrow
from train_simple import compute_local_energy, setup_system

device = "cpu"
dtype = torch.float64

# Setup
C_occ, params = setup_system(2, 1.0, device=device, dtype=dtype)
print("C_occ shape:", C_occ.shape)
print("C_occ:\n", C_occ)

# SD-only (zero Jastrow)
f_net = ZeroJastrow().to(device).to(dtype)
spin = torch.cat([torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long)]).to(device)


def psi_log_fn(y):
    lp, _ = psi_fn(f_net, y, C_occ, backflow_net=None, spin=spin, params=params)
    return lp


# Test E_L at random points
x = torch.randn(64, 2, 2, dtype=dtype, device=device) * 1.0
E_L = compute_local_energy(psi_log_fn, x, 1.0)
print(f"\nE_L at random points: {E_L.mean().item():.4f} ± {E_L.std().item():.4f}")
print(f"E_L range: [{E_L.min().item():.4f}, {E_L.max().item():.4f}]")

# Also test: what's log|Ψ| at origin?
x_zero = torch.zeros(1, 2, 2, dtype=dtype, device=device)
x_zero[0, 0, :] = torch.tensor([0.1, 0.0])
x_zero[0, 1, :] = torch.tensor([-0.1, 0.0])
lp = psi_log_fn(x_zero)
print(f"\nlog|Ψ| near origin: {lp.item():.4f}")

# Check the non-interacting energy component
x_test = torch.randn(256, 2, 2, dtype=dtype, device=device) * 0.8
E_L_test = compute_local_energy(psi_log_fn, x_test, 1.0)
good = torch.isfinite(E_L_test)
E_L_test = E_L_test[good]
print(f"\nE_L (256 samples, σ=0.8): {E_L_test.mean().item():.4f} ± {E_L_test.std().item():.4f}")
