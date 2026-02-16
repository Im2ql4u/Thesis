"""Quick sanity check for K-FAC components."""

import sys

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")
import math

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

from run_6e_kfac import (
    KFACPreconditioner,
    compute_bf_laplacian_penalty,
    damped_slogdet,
    make_nets,
    make_psi_log_fn,
)
from run_6e_residual import compute_local_energy, screened_collocation, setup_noninteracting

C_occ, params = setup_noninteracting(6, 0.5, d=2)
f_net, bf_net = make_nets(0.7, False)
psi_log_fn, spin = make_psi_log_fn(f_net, bf_net, C_occ, params)

# Test K-FAC hooks
kfac = KFACPreconditioner(bf_net, damping=1e-3, ema=0.95)
kfac.register_hooks()
print(f"K-FAC tracking {len(kfac.linear_layers)} linear layers")

sigma = 1.3 / math.sqrt(0.5)
X = screened_collocation(
    psi_log_fn, 6, 2, sigma, n_keep=64, oversampling=10, device="cpu", dtype=torch.float64
)
f_net.train()
bf_net.train()
E_L = compute_local_energy(psi_log_fn, X[:16], 0.5).view(-1)
loss = (E_L**2).mean()
loss.backward()
kfac.step()
kfac.remove_hooks()
print("K-FAC step OK")

# Test smoothness penalty
bf_net.zero_grad()
pen = compute_bf_laplacian_penalty(bf_net, X[:16], spin, n_samples=8)
pen.backward()
print(f"Smoothness penalty = {pen.item():.6f}, OK")

# Test damped slogdet
M = torch.randn(4, 3, 3, dtype=torch.float64, requires_grad=True)
sign, logabs = damped_slogdet(M, damping=1e-4)
logabs.sum().backward()
print(f"Damped slogdet grad norm = {M.grad.norm().item():.4f}, OK")

print("ALL SANITY CHECKS PASSED")
