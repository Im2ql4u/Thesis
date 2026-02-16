"""
Diagnostic: Train 2e briefly with residual, then compare E_L on:
  (a) Stratified points (what training sees)
  (b) |Ψ|²-sampled points (what VMC evaluation uses)

If there's a big gap, the sampler isn't covering the regions that matter.
"""
import math
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

from PINN import UnifiedCTNN
from functions.Neural_Networks import (
    psi_fn,
    train_model,
    _laplacian_logpsi_exact,
)
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import compute_integrals, hartree_fock_closed_shell
from functions.Energy import evaluate_energy_vmc
from run_experiments import setup_system, make_net


def local_energy_batch(f_net, x, C_occ, params, spin=None):
    """Compute E_L(x) = T(x) + V(x) for a batch using exact Laplacian."""
    N = params["n_particles"]
    d = params["d"]
    omega = params["omega"]

    if spin is None:
        up = N // 2
        spin = torch.cat([
            torch.zeros(up, dtype=torch.long),
            torch.ones(N - up, dtype=torch.long),
        ]).to(x.device)

    def psi_log_fn(y):
        logpsi, _ = psi_fn(f_net, y, C_occ, backflow_net=None, spin=spin, params=params)
        return logpsi

    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)

    # T = -0.5 * (lap_log + g²)
    T = -0.5 * (lap_log.squeeze(-1) + g2.squeeze(-1))

    # V = V_harm + V_int
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).squeeze(-1)
    V = V_harm + V_int

    E_L = T + V
    return E_L.detach()


def sample_from_psi2(f_net, C_occ, params, n_samples=5000, burn_in=500, thin=5, sigma=0.12):
    """Metropolis sampling from |Ψ|²."""
    N = params["n_particles"]
    d = params["d"]
    omega = params["omega"]
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    ell = 1.0 / math.sqrt(omega)

    up = N // 2
    spin = torch.cat([
        torch.zeros(up, dtype=torch.long),
        torch.ones(N - up, dtype=torch.long),
    ]).to(device)

    f_net.eval()

    def log_psi2(y):
        with torch.no_grad():
            lp, _ = psi_fn(f_net, y, C_occ, backflow_net=None, spin=spin, params=params)
        return 2.0 * lp  # log|Ψ|²

    # Initialize
    x = torch.randn(1, N, d, device=device, dtype=dtype) * ell
    lp = log_psi2(x)

    sig = sigma * ell
    samples = []
    accepted = 0
    total = 0

    for step in range(burn_in + n_samples * thin):
        prop = x + torch.randn_like(x) * sig
        lp_prop = log_psi2(prop)
        if torch.log(torch.rand(1, device=device, dtype=dtype)) < (lp_prop - lp):
            x = prop
            lp = lp_prop
            accepted += 1
        total += 1

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(x.clone())
            if len(samples) >= n_samples:
                break

    acc_rate = accepted / total
    return torch.cat(samples, dim=0), acc_rate


def sample_stratified(N, d, omega, n_samples=5000, device="cpu", dtype=torch.float64):
    """Simple stratified sampling (center + tails mixture) mimicking train_model."""
    ell = 1.0 / math.sqrt(omega)
    x = torch.randn(n_samples, N, d, device=device, dtype=dtype) * ell * 0.7
    return x


def main():
    device = "cpu"
    dtype = torch.float64
    N, omega = 2, 1.0

    print("=" * 70)
    print("DIAGNOSTIC: E_L gap between training sampler and VMC sampler")
    print(f"System: N={N}, ω={omega}, target E_DMC=3.0")
    print("=" * 70)
    sys.stdout.flush()

    C_occ, params = setup_system(N, omega, device=device, dtype=dtype)

    # --- Train for 200 epochs with residual ---
    net = make_net(N, omega, node_hidden=64, edge_hidden=64,
                   jastrow_hidden=32, jastrow_layers=2, device=device, dtype=dtype)

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    train_params = dict(params)
    train_params["n_epochs"] = 200
    train_params["N_collocation"] = 600
    train_params["sampler_hard_enable"] = False

    print("\n--- Training 200ep with residual, exact Laplacian ---")
    sys.stdout.flush()
    t0 = time.time()
    net, _, optimizer, hist = train_model(
        net, optimizer, C_occ,
        psi_fn=psi_fn,
        objective="residual",
        lap_mode="exact",
        backflow_net=None,
        params=train_params,
        micro_batch=200,
        grad_clip=0.5,
        print_every=50,
        alpha_start=0.0, alpha_end=0.0, alpha_decay_frac=0.0,
        quantile_trim=0.05,
        use_huber=True, huber_delta=2.0,
    )
    print(f"Training done in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    net.eval()

    # --- Compute E_L on stratified (training-like) points ---
    print("--- E_L on STRATIFIED points (what training sees) ---")
    sys.stdout.flush()
    x_strat = sample_stratified(N, 2, omega, n_samples=3000, device=device, dtype=dtype)
    EL_strat = local_energy_batch(net, x_strat, C_occ, params)
    good = torch.isfinite(EL_strat)
    EL_strat = EL_strat[good]
    print(f"  N_samples: {EL_strat.numel()}")
    print(f"  E_L mean: {EL_strat.mean():.6f}")
    print(f"  E_L std:  {EL_strat.std():.6f}")
    print(f"  E_L median: {EL_strat.median():.6f}")
    print(f"  E_L min/max: {EL_strat.min():.4f} / {EL_strat.max():.4f}")
    sys.stdout.flush()

    # --- Compute E_L on |Ψ|²-sampled points ---
    print("\n--- E_L on |Ψ|²-sampled points (what VMC sees) ---")
    sys.stdout.flush()
    x_vmc, acc_rate = sample_from_psi2(net, C_occ, params, n_samples=3000, burn_in=500, thin=5)
    print(f"  MCMC acceptance rate: {acc_rate:.3f}")
    EL_vmc = local_energy_batch(net, x_vmc, C_occ, params)
    good = torch.isfinite(EL_vmc)
    EL_vmc = EL_vmc[good]
    print(f"  N_samples: {EL_vmc.numel()}")
    print(f"  E_L mean: {EL_vmc.mean():.6f}")
    print(f"  E_L std:  {EL_vmc.std():.6f}")
    print(f"  E_L median: {EL_vmc.median():.6f}")
    print(f"  E_L min/max: {EL_vmc.min():.4f} / {EL_vmc.max():.4f}")
    sys.stdout.flush()

    # --- Also look at |Ψ|² density on both sets ---
    print("\n--- log|Ψ|² distribution comparison ---")
    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N-up, dtype=torch.long)]).to(device)
    with torch.no_grad():
        lp_strat, _ = psi_fn(net, x_strat[:1000], C_occ, backflow_net=None, spin=spin, params=params)
        lp_vmc, _ = psi_fn(net, x_vmc[:1000], C_occ, backflow_net=None, spin=spin, params=params)
    print(f"  Stratified: log|Ψ| mean={lp_strat.mean():.4f}, std={lp_strat.std():.4f}")
    print(f"  VMC:        log|Ψ| mean={lp_vmc.mean():.4f}, std={lp_vmc.std():.4f}")

    # --- Spatial distribution ---
    r_strat = x_strat[:1000].norm(dim=-1).mean(dim=-1)
    r_vmc = x_vmc[:1000].norm(dim=-1).mean(dim=-1)
    print(f"\n  Stratified <|r|>: mean={r_strat.mean():.4f}, std={r_strat.std():.4f}")
    print(f"  VMC        <|r|>: mean={r_vmc.mean():.4f}, std={r_vmc.std():.4f}")

    # --- The gap ---
    gap = float(EL_strat.mean() - EL_vmc.mean())
    print(f"\n{'='*70}")
    print(f"GAP: E_L(strat) - E_L(vmc) = {gap:.6f}")
    print(f"  Training sees:  {EL_strat.mean():.6f} ± {EL_strat.std():.6f}")
    print(f"  VMC evaluates:  {EL_vmc.mean():.6f} ± {EL_vmc.std():.6f}")
    print(f"  TARGET:         3.000000")
    print(f"{'='*70}")
    sys.stdout.flush()

    # --- Full VMC evaluation for comparison ---
    print("\n--- Full VMC evaluation (exact Laplacian) ---")
    sys.stdout.flush()
    result = evaluate_energy_vmc(
        net, C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=None,
        params=params,
        n_samples=10_000,
        batch_size=512,
        sampler_steps=50,
        sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True,
        sampler_burn_in=200,
        sampler_thin=2,
        progress=False,
    )
    print(f"  VMC E = {result['E_mean']:.6f} ± {result['E_stderr']:.6f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
