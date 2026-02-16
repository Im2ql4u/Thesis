"""
Experiment runner: UnifiedCTNN training experiments.

Targets:
  N=2, ω=1.0  → E_DMC = 3.00000  (sanity check)
  N=6, ω=0.5  → E_DMC = 11.78484

Strategy:
  1) Adam pre-training with **residual** objective (pure PDE residual minimization)
  2) SR fine-tuning with natural gradient
  3) All Laplacians computed **analytically** (exact mode)
  4) Variational bound check: E < E_GS flags a warning
"""

import math
import sys
import time

import torch

# ── project imports ──
import config
from PINN import UnifiedCTNN
from functions.Neural_Networks import psi_fn, train_model
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import compute_integrals, hartree_fock_closed_shell
from functions.Stochastic_Reconfiguration import train_model_sr_energy
from functions.Energy import evaluate_energy_vmc


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def setup_system(N, omega, d=2, n_max_basis=5, device="cpu", dtype=torch.float64):
    """Build HF orbitals and params dict for a given system."""
    L = max(8.0, 3.0 / math.sqrt(omega))
    n_grid = 50

    if N <= 2:
        nx, ny = 2, 2
    elif N <= 6:
        nx, ny = 4, 4
    elif N <= 12:
        nx, ny = 5, 5
    else:
        nx, ny = 6, 6

    params = {
        "omega": omega,
        "n_particles": N,
        "d": d,
        "device": device,
        "torch_dtype": dtype,
        "basis": "cart",
        "basis_n_max": n_max_basis,
        "L": L,
        "n_grid": n_grid,
        "nx": nx,
        "ny": ny,
        "hf_verbose": False,
    }

    Hcore, two_dirac, _info = compute_integrals(params=params)
    C_occ_np, orb_e, E_hf = hartree_fock_closed_shell(Hcore, two_dirac, params=params)
    C_occ = torch.tensor(C_occ_np, dtype=dtype, device=device)

    E_DMC = config.DMC_ENERGIES.get(N, {}).get(config._snap_omega(omega), None)
    params["E"] = E_DMC
    params["E_HF"] = float(E_hf)

    print(f"System: N={N}, ω={omega}, d={d}")
    print(f"  E_HF  = {E_hf:.6f}")
    print(f"  E_DMC = {E_DMC}")
    sys.stdout.flush()

    return C_occ, params


def evaluate(f_net, C_occ, params, n_samples=20_000, label=""):
    """Quick energy evaluation with exact Laplacian."""
    print(f"\n── Evaluating {label} ──")
    sys.stdout.flush()
    result = evaluate_energy_vmc(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=None,
        params=params,
        n_samples=n_samples,
        batch_size=512,
        sampler_steps=50,
        sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True,
        sampler_burn_in=200,
        sampler_thin=2,
        progress=True,
    )
    E = result["E_mean"]
    E_std = result["E_stderr"]
    E_ref = params.get("E", None)
    print(f"  E = {E:.6f} ± {E_std:.6f}  (target: {E_ref})")
    if E_ref is not None and E < E_ref:
        print(f"  ⚠ WARNING: E < E_GS ({E:.6f} < {E_ref}) — possible bias or statistical fluctuation")
    sys.stdout.flush()
    return result


def run_adam(
    f_net,
    C_occ,
    params,
    *,
    n_epochs=400,
    lr=5e-4,
    N_collocation=1000,
    micro_batch=256,
    objective="residual",
    disable_adaptive=True,
    grad_clip=0.5,
    print_every=50,
    alpha_start=0.0,
    alpha_end=0.0,
    alpha_decay_frac=0.0,
    lap_mode="exact",
):
    """Run Adam-based pre-training."""
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    f_net.to(device).to(dtype)
    optimizer = torch.optim.Adam(f_net.parameters(), lr=lr)

    train_params = dict(params)
    train_params["n_epochs"] = n_epochs
    train_params["N_collocation"] = N_collocation

    if disable_adaptive:
        train_params["sampler_hard_enable"] = False

    print(f"\n── Adam ({n_epochs} ep, obj={objective}, lr={lr}, lap={lap_mode}) ──")
    sys.stdout.flush()
    t0 = time.time()
    # For residual objective, alpha is irrelevant (E_eff=running mean of E_L).
    # We still pass alpha params to satisfy the API but they have no effect.
    f_net, _, optimizer, history = train_model(
        f_net,
        optimizer,
        C_occ,
        psi_fn=psi_fn,
        objective=objective,
        lap_mode=lap_mode,
        backflow_net=None,
        params=train_params,
        micro_batch=micro_batch,
        grad_clip=grad_clip,
        print_every=print_every,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_decay_frac=alpha_decay_frac,
        quantile_trim=0.05,
        use_huber=True,
        huber_delta=2.0,
    )
    dt = time.time() - t0
    print(f"  Adam done in {dt:.1f}s")
    sys.stdout.flush()
    return f_net, optimizer, history


def run_sr(
    f_net,
    C_occ,
    params,
    *,
    n_steps=200,
    step_size=0.008,
    max_param_step=0.008,
    damping=5e-3,
    total_rows=3000,
    micro_batch=600,
    sampler_steps=50,
    log_every=10,
):
    """Run SR natural-gradient fine-tuning."""
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    f_net.to(device).to(dtype)

    print(f"\n── SR ({n_steps} steps, η={step_size}, B={total_rows}) ──")
    sys.stdout.flush()
    t0 = time.time()
    f_net, _ = train_model_sr_energy(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=None,
        params=params,
        n_sr_steps=n_steps,
        log_every=log_every,
        step_size=step_size,
        max_param_step=max_param_step,
        damping=damping,
        total_rows=total_rows,
        micro_batch=micro_batch,
        sampler_steps=sampler_steps,
        sampler_step_sigma=0.10,
        lap_mode="exact",
        store_dtype=torch.float64,
    )
    dt = time.time() - t0
    print(f"  SR done in {dt:.1f}s")
    sys.stdout.flush()
    return f_net


def make_net(N, omega, node_hidden=64, edge_hidden=64, n_mp_steps=1,
             msg_layers=2, node_layers=2, jastrow_hidden=32, jastrow_layers=2,
             bf_scale_init=0.05, device="cpu", dtype=torch.float64):
    """Create a UnifiedCTNN with the given hyperparameters."""
    net = UnifiedCTNN(
        d=2,
        n_particles=N,
        omega=omega,
        node_hidden=node_hidden,
        edge_hidden=edge_hidden,
        msg_layers=msg_layers,
        node_layers=node_layers,
        n_mp_steps=n_mp_steps,
        act="silu",
        jastrow_hidden=jastrow_hidden,
        jastrow_layers=jastrow_layers,
        bf_scale_init=bf_scale_init,
    ).to(device).to(dtype)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  UnifiedCTNN params: {n_params:,}")
    sys.stdout.flush()
    return net


# ────────────────────────────────────────────────────────────────────
# Experiments
# ────────────────────────────────────────────────────────────────────

def experiment_2e_sanity(device="cpu"):
    """2e ω=1.0 sanity check. Target: E=3.0"""
    N, omega = 2, 1.0
    dtype = torch.float64
    C_occ, params = setup_system(N, omega, device=device, dtype=dtype)
    net = make_net(N, omega, node_hidden=64, edge_hidden=64,
                   jastrow_hidden=32, jastrow_layers=2, device=device, dtype=dtype)

    # Adam: 300 epochs, residual objective, exact Laplacian
    net, opt, hist = run_adam(
        net, C_occ, params,
        n_epochs=300,
        lr=5e-4,
        N_collocation=600,
        micro_batch=200,
        objective="residual",
        print_every=50,
        lap_mode="exact",
    )
    eval1 = evaluate(net, C_occ, params, n_samples=10_000, label="2e after Adam")

    # SR: 100 steps
    net = run_sr(
        net, C_occ, params,
        n_steps=100,
        step_size=0.01,
        max_param_step=0.01,
        damping=5e-3,
        total_rows=2000,
        micro_batch=500,
        sampler_steps=40,
        log_every=10,
    )
    eval2 = evaluate(net, C_occ, params, n_samples=15_000, label="2e after SR")

    E = eval2["E_mean"]
    flag = " ⚠ BELOW GS!" if E < 3.0 else ""
    print(f"\n  *** 2e RESULT: E = {E:.5f}  (target 3.00000, err = {abs(E-3.0)/3.0*100:.2f}%){flag} ***")
    sys.stdout.flush()
    return net, eval2


def experiment_6e_unified(device="cpu"):
    """6e ω=0.5 with UnifiedCTNN. Target: E_DMC=11.78484"""
    N, omega = 6, 0.5
    dtype = torch.float64
    C_occ, params = setup_system(N, omega, device=device, dtype=dtype)
    net = make_net(N, omega, node_hidden=128, edge_hidden=128,
                   msg_layers=2, node_layers=3,
                   jastrow_hidden=64, jastrow_layers=2,
                   bf_scale_init=0.05, device=device, dtype=dtype)

    # Phase 1: Adam, residual objective, fixed sampler, exact Laplacian
    net, opt, hist = run_adam(
        net, C_occ, params,
        n_epochs=500,
        lr=3e-4,
        N_collocation=1200,
        micro_batch=200,
        objective="residual",
        print_every=50,
        lap_mode="exact",
    )
    eval1 = evaluate(net, C_occ, params, n_samples=10_000, label="6e after Adam")

    # Phase 2: SR fine-tuning
    net = run_sr(
        net, C_occ, params,
        n_steps=200,
        step_size=0.008,
        max_param_step=0.008,
        damping=5e-3,
        total_rows=3000,
        micro_batch=600,
        sampler_steps=50,
        log_every=10,
    )
    eval2 = evaluate(net, C_occ, params, n_samples=20_000, label="6e after SR")

    E = eval2["E_mean"]
    target = 11.78484
    flag = " ⚠ BELOW GS!" if E < target else ""
    print(f"\n  *** 6e RESULT: E = {E:.5f}  (target {target:.5f}, err = {abs(E-target)/target*100:.2f}%){flag} ***")
    sys.stdout.flush()
    return net, eval2


if __name__ == "__main__":
    device = "cpu"
    print(f"Using device: {device}\n")
    sys.stdout.flush()

    # ── Step 1: Quick 2e sanity ──
    print("=" * 60)
    print("STEP 1: 2-electron sanity check (ω=1.0, target=3.0)")
    print("=" * 60)
    sys.stdout.flush()
    net2, res2 = experiment_2e_sanity(device=device)

    # ── Step 2: 6e experiment ──
    print("\n" + "=" * 60)
    print("STEP 2: 6-electron experiment (ω=0.5, target=11.78484)")
    print("=" * 60)
    sys.stdout.flush()
    net6, res6 = experiment_6e_unified(device=device)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    E2 = res2["E_mean"]
    E6 = res6["E_mean"]
    print(f"  2e: E = {E2:.5f} (target 3.00000, err {abs(E2-3.0)/3.0*100:.2f}%)")
    print(f"  6e: E = {E6:.5f} (target 11.78484, err {abs(E6-11.78484)/11.78484*100:.2f}%)")
    sys.stdout.flush()
