"""
Systematic benchmark: PINN, CTNN+PINN, UnifiedCTNN on 2e and 6e.

Uses verified train_model from Neural_Networks.py + non-interacting C_occ.
"""
import math
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, BackflowNet, CTNNBackflowNet, UnifiedCTNN
from functions.Neural_Networks import psi_fn, train_model
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc


def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
    """Non-interacting C_occ: occupy lowest Cartesian HO orbitals."""
    n_occ = N // 2
    nx = {2: 2, 6: 3, 12: 4, 20: 5}.get(N, 4)
    ny = nx
    n_basis = nx * ny
    L = max(8.0, 3.0 / math.sqrt(omega))

    config.update(
        omega=omega, n_particles=N, d=d,
        L=L, n_grid=80, nx=nx, ny=ny,
        basis="cart", device=str(device), dtype="float64",
    )

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


def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    """VMC evaluation with MCMC + exact Laplacian."""
    print(f"\n── VMC: {label} ──")
    result = evaluate_energy_vmc(
        f_net, C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=backflow_net, params=params,
        n_samples=n_samples, batch_size=512,
        sampler_steps=50, sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True, sampler_burn_in=300, sampler_thin=3,
        progress=True,
    )
    E = result["E_mean"]
    E_std = result["E_stderr"]
    E_ref = params.get("E")
    err = abs(E - E_ref) / abs(E_ref) * 100 if E_ref else 0
    print(f"  E = {E:.6f} ± {E_std:.6f}  (target {E_ref}, err {err:.2f}%)")
    return result


def do_train(f_net, C_occ, params, *, backflow_net=None, n_epochs=200,
             lr=3e-4, N_collocation=1024, label=""):
    """Train with train_model, return (f_net, backflow_net, result)."""
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)

    params_t = dict(params)
    params_t["n_epochs"] = n_epochs
    params_t["N_collocation"] = N_collocation

    # Collect all params for optimizer
    all_params = list(f_net.parameters())
    if backflow_net is not None:
        all_params += list(backflow_net.parameters())

    optimizer = torch.optim.Adam(all_params, lr=lr)

    n_p = sum(p.numel() for p in all_params)
    print(f"\n{'='*60}")
    print(f"Training: {label}  ({n_p:,} params, {n_epochs} epochs)")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    f_net, backflow_net_out, optimizer, hist = train_model(
        f_net, optimizer, C_occ,
        psi_fn=psi_fn,
        lap_mode="exact",
        objective="energy_var",
        micro_batch=128,
        grad_clip=0.3,
        print_every=20,
        quantile_trim=0.03,
        use_huber=True,
        huber_delta=1.0,
        backflow_net=backflow_net,
        params=params_t,
    )
    dt = time.time() - t0
    print(f"Training done in {dt:.0f}s")

    result = evaluate(f_net, C_occ, params, backflow_net=backflow_net_out, label=label)
    return f_net, backflow_net_out, result


# ─────────────────────────────────────────────────────────────────
# 2e experiments
# ─────────────────────────────────────────────────────────────────

def run_2e_unified():
    """2e with UnifiedCTNN."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(2, 1.0, device=device, dtype=dtype)

    net = UnifiedCTNN(
        d=2, n_particles=2, omega=1.0,
        node_hidden=64, edge_hidden=64,
        msg_layers=2, node_layers=2, n_mp_steps=1,
        jastrow_hidden=32, jastrow_layers=2,
        envelope_width_aho=3.0,
    ).to(device).to(dtype)

    # UnifiedCTNN is passed as f_net (psi_fn detects isinstance)
    f_net, _, result = do_train(
        net, C_occ, params,
        n_epochs=200, lr=3e-4, N_collocation=1024,
        label="2e UnifiedCTNN",
    )
    return result


# ─────────────────────────────────────────────────────────────────
# 6e experiments
# ─────────────────────────────────────────────────────────────────

def run_6e_pinn():
    """6e with PINN Jastrow only (no backflow). Benchmark."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)

    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=5, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(device).to(dtype)

    f_net, _, result = do_train(
        f_net, C_occ, params,
        n_epochs=200, lr=3e-4, N_collocation=1024,
        label="6e PINN-only",
    )
    return result


def run_6e_ctnn():
    """6e with CTNNBackflowNet + PINN Jastrow."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)

    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=5, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(device).to(dtype)

    backflow_net = CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=0.05,
        omega=0.5,
    ).to(device).to(dtype)

    f_net, bf_net, result = do_train(
        f_net, C_occ, params,
        backflow_net=backflow_net,
        n_epochs=200, lr=3e-4, N_collocation=1024,
        label="6e CTNN+PINN",
    )
    return result


def run_6e_unified():
    """6e with UnifiedCTNN."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)

    net = UnifiedCTNN(
        d=2, n_particles=6, omega=0.5,
        node_hidden=64, edge_hidden=64,
        msg_layers=1, node_layers=2, n_mp_steps=1,
        jastrow_hidden=32, jastrow_layers=2,
        envelope_width_aho=3.0,
    ).to(device).to(dtype)

    f_net, _, result = do_train(
        net, C_occ, params,
        n_epochs=200, lr=3e-4, N_collocation=1024,
        label="6e UnifiedCTNN",
    )
    return result


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    # 2e UnifiedCTNN
    print("\n" + "#"*60)
    print("# 2e: UnifiedCTNN (ω=1.0, target=3.0)")
    print("#"*60)
    results["2e_unified"] = run_2e_unified()

    # 6e PINN
    print("\n" + "#"*60)
    print("# 6e: PINN only (ω=0.5, target=11.78484)")
    print("#"*60)
    results["6e_pinn"] = run_6e_pinn()

    # 6e CTNN+PINN
    print("\n" + "#"*60)
    print("# 6e: CTNN+PINN (ω=0.5, target=11.78484)")
    print("#"*60)
    results["6e_ctnn"] = run_6e_ctnn()

    # 6e Unified
    print("\n" + "#"*60)
    print("# 6e: UnifiedCTNN (ω=0.5, target=11.78484)")
    print("#"*60)
    results["6e_unified"] = run_6e_unified()

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, r in results.items():
        E = r["E_mean"]
        E_std = r["E_stderr"]
        if "2e" in name:
            target = 3.0
        else:
            target = 11.78484
        err = abs(E - target) / target * 100
        print(f"  {name:20s}  E={E:.6f} ± {E_std:.6f}  err={err:.2f}%")
    print("="*60)
