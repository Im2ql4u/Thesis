"""
Simple VMC: non-interacting SD × PINN Jastrow, variance minimization.

Key simplifications vs train_residual.py:
  - NO Hartree-Fock: C_occ is just identity (occupy lowest orbitals)
  - Simple PINN Jastrow (not UnifiedCTNN)
  - MCMC sampling from |Ψ|²
  - Exact analytic Laplacians only
  - Adam optimizer (no SR)

For N=2, ω=1.0: the non-interacting SD has both electrons in φ₀₀.
  SD energy = 2.0, Coulomb adds ~1.0 → exact E_DMC = 3.0.
  The Jastrow captures the correlation.
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import (
    _laplacian_logpsi_exact,
    psi_fn,
)
from functions.Physics import compute_coulomb_interaction
from PINN import PINN, ZeroJastrow

# ─────────────────────────────────────────────────────────────────
# System setup — NO HF, just non-interacting orbitals
# ─────────────────────────────────────────────────────────────────


def setup_system(N, omega, d=2, device="cpu", dtype=torch.float64):
    """Set up with non-interacting C_occ (no Hartree-Fock).

    For closed-shell 2D HO:
      - Basis: Cartesian product of 1D HO eigenstates
      - Ordering: (nx,ny) = (0,0), (0,1), (1,0), (1,1), ...
      - Energies: ω(nx + ny + 1)
      - n_occ = N/2 (closed shell, each orbital holds 2 electrons)
      - C_occ: identity-like, just the lowest n_occ orbitals
    """
    # Choose basis size: enough to hold n_occ orbitals
    n_occ = N // 2
    # For N=2: n_occ=1, need 1 orbital (the ground state)
    # For N=6: n_occ=3, need 3 orbitals. Shell structure:
    #   (0,0) → E=ω, (0,1) → E=2ω, (1,0) → E=2ω, (0,2)→E=3ω, (1,1)→E=3ω, (2,0)→E=3ω
    #   So n_occ=3 fills: (0,0), (0,1), (1,0)
    nx = {2: 2, 6: 3, 12: 4, 20: 5}.get(N, 4)
    ny = nx
    n_basis = nx * ny

    L = max(8.0, 3.0 / math.sqrt(omega))

    # Set global config so @inject_params picks up correct omega
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

    params = config.get().as_dict()
    params["device"] = device
    params["torch_dtype"] = dtype

    # ── Non-interacting C_occ ──
    # Build energy ordering for Cartesian 2D HO basis
    energies = []
    for ix in range(nx):
        for iy in range(ny):
            energies.append((omega * (ix + iy + 1), ix, iy))
    energies.sort(key=lambda t: t[0])

    # C_occ: (n_basis, n_occ) — columns are the occupied orbitals
    C_occ_np = np.zeros((n_basis, n_occ), dtype=np.float64)
    for k in range(n_occ):
        _, ix, iy = energies[k]
        basis_idx = ix * ny + iy  # consistent with reshape ordering in batch_2d
        C_occ_np[basis_idx, k] = 1.0

    C_occ = torch.tensor(C_occ_np, dtype=dtype, device=device)

    # Non-interacting energy (for reference)
    E_ni = sum(energies[k][0] for k in range(n_occ)) * 2  # ×2 for spin
    E_DMC = config.DMC_ENERGIES.get(N, {}).get(config._snap_omega(omega), None)
    params["E"] = E_DMC

    occ_str = ", ".join(f"({energies[k][1]},{energies[k][2]})" for k in range(n_occ))
    print(f"System: N={N}, ω={omega}, basis={nx}×{ny}={n_basis}")
    print(f"  Occupied orbitals: {occ_str}")
    print(f"  E_non-int = {E_ni:.4f}  (no Coulomb)")
    print(f"  E_DMC     = {E_DMC}")
    sys.stdout.flush()
    return C_occ, params


# ─────────────────────────────────────────────────────────────────
# Local energy
# ─────────────────────────────────────────────────────────────────


def compute_local_energy(psi_log_fn, x, omega):
    """E_L(x) = T(x) + V(x) using exact analytic Laplacian."""
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    T = -0.5 * (lap_log.squeeze(-1) + g2.squeeze(-1))
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).squeeze(-1)
    E_L = T + V_harm + V_int
    return E_L


# ─────────────────────────────────────────────────────────────────
# MCMC sampling
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def mcmc_sample(psi_log_fn, x_chain, sigma, n_steps, device, dtype):
    """Advance persistent MCMC chain. Returns (x_chain, lp_chain, acc_rate)."""
    n_walkers = x_chain.shape[0]
    lp_chain = psi_log_fn(x_chain) * 2.0
    acc, tot = 0, 0
    for _ in range(n_steps):
        prop = x_chain + torch.randn_like(x_chain) * sigma
        lp_prop = psi_log_fn(prop) * 2.0
        log_u = torch.log(torch.rand(n_walkers, device=device, dtype=dtype) + 1e-30)
        accept = log_u < (lp_prop - lp_chain)
        x_chain = torch.where(accept.view(-1, 1, 1), prop, x_chain)
        lp_chain = torch.where(accept, lp_prop, lp_chain)
        acc += accept.sum().item()
        tot += n_walkers
    return x_chain, lp_chain, acc / max(tot, 1)


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────


def train(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    params: dict,
    *,
    n_epochs: int = 300,
    lr: float = 3e-4,
    n_walkers: int = 1024,
    rw_steps: int = 20,
    burn_in: int = 500,
    sigma_frac: float = 0.15,
    micro_batch: int = 128,
    grad_clip: float = 1.0,
    quantile_trim: float = 0.03,
    print_every: int = 10,
    huber_delta: float = 2.0,
    warmup_epochs: int = 10,
    rechain_every: int = 0,  # 0 = never re-init chain
    patience: int = 0,  # 0 = no early stopping
):
    """MCMC variance-minimization VMC with Adam."""
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    E_ref = params.get("E", None)
    ell = 1.0 / math.sqrt(omega)

    f_net.to(device).to(dtype)

    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]).to(
        device
    )

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=None, spin=spin, params=params)
        return lp

    optimizer = torch.optim.Adam(f_net.parameters(), lr=lr)

    # ── Initialize chain ──
    sigma = sigma_frac * ell
    x_chain = torch.randn(n_walkers, N, d, device=device, dtype=dtype) * ell
    f_net.eval()
    print(f"  Burn-in ({burn_in} steps, {n_walkers} walkers)...", end=" ", flush=True)
    x_chain, lp_chain, acc0 = mcmc_sample(psi_log_fn, x_chain, sigma, burn_in, device, dtype)
    print(f"acc={acc0:.3f}")

    # Quick sanity: initial E_L
    with torch.no_grad():
        x_test = x_chain[:64].clone().detach()
    x_test.requires_grad_(True)
    E_L_init = compute_local_energy(psi_log_fn, x_test, omega)
    print(f"  Initial E_L = {E_L_init.mean().item():.4f} ± {E_L_init.std().item():.4f}")
    sys.stdout.flush()

    history = []
    best_var = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training: {n_epochs} epochs, lr={lr}, {n_walkers} walkers, {rw_steps} RW/ep")
    print(f"  micro_batch={micro_batch}, grad_clip={grad_clip}")
    if E_ref:
        print(f"  Target E = {E_ref}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    t0 = time.time()

    for epoch in range(n_epochs):
        # ── LR warmup ──
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * frac
        elif epoch == warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # ── Optional re-chain ──
        if rechain_every > 0 and epoch > 0 and epoch % rechain_every == 0:
            f_net.eval()
            x_chain = torch.randn(n_walkers, N, d, device=device, dtype=dtype) * ell
            x_chain, lp_chain, _ = mcmc_sample(
                psi_log_fn, x_chain, sigma, burn_in // 2, device, dtype
            )

        # ── Step 1: Advance MCMC ──
        f_net.eval()
        x_chain, lp_chain, acc_rate = mcmc_sample(
            psi_log_fn, x_chain, sigma, rw_steps, device, dtype
        )

        # Adapt sigma
        if acc_rate > 0.6:
            sigma *= 1.02
        elif acc_rate < 0.4:
            sigma *= 0.98
        sigma = max(0.03 * ell, min(sigma, 0.5 * ell))

        X = x_chain.detach().clone()

        # ── Step 2: Compute loss ──
        f_net.train()
        optimizer.zero_grad(set_to_none=True)
        all_EL = []
        n_batches = max(1, math.ceil(X.shape[0] / micro_batch))

        for i in range(0, X.shape[0], micro_batch):
            x_mb = X[i : i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega)

            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue

            if quantile_trim > 0 and E_L.numel() > 10:
                lo = torch.quantile(E_L.detach(), quantile_trim)
                hi = torch.quantile(E_L.detach(), 1.0 - quantile_trim)
                mask = (E_L >= lo) & (E_L <= hi)
                E_L = E_L[mask]
                if E_L.numel() == 0:
                    continue

            all_EL.append(E_L.detach())
            mu = E_L.mean().detach()
            resid = E_L - mu
            abs_r = resid.abs()
            loss = torch.where(
                abs_r <= huber_delta,
                0.5 * resid**2,
                huber_delta * (abs_r - 0.5 * huber_delta),
            ).mean()
            (loss / n_batches).backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(f_net.parameters(), grad_clip)
        optimizer.step()

        # ── Refresh log|Ψ|² for chain ──
        f_net.eval()
        with torch.no_grad():
            lp_chain = psi_log_fn(x_chain) * 2.0

        # ── Logging ──
        if len(all_EL) > 0:
            EL_cat = torch.cat(all_EL)
            E_mean = EL_cat.mean().item()
            E_var = EL_cat.var().item()
            E_std = EL_cat.std().item()
        else:
            E_mean, E_var, E_std = float("nan"), float("nan"), float("nan")

        history.append({"epoch": epoch, "E_mean": E_mean, "E_var": E_var, "acc": acc_rate})

        # Best model tracking
        improved = False
        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            improved = True
        if improved:
            best_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience and epoch > warmup_epochs + 30:
            print(f"  Early stopping at epoch {epoch}")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            err_str = ""
            if E_ref and math.isfinite(E_mean):
                err_str = f"  err={abs(E_mean - E_ref)/abs(E_ref)*100:.2f}%"
            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  acc={acc_rate:.2f}  "
                f"lr={lr_now:.1e}  t={dt:.0f}s{err_str}"
            )
            sys.stdout.flush()

    # Restore best
    if best_state is not None:
        f_net.load_state_dict(best_state)
        print(f"\nRestored best (var={best_var:.3e})")

    total_time = time.time() - t0
    print(f"Training done in {total_time:.1f}s ({total_time/60:.1f}min)")
    sys.stdout.flush()
    return f_net, history


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────


def evaluate_vmc(f_net, C_occ, params, n_samples=15_000, label=""):
    """Full VMC evaluation with MCMC + exact Laplacian."""
    print(f"\n── VMC evaluation: {label} ──")
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
        sampler_burn_in=300,
        sampler_thin=3,
        progress=True,
    )
    E = result["E_mean"]
    E_std = result["E_stderr"]
    E_ref = params.get("E", None)
    print(f"  VMC E = {E:.6f} ± {E_std:.6f}")
    if E_ref:
        err = abs(E - E_ref) / abs(E_ref) * 100
        print(f"  Target: {E_ref}  error: {err:.2f}%")
    sys.stdout.flush()
    return result


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────


def test_sd_only(N, omega, device="cpu", dtype=torch.float64):
    """Test: SD-only (zero Jastrow) should give E ≈ E_non-interacting + Coulomb mean-field."""
    C_occ, params = setup_system(N, omega, device=device, dtype=dtype)
    f_net = ZeroJastrow().to(device).to(dtype)
    result = evaluate_vmc(f_net, C_occ, params, n_samples=10_000, label=f"SD-only N={N}")
    return result


def run_2e(device="cpu"):
    """2-electron ω=1.0 → target E=3.0"""
    dtype = torch.float64
    C_occ, params = setup_system(2, 1.0, device=device, dtype=dtype)

    # Simple PINN Jastrow
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

    f_net, hist = train(
        f_net,
        C_occ,
        params,
        n_epochs=400,
        lr=1e-4,
        n_walkers=1024,
        rw_steps=20,
        burn_in=500,
        sigma_frac=0.15,
        micro_batch=128,
        grad_clip=1.0,
        quantile_trim=0.03,
        print_every=20,
        huber_delta=2.0,
        warmup_epochs=10,
        rechain_every=80,
    )

    result = evaluate_vmc(f_net, C_occ, params, n_samples=15_000, label="2e final")

    E = result["E_mean"]
    print(f"\n{'='*60}")
    print(f"2e RESULT: E = {E:.5f}  (target 3.00000, err {abs(E-3.0)/3.0*100:.2f}%)")
    print(f"{'='*60}")
    return f_net, result, hist


def run_6e(device="cpu"):
    """6-electron ω=0.5 → target E=11.78484"""
    dtype = torch.float64
    C_occ, params = setup_system(6, 0.5, device=device, dtype=dtype)

    f_net = (
        PINN(
            n_particles=6,
            d=2,
            omega=0.5,
            dL=8,
            hidden_dim=128,
            n_layers=3,
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

    f_net, hist = train(
        f_net,
        C_occ,
        params,
        n_epochs=400,
        lr=2e-4,
        n_walkers=2048,
        rw_steps=20,
        burn_in=500,
        sigma_frac=0.12,
        micro_batch=128,
        grad_clip=1.0,
        quantile_trim=0.03,
        print_every=10,
        huber_delta=2.0,
        warmup_epochs=15,
    )

    result = evaluate_vmc(f_net, C_occ, params, n_samples=20_000, label="6e final")

    E = result["E_mean"]
    target = 11.78484
    print(f"\n{'='*60}")
    print(f"6e RESULT: E = {E:.5f}  (target {target:.5f}, err {abs(E-target)/target*100:.2f}%)")
    print(f"{'='*60}")
    return f_net, result, hist


if __name__ == "__main__":
    device = "cpu"

    # ── 2e with PINN Jastrow ──
    print("=" * 60)
    print("2e + PINN Jastrow → target E=3.0")
    print("=" * 60)
    run_2e(device=device)
