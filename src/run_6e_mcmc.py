"""
6e training with MCMC from |Ψ|² + energy_var targeting.

The stratified sampler fails for 6e (12D) because the fixed Gaussian mixture
has negligible overlap with |Ψ|². MCMC samples directly from |Ψ|², so every
sample gives a meaningful gradient signal.

Combined with the alpha-ramp targeting E_DMC (same schedule that made 2e
converge to 3.000), this should reach DMC for 6e.
"""
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, CTNNBackflowNet, UnifiedCTNN
from functions.Neural_Networks import (
    psi_fn,
    _laplacian_logpsi_exact,
    _make_closed_shell_spin,
)
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc


# ─────────────────────────────────────────────────────────────────
def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
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


# ─────────────────────────────────────────────────────────────────
def compute_local_energy(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    T = -0.5 * (lap_log.squeeze(-1) + g2.squeeze(-1))
    V_harm = 0.5 * omega ** 2 * (x ** 2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).squeeze(-1)
    return T + V_harm + V_int


# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def mcmc_advance(psi_log_fn, x, lp, sigma, n_steps, device, dtype):
    """Advance persistent MCMC chain n_steps. Returns (x, lp, acc_rate)."""
    nw = x.shape[0]
    acc, tot = 0, 0
    for _ in range(n_steps):
        prop = x + torch.randn_like(x) * sigma
        lp_prop = psi_log_fn(prop) * 2.0
        log_u = torch.log(torch.rand(nw, device=device, dtype=dtype) + 1e-30)
        accept = log_u < (lp_prop - lp)
        x = torch.where(accept.view(-1, 1, 1), prop, x)
        lp = torch.where(accept, lp_prop, lp)
        acc += accept.sum().item()
        tot += nw
    return x, lp, acc / max(tot, 1)


# ─────────────────────────────────────────────────────────────────
def _huber(resid, delta):
    abs_r = resid.abs()
    return torch.where(abs_r <= delta, 0.5 * resid**2,
                       delta * (abs_r - 0.5 * delta))


def train_mcmc(
    f_net, C_occ, params, *,
    backflow_net=None,
    n_epochs=200, lr=3e-4,
    n_walkers=512, rw_steps=10, burn_in=500,
    sigma_frac=0.15,
    micro_batch=128, grad_clip=0.3,
    quantile_trim=0.03, huber_delta=1.0,
    print_every=10,
    # Alpha schedule (same as train_model)
    alpha_start=0.10, alpha_end=0.90, alpha_decay_frac=0.70,
):
    """
    MCMC variance-minimization VMC with energy_var targeting.
    Same alpha schedule as train_model, but samples from |Ψ|² via MCMC.
    """
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    E_DMC = params.get("E", None)
    ell = 1.0 / math.sqrt(omega)

    f_net.to(device).to(dtype)
    if backflow_net is not None:
        backflow_net.to(device).to(dtype)

    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(device)

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net,
                        spin=spin, params=params)
        return lp

    all_params = list(f_net.parameters())
    if backflow_net is not None:
        all_params += list(backflow_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # MCMC init
    sigma = sigma_frac * ell
    x_chain = torch.randn(n_walkers, N, d, device=device, dtype=dtype) * ell
    f_net.eval()
    if backflow_net is not None:
        backflow_net.eval()
    lp_chain = psi_log_fn(x_chain) * 2.0
    print(f"  Burn-in ({burn_in} steps)...", end=" ", flush=True)
    x_chain, lp_chain, acc0 = mcmc_advance(
        psi_log_fn, x_chain, lp_chain, sigma, burn_in, device, dtype
    )
    print(f"acc={acc0:.3f}")

    # Quick sanity
    with torch.no_grad():
        x_t = x_chain[:64].clone()
    E_init = compute_local_energy(psi_log_fn, x_t, omega)
    print(f"  Initial E_L = {E_init.mean().item():.4f} ± {E_init.std().item():.4f}")

    n_p = sum(p.numel() for p in all_params)
    print(f"\n{'='*60}")
    print(f"MCMC training: {n_epochs} ep, {n_walkers} walkers, {rw_steps} RW/ep")
    print(f"  {n_p:,} params, lr={lr}, E_DMC={E_DMC}")
    print(f"  alpha: {alpha_start} → {alpha_end} over {alpha_decay_frac*100:.0f}%")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    history = []
    best_var = float("inf")
    best_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    patience = 40  # early stop if no improvement
    rechain_every = 50  # re-burn chain periodically

    for epoch in range(n_epochs):
        # ── Alpha schedule ──
        t_frac = epoch / max(1, n_epochs - 1)
        t_alpha = min(1.0, t_frac / max(1e-8, alpha_decay_frac))
        alpha = alpha_start + 0.5 * (alpha_end - alpha_start) * (
            1 - math.cos(math.pi * t_alpha)
        )
        alpha = max(0.0, min(1.0, alpha))

        # ── Periodic re-chain to prevent drift ──
        if rechain_every > 0 and epoch > 0 and epoch % rechain_every == 0:
            f_net.eval()
            if backflow_net is not None:
                backflow_net.eval()
            with torch.no_grad():
                x_chain = torch.randn(n_walkers, N, d, device=device, dtype=dtype) * ell
                lp_chain = psi_log_fn(x_chain) * 2.0
                x_chain, lp_chain, _ = mcmc_advance(
                    psi_log_fn, x_chain, lp_chain, sigma, burn_in // 2, device, dtype
                )

        # ── Advance MCMC ──
        f_net.eval()
        if backflow_net is not None:
            backflow_net.eval()
        x_chain, lp_chain, acc_rate = mcmc_advance(
            psi_log_fn, x_chain, lp_chain, sigma, rw_steps, device, dtype
        )
        if acc_rate > 0.6:
            sigma *= 1.02
        elif acc_rate < 0.4:
            sigma *= 0.98
        sigma = max(0.03 * ell, min(sigma, 0.5 * ell))

        X = x_chain.detach().clone()

        # ── Compute loss ──
        f_net.train()
        if backflow_net is not None:
            backflow_net.train()
        optimizer.zero_grad(set_to_none=True)
        all_EL = []
        n_batches = max(1, math.ceil(X.shape[0] / micro_batch))

        for i in range(0, X.shape[0], micro_batch):
            x_mb = X[i:i + micro_batch]
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
            E_eff = alpha * float(E_DMC) + (1.0 - alpha) * mu
            resid = E_L - E_eff
            loss = _huber(resid, huber_delta).mean()
            (loss / n_batches).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        # Refresh lp for chain
        f_net.eval()
        if backflow_net is not None:
            backflow_net.eval()
        with torch.no_grad():
            lp_chain = psi_log_fn(x_chain) * 2.0

        # Logging
        if len(all_EL) > 0:
            EL_cat = torch.cat(all_EL)
            E_mean = EL_cat.mean().item()
            E_var = EL_cat.var().item()
            E_std = EL_cat.std().item()
        else:
            E_mean, E_var, E_std = float("nan"), float("nan"), float("nan")

        history.append({"epoch": epoch, "E_mean": E_mean, "E_var": E_var, "acc": acc_rate})

        # ── Best model tracking ──
        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            if backflow_net is not None:
                best_bf_state = {k: v.clone() for k, v in backflow_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience and epoch > 50:
            print(f"  Early stopping at epoch {epoch} (best var={best_var:.3e})")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            err = abs(E_mean - E_DMC) / abs(E_DMC) * 100 if E_DMC else 0
            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  acc={acc_rate:.2f}  "
                f"t={dt:.0f}s  err={err:.2f}%"
            )
            sys.stdout.flush()

    # Restore best
    if best_state:
        f_net.load_state_dict(best_state)
        if backflow_net is not None and best_bf_state:
            backflow_net.load_state_dict(best_bf_state)
        print(f"Restored best model (var={best_var:.3e})")

    total_time = time.time() - t0
    print(f"Training done in {total_time:.0f}s ({total_time/60:.1f}min)")
    return f_net, backflow_net, history


# ─────────────────────────────────────────────────────────────────
def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
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
    E, E_std = result["E_mean"], result["E_stderr"]
    E_ref = params.get("E")
    err = abs(E - E_ref) / abs(E_ref) * 100 if E_ref else 0
    print(f"  E = {E:.6f} ± {E_std:.6f}  (target {E_ref}, err {err:.2f}%)")
    return result


# ─────────────────────────────────────────────────────────────────
def run_6e_pinn():
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(device).to(dtype)
    print(f"PINN params: {sum(p.numel() for p in f_net.parameters()):,}")
    f_net, _, hist = train_mcmc(
        f_net, C_occ, params,
        n_epochs=200, lr=3e-4,
        n_walkers=512, rw_steps=10, burn_in=500,
        micro_batch=128, grad_clip=0.3, print_every=10,
        alpha_start=0.05, alpha_end=0.60, alpha_decay_frac=0.80,
    )
    return evaluate(f_net, C_occ, params, label="6e PINN dL=8 MCMC")


def run_6e_ctnn_pinn():
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(device).to(dtype)
    bf_net = CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=0.05,
        omega=0.5,
    ).to(device).to(dtype)
    np_total = sum(p.numel() for p in f_net.parameters()) + sum(p.numel() for p in bf_net.parameters())
    print(f"CTNN+PINN params: {np_total:,}")
    f_net, bf_net, hist = train_mcmc(
        f_net, C_occ, params,
        backflow_net=bf_net,
        n_epochs=200, lr=3e-4,
        n_walkers=512, rw_steps=10, burn_in=500,
        micro_batch=128, grad_clip=0.3, print_every=10,
        alpha_start=0.05, alpha_end=0.60, alpha_decay_frac=0.80,
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="6e CTNN+PINN MCMC")


def run_6e_unified():
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    net = UnifiedCTNN(
        d=2, n_particles=6, omega=0.5,
        node_hidden=64, edge_hidden=64,
        msg_layers=1, node_layers=2, n_mp_steps=1,
        jastrow_hidden=32, jastrow_layers=2,
        envelope_width_aho=3.0,
    ).to(device).to(dtype)
    print(f"UnifiedCTNN params: {sum(p.numel() for p in net.parameters()):,}")
    net, _, hist = train_mcmc(
        net, C_occ, params,
        n_epochs=200, lr=2e-4,
        n_walkers=512, rw_steps=10, burn_in=500,
        micro_batch=128, grad_clip=0.3, print_every=10,
        alpha_start=0.05, alpha_end=0.60, alpha_decay_frac=0.80,
    )
    return evaluate(net, C_occ, params, label="6e UnifiedCTNN MCMC")


if __name__ == "__main__":
    results = {}

    print("#" * 60)
    print("# 6e PINN (dL=8) + MCMC")
    print("#" * 60)
    results["pinn"] = run_6e_pinn()

    print("\n" + "#" * 60)
    print("# 6e CTNN+PINN + MCMC")
    print("#" * 60)
    results["ctnn"] = run_6e_ctnn_pinn()

    print("\n" + "#" * 60)
    print("# 6e UnifiedCTNN + MCMC")
    print("#" * 60)
    results["unified"] = run_6e_unified()

    target = 11.78484
    print(f"\n{'='*60}")
    print("SUMMARY — 6e ω=0.5 (all MCMC)")
    print(f"{'='*60}")
    for name, r in results.items():
        E, se = r["E_mean"], r["E_stderr"]
        err = abs(E - target) / target * 100
        print(f"  {name:15s}  E={E:.6f} ± {se:.6f}  err={err:.2f}%")
    print(f"  {'target':15s}  E={target:.6f}")
    print(f"{'='*60}")
