"""
6e residual training: two-stream collocation for backflow.

The single-stream screened collocation (top-K by |Ψ|²) reaches 0.42%
with PINN-only but backflow HURTS because:
  • High-|Ψ|² points are in the bulk, far from nodes
  • E_L there is mostly determined by the Jastrow
  • Backflow only matters near the nodal surface, which has |Ψ|² → 0

FIX: Two collocation pools, same energy-variance loss:
  Pool 1 (BULK):      top-K by |Ψ|²  — trains the Jastrow (as before)
  Pool 2 (NEAR-NODE): band selection where |Ψ| is small-but-nonzero
                      — trains backflow to fix the nodal surface

Near-node points have high E_L variance because E_L ~ ∇²Ψ/Ψ diverges
at wrong nodes.  The energy-variance loss gradient at these points tells
backflow WHICH DIRECTION to shift the node.  This is information that
bulk points simply don't contain.

Additionally, same-spin close-pair configs are injected directly into
the near-node pool (bypassing |Ψ|² screening) because the antisymmetric
SD forces Ψ → 0 at same-spin coalescence — exactly the pair-coalescence
cusp that backflow modifies.
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
from PINN import PINN, CTNNBackflowNet

# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════


def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
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
    E_DMC = config.DMC_ENERGIES.get(N, {}).get(config._snap_omega(omega), None)
    params = config.get().as_dict()
    params["device"] = device
    params["torch_dtype"] = dtype
    params["E"] = E_DMC
    occ_str = ", ".join(f"({energies[k][1]},{energies[k][2]})" for k in range(n_occ))
    print(f"N={N}, ω={omega}, basis={nx}×{ny}={n_basis}")
    print(f"  Occupied: {occ_str}   E_DMC={E_DMC}")
    return C_occ, params


def compute_local_energy(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap_log.view(B) + g2.view(B))
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).view(B)
    return T + V_harm + V_int


def _huber(resid, delta):
    a = resid.abs()
    return torch.where(a <= delta, 0.5 * resid**2, delta * (a - 0.5 * delta))


# ══════════════════════════════════════════════════════════════════
#  Two-pool collocation:  BULK  +  NEAR-NODE
# ══════════════════════════════════════════════════════════════════


@torch.no_grad()
def two_pool_collocation(
    psi_log_fn,
    N,
    d,
    sigma,
    device,
    dtype,
    *,
    n_bulk,  # how many bulk (high |Ψ|²) points
    n_node,  # how many near-node points
    oversampling=10,
    # near-node band: select by percentile of log|Ψ| among candidates
    node_lo_pct=0.05,  # ignore extremely small |Ψ| (numerical junk)
    node_hi_pct=0.30,  # upper bound of "near-node" band
    # same-spin close-pair injection
    close_pair_n=0,  # how many same-spin close-pair configs
    close_pair_sigma=0.3,  # σ for the inter-electron displacement
    batch_size=4096,
):
    """
    Returns (X_bulk, X_node):
      X_bulk:  (n_bulk, N, d)  — high |Ψ|² region (Jastrow-relevant)
      X_node:  (n_node, N, d)  — near-node region (backflow-relevant)
    """
    n_cand = oversampling * (n_bulk + n_node)
    x_cand = torch.randn(n_cand, N, d, device=device, dtype=dtype) * sigma

    # Evaluate log|Ψ| at all candidates  (forward only)
    log_psi_parts = []
    for i in range(0, n_cand, batch_size):
        lp = psi_log_fn(x_cand[i : i + batch_size])
        log_psi_parts.append(lp)
    log_psi = torch.cat(log_psi_parts)  # (n_cand,)

    # Also compute log q(x) for the Gaussian proposal, for IS ratio
    Nd = N * d
    log_q = -0.5 * Nd * math.log(2 * math.pi * sigma**2) - x_cand.reshape(n_cand, -1).pow(2).sum(
        -1
    ) / (2 * sigma**2)

    # ── BULK pool: top by |Ψ|²/q  ──
    log_ratio = 2.0 * log_psi - log_q
    _, bulk_idx = torch.topk(log_ratio, n_bulk)
    X_bulk = x_cand[bulk_idx].clone()

    # ── NEAR-NODE pool: band of small |Ψ| ──
    # Sort log|Ψ| and select the band [lo_pct, hi_pct]
    sorted_lp, sorted_idx = torch.sort(log_psi)
    n_lo = int(node_lo_pct * n_cand)
    n_hi = int(node_hi_pct * n_cand)
    band_idx = sorted_idx[n_lo:n_hi]  # indices into x_cand

    if band_idx.numel() >= n_node:
        # Randomly sample n_node from the band
        perm = torch.randperm(band_idx.numel())[:n_node]
        node_idx = band_idx[perm]
    else:
        # Not enough in band — take what we have, pad with randn
        node_idx = band_idx
        n_pad = n_node - node_idx.numel()
        x_pad = torch.randn(n_pad, N, d, device=device, dtype=dtype) * sigma * 0.5
        X_node_pad = x_pad

    X_node = x_cand[node_idx].clone()
    if band_idx.numel() < n_node:
        X_node = torch.cat([X_node, X_node_pad], dim=0)

    # ── Same-spin close-pair injection ──
    if close_pair_n > 0:
        up = N // 2
        x_cp = torch.randn(close_pair_n, N, d, device=device, dtype=dtype) * sigma
        batch_range = torch.arange(close_pair_n)

        # Half: two same-spin-up electrons close
        n_half = close_pair_n // 2
        if up >= 2:
            i_up = torch.zeros(n_half, dtype=torch.long)  # electron 0
            j_up = torch.ones(n_half, dtype=torch.long)  # electron 1
            offset = torch.randn(n_half, d, device=device, dtype=dtype) * close_pair_sigma
            x_cp[:n_half, 1] = x_cp[:n_half, 0] + offset

        # Other half: two same-spin-down electrons close
        n_rest = close_pair_n - n_half
        down_start = up
        if N - up >= 2:
            offset2 = torch.randn(n_rest, d, device=device, dtype=dtype) * close_pair_sigma
            x_cp[n_half:, down_start + 1] = x_cp[n_half:, down_start] + offset2

        # Append to near-node pool (these bypass |Ψ|² screening)
        X_node = torch.cat([X_node[: n_node - close_pair_n], x_cp], dim=0)

    return X_bulk, X_node


# ══════════════════════════════════════════════════════════════════
#  Two-stream residual training
# ══════════════════════════════════════════════════════════════════


def train_two_stream(
    f_net,
    C_occ,
    params,
    *,
    backflow_net,
    # schedule
    n_epochs=300,
    lr=3e-4,
    lr_min_frac=0.02,
    phase1_frac=0.25,
    alpha_end=0.60,
    # collocation
    n_bulk=1536,  # bulk pool size
    n_node=512,  # near-node pool size
    oversampling=10,
    proposal_sigma_factor=1.3,
    micro_batch=256,
    # near-node settings
    node_lo_pct=0.05,
    node_hi_pct=0.30,
    close_pair_n=128,  # same-spin close-pair injections
    close_pair_sigma_ell=0.25,
    # loss weights
    node_loss_weight=0.5,  # relative weight of near-node loss
    node_huber_delta=5.0,  # Huber δ for near-node E_L (prevents blowup)
    # robustness
    grad_clip=0.5,
    quantile_trim=0.02,
    print_every=10,
):
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    E_DMC = params.get("E", None)
    ell = 1.0 / math.sqrt(omega)
    sigma = proposal_sigma_factor * ell
    cp_sigma = close_pair_sigma_ell * ell

    f_net.to(device).to(dtype)
    backflow_net.to(device).to(dtype)

    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]).to(
        device
    )

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return lp

    all_params = list(f_net.parameters()) + list(backflow_net.parameters())
    n_p = sum(p.numel() for p in all_params)
    n_bf = sum(p.numel() for p in backflow_net.parameters())

    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_min = lr * lr_min_frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (
            lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))
        )
        / lr,
    )

    phase1_end = int(phase1_frac * n_epochs)

    print(f"\n{'='*60}")
    print("Two-stream residual training")
    print(f"  {n_epochs} ep, bulk={n_bulk} + node={n_node} coll pts (×{oversampling})")
    print(f"  {n_p:,} total params ({n_bf:,} backflow)")
    print(f"  lr={lr}, cosine → {lr_min:.1e}")
    print(f"  proposal σ={sigma:.3f}  (ℓ={ell:.3f})")
    print(
        f"  near-node: band [{node_lo_pct:.0%}, {node_hi_pct:.0%}], "
        f"{close_pair_n} close-pairs (σ_cp={cp_sigma:.3f})"
    )
    print(f"  node_loss_weight={node_loss_weight}, huber_δ={node_huber_delta}")
    print(f"  phase 1 (var-min): 0–{phase1_end}")
    print(f"  phase 2 (targeting): α 0→{alpha_end}, {phase1_end}–{n_epochs}")
    print(f"  E_DMC = {E_DMC}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    history = []
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    patience = 60

    for epoch in range(n_epochs):
        # ── Alpha schedule ──
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # ── Two-pool collocation ──
        f_net.eval()
        backflow_net.eval()
        X_bulk, X_node = two_pool_collocation(
            psi_log_fn,
            N,
            d,
            sigma,
            device,
            dtype,
            n_bulk=n_bulk,
            n_node=n_node,
            oversampling=oversampling,
            node_lo_pct=node_lo_pct,
            node_hi_pct=node_hi_pct,
            close_pair_n=close_pair_n,
            close_pair_sigma=cp_sigma,
        )

        # ── Compute losses ──
        f_net.train()
        backflow_net.train()
        optimizer.zero_grad(set_to_none=True)

        # ── STREAM 1: Bulk (standard energy variance) ──
        all_EL_bulk = []
        n_b1 = max(1, math.ceil(n_bulk / micro_batch))
        for i in range(0, n_bulk, micro_batch):
            x_mb = X_bulk[i : i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega).view(-1)

            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue
            if quantile_trim > 0 and E_L.numel() > 20:
                lo = torch.quantile(E_L.detach(), quantile_trim)
                hi = torch.quantile(E_L.detach(), 1.0 - quantile_trim)
                mask = (E_L.detach() >= lo) & (E_L.detach() <= hi)
                E_L = E_L[mask]
                if E_L.numel() == 0:
                    continue

            all_EL_bulk.append(E_L.detach())
            mu = E_L.mean().detach()
            E_eff = alpha * float(E_DMC) + (1.0 - alpha) * mu
            resid = E_L - E_eff
            loss_bulk = (resid**2).mean()
            (loss_bulk / n_b1).backward()

        # ── STREAM 2: Near-node (Huber loss, targets backflow) ──
        all_EL_node = []
        n_b2 = max(1, math.ceil(n_node / micro_batch))
        for i in range(0, n_node, micro_batch):
            x_mb = X_node[i : i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega).view(-1)

            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue

            # Clip extreme E_L values (near nodes E_L can blow up)
            E_L_clamp = E_L.clamp(-100.0, 100.0)

            all_EL_node.append(E_L_clamp.detach())

            mu = E_L_clamp.mean().detach()
            E_eff = alpha * float(E_DMC) + (1.0 - alpha) * mu
            resid = E_L_clamp - E_eff
            loss_node = _huber(resid, node_huber_delta).mean()
            (node_loss_weight * loss_node / n_b2).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Logging ──
        if len(all_EL_bulk) > 0:
            EL_bulk = torch.cat(all_EL_bulk)
            E_mean = EL_bulk.mean().item()
            E_var = EL_bulk.var().item()
            E_std = EL_bulk.std().item()
        else:
            E_mean, E_var, E_std = float("nan"), float("nan"), float("nan")

        if len(all_EL_node) > 0:
            EL_node = torch.cat(all_EL_node)
            node_var = EL_node.var().item()
            node_mean = EL_node.mean().item()
        else:
            node_var, node_mean = float("nan"), float("nan")

        history.append(
            {
                "epoch": epoch,
                "E_mean": E_mean,
                "E_var": E_var,
                "node_var": node_var,
                "node_mean": node_mean,
            }
        )

        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_f_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            best_bf_state = {k: v.clone() for k, v in backflow_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience and epoch > phase1_end + 30:
            print(f"  Early stop at epoch {epoch}  (best var={best_var:.3e})")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / abs(E_DMC) * 100 if E_DMC else 0
            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  node_var={node_var:.2e}  "
                f"α={alpha:.3f}  lr={cur_lr:.1e}  t={dt:.0f}s  err={err:.2f}%"
            )
            sys.stdout.flush()

    if best_f_state:
        f_net.load_state_dict(best_f_state)
        backflow_net.load_state_dict(best_bf_state)
        print(f"Restored best model (var={best_var:.3e})")

    total = time.time() - t0
    print(f"Training done in {total:.0f}s ({total/60:.1f}min)")
    return f_net, backflow_net, history


# ══════════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════════


def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    print(f"\n── VMC eval: {label} ──")
    result = evaluate_energy_vmc(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=backflow_net,
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
    E, E_std = result["E_mean"], result["E_stderr"]
    E_ref = params.get("E")
    err = abs(E - E_ref) / abs(E_ref) * 100 if E_ref else 0
    print(f"  E = {E:.6f} ± {E_std:.6f}  (target {E_ref}, err {err:.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════════
#  Network factory
# ══════════════════════════════════════════════════════════════════


def make_nets(device="cpu", dtype=torch.float64):
    f_net = (
        PINN(
            n_particles=6,
            d=2,
            omega=0.5,
            dL=8,
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
    bf_net = (
        CTNNBackflowNet(
            d=2,
            msg_hidden=32,
            msg_layers=1,
            hidden=32,
            layers=2,
            act="silu",
            aggregation="sum",
            use_spin=True,
            same_spin_only=False,
            out_bound="tanh",
            bf_scale_init=0.05,
            omega=0.5,
        )
        .to(device)
        .to(dtype)
    )
    np_total = sum(p.numel() for p in f_net.parameters()) + sum(
        p.numel() for p in bf_net.parameters()
    )
    print(f"CTNN+PINN params: {np_total:,}")
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  Runs
# ══════════════════════════════════════════════════════════════════

COMMON = dict(
    n_epochs=300,
    lr=3e-4,
    n_bulk=1536,
    n_node=512,
    oversampling=10,
    micro_batch=256,
    grad_clip=0.5,
    print_every=10,
    phase1_frac=0.25,
    alpha_end=0.60,
    proposal_sigma_factor=1.3,
    close_pair_n=128,
    close_pair_sigma_ell=0.25,
    node_lo_pct=0.05,
    node_hi_pct=0.30,
)


def run_twostream():
    """Two-stream with default node_loss_weight=0.5."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net, bf_net = make_nets(device, dtype)
    f_net, bf_net, _ = train_two_stream(
        f_net,
        C_occ,
        params,
        backflow_net=bf_net,
        node_loss_weight=0.5,
        node_huber_delta=5.0,
        **COMMON,
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="Two-stream  λ_node=0.5  δ=5")


def run_twostream_heavy():
    """Two-stream with stronger near-node emphasis."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net, bf_net = make_nets(device, dtype)
    f_net, bf_net, _ = train_two_stream(
        f_net,
        C_occ,
        params,
        backflow_net=bf_net,
        node_loss_weight=1.0,  # equal weight to node and bulk
        node_huber_delta=3.0,  # tighter Huber → stronger signal
        n_node=768,  # more near-node points
        n_bulk=1280,
        close_pair_n=256,  # more close-pairs
        close_pair_sigma_ell=0.20,  # tighter coalescence
        node_hi_pct=0.35,  # wider near-node band
        n_epochs=300,
        lr=3e-4,
        oversampling=10,
        micro_batch=256,
        grad_clip=0.5,
        print_every=10,
        phase1_frac=0.25,
        alpha_end=0.60,
        proposal_sigma_factor=1.3,
    )
    return evaluate(
        f_net, C_occ, params, backflow_net=bf_net, label="Two-stream  λ_node=1.0  heavy"
    )


def run_twostream_wider_band():
    """Two-stream with wider near-node band and more close-pairs."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net, bf_net = make_nets(device, dtype)
    f_net, bf_net, _ = train_two_stream(
        f_net,
        C_occ,
        params,
        backflow_net=bf_net,
        node_loss_weight=0.5,
        node_huber_delta=5.0,
        node_lo_pct=0.02,  # include points closer to node
        node_hi_pct=0.40,  # wider band
        close_pair_n=192,
        close_pair_sigma_ell=0.15,  # very tight coalescence
        n_node=640,
        n_bulk=1408,
        n_epochs=300,
        lr=3e-4,
        oversampling=10,
        micro_batch=256,
        grad_clip=0.5,
        print_every=10,
        phase1_frac=0.25,
        alpha_end=0.60,
        proposal_sigma_factor=1.3,
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="Two-stream  wide-band")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = {}

    for name, fn in [
        ("twostream", run_twostream),
        ("heavy", run_twostream_heavy),
        ("wide_band", run_twostream_wider_band),
    ]:
        print(f"\n{'#'*60}")
        print(f"# {name.upper()}")
        print(f"{'#'*60}")
        results[name] = fn()

    target = 11.78484
    print(f"\n{'='*60}")
    print("SUMMARY — 6e ω=0.5  Two-stream collocation")
    print(f"{'='*60}")
    for name, r in results.items():
        E, se = r["E_mean"], r["E_stderr"]
        err = abs(E - target) / target * 100
        print(f"  {name:20s}  E={E:.6f} ± {se:.6f}  err={err:.2f}%")
    print(f"  {'PINN only (ref)':20s}  E=11.834691 ± 0.002782  err=0.42%")
    print(f"  {'CTNN base (ref)':20s}  E=11.848922 ± 0.002811  err=0.54%")
    print(f"  {'DMC target':20s}  E={target:.6f}")
    print(f"{'='*60}")
