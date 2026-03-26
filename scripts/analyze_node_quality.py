#!/usr/bin/env python3
"""
Node quality analysis
=====================
Measure how "intelligent" the nodal surface is by comparing
the kinetic energy density |∇log Ψ|² near the node for:
  - Bare Slater determinant
  - SD + Jastrow
  - SD + Backflow + Jastrow (full ansatz)

For the exact eigenstate, E_L = E₀ everywhere, so |∇log Ψ|²
is well-behaved even at the node. The divergence rate of
|∇log Ψ|² as |D| → 0 directly measures how approximate the
nodal surface is.

Usage:
  CUDA_MANUAL_DEVICE=4 python3 scripts/analyze_node_quality.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import config
from functions.Slater_Determinant import slater_determinant_closed_shell
from functions.Physics import compute_coulomb_interaction
from PINN import CTNNBackflowNet
from jastrow_architectures import CTNNJastrowVCycle

# ===================== Config =====================
N_ELEC = 6
OMEGA = 1.0
DIM = 2
DEVICE = torch.device(f"cuda:{os.environ.get('CUDA_MANUAL_DEVICE', 4)}")
DTYPE = torch.float64
CHECKPOINT = REPO / "results" / "arch_colloc" / "bf_ctnn_vcycle.pt"
OUT_DIR = REPO / "outputs" / "coalescence_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 4096
BURN_IN = 800
GRAD_BATCH = 128  # batch size for gradient computation (memory-limited)


def setup():
    n_occ = N_ELEC // 2
    nx = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
    ny = nx
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(
        n_particles=N_ELEC, omega=OMEGA, d=DIM,
        basis="cart", nx=nx, ny=ny, L=L, n_grid=80,
        device=str(DEVICE), dtype="float64", seed=42,
    )
    energies = sorted(
        [(OMEGA * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)]
    )
    C = np.zeros((nx * ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    return torch.tensor(C, dtype=DTYPE, device=DEVICE)


def load_models():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    bfc = ckpt["bf_config"]
    bf_net = CTNNBackflowNet(
        d=bfc["d"], msg_hidden=bfc["msg_hidden"], msg_layers=bfc["msg_layers"],
        hidden=bfc["hidden"], layers=bfc["layers"], act=bfc["act"],
        aggregation=bfc["aggregation"], use_spin=bfc["use_spin"],
        same_spin_only=bfc["same_spin_only"], out_bound=bfc["out_bound"],
        bf_scale_init=bfc["bf_scale_init"], zero_init_last=bfc["zero_init_last"],
        omega=OMEGA,
    ).to(DEVICE).to(DTYPE)
    bf_net.load_state_dict(ckpt["bf_state"])
    bf_net.eval()

    jas_net = CTNNJastrowVCycle(
        n_particles=N_ELEC, d=DIM, omega=OMEGA,
        node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
        n_down=1, n_up=1, msg_layers=1, node_layers=1,
        readout_hidden=64, readout_layers=2, act="silu",
    ).to(DEVICE).to(DTYPE)
    jas_net.load_state_dict(ckpt["jas_state"])
    jas_net.eval()

    return bf_net, jas_net


def make_spin():
    n_up = N_ELEC // 2
    return torch.cat([
        torch.zeros(n_up, dtype=torch.long),
        torch.ones(N_ELEC - n_up, dtype=torch.long),
    ]).to(DEVICE)


def _params():
    p = config.get().as_dict()
    p["device"] = str(DEVICE)
    p["torch_dtype"] = DTYPE
    return p


def mcmc_sample(C_occ, spin, bf_net, jas_net, n_samples, burn_in):
    """MCMC from |Ψ_full|²."""
    ell = 1.0 / math.sqrt(OMEGA)
    x = torch.randn(n_samples, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * ell
    step = 0.15 * ell
    spin_bn = spin.unsqueeze(0).expand(n_samples, -1)
    params = _params()

    with torch.no_grad():
        dx = bf_net(x, spin=spin_bn)
        _, la = slater_determinant_closed_shell(
            x_config=x + dx, C_occ=C_occ, params=params, spin=spin, normalize=True
        )
        j = jas_net(x, spin=spin_bn).squeeze(-1)
        lp = 2.0 * (la + j)

        n_acc = 0
        for step_i in range(burn_in):
            prop = x + torch.randn_like(x) * step
            dx_p = bf_net(prop, spin=spin_bn)
            _, la_p = slater_determinant_closed_shell(
                x_config=prop + dx_p, C_occ=C_occ, params=params, spin=spin,
                normalize=True,
            )
            j_p = jas_net(prop, spin=spin_bn).squeeze(-1)
            lp_p = 2.0 * (la_p + j_p)
            acc = torch.rand_like(lp).log() < (lp_p - lp)
            x = torch.where(acc.view(-1, 1, 1), prop, x)
            lp = torch.where(acc, lp_p, lp)
            n_acc += acc.sum().item()

        acc_rate = n_acc / (burn_in * n_samples)
        print(f"  MCMC acceptance rate: {acc_rate:.3f}")

    return x.detach()


def compute_grad_log_psi_batched(x_all, C_occ, spin, bf_net, jas_net, mode="full"):
    """
    Compute |∇_r log|Ψ||² for each sample.

    mode:
      "sd"   — bare Slater determinant D(r)
      "sd_j" — D(r) × exp(J(r))
      "full" — D(r + Δr(r)) × exp(J(r))

    Returns: grad_sq (N_samples,), log_D_bare (N_samples,), log_D_bf (N_samples,)
    """
    N = x_all.shape[0]
    params = _params()
    spin_bn = spin.unsqueeze(0)

    grad_sq_all = []
    logD_bare_all = []
    logD_bf_all = []

    for i in range(0, N, GRAD_BATCH):
        e = min(i + GRAD_BATCH, N)
        xb = x_all[i:e].detach().requires_grad_(True)
        B = xb.shape[0]
        sb = spin_bn.expand(B, -1)

        if mode == "sd":
            _, logabs = slater_determinant_closed_shell(
                x_config=xb, C_occ=C_occ, params=params, spin=spin, normalize=True,
            )
            log_psi = logabs
        elif mode == "sd_j":
            _, logabs = slater_determinant_closed_shell(
                x_config=xb, C_occ=C_occ, params=params, spin=spin, normalize=True,
            )
            j = jas_net(xb, spin=sb).squeeze(-1)
            log_psi = logabs + j
        elif mode == "full":
            dx = bf_net(xb, spin=sb)
            x_eff = xb + dx
            _, logabs_bf = slater_determinant_closed_shell(
                x_config=x_eff, C_occ=C_occ, params=params, spin=spin, normalize=True,
            )
            j = jas_net(xb, spin=sb).squeeze(-1)
            log_psi = logabs_bf + j
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Gradient of log|Ψ| w.r.t. electron coordinates
        grad = torch.autograd.grad(
            log_psi.sum(), xb, create_graph=False, retain_graph=False
        )[0]  # (B, N_elec, d)

        # |∇log|Ψ||² summed over all electrons and dimensions
        g_sq = (grad ** 2).sum(dim=(-1, -2))  # (B,)
        grad_sq_all.append(g_sq.detach())

        # Also compute log|D_bare| and log|D_bf| for binning (no grad needed)
        with torch.no_grad():
            _, la_bare = slater_determinant_closed_shell(
                x_config=xb.detach(), C_occ=C_occ, params=params, spin=spin,
                normalize=True,
            )
            logD_bare_all.append(la_bare)

            dx_bf = bf_net(xb.detach(), spin=sb)
            _, la_bf = slater_determinant_closed_shell(
                x_config=xb.detach() + dx_bf, C_occ=C_occ, params=params, spin=spin,
                normalize=True,
            )
            logD_bf_all.append(la_bf)

    return (
        torch.cat(grad_sq_all).cpu().numpy(),
        torch.cat(logD_bare_all).cpu().numpy(),
        torch.cat(logD_bf_all).cpu().numpy(),
    )


def compute_rmin_same_spin(x, spin):
    """Minimum same-spin pair distance per sample."""
    x_np = x.cpu().numpy()
    spin_np = spin.cpu().numpy()
    N_s = x_np.shape[0]
    rmin = np.full(N_s, np.inf)
    for i in range(N_ELEC):
        for j in range(i + 1, N_ELEC):
            if spin_np[i] == spin_np[j]:
                d = np.linalg.norm(x_np[:, i] - x_np[:, j], axis=-1)
                rmin = np.minimum(rmin, d)
    return rmin


def compute_potential(x):
    """V = ½ω²r² + Coulomb."""
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    N_s = x_np.shape[0]
    V = np.zeros(N_s)
    # Harmonic
    for i in range(N_ELEC):
        V += 0.5 * OMEGA * np.sum(x_np[:, i] ** 2, axis=-1)
    # Coulomb
    for i in range(N_ELEC):
        for j in range(i + 1, N_ELEC):
            rij = np.linalg.norm(x_np[:, i] - x_np[:, j], axis=-1)
            V += 1.0 / np.maximum(rij, 1e-12)
    return V


def binned_stats(x_vals, y_vals, n_bins=30, percentile_range=(1, 99)):
    """Compute median and IQR of y in bins of x."""
    lo, hi = np.percentile(x_vals, percentile_range)
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(n_bins, np.nan)
    q25 = np.full(n_bins, np.nan)
    q75 = np.full(n_bins, np.nan)
    means = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask = (x_vals >= edges[k]) & (x_vals < edges[k + 1])
        if mask.sum() > 5:
            vals = y_vals[mask]
            medians[k] = np.median(vals)
            q25[k] = np.percentile(vals, 25)
            q75[k] = np.percentile(vals, 75)
            means[k] = np.mean(vals)
    return centers, medians, q25, q75, means


# ===================== MAIN =====================
def main():
    print("=" * 60)
    print("NODE QUALITY ANALYSIS")
    print("=" * 60)

    C_occ = setup()
    bf_net, jas_net = load_models()
    spin = make_spin()

    # --- MCMC sampling from |Ψ_full|² ---
    print(f"\nMCMC sampling {N_SAMPLES} configs (burn-in {BURN_IN})...")
    x_samples = mcmc_sample(C_occ, spin, bf_net, jas_net, N_SAMPLES, BURN_IN)

    # --- Compute |∇logΨ|² for each ansatz ---
    print("\nComputing |∇logΨ|² for bare SD...")
    gsq_sd, logD_bare, logD_bf = compute_grad_log_psi_batched(
        x_samples, C_occ, spin, bf_net, jas_net, mode="sd"
    )
    print("Computing |∇logΨ|² for SD + Jastrow...")
    gsq_sdj, _, _ = compute_grad_log_psi_batched(
        x_samples, C_occ, spin, bf_net, jas_net, mode="sd_j"
    )
    print("Computing |∇logΨ|² for full BF + Jastrow...")
    gsq_full, _, _ = compute_grad_log_psi_batched(
        x_samples, C_occ, spin, bf_net, jas_net, mode="full"
    )

    # --- Auxiliary quantities ---
    print("Computing r_min (same-spin) and V...")
    rmin = compute_rmin_same_spin(x_samples, spin)
    V = compute_potential(x_samples)

    # Weak-form energy density: e = ½|∇logΨ|² + V
    e_sd = 0.5 * gsq_sd + V
    e_sdj = 0.5 * gsq_sdj + V
    e_full = 0.5 * gsq_full + V

    # --- Summary statistics ---
    print("\n=== Summary ===")
    print(f"{'Ansatz':<20} {'mean |∇logΨ|²':>15} {'median':>10} {'mean e_WF':>12}")
    for name, gsq, e in [("Bare SD", gsq_sd, e_sd),
                          ("SD + Jastrow", gsq_sdj, e_sdj),
                          ("Full BF+Jas", gsq_full, e_full)]:
        print(f"{name:<20} {np.mean(gsq):>15.2f} {np.median(gsq):>10.2f} {np.mean(e):>12.4f}")

    print(f"\nDMC reference energy: {config.get().E:.5f}")

    # ===================== PLOTTING =====================
    print("\nPlotting...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    colors = {"sd": "#2166ac", "sdj": "#4dac26", "full": "#b2182b"}
    labels = {"sd": "Bare SD", "sdj": "SD + Jastrow", "full": "Full (BF+Jas)"}

    # --- (a) |∇logΨ|² vs log|D_bare|: binned medians ---
    ax = axes[0, 0]
    for key, gsq in [("sd", gsq_sd), ("sdj", gsq_sdj), ("full", gsq_full)]:
        c, med, q25, q75, _ = binned_stats(logD_bare, gsq, n_bins=35)
        valid = ~np.isnan(med)
        ax.plot(c[valid], med[valid], "-o", color=colors[key], lw=2, ms=4,
                label=labels[key])
        ax.fill_between(c[valid], q25[valid], q75[valid],
                        color=colors[key], alpha=0.15)
    ax.set_xlabel("$\\ln|D_{bare}|$", fontsize=13)
    ax.set_ylabel("$|\\nabla \\ln|\\Psi||^2$", fontsize=13)
    ax.set_title("Kinetic energy density vs bare SD amplitude", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # --- (b) Same but vs log|D_bf| (BF-shifted SD amplitude) ---
    ax = axes[0, 1]
    for key, gsq in [("sd", gsq_sd), ("sdj", gsq_sdj), ("full", gsq_full)]:
        c, med, q25, q75, _ = binned_stats(logD_bf, gsq, n_bins=35)
        valid = ~np.isnan(med)
        ax.plot(c[valid], med[valid], "-o", color=colors[key], lw=2, ms=4,
                label=labels[key])
        ax.fill_between(c[valid], q25[valid], q75[valid],
                        color=colors[key], alpha=0.15)
    ax.set_xlabel("$\\ln|D_{BF}|$", fontsize=13)
    ax.set_ylabel("$|\\nabla \\ln|\\Psi||^2$", fontsize=13)
    ax.set_title("Same, but binned by BF-shifted SD amplitude", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # --- (c) |∇logΨ|² vs r_min (same-spin pair distance) ---
    ax = axes[0, 2]
    for key, gsq in [("sd", gsq_sd), ("sdj", gsq_sdj), ("full", gsq_full)]:
        c, med, q25, q75, _ = binned_stats(rmin, gsq, n_bins=35,
                                            percentile_range=(0, 99))
        valid = ~np.isnan(med)
        ax.plot(c[valid], med[valid], "-o", color=colors[key], lw=2, ms=4,
                label=labels[key])
        ax.fill_between(c[valid], q25[valid], q75[valid],
                        color=colors[key], alpha=0.15)
    ax.set_xlabel("$r_{\\min}^{\\uparrow\\uparrow}$ (min same-spin distance)", fontsize=13)
    ax.set_ylabel("$|\\nabla \\ln|\\Psi||^2$", fontsize=13)
    ax.set_title("Kinetic energy density vs same-spin proximity", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # --- (d) Weak-form energy density distribution ---
    ax = axes[1, 0]
    # Clip extreme tails for readability
    clip_lo, clip_hi = np.percentile(np.concatenate([e_sd, e_sdj, e_full]), [1, 99])
    bins = np.linspace(clip_lo, clip_hi, 80)
    for key, e_vals in [("sd", e_sd), ("sdj", e_sdj), ("full", e_full)]:
        ax.hist(e_vals, bins=bins, density=True, alpha=0.4, color=colors[key],
                label=f"{labels[key]}: mean={np.mean(e_vals):.2f}", edgecolor="none")
    ax.axvline(x=config.get().E, color="black", ls="--", lw=2, label=f"DMC = {config.get().E:.3f}")
    ax.set_xlabel("$e_{WF} = \\frac{1}{2}|\\nabla \\ln\\Psi|^2 + V$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Weak-form energy density distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # --- (e) Cumulative variance contribution ---
    ax = axes[1, 1]
    for key, e_vals in [("sd", e_sd), ("sdj", e_sdj), ("full", e_full)]:
        # Sort by |D_bare| (ascending = near node first)
        sort_idx = np.argsort(logD_bare)
        e_sorted = e_vals[sort_idx]
        mean_e = np.mean(e_vals)
        cumvar = np.cumsum((e_sorted - mean_e) ** 2) / len(e_vals)
        total_var = cumvar[-1]
        frac = np.arange(1, len(e_vals) + 1) / len(e_vals) * 100
        ax.plot(frac, cumvar / max(total_var, 1e-30) * 100,
                color=colors[key], lw=2, label=labels[key])
    ax.set_xlabel("Samples included (sorted by $\\ln|D_{bare}|$ ascending) [%]", fontsize=12)
    ax.set_ylabel("Cumulative variance [%]", fontsize=13)
    ax.set_title("Variance contribution: near-node samples dominate", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.axhline(y=50, color="gray", ls=":", alpha=0.5)

    # --- (f) log|D_bare| vs log|D_bf| scatter ---
    ax = axes[1, 2]
    sc = ax.scatter(logD_bare, logD_bf, c=gsq_full, cmap="hot_r",
                    s=3, alpha=0.4, norm=matplotlib.colors.LogNorm(
                        vmin=np.percentile(gsq_full, 5),
                        vmax=np.percentile(gsq_full, 95),
                    ))
    plt.colorbar(sc, ax=ax, label="$|\\nabla \\ln|\\Psi_{full}||^2$", shrink=0.8)
    # Diagonal
    lo = min(logD_bare.min(), logD_bf.min())
    hi = max(logD_bare.max(), logD_bf.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("$\\ln|D_{bare}(r)|$", fontsize=13)
    ax.set_ylabel("$\\ln|D_{BF}(r+\\Delta r)|$", fontsize=13)
    ax.set_title("BF amplifies SD: points above diagonal", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    # Report mean shift
    mean_shift = np.mean(logD_bf - logD_bare)
    ax.text(0.05, 0.92, f"Mean $\\Delta\\ln|D|$ = {mean_shift:.3f}\n"
            f"= {np.exp(mean_shift):.2f}× amplification",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle(
        f"Node quality analysis: N={N_ELEC}, ω={OMEGA}\n"
        f"Samples from $|\\Psi_{{full}}|^2$ (MCMC, {N_SAMPLES} configs)",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    path = OUT_DIR / "fig8_node_quality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
