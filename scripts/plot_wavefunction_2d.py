#!/usr/bin/env python3
"""
2D Wavefunction Visualization
==============================
Fix all electrons except one, scan that electron on a 2D grid.
Plot |Ψ|² and sign(Ψ) for:
  - Bare Slater determinant
  - SD × Jastrow
  - SD(x+Δx) × Jastrow  (full backflow + Jastrow)

Also: one-body density and pair correlation from MCMC.

Usage:
  CUDA_MANUAL_DEVICE=4 python3 scripts/plot_wavefunction_2d.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import config
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import slater_determinant_closed_shell
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

GRID_N = 200  # grid resolution per axis
GRID_EXTENT = 3.5  # physical extent in each direction (units of 1/√ω)


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

    print(f"BF: {sum(p.numel() for p in bf_net.parameters())} params, "
          f"out_bound={bfc['out_bound']}")
    print(f"Jas: {sum(p.numel() for p in jas_net.parameters())} params")
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


def find_equilibrium_config(C_occ, spin, bf_net, jas_net, n_pool=4096, burn_in=500):
    """MCMC sample and pick a 'typical' configuration (near median |SD|)."""
    ell = 1.0 / math.sqrt(OMEGA)
    x = torch.randn(n_pool, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * ell
    step = 0.15 * ell
    spin_bn = spin.unsqueeze(0).expand(n_pool, -1)

    with torch.no_grad():
        dx = bf_net(x, spin=spin_bn)
        x_eff = x + dx
        sign, logabs = slater_determinant_closed_shell(
            x_config=x_eff, C_occ=C_occ, params=_params(), spin=spin, normalize=True
        )
        j = jas_net(x, spin=spin_bn).squeeze(-1)
        lp = 2.0 * (logabs + j)

        for _ in range(burn_in):
            prop = x + torch.randn_like(x) * step
            dx_p = bf_net(prop, spin=spin_bn)
            x_eff_p = prop + dx_p
            sign_p, logabs_p = slater_determinant_closed_shell(
                x_config=x_eff_p, C_occ=C_occ, params=_params(), spin=spin,
                normalize=True,
            )
            j_p = jas_net(prop, spin=spin_bn).squeeze(-1)
            lp_p = 2.0 * (logabs_p + j_p)
            acc = (torch.rand_like(lp).log() < (lp_p - lp))
            x = torch.where(acc.view(-1, 1, 1), prop, x)
            lp = torch.where(acc, lp_p, lp)

        # Compute bare |SD| and pick config near the 75th percentile
        # (not too close to node, not the max — something typical)
        _, logabs_bare = slater_determinant_closed_shell(
            x_config=x, C_occ=C_occ, params=_params(), spin=spin, normalize=True,
        )
        sd_vals = logabs_bare
        target = torch.quantile(sd_vals.float(), 0.75).item()
        idx = (sd_vals - target).abs().argmin().item()

    x0 = x[idx].clone()
    print(f"Equilibrium config: log|SD| = {sd_vals[idx].item():.2f}")
    print(f"  Electron positions:")
    for i in range(N_ELEC):
        s = "↑" if spin[i] == 0 else "↓"
        print(f"    e{i} ({s}): ({x0[i,0].item():.3f}, {x0[i,1].item():.3f})")
    return x0


def eval_on_grid(x_fixed, scan_idx, C_occ, spin, bf_net, jas_net,
                 grid_n=GRID_N, extent=GRID_EXTENT):
    """
    Fix all electrons except scan_idx, scan that electron on a 2D grid.
    Returns grid coordinates and three 2D arrays:
      psi2_sd, psi2_sdj, psi2_full (all shape grid_n × grid_n)
    and:
      sign_sd, sign_full (sign of the wavefunction)
    """
    ell = 1.0 / math.sqrt(OMEGA)
    lim = extent * ell
    gx = torch.linspace(-lim, lim, grid_n, device=DEVICE, dtype=DTYPE)
    gy = torch.linspace(-lim, lim, grid_n, device=DEVICE, dtype=DTYPE)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    grid_pts = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=-1)  # (M, 2)
    M = grid_pts.shape[0]

    # Build full configs: (M, N, 2) with scan_idx replaced by grid point
    x_base = x_fixed.unsqueeze(0).expand(M, -1, -1).clone()  # (M, N, 2)
    x_base[:, scan_idx, :] = grid_pts

    params = _params()
    spin_bn = spin.unsqueeze(0).expand(M, -1)

    # Process in batches to avoid OOM
    batch_sz = 2048
    sd_sign = torch.zeros(M, device=DEVICE, dtype=DTYPE)
    sd_logabs = torch.zeros(M, device=DEVICE, dtype=DTYPE)
    jas_val = torch.zeros(M, device=DEVICE, dtype=DTYPE)
    bf_sign = torch.zeros(M, device=DEVICE, dtype=DTYPE)
    bf_logabs = torch.zeros(M, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        for i in range(0, M, batch_sz):
            e = min(i + batch_sz, M)
            xb = x_base[i:e]
            sb = spin_bn[i:e]
            Bb = xb.shape[0]

            # Bare SD
            s_i, la_i = slater_determinant_closed_shell(
                x_config=xb, C_occ=C_occ, params=params, spin=spin, normalize=True,
            )
            sd_sign[i:e] = s_i
            sd_logabs[i:e] = la_i

            # Jastrow
            j_i = jas_net(xb, spin=sb).squeeze(-1)
            jas_val[i:e] = j_i

            # BF + SD
            dx_i = bf_net(xb, spin=sb)
            x_eff_i = xb + dx_i
            s_bf, la_bf = slater_determinant_closed_shell(
                x_config=x_eff_i, C_occ=C_occ, params=params, spin=spin,
                normalize=True,
            )
            bf_sign[i:e] = s_bf
            bf_logabs[i:e] = la_bf

    # Compute |Ψ|² for each ansatz
    # 1. Bare SD: |Ψ|² = exp(2 * logabs)
    psi2_sd = torch.exp(2.0 * sd_logabs).cpu().numpy().reshape(grid_n, grid_n)
    sign_sd = sd_sign.cpu().numpy().reshape(grid_n, grid_n)

    # 2. SD + Jastrow: |Ψ|² = exp(2 * (logabs + J))
    psi2_sdj = torch.exp(2.0 * (sd_logabs + jas_val)).cpu().numpy().reshape(grid_n, grid_n)

    # 3. Full BF + Jastrow: |Ψ|² = exp(2 * (bf_logabs + J))
    psi2_full = torch.exp(2.0 * (bf_logabs + jas_val)).cpu().numpy().reshape(grid_n, grid_n)
    sign_full = bf_sign.cpu().numpy().reshape(grid_n, grid_n)

    # Also: the signed wavefunction (for node visualization)
    psi_sd = (sd_sign * torch.exp(sd_logabs)).cpu().numpy().reshape(grid_n, grid_n)
    psi_full = (bf_sign * torch.exp(bf_logabs + jas_val)).cpu().numpy().reshape(grid_n, grid_n)

    # BF displacement field for the scanned electron
    bf_dx_arr = torch.zeros(M, DIM, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        for i in range(0, M, batch_sz):
            e = min(i + batch_sz, M)
            xb = x_base[i:e]
            sb = spin_bn[i:e]
            dx_i = bf_net(xb, spin=sb)
            bf_dx_arr[i:e] = dx_i[:, scan_idx, :]  # displacement of scanned electron

    bf_dx_x = bf_dx_arr[:, 0].cpu().numpy().reshape(grid_n, grid_n)
    bf_dx_y = bf_dx_arr[:, 1].cpu().numpy().reshape(grid_n, grid_n)

    # logabs for ratio computation
    sd_logabs_np = sd_logabs.cpu().numpy().reshape(grid_n, grid_n)
    bf_logabs_np = bf_logabs.cpu().numpy().reshape(grid_n, grid_n)
    jas_val_np = jas_val.cpu().numpy().reshape(grid_n, grid_n)

    coords = gx.cpu().numpy()
    return coords, {
        "psi2_sd": psi2_sd,
        "psi2_sdj": psi2_sdj,
        "psi2_full": psi2_full,
        "psi_sd": psi_sd,
        "psi_full": psi_full,
        "sign_sd": sign_sd,
        "sign_full": sign_full,
        "bf_dx_x": bf_dx_x,
        "bf_dx_y": bf_dx_y,
        "sd_logabs": sd_logabs_np,
        "bf_logabs": bf_logabs_np,
        "jas_logabs": jas_val_np,
    }


def compute_onebody_density(C_occ, spin, bf_net, jas_net,
                            n_samples=16384, burn_in=500, mode="full"):
    """
    Estimate one-body density ρ(x,y) via histogramming MCMC samples.
    mode: 'sd' (bare SD), 'full' (BF+Jastrow)
    """
    ell = 1.0 / math.sqrt(OMEGA)
    x = torch.randn(n_samples, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * ell
    step = 0.12 * ell if mode == "sd" else 0.15 * ell
    spin_bn = spin.unsqueeze(0).expand(n_samples, -1)
    params = _params()

    with torch.no_grad():
        if mode == "sd":
            _, la = slater_determinant_closed_shell(
                x_config=x, C_occ=C_occ, params=params, spin=spin, normalize=True
            )
            lp = 2.0 * la
        else:
            dx = bf_net(x, spin=spin_bn)
            _, la = slater_determinant_closed_shell(
                x_config=x + dx, C_occ=C_occ, params=params, spin=spin, normalize=True
            )
            j = jas_net(x, spin=spin_bn).squeeze(-1)
            lp = 2.0 * (la + j)

        for _ in range(burn_in):
            prop = x + torch.randn_like(x) * step
            if mode == "sd":
                _, la_p = slater_determinant_closed_shell(
                    x_config=prop, C_occ=C_occ, params=params, spin=spin,
                    normalize=True,
                )
                lp_p = 2.0 * la_p
            else:
                dx_p = bf_net(prop, spin=spin_bn)
                _, la_p = slater_determinant_closed_shell(
                    x_config=prop + dx_p, C_occ=C_occ, params=params, spin=spin,
                    normalize=True,
                )
                j_p = jas_net(prop, spin=spin_bn).squeeze(-1)
                lp_p = 2.0 * (la_p + j_p)

            acc = (torch.rand_like(lp).log() < (lp_p - lp))
            x = torch.where(acc.view(-1, 1, 1), prop, x)
            lp = torch.where(acc, lp_p, lp)

    # Histogram all electron positions
    all_pos = x.reshape(-1, DIM).cpu().numpy()  # (N_samples * N_elec, 2)
    return all_pos


def compute_pair_correlation(C_occ, spin, bf_net, jas_net,
                             n_samples=16384, burn_in=500, mode="full"):
    """Compute same-spin pair distance distribution from MCMC."""
    ell = 1.0 / math.sqrt(OMEGA)
    x = torch.randn(n_samples, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * ell
    step = 0.12 * ell if mode == "sd" else 0.15 * ell
    spin_bn = spin.unsqueeze(0).expand(n_samples, -1)
    params = _params()

    with torch.no_grad():
        if mode == "sd":
            _, la = slater_determinant_closed_shell(
                x_config=x, C_occ=C_occ, params=params, spin=spin, normalize=True
            )
            lp = 2.0 * la
        else:
            dx = bf_net(x, spin=spin_bn)
            _, la = slater_determinant_closed_shell(
                x_config=x + dx, C_occ=C_occ, params=params, spin=spin, normalize=True
            )
            j = jas_net(x, spin=spin_bn).squeeze(-1)
            lp = 2.0 * (la + j)

        for _ in range(burn_in):
            prop = x + torch.randn_like(x) * step
            if mode == "sd":
                _, la_p = slater_determinant_closed_shell(
                    x_config=prop, C_occ=C_occ, params=params, spin=spin,
                    normalize=True,
                )
                lp_p = 2.0 * la_p
            else:
                dx_p = bf_net(prop, spin=spin_bn)
                _, la_p = slater_determinant_closed_shell(
                    x_config=prop + dx_p, C_occ=C_occ, params=params, spin=spin,
                    normalize=True,
                )
                j_p = jas_net(prop, spin=spin_bn).squeeze(-1)
                lp_p = 2.0 * (la_p + j_p)

            acc = (torch.rand_like(lp).log() < (lp_p - lp))
            x = torch.where(acc.view(-1, 1, 1), prop, x)
            lp = torch.where(acc, lp_p, lp)

    # Same-spin pair distances
    x_np = x.cpu().numpy()
    spin_np = spin.cpu().numpy()
    same_dists = []
    opp_dists = []
    for b in range(n_samples):
        for i in range(N_ELEC):
            for j in range(i + 1, N_ELEC):
                d = np.linalg.norm(x_np[b, i] - x_np[b, j])
                if spin_np[i] == spin_np[j]:
                    same_dists.append(d)
                else:
                    opp_dists.append(d)
    return np.array(same_dists), np.array(opp_dists)


# ===================== PLOTTING =====================
def plot_conditional_wavefunction(coords, data, x_fixed, spin, scan_idx):
    """
    Main figure: 2D heatmaps of the conditional wavefunction.
    Row 1: log|Ψ|² for SD, SD+J, SD+J+BF
    Row 2: Signed Ψ for SD, full, and node contour overlay
    """
    ell = 1.0 / math.sqrt(OMEGA)
    lim = GRID_EXTENT * ell
    extent = [-lim, lim, -lim, lim]
    x_np = x_fixed.cpu().numpy()
    spin_np = spin.cpu().numpy()
    scan_spin = "↑" if spin_np[scan_idx] == 0 else "↓"

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- Row 1: log|Ψ|² ---
    titles = [
        "$|\\Psi_{SD}|^2$ (bare Slater)",
        "$|\\Psi_{SD} \\cdot J|^2$ (+ Jastrow)",
        "$|\\Psi_{SD+BF} \\cdot J|^2$ (+ Backflow + Jastrow)",
    ]
    keys = ["psi2_sd", "psi2_sdj", "psi2_full"]

    # Find global color scale
    vmax = max(data[k].max() for k in keys)
    vmax = max(vmax, 1e-30)

    for col, (key, title) in enumerate(zip(keys, titles)):
        ax = axes[0, col]
        psi2 = data[key]
        # Use log scale for dynamic range
        log_psi2 = np.log10(np.clip(psi2, 1e-30, None))
        log_max = np.log10(vmax)

        im = ax.imshow(
            log_psi2.T, origin="lower", extent=extent,
            cmap="inferno", vmin=log_max - 8, vmax=log_max,
            aspect="equal", interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax, label="$\\log_{10}|\\Psi|^2$", shrink=0.8)

        # Mark fixed electrons
        for i in range(N_ELEC):
            if i == scan_idx:
                continue
            marker = "^" if spin_np[i] == 0 else "v"
            color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
            ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                    ms=12, mew=2, mec="white", zorder=10)

        # Node contour (where SD changes sign)
        if col == 0:
            ax.contour(
                coords, coords, data["sign_sd"].T,
                levels=[0], colors=["white"], linewidths=1.5,
                linestyles="--",
            )
        elif col == 2:
            ax.contour(
                coords, coords, data["sign_full"].T,
                levels=[0], colors=["white"], linewidths=1.5,
                linestyles="--",
            )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("$x$ (a.u.)")
        ax.set_ylabel("$y$ (a.u.)")

    # --- Row 2: Signed wavefunction ---
    # SD signed
    ax = axes[1, 0]
    psi = data["psi_sd"]
    vlim = np.percentile(np.abs(psi[psi != 0]), 99) if np.any(psi != 0) else 1.0
    im = ax.imshow(
        psi.T, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        aspect="equal", interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="$\\Psi_{SD}$", shrink=0.8)
    ax.contour(coords, coords, psi.T, levels=[0], colors=["black"],
               linewidths=2, linestyles="-")
    for i in range(N_ELEC):
        if i == scan_idx:
            continue
        marker = "^" if spin_np[i] == 0 else "v"
        color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
        ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                ms=12, mew=2, mec="black", zorder=10)
    ax.set_title("Signed $\\Psi_{SD}$ — nodal lines (black)", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    # Full signed
    ax = axes[1, 1]
    psi_f = data["psi_full"]
    vlim_f = np.percentile(np.abs(psi_f[psi_f != 0]), 99) if np.any(psi_f != 0) else 1.0
    im = ax.imshow(
        psi_f.T, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vlim_f, vmax=vlim_f,
        aspect="equal", interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="$\\Psi_{full}$", shrink=0.8)
    ax.contour(coords, coords, psi_f.T, levels=[0], colors=["black"],
               linewidths=2, linestyles="-")
    for i in range(N_ELEC):
        if i == scan_idx:
            continue
        marker = "^" if spin_np[i] == 0 else "v"
        color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
        ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                ms=12, mew=2, mec="black", zorder=10)
    ax.set_title("Signed $\\Psi_{full}$ — nodes shifted by BF", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    # Overlay: SD nodes vs BF nodes
    ax = axes[1, 2]
    # Background: log|Ψ_full|²
    log_psi2_f = np.log10(np.clip(data["psi2_full"], 1e-30, None))
    log_max_f = np.log10(max(data["psi2_full"].max(), 1e-30))
    ax.imshow(
        log_psi2_f.T, origin="lower", extent=extent,
        cmap="Greys", vmin=log_max_f - 6, vmax=log_max_f,
        aspect="equal", interpolation="bilinear", alpha=0.5,
    )
    # SD node contour
    ax.contour(coords, coords, data["psi_sd"].T, levels=[0],
               colors=["blue"], linewidths=2, linestyles="--")
    # BF node contour
    ax.contour(coords, coords, data["psi_full"].T, levels=[0],
               colors=["red"], linewidths=2, linestyles="-")
    for i in range(N_ELEC):
        if i == scan_idx:
            continue
        marker = "^" if spin_np[i] == 0 else "v"
        color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
        ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                ms=12, mew=2, mec="black", zorder=10)
    ax.plot([], [], "b--", lw=2, label="SD nodes (bare)")
    ax.plot([], [], "r-", lw=2, label="SD nodes (with BF)")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("Node shift: backflow moves the nodal surface", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    fig.suptitle(
        f"Conditional wavefunction: N={N_ELEC}, ω={OMEGA}\n"
        f"Scanning electron {scan_idx} ({scan_spin}), "
        f"others fixed  |  ▲=same-spin  ▼=opposite-spin",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    path = OUT_DIR / "fig5_conditional_wavefunction.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_backflow_effect(coords, data, x_fixed, spin, scan_idx):
    """
    Focused figure showing what backflow actually does:
      (a) BF displacement vector field + magnitude overlaid on SD sign
      (b) log|Ψ_BF|/|Ψ_SD| ratio — where BF enhances/suppresses
      (c) Zoomed nodal region with both contours
      (d) 1D cross-section perpendicular to node showing the node shift
      (e) BF displacement magnitude |Δx| heatmap
      (f) Jastrow contribution log|J| heatmap
    """
    ell = 1.0 / math.sqrt(OMEGA)
    lim = GRID_EXTENT * ell
    extent = [-lim, lim, -lim, lim]
    x_np = x_fixed.cpu().numpy()
    spin_np = spin.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # --- (a) BF displacement vector field ---
    ax = axes[0, 0]
    # Background: signed SD
    psi = data["psi_sd"]
    vlim = np.percentile(np.abs(psi[psi != 0]), 99) if np.any(psi != 0) else 1.0
    ax.imshow(psi.T, origin="lower", extent=extent,
              cmap="RdBu_r", vmin=-vlim, vmax=vlim,
              aspect="equal", interpolation="bilinear", alpha=0.5)
    # Node contour
    ax.contour(coords, coords, psi.T, levels=[0], colors=["black"],
               linewidths=2, linestyles="-")
    # Quiver: subsample for readability
    skip = max(1, GRID_N // 25)
    X, Y = np.meshgrid(coords[::skip], coords[::skip], indexing="ij")
    U = data["bf_dx_x"][::skip, ::skip]
    V = data["bf_dx_y"][::skip, ::skip]
    mag = np.sqrt(U**2 + V**2)
    # Scale arrows for visibility (BF displacements are small)
    scale_factor = 0.15 * lim / max(mag.max(), 1e-12)
    q = ax.quiver(X, Y, U * scale_factor, V * scale_factor,
                  mag, cmap="plasma", scale=1.0, scale_units="xy",
                  width=0.005, headwidth=4, headlength=5, zorder=5)
    plt.colorbar(q, ax=ax, label="$|\\Delta x_0|$ (a.u.)", shrink=0.8)
    # Fixed electrons
    for i in range(N_ELEC):
        if i == scan_idx:
            continue
        marker = "^" if spin_np[i] == 0 else "v"
        color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
        ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                ms=12, mew=2, mec="black", zorder=10)
    ax.set_title("BF displacement field on signed $\\Psi_{SD}$", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    # --- (b) log ratio |Ψ_BF|/|Ψ_SD| ---
    ax = axes[0, 1]
    # This is log|D(x+Δx)| - log|D(x)|, i.e. what BF does to the SD alone
    log_ratio_sd = data["bf_logabs"] - data["sd_logabs"]
    vlr = np.percentile(np.abs(log_ratio_sd[np.isfinite(log_ratio_sd)]), 98)
    vlr = max(vlr, 0.1)
    im = ax.imshow(log_ratio_sd.T, origin="lower", extent=extent,
                   cmap="RdBu_r", vmin=-vlr, vmax=vlr,
                   aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=ax,
                 label="$\\ln|D(x+\\Delta x)| - \\ln|D(x)|$", shrink=0.8)
    # Both node contours
    ax.contour(coords, coords, data["psi_sd"].T, levels=[0],
               colors=["blue"], linewidths=2, linestyles="--")
    ax.contour(coords, coords, data["psi_full"].T, levels=[0],
               colors=["red"], linewidths=2, linestyles="-")
    ax.plot([], [], "b--", lw=2, label="bare SD node")
    ax.plot([], [], "r-", lw=2, label="BF-shifted node")
    ax.legend(loc="upper right", fontsize=10)
    for i in range(N_ELEC):
        if i == scan_idx:
            continue
        marker = "^" if spin_np[i] == 0 else "v"
        color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
        ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                ms=10, mew=2, mec="black", zorder=10)
    ax.set_title("BF effect on SD: $\\ln|D_{BF}|/|D_{bare}|$", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    # --- (c) Zoomed nodal region ---
    ax = axes[0, 2]
    # Find where the SD node is and zoom in
    # Use the signed psi to find the zero crossing
    psi_sd_1d_vals = data["psi_sd"]
    # Find approximate node location from the sign change
    sign_changes_x = np.diff(np.sign(psi_sd_1d_vals), axis=0)
    sign_changes_y = np.diff(np.sign(psi_sd_1d_vals), axis=1)
    node_mask = np.zeros_like(psi_sd_1d_vals, dtype=bool)
    node_mask[:-1, :] |= sign_changes_x != 0
    node_mask[:, :-1] |= sign_changes_y != 0
    node_ij = np.argwhere(node_mask)
    if len(node_ij) > 0:
        # Zoom to region around the node center
        ci = int(np.median(node_ij[:, 0]))
        cj = int(np.median(node_ij[:, 1]))
        zoom_half = GRID_N // 5  # ±20% of the grid
        i0 = max(0, ci - zoom_half)
        i1 = min(GRID_N, ci + zoom_half)
        j0 = max(0, cj - zoom_half)
        j1 = min(GRID_N, cj + zoom_half)
        zoom_ext = [coords[i0], coords[min(i1, GRID_N - 1)],
                    coords[j0], coords[min(j1, GRID_N - 1)]]
        zoom_coords_x = coords[i0:i1]
        zoom_coords_y = coords[j0:j1]

        # Signed SD in zoom
        psi_z = data["psi_sd"][i0:i1, j0:j1]
        psi_bf_z = data["psi_full"][i0:i1, j0:j1]
        vlz = np.percentile(np.abs(psi_z[psi_z != 0]), 99) if np.any(psi_z != 0) else 1.0
        ax.imshow(psi_z.T, origin="lower", extent=zoom_ext,
                  cmap="RdBu_r", vmin=-vlz, vmax=vlz,
                  aspect="equal", interpolation="bilinear", alpha=0.6)
        ax.contour(zoom_coords_x, zoom_coords_y, psi_z.T, levels=[0],
                   colors=["blue"], linewidths=3, linestyles="--")
        ax.contour(zoom_coords_x, zoom_coords_y, psi_bf_z.T, levels=[0],
                   colors=["red"], linewidths=3, linestyles="-")

        # Quiver in zoomed region
        skip_z = max(1, (i1 - i0) // 15)
        Xz, Yz = np.meshgrid(coords[i0:i1:skip_z], coords[j0:j1:skip_z], indexing="ij")
        Uz = data["bf_dx_x"][i0:i1:skip_z, j0:j1:skip_z]
        Vz = data["bf_dx_y"][i0:i1:skip_z, j0:j1:skip_z]
        mag_z = np.sqrt(Uz**2 + Vz**2)
        zoom_width = coords[min(i1, GRID_N - 1)] - coords[i0]
        sf_z = 0.15 * zoom_width / max(mag_z.max(), 1e-12)
        ax.quiver(Xz, Yz, Uz * sf_z, Vz * sf_z,
                  color="black", scale=1.0, scale_units="xy",
                  width=0.006, headwidth=4, headlength=5, zorder=5, alpha=0.8)

        ax.plot([], [], "b--", lw=3, label="bare SD node")
        ax.plot([], [], "r-", lw=3, label="BF-shifted node")
        ax.legend(loc="upper right", fontsize=10)
        for i in range(N_ELEC):
            if i == scan_idx:
                continue
            if (zoom_ext[0] <= x_np[i, 0] <= zoom_ext[1] and
                    zoom_ext[2] <= x_np[i, 1] <= zoom_ext[3]):
                marker = "^" if spin_np[i] == 0 else "v"
                ax.plot(x_np[i, 0], x_np[i, 1], marker, color="lime",
                        ms=14, mew=2, mec="black", zorder=10)
    ax.set_title("Zoomed nodal region + BF arrows", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    # --- (d) 1D cross-section through the node ---
    ax = axes[1, 0]
    # Slice perpendicular to the node through the node center
    if len(node_ij) > 0:
        ci = int(np.median(node_ij[:, 0]))
        cj = int(np.median(node_ij[:, 1]))
        # Try a diagonal slice (the node is roughly diagonal in these HO systems)
        # Take a slice along j at fixed i = ci (horizontal cut through node)
        sd_slice = data["psi_sd"][ci, :]
        bf_slice = data["psi_full"][ci, :]
        # Also the Jastrow-only version
        sdj_logabs = data["sd_logabs"][ci, :] + data["jas_logabs"][ci, :]
        sdj_sign = data["sign_sd"][ci, :]
        sdj_slice = sdj_sign * np.exp(sdj_logabs - sdj_logabs.max())
        # Normalize for comparison
        sd_norm = sd_slice / max(np.abs(sd_slice).max(), 1e-30)
        bf_norm = bf_slice / max(np.abs(bf_slice).max(), 1e-30)
        sdj_norm = sdj_slice / max(np.abs(sdj_slice).max(), 1e-30)

        ax.plot(coords, sd_norm, "b-", lw=2, label="bare SD", alpha=0.8)
        ax.plot(coords, sdj_norm, "g--", lw=2, label="SD + Jastrow", alpha=0.8)
        ax.plot(coords, bf_norm, "r-", lw=2.5, label="SD + BF + Jastrow")
        ax.axhline(y=0, color="gray", ls=":", alpha=0.5)
        # Mark zero crossings
        for arr, c, ls in [(sd_norm, "blue", "--"), (bf_norm, "red", "-")]:
            sign_ch = np.where(np.diff(np.sign(arr)))[0]
            for sc_idx in sign_ch:
                # Linear interpolation for zero crossing
                x0 = coords[sc_idx]
                x1 = coords[sc_idx + 1]
                y0 = arr[sc_idx]
                y1 = arr[sc_idx + 1]
                if abs(y1 - y0) > 1e-15:
                    xz = x0 - y0 * (x1 - x0) / (y1 - y0)
                    ax.axvline(x=xz, color=c, ls=ls, alpha=0.6, lw=1.5)
        ax.set_xlabel("$y$ (a.u.)", fontsize=12)
        ax.set_ylabel("$\\Psi$ (normalized)", fontsize=12)
        ax.set_title(f"1D slice at $x={coords[ci]:.2f}$: node shift visible", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(coords[max(0, cj - GRID_N // 4)],
                    coords[min(GRID_N - 1, cj + GRID_N // 4)])
    else:
        ax.text(0.5, 0.5, "No node found", transform=ax.transAxes, ha="center")

    # --- (e) |Δx| magnitude heatmap ---
    ax = axes[1, 1]
    bf_mag = np.sqrt(data["bf_dx_x"]**2 + data["bf_dx_y"]**2)
    im = ax.imshow(bf_mag.T, origin="lower", extent=extent,
                   cmap="hot", aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="$|\\Delta x_0|$ (a.u.)", shrink=0.8)
    # Node contours
    ax.contour(coords, coords, data["psi_sd"].T, levels=[0],
               colors=["cyan"], linewidths=2, linestyles="--")
    for i in range(N_ELEC):
        if i == scan_idx:
            continue
        marker = "^" if spin_np[i] == 0 else "v"
        color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
        ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                ms=10, mew=2, mec="white", zorder=10)
    ax.set_title("BF displacement magnitude $|\\Delta x_0|$", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    # --- (f) Jastrow contribution ---
    ax = axes[1, 2]
    jas = data["jas_logabs"]
    jas_centered = jas - np.median(jas)
    vlj = np.percentile(np.abs(jas_centered[np.isfinite(jas_centered)]), 98)
    vlj = max(vlj, 0.1)
    im = ax.imshow(jas_centered.T, origin="lower", extent=extent,
                   cmap="PuOr_r", vmin=-vlj, vmax=vlj,
                   aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="$\\ln J(x) - \\mathrm{median}$", shrink=0.8)
    ax.contour(coords, coords, data["psi_sd"].T, levels=[0],
               colors=["black"], linewidths=1.5, linestyles="--")
    for i in range(N_ELEC):
        if i == scan_idx:
            continue
        marker = "^" if spin_np[i] == 0 else "v"
        color = "cyan" if spin_np[i] == spin_np[scan_idx] else "lime"
        ax.plot(x_np[i, 0], x_np[i, 1], marker, color=color,
                ms=10, mew=2, mec="black", zorder=10)
    ax.set_title("Jastrow factor $\\ln J$ (correlation envelope)", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    fig.suptitle(
        f"Backflow effect analysis: N={N_ELEC}, ω={OMEGA}, "
        f"bf_scale={0.156:.3f}\n"
        f"Scanning electron {scan_idx} "
        f"({'↑' if spin_np[scan_idx] == 0 else '↓'}), others fixed",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    path = OUT_DIR / "fig7_backflow_effect.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_densities(pos_sd, pos_full, same_sd, opp_sd, same_full, opp_full):
    """One-body density + pair correlation."""
    ell = 1.0 / math.sqrt(OMEGA)
    lim = 4.0 * ell

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: One-body density
    bins_2d = 120
    range_2d = [[-lim, lim], [-lim, lim]]

    for col, (pos, label) in enumerate([
        (pos_sd, "SD only"),
        (pos_full, "Full $\\Psi$ (BF+Jas)"),
    ]):
        ax = axes[0, col]
        H, xedges, yedges = np.histogram2d(
            pos[:, 0], pos[:, 1], bins=bins_2d, range=range_2d, density=True
        )
        im = ax.imshow(
            H.T, origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis", aspect="equal", interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax, label="$\\rho(x,y)$", shrink=0.8)
        ax.set_title(f"One-body density — {label}", fontsize=13)
        ax.set_xlabel("$x$ (a.u.)")
        ax.set_ylabel("$y$ (a.u.)")

    # Difference
    ax = axes[0, 2]
    H_sd, xe, ye = np.histogram2d(
        pos_sd[:, 0], pos_sd[:, 1], bins=bins_2d, range=range_2d, density=True
    )
    H_full, _, _ = np.histogram2d(
        pos_full[:, 0], pos_full[:, 1], bins=bins_2d, range=range_2d, density=True
    )
    diff = H_full - H_sd
    vlim_d = np.percentile(np.abs(diff), 99)
    im = ax.imshow(
        diff.T, origin="lower",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap="RdBu_r", vmin=-vlim_d, vmax=vlim_d,
        aspect="equal", interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="$\\Delta\\rho$", shrink=0.8)
    ax.set_title("$\\rho_{full} - \\rho_{SD}$ (correlation effect)", fontsize=13)
    ax.set_xlabel("$x$ (a.u.)")
    ax.set_ylabel("$y$ (a.u.)")

    # Row 2: Pair correlation
    bins_r = 100
    r_max = 5.0 * ell

    ax = axes[1, 0]
    ax.hist(same_sd, bins=bins_r, range=(0, r_max), density=True, alpha=0.6,
            color="blue", label="SD only", edgecolor="none")
    ax.hist(same_full, bins=bins_r, range=(0, r_max), density=True, alpha=0.6,
            color="red", label="Full $\\Psi$", edgecolor="none")
    ax.set_xlabel("Same-spin pair distance $r_{ij}$")
    ax.set_ylabel("$g_{\\uparrow\\uparrow}(r)$")
    ax.set_title("Same-spin pair correlation", fontsize=13)
    ax.legend()
    ax.axvline(x=0, color="black", ls=":", alpha=0.3)

    ax = axes[1, 1]
    ax.hist(opp_sd, bins=bins_r, range=(0, r_max), density=True, alpha=0.6,
            color="blue", label="SD only", edgecolor="none")
    ax.hist(opp_full, bins=bins_r, range=(0, r_max), density=True, alpha=0.6,
            color="red", label="Full $\\Psi$", edgecolor="none")
    ax.set_xlabel("Opposite-spin pair distance $r_{ij}$")
    ax.set_ylabel("$g_{\\uparrow\\downarrow}(r)$")
    ax.set_title("Opposite-spin pair correlation", fontsize=13)
    ax.legend()

    # Ratio g_full / g_sd for same-spin
    ax = axes[1, 2]
    h_sd, edges = np.histogram(same_sd, bins=bins_r, range=(0, r_max), density=True)
    h_full, _ = np.histogram(same_full, bins=bins_r, range=(0, r_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = h_sd > 0.01 * h_sd.max()
    ratio = np.where(mask, h_full / np.maximum(h_sd, 1e-30), np.nan)
    ax.plot(centers[mask], ratio[mask], "k-", lw=2)
    ax.axhline(y=1.0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Same-spin pair distance $r_{ij}$")
    ax.set_ylabel("$g_{full}/g_{SD}$")
    ax.set_title("Correlation ratio (how BF+J reshape pairing)", fontsize=13)

    fig.suptitle(
        f"Density and pair correlation: N={N_ELEC}, ω={OMEGA}",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    path = OUT_DIR / "fig6_densities.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


# ===================== MAIN =====================
def main():
    print("=" * 60)
    print("2D WAVEFUNCTION VISUALIZATION")
    print("=" * 60)

    C_occ = setup()
    bf_net, jas_net = load_models()
    spin = make_spin()

    # --- Find a good equilibrium config ---
    print("\nFinding equilibrium configuration...")
    x0 = find_equilibrium_config(C_occ, spin, bf_net, jas_net)

    # --- Evaluate on 2D grid (scan electron 0, spin-up) ---
    print(f"\nEvaluating on {GRID_N}×{GRID_N} grid (scanning electron 0)...")
    coords, data = eval_on_grid(x0, scan_idx=0, C_occ=C_occ, spin=spin,
                                bf_net=bf_net, jas_net=jas_net)

    # --- Plot conditional wavefunction ---
    print("\nPlotting conditional wavefunction...")
    plot_conditional_wavefunction(coords, data, x0, spin, scan_idx=0)

    # --- Plot focused backflow effect ---
    print("\nPlotting backflow effect analysis...")
    plot_backflow_effect(coords, data, x0, spin, scan_idx=0)

    # --- One-body density ---
    print("\nComputing one-body density (SD only)...")
    pos_sd = compute_onebody_density(C_occ, spin, bf_net, jas_net, mode="sd")
    print("Computing one-body density (full)...")
    pos_full = compute_onebody_density(C_occ, spin, bf_net, jas_net, mode="full")

    # --- Pair correlation ---
    print("\nComputing pair correlation (SD only)...")
    same_sd, opp_sd = compute_pair_correlation(C_occ, spin, bf_net, jas_net, mode="sd")
    print("Computing pair correlation (full)...")
    same_full, opp_full = compute_pair_correlation(C_occ, spin, bf_net, jas_net, mode="full")

    # --- Plot densities ---
    print("\nPlotting densities...")
    plot_densities(pos_sd, pos_full, same_sd, opp_sd, same_full, opp_full)

    print(f"\nAll figures saved to {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("fig*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
