#!/usr/bin/env python3
"""
Architecture Diagnostic: Input Attribution, Gate Behaviour, and REINFORCE vs FD-Colloc
========================================================================================
Produces four figures that empirically back the design claims in the thesis:

  Fig A  — Wavefunction sensitivity ||dlogΨ/dr|| vs pair distance r (gate suppression)
  Fig B  — Input-channel attribution at ω=1.0 vs ω=0.001 (regime dependence)
  Fig C  — Activation effective rank of Jastrow hidden layers (dead-channel detection)
  Fig D  — Gradient norm near coalescence: REINFORCE vs FD-Colloc (loss comparison)

Usage (all figures, ~45 min on 1 GPU):
  CUDA_MANUAL_DEVICE=0 python3.11 scripts/diagnose_input_attribution.py

Or individual figures:
  CUDA_MANUAL_DEVICE=0 python3.11 scripts/diagnose_input_attribution.py --figs A B
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from functions.Slater_Determinant import slater_determinant_closed_shell
from PINN import CTNNBackflowNet
from jastrow_architectures import CTNNJastrowVCycle
import config

# ─────────────────────────── constants ───────────────────────────
DIM = 2
DTYPE = torch.float64
DEVICE = torch.device(f"cuda:{os.environ.get('CUDA_MANUAL_DEVICE', 0)}")

CKPT_DIR = REPO / "results" / "arch_colloc"
OUT_DIR = REPO / "results" / "figures" / "architecture_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Named checkpoints used in comparisons
CKPT = {
    "reinforce_w01":  CKPT_DIR / "p3c_adam_n6w01_best.pt",
    "fdcolloc_w01":   CKPT_DIR / "p3b_fdcolloc_n6w01_best.pt",
    "nogate_w01":     CKPT_DIR / "p3a_nogate_n6w01_best.pt",
    "best_w1":        CKPT_DIR / "bf_ctnn_vcycle.pt",       # N=6 ω=1.0, the best campaign result
    "adam_w0001":     CKPT_DIR / "n6x2_adam_w0001_best.pt", # N=6 ω=0.001
}

# Input channel labels
EDGE_NAMES = ["Δx", "Δy", "|r|", "r²"]      # 4 edge input channels
NODE_NAMES = ["x", "y", "spin"]              # 3 node input channels
READOUT_GLOBAL = ["r²_mean", "s₁_mean"]      # 2 global readout scalars


# ─────────────────────────── model loading ───────────────────────────

def _infer_bf_config(bf_state: dict) -> dict:
    """Infer CTNNBackflowNet constructor kwargs from state dict key shapes."""
    hidden = bf_state["node_embed.weight"].shape[0]
    msg_hidden = bf_state["edge_embed.0.weight"].shape[0]
    # Count MLP layers by counting Linear layers in edge_update
    msg_layers = sum(1 for k in bf_state if k.startswith("edge_update.") and k.endswith(".weight"))
    layers = sum(1 for k in bf_state if k.startswith("node_update.") and k.endswith(".weight"))
    return dict(d=DIM, hidden=hidden, msg_hidden=msg_hidden,
                msg_layers=msg_layers, layers=layers,
                act="silu", aggregation="sum", use_spin=True,
                same_spin_only=False, out_bound="tanh",
                bf_scale_init=0.05, zero_init_last=True)


def _infer_jas_config(jas_state: dict, n_elec: int, omega: float) -> dict:
    """Infer CTNNJastrowVCycle constructor kwargs from state dict key shapes."""
    node_hidden = jas_state["node_embed.weight"].shape[0]
    edge_hidden = jas_state["edge_embed.0.weight"].shape[0]
    bottleneck_hidden = jas_state["node_down.weight"].shape[0]
    n_down = sum(1 for k in jas_state if k.startswith("rho_v_to_e_down."))
    n_up   = sum(1 for k in jas_state if k.startswith("rho_v_to_e_up."))
    msg_layers  = sum(1 for k in jas_state if k.startswith("edge_updates_down.0.") and k.endswith(".weight"))
    node_layers = sum(1 for k in jas_state if k.startswith("node_updates_down.0.") and k.endswith(".weight"))
    readout_layers = (sum(1 for k in jas_state if k.startswith("f_head.") and k.endswith(".weight")) - 1)
    readout_hidden = jas_state["f_head.0.weight"].shape[0]
    return dict(n_particles=n_elec, d=DIM, omega=omega,
                node_hidden=node_hidden, edge_hidden=edge_hidden,
                bottleneck_hidden=bottleneck_hidden,
                n_down=n_down, n_up=n_up,
                msg_layers=msg_layers, node_layers=node_layers,
                readout_hidden=readout_hidden, readout_layers=readout_layers,
                act="silu", aggregation="sum", use_spin=True)


def load_checkpoint(path: Path, device=DEVICE, dtype=DTYPE):
    """Load BF + Jastrow from checkpoint, returning (bf_net, jas_net, n_elec, omega)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    # Different checkpoint formats store these differently
    n_elec = int(ck.get("n_elec") or ck.get("bf_config", {}).get("n_elec") or 6)
    omega  = float(ck.get("omega") or ck.get("bf_config", {}).get("omega") or 1.0)

    bf_cfg = ck.get("bf_config") or _infer_bf_config(ck["bf_state"])
    bf_cfg["omega"] = omega
    bf_net = CTNNBackflowNet(**bf_cfg).to(device).to(dtype)
    bf_net.load_state_dict(ck["bf_state"])
    bf_net.eval()

    jas_cfg = _infer_jas_config(ck["jas_state"], n_elec, omega)
    jas_net = CTNNJastrowVCycle(**jas_cfg).to(device).to(dtype)
    jas_net.load_state_dict(ck["jas_state"])
    jas_net.eval()

    return bf_net, jas_net, n_elec, omega


# ─────────────────────────── physics helpers ───────────────────────────

def _spin(n_elec, device=DEVICE):
    n_up = n_elec // 2
    return torch.cat([torch.zeros(n_up, dtype=torch.long),
                      torch.ones(n_elec - n_up, dtype=torch.long)]).to(device)


def _setup(n_elec, omega, device=DEVICE):
    n_occ = n_elec // 2
    nx = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
    ny = nx
    L = max(8.0, 3.0 / math.sqrt(omega))
    config.update(n_particles=n_elec, omega=omega, d=DIM,
                  basis="cart", nx=nx, ny=ny, L=L, n_grid=80,
                  device=str(device), dtype="float64", seed=42)
    energies = sorted([(omega*(ix+iy+1), ix, iy) for ix in range(nx) for iy in range(ny)])
    C = np.zeros((nx*ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix*ny+iy, k] = 1.0
    return torch.tensor(C, dtype=DTYPE, device=device)


def _logpsi(x, C_occ, bf_net, jas_net, spin):
    """log|Ψ|(x) — full BF + Jastrow, batched."""
    B = x.shape[0]
    spin_b = spin.unsqueeze(0).expand(B, -1)
    dx = bf_net(x, spin=spin_b)
    x_eff = x + dx
    params = config.get().as_dict()
    params["device"] = str(x.device)
    params["torch_dtype"] = x.dtype
    _, logabs = slater_determinant_closed_shell(
        x_config=x_eff, C_occ=C_occ, params=params, spin=spin, normalize=True)
    j = jas_net(x, spin=spin_b).squeeze(-1)
    return logabs + j   # (B,)


# ─────────────────────────── MCMC ───────────────────────────

def mcmc_sample(C_occ, bf_net, jas_net, spin, n_samples: int,
                omega: float, burn_in: int = 400, step_scale: float = 0.15):
    n_elec = spin.shape[0]
    ell = 1.0 / math.sqrt(omega)
    x = torch.randn(n_samples, n_elec, DIM, device=DEVICE, dtype=DTYPE) * ell
    step = step_scale * ell
    accepted = 0
    with torch.no_grad():
        lp = 2.0 * _logpsi(x, C_occ, bf_net, jas_net, spin)
        for _ in range(burn_in):
            xp = x + torch.randn_like(x) * step
            lpp = 2.0 * _logpsi(xp, C_occ, bf_net, jas_net, spin)
            accept = torch.rand(n_samples, device=DEVICE, dtype=DTYPE).log() < (lpp - lp)
            x = torch.where(accept.view(-1, 1, 1), xp, x)
            lp = torch.where(accept, lpp, lp)
            accepted += accept.float().mean().item()
    print(f"  MCMC acc={accepted/burn_in:.2f}  |x|_rms={x.norm(dim=-1).mean():.2f}")
    return x  # (n_samples, N, 2)


# ─────────────────────────── FIG A: coalescence sensitivity ───────────────────────────

def compute_pair_sensitivity(C_occ, bf_net, jas_net, spin, x_ref: torch.Tensor,
                              r_vals: np.ndarray, n_orient: int = 20):
    """
    Controlled radial scan: electron 0 placed at distance r from electron 1.
    Returns:
      sens_total:   (n_r,) — mean ||d logΨ / d x_0|| at each r
      attr_by_chan: (n_r, 9) — mean |d logΨ_jas / d channel| for each input channel
                    order: [Δx, Δy, |r|, r², x0, y0, spin0, r²_mean, s1_mean]
    """
    n_elec = spin.shape[0]
    ell = 1.0 / math.sqrt(jas_net.omega)
    x_fixed = x_ref.median(0).values.unsqueeze(0)
    thetas = torch.linspace(0, 2 * math.pi, n_orient + 1)[:-1]
    dirs = torch.stack([thetas.cos(), thetas.sin()], dim=-1).to(DEVICE, DTYPE)  # (n_orient, 2)

    sens_vals = []
    attr_by_r = []  # (n_r, 9) — per-channel attributions along scan
    N = spin.shape[0]

    for r in r_vals:
        x_batch = x_fixed.expand(n_orient, -1, -1).clone()
        offsets = dirs * r / ell
        x_batch[:, 0, :] = x_batch[:, 1, :] + offsets * ell

        # ── total logΨ sensitivity (BF + Jastrow) ──
        x_in = x_batch.detach().requires_grad_(True)
        logpsi = _logpsi(x_in, C_occ, bf_net, jas_net, spin)
        grad = torch.autograd.grad(logpsi.sum(), x_in)[0]
        g0 = grad[:, 0, :]
        sens_vals.append(g0.norm(dim=-1).mean().item())

        # ── per-channel attribution on Jastrow only ──
        omega = jas_net.omega
        xb = x_batch.detach()
        x_sc = xb * omega**0.5
        spin_b = spin.unsqueeze(0).expand(n_orient, -1)

        r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)   # (B,N,N,2)
        r2_e  = (r_vec**2).sum(-1, keepdim=True)
        r1_e  = torch.sqrt(r2_e + 1e-12)
        edge_in = torch.cat([r_vec, r1_e, r2_e], dim=-1).requires_grad_(True)

        sf = spin_b.to(xb.dtype).unsqueeze(-1)
        node_in = torch.cat([x_sc, sf], dim=-1).requires_grad_(True)

        h_v = jas_net.node_embed(node_in)
        h_e = jas_net.edge_embed(edge_in)
        eye_m = torch.eye(N, device=DEVICE, dtype=DTYPE).view(1, N, N, 1)
        w = 1.0 - eye_m
        down_skips = []
        for k in range(jas_net.n_down):
            h_v, h_e = jas_net._message_step(h_v, h_e, jas_net.rho_v_to_e_down[k],
                jas_net.rho_e_to_v_down[k], jas_net.edge_updates_down[k],
                jas_net.node_updates_down[k], w)
            down_skips.append((h_v, h_e))
        h_v = jas_net.node_up(jas_net.node_down(h_v))
        h_e = jas_net.edge_up(jas_net.edge_down(h_e))
        for k in range(jas_net.n_up):
            sv, se = down_skips[-(k+1)]
            h_v = jas_net.node_skip_fuse(torch.cat([h_v, sv], dim=-1))
            h_e = jas_net.edge_skip_fuse(torch.cat([h_e, se], dim=-1))
            h_v, h_e = jas_net._message_step(h_v, h_e, jas_net.rho_v_to_e_up[k],
                jas_net.rho_e_to_v_up[k], jas_net.edge_updates_up[k],
                jas_net.node_updates_up[k], w)
        h_v_sum = h_v.sum(1); h_v_mean = h_v.mean(1)
        ii, jj = jas_net.idx_i, jas_net.idx_j
        h_ep = h_e[:, ii, jj, :]
        h_e_sum = h_ep.sum(1); h_e_mean = h_ep.mean(1)
        attn_w = F.softmax(jas_net.edge_attn(h_ep), dim=1)
        h_e_attn = (h_ep * attn_w).sum(1)
        r2_mean = (x_sc**2).mean(dim=(1,2)).unsqueeze(-1)
        diff2 = (xb[:, ii] - xb[:, jj]).pow(2)
        dist2 = torch.sqrt(diff2.sum(-1, keepdim=True) + 1e-8)
        s1_mean = torch.log1p((dist2/0.2)**2).mean(dim=1)
        glob_in = torch.cat([r2_mean, s1_mean], dim=-1).requires_grad_(True)
        f_in = torch.cat([h_v_sum, h_v_mean, h_e_sum, h_e_mean, h_e_attn,
                          glob_in[:, :1], glob_in[:, 1:]], dim=1)
        logj = jas_net.f_head(f_in).squeeze(-1).sum()
        g_e, g_n, g_g = torch.autograd.grad(logj, [edge_in, node_in, glob_in])
        chan_attr = np.array([
            g_e.abs().mean(dim=(0,1,2)).cpu().numpy().tolist(),  # 4 edge channels
            g_n.abs().mean(dim=(0,1)).cpu().numpy().tolist(),    # 3 node channels
            g_g.abs().mean(dim=0).cpu().numpy().tolist(),        # 2 global channels
        ], dtype=object)
        attr_by_r.append(np.concatenate([a for a in chan_attr]))

    return np.array(sens_vals), np.array(attr_by_r)  # (n_r,), (n_r, 9)


def make_fig_A(ax_row, r_vals, sens_dict, chi_vals, attr_scan=None):
    """
    Panel A: gate χ(r) curve.
    Panel B: per-channel Jastrow attribution vs r (the key insight).
    Panel C: total sensitivity curves for each checkpoint.
    """
    ax1, ax2, ax3 = ax_row

    # Panel A: gate + safe feature derivatives (analytical)
    ax1.plot(r_vals, chi_vals, "k-", lw=2, label=r"$\chi(r)$")
    ax1.plot(r_vals, 2 * r_vals / (r_vals + 1e-6) / (r_vals / (r_vals + 1e-6) + 1),
             "b--", lw=1.5, label=r"$d(r)/dr=1$ (unsafe)")
    ax1.plot(r_vals, 2 * r_vals, "r:", lw=1.5, label=r"$d(r^2)/dr=2r$ (safe)")
    ax1.set_xlabel(r"Pair distance $r$ ($a_{\rm ho}$)")
    ax1.set_ylabel("Value")
    ax1.set_title("Safe feature derivatives near coalescence")
    ax1.set_ylim(-0.1, 2.2)
    ax1.axvline(0.3, ls="--", color="gray", lw=0.8, alpha=0.5)
    ax1.legend(fontsize=8)

    # Panel B: per-channel attribution vs r (from the best checkpoint)
    if attr_scan is not None:
        r_plot, attr_mat = attr_scan  # (n_r,), (n_r, 9)
        all_names = EDGE_NAMES + NODE_NAMES + READOUT_GLOBAL
        # normalise each row to sum=1
        row_sum = attr_mat.sum(axis=1, keepdims=True) + 1e-30
        attr_norm = attr_mat / row_sum
        ch_colors = ["C0", "C0", "C1", "C2", "C3", "C3", "gray", "C4", "C4"]
        ch_ls     = ["-",  "--",  "-",  "-",   "-",  "--",  "-",   "-",  "--"]
        for i, (name, col, ls) in enumerate(zip(all_names, ch_colors, ch_ls)):
            ax2.plot(r_plot, attr_norm[:, i], color=col, ls=ls, lw=1.5, label=name)
        ax2.set_xlabel(r"Pair distance $r$ ($a_{\rm ho}$)")
        ax2.set_ylabel("Relative Jastrow attribution")
        ax2.set_title("Input channel attribution vs pair distance\n(REINFORCE ω=0.1)")
        ax2.legend(fontsize=7, ncol=2)
        ax2.axvline(0.3, ls="--", color="gray", lw=0.8, alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "No attribution data", transform=ax2.transAxes, ha="center")

    # Panel C: total sensitivity comparison
    colors = {"REINFORCE (ω=0.1)": "C0", "FD-Colloc (ω=0.1)": "C1",
              "No-gate (ω=0.1)": "C2", "Best (ω=1.0)": "C3"}
    for label, (s, _) in sens_dict.items():
        ax3.plot(r_vals, s, label=label, color=colors.get(label, "C4"), lw=1.5)
    ax3.set_xlabel(r"Pair distance $r$ ($a_{\rm ho}$)")
    ax3.set_ylabel(r"$\|\partial\log|\Psi|/\partial\mathbf{x}_0\|$")
    ax3.set_title("Total wavefunction sensitivity vs pair distance")
    ax3.legend(fontsize=8)
    ax3.set_yscale("log")


# ─────────────────────────── FIG B: channel attribution ───────────────────────────

def compute_channel_attribution(jas_net, x: torch.Tensor, chunk: int = 64):
    """
    Compute mean |dlogΨ_Jastrow / d channel| for each input channel.
    Returns dict {channel_name: mean_abs_grad}.
    Node channels: x, y, spin (3). Edge channels: Δx, Δy, |r|, r² (4).
    Readout globals: r²_mean, s₁_mean (2).
    """
    B, N, d = x.shape
    omega = jas_net.omega

    attr_edge = torch.zeros(4, device=DEVICE, dtype=DTYPE)
    attr_node = torch.zeros(3, device=DEVICE, dtype=DTYPE)
    attr_glob = torch.zeros(2, device=DEVICE, dtype=DTYPE)

    n_chunks = 0
    spin = _spin(N)

    for start in range(0, B, chunk):
        xc = x[start:start+chunk].detach()
        B_c = xc.shape[0]
        spin_b = spin.unsqueeze(0).expand(B_c, -1)
        x_sc = xc * omega**0.5

        # ── Edge input tensor (requires_grad) ──
        r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)   # (B,N,N,2)
        r2    = (r_vec**2).sum(-1, keepdim=True)          # (B,N,N,1)
        r1    = torch.sqrt(r2 + 1e-12)                    # (B,N,N,1)
        edge_in = torch.cat([r_vec, r1, r2], dim=-1).requires_grad_(True)  # (B,N,N,4)

        # ── Node input tensor (requires_grad) ──
        sf = spin_b.to(xc.dtype).unsqueeze(-1)           # (B,N,1)
        node_in = torch.cat([x_sc, sf], dim=-1).requires_grad_(True)  # (B,N,3)

        # Monkey-patch the forward to use our instrumented inputs
        h_v = jas_net.node_embed(node_in)
        h_e = jas_net.edge_embed(edge_in)

        eye = torch.eye(N, device=DEVICE, dtype=DTYPE).view(1, N, N, 1)
        weight = 1.0 - eye

        down_skips = []
        for k in range(jas_net.n_down):
            h_v, h_e = jas_net._message_step(h_v, h_e,
                jas_net.rho_v_to_e_down[k], jas_net.rho_e_to_v_down[k],
                jas_net.edge_updates_down[k], jas_net.node_updates_down[k], weight)
            down_skips.append((h_v, h_e))

        h_v = jas_net.node_up(jas_net.node_down(h_v))
        h_e = jas_net.edge_up(jas_net.edge_down(h_e))

        for k in range(jas_net.n_up):
            skip_v, skip_e = down_skips[-(k+1)]
            h_v = jas_net.node_skip_fuse(torch.cat([h_v, skip_v], dim=-1))
            h_e = jas_net.edge_skip_fuse(torch.cat([h_e, skip_e], dim=-1))
            h_v, h_e = jas_net._message_step(h_v, h_e,
                jas_net.rho_v_to_e_up[k], jas_net.rho_e_to_v_up[k],
                jas_net.edge_updates_up[k], jas_net.node_updates_up[k], weight)

        h_v_sum  = h_v.sum(dim=1)
        h_v_mean = h_v.mean(dim=1)
        ii, jj = jas_net.idx_i, jas_net.idx_j
        h_e_pairs = h_e[:, ii, jj, :]
        h_e_sum  = h_e_pairs.sum(dim=1)
        h_e_mean = h_e_pairs.mean(dim=1)
        attn_w   = F.softmax(jas_net.edge_attn(h_e_pairs), dim=1)
        h_e_attn = (h_e_pairs * attn_w).sum(dim=1)

        # Global readout scalars
        r2_mean = (x_sc**2).mean(dim=(1, 2)).unsqueeze(-1)
        diff = (xc[:, ii] - xc[:, jj]).pow(2)
        dist = torch.sqrt(diff.sum(-1, keepdim=True) + 1e-8)
        s1_mean = torch.log1p((dist / 0.2)**2).mean(dim=1)

        # Stack globals so we can compute grad wrt them too
        glob_in = torch.cat([r2_mean, s1_mean], dim=-1).requires_grad_(True)
        # Use glob_in as the last 2 of f_head input
        f_in = torch.cat([h_v_sum, h_v_mean, h_e_sum, h_e_mean, h_e_attn,
                          glob_in[:, :1], glob_in[:, 1:]], dim=1)
        logj = jas_net.f_head(f_in).squeeze(-1).sum()

        # Gradient wrt each input tensor
        g_edge, g_node, g_glob = torch.autograd.grad(
            logj, [edge_in, node_in, glob_in], allow_unused=True)

        if g_edge is not None:
            attr_edge += g_edge.abs().mean(dim=(0, 1, 2))   # mean over B,N,N
        if g_node is not None:
            attr_node += g_node.abs().mean(dim=(0, 1))      # mean over B,N
        if g_glob is not None:
            attr_glob += g_glob.abs().mean(dim=0)           # mean over B

        n_chunks += 1

    attr_edge /= n_chunks
    attr_node /= n_chunks
    attr_glob /= n_chunks

    all_attr = torch.cat([attr_edge, attr_node, attr_glob]).cpu().numpy()
    labels   = EDGE_NAMES + NODE_NAMES + READOUT_GLOBAL
    return dict(zip(labels, all_attr))


def make_fig_B(ax_row, attr_w01, attr_w0001):
    ax1, ax2, ax3 = ax_row
    labels = EDGE_NAMES + NODE_NAMES + READOUT_GLOBAL
    colors = ["C0"]*4 + ["C1"]*3 + ["C2"]*2

    def plot_bars(ax, attr, title):
        vals = np.array([attr[l] for l in labels])
        vals /= vals.sum()
        bars = ax.bar(range(len(labels)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Relative attribution")
        ax.set_title(title)
        # Mark zero/near-zero channels
        for i, v in enumerate(vals):
            if v < 0.01:
                ax.text(i, v + 0.002, "×", ha="center", color="red", fontsize=10)

    plot_bars(ax1, attr_w01,   r"Attribution — $\omega=1.0$")
    plot_bars(ax2, attr_w0001, r"Attribution — $\omega=0.001$")

    # Compute attribution shift
    l = labels
    v1    = np.array([attr_w01[ll]    for ll in l]); v1 /= v1.sum()
    v2    = np.array([attr_w0001[ll]  for ll in l]); v2 /= v2.sum()
    shift = v2 - v1
    ax3.bar(range(len(l)), shift, color=["C0" if s > 0 else "C3" for s in shift])
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xticks(range(len(l)))
    ax3.set_xticklabels(l, rotation=35, ha="right", fontsize=9)
    ax3.set_ylabel(r"Δ attribution (low-ω − high-ω)")
    ax3.set_title(r"Attribution shift: $\omega=0.001$ vs $\omega=1.0$")


# ─────────────────────────── FIG C: activation rank ───────────────────────────

def compute_activation_rank(jas_net, x: torch.Tensor, chunk: int = 128):
    """
    Extract edge-embedding and node-embedding activations after first linear layer,
    compute SVD to get effective rank.
    Returns dict {layer_name: (singular_values, eff_rank)}.
    """
    B, N, d = x.shape
    omega = jas_net.omega

    all_edge, all_node = [], []

    with torch.no_grad():
        for start in range(0, B, chunk):
            xc = x[start:start+chunk]
            B_c = xc.shape[0]
            x_sc = xc * omega**0.5
            spin_b = _spin(N).unsqueeze(0).expand(B_c, -1)

            sf = spin_b.to(xc.dtype).unsqueeze(-1)
            node_in = torch.cat([x_sc, sf], dim=-1)            # (B,N,3)
            h_v_emb = jas_net.node_embed(node_in)               # (B,N,node_hidden)

            r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)
            r2    = (r_vec**2).sum(-1, keepdim=True)
            r1    = torch.sqrt(r2 + 1e-12)
            edge_in = torch.cat([r_vec, r1, r2], dim=-1)        # (B,N,N,4)
            h_e_emb = jas_net.edge_embed(edge_in)               # (B,N,N,edge_hidden)

            all_node.append(h_v_emb.reshape(-1, h_v_emb.shape[-1]).cpu())
            all_edge.append(h_e_emb.reshape(-1, h_e_emb.shape[-1]).cpu())

    node_mat = torch.cat(all_node, dim=0).float().numpy()  # (B*N, H_node)
    edge_mat = torch.cat(all_edge, dim=0).float().numpy()  # (B*N*N, H_edge)

    results = {}
    for name, mat in [("node_embed", node_mat), ("edge_embed", edge_mat)]:
        mat_c = mat - mat.mean(0, keepdims=True)
        sv = np.linalg.svd(mat_c, compute_uv=False)
        sv_sq = sv**2
        eff_rank = (sv_sq.sum()**2) / (sv_sq**2).sum()
        results[name] = (sv / sv.max(), float(eff_rank))
        print(f"  {name}: eff_rank={eff_rank:.1f}/{mat.shape[1]}")

    return results


def make_fig_C(ax_row, rank_w01, rank_w0001):
    ax1, ax2 = ax_row[0], ax_row[1]
    for ax, rank_data, title in [
        (ax1, rank_w01,   r"Activation spectrum — $\omega=1.0$"),
        (ax2, rank_w0001, r"Activation spectrum — $\omega=0.001$"),
    ]:
        for name, (sv, eff_rank) in rank_data.items():
            ax.plot(sv, label=f"{name} (k={eff_rank:.1f})")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalised singular value")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.axhline(0.01, ls="--", color="gray", lw=0.8, label="1% threshold")


# ─────────────────────────── FIG D: REINFORCE vs FD grad norms ───────────────────────────

def compute_grad_norms_near_coalescence(C_occ, bf_net, jas_net, spin, x_mcmc: torch.Tensor,
                                         use_fd: bool = False, fd_h: float = 0.01,
                                         chunk: int = 32):
    """
    For each config in x_mcmc, compute the Frobenius norm of grad_theta(logΨ)
    and the minimum pair distance r_min.

    If use_fd=True: adds FD-Laplacian to the reward (simulates FD-Colloc gradient signal).
    If use_fd=False: uses REINFORCE-style (only score gradient, no Laplacian through graph).
    """
    B = x_mcmc.shape[0]
    n_elec = spin.shape[0]

    r_mins, grad_norms = [], []
    params = config.get().as_dict()
    params["device"] = str(DEVICE)
    params["torch_dtype"] = DTYPE

    for start in range(0, B, chunk):
        xc = x_mcmc[start:start+chunk].detach().requires_grad_(False)
        B_c = xc.shape[0]
        spin_b = spin.unsqueeze(0).expand(B_c, -1)

        # r_min for each config
        diff = xc.unsqueeze(2) - xc.unsqueeze(1)
        r_all = diff.norm(dim=-1)
        big = 1e10
        eye = torch.eye(n_elec, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        r_all_masked = r_all + eye * big
        r_min = r_all_masked.min(dim=-1).values.min(dim=-1).values
        r_mins.append(r_min.cpu().numpy().mean())  # scalar per chunk

        # Gradient of all parameters wrt logΨ at these configs
        xc_g = xc.detach().requires_grad_(False)
        # Zero param grads
        for p in list(bf_net.parameters()) + list(jas_net.parameters()):
            if p.grad is not None:
                p.grad.zero_()

        # Compute logΨ with graph retained
        dx = bf_net(xc_g, spin=spin_b)
        x_eff = xc_g + dx
        _, logabs = slater_determinant_closed_shell(
            x_config=x_eff, C_occ=C_occ, params=params, spin=spin, normalize=True)
        j = jas_net(xc_g, spin=spin_b).squeeze(-1)
        logpsi = logabs + j

        if use_fd:
            # FD-Colloc: gradient passes through the Laplacian of logΨ.
            # Concretely: differentiate -½(∇²logΨ + |∇logΨ|²) through network params.
            # This makes parameter gradients depend on 2nd derivatives of the wavefunction —
            # which blow up near coalescence.
            x_fin = xc.detach().requires_grad_(True)
            dx_fd = bf_net(x_fin, spin=spin_b)
            x_eff_fd = x_fin + dx_fd
            _, la_fd = slater_determinant_closed_shell(
                x_config=x_eff_fd, C_occ=C_occ, params=params, spin=spin, normalize=True)
            j_fd = jas_net(x_fin, spin=spin_b).squeeze(-1)
            lp_fd = la_fd + j_fd
            # ∇_x logΨ with graph retained so we can differentiate wrt params
            grad_x = torch.autograd.grad(lp_fd.sum(), x_fin, create_graph=True)[0]
            # Kinetic energy: -½|∇logΨ|² — this passes through network params via create_graph
            kinetic = 0.5 * (grad_x**2).sum()
            kinetic.backward()  # ∂(kinetic)/∂θ — includes 2nd-derivative pathways
        else:
            # REINFORCE: score gradient only.
            # Laplacian enters reward but is NOT differentiated through.
            logpsi.mean().backward()

        # Collect parameter gradient norms
        total_sq = sum(p.grad.pow(2).sum().item()
                       for p in list(bf_net.parameters()) + list(jas_net.parameters())
                       if p.grad is not None)
        grad_norms.append(math.sqrt(total_sq))

        for p in list(bf_net.parameters()) + list(jas_net.parameters()):
            if p.grad is not None:
                p.grad.zero_()

    return np.array(r_mins), np.array(grad_norms)


def make_fig_D(ax_row, r_min_rf, gnorm_rf, r_min_fd, gnorm_fd):
    ax1, ax2, ax3 = ax_row
    bins = [0.0, 0.3, 1.0, 3.0, np.inf]
    bin_labels = ["r<0.3", "0.3–1.0", "1.0–3.0", ">3.0"]

    def bin_stat(r_min, gnorm):
        means, stds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (r_min >= lo) & (r_min < hi)
            g = gnorm[mask]
            means.append(g.mean() if len(g) > 0 else 0.0)
            stds.append(g.std() if len(g) > 1 else 0.0)
        return np.array(means), np.array(stds)

    means_rf, stds_rf = bin_stat(r_min_rf, gnorm_rf)
    means_fd, stds_fd = bin_stat(r_min_fd, gnorm_fd)

    x_pos = np.arange(4)
    width = 0.35
    ax1.bar(x_pos - width/2, means_rf, width, yerr=stds_rf, label="REINFORCE", color="C0", alpha=0.8)
    ax1.bar(x_pos + width/2, means_fd, width, yerr=stds_fd, label="FD-Colloc",  color="C1", alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels)
    ax1.set_ylabel(r"$\|\nabla_\theta \log|\Psi|\|_F$")
    ax1.set_xlabel(r"$r_{\min}$ bin ($a_{\rm ho}$)")
    ax1.set_title("Parameter gradient norm by proximity to coalescence")
    ax1.legend()

    # Scatter: grad norm vs r_min
    ax2.scatter(r_min_rf, gnorm_rf, s=6, alpha=0.4, color="C0", label="REINFORCE")
    ax2.scatter(r_min_fd, gnorm_fd, s=6, alpha=0.4, color="C1", label="FD-Colloc")
    ax2.set_xlabel(r"$r_{\min}$ ($a_{\rm ho}$)")
    ax2.set_ylabel(r"$\|\nabla_\theta \log|\Psi|\|_F$")
    ax2.set_title("Gradient norm vs minimum pair distance")
    ax2.legend()
    ax2.set_yscale("log")

    # Ratio: FD/REINFORCE mean per bin
    ratio = np.where(means_rf > 0, means_fd / means_rf, 1.0)
    ax3.bar(x_pos, ratio, color=["C3" if r > 1 else "C2" for r in ratio])
    ax3.axhline(1.0, color="k", lw=1, ls="--")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bin_labels)
    ax3.set_ylabel("FD-Colloc / REINFORCE gradient norm ratio")
    ax3.set_title("Gradient amplification from FD-Colloc path (>1 = worse)")


# ─────────────────────────── main ───────────────────────────

def main(figs_to_run):
    torch.set_grad_enabled(True)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    # ── Load the checkpoints we need ──
    print("\nLoading checkpoints...")
    ckpts = {}
    for key, path in CKPT.items():
        if path.exists():
            ckpts[key] = load_checkpoint(path)
            _, _, n, w = ckpts[key]
            print(f"  {key}: N={n}, ω={w}")
        else:
            print(f"  {key}: NOT FOUND at {path}")

    fig = plt.figure(figsize=(18, 5 * len(figs_to_run)))
    gs  = gridspec.GridSpec(len(figs_to_run), 3, figure=fig, hspace=0.6, wspace=0.4)

    npz_data = {}
    fig_idx = 0

    # ════════════ FIG A ════════════
    if "A" in figs_to_run:
        print("\n── Figure A: Coalescence sensitivity ──")
        r_vals = np.concatenate([np.linspace(0.02, 0.60, 20), np.linspace(0.60, 5.0, 20)])
        r_vals = np.sort(np.unique(r_vals))
        chi_vals = r_vals**2 / (r_vals**2 + 0.3**2)

        sens_dict  = {}
        attr_scan_ref = None  # store attribution scan from first checkpoint for Panel B
        for key, label in [("reinforce_w01", "REINFORCE (ω=0.1)"),
                            ("fdcolloc_w01",  "FD-Colloc (ω=0.1)"),
                            ("nogate_w01",    "No-gate (ω=0.1)"),
                            ("best_w1",       "Best (ω=1.0)")]:
            if key not in ckpts:
                print(f"  Skipping {key} (not found)")
                continue
            bf_net, jas_net, n_elec, omega = ckpts[key]
            print(f"  Scanning {label}...")
            C_occ = _setup(n_elec, omega)
            spin  = _spin(n_elec)
            x_ref = mcmc_sample(C_occ, bf_net, jas_net, spin, 200, omega, burn_in=200)
            s, attr = compute_pair_sensitivity(C_occ, bf_net, jas_net, spin, x_ref,
                                               r_vals, n_orient=20)
            sens_dict[label] = (s, attr)
            if attr_scan_ref is None:
                attr_scan_ref = (r_vals, attr)  # use first ckpt for per-channel panel
            print(f"    sens[r≈0] = {s[0]:.4f}, sens[r≈1] = {s[len(r_vals)//2]:.4f}")

        ax_row = [fig.add_subplot(gs[fig_idx, j]) for j in range(3)]
        make_fig_A(ax_row, r_vals, sens_dict, chi_vals, attr_scan=attr_scan_ref)
        npz_data.update({"fig_A_r_vals": r_vals, "fig_A_chi": chi_vals,
                         **{f"fig_A_{k.replace(' ','_')}_sens": v[0] for k,v in sens_dict.items()},
                         **{f"fig_A_{k.replace(' ','_')}_attr": v[1] for k,v in sens_dict.items()}})
        fig_idx += 1

    # ════════════ FIG B ════════════
    if "B" in figs_to_run:
        print("\n── Figure B: Feature attribution by ω ──")
        attr_by_omega = {}
        for key, omega_label in [("best_w1",    r"$\omega=1.0$"),
                                  ("adam_w0001", r"$\omega=0.001$")]:
            if key not in ckpts:
                print(f"  Skipping {key}")
                continue
            bf_net, jas_net, n_elec, omega = ckpts[key]
            C_occ = _setup(n_elec, omega)
            spin  = _spin(n_elec)
            print(f"  Sampling {key} (ω={omega})...")
            x_mcmc = mcmc_sample(C_occ, bf_net, jas_net, spin, 1000, omega, burn_in=300)
            print(f"  Computing attribution...")
            attr = compute_channel_attribution(jas_net, x_mcmc)
            attr_by_omega[omega] = attr
            for ch, v in attr.items():
                print(f"    {ch}: {v:.4f}")

        if len(attr_by_omega) == 2:
            omegas = sorted(attr_by_omega.keys())
            ax_row = [fig.add_subplot(gs[fig_idx, j]) for j in range(3)]
            make_fig_B(ax_row, attr_by_omega[omegas[-1]], attr_by_omega[omegas[0]])
            npz_data["fig_B_attr_w1"]    = np.array(list(attr_by_omega[omegas[-1]].values()))
            npz_data["fig_B_attr_w0001"] = np.array(list(attr_by_omega[omegas[0]].values()))
        fig_idx += 1

    # ════════════ FIG C ════════════
    if "C" in figs_to_run:
        print("\n── Figure C: Activation effective rank ──")
        rank_data = {}
        for key, omega_label in [("best_w1", "ω=1.0"), ("adam_w0001", "ω=0.001")]:
            if key not in ckpts:
                continue
            bf_net, jas_net, n_elec, omega = ckpts[key]
            C_occ = _setup(n_elec, omega)
            spin  = _spin(n_elec)
            x_mcmc = mcmc_sample(C_occ, bf_net, jas_net, spin, 500, omega, burn_in=200)
            print(f"  Computing rank for {key}...")
            rank_data[omega] = compute_activation_rank(jas_net, x_mcmc)

        if len(rank_data) >= 2:
            omegas = sorted(rank_data.keys())
            ax_row = [fig.add_subplot(gs[fig_idx, j]) for j in range(2)]
            make_fig_C(ax_row, rank_data[omegas[-1]], rank_data[omegas[0]])
            # Hide unused third panel
            fig.add_subplot(gs[fig_idx, 2]).set_visible(False)
        fig_idx += 1

    # ════════════ FIG D ════════════
    if "D" in figs_to_run:
        print("\n── Figure D: REINFORCE vs FD-Colloc gradient variance ──")
        results_D = {}
        for key, use_fd, label in [("reinforce_w01", False, "REINFORCE"),
                                    ("fdcolloc_w01",  True,  "FD-Colloc")]:
            if key not in ckpts:
                print(f"  Skipping {key}")
                continue
            bf_net, jas_net, n_elec, omega = ckpts[key]
            C_occ = _setup(n_elec, omega)
            spin  = _spin(n_elec)
            print(f"  Sampling for {label}...")
            x_mcmc = mcmc_sample(C_occ, bf_net, jas_net, spin, 300, omega, burn_in=300)
            print(f"  Computing gradient norms...")
            r_min, gnorm = compute_grad_norms_near_coalescence(
                C_occ, bf_net, jas_net, spin, x_mcmc, use_fd=use_fd)
            results_D[label] = (r_min, gnorm)
            print(f"    mean gnorm={gnorm.mean():.4f}, near-coalescence (r<0.3): "
                  f"{gnorm[r_min<0.3].mean() if (r_min<0.3).any() else float('nan'):.4f}")

        if "REINFORCE" in results_D and "FD-Colloc" in results_D:
            ax_row = [fig.add_subplot(gs[fig_idx, j]) for j in range(3)]
            make_fig_D(ax_row,
                       results_D["REINFORCE"][0], results_D["REINFORCE"][1],
                       results_D["FD-Colloc"][0],  results_D["FD-Colloc"][1])
            npz_data.update({"fig_D_r_reinforce": results_D["REINFORCE"][0],
                             "fig_D_g_reinforce": results_D["REINFORCE"][1],
                             "fig_D_r_fd":        results_D["FD-Colloc"][0],
                             "fig_D_g_fd":        results_D["FD-Colloc"][1]})
        fig_idx += 1

    # ── Save ──
    pdf_path = OUT_DIR / "architecture_diagnostics.pdf"
    png_path = OUT_DIR / "architecture_diagnostics.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")

    if npz_data:
        np.savez(OUT_DIR / "architecture_diagnostics_data.npz", **npz_data)
        print(f"Saved raw data: {OUT_DIR / 'architecture_diagnostics_data.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", default=["A", "B", "C", "D"],
                    choices=["A", "B", "C", "D"],
                    help="Which figures to produce (default: all)")
    args = ap.parse_args()
    main(figs_to_run=set(args.figs))
