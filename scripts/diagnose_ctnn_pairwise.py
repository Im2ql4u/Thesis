#!/usr/bin/env python3
"""
CTNN vs Pairwise Diagnostics
==============================
Three measurements that reveal what message-passing adds over a pairwise/FFN Jastrow,
and what the CTNN backflow actually computes geometrically.

  Fig E  — Three-body angular correlation: logΨ(θ) as particle 2 orbits a fixed 0-1 pair.
            Pure pairwise Jastrow: logΨ = const. CTNN: varies → quantifies 3-body terms.

  Fig F  — Message-passing ablation: zero rho_v_to_e / rho_e_to_v weights in trained CTNN,
            re-evaluate energy. Degradation = benefit of inter-particle communication.

  Fig G  — Backflow correlation-hole geometry: direction and magnitude of Δx_i vs
            distance and spin of nearest neighbour.

Usage:
  CUDA_MANUAL_DEVICE=0 python3.11 scripts/diagnose_ctnn_pairwise.py
  CUDA_MANUAL_DEVICE=0 python3.11 scripts/diagnose_ctnn_pairwise.py --figs E F G
"""
from __future__ import annotations

import argparse
import copy
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

# ─────────────── constants ───────────────
DIM   = 2
DTYPE = torch.float64
DEVICE = torch.device(f"cuda:{os.environ.get('CUDA_MANUAL_DEVICE', 0)}")

CKPT_DIR = REPO / "results" / "arch_colloc"
OUT_DIR  = REPO / "results" / "figures" / "architecture_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the best N=6 ω=1.0 checkpoint + best ω=0.001 for cross-regime comparison
CKPTS = {
    "w1":    CKPT_DIR / "bf_ctnn_vcycle.pt",             # N=6, ω=1.0, best campaign
    "w01":   CKPT_DIR / "p3c_adam_n6w01_best.pt",        # N=6, ω=0.1, REINFORCE
    "w0001": CKPT_DIR / "n6x2_adam_w0001_best.pt",       # N=6, ω=0.001
}


# ─────────────── model loading (same as diagnose_input_attribution.py) ───────────────

def _infer_bf_config(bf_state):
    hidden    = bf_state["node_embed.weight"].shape[0]
    msg_hid   = bf_state["edge_embed.0.weight"].shape[0]
    msg_lay   = sum(1 for k in bf_state if k.startswith("edge_update.") and k.endswith(".weight"))
    node_lay  = sum(1 for k in bf_state if k.startswith("node_update.") and k.endswith(".weight"))
    return dict(d=DIM, hidden=hidden, msg_hidden=msg_hid,
                msg_layers=msg_lay, layers=node_lay,
                act="silu", aggregation="sum", use_spin=True,
                same_spin_only=False, out_bound="tanh",
                bf_scale_init=0.05, zero_init_last=True)


def _infer_jas_config(js, n_elec, omega):
    nh = js["node_embed.weight"].shape[0]
    eh = js["edge_embed.0.weight"].shape[0]
    bh = js["node_down.weight"].shape[0]
    nd = sum(1 for k in js if k.startswith("rho_v_to_e_down."))
    nu = sum(1 for k in js if k.startswith("rho_v_to_e_up."))
    ml = sum(1 for k in js if k.startswith("edge_updates_down.0.") and k.endswith(".weight"))
    nl = sum(1 for k in js if k.startswith("node_updates_down.0.") and k.endswith(".weight"))
    rl = sum(1 for k in js if k.startswith("f_head.") and k.endswith(".weight")) - 1
    rh = js["f_head.0.weight"].shape[0]
    return dict(n_particles=n_elec, d=DIM, omega=omega,
                node_hidden=nh, edge_hidden=eh, bottleneck_hidden=bh,
                n_down=nd, n_up=nu, msg_layers=ml, node_layers=nl,
                readout_hidden=rh, readout_layers=rl,
                act="silu", aggregation="sum", use_spin=True)


def load_checkpoint(path, device=DEVICE, dtype=DTYPE):
    ck = torch.load(path, map_location=device, weights_only=False)
    n_elec = int(ck.get("n_elec") or ck.get("bf_config", {}).get("n_elec") or 6)
    omega  = float(ck.get("omega") or ck.get("bf_config", {}).get("omega") or 1.0)

    bfc = ck.get("bf_config") or _infer_bf_config(ck["bf_state"])
    bfc["omega"] = omega
    bf_net = CTNNBackflowNet(**bfc).to(device).to(dtype)
    bf_net.load_state_dict(ck["bf_state"])
    bf_net.eval()

    jc = _infer_jas_config(ck["jas_state"], n_elec, omega)
    jas_net = CTNNJastrowVCycle(**jc).to(device).to(dtype)
    jas_net.load_state_dict(ck["jas_state"])
    jas_net.eval()

    return bf_net, jas_net, n_elec, omega


def _spin(n, device=DEVICE):
    return torch.cat([torch.zeros(n//2, dtype=torch.long),
                      torch.ones(n - n//2, dtype=torch.long)]).to(device)


def _setup(n, omega, device=DEVICE):
    n_occ = n // 2
    nx = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
    ny = nx
    L  = max(8., 3.0 / math.sqrt(omega))
    config.update(n_particles=n, omega=omega, d=DIM,
                  basis="cart", nx=nx, ny=ny, L=L, n_grid=80,
                  device=str(device), dtype="float64", seed=42)
    energies = sorted([(omega*(ix+iy+1), ix, iy) for ix in range(nx) for iy in range(ny)])
    C = np.zeros((nx*ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix*ny+iy, k] = 1.
    return torch.tensor(C, dtype=DTYPE, device=device)


def _logpsi(x, C_occ, bf_net, jas_net, spin):
    B = x.shape[0]
    sb = spin.unsqueeze(0).expand(B, -1)
    dx = bf_net(x, spin=sb)
    p  = config.get().as_dict(); p["device"] = str(DEVICE); p["torch_dtype"] = DTYPE
    _, la = slater_determinant_closed_shell(x+dx, C_occ, params=p, spin=spin, normalize=True)
    j  = jas_net(x, spin=sb).squeeze(-1)
    return la + j


def _logpsi_jas_only(x, jas_net, spin):
    """Jastrow only (no BF), for ablation comparisons."""
    B = x.shape[0]
    sb = spin.unsqueeze(0).expand(B, -1)
    return jas_net(x, spin=sb).squeeze(-1)


def mcmc(C_occ, bf_net, jas_net, spin, n, omega, burn=400, step=0.15):
    N = spin.shape[0]; ell = 1./math.sqrt(omega)
    x = torch.randn(n, N, DIM, device=DEVICE, dtype=DTYPE) * ell
    acc = 0
    with torch.no_grad():
        lp = 2.*_logpsi(x, C_occ, bf_net, jas_net, spin)
        for _ in range(burn):
            xp = x + torch.randn_like(x) * (step * ell)
            lpp = 2.*_logpsi(xp, C_occ, bf_net, jas_net, spin)
            a = torch.rand(n, device=DEVICE, dtype=DTYPE).log() < (lpp - lp)
            x  = torch.where(a.view(-1,1,1), xp, x)
            lp = torch.where(a, lpp, lp)
            acc += a.float().mean().item()
    print(f"  MCMC acc={acc/burn:.2f}  |x|_rms={x.norm(dim=-1).mean():.2f}")
    return x


# ═══════════════════════════════════════════════════════════════════
# FIG E — Three-body angular correlation
# ═══════════════════════════════════════════════════════════════════

def three_body_scan(jas_net, spin, r01=1.0, r_cross=1.0, n_theta=120, n_freeze=40,
                    omega=1.0, C_occ=None, bf_net=None):
    """
    Fix particles 0 and 1 at separation r01 (in a_ho).
    Place particle 2 at distance r_cross from the midpoint, sweeping angle θ.
    All other particles drawn from MCMC and frozen.

    Returns theta (n_theta,), logpsi_ctnn (n_theta,), logpsi_pair (n_theta,).
    logpsi_pair is computed with all inter-particle transport maps zeroed
    (pure pairwise-additive Jastrow baseline).
    """
    N = spin.shape[0]; ell = 1./math.sqrt(omega)
    # Frozen equilibrium positions for particles 3..N-1
    if C_occ is not None and bf_net is not None:
        x_eq = mcmc(C_occ, bf_net, jas_net, spin, n_freeze, omega, burn=300)
        x_frozen = x_eq.median(0).values  # (N, 2) representative config
    else:
        x_frozen = torch.randn(N, DIM, device=DEVICE, dtype=DTYPE) * ell

    # Place 0 and 1 symmetrically about origin on x-axis
    x_frozen[0] = torch.tensor([ r01/2 * ell,  0.], device=DEVICE, dtype=DTYPE)
    x_frozen[1] = torch.tensor([-r01/2 * ell,  0.], device=DEVICE, dtype=DTYPE)
    midpoint = (x_frozen[0] + x_frozen[1]) / 2   # ~origin

    thetas = torch.linspace(0, 2*math.pi, n_theta+1)[:-1]
    lp_ctnn = []
    lp_pair = []

    # Build the pairwise ablated Jastrow (zero all inter-particle transport)
    jas_ablated = copy.deepcopy(jas_net)
    _ablate_message_passing(jas_ablated)
    jas_ablated.eval()

    with torch.no_grad():
        for th in thetas:
            x = x_frozen.clone().unsqueeze(0)  # (1, N, 2)
            # Particle 2 orbits at radius r_cross from midpoint
            dx2 = torch.tensor([r_cross * th.cos() * ell,
                                 r_cross * th.sin() * ell],
                                device=DEVICE, dtype=DTYPE)
            x[0, 2] = midpoint + dx2

            sb = spin.unsqueeze(0)
            j_full   = jas_net(x, spin=sb).squeeze()
            j_ablate = jas_ablated(x, spin=sb).squeeze()

            lp_ctnn.append(j_full.item())
            lp_pair.append(j_ablate.item())

    return (thetas.numpy(),
            np.array(lp_ctnn) - np.mean(lp_ctnn),    # centre for comparison
            np.array(lp_pair) - np.mean(lp_pair))


def _ablate_message_passing(jas_net):
    """Zero all inter-particle transport weights (rho_v_to_e, rho_e_to_v).
    Leaves edge and node embeddings intact — simulates a pairwise/DeepSet Jastrow."""
    with torch.no_grad():
        for name, p in jas_net.named_parameters():
            if "rho_v_to_e" in name or "rho_e_to_v" in name:
                p.zero_()


def make_fig_E(axes, results_by_key, r_cross):
    """
    axes: list of 3 Axes (one per row column)
    results_by_key: dict {label: (theta, lp_ctnn, lp_pair)}
    """
    ax1, ax2, ax3 = axes
    colors = {"ω=1.0": "C0", "ω=0.1": "C1", "ω=0.001": "C3"}

    # Panel 1: logΨ_Jastrow(θ) — CTNN vs Pairwise for each ω
    for label, (theta, lp_ctnn, lp_pair) in results_by_key.items():
        c = colors.get(label, "C4")
        ax1.plot(np.degrees(theta), lp_ctnn, color=c, lw=2, label=f"CTNN {label}")
        ax1.plot(np.degrees(theta), lp_pair, color=c, lw=1, ls="--",
                 alpha=0.6, label=f"Pairwise {label}")
    ax1.set_xlabel("Orbit angle θ (degrees)")
    ax1.set_ylabel(r"$\log|\Psi_{\rm Jas}|$ (centred)")
    ax1.set_title(f"Three-body angular correlation\n"
                  f"(particle 2 orbits pair 0–1, $r_{{\\rm cross}}={r_cross}\\,a_{{\\rm ho}}$)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.axhline(0, color="k", lw=0.5, ls=":")

    # Panel 2: peak-to-peak amplitude of the angular variation (3-body signal)
    labels_k = list(results_by_key.keys())
    ctnn_amp  = [np.ptp(v[1]) for v in results_by_key.values()]
    pair_amp  = [np.ptp(v[2]) for v in results_by_key.values()]
    x_pos = np.arange(len(labels_k))
    w = 0.35
    ax2.bar(x_pos - w/2, ctnn_amp, w, label="CTNN",     color=[colors.get(l,"C4") for l in labels_k], alpha=0.9)
    ax2.bar(x_pos + w/2, pair_amp, w, label="Pairwise", color=[colors.get(l,"C4") for l in labels_k], alpha=0.4)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(labels_k)
    ax2.set_ylabel("Peak-to-peak $\\Delta\\log|\\Psi_{\\rm Jas}|$")
    ax2.set_title("3-body signal amplitude\n(0 = purely pairwise Jastrow)")
    ax2.legend()

    # Panel 3: ratio CTNN/Pairwise amplitude
    ratio = [c/p if p > 1e-6 else float("nan") for c, p in zip(ctnn_amp, pair_amp)]
    ax3.bar(x_pos, ratio, color=[colors.get(l,"C4") for l in labels_k], alpha=0.8)
    ax3.axhline(1., color="k", ls="--", lw=1)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(labels_k)
    ax3.set_ylabel("CTNN / Pairwise amplitude ratio")
    ax3.set_title("How many times more 3-body correlation\n does CTNN capture?")


# ═══════════════════════════════════════════════════════════════════
# FIG F — Message-passing ablation energy
# ═══════════════════════════════════════════════════════════════════

def importance_sampled_energy(x, C_occ, bf_net, jas_net, spin, chunk=64):
    """
    Compute importance-sampled local energy estimate over MCMC samples x.
    E ≈ mean(E_L) where E_L = T_loc + V_coul + V_trap.
    Uses only kinetic + potential — no Laplacian, just the variational energy.
    """
    p = config.get().as_dict(); p["device"] = str(DEVICE); p["torch_dtype"] = DTYPE
    N   = spin.shape[0]; ell = 1./math.sqrt(jas_net.omega)
    omega = jas_net.omega
    els = []
    for start in range(0, x.shape[0], chunk):
        with torch.enable_grad():
            xc = x[start:start+chunk].detach()
            B  = xc.shape[0]
            sb = spin.unsqueeze(0).expand(B, -1)
            # Kinetic via gradient trick: -½|∇logΨ|² (omits Laplacian — variational upper bound)
            xc_g = xc.requires_grad_(True)
            dx_g = bf_net(xc_g, spin=sb)
            xe_g = xc_g + dx_g
            _, la = slater_determinant_closed_shell(xe_g, C_occ, params=p, spin=spin, normalize=True)
            j_g  = jas_net(xc_g, spin=sb).squeeze(-1)
            lp   = la + j_g
            gl   = torch.autograd.grad(lp.sum(), xc_g)[0]  # (B,N,2)
            T_loc = 0.5 * (gl**2).sum(dim=(1,2))           # (B,) — |∇logΨ|² kinetic proxy
        # Potential (no grad needed)
        with torch.no_grad():
            V_trap = 0.5 * omega**2 * (xc**2).sum(dim=(1,2))
            diff   = xc.unsqueeze(2) - xc.unsqueeze(1)
            r_ij   = torch.sqrt((diff**2).sum(-1) + 1e-12)
            mask   = torch.triu(torch.ones(N, N, device=DEVICE, dtype=torch.bool), diagonal=1)
            V_coul = (1. / r_ij[:, mask]).sum(-1)
        E_L = T_loc.detach() + V_trap + V_coul
        els.append(E_L.detach())
    E_arr = torch.cat(els).cpu().numpy()
    return E_arr.mean(), E_arr.std() / math.sqrt(len(E_arr))


def run_ablation(x_mcmc, C_occ, bf_net, jas_net, spin, n_ablation_seeds=3):
    """
    Evaluate IS energy for:
      1. Full CTNN
      2. Ablated (message-passing zeroed)
      3. Ablated with random re-init of rho weights (random shuffled baseline)
    """
    E_full, se_full = importance_sampled_energy(x_mcmc, C_occ, bf_net, jas_net, spin)
    print(f"  Full CTNN:  E={E_full:.5f} ± {se_full:.5f}")

    results = {"Full CTNN": (E_full, se_full)}

    jas_ablated = copy.deepcopy(jas_net)
    _ablate_message_passing(jas_ablated)
    E_abl, se_abl = importance_sampled_energy(x_mcmc, C_occ, bf_net, jas_ablated, spin)
    print(f"  Ablated:    E={E_abl:.5f} ± {se_abl:.5f}  ΔE={E_abl-E_full:.5f}")
    results["Pairwise\n(ablated)"] = (E_abl, se_abl)

    # Random rho baseline: random orthogonal rho matrices (same param count, random inter-particle comm)
    random_Es = []
    for seed in range(n_ablation_seeds):
        torch.manual_seed(seed + 999)
        jas_rand = copy.deepcopy(jas_net)
        with torch.no_grad():
            for name, p in jas_rand.named_parameters():
                if "rho_v_to_e" in name or "rho_e_to_v" in name:
                    nn_init = torch.nn.init.orthogonal_
                    nn_init(p)
                    p.mul_(0.01)   # small random: slightly better than zero
        E_r, se_r = importance_sampled_energy(x_mcmc, C_occ, bf_net, jas_rand, spin)
        print(f"  Random rho s{seed}: E={E_r:.5f} ± {se_r:.5f}")
        random_Es.append((E_r, se_r))
    results["Random comm.\n(control)"] = (
        np.mean([e for e,_ in random_Es]),
        np.std([e for e,_ in random_Es]) + np.mean([s for _,s in random_Es])
    )
    return results


def make_fig_F(axes, ablation_by_key):
    """axes: 3 Axes; ablation_by_key: {omega_label: {method: (E, se)}}"""
    ax1, ax2, ax3 = axes
    colors_m = {"Full CTNN": "C0", "Pairwise\n(ablated)": "C3", "Random comm.\n(control)": "C4"}
    all_omegas = list(ablation_by_key.keys())

    # Panel 1: Energy per ω for each method
    x_pos = np.arange(len(all_omegas))
    methods = list(list(ablation_by_key.values())[0].keys())
    width = 0.25
    for i, method in enumerate(methods):
        Es  = [ablation_by_key[om].get(method, (float("nan"), 0))[0] for om in all_omegas]
        ses = [ablation_by_key[om].get(method, (float("nan"), 0))[1] for om in all_omegas]
        ax1.bar(x_pos + (i - 1) * width, Es, width, yerr=ses,
                label=method.replace("\n"," "),
                color=colors_m.get(method, f"C{i}"), alpha=0.85)
    ax1.set_xticks(x_pos); ax1.set_xticklabels(all_omegas)
    ax1.set_ylabel("IS Energy (Hartree)")
    ax1.set_title("Energy: full CTNN vs ablated message passing")
    ax1.legend(fontsize=8)

    # Panel 2: ΔE = E_ablated - E_full (energy cost of removing message passing)
    delta_abl  = []
    delta_rand = []
    for om in all_omegas:
        r = ablation_by_key[om]
        E0 = r["Full CTNN"][0]
        delta_abl.append(r.get("Pairwise\n(ablated)", (E0, 0))[0] - E0)
        delta_rand.append(r.get("Random comm.\n(control)", (E0, 0))[0] - E0)
    ax2.bar(x_pos - width/2, delta_abl,  width, label="Pairwise (ablated)",    color="C3", alpha=0.85)
    ax2.bar(x_pos + width/2, delta_rand, width, label="Random comm. (control)", color="C4", alpha=0.85)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(all_omegas)
    ax2.set_ylabel("ΔE = E_variant − E_CTNN (Hartree)")
    ax2.set_title("Energy cost of removing inter-particle communication\n(higher = message passing mattered more)")
    ax2.legend(fontsize=8)

    # Panel 3: Relative degradation in units of DMC errors
    # Approximate: each Hartree of error in E_full vs E_DMC; show ΔE as fraction of that
    E_full_vals = [ablation_by_key[om]["Full CTNN"][0] for om in all_omegas]
    rel = [de / abs(ef) * 100 if abs(ef) > 0 else 0
           for de, ef in zip(delta_abl, E_full_vals)]
    ax3.bar(x_pos, rel, color="C3", alpha=0.85)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(all_omegas)
    ax3.set_ylabel("ΔE / E_CTNN (×100 = %)")
    ax3.set_title("Relative energy degradation\nfrom removing message passing")


# ═══════════════════════════════════════════════════════════════════
# FIG G — Backflow correlation-hole geometry
# ═══════════════════════════════════════════════════════════════════

def backflow_geometry(x_mcmc, bf_net, spin):
    """
    For each config in x_mcmc:
      - Compute Δx_i = BF displacement for each electron
      - Find nearest neighbour j* of each i (minimum r_ij, j≠i)
      - Compute: cos_angle = (Δx_i · r_ij) / (|Δx_i| |r_ij|)  — positive = away, negative = toward
      - r_min_i = r_ij*   — distance to nearest neighbour
    Returns arrays: (r_min, cos_angle, |dx|, same_spin_flag)
    """
    B, N, d = x_mcmc.shape
    spin_np = spin.cpu().numpy()

    r_mins, cos_angles, dx_norms, same_spin_flags = [], [], [], []

    with torch.no_grad():
        for start in range(0, B, 64):
            xc = x_mcmc[start:start+64]
            Bc = xc.shape[0]
            sb = spin.unsqueeze(0).expand(Bc, -1)
            dx = bf_net(xc, spin=sb)    # (Bc, N, 2)

            diff = xc.unsqueeze(2) - xc.unsqueeze(1)   # (Bc,N,N,2): x_i - x_j
            r    = torch.sqrt((diff**2).sum(-1) + 1e-12) # (Bc,N,N)
            big  = torch.full_like(r, 1e10)
            eye  = torch.eye(N, device=DEVICE, dtype=torch.bool).unsqueeze(0)
            r_masked = r.clone(); r_masked[eye.expand(Bc,-1,-1)] = 1e10
            r_min_i, j_star = r_masked.min(dim=2)   # (Bc,N)

            # direction from i to nearest neighbour j*: r_ij = x_j - x_i = -diff[:,i,j*]
            j_idx = j_star.unsqueeze(-1).unsqueeze(-1).expand(Bc, N, 1, d)
            r_vec_to_nn = -torch.gather(diff, 2, j_idx).squeeze(2)  # (Bc,N,2), towards nn

            dx_norm  = dx.norm(dim=-1)                   # (Bc,N)
            r_nn_norm = r_vec_to_nn.norm(dim=-1) + 1e-12  # (Bc,N)

            cos = (dx * r_vec_to_nn).sum(-1) / (dx_norm.clamp(1e-12) * r_nn_norm)

            # Same-spin flag (same spin as nearest neighbour)
            spin_b = spin.cpu()  # (N,)
            ss = torch.zeros(Bc, N, dtype=torch.bool)
            for ii in range(N):
                j_ii = j_star[:, ii].cpu()  # (Bc,)
                ss[:, ii] = spin_b[ii] == spin_b[j_ii]

            r_mins.append(r_min_i.cpu().numpy())
            cos_angles.append(cos.cpu().numpy())
            dx_norms.append(dx_norm.cpu().numpy())
            same_spin_flags.append(ss.numpy())

    return (np.concatenate(r_mins, axis=0).ravel(),
            np.concatenate(cos_angles, axis=0).ravel(),
            np.concatenate(dx_norms, axis=0).ravel(),
            np.concatenate(same_spin_flags, axis=0).ravel())


def make_fig_G(axes, geo_by_key, bf_scale_by_key):
    ax1, ax2, ax3 = axes
    colors = {"ω=1.0": "C0", "ω=0.1": "C1", "ω=0.001": "C3"}

    # Panel 1: Histogram of cos(angle) — negative = toward nn (should peak < 0 for correlation hole)
    for label, (r_min, cos_a, dx_n, ss) in geo_by_key.items():
        c = colors.get(label, "C4")
        ax1.hist(cos_a, bins=50, density=True, alpha=0.5, color=c, label=label)
        ax1.axvline(np.median(cos_a), color=c, lw=1.5, ls="--")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("cos(Δx_i, r_to_nn)  [+1=away, -1=toward]")
    ax1.set_ylabel("Density")
    ax1.set_title("Backflow direction relative to nearest neighbour\n(negative = correlation hole deepening)")
    ax1.legend(fontsize=8)

    # Panel 2: |Δx_i| vs r_min — should increase as r_min decreases (stronger displacement near close pairs)
    for label, (r_min, cos_a, dx_n, ss) in geo_by_key.items():
        c = colors.get(label, "C4")
        # Bin r_min and plot mean |Δx| per bin
        bins = np.percentile(r_min, np.linspace(0, 100, 15))
        bins = np.unique(bins)
        means, centers = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (r_min >= lo) & (r_min < hi)
            if mask.sum() > 5:
                means.append(dx_n[mask].mean())
                centers.append((lo+hi)/2)
        ax2.plot(centers, means, "o-", color=c, label=label, lw=1.5, ms=4)
    ax2.set_xlabel(r"Nearest-neighbour distance $r_{\rm min}$ ($a_{\rm ho}$)")
    ax2.set_ylabel(r"Mean $|\Delta\mathbf{x}_i|$ ($a_{\rm ho}$)")
    ax2.set_title("Backflow displacement magnitude vs proximity\n(slope = correlation hole response strength)")
    ax2.legend(fontsize=8)

    # Panel 3: cos(angle) split by same-spin vs opposite-spin nearest neighbour
    for label, (r_min, cos_a, dx_n, ss) in geo_by_key.items():
        c = colors.get(label, "C4")
        med_same  = np.median(cos_a[ss == 1]) if (ss == 1).any() else float("nan")
        med_opp   = np.median(cos_a[ss == 0]) if (ss == 0).any() else float("nan")
        print(f"  {label}: median cos — same-spin={med_same:.3f}, opp-spin={med_opp:.3f}  "
              f"(bf_scale={bf_scale_by_key.get(label,'?'):.4f})")
    # Bar plot: median cos for same vs opposite spin, per omega
    labels_k = list(geo_by_key.keys())
    x_pos = np.arange(len(labels_k)); w = 0.35
    med_same = [np.median(geo_by_key[l][1][geo_by_key[l][3]==1])
                if (geo_by_key[l][3]==1).any() else 0 for l in labels_k]
    med_opp  = [np.median(geo_by_key[l][1][geo_by_key[l][3]==0])
                if (geo_by_key[l][3]==0).any() else 0 for l in labels_k]
    ax3.bar(x_pos - w/2, med_same, w, label="Same-spin NN",     color="C0", alpha=0.85)
    ax3.bar(x_pos + w/2, med_opp,  w, label="Opposite-spin NN", color="C3", alpha=0.85)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(labels_k)
    ax3.set_ylabel("Median cos(Δx_i, r_to_nn)")
    ax3.set_title("Backflow direction: same-spin vs opp-spin nearest neighbour\n(asymmetry = spin-aware correlation hole)")
    ax3.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main(figs_to_run):
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}\n")

    print("Loading checkpoints...")
    ckpts = {}
    for key, path in CKPTS.items():
        if path.exists():
            ckpts[key] = load_checkpoint(path)
            _, _, n, w = ckpts[key]
            print(f"  {key}: N={n}, ω={w}")
        else:
            print(f"  {key}: NOT FOUND")

    n_figs = len(figs_to_run)
    fig = plt.figure(figsize=(18, 5.5 * n_figs))
    gs  = gridspec.GridSpec(n_figs, 3, figure=fig, hspace=0.65, wspace=0.4)
    fig_idx = 0
    npz = {}

    # ════════════════ FIG E ════════════════
    if "E" in figs_to_run:
        print("\n── Figure E: Three-body angular correlation ──")
        results_E = {}
        for key, label in [("w1", "ω=1.0"), ("w01", "ω=0.1"), ("w0001", "ω=0.001")]:
            if key not in ckpts:
                print(f"  Skipping {key}"); continue
            bf_net, jas_net, n_elec, omega = ckpts[key]
            C_occ = _setup(n_elec, omega)
            spin  = _spin(n_elec)
            print(f"  Scanning {label}...")
            # r_cross in a_ho units (scale with oscillator length)
            r_cross = 1.0  # 1 a_ho
            r01 = 1.0
            theta, lp_ctnn, lp_pair = three_body_scan(
                jas_net, spin, r01=r01, r_cross=r_cross,
                n_theta=120, omega=omega, C_occ=C_occ, bf_net=bf_net)
            results_E[label] = (theta, lp_ctnn, lp_pair)
            ctnn_amp = np.ptp(lp_ctnn); pair_amp = np.ptp(lp_pair)
            print(f"    CTNN amp={ctnn_amp:.4f}, Pairwise amp={pair_amp:.4f}, "
                  f"ratio={ctnn_amp/pair_amp if pair_amp>1e-6 else 'inf':.1f}×")
            npz[f"fig_E_{label}_theta"]  = theta
            npz[f"fig_E_{label}_ctnn"]   = lp_ctnn
            npz[f"fig_E_{label}_pair"]   = lp_pair

        if results_E:
            ax_row = [fig.add_subplot(gs[fig_idx, j]) for j in range(3)]
            make_fig_E(ax_row, results_E, r_cross=1.0)
        fig_idx += 1

    # ════════════════ FIG F ════════════════
    if "F" in figs_to_run:
        print("\n── Figure F: Message-passing ablation energy ──")
        ablation_F = {}
        for key, label in [("w1", "ω=1.0"), ("w01", "ω=0.1"), ("w0001", "ω=0.001")]:
            if key not in ckpts:
                print(f"  Skipping {key}"); continue
            bf_net, jas_net, n_elec, omega = ckpts[key]
            C_occ = _setup(n_elec, omega)
            spin  = _spin(n_elec)
            print(f"  Sampling {label}...")
            x_mcmc = mcmc(C_occ, bf_net, jas_net, spin, 500, omega, burn=300)
            print(f"  Running ablation...")
            ablation_F[label] = run_ablation(x_mcmc, C_occ, bf_net, jas_net, spin, n_ablation_seeds=3)

        if ablation_F:
            ax_row = [fig.add_subplot(gs[fig_idx, j]) for j in range(3)]
            make_fig_F(ax_row, ablation_F)
        fig_idx += 1

    # ════════════════ FIG G ════════════════
    if "G" in figs_to_run:
        print("\n── Figure G: Backflow correlation-hole geometry ──")
        geo_G = {}
        bf_scale_G = {}
        for key, label in [("w1", "ω=1.0"), ("w01", "ω=0.1"), ("w0001", "ω=0.001")]:
            if key not in ckpts:
                print(f"  Skipping {key}"); continue
            bf_net, jas_net, n_elec, omega = ckpts[key]
            # BF scale
            bf_sc = float(F.softplus(
                next(p for n, p in bf_net.named_parameters() if "bf_scale_raw" in n)))
            bf_scale_G[label] = bf_sc
            print(f"  {label}: bf_scale={bf_sc:.4f}")
            C_occ = _setup(n_elec, omega)
            spin  = _spin(n_elec)
            x_mcmc = mcmc(C_occ, bf_net, jas_net, spin, 1500, omega, burn=350)
            print(f"  Computing backflow geometry...")
            geo_G[label] = backflow_geometry(x_mcmc, bf_net, spin)
            r_min, cos_a, dx_n, ss = geo_G[label]
            frac_away = (cos_a > 0).mean()  # positive cos = moving away
            print(f"    frac moving AWAY from nn: {frac_away:.2%}")
            print(f"    median |Δx|: {np.median(dx_n):.4f} a_ho")
            npz[f"fig_G_{label}_rmin"]  = r_min
            npz[f"fig_G_{label}_cos"]   = cos_a
            npz[f"fig_G_{label}_dxn"]   = dx_n
            npz[f"fig_G_{label}_ss"]    = ss.astype(np.float32)

        if geo_G:
            ax_row = [fig.add_subplot(gs[fig_idx, j]) for j in range(3)]
            make_fig_G(ax_row, geo_G, bf_scale_G)
        fig_idx += 1

    # Save
    pdf = OUT_DIR / "ctnn_pairwise_diagnostics.pdf"
    png = OUT_DIR / "ctnn_pairwise_diagnostics.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=150)
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {pdf}")
    if npz:
        np.savez(OUT_DIR / "ctnn_pairwise_data.npz", **npz)
        print(f"Saved data: {OUT_DIR / 'ctnn_pairwise_data.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", default=["E","F","G"], choices=["E","F","G"])
    args = ap.parse_args()
    main(set(args.figs))
