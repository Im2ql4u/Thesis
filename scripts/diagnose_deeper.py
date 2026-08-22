#!/usr/bin/env python3
"""
Deeper CTNN Diagnostics
========================
Four measurements completing the diagnostic picture:

  Fig E  — Environment-conditional gradient variance (proper three-body test).
            Bin pair (0,1) by distance r₀₁; within each bin, compute variance
            of ∂logΨ_Jas/∂r₀₁ across configs with the SAME r₀₁ but different
            environments. CTNN: high intra-bin variance. Pairwise ablated: near-zero.

  Fig H  — Training dynamics from saved jsonl logs.
            var_EL and energy convergence for REINFORCE (p3c) vs FD-Colloc (p3b).
            No GPU needed — reads consistency_campaign phase3 epoch files.

  Fig I  — Kinetic / potential energy decomposition in the message-passing ablation.
            Shows WHERE the 22-30% energy gain of CTNN over pairwise comes from.

  Fig J  — Backflow displacement vs classical force alignment.
            For each electron: cos(Δxᵢ, Fᵢ) where Fᵢ = −∇ᵢV(x) is the classical
            Coulomb+trap restoring force. Tests whether BF is a force corrector
            (positive cos) or orbital corrector (negative cos at Wigner regime).

Usage:
  CUDA_MANUAL_DEVICE=0 python3.11 scripts/diagnose_deeper.py
  CUDA_MANUAL_DEVICE=0 python3.11 scripts/diagnose_deeper.py --figs E H I J
"""
from __future__ import annotations

import argparse
import copy
import json
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

DIM   = 2
DTYPE = torch.float64
DEVICE = torch.device(f"cuda:{os.environ.get('CUDA_MANUAL_DEVICE', 0)}")

CKPT_DIR = REPO / "results" / "arch_colloc"
OUT_DIR  = REPO / "results" / "figures" / "architecture_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = {
    "w1":    CKPT_DIR / "bf_ctnn_vcycle.pt",
    "w01":   CKPT_DIR / "p3c_adam_n6w01_best.pt",
    "w0001": CKPT_DIR / "n6x2_adam_w0001_best.pt",
}

LOGS = {
    "p3b_fdcolloc": REPO / "outputs" / "consistency_campaign" / "phase3" / "p3b_fdcolloc_n6w01_epochs.jsonl",
    "p3c_reinforce": REPO / "outputs" / "consistency_campaign" / "phase3" / "p3c_adam_n6w01_epochs.jsonl",
}


# ─────────────── model loading ───────────────

def _infer_bf(s):
    h  = s["node_embed.weight"].shape[0]
    mh = s["edge_embed.0.weight"].shape[0]
    ml = sum(1 for k in s if k.startswith("edge_update.") and k.endswith(".weight"))
    nl = sum(1 for k in s if k.startswith("node_update.") and k.endswith(".weight"))
    return dict(d=DIM, hidden=h, msg_hidden=mh, msg_layers=ml, layers=nl,
                act="silu", aggregation="sum", use_spin=True,
                same_spin_only=False, out_bound="tanh",
                bf_scale_init=0.05, zero_init_last=True)


def _infer_jas(s, n, omega):
    nh = s["node_embed.weight"].shape[0]
    eh = s["edge_embed.0.weight"].shape[0]
    bh = s["node_down.weight"].shape[0]
    nd = sum(1 for k in s if k.startswith("rho_v_to_e_down."))
    nu = sum(1 for k in s if k.startswith("rho_v_to_e_up."))
    ml = sum(1 for k in s if k.startswith("edge_updates_down.0.") and k.endswith(".weight"))
    nl = sum(1 for k in s if k.startswith("node_updates_down.0.") and k.endswith(".weight"))
    rl = sum(1 for k in s if k.startswith("f_head.") and k.endswith(".weight")) - 1
    rh = s["f_head.0.weight"].shape[0]
    return dict(n_particles=n, d=DIM, omega=omega, node_hidden=nh, edge_hidden=eh,
                bottleneck_hidden=bh, n_down=nd, n_up=nu, msg_layers=ml, node_layers=nl,
                readout_hidden=rh, readout_layers=rl, act="silu", aggregation="sum", use_spin=True)


def load_ckpt(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    n  = int(ck.get("n_elec") or ck.get("bf_config", {}).get("n_elec") or 6)
    w  = float(ck.get("omega") or ck.get("bf_config", {}).get("omega") or 1.0)
    bfc = ck.get("bf_config") or _infer_bf(ck["bf_state"])
    bfc["omega"] = w
    bf = CTNNBackflowNet(**bfc).to(DEVICE).to(DTYPE)
    bf.load_state_dict(ck["bf_state"]); bf.eval()
    jc = _infer_jas(ck["jas_state"], n, w)
    jas = CTNNJastrowVCycle(**jc).to(DEVICE).to(DTYPE)
    jas.load_state_dict(ck["jas_state"]); jas.eval()
    return bf, jas, n, w


def _spin(n):
    return torch.cat([torch.zeros(n//2, dtype=torch.long),
                      torch.ones(n-n//2, dtype=torch.long)]).to(DEVICE)


def _setup(n, omega):
    n_occ = n//2; nx = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
    ny = nx; L = max(8., 3./math.sqrt(omega))
    config.update(n_particles=n, omega=omega, d=DIM, basis="cart",
                  nx=nx, ny=ny, L=L, n_grid=80, device=str(DEVICE), dtype="float64", seed=42)
    energies = sorted([(omega*(ix+iy+1), ix, iy) for ix in range(nx) for iy in range(ny)])
    C = np.zeros((nx*ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]; C[ix*ny+iy, k] = 1.
    return torch.tensor(C, dtype=DTYPE, device=DEVICE)


def _logpsi(x, C, bf, jas, spin):
    B = x.shape[0]; sb = spin.unsqueeze(0).expand(B,-1)
    dx = bf(x, spin=sb)
    p = config.get().as_dict(); p["device"]=str(DEVICE); p["torch_dtype"]=DTYPE
    _, la = slater_determinant_closed_shell(x+dx, C, params=p, spin=spin, normalize=True)
    j = jas(x, spin=sb).squeeze(-1)
    return la + j


def _logpsi_jas(x, jas, spin):
    B = x.shape[0]; sb = spin.unsqueeze(0).expand(B,-1)
    return jas(x, spin=sb).squeeze(-1)


def mcmc(C, bf, jas, spin, n, omega, burn=400, step=0.15):
    N = spin.shape[0]; ell = 1./math.sqrt(omega)
    x = torch.randn(n, N, DIM, device=DEVICE, dtype=DTYPE) * ell
    acc = 0
    with torch.no_grad():
        lp = 2.*_logpsi(x, C, bf, jas, spin)
        for _ in range(burn):
            xp = x + torch.randn_like(x)*(step*ell)
            lpp = 2.*_logpsi(xp, C, bf, jas, spin)
            a = torch.rand(n, device=DEVICE, dtype=DTYPE).log() < (lpp-lp)
            x = torch.where(a.view(-1,1,1), xp, x); lp = torch.where(a, lpp, lp)
            acc += a.float().mean().item()
    print(f"  MCMC acc={acc/burn:.2f}  |x|={x.norm(dim=-1).mean():.2f}")
    return x


def _ablate_mp(jas):
    jas2 = copy.deepcopy(jas)
    with torch.no_grad():
        for name, p in jas2.named_parameters():
            if "rho_v_to_e" in name or "rho_e_to_v" in name:
                p.zero_()
    return jas2


# ═══════════════════════════════════════════════════════════════════
# FIG E — Environment-conditional gradient variance (proper three-body)
# ═══════════════════════════════════════════════════════════════════

def env_conditional_variance(x_mcmc, jas, jas_pair, spin, pair=(0,1), n_bins=18, chunk=32):
    """
    For the pair (i,j), compute ∂logΨ_Jas/∂r_ij for every config in x_mcmc.
    Bin configs by r_ij. Within each bin, compute variance of the gradient.
    CTNN: intra-bin variance >> 0 (environment shifts the response).
    Pairwise: intra-bin variance ≈ 0 (response determined solely by r_ij).
    Returns: bin_centers, var_ctnn, var_pair, mean_ctnn, mean_pair
    """
    i, j = pair
    r_vals, g_ctnn, g_pair = [], [], []

    for start in range(0, x_mcmc.shape[0], chunk):
        xc = x_mcmc[start:start+chunk].detach()
        B  = xc.shape[0]

        diff = xc[:, i] - xc[:, j]                  # (B, 2)
        r_ij = diff.norm(dim=-1)                     # (B,)
        r_vals.append(r_ij.cpu().numpy())

        for net, store in [(jas, g_ctnn), (jas_pair, g_pair)]:
            xi = xc.clone().requires_grad_(True)
            lp = _logpsi_jas(xi, net, spin)
            gi = torch.autograd.grad(lp.sum(), xi)[0]  # (B,N,2)
            # Gradient wrt pair displacement direction (dot with unit vector)
            d_hat = diff / (r_ij.unsqueeze(-1) + 1e-12)
            grad_r = (gi[:, i] * d_hat).sum(-1) - (gi[:, j] * d_hat).sum(-1)
            store.append(grad_r.detach().cpu().numpy())

    r_all   = np.concatenate(r_vals)
    gc_all  = np.concatenate(g_ctnn)
    gp_all  = np.concatenate(g_pair)

    # Bin by r_ij
    r_min, r_max = r_all.min(), r_all.max()
    edges = np.linspace(r_min, r_max, n_bins+1)
    ctnn_var, pair_var, ctnn_mu, pair_mu, centers, counts = [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (r_all >= lo) & (r_all < hi)
        if mask.sum() < 5:
            continue
        centers.append((lo+hi)/2)
        counts.append(mask.sum())
        ctnn_var.append(gc_all[mask].var())
        pair_var.append(gp_all[mask].var())
        ctnn_mu.append(np.abs(gc_all[mask]).mean())
        pair_mu.append(np.abs(gp_all[mask]).mean())

    return (np.array(centers), np.array(ctnn_var), np.array(pair_var),
            np.array(ctnn_mu), np.array(pair_mu), np.array(counts))


def make_fig_E(axes, results_E, omega_labels):
    ax1, ax2, ax3 = axes
    colors = {"ω=1.0": "C0", "ω=0.1": "C1", "ω=0.001": "C3"}

    # Panel 1: intra-bin variance of ∂logΨ/∂r₀₁ — CTNN vs pairwise
    for label in omega_labels:
        if label not in results_E:
            continue
        r, vc, vp, mc, mp, cnt = results_E[label]
        c = colors.get(label, "C4")
        ax1.plot(r, vc, "o-", color=c, lw=1.5, ms=4, label=f"CTNN {label}")
        ax1.plot(r, vp, "s--", color=c, lw=1, ms=3, alpha=0.5, label=f"Pairwise {label}")
    ax1.set_xlabel(r"Pair distance $r_{01}$ ($a_{\rm ho}$)")
    ax1.set_ylabel(r"Intra-bin Var$[\partial\log|\Psi_J|/\partial r_{01}]$")
    ax1.set_title("Environment-conditional gradient variance\n"
                  "(non-zero = three-body/many-body response)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_yscale("log")

    # Panel 2: ratio Var_CTNN / Var_pair
    for label in omega_labels:
        if label not in results_E:
            continue
        r, vc, vp, mc, mp, cnt = results_E[label]
        ratio = vc / (vp + 1e-12)
        ax2.plot(r, ratio, "o-", color=colors.get(label, "C4"), lw=1.5, ms=4, label=label)
    ax2.axhline(1., color="k", ls="--", lw=0.8)
    ax2.set_xlabel(r"Pair distance $r_{01}$ ($a_{\rm ho}$)")
    ax2.set_ylabel("CTNN / Pairwise variance ratio")
    ax2.set_title("How many × more environment-sensitive is CTNN\nvs pairwise Jastrow?")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")

    # Panel 3: mean absolute gradient (shows that CTNN and pairwise have similar MEAN — difference is in variance)
    for label in omega_labels:
        if label not in results_E:
            continue
        r, vc, vp, mc, mp, cnt = results_E[label]
        c = colors.get(label, "C4")
        ax3.plot(r, mc, "o-", color=c, lw=1.5, ms=4, label=f"CTNN {label}")
        ax3.plot(r, mp, "s--", color=c, lw=1, ms=3, alpha=0.5, label=f"Pairwise {label}")
    ax3.set_xlabel(r"Pair distance $r_{01}$ ($a_{\rm ho}$)")
    ax3.set_ylabel(r"Mean $|\partial\log|\Psi_J|/\partial r_{01}|$")
    ax3.set_title("Mean gradient magnitude vs pair distance\n"
                  "(CTNN ≈ Pairwise in mean, differs in variance)")
    ax3.legend(fontsize=7, ncol=2)
    ax3.set_yscale("log")


# ═══════════════════════════════════════════════════════════════════
# FIG H — Training dynamics from jsonl logs
# ═══════════════════════════════════════════════════════════════════

def load_epochs(path):
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def make_fig_H(axes, logs):
    ax1, ax2, ax3 = axes
    colors = {"p3b_fdcolloc": "C1", "p3c_reinforce": "C0"}
    labels = {"p3b_fdcolloc": "FD-Colloc (p3b)", "p3c_reinforce": "REINFORCE (p3c)"}

    for key, recs in logs.items():
        if not recs:
            continue
        ep      = np.array([r["ep"]     for r in recs])
        E       = np.array([r["E"]      for r in recs])
        var_EL  = np.array([r["var_EL"] for r in recs])
        ess     = np.array([r["ess"]    for r in recs])
        c       = colors.get(key, "C4")
        lbl     = labels.get(key, key)

        # Smooth with rolling window
        w = 10
        E_sm   = np.convolve(E,      np.ones(w)/w, mode="valid")
        ve_sm  = np.convolve(var_EL, np.ones(w)/w, mode="valid")
        ep_sm  = ep[w-1:]

        ax1.plot(ep_sm, E_sm,  color=c, lw=1.5, label=lbl)
        ax2.plot(ep_sm, ve_sm, color=c, lw=1.5, label=lbl)
        ax3.plot(ep, ess, color=c, lw=0.8, alpha=0.5)
        ax3.plot(ep_sm, np.convolve(ess, np.ones(w)/w, mode="valid"),
                 color=c, lw=1.5, label=lbl)

    # DMC reference for N=6 ω=0.1
    e_dmc = 3.55385
    ax1.axhline(e_dmc, color="k", ls="--", lw=0.8, label="DMC ref")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("E (Hartree)")
    ax1.set_title("Energy convergence: REINFORCE vs FD-Colloc\n(N=6, ω=0.1)")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel(r"Var$[E_L]$ (Hartree²)")
    ax2.set_title("Local-energy variance during training\n"
                  "(proxy for gradient noise — lower = more stable)")
    ax2.legend(fontsize=8); ax2.set_yscale("log")

    ax3.set_xlabel("Epoch"); ax3.set_ylabel("ESS")
    ax3.set_title("Effective Sample Size during training")
    ax3.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════
# FIG I — Kinetic / potential decomposition in ablation
# ═══════════════════════════════════════════════════════════════════

def energy_decomposition(x_mcmc, C, bf, jas, jas_pair, spin, chunk=32):
    """Returns dict: {label: (E, T, V_trap, V_coul)} for CTNN and pairwise."""
    omega = jas.omega
    results = {}
    for label, net in [("CTNN", jas), ("Pairwise", jas_pair)]:
        T_list, Vtrap_list, Vcoul_list = [], [], []
        for start in range(0, x_mcmc.shape[0], chunk):
            xc = x_mcmc[start:start+chunk].detach()
            B  = xc.shape[0]; N = spin.shape[0]
            with torch.enable_grad():
                xi = xc.requires_grad_(True)
                sb = spin.unsqueeze(0).expand(B,-1)
                dx = bf(xi, spin=sb)
                xe = xi + dx
                p  = config.get().as_dict(); p["device"]=str(DEVICE); p["torch_dtype"]=DTYPE
                _, la = slater_determinant_closed_shell(xe, C, params=p, spin=spin, normalize=True)
                j  = net(xi, spin=sb).squeeze(-1)
                lp = la + j
                gl = torch.autograd.grad(lp.sum(), xi)[0]
            T    = 0.5*(gl.detach()**2).sum(dim=(1,2))
            Vtr  = 0.5*omega**2*(xc**2).sum(dim=(1,2))
            diff = xc.unsqueeze(2)-xc.unsqueeze(1)
            r_ij = torch.sqrt((diff**2).sum(-1)+1e-12)
            mask = torch.triu(torch.ones(N,N,device=DEVICE,dtype=torch.bool), diagonal=1)
            Vc   = (1./r_ij[:,mask]).sum(-1)
            T_list.append(T.cpu()); Vtrap_list.append(Vtr.cpu()); Vcoul_list.append(Vc.cpu())

        T_arr   = torch.cat(T_list).detach().numpy()
        Vtr_arr = torch.cat(Vtrap_list).detach().numpy()
        Vc_arr  = torch.cat(Vcoul_list).detach().numpy()
        E_arr   = T_arr + Vtr_arr + Vc_arr
        results[label] = (E_arr.mean(), T_arr.mean(), Vtr_arr.mean(), Vc_arr.mean(),
                          E_arr.std()/math.sqrt(len(E_arr)))
        print(f"  {label}: E={E_arr.mean():.4f}  T={T_arr.mean():.4f}  "
              f"V_trap={Vtr_arr.mean():.4f}  V_coul={Vc_arr.mean():.4f}")
    return results


def make_fig_I(axes, decomp_by_omega):
    ax1, ax2, ax3 = axes
    colors = {"ω=1.0": "C0", "ω=0.1": "C1", "ω=0.001": "C3"}
    all_labels = list(decomp_by_omega.keys())
    x_pos = np.arange(len(all_labels))
    w = 0.3

    # Panel 1: ΔE breakdown (CTNN vs pairwise)
    delta_T    = [decomp_by_omega[l]["Pairwise"][1] - decomp_by_omega[l]["CTNN"][1] for l in all_labels]
    delta_Vtr  = [decomp_by_omega[l]["Pairwise"][2] - decomp_by_omega[l]["CTNN"][2] for l in all_labels]
    delta_Vc   = [decomp_by_omega[l]["Pairwise"][3] - decomp_by_omega[l]["CTNN"][3] for l in all_labels]
    ax1.bar(x_pos - w, delta_T,   w, label="ΔT (kinetic)", color="C0", alpha=0.85)
    ax1.bar(x_pos,     delta_Vtr, w, label="ΔV_trap",      color="C2", alpha=0.85)
    ax1.bar(x_pos + w, delta_Vc,  w, label="ΔV_Coulomb",   color="C3", alpha=0.85)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x_pos); ax1.set_xticklabels(all_labels)
    ax1.set_ylabel("ΔE_component = E_pairwise − E_CTNN (Hartree)")
    ax1.set_title("Where does message passing save energy?\n"
                  "(positive = CTNN has lower this component)")
    ax1.legend(fontsize=8)

    # Panel 2: fractional contribution to ΔE_total
    delta_E = [decomp_by_omega[l]["Pairwise"][0] - decomp_by_omega[l]["CTNN"][0] for l in all_labels]
    frac_T  = [dt/de if abs(de)>1e-8 else 0 for dt, de in zip(delta_T, delta_E)]
    frac_Vt = [dv/de if abs(de)>1e-8 else 0 for dv, de in zip(delta_Vtr, delta_E)]
    frac_Vc = [dv/de if abs(de)>1e-8 else 0 for dv, de in zip(delta_Vc, delta_E)]
    bars_T  = ax2.bar(x_pos - w, frac_T,  w, label="ΔT fraction",  color="C0", alpha=0.85)
    bars_Vt = ax2.bar(x_pos,     frac_Vt, w, label="ΔV_tr fraction", color="C2", alpha=0.85)
    bars_Vc = ax2.bar(x_pos + w, frac_Vc, w, label="ΔV_C fraction",  color="C3", alpha=0.85)
    ax2.axhline(0, color="k", lw=0.8); ax2.axhline(1, color="k", ls=":", lw=0.8)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(all_labels)
    ax2.set_ylabel("Fraction of total ΔE")
    ax2.set_title("Fractional contribution to energy gain\nfrom message passing")
    ax2.legend(fontsize=8)

    # Panel 3: actual E, T, V values for CTNN and pairwise side by side
    for li, label in enumerate(all_labels):
        r = decomp_by_omega[label]
        c = colors.get(label, "C4")
        E_c, T_c, Vt_c, Vc_c = r["CTNN"][:4]
        E_p, T_p, Vt_p, Vc_p = r["Pairwise"][:4]
        ax3.annotate(f"{label}\nCTNN E={E_c:.3f}", xy=(li,0), fontsize=7, ha="center")
    ax3.bar(x_pos - w/2, [decomp_by_omega[l]["CTNN"][0]    for l in all_labels], w,
            label="CTNN E", color="C0", alpha=0.85)
    ax3.bar(x_pos + w/2, [decomp_by_omega[l]["Pairwise"][0] for l in all_labels], w,
            label="Pairwise E", color="C3", alpha=0.85)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(all_labels)
    ax3.set_ylabel("IS Energy (Hartree)")
    ax3.set_title("Total energy: CTNN vs Pairwise")
    ax3.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════
# FIG J — Backflow vs classical force alignment
# ═══════════════════════════════════════════════════════════════════

def classical_force(x, omega):
    """
    F_i = −∇_i V(x) = −ω²x_i + Σ_{j≠i} (x_i−x_j)/|x_i−x_j|³
    Returns (B, N, 2)
    """
    B, N, d = x.shape
    trap_force = -omega**2 * x                             # (B,N,2)
    diff = x.unsqueeze(2) - x.unsqueeze(1)                # (B,N,N,2)
    r    = torch.sqrt((diff**2).sum(-1, keepdim=True) + 1e-12)  # (B,N,N,1)
    r3   = r**3
    eye  = torch.eye(N, device=x.device, dtype=x.dtype).view(1,N,N,1)
    coulomb_force = (diff / (r3 + eye)).sum(2)            # (B,N,2)  — sum j≠i
    return trap_force + coulomb_force


def bf_force_alignment(x_mcmc, bf, spin, omega, chunk=64):
    """
    For each electron in each config:
      - Compute Δxᵢ from backflow
      - Compute classical force Fᵢ
      - Return arrays: cos(Δxᵢ, Fᵢ), |Δxᵢ|, |Fᵢ|, r_min_i, same_spin_flag
    """
    N = spin.shape[0]
    spin_np = spin.cpu().numpy()
    all_cos, all_dx, all_F, all_rmin, all_ss, all_cos_trap, all_cos_coul = [], [], [], [], [], [], []

    with torch.no_grad():
        for start in range(0, x_mcmc.shape[0], chunk):
            xc = x_mcmc[start:start+chunk]
            Bc = xc.shape[0]
            sb = spin.unsqueeze(0).expand(Bc,-1)
            dx = bf(xc, spin=sb)                          # (Bc,N,2)

            F_full  = classical_force(xc, omega)          # (Bc,N,2) full force
            F_trap  = -omega**2 * xc                      # trap component
            F_coul  = F_full - F_trap                     # Coulomb component

            dx_n = dx.norm(dim=-1, keepdim=True).clamp(1e-12)
            F_n  = F_full.norm(dim=-1, keepdim=True).clamp(1e-12)
            Ft_n = F_trap.norm(dim=-1, keepdim=True).clamp(1e-12)
            Fc_n = F_coul.norm(dim=-1, keepdim=True).clamp(1e-12)

            cos_full = ((dx / dx_n) * (F_full / F_n)).sum(-1)      # (Bc,N)
            cos_trap = ((dx / dx_n) * (F_trap / Ft_n)).sum(-1)
            cos_coul = ((dx / dx_n) * (F_coul / Fc_n)).sum(-1)

            # Nearest neighbour
            diff = xc.unsqueeze(2)-xc.unsqueeze(1)
            r    = torch.sqrt((diff**2).sum(-1)+1e-12)
            big  = torch.full_like(r, 1e10)
            eye  = torch.eye(N, device=DEVICE, dtype=torch.bool).unsqueeze(0)
            r_m  = r.clone(); r_m[eye.expand(Bc,-1,-1)] = 1e10
            rmin, jstar = r_m.min(2)

            # Same-spin flag
            ss = np.zeros((Bc, N), dtype=bool)
            for ii in range(N):
                j_ii = jstar[:, ii].cpu().numpy()
                ss[:, ii] = spin_np[ii] == spin_np[j_ii]

            all_cos.append(cos_full.cpu().numpy())
            all_cos_trap.append(cos_trap.cpu().numpy())
            all_cos_coul.append(cos_coul.cpu().numpy())
            all_dx.append(dx.norm(dim=-1).cpu().numpy())
            all_F.append(F_full.norm(dim=-1).cpu().numpy())
            all_rmin.append(rmin.cpu().numpy())
            all_ss.append(ss)

    return {
        "cos":      np.concatenate(all_cos).ravel(),
        "cos_trap": np.concatenate(all_cos_trap).ravel(),
        "cos_coul": np.concatenate(all_cos_coul).ravel(),
        "dx_norm":  np.concatenate(all_dx).ravel(),
        "F_norm":   np.concatenate(all_F).ravel(),
        "r_min":    np.concatenate(all_rmin).ravel(),
        "same_spin":np.concatenate(all_ss).ravel(),
    }


def make_fig_J(axes, geo_by_label, e_dmc_by_label=None):
    ax1, ax2, ax3 = axes
    colors = {"ω=1.0": "C0", "ω=0.1": "C1", "ω=0.001": "C3"}

    # Panel 1: cos(Δxᵢ, F_full) decomposed by force type — full, trap, Coulomb
    labels_k = list(geo_by_label.keys())
    x_pos = np.arange(len(labels_k)); w = 0.25
    med_full = [np.median(geo_by_label[l]["cos"])      for l in labels_k]
    med_trap = [np.median(geo_by_label[l]["cos_trap"]) for l in labels_k]
    med_coul = [np.median(geo_by_label[l]["cos_coul"]) for l in labels_k]
    ax1.bar(x_pos - w,   med_full, w, label="Full force",   color="C0", alpha=0.85)
    ax1.bar(x_pos,       med_trap, w, label="Trap force",   color="C2", alpha=0.85)
    ax1.bar(x_pos + w,   med_coul, w, label="Coulomb force",color="C3", alpha=0.85)
    ax1.axhline(0, color="k", lw=0.8); ax1.axhline(1, color="k", ls=":", lw=0.5)
    ax1.set_xticks(x_pos); ax1.set_xticklabels(labels_k)
    ax1.set_ylabel("Median cos(Δxᵢ, force)")
    ax1.set_title("Backflow direction vs classical force components\n"
                  "(+1=aligned with force, −1=opposite)")
    ax1.legend(fontsize=8)

    # Panel 2: same-spin vs opp-spin split for full force
    med_ss   = [np.median(geo_by_label[l]["cos"][geo_by_label[l]["same_spin"]==1]) for l in labels_k]
    med_opp  = [np.median(geo_by_label[l]["cos"][geo_by_label[l]["same_spin"]==0]) for l in labels_k]
    ax2.bar(x_pos - w/2, med_ss,  w, label="Same-spin NN",     color="C0", alpha=0.85)
    ax2.bar(x_pos + w/2, med_opp, w, label="Opp-spin NN",      color="C3", alpha=0.85)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(labels_k)
    ax2.set_ylabel("Median cos(Δxᵢ, F_full)")
    ax2.set_title("Force alignment split by nearest-neighbour spin\n"
                  "(BF treats same-spin and opp-spin differently)")
    ax2.legend(fontsize=8)

    # Panel 3: |Δxᵢ| × cos vs r_min (effective work done against/with force)
    for label, geo in geo_by_label.items():
        c = colors.get(label, "C4")
        # Bin r_min
        rmin = geo["r_min"]; dx_n = geo["dx_norm"]; cos = geo["cos"]
        effective_work = dx_n * cos   # |Δx|·cos(θ) = component along force
        bins = np.percentile(rmin, np.linspace(0, 100, 14))
        bins = np.unique(bins)
        c_r, c_w = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (rmin >= lo) & (rmin < hi)
            if mask.sum() >= 5:
                c_r.append((lo+hi)/2)
                c_w.append(effective_work[mask].mean())
        ax3.plot(c_r, c_w, "o-", color=c, lw=1.5, ms=4, label=label)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xlabel(r"$r_{\min}$ ($a_{\rm ho}$)")
    ax3.set_ylabel(r"$\langle|\Delta\mathbf{x}|\cos\theta\rangle$ ($a_{\rm ho}$)")
    ax3.set_title("Effective work done by BF along force direction\nvs nearest-neighbour proximity")
    ax3.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main(figs_to_run):
    print(f"Device: {DEVICE}")

    print("\nLoading checkpoints...")
    ckpts = {}
    for key, path in CKPTS.items():
        if path.exists():
            ckpts[key] = load_ckpt(path)
            _, _, n, w = ckpts[key]
            print(f"  {key}: N={n}, ω={w}")
        else:
            print(f"  {key}: NOT FOUND")

    n_figs = len(figs_to_run)
    fig    = plt.figure(figsize=(18, 5.5*n_figs))
    gs     = gridspec.GridSpec(n_figs, 3, figure=fig, hspace=0.65, wspace=0.40)
    fi     = 0
    npz    = {}

    # ════════════ FIG E ════════════
    if "E" in figs_to_run:
        print("\n── Figure E: Environment-conditional gradient variance ──")
        results_E = {}
        omega_labels = []
        for key, label in [("w1","ω=1.0"), ("w01","ω=0.1"), ("w0001","ω=0.001")]:
            if key not in ckpts:
                continue
            bf, jas, n_elec, omega = ckpts[key]
            C = _setup(n_elec, omega)
            spin = _spin(n_elec)
            print(f"  Sampling {label}...")
            x_m = mcmc(C, bf, jas, spin, 2000, omega, burn=350)
            jas_pair = _ablate_mp(jas)
            print(f"  Computing conditional variance...")
            out = env_conditional_variance(x_m, jas, jas_pair, spin)
            r, vc, vp, mc, mp, cnt = out
            results_E[label] = out
            omega_labels.append(label)
            ratio = (vc/(vp+1e-12)).mean()
            print(f"    mean var ratio CTNN/pair = {ratio:.2f}×")
            npz[f"E_{label}_r"] = r
            npz[f"E_{label}_vc"] = vc
            npz[f"E_{label}_vp"] = vp

        ax_row = [fig.add_subplot(gs[fi, j]) for j in range(3)]
        make_fig_E(ax_row, results_E, omega_labels)
        fi += 1

    # ════════════ FIG H ════════════
    if "H" in figs_to_run:
        print("\n── Figure H: Training dynamics (no GPU) ──")
        logs = {}
        for key, path in LOGS.items():
            if path.exists():
                recs = load_epochs(path)
                logs[key] = recs
                print(f"  {key}: {len(recs)} epochs  "
                      f"E_final={recs[-1]['E']:.4f}  "
                      f"var_final={recs[-1]['var_EL']:.4e}")
            else:
                print(f"  {key}: NOT FOUND at {path}")
        ax_row = [fig.add_subplot(gs[fi, j]) for j in range(3)]
        make_fig_H(ax_row, logs)
        fi += 1

    # ════════════ FIG I ════════════
    if "I" in figs_to_run:
        print("\n── Figure I: Kinetic/potential decomposition ──")
        decomp = {}
        for key, label in [("w1","ω=1.0"), ("w01","ω=0.1"), ("w0001","ω=0.001")]:
            if key not in ckpts:
                continue
            bf, jas, n_elec, omega = ckpts[key]
            C = _setup(n_elec, omega)
            spin = _spin(n_elec)
            x_m = mcmc(C, bf, jas, spin, 500, omega, burn=300)
            jas_pair = _ablate_mp(jas)
            print(f"\n  {label}:")
            decomp[label] = energy_decomposition(x_m, C, bf, jas, jas_pair, spin)
            for lbl, vals in decomp[label].items():
                E, T, Vt, Vc, se = vals
                dE = E - decomp[label]["CTNN"][0] if lbl != "CTNN" else 0.
                print(f"    {lbl}: E={E:.4f}(±{se:.4f})  T={T:.4f}  Vtrap={Vt:.4f}  Vcoul={Vc:.4f}  ΔE={dE:+.4f}")

        ax_row = [fig.add_subplot(gs[fi, j]) for j in range(3)]
        make_fig_I(ax_row, decomp)
        fi += 1

    # ════════════ FIG J ════════════
    if "J" in figs_to_run:
        print("\n── Figure J: Backflow vs classical force alignment ──")
        geo = {}
        for key, label in [("w1","ω=1.0"), ("w01","ω=0.1"), ("w0001","ω=0.001")]:
            if key not in ckpts:
                continue
            bf, jas, n_elec, omega = ckpts[key]
            C = _setup(n_elec, omega)
            spin = _spin(n_elec)
            x_m = mcmc(C, bf, jas, spin, 1500, omega, burn=350)
            print(f"  Computing force alignment {label}...")
            g = bf_force_alignment(x_m, bf, spin, omega)
            geo[label] = g
            print(f"    cos(Δx,F_full): median={np.median(g['cos']):.3f}  "
                  f"cos_trap={np.median(g['cos_trap']):.3f}  "
                  f"cos_coul={np.median(g['cos_coul']):.3f}")
            npz[f"J_{label}_cos"]  = g["cos"]
            npz[f"J_{label}_ctrap"]= g["cos_trap"]
            npz[f"J_{label}_ccoul"]= g["cos_coul"]
            npz[f"J_{label}_rmin"] = g["r_min"]
            npz[f"J_{label}_ss"]   = g["same_spin"].astype(np.float32)

        ax_row = [fig.add_subplot(gs[fi, j]) for j in range(3)]
        make_fig_J(ax_row, geo)
        fi += 1

    # Save
    pdf = OUT_DIR / "deeper_diagnostics.pdf"
    png = OUT_DIR / "deeper_diagnostics.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=150)
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {pdf}")
    if npz:
        np.savez(OUT_DIR / "deeper_diagnostics_data.npz", **npz)
        print(f"Saved data: {OUT_DIR / 'deeper_diagnostics_data.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", default=["E","H","I","J"],
                    choices=["E","H","I","J"])
    args = ap.parse_args()
    main(set(args.figs))
