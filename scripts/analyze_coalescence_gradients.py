#!/usr/bin/env python3
"""
Coalescence Gradient Pathology Analysis
========================================
Comprehensive diagnostic of the gradient signal near the Slater determinant
nodal surface for the backflow + Jastrow ansatz.

System: N=6, ω=1.0 quantum dots (3↑ + 3↓) using trained BF+Jastrow checkpoint.

Usage:
  CUDA_MANUAL_DEVICE=4 python3 scripts/analyze_coalescence_gradients.py
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
import torch.nn.functional as F

# --- Setup paths ---
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import config
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import slater_determinant_closed_shell
from PINN import CTNNBackflowNet
from jastrow_architectures import CTNNJastrowVCycle

# ===================== Configuration =====================
N_ELEC = 6
OMEGA = 1.0
DIM = 2
N_SAMPLES = 8192       # MCMC samples for scatter analysis
N_GRAD_SAMPLES = 2048  # per-sample gradient computation
N_SR_SAMPLES = 512     # samples for Fisher / SR analysis
DEVICE = torch.device(f"cuda:{os.environ.get('CUDA_MANUAL_DEVICE', 4)}")
DTYPE = torch.float64
CHECKPOINT = REPO / "results" / "arch_colloc" / "bf_ctnn_vcycle.pt"

# Fixed output directory
OUT_DIR = REPO / "outputs" / "coalescence_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Output directory: {OUT_DIR}")
print(f"Figures will be saved as:")
for name in ["fig1_controlled_scan.png", "fig2_mcmc_scatter.png",
             "fig3_sampling_comparison.png", "fig4_sr_vs_adam.png"]:
    print(f"  {OUT_DIR / name}")


# ===================== Step 1: Setup physics =====================
def setup_system():
    """Initialize config and build C_occ from analytic HO orbitals.
    Matches run_weak_form.setup() exactly.
    """
    n_occ = N_ELEC // 2
    nx = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
    ny = nx
    L = max(8.0, 3.0 / math.sqrt(OMEGA))

    cfg = config.update(
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
    C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
    print(f"Setup: N={N_ELEC}, ω={OMEGA}, basis=cart {nx}×{ny}, n_occ={n_occ}")
    return cfg, C_occ


# ===================== Step 2: Load trained models =====================
def load_models(cfg):
    """Load BF + Jastrow from checkpoint."""
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

    # Architecture inferred from state dict: n_down=1, n_up=1
    jas_net = CTNNJastrowVCycle(
        n_particles=N_ELEC, d=DIM, omega=OMEGA,
        node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
        n_down=1, n_up=1, msg_layers=1, node_layers=1,
        readout_hidden=64, readout_layers=2, act="silu",
    ).to(DEVICE).to(DTYPE)
    jas_net.load_state_dict(ckpt["jas_state"])
    jas_net.eval()

    bf_scale = F.softplus(ckpt["bf_state"]["bf_scale_raw"]).item()
    print(f"BF: {sum(p.numel() for p in bf_net.parameters())} params, "
          f"out_bound={bfc['out_bound']}, bf_scale={bf_scale:.4f}")
    print(f"Jas: {sum(p.numel() for p in jas_net.parameters())} params")
    print(f"Checkpoint: E={ckpt['E']:.6f}, err={ckpt['err']:.4f}%")
    return bf_net, jas_net


# ===================== Helpers =====================
def make_spin():
    n_up = N_ELEC // 2
    return torch.cat([
        torch.zeros(n_up, dtype=torch.long),
        torch.ones(N_ELEC - n_up, dtype=torch.long),
    ]).to(DEVICE)


def _params_dict():
    p = config.get().as_dict()
    p["device"] = str(DEVICE)
    p["torch_dtype"] = DTYPE
    return p


def eval_sd(x, C_occ, spin):
    return slater_determinant_closed_shell(
        x_config=x, C_occ=C_occ, params=_params_dict(), spin=spin, normalize=True
    )


def eval_full_logpsi(x, C_occ, spin, bf_net, jas_net):
    spin_bn = spin.unsqueeze(0).expand(x.shape[0], -1)
    dx = bf_net(x, spin=spin_bn)
    x_eff = x + dx
    sign, logabs = slater_determinant_closed_shell(
        x_config=x_eff, C_occ=C_occ, params=_params_dict(), spin=spin, normalize=True
    )
    j = jas_net(x, spin=spin_bn).squeeze(-1)
    return sign, logabs + j


def compute_pairwise_distances(x, spin):
    B, N, d = x.shape
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    dist = torch.sqrt((diff**2).sum(-1) + 1e-30)
    s = spin.unsqueeze(0).expand(B, -1)
    same = (s.unsqueeze(2) == s.unsqueeze(1))
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    same = same & ~eye
    big = torch.tensor(1e10, device=x.device, dtype=x.dtype)
    d_same = dist.clone()
    d_same[~same] = big
    min_same = d_same.min(dim=-1).values.min(dim=-1).values
    return min_same


# ===================== Sampling =====================
def mcmc_sample(psi_log_fn, n_samples, burn_in=300, step_sigma_ell=0.15):
    """MCMC from |Ψ|² with given log-psi function."""
    ell = 1.0 / math.sqrt(OMEGA)
    x = torch.randn(n_samples, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * ell
    step_sigma = step_sigma_ell * ell
    with torch.no_grad():
        lp = 2.0 * psi_log_fn(x)
        acc_total = 0
        for _ in range(burn_in):
            prop = x + torch.randn_like(x) * step_sigma
            lp_prop = 2.0 * psi_log_fn(prop)
            accept = (torch.rand_like(lp).log() < (lp_prop - lp))
            x = torch.where(accept.view(-1, 1, 1), prop, x)
            lp = torch.where(accept, lp_prop, lp)
            acc_total += accept.float().mean().item()
    print(f"  MCMC accept rate: {acc_total / burn_in:.3f}")
    return x.detach()


# ===================== Controlled Coalescence Scan =====================
def controlled_coalescence_scan(C_occ, spin, bf_net, jas_net):
    """
    Fix a base config with LARGE |SD|, then systematically bring
    two same-spin electrons together. Measures everything vs distance d.
    """
    print("  Finding base configuration with large |SD|...")

    # Sample many configs, pick the one with the LARGEST |SD|
    def full_logpsi(x):
        _, lp = eval_full_logpsi(x, C_occ, spin, bf_net, jas_net)
        return lp

    x_pool = mcmc_sample(full_logpsi, n_samples=4096, burn_in=400)
    with torch.no_grad():
        _, logabs_pool = eval_sd(x_pool, C_occ, spin)
    best_idx = logabs_pool.argmax().item()
    x0 = x_pool[best_idx].clone()
    sd_best = torch.exp(logabs_pool[best_idx]).item()
    print(f"  Best |SD| = {sd_best:.4e}")

    # Electrons 0,1,2 are spin-up. Move electron 1 toward electron 0.
    r0 = x0[0].clone()
    r1_orig = x0[1].clone()
    d_orig = (r1_orig - r0).norm().item()
    direction = (r1_orig - r0) / (r1_orig - r0).norm()

    # Scan from the original distance down to nearly zero
    distances = np.concatenate([
        np.logspace(np.log10(max(d_orig * 1.5, 2.0)), -0.5, 40),   # far field
        np.logspace(-0.5, -3.0, 60),                                 # near node
    ])
    distances = np.unique(np.sort(distances)[::-1])  # large to small

    params = _params_dict()
    spin_1 = spin.unsqueeze(0)

    keys = ["d", "sd_bare", "sd_bf", "logpsi_full",
            "T_loc", "V_coul", "V_trap", "E_L",
            "bf_grad_norm", "jas_grad_norm", "grad_logsd_bf_norm"]
    results = {k: [] for k in keys}

    for di, dist in enumerate(distances):
        xi = x0.clone().unsqueeze(0)
        xi[0, 1] = r0 + dist * direction

        # 1. Bare SD
        with torch.no_grad():
            _, logabs_bare = eval_sd(xi, C_occ, spin)
            results["sd_bare"].append(torch.exp(logabs_bare).item())

        # 2. SD with backflow
        with torch.no_grad():
            dx = bf_net(xi, spin=spin_1)
            x_eff = xi + dx
            _, logabs_bf = slater_determinant_closed_shell(
                x_config=x_eff, C_occ=C_occ, params=params, spin=spin, normalize=True
            )
            results["sd_bf"].append(torch.exp(logabs_bf).item())

        # 3. Full logpsi
        with torch.no_grad():
            _, lp_full = eval_full_logpsi(xi, C_occ, spin, bf_net, jas_net)
            results["logpsi_full"].append(lp_full.item())

        # 4. Local energy components
        try:
            xi_g = xi.detach().requires_grad_(True)
            dx_g = bf_net(xi_g, spin=spin_1)
            x_eff_g = xi_g + dx_g
            _, logabs_g = slater_determinant_closed_shell(
                x_config=x_eff_g, C_occ=C_occ, params=params, spin=spin, normalize=True
            )
            j_g = jas_net(xi_g, spin=spin_1).squeeze(-1)
            logpsi_g = logabs_g + j_g

            grad_lp = torch.autograd.grad(
                logpsi_g.sum(), xi_g, create_graph=True, retain_graph=True
            )[0]
            g2 = (grad_lp**2).sum()

            lap = torch.zeros(1, device=DEVICE, dtype=DTYPE)
            for ii in range(N_ELEC):
                for kk in range(DIM):
                    gi = grad_lp[0, ii, kk]
                    hii = torch.autograd.grad(
                        gi, xi_g, retain_graph=True, create_graph=False
                    )[0][0, ii, kk]
                    lap = lap + hii

            T = (-0.5 * (lap + g2)).item()
            V_c = compute_coulomb_interaction(xi_g).item()
            V_t = (0.5 * OMEGA**2 * (xi_g**2).sum()).item()
            results["T_loc"].append(T)
            results["V_coul"].append(V_c)
            results["V_trap"].append(V_t)
            results["E_L"].append(T + V_c + V_t)
        except Exception:
            for k in ["T_loc", "V_coul", "V_trap", "E_L"]:
                results[k].append(float("nan"))

        # 5. BF + Jas parameter gradient norms
        bf_net.zero_grad()
        jas_net.zero_grad()
        xi_p = xi.detach().requires_grad_(True)
        dx_p = bf_net(xi_p, spin=spin_1)
        x_eff_p = xi_p + dx_p
        _, logabs_p = slater_determinant_closed_shell(
            x_config=x_eff_p, C_occ=C_occ, params=params, spin=spin, normalize=True
        )
        j_p = jas_net(xi_p, spin=spin_1).squeeze(-1)
        (logabs_p + j_p).backward()
        bg = [p.grad.view(-1) for p in bf_net.parameters() if p.grad is not None]
        jg = [p.grad.view(-1) for p in jas_net.parameters() if p.grad is not None]
        results["bf_grad_norm"].append(torch.cat(bg).norm().item() if bg else 0.0)
        results["jas_grad_norm"].append(torch.cat(jg).norm().item() if jg else 0.0)

        # 6. Gradient of logSD(x+Δx) w.r.t. BF params only
        bf_net.zero_grad()
        xi_s = xi.detach().requires_grad_(True)
        dx_s = bf_net(xi_s, spin=spin_1)
        x_eff_s = xi_s + dx_s
        _, logabs_s = slater_determinant_closed_shell(
            x_config=x_eff_s, C_occ=C_occ, params=params, spin=spin, normalize=True
        )
        logabs_s.backward()
        bg_sd = [p.grad.view(-1) for p in bf_net.parameters() if p.grad is not None]
        results["grad_logsd_bf_norm"].append(
            torch.cat(bg_sd).norm().item() if bg_sd else 0.0
        )

        results["d"].append(dist)
        if di % 20 == 0:
            print(f"    scan {di+1}/{len(distances)}: d={dist:.4f}, "
                  f"|SD_bare|={results['sd_bare'][-1]:.2e}, "
                  f"|SD_bf|={results['sd_bf'][-1]:.2e}, "
                  f"|∂logSD/∂θ|={results['grad_logsd_bf_norm'][-1]:.2e}")

    return results


# ===================== Per-sample parameter gradients =====================
def compute_parameter_gradients(x, C_occ, spin, bf_net, jas_net):
    """Per-sample ∂logΨ/∂θ for BF and Jastrow. Returns norms + flat grads for Fisher."""
    B = x.shape[0]
    params = _params_dict()
    spin_bn = spin.unsqueeze(0)

    bf_grad_norms = torch.zeros(B, dtype=DTYPE)
    jas_grad_norms = torch.zeros(B, dtype=DTYPE)
    bf_grads_list = []
    jas_grads_list = []

    for si in range(B):
        xi = x[si:si+1].detach().requires_grad_(True)
        sp = spin_bn.expand(1, -1)

        bf_net.zero_grad()
        jas_net.zero_grad()

        dx = bf_net(xi, spin=sp)
        x_eff = xi + dx
        _, logabs = slater_determinant_closed_shell(
            x_config=x_eff, C_occ=C_occ, params=params, spin=spin, normalize=True
        )
        j = jas_net(xi, spin=sp).squeeze(-1)
        (logabs + j).backward()

        bg = [p.grad.detach().view(-1) for p in bf_net.parameters() if p.grad is not None]
        jg = [p.grad.detach().view(-1) for p in jas_net.parameters() if p.grad is not None]

        if bg:
            bg_flat = torch.cat(bg)
            bf_grad_norms[si] = bg_flat.norm().cpu()
            if si < N_SR_SAMPLES:
                bf_grads_list.append(bg_flat.cpu())
        if jg:
            jg_flat = torch.cat(jg)
            jas_grad_norms[si] = jg_flat.norm().cpu()
            if si < N_SR_SAMPLES:
                jas_grads_list.append(jg_flat.cpu())

        if si % 256 == 0 and si > 0:
            print(f"    gradients: {si}/{B}")

    bf_grads = torch.stack(bf_grads_list) if bf_grads_list else None
    jas_grads = torch.stack(jas_grads_list) if jas_grads_list else None
    return bf_grad_norms, jas_grad_norms, bf_grads, jas_grads


# ===================== Local energy in batches =====================
def compute_local_energy_batched(x, C_occ, spin, bf_net, jas_net, batch_sz=256):
    """Returns T_loc, V_coul per sample."""
    B = x.shape[0]
    params = _params_dict()
    T_all, Vc_all = [], []

    for i in range(0, B, batch_sz):
        e = min(i + batch_sz, B)
        xb = x[i:e].detach().requires_grad_(True)
        Bb = xb.shape[0]
        spin_bn = spin.unsqueeze(0).expand(Bb, -1)

        dx = bf_net(xb, spin=spin_bn)
        x_eff = xb + dx
        _, logabs = slater_determinant_closed_shell(
            x_config=x_eff, C_occ=C_occ, params=params, spin=spin, normalize=True
        )
        j = jas_net(xb, spin=spin_bn).squeeze(-1)
        logpsi = logabs + j

        grad_lp = torch.autograd.grad(
            logpsi.sum(), xb, create_graph=True, retain_graph=True
        )[0]
        g2 = (grad_lp**2).sum(dim=(1, 2))

        N, d = xb.shape[1], xb.shape[2]
        lap = torch.zeros(Bb, device=DEVICE, dtype=DTYPE)
        for ii in range(N):
            for kk in range(d):
                gi = grad_lp[:, ii, kk]
                hii = torch.autograd.grad(
                    gi, xb, grad_outputs=torch.ones_like(gi),
                    retain_graph=True, create_graph=False
                )[0][:, ii, kk]
                lap = lap + hii

        T_loc = -0.5 * (lap + g2)
        with torch.no_grad():
            V_coul = compute_coulomb_interaction(xb).view(-1)

        T_all.append(T_loc.detach().cpu())
        Vc_all.append(V_coul.cpu())

        if i % 1024 == 0:
            print(f"    energy: {i}/{B}")

    return torch.cat(T_all).numpy(), torch.cat(Vc_all).numpy()


# ===================== SR vs Adam =====================
def sr_vs_adam_analysis(bf_grads, min_same_dist):
    """Compare raw (Adam) gradient vs SR-preconditioned gradient using Woodbury."""
    if bf_grads is None or bf_grads.shape[0] < 32:
        print("  Not enough samples for SR analysis.")
        return None

    G = bf_grads.to(torch.float64)  # (K, P) on CPU
    K, P = G.shape
    print(f"  SR analysis: K={K} samples, P={P} params")

    g_mean = G.mean(dim=0)  # Adam direction

    # Woodbury: (G^T G / K + λI)^{-1} g via K×K system
    lam = 1e-3 * (G * G).mean()
    A = (G @ G.T) / K + lam * torch.eye(K, dtype=G.dtype)  # (K, K)
    Gg = G @ g_mean  # (K,)

    try:
        y = torch.linalg.solve(A, Gg / K)
        g_sr = (g_mean - G.T @ y) / lam
    except Exception as e:
        print(f"  Woodbury failed: {e}, using diagonal Fisher")
        diag_fisher = (G * G).mean(dim=0)
        g_sr = g_mean / (diag_fisher + lam)

    adam_proj = (G @ g_mean) / (g_mean.norm() + 1e-30)
    sr_proj = (G @ g_sr) / (g_sr.norm() + 1e-30)

    try:
        fisher_cond = torch.linalg.cond(A).item()
    except Exception:
        fisher_cond = float("nan")

    return {
        "adam_proj": adam_proj.numpy(),
        "sr_proj": sr_proj.numpy(),
        "g_mean_norm": g_mean.norm().item(),
        "g_sr_norm": g_sr.norm().item(),
        "fisher_cond": fisher_cond,
        "min_same_dist": min_same_dist[:K],
    }


# ===================== PLOTTING =====================
def plot_all(scan, scatter, sr_data):
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "legend.fontsize": 10,
    })

    d = np.array(scan["d"])
    sd_bare = np.array(scan["sd_bare"])
    sd_bf = np.array(scan["sd_bf"])

    # ===== Figure 1: Controlled Scan =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) |SD| vs distance
    ax = axes[0, 0]
    ax.loglog(d, sd_bare, "b-", lw=2, label="$|D(\\mathbf{x})|$ (bare)")
    ax.loglog(d, sd_bf, "r-", lw=2, label="$|D(\\mathbf{x}+\\Delta\\mathbf{x})|$ (with BF)")
    ax.set_xlabel("Same-spin pair distance $d_{12}$")
    ax.set_ylabel("$|D|$")
    ax.set_title("(a) Slater determinant near coalescence")
    ax.legend()

    # (b) Local energy decomposition
    ax = axes[0, 1]
    T = np.array(scan["T_loc"])
    Vc = np.array(scan["V_coul"])
    EL = np.array(scan["E_L"])
    m = np.isfinite(T) & np.isfinite(Vc)
    ax.semilogx(d[m], T[m], "r-", lw=2, label="$T_{\\mathrm{loc}}$ (kinetic)")
    ax.semilogx(d[m], Vc[m], "g-", lw=2, label="$V_{\\mathrm{Coulomb}}$")
    ax.semilogx(d[m], EL[m], "k--", lw=2, label="$E_L$ (total)")
    ax.set_xlabel("Same-spin pair distance $d_{12}$")
    ax.set_ylabel("Energy (a.u.)")
    ax.set_title("(b) Local energy diverges at coalescence")
    ax.legend()
    # Auto-clip to reasonable range
    fin = EL[m & np.isfinite(EL)]
    if len(fin) > 5:
        q5, q95 = np.percentile(fin, [2, 98])
        margin = max(abs(q95 - q5) * 0.5, 5)
        ax.set_ylim(q5 - margin, q95 + margin)

    # (c) Gradient norms vs distance
    ax = axes[1, 0]
    bf_gn = np.array(scan["bf_grad_norm"])
    jas_gn = np.array(scan["jas_grad_norm"])
    sd_gn = np.array(scan["grad_logsd_bf_norm"])
    ax.loglog(d, bf_gn, "r-", lw=2, label="$|\\partial\\log\\Psi/\\partial\\theta_{BF}|$")
    ax.loglog(d, jas_gn, "b-", lw=2, label="$|\\partial\\log\\Psi/\\partial\\theta_{Jas}|$")
    ax.loglog(d, sd_gn, "k--", lw=1.5, label="$|\\partial\\log D/\\partial\\theta_{BF}|$ (SD only)")
    ax.set_xlabel("Same-spin pair distance $d_{12}$")
    ax.set_ylabel("Parameter gradient norm")
    ax.set_title("(c) Gradient norms vs coalescence distance")
    ax.legend()

    # (d) |∂logSD/∂θ| vs |SD(x+Δx)| — the D^{-1} scaling test
    ax = axes[1, 1]
    m2 = (sd_bf > 1e-30) & (sd_gn > 0)
    ax.loglog(sd_bf[m2], sd_gn[m2], "ko", ms=4, alpha=0.7, label="Data")
    # Fit D^{-1} reference line
    if m2.sum() > 2:
        log_sd = np.log(sd_bf[m2])
        log_gn = np.log(sd_gn[m2])
        # Least-squares fit: log(gn) = a * log(sd) + b
        A_fit = np.vstack([log_sd, np.ones_like(log_sd)]).T
        slope, intercept = np.linalg.lstsq(A_fit, log_gn, rcond=None)[0]
        sd_ref = np.logspace(np.log10(sd_bf[m2].min()), np.log10(sd_bf[m2].max()), 50)
        ax.loglog(sd_ref, np.exp(intercept) * sd_ref**slope, "r--", lw=1.5,
                  label=f"Fit: slope={slope:.2f}")
        ax.loglog(sd_ref, np.exp(intercept + np.log(sd_ref.max()) * (slope + 1)) / sd_ref,
                  "b:", lw=1, alpha=0.5, label="$\\propto |D|^{-1}$ (reference)")
    ax.set_xlabel("$|D(\\mathbf{x}+\\Delta\\mathbf{x})|$")
    ax.set_ylabel("$|\\partial\\log D / \\partial\\theta_{BF}|$")
    ax.set_title("(d) Gradient scaling with determinant")
    ax.legend()

    fig.suptitle(
        f"Coalescence Gradient Pathology — N={N_ELEC}, ω={OMEGA}\n"
        "Controlled scan: move two same-spin electrons together",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    path = OUT_DIR / "fig1_controlled_scan.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # ===== Figure 2: MCMC Scatter =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    min_d = scatter["min_same_dist"]
    log_sd = np.log10(np.clip(scatter["sd_bare"], 1e-30, None))
    n_pts = len(min_d)

    # (a) SD distribution
    ax = axes[0, 0]
    ax.hist(log_sd, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    med = np.median(log_sd)
    ax.axvline(x=med, color="red", ls="--", lw=1.5, label=f"median = {med:.1f}")
    ax.set_xlabel("$\\log_{10}|D|$")
    ax.set_ylabel("Density")
    ax.set_title(f"(a) SD distribution ({n_pts} samples from $|\\Psi|^2$)")
    ax.legend()

    # (b) |SD| vs min same-spin distance
    ax = axes[0, 1]
    ax.scatter(min_d, log_sd, s=2, alpha=0.3, c="steelblue", rasterized=True)
    ax.set_xlabel("Min same-spin distance")
    ax.set_ylabel("$\\log_{10}|D|$")
    ax.set_title("(b) SD vanishes at coalescence")

    # (c) Coulomb vs min distance
    ax = axes[0, 2]
    ax.scatter(min_d, scatter["V_coul"], s=2, alpha=0.3, c="green", rasterized=True)
    ax.set_xlabel("Min same-spin distance")
    ax.set_ylabel("$V_{\\mathrm{Coulomb}}$")
    ax.set_title("(c) Coulomb diverges at coalescence")
    fin_vc = scatter["V_coul"][np.isfinite(scatter["V_coul"])]
    if len(fin_vc) > 0:
        ax.set_ylim(top=np.percentile(fin_vc, 99.5) * 1.2)

    # (d) BF gradient norm vs |SD|
    ax = axes[1, 0]
    bf_gn_s = scatter["bf_grad_norm"]
    m3 = bf_gn_s > 0
    ax.scatter(log_sd[m3], np.log10(bf_gn_s[m3] + 1e-30), s=2, alpha=0.3, c="red", rasterized=True)
    ax.set_xlabel("$\\log_{10}|D|$")
    ax.set_ylabel("$\\log_{10}|\\partial\\log\\Psi/\\partial\\theta_{BF}|$")
    ax.set_title("(d) BF gradients explode near node")

    # (e) Jastrow gradient norm vs |SD|
    ax = axes[1, 1]
    jas_gn_s = scatter["jas_grad_norm"]
    m4 = jas_gn_s > 0
    ax.scatter(log_sd[m4], np.log10(jas_gn_s[m4] + 1e-30), s=2, alpha=0.3, c="blue", rasterized=True)
    ax.set_xlabel("$\\log_{10}|D|$")
    ax.set_ylabel("$\\log_{10}|\\partial\\log\\Psi/\\partial\\theta_{Jas}|$")
    ax.set_title("(e) Jastrow gradients also affected")

    # (f) |T_loc| vs |SD|
    ax = axes[1, 2]
    T_abs = np.abs(scatter["T_loc"])
    m5 = T_abs > 0
    ax.scatter(log_sd[m5], np.log10(T_abs[m5] + 1e-30), s=2, alpha=0.3, c="orange", rasterized=True)
    ax.set_xlabel("$\\log_{10}|D|$")
    ax.set_ylabel("$\\log_{10}|T_{\\mathrm{loc}}|$")
    ax.set_title("(f) Kinetic energy divergence")

    fig.suptitle(
        f"MCMC samples from $|\\Psi|^2$: N={N_ELEC}, ω={OMEGA}, {n_pts} samples",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    path = OUT_DIR / "fig2_mcmc_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # ===== Figure 3: SD-only vs full Ψ sampling =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(scatter["min_same_sd_only"], bins=80, density=True, alpha=0.6,
            color="blue", label="SD-only sampling", edgecolor="none")
    ax.hist(scatter["min_same_dist"], bins=80, density=True, alpha=0.6,
            color="red", label="Full $\\Psi$ sampling", edgecolor="none")
    ax.set_xlabel("Min same-spin distance")
    ax.set_ylabel("Density")
    ax.set_title("(a) Where do samples live?")
    ax.legend()

    ax = axes[1]
    log_sd_only = np.log10(np.clip(scatter["sd_bare_sd_only"], 1e-30, None))
    ax.hist(log_sd_only, bins=80, density=True, alpha=0.6,
            color="blue", label="SD-only", edgecolor="none")
    ax.hist(log_sd, bins=80, density=True, alpha=0.6,
            color="red", label="Full $\\Psi$", edgecolor="none")
    ax.set_xlabel("$\\log_{10}|D|$")
    ax.set_ylabel("Density")
    ax.set_title("(b) SD magnitude at sampled points")
    ax.legend()

    ax = axes[2]
    ax.scatter(scatter["min_same_sd_only"], log_sd_only, s=2, alpha=0.2,
               c="blue", label="SD-only", rasterized=True)
    ax.scatter(scatter["min_same_dist"], log_sd, s=2, alpha=0.2,
               c="red", label="Full $\\Psi$", rasterized=True)
    ax.set_xlabel("Min same-spin distance")
    ax.set_ylabel("$\\log_{10}|D|$")
    ax.set_title("(c) Jastrow+BF reshapes the distribution")
    ax.legend()

    fig.suptitle("How Jastrow + Backflow reshape the sampling distribution", fontsize=15)
    plt.tight_layout()
    path = OUT_DIR / "fig3_sampling_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # ===== Figure 4: SR vs Adam =====
    if sr_data is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        min_d_sr = sr_data["min_same_dist"]
        K = len(sr_data["adam_proj"])

        ax = axes[0]
        ax.scatter(min_d_sr, sr_data["adam_proj"], s=3, alpha=0.5,
                   c="orange", label="Adam (raw gradient)")
        ax.scatter(min_d_sr, sr_data["sr_proj"], s=3, alpha=0.5,
                   c="purple", label="SR (Fisher-preconditioned)")
        ax.set_xlabel("Min same-spin distance")
        ax.set_ylabel("Projected gradient")
        ax.set_title("(a) Per-sample gradient projection")
        ax.legend()
        all_p = np.concatenate([sr_data["adam_proj"], sr_data["sr_proj"]])
        fin_p = all_p[np.isfinite(all_p)]
        if len(fin_p) > 0:
            q1, q99 = np.percentile(fin_p, [1, 99])
            margin = max(abs(q99 - q1) * 0.3, 0.5)
            ax.set_ylim(q1 - margin, q99 + margin)

        ax = axes[1]
        ax.hist(sr_data["adam_proj"], bins=80, density=True, alpha=0.5,
                color="orange", label="Adam", edgecolor="none")
        ax.hist(sr_data["sr_proj"], bins=80, density=True, alpha=0.5,
                color="purple", label="SR", edgecolor="none")
        ax.set_xlabel("Projected gradient")
        ax.set_ylabel("Density")
        ax.set_title(f"(b) SR suppresses outlier gradients\nFisher cond = {sr_data['fisher_cond']:.1e}")
        ax.legend()
        if len(fin_p) > 0:
            ax.set_xlim(q1 - margin, q99 + margin)

        # (c) Gradient norm ratio
        ax = axes[2]
        adam_norms = np.abs(sr_data["adam_proj"])
        sr_norms = np.abs(sr_data["sr_proj"])
        ratio = adam_norms / (sr_norms + 1e-30)
        ax.scatter(min_d_sr, ratio, s=3, alpha=0.5, c="green")
        ax.axhline(y=1.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Min same-spin distance")
        ax.set_ylabel("|Adam proj| / |SR proj|")
        ax.set_title("(c) SR reweighting vs distance to node")
        ax.set_yscale("log")

        fig.suptitle(
            f"Stochastic Reconfiguration dampens near-node gradient noise\n"
            f"|g_Adam| = {sr_data['g_mean_norm']:.2e}, "
            f"|g_SR| = {sr_data['g_sr_norm']:.2e}",
            fontsize=14,
        )
        plt.tight_layout()
        path = OUT_DIR / "fig4_sr_vs_adam.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


# ===================== MAIN =====================
def main():
    print("=" * 60)
    print("COALESCENCE GRADIENT ANALYSIS")
    print("=" * 60)

    cfg, C_occ = setup_system()
    bf_net, jas_net = load_models(cfg)
    spin = make_spin()

    # --- Part A: Controlled coalescence scan ---
    print("\n--- Part A: Controlled coalescence scan ---")
    scan = controlled_coalescence_scan(C_occ, spin, bf_net, jas_net)

    # --- Part B: MCMC scatter ---
    print("\n--- Part B: MCMC sampling from |Ψ|² ---")
    def full_logpsi(x):
        _, lp = eval_full_logpsi(x, C_occ, spin, bf_net, jas_net)
        return lp

    x_full = mcmc_sample(full_logpsi, n_samples=N_SAMPLES, burn_in=400)
    min_same = compute_pairwise_distances(x_full, spin)

    print("  Computing |SD| at sampled points...")
    with torch.no_grad():
        _, logabs = eval_sd(x_full, C_occ, spin)
        sd_bare = torch.exp(logabs).cpu().numpy()

    print("  Computing local energy components...")
    T_all, Vc_all = compute_local_energy_batched(x_full, C_occ, spin, bf_net, jas_net)

    print("  Computing per-sample parameter gradients...")
    x_grad = x_full[:N_GRAD_SAMPLES]
    bf_gn, jas_gn, bf_grads, jas_grads = compute_parameter_gradients(
        x_grad, C_occ, spin, bf_net, jas_net
    )

    # SD-only sampling for comparison
    print("\n  Sampling from |SD|² only...")
    def sd_logpsi(x):
        _, la = eval_sd(x, C_occ, spin)
        return la
    x_sd = mcmc_sample(sd_logpsi, n_samples=N_SAMPLES, burn_in=400, step_sigma_ell=0.12)
    min_same_sd = compute_pairwise_distances(x_sd, spin)
    with torch.no_grad():
        _, logabs_sd = eval_sd(x_sd, C_occ, spin)
        sd_bare_sd = torch.exp(logabs_sd).cpu().numpy()

    scatter = {
        "min_same_dist": min_same[:N_GRAD_SAMPLES].cpu().numpy(),
        "sd_bare": sd_bare[:N_GRAD_SAMPLES],
        "T_loc": T_all[:N_GRAD_SAMPLES],
        "V_coul": Vc_all[:N_GRAD_SAMPLES],
        "bf_grad_norm": bf_gn.numpy(),
        "jas_grad_norm": jas_gn.numpy(),
        "min_same_sd_only": min_same_sd.cpu().numpy(),
        "sd_bare_sd_only": sd_bare_sd,
    }

    # --- Part C: SR vs Adam ---
    print("\n--- Part C: SR vs Adam analysis ---")
    sr_data = sr_vs_adam_analysis(
        bf_grads,
        min_same[:N_SR_SAMPLES].cpu().numpy() if bf_grads is not None else np.array([]),
    )

    # --- Plotting ---
    print("\n--- Generating plots ---")
    plot_all(scan, scatter, sr_data)

    # Save raw data
    np.savez(OUT_DIR / "analysis_data.npz", **{
        k: np.array(v) for k, v in scan.items()
    })
    print(f"\nAll outputs in: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
