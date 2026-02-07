#!/usr/bin/env python
"""
Avoided Crossing Study v3 — Publication-quality 2D double-well analysis.

New in v3 (supervisor round 4):
 A) 5 seeds, full uncertainty for every observable
 B) Retrain-on-hop with logged diagnostics + branch-consistency plot
 C) HO-basis overlap decomposition → where missing W_sub goes
 D) H_eff in {|10⟩,|01⟩} and extended subspace → Δ_eff(λ)
 E) Two-regime comparison: quasi-1D vs full-2D
 F) Ablation: SD / SD+J / SD+BF+J near resonance
 G) 5 publication-ready figures

Usage:
    python avoided_crossing_study.py --dense              # 2D, 5 seeds
    python avoided_crossing_study.py --dense --quasi1d    # quasi-1D, 5 seeds
    python avoided_crossing_study.py --ablation           # 3-model ablation
    python avoided_crossing_study.py --quick              # quick 7-pt
    python avoided_crossing_study.py --tiny               # smoke test
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_MANUAL_DEVICE"] = "0"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Neural_Networks import psi_fn
from PINN import PINN, CTNNBackflowNet

torch.set_num_threads(4)

# ============================================================
# Constants
# ============================================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
OMEGA_BASE = 1.0
N_PARTICLES = 2
D = 2
W_SUB_THRESHOLD = 0.8

RESULTS_DIR = Path("/Users/aleksandersekkelsten/thesis/results/avoided_crossing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Double-Well Configuration
# ============================================================
@dataclass
class DoubleWellConfig:
    """Asymmetric double-well with optional quasi-1D confinement."""

    well_separation: float = 4.0
    omega_base: float = 1.0
    asymmetry_strength: float = 0.3
    lam: float = 0.0
    softening: float = 1e-6
    quasi_1d: bool = False
    omega_y_factor: float = 5.0

    @property
    def omega_left(self) -> float:
        return self.omega_base * (1.0 + self.lam * self.asymmetry_strength)

    @property
    def omega_right(self) -> float:
        return self.omega_base * (1.0 - self.lam * self.asymmetry_strength)

    @property
    def omega_y(self) -> float:
        return self.omega_base * self.omega_y_factor if self.quasi_1d else self.omega_base

    @property
    def ell_base(self) -> float:
        return 1.0 / math.sqrt(self.omega_base)

    @property
    def sep_physical(self) -> float:
        return self.well_separation * self.ell_base

    def E_nonint(self, nxL, nyL, nxR, nyR):
        """Non-interacting energy for config (nxL,nyL)_left ⊗ (nxR,nyR)_right."""
        wL, wR, wy = self.omega_left, self.omega_right, self.omega_y
        if self.quasi_1d:
            eL = wL * (nxL + 0.5) + wy * (nyL + 0.5)
            eR = wR * (nxR + 0.5) + wy * (nyR + 0.5)
        else:
            eL = wL * (nxL + nyL + 1)  # d=2 isotropic: ω(nx+ny+1)
            eR = wR * (nxR + nyR + 1)
        return eL + eR

    def detuning(self) -> float:
        """δ(λ) = E_10 - E_01 (non-interacting)."""
        return self.E_nonint(1, 0, 0, 0) - self.E_nonint(0, 0, 1, 0)


# ============================================================
# Potential
# ============================================================
def asymmetric_double_well_potential(x, cfg):
    sep = cfg.sep_physical
    r0 = x[:, 0, :].clone()
    r0[:, 0] += sep / 2
    r1 = x[:, 1, :].clone()
    r1[:, 0] -= sep / 2
    if cfg.quasi_1d:
        oy2 = cfg.omega_y**2
        V_L = 0.5 * cfg.omega_left**2 * r0[:, 0] ** 2 + 0.5 * oy2 * r0[:, 1] ** 2
        V_R = 0.5 * cfg.omega_right**2 * r1[:, 0] ** 2 + 0.5 * oy2 * r1[:, 1] ** 2
    else:
        V_L = 0.5 * cfg.omega_left**2 * (r0**2).sum(-1)
        V_R = 0.5 * cfg.omega_right**2 * (r1**2).sum(-1)
    return V_L + V_R


def softened_coulomb(x, eps=1e-6):
    r2 = ((x[:, 0, :] - x[:, 1, :]) ** 2).sum(-1)
    return 1.0 / torch.sqrt(r2 + eps**2)


# ============================================================
# Sampling
# ============================================================
def sample_positions(B, cfg, device=DEVICE, dtype=DTYPE):
    sep = cfg.sep_physical
    sL = 0.5 / math.sqrt(cfg.omega_left)
    sR = 0.5 / math.sqrt(cfg.omega_right)
    x = torch.zeros(B, N_PARTICLES, D, device=device, dtype=dtype)
    x[:, 0, :] = torch.randn(B, D, device=device, dtype=dtype) * sL
    x[:, 0, 0] -= sep / 2
    x[:, 1, :] = torch.randn(B, D, device=device, dtype=dtype) * sR
    x[:, 1, 0] += sep / 2
    if cfg.quasi_1d:
        sy = 0.5 / math.sqrt(cfg.omega_y)
        x[:, 0, 1] = torch.randn(B, device=device, dtype=dtype) * sy
        x[:, 1, 1] = torch.randn(B, device=device, dtype=dtype) * sy
    return x


def mcmc_equilibrate(x, psi_log_fn, sigma, n_steps=100):
    """In-place MCMC burn-in, returns (x, logp, n_accepted)."""
    B = x.shape[0]
    logp = 2.0 * psi_log_fn(x)
    acc_total = 0
    for _ in range(n_steps):
        xp = x + torch.randn_like(x) * sigma
        lp = 2.0 * psi_log_fn(xp)
        accept = torch.rand(B, device=x.device).log() < (lp - logp)
        x = torch.where(accept.view(-1, 1, 1), xp, x)
        logp = torch.where(accept, lp, logp)
        acc_total += int(accept.sum())
    return x, logp, acc_total / (B * n_steps)


# ============================================================
# Model Building
# ============================================================
def make_C_occ(nx=2, ny=2, n_occ=1, device=DEVICE, dtype=DTYPE):
    pairs = sorted(
        [(i, j) for i in range(nx) for j in range(ny)],
        key=lambda t: (t[0] + t[1], t[0]),
    )[:n_occ]
    C = torch.zeros(nx * ny, n_occ, dtype=dtype, device=device)
    for j, (ix, iy) in enumerate(pairs):
        C[ix * ny + iy, j] = 1.0
    return C


class _ZeroJastrow(nn.Module):
    def forward(self, x, spin=None):
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)


def build_model(omega=OMEGA_BASE, mode="full"):
    """Modes: 'full' (SD+BF+J), 'sd_jastrow' (SD+J), 'sd_only'."""
    if mode == "sd_only":
        f_net = _ZeroJastrow().to(DEVICE, DTYPE)
        bf_net = None
    else:
        f_net = PINN(
            n_particles=N_PARTICLES,
            d=D,
            omega=omega,
            dL=5,
            hidden_dim=128,
            n_layers=2,
            act="gelu",
            init="xavier",
            use_gate=True,
        ).to(DEVICE, DTYPE)
        bf_net = None
    if mode == "full":
        bf_net = CTNNBackflowNet(
            d=D,
            msg_hidden=128,
            msg_layers=2,
            hidden=128,
            layers=3,
            act="gelu",
            aggregation="mean",
            use_spin=True,
            same_spin_only=False,
            out_bound="tanh",
            bf_scale_init=0.3,
            zero_init_last=True,
            omega=omega,
        ).to(DEVICE, DTYPE)
    return f_net, bf_net


def _params_of(f_net, bf_net):
    ps = [p for p in f_net.parameters() if p.requires_grad]
    if bf_net is not None:
        ps += [p for p in bf_net.parameters() if p.requires_grad]
    return ps


def make_psi_log_fn(f_net, C_occ, bf_net, spin, params, cfg):
    sep = cfg.sep_physical

    def fn(x):
        xs = x.clone()
        xs[:, 0, 0] = x[:, 0, 0] + sep / 2
        xs[:, 1, 0] = x[:, 1, 0] - sep / 2
        logpsi, _ = psi_fn(f_net, xs, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return logpsi

    return fn


def make_psi_full_fn(f_net, C_occ, bf_net, spin, params, cfg):
    """Returns (logpsi, psi_with_sign) for overlap computations."""
    sep = cfg.sep_physical

    def fn(x):
        xs = x.clone()
        xs[:, 0, 0] = x[:, 0, 0] + sep / 2
        xs[:, 1, 0] = x[:, 1, 0] - sep / 2
        logpsi, psi = psi_fn(f_net, xs, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return logpsi, psi

    return fn


# ============================================================
# Energy Computation
# ============================================================
def compute_laplacian_logpsi(psi_log_fn, x):
    x = x.requires_grad_(True)
    lp = psi_log_fn(x)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True, retain_graph=True)[0]
    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            sec = torch.autograd.grad(g[:, i, j].sum(), x, create_graph=True, retain_graph=True)[0]
            lap += sec[:, i, j]
    return lap, (g**2).sum(dim=(1, 2)), g


def local_energy(psi_log_fn, x, cfg):
    lap, g2, _ = compute_laplacian_logpsi(psi_log_fn, x)
    T = -0.5 * (lap + g2)
    return T + asymmetric_double_well_potential(x, cfg) + softened_coulomb(x, cfg.softening), T


# ============================================================
# Training — Ground & Excited
# ============================================================
def train_ground_state(
    f_net, bf_net, C_occ, cfg, params, *, n_epochs=300, n_coll=256, lr=5e-4, print_every=100
):
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    all_p = _params_of(f_net, bf_net)
    has_p = len(all_p) > 0
    if has_p:
        opt = optim.Adam(all_p, lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr / 10)

    psi_log = make_psi_log_fn(f_net, C_occ, bf_net, spin, params, cfg)
    sig = 0.15 * cfg.ell_base
    best_E, best_st = float("inf"), None
    x = sample_positions(n_coll, cfg)
    n_run = n_epochs if has_p else min(n_epochs, 30)
    final_var = final_acc = 0.0

    for ep in range(n_run):
        f_net.train()
        if bf_net is not None:
            bf_net.train()
        with torch.no_grad():
            x, _, acc_rate = mcmc_equilibrate(x, psi_log, sig, 20)
        if has_p:
            opt.zero_grad()
            xb = x.detach().requires_grad_(True)
            EL, _ = local_energy(psi_log, xb, cfg)
            Em = EL.mean().detach()
            ((EL - Em) ** 2).mean().backward()
            torch.nn.utils.clip_grad_norm_(all_p, 1.0)
            opt.step()
            sch.step()
        else:
            with torch.set_grad_enabled(True):
                xb = x.detach().requires_grad_(True)
                EL, _ = local_energy(psi_log, xb, cfg)
            Em = EL.mean().detach()
        ev = float(Em)
        final_var = float(EL.var())
        final_acc = acc_rate
        if ev < best_E:
            best_E = ev
            best_st = {
                "f_net": copy.deepcopy(f_net.state_dict()),
                "bf_net": copy.deepcopy(bf_net.state_dict()) if bf_net else None,
            }
        if ep % print_every == 0:
            print(f"    Ep {ep:4d} | E={ev:.5f} ± {float(EL.std()):.4f} | acc={acc_rate:.2f}")

    if best_st:
        f_net.load_state_dict(best_st["f_net"])
        if bf_net and best_st["bf_net"]:
            bf_net.load_state_dict(best_st["bf_net"])
    diag = {"final_var": final_var, "final_acc": final_acc}
    return best_E, best_st, diag


def _overlap_penalty(psi_exc, psi_low, x):
    le = psi_exc(x)
    with torch.no_grad():
        ll = psi_low(x)
    return torch.exp(torch.clamp(2.0 * (ll - le), -40, 40)).mean()


def overlap_scalar(psi1, psi2, x):
    with torch.no_grad():
        return float(torch.exp(torch.clamp(psi2(x) - psi1(x), -20, 20)).mean())


def train_excited_state(
    f_net,
    bf_net,
    C_occ,
    cfg,
    params,
    lower_states,
    *,
    n_epochs=400,
    n_coll=256,
    lr=3e-4,
    penalty=50.0,
    print_every=100,
):
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    all_p = _params_of(f_net, bf_net)
    if not all_p:
        return float("inf"), None, {"final_var": 0, "final_acc": 0, "final_orthoverlaps": []}

    opt = optim.Adam(all_p, lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr / 10)
    psi_exc = make_psi_log_fn(f_net, C_occ, bf_net, spin, params, cfg)
    sig = 0.15 * cfg.ell_base
    best_E, best_st = float("inf"), None
    x = sample_positions(n_coll, cfg)
    final_var = final_acc = 0.0
    final_ovs = []

    for ep in range(n_epochs):
        f_net.train()
        if bf_net is not None:
            bf_net.train()
        with torch.no_grad():
            x, _, acc_rate = mcmc_equilibrate(x, psi_exc, sig, 20)
        opt.zero_grad()
        xb = x.detach().requires_grad_(True)
        EL, _ = local_energy(psi_exc, xb, cfg)
        Em = EL.mean()
        loss = ((EL - Em.detach()) ** 2).mean()
        orth = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        for _, _, pl in lower_states:
            orth = orth + _overlap_penalty(psi_exc, pl, x.detach())
        (loss + penalty * orth).backward()
        torch.nn.utils.clip_grad_norm_(all_p, 1.0)
        opt.step()
        sch.step()
        ev = float(Em.detach())
        final_var = float(EL.var())
        final_acc = acc_rate
        if ev < best_E:
            best_E = ev
            best_st = {
                "f_net": copy.deepcopy(f_net.state_dict()),
                "bf_net": copy.deepcopy(bf_net.state_dict()) if bf_net else None,
            }
        if ep % print_every == 0:
            ovs = [overlap_scalar(psi_exc, p, x.detach()) for _, _, p in lower_states]
            final_ovs = ovs
            print(f"    Ep {ep:4d} | E={ev:.5f} | ortho={ovs}")

    if best_st:
        f_net.load_state_dict(best_st["f_net"])
        if bf_net and best_st["bf_net"]:
            bf_net.load_state_dict(best_st["bf_net"])
    diag = {"final_var": final_var, "final_acc": final_acc, "final_orthoverlaps": final_ovs}
    return best_E, best_st, diag


# ============================================================
# Diagnostics — Energy
# ============================================================
def evaluate_energy_precise(psi_log_fn, cfg, n_samples=50000):
    sig = 0.12 * cfg.ell_base
    bs = 1024
    sE = sE2 = 0.0
    tot = 0
    x = sample_positions(bs, cfg)
    x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 50)
    while tot < n_samples:
        with torch.no_grad():
            x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 10)
        with torch.set_grad_enabled(True):
            xe = x.detach().requires_grad_(True)
            EL, _ = local_energy(psi_log_fn, xe, cfg)
        E = EL.detach()
        sE += float(E.sum())
        sE2 += float((E**2).sum())
        tot += bs
    mu = sE / tot
    var = max(sE2 / tot - mu**2, 0.0)
    return mu, math.sqrt(var / tot)


# ============================================================
# Diagnostics — Radial Classification (W_sub, a², b²)
# ============================================================
def compute_mixing(psi_log_fn, cfg, n_samples=5000):
    sig = 0.12 * cfg.ell_base
    bs = 1024
    sep = cfg.sep_physical
    sL = 1.0 / math.sqrt(cfg.omega_left)
    sR = 1.0 / math.sqrt(cfg.omega_right)
    x = sample_positions(bs, cfg)
    x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 100)
    thr_g, thr_e_lo, thr_e_hi = 1.2, 1.2, 3.0
    counts = {"00": 0, "10": 0, "01": 0, "11": 0, "other": 0}
    tot = 0
    while tot < n_samples:
        with torch.no_grad():
            x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 5)
            rL = torch.sqrt((x[:, 0, 0] + sep / 2) ** 2 + x[:, 0, 1] ** 2) / sL
            rR = torch.sqrt((x[:, 1, 0] - sep / 2) ** 2 + x[:, 1, 1] ** 2) / sR
            Lg, Le = rL < thr_g, (rL >= thr_e_lo) & (rL < thr_e_hi)
            Rg, Re = rR < thr_g, (rR >= thr_e_lo) & (rR < thr_e_hi)
            counts["00"] += int((Lg & Rg).sum())
            counts["10"] += int((Le & Rg).sum())
            counts["01"] += int((Lg & Re).sum())
            counts["11"] += int((Le & Re).sum())
            counts["other"] += int((~((Lg | Le) & (Rg | Re))).sum())
            tot += bs
    tc = sum(counts.values()) - counts["other"]
    p10 = counts["10"] / tc if tc > 0 else 0.25
    p01 = counts["01"] / tc if tc > 0 else 0.25
    W = p10 + p01
    a2 = p10 / W if W > 1e-6 else 0.5
    b2 = p01 / W if W > 1e-6 else 0.5
    return {
        "W_sub": W,
        "a_sq": a2,
        "b_sq": b2,
        "p_00": counts["00"] / tc if tc > 0 else 0.25,
        "p_10": p10,
        "p_01": p01,
        "p_11": counts["11"] / tc if tc > 0 else 0.25,
    }


def S_sub(a2, b2):
    """Subspace entanglement entropy S = -a² ln a² - b² ln b²."""
    s = 0.0
    if a2 > 1e-10:
        s -= a2 * math.log(a2)
    if b2 > 1e-10:
        s -= b2 * math.log(b2)
    return s


def compute_leakage(psi_log_fn, cfg, n_samples=5000):
    """L = ⟨y²⟩/⟨r²⟩.  L≈0.5: isotropic 2D.  L→0: quasi-1D."""
    sig = 0.12 * cfg.ell_base
    bs = 1024
    sep = cfg.sep_physical
    x = sample_positions(bs, cfg)
    x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 100)
    sy2 = sr2 = 0.0
    tot = 0
    while tot < n_samples:
        with torch.no_grad():
            x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 5)
            y2 = x[:, 0, 1] ** 2 + x[:, 1, 1] ** 2
            xd0 = x[:, 0, 0] + sep / 2
            xd1 = x[:, 1, 0] - sep / 2
            r2 = xd0**2 + x[:, 0, 1] ** 2 + xd1**2 + x[:, 1, 1] ** 2
            sy2 += float(y2.sum())
            sr2 += float(r2.sum())
            tot += bs
    return sy2 / sr2 if sr2 > 0 else 0.0


# ============================================================
# NEW: HO Basis Overlap Decomposition  (Task C)
# ============================================================
def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x), torch."""
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2.0 * x
    h0, h1 = torch.ones_like(x), 2.0 * x
    for k in range(2, n + 1):
        h0, h1 = h1, 2.0 * x * h1 - 2.0 * (k - 1) * h0
    return h1


def _ho_1d(n, x, omega):
    """1D HO ψ_n(x; ω), torch, unnormalized-safe."""
    xi = math.sqrt(omega) * x
    Hn = _hermite(n, xi)
    norm = (omega / math.pi) ** 0.25 / math.sqrt(2.0**n * math.factorial(n))
    return norm * torch.exp(-0.5 * omega * x**2) * Hn


# Basis configs: (nxL, nyL, nxR, nyR)
BASIS_CONFIGS = [
    (0, 0, 0, 0),  # |00⟩
    (1, 0, 0, 0),  # |10⟩  left x-excited
    (0, 0, 1, 0),  # |01⟩  right x-excited
    (0, 1, 0, 0),  # |10_y⟩  left y-excited
    (0, 0, 0, 1),  # |01_y⟩  right y-excited
    (1, 0, 1, 0),  # |11⟩  both x-excited
    (1, 1, 0, 0),  # |20⟩  left (1,1)
    (0, 0, 1, 1),  # |02⟩  right (1,1)
    (2, 0, 0, 0),  # left (2,0)
    (0, 0, 2, 0),  # right (2,0)
]

LABEL_MAP = {
    (0, 0, 0, 0): "|00⟩",
    (1, 0, 0, 0): "|10_x⟩",
    (0, 0, 1, 0): "|01_x⟩",
    (0, 1, 0, 0): "|10_y⟩",
    (0, 0, 0, 1): "|01_y⟩",
    (1, 0, 1, 0): "|11_xx⟩",
    (1, 1, 0, 0): "|20_xy⟩",
    (0, 0, 1, 1): "|02_xy⟩",
    (2, 0, 0, 0): "|20_x⟩",
    (0, 0, 2, 0): "|02_x⟩",
}


def _basis_state_value(x, cfg, nxL, nyL, nxR, nyR):
    """Φ_{config}(r1,r2) for batch x: (B, 2, 2)."""
    sep = cfg.sep_physical
    wL, wR = cfg.omega_left, cfg.omega_right
    wy = cfg.omega_y if cfg.quasi_1d else wL  # isotropic: ωy = ωx for each well

    x1x = x[:, 0, 0] + sep / 2
    x1y = x[:, 0, 1]
    x2x = x[:, 1, 0] - sep / 2
    x2y = x[:, 1, 1]

    if cfg.quasi_1d:
        phi = (
            _ho_1d(nxL, x1x, wL)
            * _ho_1d(nyL, x1y, wy)
            * _ho_1d(nxR, x2x, wR)
            * _ho_1d(nyR, x2y, wy)
        )
    else:
        phi = (
            _ho_1d(nxL, x1x, wL)
            * _ho_1d(nyL, x1y, wL)
            * _ho_1d(nxR, x2x, wR)
            * _ho_1d(nyR, x2y, wR)
        )
    return phi  # (B,)


def compute_basis_overlaps(psi_full_fn, psi_log_fn, cfg, n_samples=8000):
    """Compute |⟨Φ_i|ψ⟩|² for each basis config via MC from |ψ|².

    Uses: ⟨Φ|ψ⟩ = E_{|ψ|²}[Φ(r) × sign(ψ) / |ψ(r)|]
    """
    sig = 0.12 * cfg.ell_base
    bs = 1024
    x = sample_positions(bs, cfg)
    x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 100)

    # Accumulate overlap estimators
    overlap_sum = {c: 0.0 for c in BASIS_CONFIGS}
    tot = 0

    while tot < n_samples:
        with torch.no_grad():
            x, _, _ = mcmc_equilibrate(x, psi_log_fn, sig, 5)
            logpsi, psi_signed = psi_full_fn(x)
            sign = torch.sign(psi_signed)
            abs_psi = torch.exp(logpsi)  # |ψ(r)|
            ratio = sign / (abs_psi + 1e-30)  # sign(ψ)/|ψ|

            for c in BASIS_CONFIGS:
                phi = _basis_state_value(x, cfg, *c)
                overlap_sum[c] += float((phi * ratio).sum())
            tot += bs

    overlaps = {}
    for c in BASIS_CONFIGS:
        ov = overlap_sum[c] / tot
        overlaps[c] = ov**2  # |⟨Φ|ψ⟩|²

    # Normalize to probabilities
    total_w = sum(overlaps.values())
    if total_w > 0:
        for c in overlaps:
            overlaps[c] /= total_w

    return overlaps


def top_overlaps(overlaps, n=5):
    """Return top-n configs sorted by weight."""
    return sorted(overlaps.items(), key=lambda kv: -kv[1])[:n]


def extended_W_sub(overlaps, extra_configs=None):
    """W_sub in {|10_x⟩, |01_x⟩} + optional extra configs."""
    core = [(1, 0, 0, 0), (0, 0, 1, 0)]
    configs = core + (extra_configs or [])
    return sum(overlaps.get(c, 0.0) for c in configs)


# ============================================================
# NEW: Effective Hamiltonian (Task D)
# ============================================================
def compute_H_eff(cfg, configs=None, n_mc=50000):
    """H_eff_{ij} = ⟨Φ_i|H|Φ_j⟩ via MC integration.

    Non-interacting part: diagonal, analytic.
    Coulomb part: ⟨Φ_i|V_C|Φ_j⟩ via sampling from a broad Gaussian.
    """
    if configs is None:
        configs = [(1, 0, 0, 0), (0, 0, 1, 0)]  # minimal 2-level

    n = len(configs)
    sep = cfg.sep_physical

    # Diagonal: non-interacting energies
    H = np.zeros((n, n))
    for i, c in enumerate(configs):
        H[i, i] = cfg.E_nonint(*c)

    # Coulomb matrix elements via MC
    # Sample from product of broad Gaussians
    bs = 4096
    tot = 0
    V_acc = np.zeros((n, n))

    while tot < n_mc:
        sig_sample = 1.5 * cfg.ell_base
        x = torch.randn(bs, 2, 2, device=DEVICE, dtype=DTYPE) * sig_sample
        x[:, 0, 0] -= sep / 2
        x[:, 1, 0] += sep / 2

        with torch.no_grad():
            Vc = softened_coulomb(x, cfg.softening)

            # Importance weight: q(r) = product of Gaussians with sig_sample
            log_q = -(x**2).sum(dim=(1, 2)) / (2 * sig_sample**2)

            phis = []
            for c in configs:
                phi = _basis_state_value(x, cfg, *c)
                phis.append(phi)

            for i in range(n):
                for j in range(i, n):
                    integrand = phis[i] * Vc * phis[j]
                    # Weight by |Φ_i|²/q  (importance sampling correction)
                    # Actually: ∫ Φ_i V_C Φ_j dr = E_q[Φ_i V_C Φ_j / q]
                    # Direct MC: sample from q, accumulate
                    val = float(integrand.sum()) / bs
                    # Need to multiply by the volume element of q
                    # V_q = (2π sig²)^(d_total/2) = (2π sig²)^2
                    vol = (2 * math.pi * sig_sample**2) ** 2
                    V_acc[i, j] += val * vol * bs
                    if i != j:
                        V_acc[j, i] = V_acc[i, j]
            tot += bs

    V_coul = V_acc / tot * (2 * math.pi * (1.5 * cfg.ell_base) ** 2) ** 2
    H += V_coul

    # Eigenvalues
    eigvals = np.sort(np.linalg.eigvalsh(H))
    delta_eff = eigvals[-1] - eigvals[0] if len(eigvals) > 1 else 0.0

    return {
        "H_eff": H.tolist(),
        "eigvals": eigvals.tolist(),
        "delta_eff": delta_eff,
        "configs": [str(c) for c in configs],
    }


def compute_delta_eff_simple(cfg):
    """Δ_eff from 2×2 H_eff in {|10_x⟩,|01_x⟩} with analytic + numerical Coulomb."""
    sep = cfg.sep_physical

    E10 = cfg.E_nonint(1, 0, 0, 0)
    E01 = cfg.E_nonint(0, 0, 1, 0)

    # Coulomb coupling V_{10,01} via MC
    bs = 8192
    n_mc = 40000
    sig_s = 1.2 * cfg.ell_base
    V11 = V22 = V12 = 0.0
    tot = 0

    while tot < n_mc:
        x = torch.randn(bs, 2, 2, device=DEVICE, dtype=DTYPE) * sig_s
        x[:, 0, 0] -= sep / 2
        x[:, 1, 0] += sep / 2

        with torch.no_grad():
            Vc = softened_coulomb(x, cfg.softening)
            phi10 = _basis_state_value(x, cfg, 1, 0, 0, 0)
            phi01 = _basis_state_value(x, cfg, 0, 0, 1, 0)

            # ⟨Φ_i|V|Φ_j⟩ via direct MC from Gaussian sampling
            vol = (2 * math.pi * sig_s**2) ** 2
            V11 += float((phi10**2 * Vc).sum()) / bs * vol
            V22 += float((phi01**2 * Vc).sum()) / bs * vol
            V12 += float((phi10 * phi01 * Vc).sum()) / bs * vol
            tot += bs

    V11 /= tot / bs
    V22 /= tot / bs
    V12 /= tot / bs

    H = np.array([[E10 + V11, V12], [V12, E01 + V22]])
    eigs = np.sort(np.linalg.eigvalsh(H))
    delta_eff = eigs[1] - eigs[0]
    detuning = E10 - E01

    return {
        "H_eff": H.tolist(),
        "eigvals": eigs.tolist(),
        "delta_eff": float(delta_eff),
        "detuning": float(detuning),
        "V11": V11,
        "V22": V22,
        "V12": V12,
    }


# ============================================================
# Train all states at one λ → full result dict
# ============================================================
def train_all_states(
    cfg,
    C_occ,
    params,
    *,
    n_ep_g=200,
    n_ep_e=250,
    n_eval=10000,
    n_diag=5000,
    mode="full",
    warm=None,
    pev=100,
):
    """Returns (result_dict, model_states, psi_pair)."""
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    sd_only = mode == "sd_only"

    # Ground
    print("    [Ground]")
    f0, b0 = build_model(OMEGA_BASE, mode=mode)
    if warm and warm.get("ground"):
        _load_warm(f0, b0, warm["ground"])
    _, st0, dg0 = train_ground_state(f0, b0, C_occ, cfg, params, n_epochs=n_ep_g, print_every=pev)
    psi0 = make_psi_log_fn(f0, C_occ, b0, spin, params, cfg)
    E0, E0e = evaluate_energy_precise(psi0, cfg, n_eval)
    print(f"    E0 = {E0:.5f} ± {E0e:.5f}")

    if sd_only:
        return _pack_sd_only(cfg, E0, E0e, dg0), {"ground": st0, "lower": None, "upper": None}, None

    # Exc-1
    print("    [Exc-1]")
    f1, b1 = build_model(OMEGA_BASE, mode=mode)
    if warm and warm.get("lower"):
        _load_warm(f1, b1, warm["lower"])
    _, st1, dg1 = train_excited_state(
        f1, b1, C_occ, cfg, params, lower_states=[(f0, b0, psi0)], n_epochs=n_ep_e, print_every=pev
    )
    psi1 = make_psi_log_fn(f1, C_occ, b1, spin, params, cfg)
    psi1_full = make_psi_full_fn(f1, C_occ, b1, spin, params, cfg)
    E1, E1e = evaluate_energy_precise(psi1, cfg, n_eval)
    print(f"    E1 = {E1:.5f} ± {E1e:.5f}")

    # Exc-2
    print("    [Exc-2]")
    f2, b2 = build_model(OMEGA_BASE, mode=mode)
    if warm and warm.get("upper"):
        _load_warm(f2, b2, warm["upper"])
    _, st2, dg2 = train_excited_state(
        f2,
        b2,
        C_occ,
        cfg,
        params,
        lower_states=[(f0, b0, psi0), (f1, b1, psi1)],
        n_epochs=n_ep_e,
        print_every=pev,
    )
    psi2 = make_psi_log_fn(f2, C_occ, b2, spin, params, cfg)
    psi2_full = make_psi_full_fn(f2, C_occ, b2, spin, params, cfg)
    E2, E2e = evaluate_energy_precise(psi2, cfg, n_eval)
    print(f"    E2 = {E2:.5f} ± {E2e:.5f}")

    # Sort
    if E1 <= E2:
        Elo, Eloe, plo, stlo, dglo, pflo = E1, E1e, psi1, st1, dg1, psi1_full
        Ehi, Ehie, phi_, sthi, dghi, pfhi = E2, E2e, psi2, st2, dg2, psi2_full
    else:
        Elo, Eloe, plo, stlo, dglo, pflo = E2, E2e, psi2, st2, dg2, psi2_full
        Ehi, Ehie, phi_, sthi, dghi, pfhi = E1, E1e, psi1, st1, dg1, psi1_full
    gap = Ehi - Elo

    # Diagnostics
    print("    [Diag]")
    print(f"      Δ = {gap:.5f}")

    mx_lo = compute_mixing(plo, cfg, n_diag)
    mx_hi = compute_mixing(phi_, cfg, n_diag)
    Slo = S_sub(mx_lo["a_sq"], mx_lo["b_sq"])
    Shi = S_sub(mx_hi["a_sq"], mx_hi["b_sq"])
    lk0 = compute_leakage(psi0, cfg, n_diag)
    lk_lo = compute_leakage(plo, cfg, n_diag)
    lk_hi = compute_leakage(phi_, cfg, n_diag)

    # Basis overlaps
    ov_lo = compute_basis_overlaps(pflo, plo, cfg, n_samples=min(n_diag, 4000))
    ov_hi = compute_basis_overlaps(pfhi, phi_, cfg, n_samples=min(n_diag, 4000))

    # Extended W_sub: add top-2 non-core configs
    top_extra_lo = [c for c, _ in top_overlaps(ov_lo, 5) if c not in [(1, 0, 0, 0), (0, 0, 1, 0)]][
        :2
    ]
    top_extra_hi = [c for c, _ in top_overlaps(ov_hi, 5) if c not in [(1, 0, 0, 0), (0, 0, 1, 0)]][
        :2
    ]
    W_ext_lo = extended_W_sub(ov_lo, top_extra_lo)
    W_ext_hi = extended_W_sub(ov_hi, top_extra_hi)

    # H_eff
    heff = compute_delta_eff_simple(cfg)

    detuning = cfg.detuning()
    W_clean = min(mx_lo["W_sub"], mx_hi["W_sub"]) >= W_SUB_THRESHOLD

    print(
        f"      Lo: W={mx_lo['W_sub']:.3f} a²={mx_lo['a_sq']:.3f} b²={mx_lo['b_sq']:.3f} S={Slo:.4f}"
    )
    print(
        f"      Hi: W={mx_hi['W_sub']:.3f} a²={mx_hi['a_sq']:.3f} b²={mx_hi['b_sq']:.3f} S={Shi:.4f}"
    )
    print(f"      W_ext: lo={W_ext_lo:.3f} hi={W_ext_hi:.3f}")
    print(f"      Δ_eff={heff['delta_eff']:.5f}  δ(λ)={detuning:+.5f}  clean={W_clean}")
    print(f"      Leak: E0={lk0:.3f} lo={lk_lo:.3f} hi={lk_hi:.3f}")

    # Top overlaps for logging
    top_lo_str = "; ".join(f"{LABEL_MAP.get(c, str(c))}={w:.3f}" for c, w in top_overlaps(ov_lo, 5))
    top_hi_str = "; ".join(f"{LABEL_MAP.get(c, str(c))}={w:.3f}" for c, w in top_overlaps(ov_hi, 5))
    print(f"      Overlaps lo: {top_lo_str}")
    print(f"      Overlaps hi: {top_hi_str}")

    result = {
        "lambda": cfg.lam,
        "omega_left": cfg.omega_left,
        "omega_right": cfg.omega_right,
        "quasi_1d": cfg.quasi_1d,
        "detuning": detuning,
        "E0": E0,
        "E0_err": E0e,
        "E_lower": Elo,
        "E_lower_err": Eloe,
        "E_upper": Ehi,
        "E_upper_err": Ehie,
        "gap": gap,
        "lower_W_sub": mx_lo["W_sub"],
        "lower_a_sq": mx_lo["a_sq"],
        "lower_b_sq": mx_lo["b_sq"],
        "upper_W_sub": mx_hi["W_sub"],
        "upper_a_sq": mx_hi["a_sq"],
        "upper_b_sq": mx_hi["b_sq"],
        "S_sub_lower": Slo,
        "S_sub_upper": Shi,
        "E0_leakage": lk0,
        "lower_leakage": lk_lo,
        "upper_leakage": lk_hi,
        "two_level_clean": W_clean,
        "delta_eff": heff["delta_eff"],
        "H_eff": heff["H_eff"],
        "W_ext_lower": W_ext_lo,
        "W_ext_upper": W_ext_hi,
        "basis_overlaps_lo": {str(k): v for k, v in ov_lo.items()},
        "basis_overlaps_hi": {str(k): v for k, v in ov_hi.items()},
        "train_diag_ground": dg0,
        "train_diag_lower": dglo,
        "train_diag_upper": dghi,
    }

    new_warm = {"ground": st0, "lower": stlo, "upper": sthi}
    return result, new_warm, (plo, phi_)


def _load_warm(f, b, st):
    if st is None:
        return
    try:
        f.load_state_dict(st["f_net"])
    except Exception:
        pass
    if b is not None and st.get("bf_net"):
        try:
            b.load_state_dict(st["bf_net"])
        except Exception:
            pass


def _pack_sd_only(cfg, E0, E0e, dg):
    return {
        "lambda": cfg.lam,
        "omega_left": cfg.omega_left,
        "omega_right": cfg.omega_right,
        "quasi_1d": cfg.quasi_1d,
        "detuning": cfg.detuning(),
        "E0": E0,
        "E0_err": E0e,
        "E_lower": None,
        "E_lower_err": None,
        "E_upper": None,
        "E_upper_err": None,
        "gap": None,
        "lower_W_sub": None,
        "lower_a_sq": None,
        "lower_b_sq": None,
        "upper_W_sub": None,
        "upper_a_sq": None,
        "upper_b_sq": None,
        "S_sub_lower": None,
        "S_sub_upper": None,
        "E0_leakage": None,
        "lower_leakage": None,
        "upper_leakage": None,
        "two_level_clean": False,
        "delta_eff": None,
        "H_eff": None,
        "W_ext_lower": None,
        "W_ext_upper": None,
        "basis_overlaps_lo": None,
        "basis_overlaps_hi": None,
        "train_diag_ground": dg,
        "train_diag_lower": None,
        "train_diag_upper": None,
    }


# ============================================================
# Branch overlap helper
# ============================================================
def _branch_overlap(pair, prev, cfg):
    x = sample_positions(1024, cfg)
    x, _, _ = mcmc_equilibrate(x, pair[0], 0.12, 50)
    return overlap_scalar(pair[0], prev[0], x), overlap_scalar(pair[1], prev[1], x)


# ============================================================
# Dense Sweep  (Task A + B + E)
# ============================================================
def run_dense_sweep(
    lambda_values=None,
    *,
    n_seeds=5,
    n_ep_g=200,
    n_ep_e=250,
    n_ep_g_warm=100,
    n_ep_e_warm=150,
    n_eval=10000,
    n_diag=5000,
    quasi_1d=False,
    omega_y_factor=5.0,
):
    if lambda_values is None:
        lambda_values = [round(v, 2) for v in np.arange(-0.10, 0.101, 0.02)]

    tag = f"DENSE SWEEP  {'quasi-1D' if quasi_1d else '2D'}  {n_seeds} seeds"
    print("=" * 70)
    print(tag)
    print("=" * 70)
    print(f"  λ: {lambda_values}")
    print(f"  cold: g={n_ep_g} e={n_ep_e}  warm: g={n_ep_g_warm} e={n_ep_e_warm}")
    sys.stdout.flush()

    _init_config()
    params = config.get().as_dict()
    C = make_C_occ()
    all_seeds = []

    for seed in range(n_seeds):
        print(f"\n{'#' * 70}\n  SEED {seed}\n{'#' * 70}")
        sys.stdout.flush()
        torch.manual_seed(seed * 42)
        warm = None
        prev_psi = None
        sres = []

        for i, lam in enumerate(lambda_values):
            print(f"\n  === λ={lam:+.3f}  (seed {seed}, {i + 1}/{len(lambda_values)}) ===")
            sys.stdout.flush()
            cfg = DoubleWellConfig(lam=lam, quasi_1d=quasi_1d, omega_y_factor=omega_y_factor)
            use_w = warm is not None
            ng = n_ep_g_warm if use_w else n_ep_g
            ne = n_ep_e_warm if use_w else n_ep_e

            result, new_warm, pair = train_all_states(
                cfg,
                C,
                params,
                n_ep_g=ng,
                n_ep_e=ne,
                n_eval=n_eval,
                n_diag=n_diag,
                warm=warm if use_w else None,
                pev=max(ng, ne),
            )

            # Hop detection + retrain-on-hop
            hop = retrained = False
            if prev_psi and pair:
                ov_lo, ov_hi = _branch_overlap(pair, prev_psi, cfg)
                hop = abs(ov_lo) < 0.3 or abs(ov_hi) < 0.3
                print(f"      Branch: lo={ov_lo:.3f} hi={ov_hi:.3f} {'⚠ HOP' if hop else '✓'}")
                if hop:
                    print("      → Retraining cold")
                    sys.stdout.flush()
                    result, new_warm, pair = train_all_states(
                        cfg,
                        C,
                        params,
                        n_ep_g=n_ep_g,
                        n_ep_e=n_ep_e,
                        n_eval=n_eval,
                        n_diag=n_diag,
                        warm=None,
                        pev=max(n_ep_g, n_ep_e),
                    )
                    retrained = True
                    if pair:
                        ov_lo, ov_hi = _branch_overlap(pair, prev_psi, cfg)
                        print(f"      Post-retrain: lo={ov_lo:.3f} hi={ov_hi:.3f}")
                result["overlap_lo_prev"] = ov_lo
                result["overlap_hi_prev"] = ov_hi
            else:
                result["overlap_lo_prev"] = result["overlap_hi_prev"] = None
            result["hop_flag"] = hop
            result["retrained"] = retrained
            result["seed"] = seed

            sres.append(result)
            warm = new_warm
            if pair:
                prev_psi = pair
            sys.stdout.flush()

        all_seeds.append(sres)
        sfx = "_q1d" if quasi_1d else "_2d"
        with open(RESULTS_DIR / f"dense_seed{seed}{sfx}.json", "w") as f:
            json.dump(sres, f, indent=2, default=str)

    agg = _aggregate(all_seeds, lambda_values)
    with open(RESULTS_DIR / f"dense_agg{sfx}.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)
    return agg, all_seeds


def _aggregate(all_seeds, lams):
    agg = []
    for i, lam in enumerate(lams):
        rows = [s[i] for s in all_seeds if i < len(s)]

        def _s(k, _rows=rows):
            vs = [r[k] for r in _rows if r.get(k) is not None]
            return (float(np.mean(vs)), float(np.std(vs))) if vs else (None, None)

        entry = {
            "lambda": lam,
            "n_seeds": len(rows),
            "detuning": _s("detuning")[0],
            "E0_mean": _s("E0")[0],
            "E0_std": _s("E0")[1],
            "E_lower_mean": _s("E_lower")[0],
            "E_lower_std": _s("E_lower")[1],
            "E_upper_mean": _s("E_upper")[0],
            "E_upper_std": _s("E_upper")[1],
            "gap_mean": _s("gap")[0],
            "gap_std": _s("gap")[1],
            "lower_W_sub_mean": _s("lower_W_sub")[0],
            "lower_W_sub_std": _s("lower_W_sub")[1],
            "upper_W_sub_mean": _s("upper_W_sub")[0],
            "upper_W_sub_std": _s("upper_W_sub")[1],
            "lower_a_sq_mean": _s("lower_a_sq")[0],
            "lower_a_sq_std": _s("lower_a_sq")[1],
            "lower_b_sq_mean": _s("lower_b_sq")[0],
            "lower_b_sq_std": _s("lower_b_sq")[1],
            "S_sub_lower_mean": _s("S_sub_lower")[0],
            "S_sub_lower_std": _s("S_sub_lower")[1],
            "S_sub_upper_mean": _s("S_sub_upper")[0],
            "S_sub_upper_std": _s("S_sub_upper")[1],
            "E0_leakage_mean": _s("E0_leakage")[0],
            "lower_leakage_mean": _s("lower_leakage")[0],
            "upper_leakage_mean": _s("upper_leakage")[0],
            "delta_eff_mean": _s("delta_eff")[0],
            "delta_eff_std": _s("delta_eff")[1],
            "W_ext_lower_mean": _s("W_ext_lower")[0],
            "W_ext_upper_mean": _s("W_ext_upper")[0],
            "hop_rate": sum(1 for r in rows if r.get("hop_flag")) / max(len(rows), 1),
            "two_level_clean": all(r.get("two_level_clean", False) for r in rows),
        }
        agg.append(entry)
    return agg


# ============================================================
# Ablation  (Task F)
# ============================================================
def run_ablation(lambda_values=None, *, n_ep_g=200, n_ep_e=250, n_eval=8000, n_diag=4000):
    if lambda_values is None:
        lambda_values = [-0.06, -0.02, 0.0, 0.02, 0.06]
    modes = ["sd_only", "sd_jastrow", "full"]
    labels = {"sd_only": "SD", "sd_jastrow": "SD+J", "full": "SD+BF+J"}
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    sys.stdout.flush()

    _init_config()
    params = config.get().as_dict()
    C = make_C_occ()
    ablation = {}

    for m in modes:
        print(f"\n{'#' * 70}\n  {labels[m]}\n{'#' * 70}")
        torch.manual_seed(0)
        mres = []
        for lam in lambda_values:
            cfg = DoubleWellConfig(lam=lam)
            print(f"\n  λ={lam:+.3f}  [{labels[m]}]")
            sys.stdout.flush()
            r, _, _ = train_all_states(
                cfg,
                C,
                params,
                n_ep_g=n_ep_g,
                n_ep_e=n_ep_e,
                n_eval=n_eval,
                n_diag=n_diag,
                mode=m,
                pev=max(n_ep_g, n_ep_e),
            )
            r["model"] = m
            mres.append(r)
        ablation[m] = mres

    with open(RESULTS_DIR / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2, default=str)
    return ablation


# ============================================================
# Helper
# ============================================================
def _init_config():
    config.update(
        device=DEVICE, omega=OMEGA_BASE, n_particles=N_PARTICLES, d=D, basis="cart", nx=2, ny=2
    )


# ============================================================
# Publication Figures  (Task G)
# ============================================================
def make_all_figures(agg, agg_q1d=None, ablation=None, save_dir=RESULTS_DIR):
    """Generate Figures 1–5 for the paper."""
    _fig1_energy_spectrum(agg, save_dir)
    _fig2_mixing_entanglement(agg, save_dir)
    _fig3_wsub_leakage(agg, save_dir)
    _fig4_delta_eff(agg, agg_q1d, save_dir)
    if ablation:
        _fig5_ablation(ablation, save_dir)
    print("All figures saved.")


def _extract(agg, key, err_key=None):
    vals = [r.get(key) or 0 for r in agg]
    errs = [r.get(err_key) or 0 for r in agg] if err_key else None
    return vals, errs


def _fig1_energy_spectrum(agg, sd):
    lams = [r["lambda"] for r in agg]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    E0, _ = _extract(agg, "E0_mean")
    Elo, Eloe = _extract(agg, "E_lower_mean", "E_lower_std")
    Ehi, Ehie = _extract(agg, "E_upper_mean", "E_upper_std")
    ax1.plot(lams, E0, "b-o", label=r"$E_0$", ms=4)
    ax1.errorbar(lams, Elo, yerr=Eloe, fmt="r-s", label=r"$E_{\rm lo}$", ms=4, capsize=3)
    ax1.errorbar(lams, Ehi, yerr=Ehie, fmt="g-^", label=r"$E_{\rm hi}$", ms=4, capsize=3)
    ax1.set(xlabel=r"$\lambda$", ylabel="Energy", title=r"(a) Energy spectrum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    gaps, gape = _extract(agg, "gap_mean", "gap_std")
    ax2.errorbar(lams, gaps, yerr=gape, fmt="k-o", ms=5, capsize=3)
    valid = [(g, lv) for g, lv in zip(gaps, lams, strict=False) if g]
    if valid:
        mi = min(range(len(valid)), key=lambda j: valid[j][0])
        ax2.axvline(
            valid[mi][1],
            color="r",
            ls="--",
            alpha=0.5,
            label=rf"$\Delta^*={valid[mi][0]:.4f}$ at $\lambda={valid[mi][1]:.2f}$",
        )
    ax2.set(xlabel=r"$\lambda$", ylabel=r"$\Delta$", title=r"(b) Gap $\Delta(\lambda)$")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sd / "fig1_energy.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(sd / "fig1_energy.png", dpi=150, bbox_inches="tight")
    plt.close()


def _fig2_mixing_entanglement(agg, sd):
    lams = [r["lambda"] for r in agg]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    a2, a2e = _extract(agg, "lower_a_sq_mean", "lower_a_sq_std")
    b2, b2e = _extract(agg, "lower_b_sq_mean", "lower_b_sq_std")
    ax1.errorbar(lams, a2, yerr=a2e, fmt="b-o", label=r"$|a|^2$ (left)", ms=4, capsize=3)
    ax1.errorbar(lams, b2, yerr=b2e, fmt="r-s", label=r"$|b|^2$ (right)", ms=4, capsize=3)
    ax1.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax1.set(xlabel=r"$\lambda$", ylabel="Weight", title=r"(a) Mixing: $|a|^2, |b|^2$ (lower)")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    Slo, Sloe = _extract(agg, "S_sub_lower_mean", "S_sub_lower_std")
    Shi, Shie = _extract(agg, "S_sub_upper_mean", "S_sub_upper_std")
    ax2.errorbar(lams, Slo, yerr=Sloe, fmt="r-s", label=r"$S_{\rm sub}^{\rm lo}$", ms=4, capsize=3)
    ax2.errorbar(lams, Shi, yerr=Shie, fmt="g-^", label=r"$S_{\rm sub}^{\rm hi}$", ms=4, capsize=3)
    ax2.axhline(math.log(2), color="gray", ls=":", alpha=0.5, label=r"$\ln 2$")
    ax2.set(xlabel=r"$\lambda$", ylabel=r"$S_{\rm sub}$", title="(b) Subspace entanglement")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sd / "fig2_mixing.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(sd / "fig2_mixing.png", dpi=150, bbox_inches="tight")
    plt.close()


def _fig3_wsub_leakage(agg, sd):
    lams = [r["lambda"] for r in agg]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    Wlo, Wloe = _extract(agg, "lower_W_sub_mean", "lower_W_sub_std")
    Whi, _ = _extract(agg, "upper_W_sub_mean")
    Wext_lo, _ = _extract(agg, "W_ext_lower_mean")
    Wext_hi, _ = _extract(agg, "W_ext_upper_mean")
    ax1.plot(lams, Wlo, "r-s", label=r"$W_{\rm sub}^{\rm lo}$", ms=4)
    ax1.plot(lams, Whi, "g-^", label=r"$W_{\rm sub}^{\rm hi}$", ms=4)
    ax1.plot(lams, Wext_lo, "r--s", label=r"$W_{\rm ext}^{\rm lo}$", ms=3, alpha=0.6)
    ax1.plot(lams, Wext_hi, "g--^", label=r"$W_{\rm ext}^{\rm hi}$", ms=3, alpha=0.6)
    ax1.axhline(W_SUB_THRESHOLD, color="gray", ls=":", label=f"threshold={W_SUB_THRESHOLD}")
    ax1.set(xlabel=r"$\lambda$", ylabel=r"$W$", title=r"(a) Subspace weight")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    lk0, _ = _extract(agg, "E0_leakage_mean")
    lklo, _ = _extract(agg, "lower_leakage_mean")
    lkhi, _ = _extract(agg, "upper_leakage_mean")
    ax2.plot(lams, lk0, "b-o", label=r"$E_0$", ms=4)
    ax2.plot(lams, lklo, "r-s", label=r"$E_{\rm lo}$", ms=4)
    ax2.plot(lams, lkhi, "g-^", label=r"$E_{\rm hi}$", ms=4)
    ax2.axhline(0.5, color="gray", ls=":", label="2D isotropic")
    ax2.set(
        xlabel=r"$\lambda$",
        ylabel=r"$\langle y^2\rangle/\langle r^2\rangle$",
        title="(b) Transverse leakage",
    )
    ax2.set_ylim(0, 0.6)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sd / "fig3_wsub_leakage.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(sd / "fig3_wsub_leakage.png", dpi=150, bbox_inches="tight")
    plt.close()


def _fig4_delta_eff(agg, agg_q1d, sd):
    lams = [r["lambda"] for r in agg]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    gaps, gape = _extract(agg, "gap_mean", "gap_std")
    deff, deffe = _extract(agg, "delta_eff_mean", "delta_eff_std")
    ax1.errorbar(lams, gaps, yerr=gape, fmt="k-o", ms=5, capsize=3, label=r"$\Delta$ (full VMC)")
    ax1.errorbar(
        lams, deff, yerr=deffe, fmt="b-s", ms=4, capsize=3, label=r"$\Delta_{\rm eff}$ (2-level)"
    )
    ax1.set(xlabel=r"$\lambda$", ylabel="Gap", title=r"(a) Full gap vs $\Delta_{\rm eff}$")
    ax1.set_ylim(bottom=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if agg_q1d:
        lams_q = [r["lambda"] for r in agg_q1d]
        gaps_q, gape_q = _extract(agg_q1d, "gap_mean", "gap_std")
        deff_q, deffe_q = _extract(agg_q1d, "delta_eff_mean", "delta_eff_std")
        ax2.errorbar(lams, gaps, yerr=gape, fmt="k-o", ms=4, capsize=3, label=r"2D $\Delta$")
        ax2.errorbar(lams_q, gaps_q, yerr=gape_q, fmt="r-s", ms=4, capsize=3, label=r"q1D $\Delta$")
        ax2.errorbar(
            lams,
            deff,
            yerr=deffe,
            fmt="k--o",
            ms=3,
            capsize=2,
            alpha=0.5,
            label=r"2D $\Delta_{\rm eff}$",
        )
        ax2.errorbar(
            lams_q,
            deff_q,
            yerr=deffe_q,
            fmt="r--s",
            ms=3,
            capsize=2,
            alpha=0.5,
            label=r"q1D $\Delta_{\rm eff}$",
        )
        ax2.set(xlabel=r"$\lambda$", ylabel="Gap", title="(b) 2D vs quasi-1D")
        ax2.set_ylim(bottom=0)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
    else:
        det, _ = _extract(agg, "detuning")
        ax2.plot(lams, det, "k-o", ms=5)
        ax2.axhline(0, color="r", ls="--", alpha=0.5)
        ax2.set(
            xlabel=r"$\lambda$", ylabel=r"$\delta(\lambda)$", title="(b) Non-interacting detuning"
        )
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sd / "fig4_delta_eff.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(sd / "fig4_delta_eff.png", dpi=150, bbox_inches="tight")
    plt.close()


def _fig5_ablation(ablation, sd):
    labels = {"sd_only": "SD", "sd_jastrow": "SD+J", "full": "SD+BF+J"}
    colors = {"sd_only": "C0", "sd_jastrow": "C1", "full": "C2"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for m, res in ablation.items():
        lams = [r["lambda"] for r in res]
        c, lb = colors[m], labels[m]
        gaps = [r.get("gap") for r in res]
        W = [r.get("lower_W_sub") for r in res]
        S = [r.get("S_sub_lower") for r in res]
        de = [r.get("delta_eff") for r in res]

        if any(v is None for v in gaps):
            continue
        axes[0, 0].plot(lams, gaps, "-o", color=c, label=lb, ms=5)
        axes[0, 1].plot(lams, W, "-o", color=c, label=lb, ms=5)
        axes[1, 0].plot(lams, S, "-o", color=c, label=lb, ms=5)
        if de and de[0] is not None:
            axes[1, 1].plot(lams, de, "-o", color=c, label=lb, ms=5)

    ttl = [
        ("(a) Gap", r"$\Delta$"),
        ("(b) $W_{\\rm sub}$", r"$W_{\\rm sub}$"),
        ("(c) $S_{\\rm sub}$", r"$S_{\\rm sub}$"),
        ("(d) $\\Delta_{\\rm eff}$", r"$\\Delta_{\\rm eff}$"),
    ]
    for ax, (t, yl) in zip(axes.flat, ttl, strict=False):
        ax.set(xlabel=r"$\lambda$", ylabel=yl, title=t)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].axhline(W_SUB_THRESHOLD, color="gray", ls=":", alpha=0.5)
    axes[1, 0].axhline(math.log(2), color="gray", ls=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(sd / "fig5_ablation.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(sd / "fig5_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Convenience modes
# ============================================================
def tiny_test():
    agg, _ = run_dense_sweep(
        [-0.5, 0.0, 0.5],
        n_seeds=1,
        n_ep_g=50,
        n_ep_e=80,
        n_ep_g_warm=50,
        n_ep_e_warm=80,
        n_eval=3000,
        n_diag=2000,
    )
    make_all_figures(agg)
    print("\nTINY COMPLETE")


def quick_test():
    agg, _ = run_dense_sweep(
        [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6],
        n_seeds=1,
        n_ep_g=150,
        n_ep_e=200,
        n_ep_g_warm=100,
        n_ep_e_warm=150,
        n_eval=8000,
        n_diag=4000,
    )
    make_all_figures(agg)
    print("\nQUICK COMPLETE")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Avoided Crossing v3")
    p.add_argument("--dense", action="store_true", help="Dense 11-pt sweep")
    p.add_argument("--ablation", action="store_true", help="Ablation near resonance")
    p.add_argument("--quasi1d", action="store_true", help="Quasi-1D (ω_y/ω_x=5)")
    p.add_argument("--seeds", type=int, default=5, help="Seeds (default 5)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()

    if args.tiny:
        tiny_test()
    elif args.quick:
        quick_test()
    elif args.dense:
        agg, _ = run_dense_sweep(n_seeds=args.seeds, quasi_1d=args.quasi1d)
        make_all_figures(agg)
    elif args.ablation:
        ab = run_ablation()
        _fig5_ablation(ab, RESULTS_DIR)
    else:
        print("Running: 2D dense → quasi-1D dense → ablation → figures\n")
        agg_2d, _ = run_dense_sweep(n_seeds=args.seeds, quasi_1d=False)
        agg_q1d, _ = run_dense_sweep(n_seeds=args.seeds, quasi_1d=True)
        ab = run_ablation()
        make_all_figures(agg_2d, agg_q1d, ab)
