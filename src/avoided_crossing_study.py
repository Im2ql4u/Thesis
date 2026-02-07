#!/usr/bin/env python
"""
Avoided Crossing Study v2 — Comparison with Jonny's Thesis

Improvements over v1:
1. Dense λ grid near resonance (11 points in [-0.10, 0.10])
2. Warm-starting from adjacent λ to stabilise branch tracking
3. Branch-overlap hop detection with automatic flagging
4. Multi-seed error bars on Δ(λ) near resonance
5. Quasi-1D confinement option (ω_y >> ω_x) to push W_sub ≳ 0.8
6. Proper 2-level entanglement: S_sub = -a² ln(a²) - b² ln(b²)
7. Ablation study: SD / SD+Jastrow / SD+BF+Jastrow near resonance

Usage:
    python avoided_crossing_study.py --dense             # Dense sweep (~2h)
    python avoided_crossing_study.py --dense --quasi1d   # Quasi-1D regime
    python avoided_crossing_study.py --ablation          # Ablation (~1h)
    python avoided_crossing_study.py --quick             # Quick 7-pt (~40m)
    python avoided_crossing_study.py --tiny              # Smoke test (~5m)
"""

import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_MANUAL_DEVICE"] = "0"

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

RESULTS_DIR = Path("/Users/aleksandersekkelsten/thesis/results/avoided_crossing")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Double-Well Configuration
# ============================================================


@dataclass
class DoubleWellConfig:
    """Asymmetric double-well with optional quasi-1D confinement.

    ω_left  = ω_base × (1 + λ α)
    ω_right = ω_base × (1 − λ α)

    When quasi_1d=True the transverse (y) confinement is tightened:
        ω_y = ω_base × omega_y_factor   (λ-independent, same for both wells)
    This freezes out y-modes and pushes W_sub toward 1.
    """

    well_separation: float = 4.0
    omega_base: float = 1.0
    asymmetry_strength: float = 0.3
    lam: float = 0.0
    softening: float = 1e-6
    quasi_1d: bool = False
    omega_y_factor: float = 5.0  # only used when quasi_1d=True

    @property
    def omega_left(self) -> float:
        return self.omega_base * (1.0 + self.lam * self.asymmetry_strength)

    @property
    def omega_right(self) -> float:
        return self.omega_base * (1.0 - self.lam * self.asymmetry_strength)

    @property
    def omega_y(self) -> float:
        if self.quasi_1d:
            return self.omega_base * self.omega_y_factor
        return self.omega_base  # not actually used in isotropic mode

    @property
    def ell_base(self) -> float:
        return 1.0 / math.sqrt(self.omega_base)

    @property
    def sep_physical(self) -> float:
        return self.well_separation * self.ell_base

    def expected_E_asymptotic(self) -> tuple[float, float, float]:
        """Non-interacting energies (E_00, E_10, E_01)."""
        if self.quasi_1d:
            ey = 0.5 * self.omega_y
            E_L0 = 0.5 * self.omega_left + ey
            E_R0 = 0.5 * self.omega_right + ey
        else:
            E_L0 = self.omega_left  # d/2 * ω  with d=2
            E_R0 = self.omega_right
        E_L1 = E_L0 + self.omega_left
        E_R1 = E_R0 + self.omega_right
        return E_L0 + E_R0, E_L1 + E_R0, E_L0 + E_R1


# ============================================================
# Potential
# ============================================================


def asymmetric_double_well_potential(
    x: torch.Tensor, cfg: DoubleWellConfig,
) -> torch.Tensor:
    """V_trap for asymmetric double-well.  Supports isotropic and quasi-1D."""
    sep = cfg.sep_physical
    r0 = x[:, 0, :].clone()
    r0[:, 0] += sep / 2
    r1 = x[:, 1, :].clone()
    r1[:, 0] -= sep / 2

    if cfg.quasi_1d:
        oy2 = (cfg.omega_base * cfg.omega_y_factor) ** 2
        V_left = (
            0.5 * cfg.omega_left ** 2 * r0[:, 0] ** 2
            + 0.5 * oy2 * r0[:, 1] ** 2
        )
        V_right = (
            0.5 * cfg.omega_right ** 2 * r1[:, 0] ** 2
            + 0.5 * oy2 * r1[:, 1] ** 2
        )
    else:
        V_left = 0.5 * cfg.omega_left ** 2 * (r0 ** 2).sum(dim=-1)
        V_right = 0.5 * cfg.omega_right ** 2 * (r1 ** 2).sum(dim=-1)

    return V_left + V_right


def softened_coulomb(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r2 = ((x[:, 0, :] - x[:, 1, :]) ** 2).sum(dim=-1)
    return 1.0 / torch.sqrt(r2 + eps ** 2)


# ============================================================
# Sampling
# ============================================================


def sample_asymmetric_positions(B, cfg, device=DEVICE, dtype=DTYPE):
    sep = cfg.sep_physical
    sig_L = 0.5 / math.sqrt(cfg.omega_left)
    sig_R = 0.5 / math.sqrt(cfg.omega_right)

    x = torch.zeros(B, N_PARTICLES, D, device=device, dtype=dtype)
    x[:, 0, :] = torch.randn(B, D, device=device, dtype=dtype) * sig_L
    x[:, 0, 0] -= sep / 2
    x[:, 1, :] = torch.randn(B, D, device=device, dtype=dtype) * sig_R
    x[:, 1, 0] += sep / 2

    if cfg.quasi_1d:
        sig_y = 0.5 / math.sqrt(cfg.omega_base * cfg.omega_y_factor)
        x[:, 0, 1] = torch.randn(B, device=device, dtype=dtype) * sig_y
        x[:, 1, 1] = torch.randn(B, device=device, dtype=dtype) * sig_y

    return x


# ============================================================
# Model Building
# ============================================================


def make_cartesian_C_occ(nx, ny, n_occ, device=DEVICE, dtype=DTYPE):
    pairs = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    pairs.sort(key=lambda t: (t[0] + t[1], t[0]))
    sel = pairs[:n_occ]
    cols = [ix * ny + iy for (ix, iy) in sel]
    C = torch.zeros(nx * ny, n_occ, dtype=dtype, device=device)
    for j, c in enumerate(cols):
        C[c, j] = 1.0
    return C


class _ZeroJastrow(nn.Module):
    """f(x) ≡ 0  — gives bare Slater determinant.  Accepts spin kwarg."""

    def forward(self, x, spin=None):
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)


def build_model(omega=OMEGA_BASE, mode="full"):
    """Build model.

    Modes
    -----
    "full"       : SD + Backflow + Jastrow  (CTNN + PINN)
    "sd_jastrow" : SD + Jastrow             (PINN only, no backflow)
    "sd_only"    : SD only                  (no learnable Jastrow or backflow)
    """
    if mode == "sd_only":
        f_net = _ZeroJastrow().to(DEVICE, DTYPE)
        backflow_net = None
    else:
        f_net = PINN(
            n_particles=N_PARTICLES, d=D, omega=omega,
            dL=5, hidden_dim=128, n_layers=2,
            act="gelu", init="xavier", use_gate=True,
        ).to(DEVICE, DTYPE)
        backflow_net = None

    if mode == "full":
        backflow_net = CTNNBackflowNet(
            d=D, msg_hidden=128, msg_layers=2, hidden=128, layers=3,
            act="gelu", aggregation="mean", use_spin=True,
            same_spin_only=False, out_bound="tanh",
            bf_scale_init=0.3, zero_init_last=True, omega=omega,
        ).to(DEVICE, DTYPE)

    return f_net, backflow_net


def _learnable_params(f_net, backflow_net):
    """Collect learnable parameters from both nets."""
    ps = [p for p in f_net.parameters() if p.requires_grad]
    if backflow_net is not None:
        ps += [p for p in backflow_net.parameters() if p.requires_grad]
    return ps


def make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, cfg):
    sep = cfg.sep_physical

    def psi_log_fn(x):
        xs = x.clone()
        xs[:, 0, 0] = x[:, 0, 0] + sep / 2
        xs[:, 1, 0] = x[:, 1, 0] - sep / 2
        logpsi, _ = psi_fn(
            f_net, xs, C_occ,
            backflow_net=backflow_net, spin=spin, params=params,
        )
        return logpsi

    return psi_log_fn


# ============================================================
# Energy Computation
# ============================================================


def compute_laplacian_logpsi(psi_log_fn, x):
    x = x.requires_grad_(True)
    lp = psi_log_fn(x)
    g = torch.autograd.grad(
        lp.sum(), x, create_graph=True, retain_graph=True,
    )[0]
    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            sec = torch.autograd.grad(
                g[:, i, j].sum(), x, create_graph=True, retain_graph=True,
            )[0]
            lap += sec[:, i, j]
    return lap, (g ** 2).sum(dim=(1, 2)), g


def local_energy(psi_log_fn, x, cfg):
    lap, g2, _ = compute_laplacian_logpsi(psi_log_fn, x)
    T = -0.5 * (lap + g2)
    V_trap = asymmetric_double_well_potential(x, cfg)
    V_coul = softened_coulomb(x, cfg.softening)
    return T + V_trap + V_coul, T, V_trap, V_coul


# ============================================================
# Training — Ground State
# ============================================================


def train_ground_state(
    f_net, backflow_net, C_occ, cfg, params, *,
    n_epochs=300, n_collocation=256, lr=5e-4, print_every=100,
):
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    all_p = _learnable_params(f_net, backflow_net)
    has_params = len(all_p) > 0

    if has_params:
        optimizer = optim.Adam(all_p, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr / 10,
        )

    psi_log_fn = make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, cfg)
    mcmc_sigma = 0.15 * cfg.ell_base
    best_energy, best_state = float("inf"), None
    x = sample_asymmetric_positions(n_collocation, cfg)

    n_run = n_epochs if has_params else min(n_epochs, 30)

    for epoch in range(n_run):
        f_net.train()
        if backflow_net is not None:
            backflow_net.train()

        with torch.no_grad():
            logp = 2.0 * psi_log_fn(x)
            for _ in range(20):
                xp = x + torch.randn_like(x) * mcmc_sigma
                lp = 2.0 * psi_log_fn(xp)
                acc = torch.rand(n_collocation, device=DEVICE).log() < (lp - logp)
                x = torch.where(acc.view(-1, 1, 1), xp, x)
                logp = torch.where(acc, lp, logp)

        if has_params:
            optimizer.zero_grad()
            xb = x.detach().requires_grad_(True)
            E_L, _, _, _ = local_energy(psi_log_fn, xb, cfg)
            E_mean = E_L.mean().detach()
            loss = ((E_L - E_mean) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_p, 1.0)
            optimizer.step()
            scheduler.step()
        else:
            with torch.set_grad_enabled(True):
                xb = x.detach().requires_grad_(True)
                E_L, _, _, _ = local_energy(psi_log_fn, xb, cfg)
            E_mean = E_L.mean().detach()

        ev = float(E_mean.item())
        if ev < best_energy:
            best_energy = ev
            best_state = {
                "f_net": copy.deepcopy(f_net.state_dict()),
                "bf_net": (
                    copy.deepcopy(backflow_net.state_dict())
                    if backflow_net
                    else None
                ),
            }

        if epoch % print_every == 0:
            print(
                f"    Ep {epoch:4d} | E = {ev:.5f}"
                f" ± {float(E_L.std()):.4f}"
            )

    if best_state:
        f_net.load_state_dict(best_state["f_net"])
        if backflow_net and best_state["bf_net"]:
            backflow_net.load_state_dict(best_state["bf_net"])

    return best_energy, best_state


# ============================================================
# Training — Excited State
# ============================================================


def compute_overlap_penalty_differentiable(psi_exc, psi_lower, x):
    log_exc = psi_exc(x)
    with torch.no_grad():
        log_low = psi_lower(x)
    log_ratio_sq = torch.clamp(2.0 * (log_low - log_exc), -40, 40)
    return torch.exp(log_ratio_sq).mean()


def compute_overlap(psi1, psi2, x):
    """Non-differentiable overlap ratio for monitoring / hop detection."""
    with torch.no_grad():
        lr = torch.clamp(psi2(x) - psi1(x), -20, 20)
        return torch.exp(lr).mean()


def train_excited_state(
    f_net, backflow_net, C_occ, cfg, params, lower_states, *,
    n_epochs=400, n_collocation=256, lr=3e-4,
    orthog_penalty=50.0, print_every=100,
):
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    all_p = _learnable_params(f_net, backflow_net)
    if not all_p:
        # SD-only — can't optimise excited states
        return float("inf"), None

    optimizer = optim.Adam(all_p, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 10,
    )
    psi_exc = make_psi_log_fn(f_net, C_occ, backflow_net, spin, params, cfg)
    mcmc_sigma = 0.15 * cfg.ell_base
    best_energy, best_state = float("inf"), None
    x = sample_asymmetric_positions(n_collocation, cfg)

    for epoch in range(n_epochs):
        f_net.train()
        if backflow_net is not None:
            backflow_net.train()

        with torch.no_grad():
            logp = 2.0 * psi_exc(x)
            for _ in range(20):
                xp = x + torch.randn_like(x) * mcmc_sigma
                lp = 2.0 * psi_exc(xp)
                acc = torch.rand(n_collocation, device=DEVICE).log() < (lp - logp)
                x = torch.where(acc.view(-1, 1, 1), xp, x)
                logp = torch.where(acc, lp, logp)

        optimizer.zero_grad()
        xb = x.detach().requires_grad_(True)
        E_L, _, _, _ = local_energy(psi_exc, xb, cfg)
        E_mean = E_L.mean()
        loss = ((E_L - E_mean.detach()) ** 2).mean()

        orth = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        for _, _, psi_low in lower_states:
            orth = orth + compute_overlap_penalty_differentiable(
                psi_exc, psi_low, x.detach(),
            )
        (loss + orthog_penalty * orth).backward()
        torch.nn.utils.clip_grad_norm_(all_p, 1.0)
        optimizer.step()
        scheduler.step()

        ev = float(E_mean.detach().item())
        if ev < best_energy:
            best_energy = ev
            best_state = {
                "f_net": copy.deepcopy(f_net.state_dict()),
                "bf_net": (
                    copy.deepcopy(backflow_net.state_dict())
                    if backflow_net
                    else None
                ),
            }

        if epoch % print_every == 0:
            ovs = ", ".join(
                f"{float(compute_overlap(psi_exc, p, x.detach())):.3f}"
                for _, _, p in lower_states
            )
            print(f"    Ep {epoch:4d} | E = {ev:.5f} | [{ovs}]")

    if best_state:
        f_net.load_state_dict(best_state["f_net"])
        if backflow_net and best_state["bf_net"]:
            backflow_net.load_state_dict(best_state["bf_net"])

    return best_energy, best_state


# ============================================================
# Diagnostics
# ============================================================


def evaluate_energy_precise(psi_log_fn, cfg, n_samples=50000):
    """Precise energy ± stderr via extended MCMC."""
    mcmc_sigma = 0.12 * cfg.ell_base
    bs = 1024
    sum_E = sum_E2 = 0.0
    total = 0
    x = sample_asymmetric_positions(bs, cfg)

    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(50):
            xp = x + torch.randn_like(x) * mcmc_sigma
            lp = 2.0 * psi_log_fn(xp)
            acc = torch.rand(bs, device=DEVICE).log() < (lp - logp)
            x = torch.where(acc.view(-1, 1, 1), xp, x)
            logp = torch.where(acc, lp, logp)

    while total < n_samples:
        with torch.no_grad():
            for _ in range(10):
                xp = x + torch.randn_like(x) * mcmc_sigma
                lp = 2.0 * psi_log_fn(xp)
                acc = torch.rand(bs, device=DEVICE).log() < (lp - logp)
                x = torch.where(acc.view(-1, 1, 1), xp, x)
                logp = torch.where(acc, lp, logp)
        with torch.set_grad_enabled(True):
            xe = x.detach().requires_grad_(True)
            E_L, _, _, _ = local_energy(psi_log_fn, xe, cfg)
        E = E_L.detach()
        sum_E += float(E.sum())
        sum_E2 += float((E ** 2).sum())
        total += bs

    mu = sum_E / total
    var = max(sum_E2 / total - mu ** 2, 0.0)
    return mu, math.sqrt(var) / math.sqrt(total)


def compute_mixing_weights_proper(psi_log_fn, cfg, n_samples=5000):
    """W_sub, a², b² via radial classification in {|10⟩, |01⟩}."""
    mcmc_sigma = 0.12 * cfg.ell_base
    bs = 1024
    sep = cfg.sep_physical
    sig_L = 1.0 / math.sqrt(cfg.omega_left)
    sig_R = 1.0 / math.sqrt(cfg.omega_right)
    x = sample_asymmetric_positions(bs, cfg)

    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(100):
            xp = x + torch.randn_like(x) * mcmc_sigma
            lp = 2.0 * psi_log_fn(xp)
            acc = torch.rand(bs, device=DEVICE).log() < (lp - logp)
            x = torch.where(acc.view(-1, 1, 1), xp, x)
            logp = torch.where(acc, lp, logp)

    thr_g, thr_e_lo, thr_e_hi = 1.2, 1.2, 3.0
    counts = {"00": 0, "10": 0, "01": 0, "11": 0, "other": 0}
    total = 0

    while total < n_samples:
        with torch.no_grad():
            for _ in range(5):
                xp = x + torch.randn_like(x) * mcmc_sigma
                lp = 2.0 * psi_log_fn(xp)
                acc = torch.rand(bs, device=DEVICE).log() < (lp - logp)
                x = torch.where(acc.view(-1, 1, 1), xp, x)
                logp = torch.where(acc, lp, logp)

            rL = (
                torch.sqrt((x[:, 0, 0] + sep / 2) ** 2 + x[:, 0, 1] ** 2)
                / sig_L
            )
            rR = (
                torch.sqrt((x[:, 1, 0] - sep / 2) ** 2 + x[:, 1, 1] ** 2)
                / sig_R
            )

            Lg = rL < thr_g
            Le = (rL >= thr_e_lo) & (rL < thr_e_hi)
            Rg = rR < thr_g
            Re = (rR >= thr_e_lo) & (rR < thr_e_hi)
            counts["00"] += int((Lg & Rg).sum())
            counts["10"] += int((Le & Rg).sum())
            counts["01"] += int((Lg & Re).sum())
            counts["11"] += int((Le & Re).sum())
            clean = (Lg | Le) & (Rg | Re)
            counts["other"] += int((~clean).sum())
            total += bs

    tc = sum(counts.values()) - counts["other"]
    if tc > 0:
        p10, p01 = counts["10"] / tc, counts["01"] / tc
    else:
        p10 = p01 = 0.25
    W_sub = p10 + p01
    a_sq = p10 / W_sub if W_sub > 1e-6 else 0.5
    b_sq = p01 / W_sub if W_sub > 1e-6 else 0.5
    return {
        "W_sub": W_sub,
        "a_sq": a_sq,
        "b_sq": b_sq,
        "p_00": counts["00"] / tc if tc > 0 else 0.25,
        "p_10": p10,
        "p_01": p01,
        "p_11": counts["11"] / tc if tc > 0 else 0.25,
    }


def compute_subspace_entanglement(a_sq: float, b_sq: float) -> float:
    """Entanglement entropy within the {|10⟩, |01⟩} projected subspace.

    S_sub = −a² ln a² − b² ln b²

    S = 0        → fully |10⟩ or |01⟩ (product state)
    S = ln 2     → equal superposition   (maximally entangled)

    This is the *only* entanglement metric we report — clean, cheap,
    and directly comparable to Jonny's 1D two-level picture.
    """
    S = 0.0
    if a_sq > 1e-10:
        S -= a_sq * math.log(a_sq)
    if b_sq > 1e-10:
        S -= b_sq * math.log(b_sq)
    return S


def compute_transverse_leakage(psi_log_fn, cfg, n_samples=5000):
    """L = ⟨y²⟩ / ⟨r²⟩.   L→0.5: isotropic 2D.  L→0: quasi-1D."""
    mcmc_sigma = 0.12 * cfg.ell_base
    bs = 1024
    sep = cfg.sep_physical
    x = sample_asymmetric_positions(bs, cfg)

    with torch.no_grad():
        logp = 2.0 * psi_log_fn(x)
        for _ in range(100):
            xp = x + torch.randn_like(x) * mcmc_sigma
            lp = 2.0 * psi_log_fn(xp)
            acc = torch.rand(bs, device=DEVICE).log() < (lp - logp)
            x = torch.where(acc.view(-1, 1, 1), xp, x)
            logp = torch.where(acc, lp, logp)

    sy2 = sr2 = 0.0
    total = 0
    while total < n_samples:
        with torch.no_grad():
            for _ in range(5):
                xp = x + torch.randn_like(x) * mcmc_sigma
                lp = 2.0 * psi_log_fn(xp)
                acc = torch.rand(bs, device=DEVICE).log() < (lp - logp)
                x = torch.where(acc.view(-1, 1, 1), xp, x)
                logp = torch.where(acc, lp, logp)
            y2 = x[:, 0, 1] ** 2 + x[:, 1, 1] ** 2
            xd0 = x[:, 0, 0] + sep / 2
            xd1 = x[:, 1, 0] - sep / 2
            r2 = xd0 ** 2 + x[:, 0, 1] ** 2 + xd1 ** 2 + x[:, 1, 1] ** 2
            sy2 += float(y2.sum())
            sr2 += float(r2.sum())
            total += bs
    return sy2 / sr2 if sr2 > 0 else 0.0


# Jonny-comparability criterion
W_SUB_THRESHOLD = 0.8


def compute_detuning(cfg: DoubleWellConfig) -> float:
    """Non-interacting detuning δ(λ) = E_10^(0) − E_01^(0).

    If δ(0) ≠ 0, the resonance is not at λ=0 and the gap minimum
    will be shifted.  This is essential for interpreting the sweep.

    For 2D isotropic:  E_10 = 2ω_L + ω_R,  E_01 = ω_L + 2ω_R
        => δ = ω_L − ω_R = 2 ω_base α λ

    For quasi-1D:  E_10 = 1.5ω_L + 0.5ω_R + ω_y,  E_01 = 0.5ω_L + 1.5ω_R + ω_y
        => δ = ω_L − ω_R  (same)
    """
    _, E10, E01 = cfg.expected_E_asymptotic()
    return E10 - E01


def compute_resonance_lambda(cfg_template: DoubleWellConfig) -> float:
    """Find λ* where δ(λ*)=0 analytically.

    δ = ω_L − ω_R = ω_base[(1+λα) − (1−λα)] = 2 ω_base α λ
    => λ*=0 always, by construction.  But report it explicitly.
    """
    return 0.0  # analytic; kept as function for clarity/logging


# ============================================================
# Train all three states at a single λ
# ============================================================


def train_all_states(
    cfg,
    C_occ,
    params,
    *,
    n_epochs_ground=200,
    n_epochs_excited=250,
    n_eval_samples=10000,
    n_diag_samples=5000,
    model_mode="full",
    warm_state=None,
    print_every=100,
):
    """Train E0, E1, E2 at one λ.

    Returns (result_dict, model_states, psi_pair).

    warm_state : dict with keys "ground", "lower", "upper" each containing
                 {"f_net": state_dict, "bf_net": state_dict|None}
    """
    spin = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)
    is_sd_only = model_mode == "sd_only"

    # --- Ground ---
    print("    [Ground]")
    f0, bf0 = build_model(OMEGA_BASE, mode=model_mode)
    if warm_state and warm_state.get("ground"):
        _load_warm(f0, bf0, warm_state["ground"])
    E0_best, st0 = train_ground_state(
        f0, bf0, C_occ, cfg, params,
        n_epochs=n_epochs_ground, print_every=print_every,
    )
    psi0 = make_psi_log_fn(f0, C_occ, bf0, spin, params, cfg)
    E0, E0_err = evaluate_energy_precise(psi0, cfg, n_eval_samples)
    print(f"    E0 = {E0:.5f} ± {E0_err:.5f}")

    if is_sd_only:
        result = _pack_result_sd_only(cfg, E0, E0_err)
        return result, {"ground": st0, "lower": None, "upper": None}, None

    # --- First excited ---
    print("    [Exc-1]")
    f1, bf1 = build_model(OMEGA_BASE, mode=model_mode)
    if warm_state and warm_state.get("lower"):
        _load_warm(f1, bf1, warm_state["lower"])
    _, st1 = train_excited_state(
        f1, bf1, C_occ, cfg, params,
        lower_states=[(f0, bf0, psi0)],
        n_epochs=n_epochs_excited, print_every=print_every,
    )
    psi1 = make_psi_log_fn(f1, C_occ, bf1, spin, params, cfg)
    E1, E1_err = evaluate_energy_precise(psi1, cfg, n_eval_samples)
    print(f"    E1 = {E1:.5f} ± {E1_err:.5f}")

    # --- Second excited ---
    print("    [Exc-2]")
    f2, bf2 = build_model(OMEGA_BASE, mode=model_mode)
    if warm_state and warm_state.get("upper"):
        _load_warm(f2, bf2, warm_state["upper"])
    _, st2 = train_excited_state(
        f2, bf2, C_occ, cfg, params,
        lower_states=[(f0, bf0, psi0), (f1, bf1, psi1)],
        n_epochs=n_epochs_excited, print_every=print_every,
    )
    psi2 = make_psi_log_fn(f2, C_occ, bf2, spin, params, cfg)
    E2, E2_err = evaluate_energy_precise(psi2, cfg, n_eval_samples)
    print(f"    E2 = {E2:.5f} ± {E2_err:.5f}")

    # --- Sort by energy ---
    if E1 <= E2:
        E_lo, E_lo_err, psi_lo, st_lo = E1, E1_err, psi1, st1
        E_hi, E_hi_err, psi_hi, st_hi = E2, E2_err, psi2, st2
    else:
        E_lo, E_lo_err, psi_lo, st_lo = E2, E2_err, psi2, st2
        E_hi, E_hi_err, psi_hi, st_hi = E1, E1_err, psi1, st1

    gap = E_hi - E_lo

    # --- Diagnostics ---
    print("    [Diag]")
    print(f"      Δ = {gap:.5f}  (E_lo={E_lo:.5f}, E_hi={E_hi:.5f})")

    mx_lo = compute_mixing_weights_proper(psi_lo, cfg, n_samples=n_diag_samples)
    mx_hi = compute_mixing_weights_proper(psi_hi, cfg, n_samples=n_diag_samples)
    S_lo = compute_subspace_entanglement(mx_lo["a_sq"], mx_lo["b_sq"])
    S_hi = compute_subspace_entanglement(mx_hi["a_sq"], mx_hi["b_sq"])

    print(
        f"      Lo: W={mx_lo['W_sub']:.3f} a²={mx_lo['a_sq']:.3f}"
        f" b²={mx_lo['b_sq']:.3f}  S={S_lo:.4f}"
    )
    print(
        f"      Hi: W={mx_hi['W_sub']:.3f} a²={mx_hi['a_sq']:.3f}"
        f" b²={mx_hi['b_sq']:.3f}  S={S_hi:.4f}"
    )

    lk0 = compute_transverse_leakage(psi0, cfg, n_samples=n_diag_samples)
    lk_lo = compute_transverse_leakage(psi_lo, cfg, n_samples=n_diag_samples)
    lk_hi = compute_transverse_leakage(psi_hi, cfg, n_samples=n_diag_samples)
    print(f"      Leak: E0={lk0:.3f}  lo={lk_lo:.3f}  hi={lk_hi:.3f}")

    # Non-interacting detuning
    detuning = compute_detuning(cfg)
    _, E10_ref, E01_ref = cfg.expected_E_asymptotic()
    W_clean = min(mx_lo["W_sub"], mx_hi["W_sub"]) >= W_SUB_THRESHOLD
    print(
        f"      δ(λ)={detuning:+.5f}  (E10⁰={E10_ref:.4f}  E01⁰={E01_ref:.4f})"
    )
    print(
        f"      Two-level clean: {W_clean}"
        f"  (min W_sub={min(mx_lo['W_sub'], mx_hi['W_sub']):.3f},"
        f" threshold={W_SUB_THRESHOLD})"
    )

    result = {
        "lambda": cfg.lam,
        "omega_left": cfg.omega_left,
        "omega_right": cfg.omega_right,
        "quasi_1d": cfg.quasi_1d,
        "detuning": detuning,
        "E10_ref": E10_ref,
        "E01_ref": E01_ref,
        "E0": E0,
        "E0_err": E0_err,
        "E_lower": E_lo,
        "E_lower_err": E_lo_err,
        "E_upper": E_hi,
        "E_upper_err": E_hi_err,
        "gap": gap,
        "lower_W_sub": mx_lo["W_sub"],
        "lower_a_sq": mx_lo["a_sq"],
        "lower_b_sq": mx_lo["b_sq"],
        "upper_W_sub": mx_hi["W_sub"],
        "upper_a_sq": mx_hi["a_sq"],
        "upper_b_sq": mx_hi["b_sq"],
        "S_sub_lower": S_lo,
        "S_sub_upper": S_hi,
        "E0_leakage": lk0,
        "lower_leakage": lk_lo,
        "upper_leakage": lk_hi,
        "two_level_clean": W_clean,
    }

    new_warm = {"ground": st0, "lower": st_lo, "upper": st_hi}
    return result, new_warm, (psi_lo, psi_hi)


def _load_warm(f_net, bf_net, state):
    """Load warm-start state dicts, silently skipping mismatches."""
    if state is None:
        return
    try:
        f_net.load_state_dict(state["f_net"])
    except Exception:
        pass
    if bf_net is not None and state.get("bf_net"):
        try:
            bf_net.load_state_dict(state["bf_net"])
        except Exception:
            pass


def _check_branch_overlap(psi_pair, prev_psi, cfg):
    """Compute branch-tracking overlap between consecutive λ steps."""
    x_t = sample_asymmetric_positions(1024, cfg)
    with torch.no_grad():
        logp = 2.0 * psi_pair[0](x_t)
        for _ in range(50):
            xp = x_t + torch.randn_like(x_t) * 0.12
            lp = 2.0 * psi_pair[0](xp)
            acc = torch.rand(1024, device=DEVICE).log() < (lp - logp)
            x_t = torch.where(acc.view(-1, 1, 1), xp, x_t)
            logp = torch.where(acc, lp, logp)
    ov_lo = float(compute_overlap(psi_pair[0], prev_psi[0], x_t))
    ov_hi = float(compute_overlap(psi_pair[1], prev_psi[1], x_t))
    return ov_lo, ov_hi


def _pack_result_sd_only(cfg, E0, E0_err):
    """Result dict for sd_only mode (no excited states)."""
    detuning = compute_detuning(cfg)
    _, E10_ref, E01_ref = cfg.expected_E_asymptotic()
    return {
        "lambda": cfg.lam,
        "omega_left": cfg.omega_left,
        "omega_right": cfg.omega_right,
        "quasi_1d": cfg.quasi_1d,
        "detuning": detuning,
        "E10_ref": E10_ref,
        "E01_ref": E01_ref,
        "E0": E0,
        "E0_err": E0_err,
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
    }


# ============================================================
# Dense Sweep — warm-start + hop detection + multi-seed
# ============================================================


def run_dense_sweep(
    lambda_values=None,
    *,
    n_seeds=3,
    n_epochs_ground=200,
    n_epochs_excited=250,
    n_epochs_ground_warm=100,
    n_epochs_excited_warm=150,
    n_eval_samples=10000,
    n_diag_samples=5000,
    quasi_1d=False,
    omega_y_factor=5.0,
):
    """Dense sweep near resonance with warm-starting and multi-seed error bars."""
    if lambda_values is None:
        lambda_values = [round(v, 2) for v in np.arange(-0.10, 0.101, 0.02)]

    tag = "DENSE SWEEP"
    if quasi_1d:
        tag += f"  (quasi-1D, ω_y/ω_x={omega_y_factor})"
    print("=" * 70)
    print(tag)
    print("=" * 70)
    print(f"  λ values ({len(lambda_values)}): {lambda_values}")
    print(f"  seeds = {n_seeds}")
    print(f"  epochs cold : gnd={n_epochs_ground}  exc={n_epochs_excited}")
    print(f"  epochs warm : gnd={n_epochs_ground_warm}  exc={n_epochs_excited_warm}")
    print()

    _init_config()
    params = config.get().as_dict()
    C_occ = make_cartesian_C_occ(2, 2, 1, device=DEVICE, dtype=DTYPE)

    all_seed_results = []

    for seed in range(n_seeds):
        print(f"\n{'#' * 70}")
        print(f"  SEED {seed}")
        print(f"{'#' * 70}")
        torch.manual_seed(seed * 42)

        warm_state = None
        prev_psi = None
        seed_results = []

        for i, lam in enumerate(lambda_values):
            print(
                f"\n  === λ={lam:+.3f}"
                f"  (seed {seed}, {i + 1}/{len(lambda_values)}) ==="
            )

            cfg = DoubleWellConfig(
                well_separation=4.0,
                omega_base=OMEGA_BASE,
                asymmetry_strength=0.3,
                lam=lam,
                quasi_1d=quasi_1d,
                omega_y_factor=omega_y_factor,
            )

            use_warm = warm_state is not None
            ne_g = n_epochs_ground_warm if use_warm else n_epochs_ground
            ne_e = n_epochs_excited_warm if use_warm else n_epochs_excited

            result, new_warm, psi_pair = train_all_states(
                cfg,
                C_occ,
                params,
                n_epochs_ground=ne_g,
                n_epochs_excited=ne_e,
                n_eval_samples=n_eval_samples,
                n_diag_samples=n_diag_samples,
                warm_state=warm_state if use_warm else None,
                print_every=max(ne_g, ne_e),
            )

            # --- Hop detection + retrain-on-hop ---
            hop = False
            retrained = False
            if prev_psi is not None and psi_pair is not None:
                ov_lo, ov_hi = _check_branch_overlap(
                    psi_pair, prev_psi, cfg,
                )
                hop = abs(ov_lo) < 0.3 or abs(ov_hi) < 0.3
                sym = "⚠ HOP" if hop else "✓"
                print(
                    f"      Branch overlap: lo={ov_lo:.3f}"
                    f" hi={ov_hi:.3f}  {sym}"
                )

                if hop:
                    print("      → Retraining from SCRATCH (no warm-start)")
                    result, new_warm, psi_pair = train_all_states(
                        cfg,
                        C_occ,
                        params,
                        n_epochs_ground=n_epochs_ground,
                        n_epochs_excited=n_epochs_excited,
                        n_eval_samples=n_eval_samples,
                        n_diag_samples=n_diag_samples,
                        warm_state=None,
                        print_every=max(n_epochs_ground, n_epochs_excited),
                    )
                    retrained = True
                    # Re-check overlap after retrain
                    if psi_pair is not None:
                        ov_lo, ov_hi = _check_branch_overlap(
                            psi_pair, prev_psi, cfg,
                        )
                        hop = abs(ov_lo) < 0.3 or abs(ov_hi) < 0.3
                        print(
                            f"      Post-retrain overlap:"
                            f" lo={ov_lo:.3f} hi={ov_hi:.3f}"
                        )

                result["overlap_lo_prev"] = ov_lo
                result["overlap_hi_prev"] = ov_hi
            else:
                result["overlap_lo_prev"] = None
                result["overlap_hi_prev"] = None

            result["hop_flag"] = hop
            result["retrained_from_scratch"] = retrained

            result["seed"] = seed
            seed_results.append(result)
            warm_state = new_warm
            if psi_pair is not None:
                prev_psi = psi_pair

        all_seed_results.append(seed_results)
        with open(RESULTS_DIR / f"dense_seed{seed}.json", "w") as f:
            json.dump(seed_results, f, indent=2, default=str)

    aggregated = _aggregate_seeds(all_seed_results, lambda_values)
    with open(RESULTS_DIR / "dense_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    return aggregated, all_seed_results


def _aggregate_seeds(all_seed_results, lambda_values):
    """mean ± std across seeds for each λ."""
    agg = []
    for i, lam in enumerate(lambda_values):
        rows = [sr[i] for sr in all_seed_results if i < len(sr)]
        n = len(rows)

        def _stat(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))

        gm, gs = _stat("gap")
        entry = {
            "lambda": lam,
            "n_seeds": n,
            "E0_mean": _stat("E0")[0],
            "E0_std": _stat("E0")[1],
            "E_lower_mean": _stat("E_lower")[0],
            "E_lower_std": _stat("E_lower")[1],
            "E_upper_mean": _stat("E_upper")[0],
            "E_upper_std": _stat("E_upper")[1],
            "gap_mean": gm,
            "gap_std": gs,
            "lower_W_sub_mean": _stat("lower_W_sub")[0],
            "lower_W_sub_std": _stat("lower_W_sub")[1],
            "upper_W_sub_mean": _stat("upper_W_sub")[0],
            "upper_W_sub_std": _stat("upper_W_sub")[1],
            "lower_a_sq_mean": _stat("lower_a_sq")[0],
            "lower_b_sq_mean": _stat("lower_b_sq")[0],
            "S_sub_lower_mean": _stat("S_sub_lower")[0],
            "S_sub_lower_std": _stat("S_sub_lower")[1],
            "S_sub_upper_mean": _stat("S_sub_upper")[0],
            "S_sub_upper_std": _stat("S_sub_upper")[1],
            "E0_leakage_mean": _stat("E0_leakage")[0],
            "lower_leakage_mean": _stat("lower_leakage")[0],
            "upper_leakage_mean": _stat("upper_leakage")[0],
            "hop_flags": [r.get("hop_flag", False) for r in rows],
            "detuning": _stat("detuning")[0],
            "two_level_clean": all(
                r.get("two_level_clean", False) for r in rows
            ),
        }
        agg.append(entry)
    return agg


# ============================================================
# Ablation Study
# ============================================================


def run_ablation(
    lambda_values=None,
    *,
    n_epochs_ground=200,
    n_epochs_excited=250,
    n_eval_samples=8000,
    n_diag_samples=4000,
):
    """SD / SD+Jastrow / SD+BF+Jastrow  at λ values near resonance."""
    if lambda_values is None:
        lambda_values = [-0.06, -0.02, 0.0, 0.02, 0.06]

    modes = ["sd_only", "sd_jastrow", "full"]
    labels = {"sd_only": "SD", "sd_jastrow": "SD+J", "full": "SD+BF+J"}

    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"  λ values: {lambda_values}")
    print(f"  Models: {[labels[m] for m in modes]}")
    print()

    _init_config()
    params = config.get().as_dict()
    C_occ = make_cartesian_C_occ(2, 2, 1, device=DEVICE, dtype=DTYPE)

    ablation = {}
    for mode in modes:
        print(f"\n{'#' * 70}")
        print(f"  Model: {labels[mode]}")
        print(f"{'#' * 70}")
        torch.manual_seed(0)
        mode_results = []

        for lam in lambda_values:
            cfg = DoubleWellConfig(
                well_separation=4.0,
                omega_base=OMEGA_BASE,
                asymmetry_strength=0.3,
                lam=lam,
            )
            print(f"\n  λ={lam:+.3f}  [{labels[mode]}]")

            result, _, _ = train_all_states(
                cfg,
                C_occ,
                params,
                n_epochs_ground=n_epochs_ground,
                n_epochs_excited=n_epochs_excited,
                n_eval_samples=n_eval_samples,
                n_diag_samples=n_diag_samples,
                model_mode=mode,
                print_every=max(n_epochs_ground, n_epochs_excited),
            )
            result["model"] = mode
            mode_results.append(result)

        ablation[mode] = mode_results

    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2, default=str)

    return ablation


# ============================================================
# Helper
# ============================================================


def _init_config():
    config.update(
        device=DEVICE,
        omega=OMEGA_BASE,
        n_particles=N_PARTICLES,
        d=D,
        basis="cart",
        nx=2,
        ny=2,
    )


# ============================================================
# Plotting
# ============================================================


def create_dense_plots(aggregated, save_dir=RESULTS_DIR, suffix=""):
    """6-panel plot with multi-seed error bars."""
    lams = [r["lambda"] for r in aggregated]
    E0 = [r["E0_mean"] for r in aggregated]
    Elo = [r["E_lower_mean"] for r in aggregated]
    Ehi = [r["E_upper_mean"] for r in aggregated]
    Elo_e = [r.get("E_lower_std") or 0 for r in aggregated]
    Ehi_e = [r.get("E_upper_std") or 0 for r in aggregated]
    gaps = [r["gap_mean"] for r in aggregated]
    gap_e = [r.get("gap_std") or 0 for r in aggregated]
    a_lo = [r.get("lower_a_sq_mean") or 0.5 for r in aggregated]
    b_lo = [r.get("lower_b_sq_mean") or 0.5 for r in aggregated]
    W_lo = [r.get("lower_W_sub_mean") or 0 for r in aggregated]
    W_hi = [r.get("upper_W_sub_mean") or 0 for r in aggregated]
    S_lo = [r.get("S_sub_lower_mean") or 0 for r in aggregated]
    S_hi = [r.get("S_sub_upper_mean") or 0 for r in aggregated]
    S_lo_e = [r.get("S_sub_lower_std") or 0 for r in aggregated]
    S_hi_e = [r.get("S_sub_upper_std") or 0 for r in aggregated]
    lk0 = [r.get("E0_leakage_mean") or 0 for r in aggregated]
    lk_lo = [r.get("lower_leakage_mean") or 0 for r in aggregated]
    lk_hi = [r.get("upper_leakage_mean") or 0 for r in aggregated]

    detunings = [r.get("detuning") or 0 for r in aggregated]
    clean = [r.get("two_level_clean", False) for r in aggregated]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # (a) Energy spectrum
    ax = axes[0, 0]
    ax.plot(lams, E0, "b-o", label=r"$E_0$", ms=4)
    ax.errorbar(
        lams, Elo, yerr=Elo_e, fmt="r-s",
        label=r"$E_{\rm lo}$", ms=4, capsize=3,
    )
    ax.errorbar(
        lams, Ehi, yerr=Ehi_e, fmt="g-^",
        label=r"$E_{\rm hi}$", ms=4, capsize=3,
    )
    ax.set(xlabel=r"$\lambda$", ylabel="Energy", title="(a) Energy spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Gap Δ with error bars
    ax = axes[0, 1]
    ax.errorbar(lams, gaps, yerr=gap_e, fmt="k-o", ms=5, capsize=3)
    valid = [(g, l) for g, l in zip(gaps, lams) if g is not None]
    if valid:
        mi = min(range(len(valid)), key=lambda j: valid[j][0])
        ax.axvline(
            valid[mi][1], color="r", ls="--", alpha=0.5,
            label=(
                rf"$\lambda^*={valid[mi][1]:.2f}$,"
                rf" $\Delta^*={valid[mi][0]:.4f}$"
            ),
        )
    ax.set(
        xlabel=r"$\lambda$",
        ylabel=r"$\Delta = E_{\rm hi} - E_{\rm lo}$",
        title=r"(b) Gap $\Delta(\lambda)$",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # (c) Mixing (a², b²) for E_lower
    ax = axes[0, 2]
    ax.plot(lams, a_lo, "b-o", label=r"$|a|^2$ (left)", ms=4)
    ax.plot(lams, b_lo, "r-s", label=r"$|b|^2$ (right)", ms=4)
    ax.plot(lams, W_lo, "k--", label=r"$W_{\rm sub}$", ms=3, alpha=0.7)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set(
        xlabel=r"$\lambda$", ylabel="Weight",
        title=r"(c) $E_{\rm lo}$: 2-level mixing",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # (d) W_sub for both states
    ax = axes[1, 0]
    ax.plot(lams, W_lo, "r-s", label=r"$W_{\rm sub}^{\rm lo}$", ms=4)
    ax.plot(lams, W_hi, "g-^", label=r"$W_{\rm sub}^{\rm hi}$", ms=4)
    ax.axhline(0.8, color="gray", ls=":", alpha=0.5, label="target 0.8")
    ax.set(
        xlabel=r"$\lambda$", ylabel=r"$W_{\rm sub}$",
        title=r"(d) Subspace weight",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # (e) Subspace entanglement entropy
    ax = axes[1, 1]
    ax.errorbar(
        lams, S_lo, yerr=S_lo_e, fmt="r-s",
        label=r"$S_{\rm sub}^{\rm lo}$", ms=4, capsize=3,
    )
    ax.errorbar(
        lams, S_hi, yerr=S_hi_e, fmt="g-^",
        label=r"$S_{\rm sub}^{\rm hi}$", ms=4, capsize=3,
    )
    ax.axhline(math.log(2), color="gray", ls=":", alpha=0.5, label=r"$\ln 2$")
    ax.set(
        xlabel=r"$\lambda$", ylabel=r"$S_{\rm sub}$",
        title="(e) Subspace entanglement",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Transverse leakage
    ax = axes[1, 2]
    ax.plot(lams, lk0, "b-o", label=r"$E_0$", ms=4)
    ax.plot(lams, lk_lo, "r-s", label=r"$E_{\rm lo}$", ms=4)
    ax.plot(lams, lk_hi, "g-^", label=r"$E_{\rm hi}$", ms=4)
    ax.axhline(0.5, color="gray", ls=":", label="2D isotropic")
    ax.set(
        xlabel=r"$\lambda$",
        ylabel=r"$L = \langle y^2\rangle/\langle r^2\rangle$",
        title="(f) Transverse leakage",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)

    # (g) Non-interacting detuning δ(λ)
    ax = axes[0, 3]
    ax.plot(lams, detunings, "k-o", ms=5)
    ax.axhline(0, color="r", ls="--", alpha=0.5, label=r"$\delta=0$ (resonance)")
    ax.set(
        xlabel=r"$\lambda$",
        ylabel=r"$\delta(\lambda) = E_{10}^{(0)} - E_{01}^{(0)}$",
        title=r"(g) Non-interacting detuning",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (h) Two-level criterion summary
    ax = axes[1, 3]
    W_min = [min(wl, wh) for wl, wh in zip(W_lo, W_hi)]
    colors_pts = ["green" if c else "red" for c in clean]
    ax.scatter(lams, W_min, c=colors_pts, s=60, zorder=5,
              label=r"$\min(W_{\rm sub}^{\rm lo}, W_{\rm sub}^{\rm hi})$")
    ax.plot(lams, W_min, "k-", alpha=0.3)
    ax.axhline(W_SUB_THRESHOLD, color="gray", ls="--",
               label=f"Jonny threshold ({W_SUB_THRESHOLD})")
    ax.set(
        xlabel=r"$\lambda$", ylabel=r"$\min\, W_{\rm sub}$",
        title="(h) Two-level criterion",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    n_clean = sum(clean)
    ax.text(
        0.05, 0.95,
        f"Clean: {n_clean}/{len(clean)} pts"
        + ("\n→ Jonny-comparable" if n_clean == len(clean)
           else "\n→ 2D leakage regime"),
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.8),
    )

    plt.tight_layout()
    name = f"dense_sweep{suffix}"
    plt.savefig(save_dir / f"{name}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {name}.pdf / .png")


def create_ablation_plots(ablation, save_dir=RESULTS_DIR):
    """4-panel ablation comparison."""
    labels = {"sd_only": "SD", "sd_jastrow": "SD+J", "full": "SD+BF+J"}
    colors = {"sd_only": "C0", "sd_jastrow": "C1", "full": "C2"}
    markers = {"sd_only": "o", "sd_jastrow": "s", "full": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for mode, results in ablation.items():
        lams = [r["lambda"] for r in results]
        gaps = [r["gap"] for r in results]
        W_lo = [r["lower_W_sub"] for r in results]
        S_lo = [r["S_sub_lower"] for r in results]
        E0_errs = [r["E0_err"] for r in results]
        c, m, lb = colors[mode], markers[mode], labels[mode]

        # SD-only: can only plot E0_err
        if any(g is None for g in gaps):
            axes[1, 1].plot(lams, E0_errs, f"-{m}", color=c, label=lb, ms=5)
            continue

        axes[0, 0].plot(lams, gaps, f"-{m}", color=c, label=lb, ms=5)
        axes[0, 1].plot(lams, W_lo, f"-{m}", color=c, label=lb, ms=5)
        axes[1, 0].plot(lams, S_lo, f"-{m}", color=c, label=lb, ms=5)
        axes[1, 1].plot(lams, E0_errs, f"-{m}", color=c, label=lb, ms=5)

    titles = [
        (r"(a) Resolved gap $\Delta^*$", r"$\Delta$"),
        (r"(b) Subspace weight $W_{\rm sub}$", r"$W_{\rm sub}$"),
        ("(c) Subspace entanglement", r"$S_{\rm sub}$"),
        (r"(d) Ground-state $\sigma_E$", r"$\sigma_E$"),
    ]
    for ax, (t, yl) in zip(axes.flat, titles):
        ax.set(xlabel=r"$\lambda$", ylabel=yl, title=t)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].axhline(0.8, color="gray", ls=":", alpha=0.5)
    axes[1, 0].axhline(math.log(2), color="gray", ls=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / "ablation_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(save_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved ablation_comparison.pdf / .png")


# ============================================================
# Legacy convenience modes
# ============================================================


def tiny_test():
    """Smoke test: 3 λ, 1 seed, minimal epochs.  ~5 min."""
    agg, _ = run_dense_sweep(
        lambda_values=[-0.5, 0.0, 0.5],
        n_seeds=1,
        n_epochs_ground=50,
        n_epochs_excited=80,
        n_epochs_ground_warm=50,
        n_epochs_excited_warm=80,
        n_eval_samples=3000,
        n_diag_samples=2000,
    )
    create_dense_plots(agg, suffix="_tiny")
    print("\nTINY TEST COMPLETE")


def quick_test():
    """Quick: 7 λ, 1 seed, moderate epochs.  ~40 min."""
    agg, _ = run_dense_sweep(
        lambda_values=[-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6],
        n_seeds=1,
        n_epochs_ground=150,
        n_epochs_excited=200,
        n_epochs_ground_warm=100,
        n_epochs_excited_warm=150,
        n_eval_samples=8000,
        n_diag_samples=4000,
    )
    create_dense_plots(agg, suffix="_quick")
    print("\nQUICK TEST COMPLETE")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Avoided Crossing Study v2")
    p.add_argument(
        "--dense", action="store_true",
        help="Dense 11-pt sweep + multi-seed (~2h)",
    )
    p.add_argument(
        "--ablation", action="store_true",
        help="Ablation: SD/SD+J/SD+BF+J (~1h)",
    )
    p.add_argument(
        "--quasi1d", action="store_true",
        help="Quasi-1D confinement (ω_y/ω_x=5)",
    )
    p.add_argument(
        "--seeds", type=int, default=3,
        help="Seeds for error bars (default 3)",
    )
    p.add_argument("--quick", action="store_true", help="Quick 7-pt (~40m)")
    p.add_argument("--tiny", action="store_true", help="Smoke test (~5m)")
    args = p.parse_args()

    if args.tiny:
        tiny_test()
    elif args.quick:
        quick_test()
    elif args.dense:
        agg, raw = run_dense_sweep(
            n_seeds=args.seeds,
            quasi_1d=args.quasi1d,
        )
        sfx = "_quasi1d" if args.quasi1d else ""
        create_dense_plots(agg, suffix=sfx)
    elif args.ablation:
        ab = run_ablation()
        create_ablation_plots(ab)
    else:
        # Default: dense sweep + ablation
        print("Running dense sweep then ablation...\n")
        agg, _ = run_dense_sweep(n_seeds=args.seeds, quasi_1d=args.quasi1d)
        sfx = "_quasi1d" if args.quasi1d else ""
        create_dense_plots(agg, suffix=sfx)
        ab = run_ablation()
        create_ablation_plots(ab)
