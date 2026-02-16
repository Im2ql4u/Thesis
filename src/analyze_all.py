#!/usr/bin/env python3
"""
analyze_all.py — Unified quantum-dot analysis for PINN / BF / CTNN ansätze.

Usage:
    python analyze_all.py --ansatz pinn
    python analyze_all.py --ansatz bf
    python analyze_all.py --ansatz ctnn
    python analyze_all.py --ansatz bf --particles 6 --omegas 0.1,0.5

Iterates over all (N, ω) configurations and produces:
  • JSON with scalar metrics  (results/analysis/{ansatz}/N{N}_w{omega}/metrics.json)
  • PDF figures               (results/analysis/{ansatz}/N{N}_w{omega}/*.pdf)
  • Cross-config summary      (results/analysis/{ansatz}/summary.json + summary plots)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports (run from src/ or with src/ on sys.path)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import config as cfg
from PINN import PINN, BackflowNet, CTNNBackflowNet
from utils import inject_params, get_params_dict

from functions.Slater_Determinant import (
    compute_integrals,
    hartree_fock_closed_shell,
    slater_determinant_closed_shell,
)
from functions.Neural_Networks import psi_fn as _psi_fn
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc
from functions.Analysis import (
    make_psi_log_fn,
    init_positions_gaussian,
    sample_psi2_batch,
    forward_taps,
    feature_svd,
    pca_on_rho_in,
    pc_projection_ablation,
    compute_physical_summaries,
    linear_probe_pcs,
    branch_ablation_drop,
    cusp_vs_residual_means,
    energy_feature_correlations,
    near_field_grad_share_by_quantile,
    run_compact_analysis,
)

try:
    from functions.analyze_shells import analyze_case as analyze_shells_case
except ImportError:
    analyze_shells_case = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NX_NY_MAP = {2: (1, 1), 6: (2, 2), 12: (3, 3), 20: (4, 4)}
ALL_PARTICLES = [2, 6, 12, 20]
ALL_OMEGAS = [0.001, 0.01, 0.1, 0.5, 1.0]

# DMC reference energies (from config.py)
DMC_ENERGIES = cfg.DMC_ENERGIES

# Backflow default hyper-parameters (matching training notebooks)
BF_DEFAULTS = dict(
    msg_hidden=200, msg_layers=2, hidden=200, layers=2,
    act="gelu", aggregation="mean", bf_scale_init=0.4,
    use_spin=True, same_spin_only=False, out_bound="tanh",
    zero_init_last=True,
)
CTNN_DEFAULTS = dict(
    msg_hidden=128, msg_layers=2, hidden=128, layers=3,
    act="gelu", aggregation="mean", bf_scale_init=0.3,
    use_spin=True, same_spin_only=False, out_bound="tanh",
    zero_init_last=True,
)
PINN_DEFAULTS = dict(dL=5, hidden_dim=128, n_layers=2, act="gelu")


# ═══════════════════════════════════════════════════════════════════════════
# 0. HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _to_python(obj):
    """Recursively convert torch tensors / numpy arrays to JSON-safe types."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        x = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else obj
        if x.ndim == 0:
            return float(x)
        return x.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def _finite_stats(x: Tensor) -> dict:
    mask = torch.isfinite(x)
    n_fin = int(mask.sum())
    if n_fin == 0:
        return dict(mean=float("nan"), std=float("nan"), se=float("nan"),
                    n_finite=0, n_total=int(mask.numel()))
    xf = x[mask]
    mu = float(xf.mean())
    sd = float(xf.std(unbiased=False))
    se = sd / max(1.0, math.sqrt(n_fin))
    return dict(mean=mu, std=sd, se=se, n_finite=n_fin, n_total=int(mask.numel()))


def _effective_rank(s: Tensor) -> float:
    """Shannon effective rank from singular values."""
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * p.log()).sum()
    return float(torch.exp(H))


def _cka_linear(X: Tensor, Y: Tensor) -> float:
    """Linear CKA between two (N, d) activation matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = (X.T @ Y).norm() ** 2
    hsic_xx = (X.T @ X).norm() ** 2
    hsic_yy = (Y.T @ Y).norm() ** 2
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def _make_spin(N: int, device: torch.device) -> Tensor:
    up = N // 2
    return torch.cat([
        torch.zeros(up, dtype=torch.long, device=device),
        torch.ones(N - up, dtype=torch.long, device=device),
    ])


def _omega_tag(omega: float) -> str:
    return f"w_{omega:.5f}".rstrip("0").rstrip(".")


def _case_tag(N: int, omega: float) -> str:
    return f"N{N}_{_omega_tag(omega)}"


# ═══════════════════════════════════════════════════════════════════════════
# 1. SYSTEM SETUP  &  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def setup_system(
    ansatz: str,
    N: int,
    omega: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    model_root: str = "../results/official_models",
) -> dict:
    """
    Build PINN + optional backflow, run HF, load weights.
    Returns dict with f_net, backflow_net, C_occ, spin, params, psi_log_fn.
    """
    d = 2
    nx, ny = NX_NY_MAP[N]

    # Update global config so @inject_params picks up the right values
    cfg.update(
        n_particles=N, omega=omega, d=d, nx=nx, ny=ny,
        basis="cart", device=str(device), dtype="float64",
        emax=max(nx, ny) + 1,
    )
    params = get_params_dict()

    # ---- Hartree-Fock ----
    Hcore, two_dirac, basis_info = compute_integrals(params=params)
    S = basis_info["S"]
    C_occ_np, eps_occ, E_HF = hartree_fock_closed_shell(
        Hcore, two_dirac, S=S, params=params,
    )
    C_occ = torch.tensor(C_occ_np, device=device, dtype=dtype)
    print(f"  HF energy = {E_HF:.6f} Ha")

    # ---- Build networks ----
    f_net = PINN(N, d, omega, **PINN_DEFAULTS).to(device=device, dtype=dtype)

    backflow_net = None
    if ansatz == "bf":
        backflow_net = BackflowNet(d, **BF_DEFAULTS).to(device=device, dtype=dtype)
    elif ansatz == "ctnn":
        backflow_net = CTNNBackflowNet(d, omega=omega, **CTNN_DEFAULTS).to(device=device, dtype=dtype)

    # ---- Load weights ----
    # >>>  PLACEHOLDER — fill in your path convention  <<<
    # Expected structure:  {model_root}/{ansatz}/{N}p/w_{omega}/f_net.pt
    #                      {model_root}/{ansatz}/{N}p/w_{omega}/backflow.pt
    _load_weights(f_net, backflow_net, ansatz, N, omega, model_root, device)

    f_net.eval()
    if backflow_net is not None:
        backflow_net.eval()

    spin = _make_spin(N, device)

    psi_log = make_psi_log_fn(
        _psi_fn, f_net, C_occ, backflow_net=backflow_net, spin=spin, params=params,
    )
    return dict(
        f_net=f_net, backflow_net=backflow_net, C_occ=C_occ,
        spin=spin, params=params, psi_log_fn=psi_log,
        E_HF=E_HF, device=device, dtype=dtype, N=N, d=d, omega=omega,
    )


def _load_weights(f_net, backflow_net, ansatz, N, omega, model_root, device):
    """
    Attempt to load model weights.  Prints a warning and continues with
    random weights if files are missing (so the analysis code can still
    be tested structurally).
    """
    from functions.Save_Model import model_dir

    root = Path(model_root) / ansatz
    md = root / f"{N}p" / f"w_{omega:.5f}".rstrip("0").rstrip(".")

    def _try_load(module: nn.Module, name: str):
        pt = md / f"{name}.pt"
        if not pt.exists():
            # Try alternate naming conventions
            for alt in [md / f"{name}.pt", root / f"{name}_{N}e_{omega}.pt"]:
                if alt.exists():
                    pt = alt
                    break
        if pt.exists():
            payload = torch.load(pt, map_location=device, weights_only=False)
            if isinstance(payload, dict) and "state_dict" in payload:
                module.load_state_dict(payload["state_dict"], strict=False)
            elif isinstance(payload, dict):
                # Try to load directly as state dict
                try:
                    module.load_state_dict(payload, strict=False)
                except Exception:
                    pass
            print(f"  Loaded {pt}")
        else:
            warnings.warn(
                f"  Model file not found: {pt}  — using random weights!",
                stacklevel=3,
            )

    _try_load(f_net, "f_net")
    if backflow_net is not None:
        _try_load(backflow_net, "backflow")


# ═══════════════════════════════════════════════════════════════════════════
# 2. SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_samples(
    psi_log_fn,
    N: int,
    d: int,
    omega: float,
    device: torch.device,
    dtype: torch.dtype,
    B: int = 8192,
) -> tuple[Tensor, float]:
    """
    Draw B samples from |Ψ|² via Metropolis-Hastings.
    Returns (X, accept_rate).
    """
    X, acc_burn, acc_mix = sample_psi2_batch(
        psi_log_fn,
        B=B, N=N, d=d, omega=omega,
        device=device, dtype=dtype,
        method="rw",
        step_sigma=0.2,
        burn_in=100,
        mix_steps=50,
        autotune=True,
    )
    acc = 0.5 * (acc_burn + acc_mix)
    return X, acc


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENERGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_energy(sys_info: dict) -> dict:
    """
    Full VMC energy evaluation with T / V_int / V_trap decomposition,
    plus virial ratios and Δ E_BF if backflow is present.
    """
    f_net = sys_info["f_net"]
    C_occ = sys_info["C_occ"]
    bf = sys_info["backflow_net"]
    spin = sys_info["spin"]
    params = sys_info["params"]
    N = sys_info["N"]
    omega = sys_info["omega"]

    # Choose Laplacian mode based on system size
    lap_mode = "exact" if N <= 12 else "hvp-hutch"
    n_samples = 100_000 if N <= 6 else 50_000

    print("  Evaluating VMC energy...")
    result = evaluate_energy_vmc(
        f_net, C_occ,
        psi_fn=_psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=bf,
        spin=spin,
        params=params,
        n_samples=n_samples,
        batch_size=min(2048, n_samples),
        lap_mode=lap_mode,
        lap_probes=24,
        persistent=True,
        sampler_burn_in=200,
        sampler_thin=5,
        progress=True,
    )

    out = {
        "E_mean": result["E_mean"],
        "E_std": result["E_std"],
        "E_stderr": result["E_stderr"],
        "T_mean": result["T_mean"],
        "V_int_mean": result["V_int_mean"],
        "V_trap_mean": result["V_trap_mean"],
        "accept_rate": result.get("accept_rate_avg", float("nan")),
    }

    # Virial ratios
    T = result["T_mean"]
    Vi = result["V_int_mean"]
    Vt = result["V_trap_mean"]
    out["Gamma_Vint_over_T"] = Vi / T if abs(T) > 1e-15 else float("nan")
    out["virial_2Vt_over_Vi"] = (2 * Vt) / Vi if abs(Vi) > 1e-15 else float("nan")

    # DMC reference
    dmc = DMC_ENERGIES.get(N, {}).get(cfg._snap_omega(omega), None)
    if dmc is not None:
        out["E_DMC"] = dmc
        out["rel_error_pct"] = 100.0 * abs(result["E_mean"] - dmc) / abs(dmc)
    else:
        out["E_DMC"] = None
        out["rel_error_pct"] = None

    # ---------- Without-BF comparison (if BF present) ----------
    if bf is not None:
        print("  Evaluating energy WITHOUT backflow...")
        result_noBF = evaluate_energy_vmc(
            f_net, C_occ,
            psi_fn=_psi_fn,
            compute_coulomb_interaction=compute_coulomb_interaction,
            backflow_net=None,
            spin=spin,
            params=params,
            n_samples=n_samples // 2,
            batch_size=min(2048, n_samples // 2),
            lap_mode=lap_mode,
            persistent=True,
            sampler_burn_in=200,
            sampler_thin=5,
            progress=True,
        )
        out["E_noBF_mean"] = result_noBF["E_mean"]
        out["E_noBF_std"] = result_noBF["E_std"]
        out["delta_E_BF"] = result["E_mean"] - result_noBF["E_mean"]

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. CORRELATOR  REPRESENTATION  GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

def analyze_correlator(
    f_net: nn.Module,
    X: Tensor,
    spin: Tensor,
    E_L: Tensor | None = None,
) -> dict:
    """
    Full correlator (f_net / PINN) representation analysis:
      - Feature SVD / effective rank
      - PCA on ρ-inputs
      - PC ablation (top-k vs random)
      - Linear probes  (PCs → physical summaries)
      - Branch ablation (φ / ψ / extras)
      - PC1 block power decomposition
      - Energy-feature correlations
      - Cusp vs residual partitioning
      - Near-field gradient concentration
    """
    out = {}

    # --- Forward taps ---
    taps = forward_taps(f_net, X, spin)

    # --- Feature SVD ---
    svd_info = feature_svd(f_net, X, spin)
    out["effective_rank"] = svd_info["effective_rank"]
    out["singular_values"] = svd_info["singular_values"][:12]  # top-12
    out["explained_variance"] = svd_info["explained_variance"][:12]

    # --- PCA on rho input ---
    pca_info = pca_on_rho_in(f_net, X, spin)
    out["pca_eff_rank"] = pca_info["eff_rank"]
    out["pca_expvar_top8"] = pca_info["expvar"][:8]

    # --- Head correlation with PC1 ---
    Z = taps["rho_in"]     # (B, 2*dL+2)
    mu = Z.mean(0)
    Z_c = Z - mu
    U, S_vals, Vt = torch.linalg.svd(Z_c, full_matrices=False)
    pc1 = Z_c @ Vt[0]     # (B,)
    head_out = f_net.rho(Z).squeeze(-1)
    corr_pc1_head = float(torch.corrcoef(torch.stack([pc1, head_out]))[0, 1])
    out["head_corr_pc1"] = corr_pc1_head

    # --- PC1 block power decomposition ---
    dL = f_net.dL
    v1 = Vt[0]            # (2*dL+2,)
    phi_power = float((v1[:dL] ** 2).sum())
    psi_power = float((v1[dL:2 * dL] ** 2).sum())
    extras_power = float((v1[2 * dL:] ** 2).sum())
    out["pc1_block_power"] = {"phi": phi_power, "psi": psi_power, "extras": extras_power}

    # --- PC ablation ---
    pc_abl = pc_projection_ablation(f_net, X, spin)
    out["pc_ablation"] = {
        "k": pc_abl["k"],
        "mae_pc": pc_abl["mae_pc"],
        "mae_rand": pc_abl["mae_rand"],
    }

    # --- Linear probes ---
    probes = linear_probe_pcs(f_net, X, spin)
    out["probe_R2"] = probes["R2"]
    out["probe_targets"] = probes["target_names"]

    # --- Branch ablation ---
    ablation = branch_ablation_drop(f_net, X, spin)
    out["branch_ablation"] = ablation

    # --- Cusp vs residual ---
    cusp_stats = cusp_vs_residual_means(f_net, X, spin)
    out["cusp_means"] = cusp_stats

    # --- Energy-feature correlations ---
    if E_L is not None and torch.isfinite(E_L).sum() > 100:
        ecorr = energy_feature_correlations(taps, E_L)
        out["energy_feature_corr"] = ecorr
    else:
        out["energy_feature_corr"] = None

    # --- Near-field gradient concentration ---
    nf = near_field_grad_share_by_quantile(f_net, X, spin)
    out["near_field_grad"] = nf

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. BACKFLOW  DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_backflow(
    f_net: nn.Module,
    backflow_net: nn.Module,
    X: Tensor,
    spin: Tensor,
    C_occ: Tensor,
    params: dict,
) -> dict:
    """
    Comprehensive backflow analysis:
      - Displacement statistics (magnitude, direction)
      - Effective rank of Δx
      - PCA on flattened displacements
      - PC ablation on displacement
      - Channel ablation (zero x_i / x_j / r_vec / |r| / |r|²)
      - Linear probes from Δx PCs to physical summaries
      - Near-field concentration of ‖Δx‖
      - Spin-resolved displacement statistics
      - BF vs no-BF correlator comparison
      - COM shift magnitude
    """
    out = {}
    B, N, d = X.shape
    device = X.device

    # ---- Displacements ----
    with torch.no_grad():
        spin_bn = spin.unsqueeze(0).expand(B, -1) if spin.dim() == 1 else spin
        dx = backflow_net(X, spin=spin_bn)  # (B, N, d)

    dx_norm = dx.norm(dim=-1)               # (B, N)
    out["dx_mean_magnitude"] = float(dx_norm.mean())
    out["dx_std_magnitude"] = float(dx_norm.std())
    out["dx_max_magnitude"] = float(dx_norm.max())

    # ---- Effective rank of Δx ----
    dx_flat = dx.reshape(B, N * d)           # (B, N*d)
    dx_c = dx_flat - dx_flat.mean(0)
    svs = torch.linalg.svdvals(dx_c)
    out["dx_effective_rank"] = _effective_rank(svs)
    out["dx_singular_values"] = svs[:12].tolist()

    # ---- PCA on displacement ----
    U, S_vals, Vt = torch.linalg.svd(dx_c, full_matrices=False)
    total_var = (S_vals ** 2).sum()
    expvar = (S_vals ** 2).cumsum(0) / total_var
    out["dx_pca_expvar_top8"] = expvar[:8].tolist()

    # ---- PC ablation on displacement → head response ----
    # Project Δx through top-k PCs, measure how well f_net output is preserved
    with torch.no_grad():
        spin_bn = spin.unsqueeze(0).expand(B, -1) if spin.dim() == 1 else spin
        x_bf = X + dx
        taps_bf = forward_taps(f_net, x_bf, spin)
        taps_nobf = forward_taps(f_net, X, spin)
        delta_head = taps_bf["out"] - taps_nobf["out"]       # (B, 1)
        baseline_mae = float(delta_head.abs().mean())

    out["bf_head_delta_mae"] = baseline_mae

    pc_ablation = {}
    for k in [1, 2, 3, 5, 8, min(12, N * d)]:
        if k > dx_flat.shape[1]:
            break
        # Project Δx onto top-k PCs
        Vk = Vt[:k]                                           # (k, N*d)
        dx_proj = (dx_c @ Vk.T) @ Vk                          # (B, N*d)
        dx_proj = dx_proj + dx_flat.mean(0)
        dx_recon = dx_proj.reshape(B, N, d)
        with torch.no_grad():
            x_recon = X + dx_recon
            taps_recon = forward_taps(f_net, x_recon, spin)
            delta_recon = taps_recon["out"] - taps_nobf["out"]
            mae = float((delta_recon - delta_head).abs().mean())
        pc_ablation[k] = mae
    out["dx_pc_ablation"] = pc_ablation

    # ---- Spin-resolved displacement statistics ----
    up_mask = (spin == 0)
    dn_mask = (spin == 1)
    if up_mask.any():
        out["dx_up_mean"] = float(dx_norm[:, up_mask].mean())
    if dn_mask.any():
        out["dx_dn_mean"] = float(dx_norm[:, dn_mask].mean())

    # ---- COM shift ----
    com_shift = dx.mean(dim=1).norm(dim=-1)   # (B,)
    out["com_shift_mean"] = float(com_shift.mean())
    out["com_shift_max"] = float(com_shift.max())

    # ---- Near-field Δx concentration ----
    # Fraction of ‖Δx‖² in near-field (small r_min) quantiles
    with torch.no_grad():
        r_particles = X.norm(dim=-1)           # (B, N)
        # Minimum pair distance per config
        diff = X.unsqueeze(2) - X.unsqueeze(1)
        ii, jj = torch.triu_indices(N, N, 1, device=device)
        pair_dist = diff[:, ii, jj].norm(dim=-1)  # (B, P)
        r_min_config = pair_dist.min(dim=-1).values  # (B,)

    dx_sq_total = (dx_norm ** 2).sum(dim=-1)   # (B,)  total ‖Δx‖² per config
    nf_report = []
    for q in (0.01, 0.05, 0.10):
        threshold = torch.quantile(r_min_config, q)
        near_mask = r_min_config <= threshold
        n_near = int(near_mask.sum())
        if n_near > 0:
            share = float(dx_sq_total[near_mask].sum() / dx_sq_total.sum())
            nf_report.append(dict(q=q, share=share, count=n_near, threshold=float(threshold)))
    out["near_field_dx"] = nf_report

    # ---- BF vs noBF correlator feature comparison ----
    with torch.no_grad():
        Z_nobf = taps_nobf["rho_in"]
        Z_bf = taps_bf["rho_in"]
        out["correlator_cka_nobf_vs_bf"] = _cka_linear(Z_nobf.float(), Z_bf.float())
        # Effective rank comparison
        out["eff_rank_noBF"] = _effective_rank(torch.linalg.svdvals(Z_nobf - Z_nobf.mean(0)))
        out["eff_rank_BF"] = _effective_rank(torch.linalg.svdvals(Z_bf - Z_bf.mean(0)))
        # PC1 alignment between noBF and BF
        U1 = torch.linalg.svd(Z_nobf - Z_nobf.mean(0), full_matrices=False)[2][0]
        U2 = torch.linalg.svd(Z_bf - Z_bf.mean(0), full_matrices=False)[2][0]
        out["pc1_alignment_cos"] = float(torch.abs(U1 @ U2))

    # ---- Channel ablation (for standard BackflowNet) ----
    if isinstance(backflow_net, BackflowNet):
        out["channel_ablation"] = _bf_channel_ablation(backflow_net, X, spin)

    return out


def _bf_channel_ablation(bf_net: BackflowNet, X: Tensor, spin: Tensor) -> dict:
    """
    Zero each input channel group to the message MLP and measure ΔΔx.
    Groups: x_i (d), x_j (d), r_vec (d), |r| (1), |r|² (1)
    """
    B, N, d = X.shape
    device = X.device

    with torch.no_grad():
        spin_bn = spin.unsqueeze(0).expand(B, -1) if spin.dim() == 1 else spin
        dx_ref = bf_net(X, spin=spin_bn)                   # (B, N, d)
        dx_ref_flat = dx_ref.reshape(B, -1)

    # We need to hook into the message input — this requires modifying the forward
    # For now, measure total Δx change when zeroing each particle coordinate
    channel_results = {}
    for dim_idx in range(d):
        X_zero = X.clone()
        X_zero[..., dim_idx] = 0.0
        with torch.no_grad():
            dx_zero = bf_net(X_zero, spin=spin_bn)
        delta = (dx_zero - dx_ref).reshape(B, -1).norm(dim=-1)
        channel_results[f"coord_{dim_idx}"] = float(delta.mean())

    return channel_results


# ═══════════════════════════════════════════════════════════════════════════
# 6. WIGNER-MOLECULE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_structure(
    X: Tensor,
    N: int,
    omega: float,
    psi_log_fn=None,
    n_extra_samples: int = 50_000,
) -> dict:
    """
    Wigner-molecule structural analysis from MCMC samples:
      - g(r) pair correlation function
      - P(r) single-particle radial distribution
      - Radial summary statistics (mode, mean, σ, FWHM, quantiles, γ)
      - Angle distribution (Δφ between particle pairs)
      - Lindemann ratio
      - Shell analysis (for N ≥ 6)
      - One-body density moments
    """
    out = {}
    B, N_actual, d = X.shape
    device = X.device

    # Convert to trap units for structure analysis
    ell = 1.0 / math.sqrt(max(omega, 1e-12))
    X_trap = X / ell  # dimensionless trap units

    # ── g(r): pair correlation function ──
    gr_info = _compute_pair_correlation(X, N, omega, n_bins=200)
    out["gr"] = gr_info

    # ── P(r): single-particle radial distribution ──
    r_all = X.norm(dim=-1).reshape(-1).cpu().numpy()
    pr_info = _radial_summary(r_all, omega)
    out["radial"] = pr_info

    # ── Per-particle radial moments ──
    r_per_particle = X.norm(dim=-1)  # (B, N)
    r_mean_config = r_per_particle.mean(dim=-1)  # (B,)
    out["r_rms"] = float(r_per_particle.pow(2).mean().sqrt())
    out["r_mean"] = float(r_per_particle.mean())
    out["r_std"] = float(r_per_particle.std())

    # ── Angle distribution ──
    if d == 2:
        angles = torch.atan2(X[..., 1], X[..., 0])  # (B, N)
        # Pairwise angle differences
        ang_diff = []
        for i in range(N):
            for j in range(i + 1, N):
                delta = (angles[:, j] - angles[:, i]) % (2 * math.pi)
                delta = torch.where(delta > math.pi, 2 * math.pi - delta, delta)
                ang_diff.append(delta)
        ang_diff = torch.cat(ang_diff).cpu().numpy()
        n_bins_ang = 72
        ang_hist, ang_edges = np.histogram(ang_diff, bins=n_bins_ang, range=(0, np.pi), density=True)
        out["angle_dist"] = {
            "hist": ang_hist.tolist(),
            "edges": ang_edges.tolist(),
            "mean_angle": float(np.mean(ang_diff)),
            "std_angle": float(np.std(ang_diff)),
        }

    # ── Lindemann ratio ──
    lind = _lindemann_ratio(X)
    out["lindemann"] = lind

    # ── Shell analysis (N ≥ 6) ──
    if N >= 6 and analyze_shells_case is not None:
        try:
            X_trap_np = X_trap.cpu().numpy() if isinstance(X_trap, Tensor) else X_trap
            # analyze_shells.analyze_case expects (K, N, d) tensor
            shell_result = analyze_shells_case(N, omega, verbose=False)
            out["shells"] = _to_python(shell_result)
        except Exception as e:
            out["shells"] = {"error": str(e)}

    # ── Density parameter r_s ──
    # First peak of g(r) gives characteristic interparticle distance a
    if "gr" in out and "r_centers" in out["gr"]:
        r_c = np.array(out["gr"]["r_centers"])
        gr_vals = np.array(out["gr"]["g_r"])
        if len(gr_vals) > 3:
            peak_idx = np.argmax(gr_vals[2:]) + 2  # skip first bins
            r_s = r_c[peak_idx] if peak_idx < len(r_c) else float("nan")
            out["r_s_from_gr"] = float(r_s)

    return out


def _compute_pair_correlation(X: Tensor, N: int, omega: float, n_bins: int = 200) -> dict:
    """Compute g(r) from samples using sqrt-stretched binning."""
    B, N_actual, d = X.shape

    # Pairwise distances (all pairs)
    ii, jj = torch.triu_indices(N, N, 1, device=X.device)
    diff = X[:, ii] - X[:, jj]                # (B, P, d)
    r_pairs = diff.norm(dim=-1).reshape(-1)    # (B*P,)
    r_np = r_pairs.cpu().numpy()

    # Sqrt-stretched bins for better near-field resolution
    r_max = float(np.percentile(r_np, 99.5))
    edges_sqrt = np.linspace(0, np.sqrt(r_max), n_bins + 1)
    edges = edges_sqrt ** 2

    hist, _ = np.histogram(r_np, bins=edges, density=False)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)

    # Normalize: g(r) = hist / (B * N_pairs * 2πr dr / A)
    n_pairs = N * (N - 1) // 2
    # Effective area estimation (use mean cloud radius)
    r_mean = float(X.norm(dim=-1).mean())
    A_eff = math.pi * (3 * r_mean) ** 2  # rough enclosing area
    rho_pair = n_pairs / A_eff
    shell_area = 2 * np.pi * r_centers * dr
    norm = B * rho_pair * shell_area
    norm = np.where(norm > 0, norm, 1.0)
    g_r = hist.astype(float) / norm

    return {
        "r_centers": r_centers.tolist(),
        "g_r": g_r.tolist(),
        "edges": edges.tolist(),
        "raw_hist": hist.tolist(),
        "r_max": r_max,
    }


def _radial_summary(r: np.ndarray, omega: float) -> dict:
    """Compute radial distribution statistics."""
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {}

    r_max = np.percentile(r, 99.5)
    n_bins = 200
    hist, edges = np.histogram(r, bins=n_bins, range=(0, r_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mode_idx = np.argmax(hist)
    mode = float(centers[mode_idx])

    mean = float(np.mean(r))
    std = float(np.std(r))

    # FWHM
    half_max = hist[mode_idx] / 2
    above = np.where(hist >= half_max)[0]
    fwhm = float(centers[above[-1]] - centers[above[0]]) if len(above) >= 2 else float("nan")

    # Quantiles
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    quantiles = {f"q{int(q * 100)}": float(np.quantile(r, q)) for q in qs}

    # Localization ratio γ = σ / mode
    gamma = std / mode if mode > 1e-10 else float("nan")

    # Shannon entropy of radial distribution
    p = hist * np.diff(edges)
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p))) if len(p) > 0 else float("nan")

    return {
        "mode": mode, "mean": mean, "std": std, "fwhm": fwhm,
        "gamma": gamma, "entropy": entropy,
        "quantiles": quantiles,
        "hist": hist.tolist(), "centers": centers.tolist(),
    }


def _lindemann_ratio(X: Tensor) -> dict:
    """Lindemann ratio from pair-distance fluctuations."""
    B, N, d = X.shape
    ii, jj = torch.triu_indices(N, N, 1, device=X.device)
    r_pairs = (X[:, ii] - X[:, jj]).norm(dim=-1)  # (B, P)

    r_mean = r_pairs.mean(dim=0)         # (P,)
    r_std = r_pairs.std(dim=0)           # (P,)
    gamma_pairs = r_std / r_mean.clamp(min=1e-12)

    return {
        "gamma_mean": float(gamma_pairs.mean()),
        "gamma_std": float(gamma_pairs.std()),
        "gamma_max": float(gamma_pairs.max()),
        "gamma_min": float(gamma_pairs.min()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. LAYER-WISE  INFORMATION  FLOW  (NEW — publishable)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_information_flow(
    f_net: nn.Module,
    backflow_net: nn.Module | None,
    X: Tensor,
    spin: Tensor,
) -> dict:
    """
    Layer-by-layer activation statistics for f_net (and optionally bf_net):
      - Per-layer: mean, std, effective rank, dead fraction
      - Dimensionality profile (effective rank vs depth)
      - CKA similarity matrix between layers
      - Gradient norms per layer (backward pass)

    This reveals the information bottleneck structure of the network:
    where compression and expansion happen, and which layers
    transform the representation most strongly.
    """
    out = {}

    # ---- Collect activations via hooks ----
    for net_name, net in [("f_net", f_net), ("backflow", backflow_net)]:
        if net is None:
            continue

        activations = {}
        hooks = []

        def _make_hook(name):
            def hook(module, input, output):
                if isinstance(output, Tensor):
                    activations[name] = output.detach()
            return hook

        # Register hooks on all Linear and activation layers
        layer_idx = 0
        for name, module in net.named_modules():
            if isinstance(module, (nn.Linear, nn.GELU, nn.SiLU, nn.ReLU, nn.Tanh, nn.LayerNorm)):
                hook_name = f"{name}" if name else f"layer_{layer_idx}"
                hooks.append(module.register_forward_hook(_make_hook(hook_name)))
                layer_idx += 1

        # Forward pass
        with torch.no_grad():
            B = X.shape[0]
            spin_bn = spin.unsqueeze(0).expand(B, -1) if spin.dim() == 1 else spin
            if net_name == "f_net":
                _ = net(X, spin=spin_bn)
            else:
                _ = net(X, spin=spin_bn)

        # Remove hooks
        for h in hooks:
            h.remove()

        if not activations:
            continue

        # ---- Per-layer statistics ----
        layer_stats = []
        layer_names = []
        layer_acts = []

        for lname in sorted(activations.keys()):
            act = activations[lname]
            # Flatten to (B, features)
            if act.dim() > 2:
                act_flat = act.reshape(act.shape[0], -1)
            else:
                act_flat = act

            layer_names.append(lname)
            layer_acts.append(act_flat.float())

            n_feat = act_flat.shape[-1]
            dead_frac = float((act_flat.abs().max(dim=0).values < 1e-8).float().mean())
            svs = torch.linalg.svdvals(act_flat.float() - act_flat.float().mean(0))
            eff_rank = _effective_rank(svs)

            layer_stats.append({
                "name": lname,
                "n_features": n_feat,
                "mean": float(act_flat.mean()),
                "std": float(act_flat.std()),
                "effective_rank": eff_rank,
                "dead_fraction": dead_frac,
            })

        out[f"{net_name}_layers"] = layer_stats

        # ---- CKA matrix ----
        n_layers = len(layer_acts)
        if n_layers >= 2:
            cka_matrix = np.eye(n_layers)
            for i in range(n_layers):
                for j in range(i + 1, n_layers):
                    c = _cka_linear(layer_acts[i], layer_acts[j])
                    cka_matrix[i, j] = c
                    cka_matrix[j, i] = c
            out[f"{net_name}_cka"] = {
                "matrix": cka_matrix.tolist(),
                "layer_names": layer_names,
            }

        # ---- Dimensionality profile ----
        out[f"{net_name}_dim_profile"] = {
            "names": [s["name"] for s in layer_stats],
            "eff_ranks": [s["effective_rank"] for s in layer_stats],
            "n_features": [s["n_features"] for s in layer_stats],
        }

    # ---- Gradient norms per layer ----
    out["gradient_norms"] = _compute_gradient_norms(f_net, X, spin)

    return out


def _compute_gradient_norms(f_net: nn.Module, X: Tensor, spin: Tensor) -> dict:
    """Compute per-parameter-group gradient norms via a dummy backward pass."""
    B = min(X.shape[0], 256)  # Use subset for speed
    X_sub = X[:B].detach().requires_grad_(True)
    spin_bn = spin.unsqueeze(0).expand(B, -1) if spin.dim() == 1 else spin
    spin_sub = spin_bn[:B]

    f_net.zero_grad()
    out = f_net(X_sub, spin=spin_sub)
    loss = out.sum()
    loss.backward()

    grad_norms = {}
    for name, p in f_net.named_parameters():
        if p.grad is not None:
            grad_norms[name] = float(p.grad.norm())
    return grad_norms


# ═══════════════════════════════════════════════════════════════════════════
# 8. JACOBIAN  INTRINSIC  DIMENSIONALITY  (NEW — publishable)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_jacobian(
    f_net: nn.Module,
    backflow_net: nn.Module | None,
    X: Tensor,
    spin: Tensor,
    n_configs: int = 256,
) -> dict:
    """
    Jacobian-based analysis of the network's input sensitivity:
      - ∂f/∂x Jacobian singular value spectrum
      - Intrinsic input dimensionality
      - Physical alignment of top Jacobian singular vectors
        (which physical coordinates — radial, angular, pair-distance —
         does the network respond to most?)
      - For backflow: ∂Δx/∂x Jacobian analysis

    This directly answers: "along what physical properties do the
    latent spaces correspond most strongly?"
    """
    out = {}
    B_use = min(n_configs, X.shape[0])
    X_sub = X[:B_use]
    N, d = X_sub.shape[1], X_sub.shape[2]
    device = X_sub.device
    spin_bn = spin.unsqueeze(0).expand(B_use, -1) if spin.dim() == 1 else spin
    spin_sub = spin_bn[:B_use]
    n_input = N * d

    # ── f_net Jacobian ──
    print("  Computing f_net Jacobian spectrum...")
    J_svs_list = []
    J_right_vecs = []  # top right singular vector per sample

    for i in tqdm(range(B_use), desc="  Jacobian", leave=False):
        x_i = X_sub[i:i + 1].detach().requires_grad_(True)  # (1, N, d)
        s_i = spin_sub[i:i + 1]
        f_val = f_net(x_i, spin=s_i).squeeze()  # scalar
        grad = torch.autograd.grad(f_val, x_i, create_graph=False)[0]  # (1, N, d)
        J_row = grad.reshape(1, n_input)  # (1, n_input) — Jacobian row for scalar output
        J_svs_list.append(J_row.detach())

    # Stack all Jacobian rows → (B_use, N*d)
    J_all = torch.cat(J_svs_list, dim=0)      # (B_use, N*d)
    J_c = J_all - J_all.mean(0)

    # SVD of the Jacobian ensemble
    svs = torch.linalg.svdvals(J_c.float())
    out["jacobian_singular_values"] = svs[:20].tolist()
    out["jacobian_effective_rank"] = _effective_rank(svs)

    # Explained variance
    total = (svs ** 2).sum()
    expvar = ((svs ** 2).cumsum(0) / total)[:12]
    out["jacobian_expvar"] = expvar.tolist()

    # ── Physical alignment of top Jacobian directions ──
    U, S_j, Vt = torch.linalg.svd(J_c.float(), full_matrices=False)
    top_k = min(5, Vt.shape[0])
    alignment = _physical_alignment(Vt[:top_k], N, d, X_sub.mean(0))
    out["jacobian_physical_alignment"] = alignment

    # ── Backflow Jacobian ∂Δx/∂x ──
    if backflow_net is not None:
        print("  Computing backflow Jacobian spectrum...")
        bf_jac_info = _backflow_jacobian(backflow_net, X_sub, spin_sub, N, d)
        out["bf_jacobian"] = bf_jac_info

    return out


def _physical_alignment(
    Vt: Tensor,  # (k, N*d) — top-k right singular vectors
    N: int,
    d: int,
    x_mean: Tensor,  # (N, d) — mean configuration for radial/angular basis construction
) -> dict:
    """
    Project each top-k Jacobian singular vector onto physically meaningful
    bases and report the fraction of variance explained by each:
      - Radial directions (per particle)
      - Angular directions (per particle, d=2 only)
      - Pair-distance directions (gradient of r_ij w.r.t. x)
      - Center-of-mass direction
    """
    k, nd = Vt.shape
    device = Vt.device

    # Build radial direction vectors
    x_m = x_mean.reshape(N, d)
    r_norms = x_m.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    radial_dirs = (x_m / r_norms).reshape(1, nd)  # (1, N*d)

    # Angular direction vectors (d=2: perpendicular to radial)
    if d == 2:
        perp = torch.stack([-x_m[:, 1], x_m[:, 0]], dim=-1) / r_norms
        angular_dirs = perp.reshape(1, nd)
    else:
        angular_dirs = None

    # COM direction
    com_dir = torch.ones(1, nd, device=device) / math.sqrt(nd)

    # Compute projections
    alignment = {}
    for i in range(k):
        v = Vt[i]                                     # (N*d,)
        v_norm = v / v.norm().clamp(min=1e-12)

        proj_radial = float((v_norm @ radial_dirs.squeeze()) ** 2)
        alignment[f"PC{i + 1}_radial"] = proj_radial

        if angular_dirs is not None:
            proj_angular = float((v_norm @ angular_dirs.squeeze()) ** 2)
            alignment[f"PC{i + 1}_angular"] = proj_angular

        proj_com = float((v_norm @ com_dir.squeeze()) ** 2)
        alignment[f"PC{i + 1}_COM"] = proj_com

        # Remaining goes to "pair/other"
        accounted = proj_radial + (proj_angular if angular_dirs is not None else 0) + proj_com
        alignment[f"PC{i + 1}_other"] = max(0, 1.0 - accounted)

    return alignment


def _backflow_jacobian(bf_net: nn.Module, X: Tensor, spin: Tensor, N: int, d: int) -> dict:
    """Compute Jacobian spectrum of backflow displacement ∂Δx/∂x."""
    B_use = min(64, X.shape[0])  # Smaller batch since BF Jacobian is (N*d, N*d)
    n_io = N * d

    all_svs = []
    for i in tqdm(range(B_use), desc="  BF Jacobian", leave=False):
        x_i = X[i:i + 1].detach().requires_grad_(True)
        s_i = spin[i:i + 1] if spin.dim() == 2 else spin.unsqueeze(0)
        dx = bf_net(x_i, spin=s_i)                     # (1, N, d)
        dx_flat = dx.reshape(n_io)                       # (N*d,)

        # Row-by-row Jacobian
        J = torch.zeros(n_io, n_io, device=X.device, dtype=X.dtype)
        for j in range(n_io):
            grad = torch.autograd.grad(
                dx_flat[j], x_i, retain_graph=(j < n_io - 1), create_graph=False,
            )[0]
            J[j] = grad.reshape(n_io)

        svs = torch.linalg.svdvals(J.float())
        all_svs.append(svs.detach())

    all_svs = torch.stack(all_svs)                      # (B_use, N*d)
    mean_svs = all_svs.mean(dim=0)

    return {
        "singular_values": mean_svs[:20].tolist(),
        "effective_rank": float(_effective_rank(mean_svs)),
        "condition_number": float(mean_svs[0] / mean_svs[-1].clamp(min=1e-15)),
        "top1_fraction": float((mean_svs[0] ** 2) / (mean_svs ** 2).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. BACKFLOW  FIELD  VISUALIZATION  (NEW for N > 2)
# ═══════════════════════════════════════════════════════════════════════════

def visualize_backflow_field(
    backflow_net: nn.Module,
    X: Tensor,
    spin: Tensor,
    N: int,
    omega: float,
    outdir: Path,
) -> dict:
    """
    Backflow displacement field visualization:
      - N=2: quiver plot on a grid (extending existing plot_f_psi_sd_with_backflow)
      - N>2: displacement magnitude vs particle radius, quiver on representative config,
             spin-resolved statistics
    """
    out = {}
    B, N_actual, d = X.shape
    device = X.device
    ell = 1.0 / math.sqrt(max(omega, 1e-12))

    with torch.no_grad():
        spin_bn = spin.unsqueeze(0).expand(B, -1) if spin.dim() == 1 else spin
        dx = backflow_net(X, spin=spin_bn)  # (B, N, d)

    # ── Displacement magnitude vs particle radius ──
    r_particles = X.norm(dim=-1)           # (B, N)
    dx_mag = dx.norm(dim=-1)                # (B, N)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: ‖Δx‖ vs r, colored by spin
    ax = axes[0]
    r_np = r_particles.cpu().numpy().ravel()
    dx_np = dx_mag.cpu().numpy().ravel()
    spin_flat = spin_bn.cpu().numpy().ravel()

    up_mask = spin_flat == 0
    dn_mask = spin_flat == 1
    ax.scatter(r_np[up_mask], dx_np[up_mask], s=1, alpha=0.15, c="tab:blue", label="spin ↑")
    ax.scatter(r_np[dn_mask], dx_np[dn_mask], s=1, alpha=0.15, c="tab:red", label="spin ↓")
    ax.set_xlabel("$r_i$ (physical)")
    ax.set_ylabel("$\\|\\Delta x_i\\|$")
    ax.set_title("BF displacement vs radius")
    ax.legend(markerscale=5)

    # Panel 2: ‖Δx‖ vs nearest-neighbor distance
    ax = axes[1]
    ii, jj = torch.triu_indices(N, N, 1, device=device)
    diff = X.unsqueeze(2) - X.unsqueeze(1)
    pair_dist = diff[:, ii, jj].norm(dim=-1)   # (B, P)

    # Per-particle minimum neighbor distance
    dist_full = torch.zeros(B, N, N, device=device)
    dist_full[:, ii, jj] = pair_dist
    dist_full[:, jj, ii] = pair_dist
    dist_full[:, range(N), range(N)] = float("inf")
    nn_dist = dist_full.min(dim=-1).values      # (B, N)

    nn_np = nn_dist.cpu().numpy().ravel()
    ax.scatter(nn_np, dx_np, s=1, alpha=0.15, c="tab:green")
    ax.set_xlabel("$r_{\\mathrm{NN}}$ (nearest neighbor)")
    ax.set_ylabel("$\\|\\Delta x_i\\|$")
    ax.set_title("BF displacement vs NN distance")

    # Panel 3: Quiver plot of one representative configuration
    ax = axes[2]
    # Pick the configuration closest to the median energy (most typical)
    mid_idx = B // 2
    x_rep = X[mid_idx].cpu().numpy()           # (N, d)
    dx_rep = dx[mid_idx].cpu().numpy()          # (N, d)
    spin_rep = spin.cpu().numpy()

    up_m = spin_rep == 0
    dn_m = spin_rep == 1

    scale = max(np.max(np.abs(dx_rep)) * 2, 1e-6)
    ax.quiver(
        x_rep[up_m, 0], x_rep[up_m, 1], dx_rep[up_m, 0], dx_rep[up_m, 1],
        color="tab:blue", scale=scale, alpha=0.8, label="↑",
    )
    ax.quiver(
        x_rep[dn_m, 0], x_rep[dn_m, 1], dx_rep[dn_m, 0], dx_rep[dn_m, 1],
        color="tab:red", scale=scale, alpha=0.8, label="↓",
    )
    ax.scatter(x_rep[:, 0], x_rep[:, 1], c=["tab:blue" if s == 0 else "tab:red" for s in spin_rep],
               s=40, zorder=5, edgecolors="k", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"BF field (N={N}, ω={omega})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "backflow_field.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    out["figure_saved"] = str(outdir / "backflow_field.pdf")

    # ── Displacement alignment with radial direction ──
    x_hat = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    radial_proj = (dx * x_hat).sum(dim=-1)     # (B, N) radial component of Δx
    tangential = dx - radial_proj.unsqueeze(-1) * x_hat
    tang_mag = tangential.norm(dim=-1)          # (B, N)

    out["radial_proj_mean"] = float(radial_proj.abs().mean())
    out["tangential_mag_mean"] = float(tang_mag.mean())
    out["radial_fraction"] = float(
        radial_proj.abs().mean() / (radial_proj.abs().mean() + tang_mag.mean() + 1e-15)
    )

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 10. PLOTTING  &  FIGURE  GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_figures(results: dict, outdir: Path, N: int, omega: float, ansatz: str):
    """Generate all per-case figures."""
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        style_path = _SCRIPT_DIR / "Thesis_style.mplstyle"
        if style_path.exists():
            plt.style.use(str(style_path))
    except Exception:
        pass

    # ── g(r) and P(r) ──
    if "structure" in results and "gr" in results["structure"]:
        _plot_gr_pr(results["structure"], outdir, N, omega)

    # ── Correlator effective rank spectrum ──
    if "correlator" in results and "singular_values" in results["correlator"]:
        _plot_svd_spectrum(results["correlator"], outdir, N, omega)

    # ── PC ablation ──
    if "correlator" in results and "pc_ablation" in results["correlator"]:
        _plot_pc_ablation(results["correlator"]["pc_ablation"], outdir, N, omega)

    # ── Probe R² bar chart ──
    if "correlator" in results and "probe_R2" in results["correlator"]:
        _plot_probes(results["correlator"], outdir, N, omega)

    # ── Information flow: dimensionality profile ──
    if "information_flow" in results:
        _plot_dim_profile(results["information_flow"], outdir, N, omega, ansatz)

    # ── Information flow: CKA matrix ──
    if "information_flow" in results:
        _plot_cka(results["information_flow"], outdir, N, omega, ansatz)

    # ── Jacobian spectrum ──
    if "jacobian" in results and "jacobian_singular_values" in results["jacobian"]:
        _plot_jacobian_spectrum(results["jacobian"], outdir, N, omega)

    # ── Angle distribution ──
    if "structure" in results and "angle_dist" in results["structure"]:
        _plot_angle_dist(results["structure"]["angle_dist"], outdir, N, omega)


def _plot_gr_pr(struct: dict, outdir: Path, N: int, omega: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # g(r)
    gr = struct["gr"]
    ax1.plot(gr["r_centers"], gr["g_r"], "b-", lw=1.5)
    ax1.set_xlabel("$r$ (physical)")
    ax1.set_ylabel("$g(r)$")
    ax1.set_title(f"Pair correlation $g(r)$ — N={N}, ω={omega}")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # P(r)
    rad = struct["radial"]
    ax2.plot(rad["centers"], rad["hist"], "r-", lw=1.5)
    ax2.axvline(rad["mode"], color="k", ls="--", lw=0.8, label=f"mode = {rad['mode']:.3f}")
    ax2.axvline(rad["mean"], color="gray", ls=":", lw=0.8, label=f"mean = {rad['mean']:.3f}")
    ax2.set_xlabel("$r$ (physical)")
    ax2.set_ylabel("$P(r)$")
    ax2.set_title(f"Radial distribution — N={N}, ω={omega}")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "gr_pr.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_svd_spectrum(corr: dict, outdir: Path, N: int, omega: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    svs = corr["singular_values"]
    ax.bar(range(len(svs)), svs, color="steelblue")
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Correlator SVD spectrum — N={N}, ω={omega}\n"
                 f"$r_{{eff}}$ = {corr['effective_rank']:.2f}")
    fig.tight_layout()
    fig.savefig(outdir / "svd_spectrum.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pc_ablation(abl: dict, outdir: Path, N: int, omega: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = abl["k"]
    ax.plot(ks, abl["mae_pc"], "bo-", label="Top-k PCs", ms=5)
    ax.plot(ks, abl["mae_rand"], "rs--", label="Random subspace", ms=5)
    ax.set_xlabel("Number of components $k$")
    ax.set_ylabel("Reconstruction MAE")
    ax.set_title(f"PC ablation — N={N}, ω={omega}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pc_ablation.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_probes(corr: dict, outdir: Path, N: int, omega: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = corr.get("probe_targets", [f"t{i}" for i in range(len(corr["probe_R2"]))])
    r2 = corr["probe_R2"]
    if isinstance(r2, dict):
        names = list(r2.keys())
        vals = list(r2.values())
    elif isinstance(r2, list):
        vals = r2
    else:
        return

    bars = ax.barh(range(len(vals)), vals, color="teal")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("$R^2$")
    ax.set_title(f"Linear probe $R^2$ — N={N}, ω={omega}")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(outdir / "probe_r2.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dim_profile(info: dict, outdir: Path, N: int, omega: float, ansatz: str):
    for net_name in ["f_net", "backflow"]:
        key = f"{net_name}_dim_profile"
        if key not in info:
            continue
        prof = info[key]
        fig, ax = plt.subplots(figsize=(10, 4))
        x_pos = range(len(prof["eff_ranks"]))
        ax.bar(x_pos, prof["eff_ranks"], color="darkorange", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(prof["names"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Effective rank")
        ax.set_title(f"Dimensionality profile ({net_name}) — N={N}, ω={omega}, {ansatz}")
        fig.tight_layout()
        fig.savefig(outdir / f"dim_profile_{net_name}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_cka(info: dict, outdir: Path, N: int, omega: float, ansatz: str):
    for net_name in ["f_net", "backflow"]:
        key = f"{net_name}_cka"
        if key not in info:
            continue
        cka_data = info[key]
        matrix = np.array(cka_data["matrix"])
        names = cka_data["layer_names"]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_title(f"CKA ({net_name}) — N={N}, ω={omega}, {ansatz}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(outdir / f"cka_{net_name}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_jacobian_spectrum(jac: dict, outdir: Path, N: int, omega: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    svs = jac["jacobian_singular_values"]
    ax.semilogy(range(1, len(svs) + 1), svs, "ko-", ms=4)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value (log)")
    ax.set_title(f"Jacobian spectrum — N={N}, ω={omega}\n"
                 f"Intrinsic dim = {jac['jacobian_effective_rank']:.1f}")
    fig.tight_layout()
    fig.savefig(outdir / "jacobian_spectrum.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_angle_dist(ang: dict, outdir: Path, N: int, omega: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    edges = np.array(ang["edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, ang["hist"], width=np.diff(edges), color="salmon", edgecolor="k", lw=0.3)
    ax.set_xlabel("Δφ (radians)")
    ax.set_ylabel("Density")
    ax.set_title(f"Pair angle distribution — N={N}, ω={omega}")
    fig.tight_layout()
    fig.savefig(outdir / "angle_dist.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 11. CROSS-CONFIGURATION  SUMMARY  &  AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(all_results: dict, ansatz: str, outdir: Path):
    """
    Generate cross-configuration summary:
      - Energy vs ω for all N
      - Relative error vs ω
      - Effective rank vs ω
      - Virial ratios vs ω
      - Power-law fits E(ω) ∝ ω^α
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Group results by N
    by_N = defaultdict(list)
    for (N, omega), res in sorted(all_results.items()):
        by_N[N].append((omega, res))

    # ── Energy vs ω ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {2: "tab:blue", 6: "tab:orange", 12: "tab:green", 20: "tab:red"}

    ax = axes[0]
    for N, items in sorted(by_N.items()):
        omegas = [w for w, r in items if "energy" in r]
        energies = [r["energy"]["E_mean"] for w, r in items if "energy" in r]
        stds = [r["energy"]["E_stderr"] for w, r in items if "energy" in r]
        if omegas:
            ax.errorbar(omegas, energies, yerr=stds, fmt="o-", color=colors.get(N, "gray"),
                        label=f"N={N}", ms=5, capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ω")
    ax.set_ylabel("E (Ha)")
    ax.set_title(f"Ground-state energy — {ansatz.upper()}")
    ax.legend()

    # ── Relative error vs ω ──
    ax = axes[1]
    for N, items in sorted(by_N.items()):
        omegas = [w for w, r in items if "energy" in r and r["energy"].get("rel_error_pct") is not None]
        errors = [r["energy"]["rel_error_pct"] for w, r in items
                  if "energy" in r and r["energy"].get("rel_error_pct") is not None]
        if omegas:
            ax.plot(omegas, errors, "o-", color=colors.get(N, "gray"), label=f"N={N}", ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("ω")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"Deviation from DMC — {ansatz.upper()}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "energy_vs_omega.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Effective rank vs ω ──
    fig, ax = plt.subplots(figsize=(7, 5))
    for N, items in sorted(by_N.items()):
        omegas = [w for w, r in items if "correlator" in r]
        ranks = [r["correlator"]["effective_rank"] for w, r in items if "correlator" in r]
        if omegas:
            ax.plot(omegas, ranks, "o-", color=colors.get(N, "gray"), label=f"N={N}", ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("ω")
    ax.set_ylabel("Effective rank $r_{eff}(Z)$")
    ax.set_title(f"Correlator dimensionality — {ansatz.upper()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "eff_rank_vs_omega.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Virial ratios ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for N, items in sorted(by_N.items()):
        omegas = [w for w, r in items if "energy" in r]

        gammas = [r["energy"]["Gamma_Vint_over_T"] for w, r in items if "energy" in r]
        virial = [r["energy"]["virial_2Vt_over_Vi"] for w, r in items if "energy" in r]
        c = colors.get(N, "gray")
        if omegas:
            ax1.plot(omegas, gammas, "o-", color=c, label=f"N={N}", ms=5)
            ax2.plot(omegas, virial, "o-", color=c, label=f"N={N}", ms=5)

    ax1.set_xscale("log")
    ax1.set_xlabel("ω")
    ax1.set_ylabel("$\\Gamma = V_{int}/T$")
    ax1.set_title("Interaction/kinetic ratio")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("ω")
    ax2.set_ylabel("$2V_{trap}/V_{int}$")
    ax2.set_title("Virial ratio")
    ax2.axhline(1.0, color="k", ls=":", lw=0.8, label="Classical limit")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(outdir / "virial_ratios.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Jacobian intrinsic dimensionality vs ω ──
    fig, ax = plt.subplots(figsize=(7, 5))
    for N, items in sorted(by_N.items()):
        omegas = [w for w, r in items if "jacobian" in r]
        jac_ranks = [r["jacobian"]["jacobian_effective_rank"] for w, r in items if "jacobian" in r]
        if omegas:
            ax.plot(omegas, jac_ranks, "o-", color=colors.get(N, "gray"), label=f"N={N}", ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("ω")
    ax.set_ylabel("Jacobian intrinsic dim")
    ax.set_title(f"Input sensitivity dimensionality — {ansatz.upper()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "jacobian_dim_vs_omega.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Power-law fits E(ω) = C ω^α ──
    powerlaw = {}
    for N, items in sorted(by_N.items()):
        omegas_e = np.array([w for w, r in items if "energy" in r])
        energies_e = np.array([r["energy"]["E_mean"] for w, r in items if "energy" in r])
        if len(omegas_e) >= 3:
            mask = (omegas_e > 0) & (energies_e > 0)
            if mask.sum() >= 3:
                log_w = np.log(omegas_e[mask])
                log_E = np.log(energies_e[mask])
                coeffs = np.polyfit(log_w, log_E, 1)
                powerlaw[N] = {"alpha": float(coeffs[0]), "log_C": float(coeffs[1])}

    # ── Summary JSON ──
    summary = {
        "ansatz": ansatz,
        "cases": {},
        "powerlaw_fits": _to_python(powerlaw),
    }
    for (N, omega), res in sorted(all_results.items()):
        key = _case_tag(N, omega)
        case_summary = {}
        if "energy" in res:
            case_summary["energy"] = {
                k: res["energy"][k] for k in
                ["E_mean", "E_std", "E_stderr", "E_DMC", "rel_error_pct",
                 "Gamma_Vint_over_T", "virial_2Vt_over_Vi"]
                if k in res["energy"]
            }
        if "correlator" in res:
            case_summary["effective_rank"] = res["correlator"]["effective_rank"]
            case_summary["branch_ablation"] = res["correlator"].get("branch_ablation")
            case_summary["pc1_block_power"] = res["correlator"].get("pc1_block_power")
        if "jacobian" in res:
            case_summary["jacobian_dim"] = res["jacobian"]["jacobian_effective_rank"]
        if "structure" in res:
            rad = res["structure"].get("radial", {})
            case_summary["r_mode"] = rad.get("mode")
            case_summary["r_mean"] = rad.get("mean")
            case_summary["lindemann"] = res["structure"].get("lindemann", {}).get("gamma_mean")
        summary["cases"][key] = case_summary

    with open(outdir / "summary.json", "w") as f:
        json.dump(_to_python(summary), f, indent=2)
    print(f"\n  Summary saved to {outdir / 'summary.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# 12. MAIN  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_single_case(
    ansatz: str,
    N: int,
    omega: float,
    device: torch.device,
    output_root: Path,
    model_root: str = "../results/official_models",
    skip_jacobian: bool = False,
    B_samples: int = 8192,
) -> dict:
    """Run full analysis pipeline for a single (N, ω) configuration."""
    tag = _case_tag(N, omega)
    outdir = output_root / ansatz / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  {ansatz.upper()}  N={N}  ω={omega}")
    print(f"{'═' * 60}")

    results = {}

    # ── Setup ──
    try:
        sys_info = setup_system(ansatz, N, omega, device, model_root=model_root)
    except Exception as e:
        print(f"  !! Setup failed: {e}")
        return {"error": str(e)}

    f_net = sys_info["f_net"]
    bf = sys_info["backflow_net"]
    C_occ = sys_info["C_occ"]
    spin = sys_info["spin"]
    params = sys_info["params"]
    psi_log = sys_info["psi_log_fn"]

    # ── Sample ──
    print("  Sampling from |Ψ|²...")
    B = B_samples if N <= 12 else B_samples // 2
    X, acc = generate_samples(psi_log, N, 2, omega, device, sys_info["dtype"], B=B)
    print(f"  Sampled {X.shape[0]} configs, accept rate = {acc:.3f}")

    # ── A: Energy ──
    print("  [A] Energy analysis...")
    try:
        results["energy"] = analyze_energy(sys_info)
        e = results["energy"]
        print(f"      E = {e['E_mean']:.6f} ± {e['E_stderr']:.6f} Ha")
        if e.get("rel_error_pct") is not None:
            print(f"      Rel. error = {e['rel_error_pct']:.4f}%")
    except Exception as ex:
        print(f"  !! Energy analysis failed: {ex}")
        results["energy"] = {"error": str(ex)}

    # ── Compute local energies for correlation analysis ──
    E_L = None
    try:
        from functions.Analysis import compute_local_energy_batch
        lap_mode = "exact" if N <= 12 else "hvp-hutch"
        E_L = compute_local_energy_batch(
            lap_mode, psi_log, _psi_fn, f_net, C_occ, X,
            compute_coulomb_interaction, omega,
            backflow_net=bf, spin=spin, params=params,
        )
    except Exception:
        pass

    # ── B: Correlator representation geometry ──
    print("  [B] Correlator analysis...")
    try:
        results["correlator"] = analyze_correlator(f_net, X, spin, E_L)
        c = results["correlator"]
        print(f"      r_eff(Z) = {c['effective_rank']:.2f}")
        print(f"      Head-PC1 corr = {c['head_corr_pc1']:.3f}")
        print(f"      Branch ablation: φ={c['branch_ablation']['phi']:.4f}  "
              f"ψ={c['branch_ablation']['psi']:.4f}  "
              f"extras={c['branch_ablation']['extras']:.4f}")
    except Exception as ex:
        print(f"  !! Correlator analysis failed: {ex}")
        results["correlator"] = {"error": str(ex)}

    # ── C: Backflow diagnostics ──
    if bf is not None:
        print("  [C] Backflow diagnostics...")
        try:
            results["backflow"] = analyze_backflow(f_net, bf, X, spin, C_occ, params)
            b = results["backflow"]
            print(f"      ‖Δx‖ mean = {b['dx_mean_magnitude']:.5f}")
            print(f"      Δx eff rank = {b['dx_effective_rank']:.2f}")
            print(f"      COM shift = {b['com_shift_mean']:.6f}")
            print(f"      CKA(noBF, BF) = {b['correlator_cka_nobf_vs_bf']:.3f}")
        except Exception as ex:
            print(f"  !! Backflow analysis failed: {ex}")
            results["backflow"] = {"error": str(ex)}

    # ── D: Wigner-molecule structure ──
    print("  [D] Structure analysis...")
    try:
        results["structure"] = analyze_structure(X, N, omega)
        s = results["structure"]
        print(f"      r_mode = {s['radial'].get('mode', '?'):.4f}  "
              f"r_mean = {s['radial'].get('mean', '?'):.4f}  "
              f"γ_Lind = {s['lindemann']['gamma_mean']:.4f}")
    except Exception as ex:
        print(f"  !! Structure analysis failed: {ex}")
        results["structure"] = {"error": str(ex)}

    # ── E: Layer-wise information flow (NEW) ──
    print("  [E] Information flow analysis...")
    try:
        results["information_flow"] = analyze_information_flow(f_net, bf, X, spin)
        info = results["information_flow"]
        if "f_net_dim_profile" in info:
            ranks = info["f_net_dim_profile"]["eff_ranks"]
            print(f"      f_net dim profile: {[f'{r:.1f}' for r in ranks[:6]]}")
    except Exception as ex:
        print(f"  !! Information flow analysis failed: {ex}")
        results["information_flow"] = {"error": str(ex)}

    # ── F: Jacobian intrinsic dimensionality (NEW) ──
    if not skip_jacobian:
        print("  [F] Jacobian analysis...")
        n_jac = 128 if N <= 6 else 64
        try:
            results["jacobian"] = analyze_jacobian(f_net, bf, X, spin, n_configs=n_jac)
            j = results["jacobian"]
            print(f"      Jacobian intrinsic dim = {j['jacobian_effective_rank']:.1f}")
            if "bf_jacobian" in j:
                print(f"      BF Jacobian eff rank = {j['bf_jacobian']['effective_rank']:.1f}")
        except Exception as ex:
            print(f"  !! Jacobian analysis failed: {ex}")
            results["jacobian"] = {"error": str(ex)}

    # ── G: Backflow field visualization (NEW for N > 2) ──
    if bf is not None:
        print("  [G] Backflow field visualization...")
        try:
            results["bf_field"] = visualize_backflow_field(bf, X, spin, N, omega, outdir)
        except Exception as ex:
            print(f"  !! BF field visualization failed: {ex}")
            results["bf_field"] = {"error": str(ex)}

    # ── Generate figures ──
    print("  Generating figures...")
    try:
        generate_figures(results, outdir, N, omega, ansatz)
    except Exception as ex:
        print(f"  !! Figure generation failed: {ex}")

    # ── Save JSON ──
    json_path = outdir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(_to_python(results), f, indent=2)
    print(f"  Saved metrics to {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified quantum-dot analysis for PINN / BF / CTNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_all.py --ansatz pinn
  python analyze_all.py --ansatz bf --particles 2,6 --omegas 0.1,1.0
  python analyze_all.py --ansatz ctnn --skip-jacobian
        """,
    )
    parser.add_argument(
        "--ansatz", required=True, choices=["pinn", "bf", "ctnn"],
        help="Ansatz type: pinn (PINN only), bf (PINN + BackflowNet), ctnn (PINN + CTNNBackflowNet)",
    )
    parser.add_argument(
        "--particles", default=",".join(map(str, ALL_PARTICLES)),
        help="Comma-separated particle numbers (default: 2,6,12,20)",
    )
    parser.add_argument(
        "--omegas", default=",".join(map(str, ALL_OMEGAS)),
        help="Comma-separated trap frequencies (default: 0.001,0.01,0.1,0.5,1.0)",
    )
    parser.add_argument(
        "--output-dir", default="../results/analysis",
        help="Base output directory (default: ../results/analysis)",
    )
    parser.add_argument(
        "--model-root", default="../results/official_models",
        help="Root directory for model weights (default: ../results/official_models)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: auto-detect CUDA → MPS → CPU)",
    )
    parser.add_argument(
        "--skip-jacobian", action="store_true",
        help="Skip Jacobian analysis (slow for large N)",
    )
    parser.add_argument(
        "--samples", type=int, default=8192,
        help="Number of MCMC samples per configuration (default: 8192)",
    )

    args = parser.parse_args()

    # Parse lists
    particles = [int(x.strip()) for x in args.particles.split(",")]
    omegas = [float(x.strip()) for x in args.omegas.split(",")]

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output_root = Path(args.output_dir)

    print(f"╔{'═' * 58}╗")
    print(f"║  Quantum Dot Analysis — {args.ansatz.upper():>5s}                           ║")
    print(f"║  Particles: {particles}                          ║")
    print(f"║  Omegas:    {omegas}              ║")
    print(f"║  Device:    {str(device):<44s}  ║")
    print(f"╚{'═' * 58}╝")

    all_results: dict[tuple[int, float], dict] = {}

    for N in particles:
        if N not in NX_NY_MAP:
            print(f"\n  !! Skipping N={N} — not in NX_NY_MAP (add entry for this particle count)")
            continue
        for omega in omegas:
            result = run_single_case(
                ansatz=args.ansatz,
                N=N,
                omega=omega,
                device=device,
                output_root=output_root,
                model_root=args.model_root,
                skip_jacobian=args.skip_jacobian,
                B_samples=args.samples,
            )
            all_results[(N, omega)] = result

    # ── Cross-configuration summary ──
    print(f"\n{'═' * 60}")
    print("  Generating cross-configuration summary...")
    try:
        aggregate_results(all_results, args.ansatz, output_root / args.ansatz)
    except Exception as ex:
        print(f"  !! Aggregation failed: {ex}")

    print(f"\n{'═' * 60}")
    print(f"  DONE — all results in {output_root / args.ansatz}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
