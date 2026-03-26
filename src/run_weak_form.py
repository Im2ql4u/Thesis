"""
Weak-form collocation training for BF + Jastrow
================================================
N=6, ω=1.0, E_DMC=20.15932

Replaces the strong-form collocation loss (which requires the Laplacian
∇²Ψ and backpropagates through it — 4th-order sensitivity) with the
weak-form / Rayleigh-quotient loss that only needs ∇Ψ (first derivatives).

Integration by parts on ⟨Ψ|H|Ψ⟩:
  -½ ∫ Ψ∇²Ψ dx = +½ ∫ |∇Ψ|² dx

so the energy functional becomes:
  E[Ψ] = ∫ [½|∇Ψ|² + V|Ψ|²] dx / ∫ |Ψ|² dx

With logΨ representation and importance sampling from q(x):
  E = Σ wᵢ [½|∇logΨ(xᵢ)|² + V(xᵢ)] / Σ wᵢ
  wᵢ = |Ψ(xᵢ)|² / q(xᵢ) = exp(2·logΨ(xᵢ) - logq(xᵢ))

This requires ONLY first derivatives of Ψ w.r.t. x — no Laplacian.
Parameter gradients are at most second-order (vs fourth-order in
the strong-form approach). This eliminates the conditioning catastrophe
that destabilized backflow training.

Supports three wavefunction types:
  1. BF + Jastrow (Slater × Jastrow, backflow shifts coordinates)
  2. Neural Pfaffian + Jastrow (Pfaffian × Jastrow)
  3. Jastrow only (Slater × Jastrow, no backflow — sanity check)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import (
    colloc_fd_loss as nn_colloc_fd_loss,
    importance_resample as nn_importance_resample,
    lookup_dmc_energy,
    psi_fn,
    rayleigh_hybrid_loss as nn_rayleigh_hybrid_loss,
    safe_percent_err as nn_safe_percent_err,
)
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import evaluate_basis_functions_torch_batch_2d
from jastrow_architectures import CTNNJastrowVCycle, CTNNJastrow
from PINN import CTNNBackflowNet, UnifiedCTNN

# ─── Constants ───
_manual = os.environ.get("CUDA_MANUAL_DEVICE")
if _manual is not None and torch.cuda.is_available():
    DEVICE = f"cuda:{_manual}" if _manual.isdigit() else _manual
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
DTYPE = torch.float64
N_ELEC = 6
DIM = 2
OMEGA = 1.0
E_DMC = 20.15932

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"


def default_dmc(n_elec, omega):
    return lookup_dmc_energy(int(n_elec), float(omega), allow_missing=True)


def _choose_basis_dims(n_occ):
    n_side = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
    return n_side, n_side


def safe_percent_err(E, E_ref):
    return nn_safe_percent_err(E, E_ref)


def finite_or(v, default=float("nan")):
    try:
        fv = float(v)
    except Exception:
        return default
    return fv if math.isfinite(fv) else default


def setup(n_elec=None, omega=None, e_dmc=None, seed=None):
    N = int(N_ELEC if n_elec is None else n_elec)
    d = DIM
    om = float(OMEGA if omega is None else omega)
    e_ref = default_dmc(N, om) if e_dmc is None else float(e_dmc)
    n_occ = N // 2
    nx, ny = _choose_basis_dims(n_occ)
    L = max(8.0, 3.0 / math.sqrt(om))
    config.update(
        omega=om, n_particles=N, d=d, L=L, n_grid=80,
        nx=nx, ny=ny, basis="cart", device=DEVICE, dtype="float64",
        seed=seed,
    )
    energies = sorted([(om * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)])
    C = np.zeros((nx * ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
    p = config.get().as_dict()
    p.update(device=DEVICE, torch_dtype=DTYPE, E=e_ref)
    return C_occ, p, nx, ny


# ═══════════════════════════════════════════════════════════════
#  Weak-form energy functional — the core innovation
# ═══════════════════════════════════════════════════════════════

def compute_grad_logpsi(psi_log_fn, x):
    """Compute ∇log|Ψ(x)| — only first derivatives, no Laplacian.

    Returns
    -------
    g  : (B, N, d) — gradient of logΨ w.r.t. x
    g2 : (B,) — |∇logΨ|² summed over all particles and dimensions
    """
    x = x.detach().requires_grad_(True)
    lp = psi_log_fn(x)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True)[0]  # (B, N, d)
    g2 = (g ** 2).sum(dim=(1, 2))  # (B,)
    return g, g2


def weak_form_local_energy(psi_log_fn, x, omega, params):
    """Compute the weak-form 'local energy' per sample.

    ẽ(x) = ½|∇logΨ(x)|² + V(x)

    This is the integrand of E[Ψ] under |Ψ|²-weighted sampling.
    Only requires first derivatives of Ψ w.r.t. x.

    Note: This equals E_L plus the quantum force correction:
      E_L = -½(∇²logΨ + |∇logΨ|²) + V
      ẽ    = ½|∇logΨ|² + V
    The difference is the Laplacian term, which averages to zero
    under |Ψ|² sampling (integration by parts). So ⟨ẽ⟩_{|Ψ|²} = ⟨E_L⟩_{|Ψ|²} = E.
    """
    _, g2 = compute_grad_logpsi(psi_log_fn, x)
    B = x.shape[0]
    T_weak = 0.5 * g2  # kinetic: ½|∇logΨ|²
    V = 0.5 * omega ** 2 * (x ** 2).sum(dim=(1, 2)) + compute_coulomb_interaction(
        x, params=params
    ).view(B)
    return T_weak + V  # (B,)


# ═══════════════════════════════════════════════════════════════
#  Sampling
# ═══════════════════════════════════════════════════════════════

def sample_gauss(n, omega, sigma_f=1.3):
    s = sigma_f / math.sqrt(omega)
    x = torch.randn(n, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * s
    Nd = N_ELEC * DIM
    lq = -0.5 * Nd * math.log(2 * math.pi * s ** 2) - x.reshape(n, -1).pow(2).sum(-1) / (2 * s ** 2)
    return x, lq


def _eval_mixture_logq_local(x, omega, sigma_fs):
    """Evaluate log-density of the Gaussian mixture at arbitrary points.

    Uses module-level N_ELEC, DIM globals. This is the local equivalent of
    eval_mixture_logq from Neural_Networks.py.
    """
    nc = len(sigma_fs)
    Nd = N_ELEC * DIM
    x_flat = x.reshape(x.shape[0], -1)  # (B, Nd)
    log_components = []
    for sf in sigma_fs:
        s = sf / math.sqrt(float(omega))
        log_norm = -0.5 * Nd * math.log(2 * math.pi * s ** 2)
        log_exp = -x_flat.pow(2).sum(-1) / (2 * s ** 2)
        log_components.append(log_norm + log_exp)
    log_stack = torch.stack(log_components, dim=-1)  # (B, K)
    return torch.logsumexp(log_stack, dim=-1) - math.log(nc)


def adapt_sigma_fs(omega, sigma_fs_default=(0.8, 1.3, 2.0)):
    """Adaptively widen Gaussian mixture for low-omega regimes.
    
    Root cause: wavefunction width scales as ~1/√ω, but fixed proposal doesn't.
    At ω=0.01, effective widths become ~25x narrower than oscillator trap.
    At ω=0.001, effective widths become ~80x narrower.
    
    Auto-widening invokes when:
    - omega is low (ω < 0.5)
    - sigma_fs is still at the DEFAULT value (user hasn't overridden)
    
    Adaptation tiers (from JOURNAL 2026-03-17 + extensions for ω<<0.01):
    - ω ≤ 0.15 → (0.4, 0.7, 1.0, 1.5, 2.5, 4.0)
    - ω ≤ 0.05 → (0.3, 0.5, 0.8, 1.2, 2.0, 3.5, 6.0)
    - ω ≤ 0.01 → (0.2, 0.4, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0)
    - ω ≤ 0.002 → (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
    - ω ≤ 0.0005 → (0.08, 0.15, 0.25, 0.4, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0)
    """
    # Only adapt if at DEFAULT (user didn't override)
    if sigma_fs_default != (0.8, 1.3, 2.0):
        return sigma_fs_default
    
    if omega >= 0.5:
        return sigma_fs_default
    elif omega > 0.15:
        return (0.4, 0.7, 1.0, 1.5, 2.5, 4.0)
    elif omega > 0.05:
        return (0.3, 0.5, 0.8, 1.2, 2.0, 3.5, 6.0)
    elif omega > 0.01:
        return (0.2, 0.4, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0)
    elif omega > 0.002:
        return (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
    else:  # omega <= 0.002
        return (0.08, 0.15, 0.25, 0.4, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0)


def sample_mixture(n, omega, sigma_fs=(0.8, 1.3, 2.0)):
    """Sample from Gaussian mixture, return (x, log_q).

    log_q is the log-density of the full mixture q(x) = (1/K) sum_k N(x; 0, s_k^2 I),
    NOT the component density. This is critical for correct importance weights.
    """
    nc = len(sigma_fs)
    xs = []
    for i, sf in enumerate(sigma_fs):
        ni = n // nc if i < nc - 1 else n - (n // nc) * (nc - 1)
        xi, _ = sample_gauss(ni, omega, sf)
        xs.append(xi)
    x_all = torch.cat(xs)
    perm = torch.randperm(x_all.shape[0], device=x_all.device)
    x_out = x_all[perm[:n]]
    # Evaluate the correct mixture log-density at all returned points
    lq_out = _eval_mixture_logq_local(x_out, omega, sigma_fs)
    return x_out, lq_out


def _min_pair_distance(x):
    """x: (B,N,d) -> (B,) minimum pair distance."""
    B, N, _ = x.shape
    dmat = torch.cdist(x, x, p=2.0)
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    dmat = dmat.masked_fill(eye, float("inf"))
    return dmat.amin(dim=(1, 2))


@torch.no_grad()
def importance_resample(
    psi_log_fn,
    n_keep,
    omega,
    n_cand_mult=8,
    sigma_fs=(0.8, 1.3, 2.0),
    min_pair_cutoff=0.0,
    weight_temp=1.0,
    logw_clip_q=0.0,
    langevin_steps=0,
    langevin_step_size=0.01,
    return_stats=False,
):
    """Multinomial resampling: sample from q, resample ∝ |Ψ|²/q.

    Returns points approximately distributed as |Ψ|², without MCMC.
    If langevin_steps > 0, proposal samples are first refined via
    overdamped Langevin dynamics toward |Ψ|² before resampling.
    """
    return nn_importance_resample(
        psi_log_fn,
        n_keep,
        N_ELEC,
        DIM,
        omega,
        device=DEVICE,
        dtype=DTYPE,
        n_cand_mult=n_cand_mult,
        sigma_fs=sigma_fs,
        min_pair_cutoff=min_pair_cutoff,
        weight_temp=weight_temp,
        logw_clip_q=logw_clip_q,
        langevin_steps=langevin_steps,
        langevin_step_size=langevin_step_size,
        return_stats=return_stats,
    )


# ═══════════════════════════════════════════════════════════════
#  Loss: FD-Laplacian collocation (per-sample gradients)
# ═══════════════════════════════════════════════════════════════

def colloc_fd_loss(psi_log_fn, x, omega, params, h=0.01,
                   huber_delta=0.0, lp_prev=None, prox_mu=0.0):
    """Collocation loss with finite-difference Laplacian.

    E_L is computed entirely via forward passes of logΨ at shifted points.
    The Laplacian is in the computational graph via forward-pass evaluations
    only — no higher-order autograd. This gives numerically stable first-order
    gradients for the collocation loss.

    Parameters
    ----------
    h : float
        Finite-difference step size (default 0.01). O(h²) bias.
    huber_delta : float
        If >0, use Huber loss on E_L residuals instead of variance (MSE).
        Caps per-sample gradient contribution for outlier E_L.
    lp_prev : Tensor or None
        If given (B,), detached logΨ values from before the gradient step.
        Used with prox_mu for proximal penalty.
    prox_mu : float
        Coefficient for proximal penalty: μ·mean((logΨ - logΨ_prev)²).

    Returns (L, E_mean, E_L_det, var_EL).
    """
    return nn_colloc_fd_loss(
        psi_log_fn,
        x,
        omega,
        params,
        h=h,
        huber_delta=huber_delta,
        lp_prev=lp_prev,
        prox_mu=prox_mu,
    )


# ═══════════════════════════════════════════════════════════════
#  Loss: hybrid REINFORCE + direct gradient
# ═══════════════════════════════════════════════════════════════

def rayleigh_hybrid_loss(psi_log_fn, x, omega, params, direct_weight=0.1,
                         clip_el=5.0, reward_qtrim=0.0):
    """Hybrid REINFORCE + weak-form direct gradient.

    Uses E_L (with Laplacian, FORWARD-ONLY) for low-variance REINFORCE
    reward signal, and ẽ = ½g²+V for direct kinetic energy gradient.

    The backward pass goes through logΨ (REINFORCE) and g² (direct) only.
    The Laplacian is NEVER in the backward graph — this avoids the
    conditioning catastrophe that plagued strong-form collocation.

    Parameters
    ----------
    direct_weight : float
        Coefficient β for the direct gradient term. 0 = pure REINFORCE.
    clip_el : float
        Clip |E_L - median| > clip_el * MAD to reduce outlier influence.

    Returns (L, EL_mean, EL_det, e_weak_det).
    """
    return nn_rayleigh_hybrid_loss(
        psi_log_fn,
        x,
        omega,
        params,
        direct_weight=direct_weight,
        clip_el=clip_el,
        reward_qtrim=reward_qtrim,
    )


# ═══════════════════════════════════════════════════════════════
#  Also keep strong-form E_L for monitoring (no backprop through it)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_EL_monitor(psi_log_fn, x, omega, params, n_max=256):
    """Compute E_L for monitoring only using exact Laplacian. No gradients."""
    from functions.Neural_Networks import _laplacian_logpsi_exact as lap_exact
    x_sub = x[:n_max].detach()
    with torch.enable_grad():
        x_sub = x_sub.requires_grad_(True)
        _, g2, lap = lap_exact(psi_log_fn, x_sub)
    B = x_sub.shape[0]
    T = -0.5 * (lap.view(B) + g2.view(B))
    V = 0.5 * omega ** 2 * (x_sub ** 2).sum(dim=(1, 2)) + compute_coulomb_interaction(
        x_sub, params=params
    ).view(B)
    EL = T + V
    ok = torch.isfinite(EL)
    if ok.sum() > 0:
        EL = EL[ok]
        return EL.mean().item(), EL.std().item(), EL.var().item()
    return float("nan"), float("nan"), float("nan")


@torch.no_grad()
def _geometry_invariants(x):
    """Permutation-invariant geometry descriptors for replay stratification."""
    B, N, _ = x.shape
    dmat = torch.cdist(x, x, p=2.0)
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    dmat = dmat.masked_fill(eye, float("inf"))
    min_pair = dmat.amin(dim=(1, 2))
    r_mean = x.norm(dim=-1).mean(dim=1)
    com_r = x.mean(dim=1).norm(dim=-1)
    return min_pair, r_mean, com_r


@torch.no_grad()
def _geometry_bucket_ids(x, n_bins=3):
    """2D quantile buckets on (min_pair, r_mean) invariants."""
    n_bins = int(max(2, n_bins))
    min_pair, r_mean, _ = _geometry_invariants(x)
    if x.shape[0] < n_bins * 2:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device), 1

    q = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)[1:-1]
    th_min = torch.quantile(min_pair, q)
    th_rm = torch.quantile(r_mean, q)
    b0 = torch.bucketize(min_pair, th_min)
    b1 = torch.bucketize(r_mean, th_rm)
    bucket_ids = b0 * n_bins + b1
    return bucket_ids, n_bins * n_bins


@torch.no_grad()
def _stratified_topk_indices(scores, bucket_ids, k):
    """Select approximately top-k by score, balanced across non-empty buckets."""
    dev = scores.device
    scores_cpu = scores.detach().cpu()
    buckets_cpu = bucket_ids.detach().cpu()
    n = int(scores_cpu.numel())
    k = int(max(1, min(int(k), n)))

    uniq = torch.unique(buckets_cpu)
    if uniq.numel() <= 1:
        return torch.topk(scores_cpu, k=k).indices.to(dev)

    picks = []
    for b in uniq.tolist():
        idx = torch.nonzero(buckets_cpu == b, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        kb = max(1, int(round(float(k) * float(idx.numel()) / float(n))))
        kb = min(kb, int(idx.numel()))
        sb = scores_cpu[idx]
        sel = idx[torch.topk(sb, k=kb).indices]
        picks.append(sel)

    if len(picks) == 0:
        return torch.topk(scores_cpu, k=k).indices.to(dev)

    out = torch.cat(picks)
    if out.numel() > k:
        out = out[torch.topk(scores_cpu[out], k=k).indices]
    elif out.numel() < k:
        mask = torch.ones(n, dtype=torch.bool)
        mask[out] = False
        rest = torch.nonzero(mask, as_tuple=False).view(-1)
        if rest.numel() > 0:
            need = min(k - int(out.numel()), int(rest.numel()))
            out = torch.cat([out, rest[torch.topk(scores_cpu[rest], k=need).indices]])
    return out.to(dev)


@torch.no_grad()
def _bf_coalescence_diag(backflow_net, x, spin, omega, q=0.10):
    """Return mean ||Δx_i-Δx_j||/r_ij over smallest-r quantile and its cutoff."""
    if backflow_net is None or x is None or x.numel() == 0:
        return float("nan"), float("nan")
    xs = x[: min(256, x.shape[0])]
    dx = backflow_net(xs, spin=spin)

    rij = torch.cdist(xs, xs, p=2.0) * (float(omega) ** 0.5)
    dd = dx.unsqueeze(2) - dx.unsqueeze(1)
    ddn = torch.sqrt((dd * dd).sum(dim=-1) + 1e-12)

    B, N, _ = xs.shape
    mask = ~torch.eye(N, device=xs.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    rvals = rij[mask]
    dvals = ddn[mask]
    if rvals.numel() == 0:
        return float("nan"), float("nan")

    rcut = torch.quantile(rvals, torch.tensor(float(q), device=rvals.device, dtype=rvals.dtype)).item()
    hard = rvals <= rcut
    if hard.sum().item() == 0:
        return float("nan"), float(rcut)
    ratio = (dvals[hard] / rvals[hard].clamp_min(1e-10)).mean().item()
    return float(ratio), float(rcut)


# ═══════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════

def train_weak_form(
    f_net, C_occ, params,
    *,
    backflow_net=None,
    npf_net=None,
    psi_log_fn_factory=None,
    n_epochs=800,
    lr=5e-4,
    lr_jas=5e-5,
    lr_min_frac=0.02,
    n_coll=2048,
    oversample=8,
    micro_batch=512,
    grad_clip=1.0,
    direct_weight=0.1,
    clip_el=0.0,
    reward_qtrim=0.0,
    loss_type="reinforce",
    fd_h=0.01,
    fd_huber_delta=0.0,
    prox_mu=0.0,
    min_ess=0,
    sigma_fs=(0.8, 1.3, 2.0),
    min_pair_cutoff=0.0,
    ess_floor_ratio=0.0,
    ess_oversample_max=0,
    ess_oversample_step=2,
    ess_resample_tries=1,
    resample_weight_temp=1.0,
    resample_logw_clip_q=0.0,
    langevin_steps=0,
    langevin_step_size=0.01,
    replay_frac=0.0,
    replay_top_frac=0.25,
    replay_stratified=False,
    replay_geo_bins=3,
    rollback_decay=1.0,
    rollback_err_pct=0.0,
    rollback_jump_sigma=0.0,
    bf_cusp_reg=0.0,
    bf_cusp_radius_aho=0.30,
    bf_diag_q=0.10,
    print_every=10,
    patience=300,
    vmc_every=50,
    vmc_n=10000,
    vmc_select_n=0,
    bf_warmup=0,
    n_elec=None,
    omega=None,
    e_ref=None,
    tag="weak_form",
    # ── Natural gradient / SR ──
    natural_grad=False,
    sr_mode="diagonal",
    fisher_damping=1e-3,
    fisher_damping_end=0.0,
    fisher_damping_anneal=0,
    fisher_ema=0.95,
    fisher_probes=4,
    fisher_subsample=256,
    fisher_max=1e6,
    nat_momentum=0.9,
    sr_max_param_change=0.1,
    sr_trust_region=1.0,
    sr_cg_iters=15,
    sr_center=True,
):
    omega = float(OMEGA if omega is None else omega)
    n_elec = int(N_ELEC if n_elec is None else n_elec)
    E_ref = default_dmc(n_elec, omega) if e_ref is None else float(e_ref)
    up = n_elec // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(n_elec - up, dtype=torch.long)]
    ).to(DEVICE)

    # Build the psi_log_fn
    if psi_log_fn_factory is not None:
        psi_log_fn = psi_log_fn_factory()
    elif npf_net is not None:
        from run_neural_pfaffian import psi_fn_npf
        def psi_log_fn(y):
            lp, _ = psi_fn_npf(npf_net, f_net, y, params, spin=spin, bf_net=backflow_net)
            return lp
    else:
        def psi_log_fn(y):
            lp, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params)
            return lp

    # When using FD-colloc with Pfaffian + proximal penalty: sample from full
    # |Ψ|² (correct objective) — the proximal penalty prevents mode collapse.
    # Without proximal: sample from frozen Jastrow (decoupled, prevents
    # feedback loop but may optimize wrong objective).
    if loss_type == "fd-colloc" and npf_net is not None and prox_mu == 0:
        def psi_log_sample_fn(y):
            lp, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params)
            return lp
        print("  FD-colloc: sampling from frozen Jastrow (no proximal)")
    else:
        psi_log_sample_fn = psi_log_fn
        if loss_type == "fd-colloc" and npf_net is not None and prox_mu > 0:
            print(f"  FD-colloc: sampling from full |Ψ|² + proximal μ={prox_mu}")

    # ── Collect trainable params ──
    trainable_groups = []
    all_trainable = []

    if npf_net is not None:
        pf_params = list(npf_net.parameters())
        trainable_groups.append({"params": pf_params, "lr": lr})
        all_trainable.extend(pf_params)
        n_pf = sum(p.numel() for p in pf_params)
        print(f"  Pfaffian params: {n_pf:,}")
    else:
        n_pf = 0

    if backflow_net is not None:
        bf_params = [p for p in backflow_net.parameters() if p.requires_grad]
        n_bf = sum(p.numel() for p in bf_params)
        if bf_warmup > 0:
            # Freeze BF during warmup (Jastrow-only first)
            pass
        trainable_groups.append({"params": bf_params, "lr": lr})
        all_trainable.extend(bf_params)
        print(f"  Backflow params: {n_bf:,}")
    else:
        bf_params = []
        n_bf = 0

    jas_params = [p for p in f_net.parameters() if p.requires_grad]
    n_jas = sum(p.numel() for p in jas_params)
    if n_jas > 0:
        trainable_groups.append({"params": jas_params, "lr": lr_jas})
        all_trainable.extend(jas_params)
    print(f"  Jastrow params: {n_jas:,} (lr={lr_jas})")
    print(f"  Total trainable: {sum(p.numel() for p in all_trainable):,}")

    fisher_precond = None
    if natural_grad:
        opt = torch.optim.SGD(trainable_groups, momentum=nat_momentum)
        if sr_mode == "woodbury":
            from sr_preconditioner import WoodburySR
            fisher_precond = WoodburySR(
                all_trainable,
                damping=fisher_damping,
                damping_end=fisher_damping_end,
                damping_anneal_epochs=fisher_damping_anneal,
                max_param_change=sr_max_param_change,
                trust_region=sr_trust_region,
                subsample=fisher_subsample,
                center_gradients=sr_center,
            )
        elif sr_mode == "cg":
            from sr_preconditioner import CGSR
            fisher_precond = CGSR(
                all_trainable,
                damping=fisher_damping,
                damping_end=fisher_damping_end,
                damping_anneal_epochs=fisher_damping_anneal,
                n_cg_iters=sr_cg_iters,
                max_param_change=sr_max_param_change,
                trust_region=sr_trust_region,
                subsample=fisher_subsample,
                center_gradients=sr_center,
            )
        else:  # diagonal
            from fisher_preconditioner import DiagonalFisherPreconditioner
            fisher_precond = DiagonalFisherPreconditioner(
                all_trainable,
                damping=fisher_damping,
                ema_decay=fisher_ema,
                n_probes=fisher_probes,
                subsample=fisher_subsample,
                max_fisher=fisher_max,
            )
    else:
        opt = torch.optim.Adam(trainable_groups)

    def lr_lambda(ep):
        lr_min_abs = lr * lr_min_frac
        return (lr_min_abs + 0.5 * (lr - lr_min_abs) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))) / lr

    sch = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda] * len(trainable_groups))

    opt_name = "SGD+Fisher" if natural_grad else "Adam"
    print(f"  Training: {n_epochs} ep, {n_coll} samples, LR={lr} opt={opt_name}")
    if loss_type == "fd-colloc":
        loss_str = "Huber" if fd_huber_delta > 0 else "Var"
        print(f"  FD-COLLOC loss: {loss_str}(E_L) via FD Laplacian"
              f"  h={fd_h}  huber_δ={fd_huber_delta}  prox_μ={prox_mu}")
    else:
        print(f"  HYBRID loss: REINFORCE(E_L, no-backprop-lap) + direct(½g², β={direct_weight})"
              f"  clip_el={clip_el}  reward_qtrim={reward_qtrim}")
    if replay_frac > 0:
        print(f"  Hard-sample replay: frac={replay_frac:.2f} top_frac={replay_top_frac:.2f}")
        if replay_stratified:
            print(f"    Stratified replay: geometry buckets={int(max(2, replay_geo_bins))}x{int(max(2, replay_geo_bins))}")
    if ess_floor_ratio > 0:
        print("  ESS adaptive sampler: "
              f"floor={ess_floor_ratio:.3f} oversample={oversample}->{max(oversample, int(ess_oversample_max))} "
              f"step={int(max(1, ess_oversample_step))} tries={int(max(1, ess_resample_tries))}")
    if resample_weight_temp != 1.0 or resample_logw_clip_q > 0:
        print("  Resample regularization: "
              f"temp={resample_weight_temp:.3f} logw_clip_q={resample_logw_clip_q:.4f}")
    if langevin_steps > 0:
        print(f"  Langevin proposal refinement: {langevin_steps} steps, "
              f"ε={langevin_step_size:.4f}")
    if rollback_decay < 1.0:
        print("  Instability rollback: "
              f"decay={rollback_decay:.3f} err_pct>{rollback_err_pct:.2f} jump_sigma={rollback_jump_sigma:.1f}")
    if bf_cusp_reg > 0 and backflow_net is not None:
        print(f"  BF cusp regularizer: λ={bf_cusp_reg:.3e} radius={bf_cusp_radius_aho:.2f} a_ho")
    if natural_grad:
        print(f"  Natural gradient ({sr_mode}): damping={fisher_damping:.1e} "
              f"subsample={fisher_subsample} momentum={nat_momentum}")
        if sr_mode == "diagonal":
            print(f"    Diagonal: ema={fisher_ema:.3f} probes={fisher_probes}")
        elif sr_mode == "woodbury":
            print(f"    Woodbury: max_Δθ={sr_max_param_change} trust_r={sr_trust_region}")
        elif sr_mode == "cg":
            print(f"    CG: iters={sr_cg_iters} max_Δθ={sr_max_param_change} trust_r={sr_trust_region}")
        if fisher_damping_anneal > 0:
            print(f"    Damping anneal: {fisher_damping:.1e} → {fisher_damping_end:.1e} over {fisher_damping_anneal} epochs")
    sys.stdout.flush()

    t0 = time.time()
    hist = []
    best_R = float("inf")
    best_vmc_err = float("inf")
    best_state = best_vmc_state = {}
    best_vmc_E = None
    no_imp = 0
    n_ess_reject = 0
    n_rollbacks = 0
    replay_X = None
    prev_stable_E = None
    curr_oversample = int(max(1, oversample))
    max_oversample = int(max(curr_oversample, ess_oversample_max if ess_oversample_max > 0 else curr_oversample))
    ess_step = int(max(1, ess_oversample_step))
    n_resample_tries = int(max(1, ess_resample_tries))
    last_rs_stats = {}

    # Save initial state for ESS-gated rollback
    def _save_state():
        st = {"jas_state": {k: v.clone() for k, v in f_net.state_dict().items()}}
        if backflow_net is not None:
            st["bf_state"] = {k: v.clone() for k, v in backflow_net.state_dict().items()}
        if npf_net is not None:
            st["pf_state"] = {k: v.clone() for k, v in npf_net.state_dict().items()}
        if fisher_precond is not None:
            st["fisher_state"] = fisher_precond.state_dict()
        return st

    def _restore_state(st):
        f_net.load_state_dict(st["jas_state"])
        if backflow_net is not None and "bf_state" in st:
            backflow_net.load_state_dict(st["bf_state"])
        if npf_net is not None and "pf_state" in st:
            npf_net.load_state_dict(st["pf_state"])
        if fisher_precond is not None and "fisher_state" in st:
            fisher_precond.load_state_dict(st["fisher_state"])

    last_good_state = _save_state()

    for ep in range(n_epochs):
        ept0 = time.time()

        # ── Resample from approximate |Ψ|² ──
        for net in [f_net, backflow_net, npf_net]:
            if net is not None:
                net.eval()

        target_ess = ess_floor_ratio * float(n_coll) if ess_floor_ratio > 0 else 0.0
        used_oversample = curr_oversample
        X = None
        ess = 0.0
        # Adapt sigma_fs for low-omega regimes (fixes ESS collapse at ω<<1)
        sigma_fs_adapted = adapt_sigma_fs(omega, sigma_fs)
        for _ in range(n_resample_tries):
            rs = importance_resample(
                psi_log_sample_fn,
                n_coll,
                omega,
                n_cand_mult=used_oversample,
                sigma_fs=sigma_fs_adapted,
                min_pair_cutoff=min_pair_cutoff,
                weight_temp=resample_weight_temp,
                logw_clip_q=resample_logw_clip_q,
                langevin_steps=langevin_steps,
                langevin_step_size=langevin_step_size,
                return_stats=True,
            )
            X, ess, last_rs_stats = rs
            if target_ess <= 0 or ess >= target_ess:
                break
            if used_oversample >= max_oversample:
                break
            used_oversample = min(max_oversample, used_oversample + ess_step)
        curr_oversample = used_oversample

        if replay_frac > 0 and replay_X is not None and replay_X.numel() > 0:
            n_rep = int(min(n_coll - 1, round(replay_frac * n_coll)))
            if n_rep > 0:
                n_new = n_coll - n_rep
                if replay_X.shape[0] >= n_rep:
                    idx = torch.randperm(replay_X.shape[0], device=replay_X.device)[:n_rep]
                    rep = replay_X[idx]
                else:
                    ridx = torch.randint(0, replay_X.shape[0], (n_rep,), device=replay_X.device)
                    rep = replay_X[ridx]
                X = torch.cat([X[:n_new], rep], dim=0)

        # ── ESS gate: skip step if ESS too low (revert to last good state) ──
        if min_ess > 0 and ess < min_ess:
            _restore_state(last_good_state)
            n_ess_reject += 1
            if ep % print_every == 0 or ep == 0:
                print(f"  [{ep:4d}] ESS={ess:.0f} < {min_ess} → SKIP (reverted, "
                      f"{n_ess_reject} rejects total)")
                sys.stdout.flush()
            continue

        fisher_stats = {}

        # ── Pre-compute logΨ for proximal penalty (before gradient step) ──
        lp_prev_all = None
        if loss_type == "fd-colloc" and prox_mu > 0:
            with torch.no_grad():
                lp_prev_all = psi_log_fn(X)  # (n_coll,)

        # ── Training step: hybrid loss ──
        for net in [f_net, backflow_net, npf_net]:
            if net is not None:
                net.train()

        opt.zero_grad(set_to_none=True)
        nmb = max(1, math.ceil(n_coll / micro_batch))
        ep_loss = 0.0
        ep_cusp_pen = 0.0
        all_EL = []
        all_e_weak = []

        for i in range(0, n_coll, micro_batch):
            xb = X[i:i + micro_batch]

            if loss_type == "fd-colloc":
                lp_prev_mb = None
                if lp_prev_all is not None:
                    lp_prev_mb = lp_prev_all[i:i + micro_batch]
                L, R_batch, EL_det, var_batch = colloc_fd_loss(
                    psi_log_fn, xb, omega, params, h=fd_h,
                    huber_delta=fd_huber_delta,
                    lp_prev=lp_prev_mb, prox_mu=prox_mu,
                )
                ew_det = EL_det  # no separate weak-form energy for FD
            else:
                L, R_batch, EL_det, ew_det = rayleigh_hybrid_loss(
                    psi_log_fn, xb, omega, params,
                    direct_weight=direct_weight,
                    clip_el=clip_el,
                    reward_qtrim=reward_qtrim,
                )

            if bf_cusp_reg > 0 and backflow_net is not None:
                dx_b = backflow_net(xb, spin=spin)
                rij = xb.unsqueeze(2) - xb.unsqueeze(1)
                r2 = (rij * rij).sum(dim=-1, keepdim=True)
                ddx = dx_b.unsqueeze(2) - dx_b.unsqueeze(1)
                ddx2 = (ddx * ddx).sum(dim=-1, keepdim=True)
                rc = float(bf_cusp_radius_aho) / math.sqrt(float(omega))
                w_close = torch.exp(-r2 / (rc * rc + 1e-12))
                eye = torch.eye(xb.shape[1], device=xb.device, dtype=xb.dtype).view(1, xb.shape[1], xb.shape[1], 1)
                w_close = w_close * (1.0 - eye)
                cusp_pen = ((ddx2 / (r2 + 1e-12)) * w_close).sum() / (w_close.sum() + 1e-12)
                L = L + bf_cusp_reg * cusp_pen
                ep_cusp_pen += float(cusp_pen.detach().item()) / nmb

            (L / nmb).backward()
            ep_loss += L.item() / nmb
            all_EL.append(EL_det)
            all_e_weak.append(ew_det)

        # ── Natural gradient: update Fisher estimate and precondition ──
        if fisher_precond is not None:
            fisher_stats = fisher_precond.update(psi_log_fn, X, all_trainable)
            fisher_precond.precondition(all_trainable)

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_trainable, grad_clip)
        opt.step()
        sch.step()

        # Epoch statistics from the hybrid loss
        EL_all = torch.cat(all_EL)
        Em = EL_all.mean().item()  # E_L mean (accurate energy estimate)
        Ev = EL_all.var().item()   # E_L variance
        Es = EL_all.std().item()

        unstable = False
        unstable_msg = ""
        if not (math.isfinite(Em) and math.isfinite(Ev) and math.isfinite(Es)):
            unstable = True
            unstable_msg = "non-finite epoch stats"

        err_now = safe_percent_err(Em, E_ref)
        if (not unstable and rollback_err_pct > 0 and math.isfinite(err_now)
                and abs(err_now) > rollback_err_pct):
            unstable = True
            unstable_msg = f"|err|={abs(err_now):.2f}% > {rollback_err_pct:.2f}%"

        if (not unstable and rollback_jump_sigma > 0 and prev_stable_E is not None
                and math.isfinite(prev_stable_E)):
            jump = abs(Em - prev_stable_E)
            jump_ref = max(1e-6, rollback_jump_sigma * max(Es, 1e-3))
            if jump > jump_ref:
                unstable = True
                unstable_msg = f"jump={jump:.3e} > {jump_ref:.3e}"

        if unstable:
            _restore_state(last_good_state)
            n_rollbacks += 1
            if rollback_decay < 1.0:
                for gi, g in enumerate(opt.param_groups):
                    g["lr"] = max(g["lr"] * rollback_decay, 1e-7)
                    if gi < len(sch.base_lrs):
                        sch.base_lrs[gi] = max(sch.base_lrs[gi] * rollback_decay, 1e-7)
            if ep % print_every == 0 or ep == 0:
                print(f"  [{ep:4d}] rollback: {unstable_msg} (count={n_rollbacks}, lr_decay={rollback_decay:.3f})")
                sys.stdout.flush()
            continue
        prev_stable_E = Em

        if replay_frac > 0 and EL_all.numel() >= 16:
            n_valid = int(min(X.shape[0], EL_all.shape[0]))
            if n_valid >= 16:
                X_replay = X[:n_valid]
                EL_replay = EL_all[:n_valid]
                resid = (EL_replay - EL_replay.mean()).abs()
                k = int(max(8, min(n_valid, round(replay_top_frac * n_valid))))
                if replay_stratified:
                    bucket_ids, _ = _geometry_bucket_ids(X_replay, n_bins=replay_geo_bins)
                    top_idx = _stratified_topk_indices(resid, bucket_ids, k)
                else:
                    top_idx = torch.topk(resid, k=k).indices
                replay_X = X_replay[top_idx].detach().clone()

        replay_min_pair_mean = float("nan")
        replay_r_mean = float("nan")
        replay_bucket_entropy = float("nan")
        bf_coal_ratio_q = float("nan")
        bf_coal_rcut_q = float("nan")
        if replay_X is not None and replay_X.numel() > 0:
            rp_min_pair, rp_r_mean, _ = _geometry_invariants(replay_X)
            replay_min_pair_mean = float(rp_min_pair.mean().item())
            replay_r_mean = float(rp_r_mean.mean().item())
            if replay_stratified:
                replay_bucket_ids, n_bucket_total = _geometry_bucket_ids(replay_X, n_bins=replay_geo_bins)
                counts = torch.bincount(replay_bucket_ids.detach().cpu(), minlength=n_bucket_total).to(torch.float64)
                probs = counts / counts.sum().clamp_min(1.0)
                nz = probs > 0
                ent = -(probs[nz] * torch.log(probs[nz])).sum().item()
                replay_bucket_entropy = float(ent / math.log(max(2, n_bucket_total)))

        if backflow_net is not None:
            bf_coal_ratio_q, bf_coal_rcut_q = _bf_coalescence_diag(
                backflow_net, X, spin, omega, q=bf_diag_q
            )

        epdt = time.time() - ept0
        entry = dict(
            ep=ep,
            E=Em,
            var_EL=Ev,
            ess=ess,
            ess_target=target_ess,
            dt=epdt,
            loss=ep_loss,
            oversample=curr_oversample,
            ess_raw=finite_or(last_rs_stats.get("ess_raw")),
            ess_eff=finite_or(last_rs_stats.get("ess_eff")),
            rs_top1_mass=finite_or(last_rs_stats.get("top1_mass")),
            rs_top10_mass=finite_or(last_rs_stats.get("top10_mass")),
            rs_logw_clip_thr=finite_or(last_rs_stats.get("logw_clip_thr")),
            rs_temp=finite_or(last_rs_stats.get("weight_temp")),
            cusp_pen=ep_cusp_pen,
            replay_min_pair_mean=replay_min_pair_mean,
            replay_r_mean=replay_r_mean,
            replay_bucket_entropy=replay_bucket_entropy,
            bf_coal_ratio_q=bf_coal_ratio_q,
            bf_coal_rcut_q=bf_coal_rcut_q,
        )
        if fisher_precond is not None and fisher_precond._n_updates > 0:
            entry.update(
                fisher_mean=fisher_stats.get("fisher_mean", float("nan")),
                fisher_max=fisher_stats.get("fisher_max", float("nan")),
                fisher_median=fisher_stats.get("fisher_median", float("nan")),
            )

        hist.append(entry)

        # ── Checkpointing ──
        if math.isfinite(Em) and Em < best_R * 0.9999:
            best_R = Em
            best_state = _save_state()
            no_imp = 0
        else:
            no_imp += 1

        # Update last-good state for ESS rollback
        if min_ess > 0:
            last_good_state = _save_state()
        if target_ess > 0:
            if ess < target_ess and curr_oversample < max_oversample:
                curr_oversample = min(max_oversample, curr_oversample + ess_step)
            elif ess > 1.6 * target_ess and curr_oversample > oversample:
                curr_oversample = max(oversample, curr_oversample - 1)

        # ── VMC probe ──
        if vmc_every > 0 and ep > 0 and ep % vmc_every == 0:
            try:
                for net in [f_net, backflow_net, npf_net]:
                    if net is not None:
                        net.eval()
                if npf_net is not None:
                    from run_neural_pfaffian import psi_fn_npf as _psi_fn_npf
                    def _psi_wrap(f_, x_, C_, *, backflow_net=None, orbital_bf_net=None, spin=None, params=None):
                        return _psi_fn_npf(npf_net, f_net, x_, params, spin=spin, bf_net=backflow_net)
                    C_dummy = torch.eye(9, 3, device=DEVICE, dtype=DTYPE)
                    vp = evaluate_energy_vmc(
                        f_net, C_dummy, psi_fn=_psi_wrap,
                        compute_coulomb_interaction=compute_coulomb_interaction,
                        params=params, n_samples=vmc_n, batch_size=512,
                        sampler_steps=60, sampler_step_sigma=0.12, lap_mode="exact",
                        persistent=True, sampler_burn_in=400, sampler_thin=3, progress=False,
                    )
                else:
                    vp = evaluate_energy_vmc(
                        f_net, C_occ, psi_fn=psi_fn,
                        compute_coulomb_interaction=compute_coulomb_interaction,
                        backflow_net=backflow_net, params=params,
                        n_samples=vmc_n, batch_size=512,
                        sampler_steps=60, sampler_step_sigma=0.12, lap_mode="exact",
                        persistent=True, sampler_burn_in=400, sampler_thin=3, progress=False,
                    )
                vE = float(vp["E_mean"])
                vErr = abs(vE - E_ref) / abs(E_ref) if math.isfinite(E_ref) and E_ref != 0 else float("nan")
                entry.update(vmc_E=vE, vmc_err=vErr)
                vSelE = None
                vSelErr = None
                if vmc_select_n > 0:
                    vp_sel = evaluate_energy_vmc(
                        f_net,
                        C_dummy if npf_net is not None else C_occ,
                        psi_fn=_psi_wrap if npf_net is not None else psi_fn,
                        compute_coulomb_interaction=compute_coulomb_interaction,
                        backflow_net=None if npf_net is not None else backflow_net,
                        params=params,
                        n_samples=vmc_select_n,
                        batch_size=512,
                        sampler_steps=60,
                        sampler_step_sigma=0.10,
                        lap_mode="exact",
                        persistent=True,
                        sampler_burn_in=250,
                        sampler_thin=2,
                        progress=False,
                    )
                    vSelE = float(vp_sel["E_mean"])
                    vSelErr = abs(vSelE - E_ref) / abs(E_ref) if math.isfinite(E_ref) and E_ref != 0 else float("nan")
                    entry.update(vmc_sel_E=vSelE, vmc_sel_err=vSelErr)
                # When E_ref is NaN (no DMC reference), fall back to minimizing vE directly
                if math.isfinite(E_ref) and E_ref != 0:
                    metric = vSelErr if vSelErr is not None and math.isfinite(vSelErr) else vErr
                    metric_E = vSelE if vSelE is not None and math.isfinite(vSelE) else vE
                    if math.isfinite(metric) and metric < best_vmc_err:
                        best_vmc_err = metric
                        best_vmc_E = metric_E
                        best_vmc_state = _save_state()
                else:
                    metric_E = vSelE if vSelE is not None and math.isfinite(vSelE) else vE
                    if best_vmc_E is None or metric_E < best_vmc_E:
                        best_vmc_E = metric_E
                        best_vmc_state = _save_state()
            except Exception as e:
                print(f"  VMC probe failed: {e}")

        # ── Early stop ──
        if patience > 0 and no_imp >= patience and ep > 60:
            print(f"  Early stop ep {ep}")
            sys.stdout.flush()
            break

        # ── Logging ──
        if ep % print_every == 0:
            err = safe_percent_err(Em, E_ref) if math.isfinite(Em) else float("nan")
            vs = ""
            if "vmc_E" in entry:
                vs = f"  vmc={entry['vmc_E']:.4f}({entry['vmc_err'] * 100:.2f}%)"
                if "vmc_sel_E" in entry:
                    vs += f" sel={entry['vmc_sel_E']:.4f}({entry['vmc_sel_err'] * 100:.2f}%)"
            eta = epdt * (n_epochs - ep - 1) / 60
            print(
                f"  [{ep:4d}] E={Em:.4f}±{Es:.3f} var={Ev:.2e} ESS={ess:.0f} "
                f"loss={ep_loss:.3e} {epdt:.1f}s err={err:+.2f}% "
                f"eta={eta:.0f}m{vs}"
            )
            sys.stdout.flush()

    # ── Restore best ──
    if best_vmc_state:
        f_net.load_state_dict(best_vmc_state["jas_state"])
        if backflow_net is not None and "bf_state" in best_vmc_state:
            backflow_net.load_state_dict(best_vmc_state["bf_state"])
        if npf_net is not None and "pf_state" in best_vmc_state:
            npf_net.load_state_dict(best_vmc_state["pf_state"])
        _err_str = f"{best_vmc_err * 100:.3f}%" if math.isfinite(best_vmc_err) else "nan%"
        print(f"  Restored VMC-best E={best_vmc_E:.5f} err={_err_str}")
    elif best_state:
        f_net.load_state_dict(best_state["jas_state"])
        if backflow_net is not None and "bf_state" in best_state:
            backflow_net.load_state_dict(best_state["bf_state"])
        if npf_net is not None and "pf_state" in best_state:
            npf_net.load_state_dict(best_state["pf_state"])
        print(f"  Restored var-best R={best_R:.5f}")

    tot = time.time() - t0
    print(f"  Done {tot:.0f}s ({tot / 60:.1f}min)")
    sys.stdout.flush()
    return f_net, backflow_net, npf_net, hist


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Weak-form collocation training")
    # Mode
    ap.add_argument("--mode", choices=["bf", "pfaffian", "jastrow"], default="bf",
                    help="bf: BF+Jastrow, pfaffian: NeuralPfaffian+Jastrow, jastrow: Jastrow-only")
    # Training
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--n-coll", type=int, default=4096)
    ap.add_argument("--oversample", type=int, default=8,
                    help="Candidate multiplier for importance resampling")
    ap.add_argument("--micro-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-jas", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--direct-weight", type=float, default=0.1,
                    help="β for direct gradient term (0=pure REINFORCE, 1=full dual)")
    ap.add_argument("--clip-el", type=float, default=0.0,
                    help="E_L clipping in MAD units (0=off, 5=moderate, 3=aggressive)")
    ap.add_argument("--reward-qtrim", type=float, default=0.0,
                    help="Trim REINFORCE reward tails by quantiles q..1-q (0=off)")
    ap.add_argument("--loss-type", choices=["reinforce", "fd-colloc"], default="reinforce",
                    help="reinforce: REINFORCE with E_L reward, fd-colloc: FD-Laplacian collocation (per-sample grad)")
    ap.add_argument("--fd-h", type=float, default=0.01,
                    help="Finite-difference step size for fd-colloc loss")
    ap.add_argument("--fd-huber-delta", type=float, default=0.0,
                    help="Huber delta for fd-colloc (0=variance, >0=Huber)")
    ap.add_argument("--prox-mu", type=float, default=0.0,
                    help="Proximal penalty mu for fd-colloc (0=off)")
    ap.add_argument("--min-ess", type=int, default=0,
                    help="Min ESS to accept a training step (0=off, 20=conservative)")
    ap.add_argument("--sigma-fs", type=str, default="0.8,1.3,2.0",
                    help="Gaussian-mixture sigmas for proposal, comma-separated")
    ap.add_argument("--min-pair-cutoff", type=float, default=0.0,
                    help="Drop candidate samples with min pair distance below cutoff (0=off)")
    ap.add_argument("--ess-floor-ratio", type=float, default=0.0,
                    help="Adaptive resample target as fraction of n_coll ESS (0=off)")
    ap.add_argument("--ess-oversample-max", type=int, default=0,
                    help="Max oversample multiplier for adaptive ESS resampling (0=use --oversample)")
    ap.add_argument("--ess-oversample-step", type=int, default=2,
                    help="Oversample increment when ESS falls below floor")
    ap.add_argument("--ess-resample-tries", type=int, default=1,
                    help="Resampling attempts per epoch when ESS floor is active")
    ap.add_argument("--resample-weight-temp", type=float, default=1.0,
                    help="Importance-weight tempering exponent (alpha); <1 flattens spikes")
    ap.add_argument("--resample-logw-clip-q", type=float, default=0.0,
                    help="Upper quantile for clipping log-weights before resampling (0=off)")
    ap.add_argument("--langevin-steps", type=int, default=0,
                    help="Langevin refinement steps on proposal samples (0=off)")
    ap.add_argument("--langevin-step-size", type=float, default=0.01,
                    help="Langevin step size (auto-scaled by 1/omega)")
    ap.add_argument("--replay-frac", type=float, default=0.0,
                    help="Fraction of collocation batch replayed from prior hard samples")
    ap.add_argument("--replay-top-frac", type=float, default=0.25,
                    help="Top residual fraction retained in replay buffer")
    ap.add_argument("--replay-stratified", action="store_true",
                    help="Stratify replay by invariant geometry buckets")
    ap.add_argument("--replay-geo-bins", type=int, default=3,
                    help="Quantile bins per invariant axis for stratified replay")
    ap.add_argument("--rollback-decay", type=float, default=1.0,
                    help="LR decay factor applied on instability rollback (1=off)")
    ap.add_argument("--rollback-err-pct", type=float, default=0.0,
                    help="Rollback if |epoch err| exceeds this percent threshold (0=off)")
    ap.add_argument("--rollback-jump-sigma", type=float, default=0.0,
                    help="Rollback if epoch E jump exceeds jump_sigma * std(E_L) (0=off)")
    ap.add_argument("--bf-cusp-reg", type=float, default=0.0,
                    help="Penalty weight to keep BF relative displacement smooth at short range")
    ap.add_argument("--bf-cusp-radius-aho", type=float, default=0.30,
                    help="Short-range radius in oscillator units for BF cusp regularizer")
    ap.add_argument("--bf-hard-cusp-gate", action="store_true",
                    help="Apply structural gate to suppress BF displacement near coalescence")
    ap.add_argument("--bf-cusp-gate-radius-aho", type=float, default=0.30,
                    help="Radius (a_ho units) for hard BF cusp gate")
    ap.add_argument("--bf-cusp-gate-power", type=float, default=2.0,
                    help="Exponent for hard BF cusp gate; >=2 gives smooth near-zero suppression")
    ap.add_argument("--bf-diag-q", type=float, default=0.10,
                    help="Quantile for BF coalescence ratio diagnostics")
    # Pfaffian args
    ap.add_argument("--K-det", type=int, default=1)
    ap.add_argument("--K-emb", type=int, default=32)
    # BF args
    ap.add_argument("--use-backflow", action="store_true", help="Load BF (bf mode or pfaffian+bf)")
    ap.add_argument("--train-jas", action="store_true",
                    help="In pfaffian mode, keep Jastrow trainable (default: frozen)")
    ap.add_argument("--train-bf", action="store_true",
                    help="In pfaffian mode with --use-backflow, keep BF trainable (default: frozen)")
    # Resume
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from (loads pf_state, bf_state, jas_state)")
    # Eval
    ap.add_argument("--vmc-every", type=int, default=50)
    ap.add_argument("--vmc-n", type=int, default=10000)
    ap.add_argument("--vmc-select-n", type=int, default=0,
                    help="Optional additional VMC samples for checkpoint selection at probe epochs (0=off)")
    ap.add_argument("--n-eval", type=int, default=30000)
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="weak_form")
    ap.add_argument("--n-elec", type=int, default=6,
                    help="Number of electrons")
    ap.add_argument("--omega", type=float, default=1.0,
                    help="Trap frequency")
    ap.add_argument("--e-dmc", type=float, default=float("nan"),
                    help="Reference DMC energy for error reporting; NaN disables err-based model selection")
    # Natural gradient
    ap.add_argument("--natural-grad", action="store_true",
                    help="Use diagonal Fisher preconditioning (natural gradient) instead of Adam")
    ap.add_argument("--sr-mode", choices=["diagonal", "woodbury", "cg"], default="diagonal",
                    help="SR mode: diagonal=Hutchinson diagonal, woodbury=exact Woodbury, cg=CG solve")
    ap.add_argument("--fisher-damping", type=float, default=1e-3,
                    help="Tikhonov damping for Fisher/SR (1e-4 to 1e-2)")
    ap.add_argument("--fisher-damping-end", type=float, default=0.0,
                    help="Final damping after annealing (0=no anneal, use for woodbury/cg)")
    ap.add_argument("--fisher-damping-anneal", type=int, default=0,
                    help="Epochs over which to anneal damping (0=constant)")
    ap.add_argument("--fisher-ema", type=float, default=0.95,
                    help="EMA decay for running Fisher estimate (diagonal mode only)")
    ap.add_argument("--fisher-probes", type=int, default=4,
                    help="Hutchinson probes per Fisher update (diagonal mode only)")
    ap.add_argument("--fisher-subsample", type=int, default=1024,
                    help="Collocation points subsampled for Fisher/SR estimation (higher = better rank)")
    ap.add_argument("--fisher-max", type=float, default=1e6,
                    help="Upper clamp for Fisher diagonal entries (diagonal mode only)")
    ap.add_argument("--nat-momentum", type=float, default=0.9,
                    help="SGD momentum when using natural gradient")
    ap.add_argument("--sr-max-param-change", type=float, default=0.1,
                    help="Max per-parameter change per step (woodbury/cg mode)")
    ap.add_argument("--sr-trust-region", type=float, default=1.0,
                    help="Trust region radius for SR update norm (woodbury/cg mode)")
    ap.add_argument("--sr-cg-iters", type=int, default=100,
                    help="CG iterations for cg mode (100 is standard for VMC)")
    ap.add_argument("--sr-center", action="store_true", default=True,
                    help="Center per-sample gradients (subtract mean O) in SR")
    # Architecture
    ap.add_argument("--jas-arch", choices=["vcycle", "ctnn"], default="vcycle",
                    help="Jastrow architecture: vcycle=CTNNJastrowVCycle, ctnn=CTNNJastrow")
    ap.add_argument("--jas-hidden", type=int, default=0,
                    help="Jastrow node/edge hidden dim (0=arch default: vcycle=24, ctnn=64)")
    ap.add_argument("--jas-mp-steps", type=int, default=0,
                    help="Jastrow message-passing steps (0=arch default: vcycle n_down/n_up=1, ctnn=2)")
    ap.add_argument("--jas-readout-hidden", type=int, default=64,
                    help="Jastrow readout MLP hidden dim")
    ap.add_argument("--bf-hidden", type=int, default=128,
                    help="Backflow node hidden dim")
    ap.add_argument("--bf-msg-hidden", type=int, default=0,
                    help="Backflow message hidden dim (0=same as --bf-hidden)")
    ap.add_argument("--bf-layers", type=int, default=3,
                    help="Backflow update MLP layers")
    ap.add_argument("--unified", action="store_true",
                    help="Use UnifiedCTNN (shared backbone for BF+Jastrow) instead of separate networks")
    ap.add_argument("--unified-mp-steps", type=int, default=2,
                    help="UnifiedCTNN message-passing iterations")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Do not load default pretrained initialization checkpoints")
    ap.add_argument("--init-jas", type=str, default=None,
                    help="Optional checkpoint path to initialize Jastrow (expects jas_state or state)")
    ap.add_argument("--init-bf", type=str, default=None,
                    help="Optional checkpoint path to initialize backflow (expects bf_state)")
    a = ap.parse_args()

    # NOTE: Low-omega SR instability was caused by incorrect mixture importance
    # weights (using component density instead of mixture density). Now fixed.
    # SR is allowed at all omega values.

    global N_ELEC, OMEGA, E_DMC
    N_ELEC = int(a.n_elec)
    OMEGA = float(a.omega)
    E_DMC = default_dmc(N_ELEC, OMEGA) if not math.isfinite(a.e_dmc) else float(a.e_dmc)

    sigma_fs = tuple(float(s) for s in a.sigma_fs.split(",") if s.strip())
    if len(sigma_fs) == 0:
        raise ValueError("--sigma-fs must contain at least one value")
    # Note: sigma_fs values are in oscillator-length units.
    # sample_gauss already scales by 1/sqrt(omega), so defaults (0.8,1.3,2.0)
    # are appropriate for all omega — no auto-widening needed.

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C_occ, params, nx, ny = setup(n_elec=N_ELEC, omega=OMEGA, e_dmc=E_DMC, seed=a.seed)
    n_basis = nx * ny
    n_occ = N_ELEC // 2

    def build_jastrow_model():
        if a.jas_arch == "ctnn":
            jh = a.jas_hidden if a.jas_hidden > 0 else 64
            jmp = a.jas_mp_steps if a.jas_mp_steps > 0 else 2
            return CTNNJastrow(
                n_particles=N_ELEC, d=DIM, omega=OMEGA,
                node_hidden=jh, edge_hidden=jh,
                n_mp_steps=jmp, msg_layers=2, node_layers=2,
                readout_hidden=a.jas_readout_hidden, readout_layers=2, act="silu",
            ).to(DEVICE).to(DTYPE)
        else:  # vcycle (default)
            jh = a.jas_hidden if a.jas_hidden > 0 else 24
            jmp = a.jas_mp_steps if a.jas_mp_steps > 0 else 1
            return CTNNJastrowVCycle(
                n_particles=N_ELEC, d=DIM, omega=OMEGA,
                node_hidden=jh, edge_hidden=jh, bottleneck_hidden=max(jh // 2, 8),
                n_down=jmp, n_up=jmp, msg_layers=1, node_layers=1,
                readout_hidden=a.jas_readout_hidden, readout_layers=2, act="silu",
            ).to(DEVICE).to(DTYPE)

    bf_msg_h = a.bf_msg_hidden if a.bf_msg_hidden > 0 else a.bf_hidden

    def build_default_backflow():
        return CTNNBackflowNet(
            d=DIM,
            msg_hidden=bf_msg_h,
            msg_layers=2,
            hidden=a.bf_hidden,
            layers=a.bf_layers,
            act="silu",
            aggregation="sum",
            use_spin=True,
            same_spin_only=False,
            out_bound="tanh",
            bf_scale_init=0.05,
            zero_init_last=True,
            omega=OMEGA,
            hard_cusp_gate=a.bf_hard_cusp_gate,
            cusp_gate_radius_aho=a.bf_cusp_gate_radius_aho,
            cusp_gate_power=a.bf_cusp_gate_power,
        ).to(DEVICE).to(DTYPE)

    def build_unified_model():
        uh = a.bf_hidden
        return UnifiedCTNN(
            n_particles=N_ELEC, d=DIM, omega=OMEGA,
            node_hidden=uh, edge_hidden=uh,
            msg_layers=2, node_layers=3,
            n_mp_steps=a.unified_mp_steps,
            act="silu", aggregation="sum",
            use_spin=True, same_spin_only=False,
            out_bound="tanh", bf_scale_init=0.05, zero_init_last=True,
            jastrow_hidden=a.jas_readout_hidden, jastrow_layers=2,
        ).to(DEVICE).to(DTYPE)

    def try_load_state(module, state, what):
        try:
            module.load_state_dict(state)
            return True
        except RuntimeError as e:
            print(f"    WARNING: could not load {what} (shape mismatch): {e}")
            return False

    def build_backflow_from_config(bfc):
        return CTNNBackflowNet(
            d=bfc["d"],
            msg_hidden=bfc["msg_hidden"],
            msg_layers=bfc["msg_layers"],
            hidden=bfc["hidden"],
            layers=bfc["layers"],
            act=bfc["act"],
            aggregation=bfc["aggregation"],
            use_spin=bfc["use_spin"],
            same_spin_only=bfc["same_spin_only"],
            out_bound=bfc["out_bound"],
            bf_scale_init=bfc["bf_scale_init"],
            zero_init_last=bfc["zero_init_last"],
            omega=OMEGA,
            hard_cusp_gate=bfc.get("hard_cusp_gate", False),
            cusp_gate_radius_aho=bfc.get("cusp_gate_radius_aho", 0.30),
            cusp_gate_power=bfc.get("cusp_gate_power", 2.0),
        ).to(DEVICE).to(DTYPE)

    def maybe_load_bf_from_ckpt(module, ckpt_obj, what, *, required=False):
        if "bf_state" not in ckpt_obj:
            msg = f"{what} has no bf_state"
            if required:
                raise RuntimeError(msg)
            print(f"    WARNING: {msg}")
            return module, False

        # Prefer architecture-aware load first when config is present (or can be
        # inferred from canonical pretrained BF), to avoid noisy mismatch warnings.
        if "bf_config" in ckpt_obj:
            try:
                rebuilt = build_backflow_from_config(ckpt_obj["bf_config"])
                loaded_cfg = try_load_state(rebuilt, ckpt_obj["bf_state"], f"{what}(cfg-first)")
                if loaded_cfg:
                    return rebuilt, True
            except Exception as e:
                print(f"    WARNING: could not use bf_config for {what}: {e}")
        else:
            bf_ckpt_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
            if bf_ckpt_path.exists():
                try:
                    base = torch.load(bf_ckpt_path, map_location=DEVICE)
                    if "bf_config" in base:
                        rebuilt = build_backflow_from_config(base["bf_config"])
                        loaded_base = try_load_state(rebuilt, ckpt_obj["bf_state"], f"{what}(base-config)")
                        if loaded_base:
                            return rebuilt, True
                except Exception as e:
                    print(f"    WARNING: could not use base bf_config fallback for {what}: {e}")

        loaded = try_load_state(module, ckpt_obj["bf_state"], what)
        if loaded:
            return module, True

        if "bf_config" in ckpt_obj:
            try:
                rebuilt = build_backflow_from_config(ckpt_obj["bf_config"])
                loaded2 = try_load_state(rebuilt, ckpt_obj["bf_state"], f"{what}(rebuilt)")
                if loaded2:
                    return rebuilt, True
            except Exception as e:
                print(f"    WARNING: could not rebuild backflow for {what}: {e}")

        if required:
            raise RuntimeError(f"Failed to load required backflow state from {what}")
        return module, False

    def maybe_load_jas_from_ckpt(module, ckpt_obj, what):
        if "jas_state" in ckpt_obj:
            return try_load_state(module, ckpt_obj["jas_state"], what)
        if "state" in ckpt_obj:
            return try_load_state(module, ckpt_obj["state"], what)
        print(f"    WARNING: {what} has no jas_state/state")
        return False

    print(f"{'=' * 64}")
    print(f"  Weak-Form Collocation — {a.tag}")
    print(f"  Mode: {a.mode}   Device: {DEVICE}   Seed: {a.seed}")
    print(f"  N={N_ELEC}  omega={OMEGA}  basis={nx}x{ny} (n_basis={n_basis}, n_occ={n_occ})")
    if math.isfinite(E_DMC):
        print(f"  DMC reference: {E_DMC}")
    else:
        print("  DMC reference: disabled (err shown as NaN)")
    print(f"{'=' * 64}")
    sys.stdout.flush()

    # ── Load Jastrow (always) ──
    backflow_net = None
    npf_net = None
    f_net = build_jastrow_model()

    if a.mode == "bf":
        backflow_net = build_default_backflow()

        if a.resume:
            print("  Resume requested: skipping optional pretrained BF+Jastrow init")
        elif not a.no_pretrained:
            bf_ckpt_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
            if bf_ckpt_path.exists():
                print(f"  Trying pretrained BF+Jastrow init from {bf_ckpt_path}")
                ckpt = torch.load(bf_ckpt_path, map_location=DEVICE)
                maybe_load_jas_from_ckpt(f_net, ckpt, "pretrained jas")
                backflow_net, _ = maybe_load_bf_from_ckpt(
                    backflow_net, ckpt, "pretrained bf", required=False
                )

        if a.init_jas:
            print(f"  Init Jastrow from {a.init_jas}")
            jckpt = torch.load(a.init_jas, map_location=DEVICE)
            if not maybe_load_jas_from_ckpt(f_net, jckpt, "init jas"):
                raise RuntimeError(f"Failed to load required Jastrow init from {a.init_jas}")

        if a.init_bf:
            print(f"  Init Backflow from {a.init_bf}")
            bckpt = torch.load(a.init_bf, map_location=DEVICE)
            backflow_net, _ = maybe_load_bf_from_ckpt(
                backflow_net, bckpt, "init bf", required=True
            )

        n_bf = sum(p.numel() for p in backflow_net.parameters())
        n_jas = sum(p.numel() for p in f_net.parameters())
        print(f"  Jastrow: {n_jas:,} params")
        print(f"  Backflow: {n_bf:,} params")

    elif a.mode == "pfaffian":
        if N_ELEC != 6:
            raise ValueError("pfaffian mode currently supports only N=6 in this script")
        from run_neural_pfaffian import NeuralPfaffianNet

        if a.init_jas:
            print(f"  Init Jastrow from {a.init_jas}")
            jckpt = torch.load(a.init_jas, map_location=DEVICE)
            maybe_load_jas_from_ckpt(f_net, jckpt, "init jas")

        if a.use_backflow:
            bf_ckpt_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
            backflow_net = build_default_backflow()
            if not a.no_pretrained and bf_ckpt_path.exists():
                ckpt = torch.load(bf_ckpt_path, map_location=DEVICE)
                maybe_load_jas_from_ckpt(f_net, ckpt, "pretrained jas")
                if "bf_state" in ckpt:
                    try_load_state(backflow_net, ckpt["bf_state"], "pretrained bf")
            if a.init_bf:
                print(f"  Init Backflow from {a.init_bf}")
                bckpt = torch.load(a.init_bf, map_location=DEVICE)
                if "bf_state" in bckpt:
                    try_load_state(backflow_net, bckpt["bf_state"], "init bf")
            if not a.train_bf:
                for p in backflow_net.parameters():
                    p.requires_grad = False
                print(f"  Backflow: {sum(p.numel() for p in backflow_net.parameters()):,} params (FROZEN)")
            else:
                print(f"  Backflow: {sum(p.numel() for p in backflow_net.parameters()):,} params (TRAINABLE)")
        else:
            if not a.no_pretrained:
                jas_path = RESULTS_DIR / "ctnn_vcycle.pt"
                if jas_path.exists():
                    jas_ckpt = torch.load(jas_path, map_location=DEVICE)
                    maybe_load_jas_from_ckpt(f_net, jas_ckpt, "pretrained jas")

        if not a.train_jas:
            for p in f_net.parameters():
                p.requires_grad = False
            print(f"  Jastrow: {sum(p.numel() for p in f_net.parameters()):,} params (FROZEN)")
        else:
            print(f"  Jastrow: {sum(p.numel() for p in f_net.parameters()):,} params (TRAINABLE)")

        npf_net = NeuralPfaffianNet(
            n_basis, n_occ, C_occ, nx, ny,
            K_emb=a.K_emb, K_det=a.K_det,
            embed_hidden=64, embed_layers=2, spin_dim=8,
        ).to(DEVICE).to(DTYPE)
        print(f"  Pfaffian: {npf_net.param_summary()}")

    elif a.mode == "jastrow":
        if not a.no_pretrained:
            jas_path = RESULTS_DIR / "ctnn_vcycle.pt"
            if jas_path.exists():
                print(f"  Trying pretrained Jastrow from {jas_path}")
                jas_ckpt = torch.load(jas_path, map_location=DEVICE)
                maybe_load_jas_from_ckpt(f_net, jas_ckpt, "pretrained jas")
        if a.init_jas:
            print(f"  Init Jastrow from {a.init_jas}")
            jckpt = torch.load(a.init_jas, map_location=DEVICE)
            maybe_load_jas_from_ckpt(f_net, jckpt, "init jas")
        n_jas = sum(p.numel() for p in f_net.parameters())
        print(f"  Jastrow: {n_jas:,} params")
    else:
        raise ValueError(f"Unknown mode: {a.mode}")

    # ── Resume from checkpoint ──
    if a.resume:
        print(f"  Resuming from {a.resume}")
        rckpt = torch.load(a.resume, map_location=DEVICE)
        if "pf_state" in rckpt and npf_net is not None:
            npf_net.load_state_dict(rckpt["pf_state"])
            print(f"    Loaded pf_state")
        if "bf_state" in rckpt and backflow_net is not None:
            backflow_net, _ = maybe_load_bf_from_ckpt(
                backflow_net, rckpt, "resume bf", required=True
            )
            print(f"    Loaded bf_state")
        if "jas_state" in rckpt:
            if not try_load_state(f_net, rckpt["jas_state"], "resume jas"):
                raise RuntimeError(f"Failed to load required Jastrow state from resume checkpoint: {a.resume}")
            print("    Loaded jas_state")

    sys.stdout.flush()

    # ── Train ──
    f_net, backflow_net, npf_net, hist = train_weak_form(
        f_net, C_occ, params,
        backflow_net=backflow_net,
        npf_net=npf_net,
        n_epochs=a.epochs, lr=a.lr, lr_jas=a.lr_jas,
        n_coll=a.n_coll,
        oversample=a.oversample,
        micro_batch=a.micro_batch,
        grad_clip=a.grad_clip,
        direct_weight=a.direct_weight,
        clip_el=a.clip_el,
        reward_qtrim=a.reward_qtrim,
        loss_type=a.loss_type,
        fd_h=a.fd_h,
        fd_huber_delta=a.fd_huber_delta,
        prox_mu=a.prox_mu,
        min_ess=a.min_ess,
        sigma_fs=sigma_fs,
        min_pair_cutoff=a.min_pair_cutoff,
        ess_floor_ratio=a.ess_floor_ratio,
        ess_oversample_max=a.ess_oversample_max,
        ess_oversample_step=a.ess_oversample_step,
        ess_resample_tries=a.ess_resample_tries,
        resample_weight_temp=a.resample_weight_temp,
        resample_logw_clip_q=a.resample_logw_clip_q,
        langevin_steps=a.langevin_steps,
        langevin_step_size=a.langevin_step_size,
        replay_frac=a.replay_frac,
        replay_top_frac=a.replay_top_frac,
        replay_stratified=a.replay_stratified,
        replay_geo_bins=a.replay_geo_bins,
        rollback_decay=a.rollback_decay,
        rollback_err_pct=a.rollback_err_pct,
        rollback_jump_sigma=a.rollback_jump_sigma,
        bf_cusp_reg=a.bf_cusp_reg,
        bf_cusp_radius_aho=a.bf_cusp_radius_aho,
        bf_diag_q=a.bf_diag_q,
        n_elec=N_ELEC,
        omega=OMEGA,
        e_ref=E_DMC,
        print_every=10, patience=a.patience,
        vmc_every=a.vmc_every, vmc_n=a.vmc_n, tag=a.tag,
        vmc_select_n=a.vmc_select_n,
        natural_grad=a.natural_grad,
        sr_mode=a.sr_mode,
        fisher_damping=a.fisher_damping,
        fisher_damping_end=a.fisher_damping_end,
        fisher_damping_anneal=a.fisher_damping_anneal,
        fisher_ema=a.fisher_ema,
        fisher_probes=a.fisher_probes,
        fisher_subsample=a.fisher_subsample,
        fisher_max=a.fisher_max,
        nat_momentum=a.nat_momentum,
        sr_max_param_change=a.sr_max_param_change,
        sr_trust_region=a.sr_trust_region,
        sr_cg_iters=a.sr_cg_iters,
        sr_center=a.sr_center,
    )

    # ── Final heavy VMC eval ──
    E = se = err = float("nan")
    if a.n_eval > 0:
        print(f"\n  Final VMC eval: {a.n_eval} samples ...")
        sys.stdout.flush()

        for net in [f_net, backflow_net, npf_net]:
            if net is not None:
                net.eval()

        if npf_net is not None:
            from run_neural_pfaffian import psi_fn_npf as _psi_fn_npf
            def _psi_wrap(f_, x_, C_, *, backflow_net=None, orbital_bf_net=None, spin=None, params=None):
                return _psi_fn_npf(npf_net, f_net, x_, params, spin=spin, bf_net=backflow_net)
            C_dummy = torch.eye(n_basis, n_occ, device=DEVICE, dtype=DTYPE)
            vmc = evaluate_energy_vmc(
                f_net, C_dummy, psi_fn=_psi_wrap,
                compute_coulomb_interaction=compute_coulomb_interaction,
                params=params, n_samples=a.n_eval, batch_size=512,
                sampler_steps=120, sampler_step_sigma=0.08, lap_mode="exact",
                persistent=True, sampler_burn_in=800, sampler_thin=5, progress=True,
            )
        else:
            vmc = evaluate_energy_vmc(
                f_net, C_occ, psi_fn=psi_fn,
                compute_coulomb_interaction=compute_coulomb_interaction,
                backflow_net=backflow_net, params=params,
                n_samples=a.n_eval, batch_size=512,
                sampler_steps=120, sampler_step_sigma=0.08, lap_mode="exact",
                persistent=True, sampler_burn_in=800, sampler_thin=5, progress=True,
            )
        E = float(vmc["E_mean"])
        se = float(vmc["E_stderr"])
        err = safe_percent_err(E, E_DMC)
        print(f"\n  *** Final: E = {E:.5f} ± {se:.5f}   err = {err:+.3f}%")

    # ── Save ──
    save_path = RESULTS_DIR / f"{a.tag}.pt"
    ckpt_dict = dict(
        tag=a.tag, mode=a.mode,
        jas_state=f_net.state_dict(),
        E=E, se=se, err=err, hist=hist,
        seed=a.seed,
        n_elec=N_ELEC,
        omega=OMEGA,
        e_dmc=E_DMC,
        nx=nx,
        ny=ny,
    )
    if backflow_net is not None:
        ckpt_dict["bf_state"] = backflow_net.state_dict()
    if npf_net is not None:
        ckpt_dict["pf_state"] = npf_net.state_dict()
    torch.save(ckpt_dict, save_path)
    print(f"  Saved → {save_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  WEAK-FORM RESULT  N={N_ELEC}  ω={OMEGA}  E_DMC={E_DMC}")
    print(f"  Mode: {a.mode}   Tag: {a.tag}")
    print(f"  E = {E:.6f} ± {se:.6f}   err = {err:+.3f}%")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
