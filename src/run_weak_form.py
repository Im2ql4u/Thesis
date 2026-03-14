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
from functions.Neural_Networks import psi_fn
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import evaluate_basis_functions_torch_batch_2d
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet

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

DEFAULT_DMC = {
    (6, 1.0): 20.15932,
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"


def default_dmc(n_elec, omega):
    key = (int(n_elec), float(omega))
    if key in DEFAULT_DMC:
        return DEFAULT_DMC[key]
    return float("nan")


def _choose_basis_dims(n_occ):
    n_side = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
    return n_side, n_side


def safe_percent_err(E, E_ref):
    if not math.isfinite(E_ref) or E_ref == 0.0:
        return float("nan")
    return (E - E_ref) / abs(E_ref) * 100.0


def setup(n_elec=None, omega=None, e_dmc=None):
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


def sample_mixture(n, omega, sigma_fs=(0.8, 1.3, 2.0)):
    """Sample from Gaussian mixture, return (x, log_q)."""
    nc = len(sigma_fs)
    xs, lqs = [], []
    for i, sf in enumerate(sigma_fs):
        ni = n // nc if i < nc - 1 else n - (n // nc) * (nc - 1)
        xi, lqi = sample_gauss(ni, omega, sf)
        xs.append(xi)
        lqs.append(lqi)
    x_all = torch.cat(xs)
    lq_all = torch.cat(lqs)
    # Shuffle
    perm = torch.randperm(x_all.shape[0], device=x_all.device)
    return x_all[perm[:n]], lq_all[perm[:n]]


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
):
    """Multinomial resampling: sample from q, resample ∝ |Ψ|²/q.

    Returns points approximately distributed as |Ψ|², without MCMC.
    This avoids the double-weighting problem of the old approach
    (top-k selection + |Ψ|² weights in loss = |Ψ|⁴ effective weighting).

    With resampled points, the Rayleigh quotient gradient is:
      ∂R/∂θ = 2⟨(ẽ-R)·∂logΨ/∂θ⟩ + ⟨∂ẽ/∂θ⟩
    estimated as a simple sample mean (no importance weights needed).
    """
    n_cand = n_cand_mult * n_keep
    x_all, lq_all = sample_mixture(n_cand, omega, sigma_fs)

    # Optional near-overlap exclusion to reduce Coulomb singular tails.
    if min_pair_cutoff > 0:
        mp = _min_pair_distance(x_all)
        keep = mp >= min_pair_cutoff
        if int(keep.sum().item()) >= n_keep:
            x_all = x_all[keep]
            lq_all = lq_all[keep]

    # Compute log importance weights
    lp2 = []
    for i in range(0, len(x_all), 4096):
        lp2.append(2.0 * psi_log_fn(x_all[i:i + 4096]))
    lp2 = torch.cat(lp2)

    log_w = lp2 - lq_all  # log(|Ψ|²/q)
    log_w = log_w - log_w.max()  # numerical stability
    w = torch.exp(log_w)
    probs = w / w.sum()

    # ESS of the proposal → target matching
    ess = (w.sum() ** 2 / (w ** 2).sum()).item()

    # Multinomial resampling (with replacement) → approx |Ψ|² samples
    idx = torch.multinomial(probs, n_keep, replacement=True)
    return x_all[idx].clone(), ess


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
    x = x.detach()  # inputs are fixed collocation points
    B, N, d = x.shape
    Nd = N * d
    x_flat = x.reshape(B, Nd)

    # Evaluate logΨ at original points (in graph w.r.t. θ)
    lp0 = psi_log_fn(x)  # (B,)

    # Component-by-component FD Laplacian and gradient-squared
    h2_inv = 1.0 / (h * h)
    h2_inv_grad = 1.0 / (2.0 * h)
    lap_fd = torch.zeros(B, device=x.device, dtype=x.dtype)
    g2_fd = torch.zeros(B, device=x.device, dtype=x.dtype)

    for i in range(Nd):
        ei = torch.zeros(1, Nd, device=x.device, dtype=x.dtype)
        ei[0, i] = h
        xp = (x_flat + ei).reshape(B, N, d)
        xm = (x_flat - ei).reshape(B, N, d)

        lp_p = psi_log_fn(xp)  # (B,) — in graph
        lp_m = psi_log_fn(xm)  # (B,) — in graph

        # ∂²logΨ/∂x_i²  (centered FD)
        lap_fd = lap_fd + (lp_p + lp_m - 2.0 * lp0) * h2_inv
        # (∂logΨ/∂x_i)²  (centered FD)
        gi = (lp_p - lp_m) * h2_inv_grad
        g2_fd = g2_fd + gi * gi

    # Potential energy (detached — doesn't depend on θ)
    V = 0.5 * omega ** 2 * (x ** 2).sum(dim=(1, 2)) + compute_coulomb_interaction(
        x, params=params
    ).view(B)

    # E_L = -½(∇²logΨ + |∇logΨ|²) + V  — IN THE GRAPH via forward passes!
    E_L = -0.5 * (lap_fd + g2_fd) + V  # (B,)

    # Loss
    E_mean = E_L.mean()
    if huber_delta > 0:
        resid = E_L - E_mean.detach()
        L = torch.nn.functional.huber_loss(
            resid, torch.zeros_like(resid), delta=huber_delta, reduction="mean"
        )
    else:
        L = ((E_L - E_mean) ** 2).mean()  # variance

    # Proximal penalty: keep logΨ close to its pre-step values
    if lp_prev is not None and prox_mu > 0:
        L = L + prox_mu * ((lp0 - lp_prev) ** 2).mean()

    return L, E_mean.item(), E_L.detach(), L.item()


# ═══════════════════════════════════════════════════════════════
#  Loss: hybrid REINFORCE + direct gradient
# ═══════════════════════════════════════════════════════════════

def rayleigh_hybrid_loss(psi_log_fn, x, omega, params, direct_weight=0.1,
                         clip_el=5.0):
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
    x = x.detach().requires_grad_(True)
    lp = psi_log_fn(x)  # (B,) — in graph (for REINFORCE through logΨ)

    # First gradient with create_graph (for direct term through g²)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True)[0]  # (B,N,d)
    g2 = (g ** 2).sum(dim=(1, 2))  # (B,) — in graph

    B = x.shape[0]
    V = 0.5 * omega ** 2 * (x ** 2).sum(dim=(1, 2)) + compute_coulomb_interaction(
        x, params=params
    ).view(B)

    # Weak-form energy (in graph for direct gradient of kinetic energy)
    e_weak = 0.5 * g2 + V  # (B,) — in graph

    # ── Laplacian (NOT in backward graph — create_graph=False) ──
    g_flat = g.reshape(B, -1)  # (B, N*d)
    Nd = g_flat.shape[1]
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(Nd):
        gg = torch.autograd.grad(g_flat[:, i].sum(), x,
                                 retain_graph=True, create_graph=False)[0]
        lap = lap + gg.reshape(B, -1)[:, i]

    # E_L = -½(∇²logΨ + |∇logΨ|²) + V — detached, forward-only
    E_L = (-0.5 * (lap + g2.detach()) + V).detach()  # (B,) detached

    # Robust clipping: remove outliers using median absolute deviation
    med = E_L.median()
    mad = (E_L - med).abs().median()
    if mad > 0 and clip_el > 0:
        E_L = E_L.clamp(med - clip_el * mad, med + clip_el * mad)

    R = E_L.mean()  # baseline

    # ── Dual loss ──
    # REINFORCE: 2⟨(E_L - R̄)·logΨ⟩  → gradient = 2⟨(E_L-R)·∂logΨ/∂θ⟩
    L_reinforce = 2.0 * ((E_L - R) * lp).mean()
    # Direct:    β·mean(ẽ)           → gradient = β·⟨½·∂g²/∂θ⟩
    L_direct = direct_weight * e_weak.mean()
    L = L_reinforce + L_direct

    return L, R.item(), E_L.detach(), e_weak.detach()


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
    loss_type="reinforce",
    fd_h=0.01,
    fd_huber_delta=0.0,
    prox_mu=0.0,
    min_ess=0,
    sigma_fs=(0.8, 1.3, 2.0),
    min_pair_cutoff=0.0,
    print_every=10,
    patience=300,
    vmc_every=50,
    vmc_n=10000,
    bf_warmup=0,
    n_elec=None,
    omega=None,
    e_ref=None,
    tag="weak_form",
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

    opt = torch.optim.Adam(trainable_groups)

    def lr_lambda(ep):
        lr_min_abs = lr * lr_min_frac
        return (lr_min_abs + 0.5 * (lr - lr_min_abs) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))) / lr

    sch = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda] * len(trainable_groups))

    print(f"  Training: {n_epochs} ep, {n_coll} samples, LR={lr}")
    if loss_type == "fd-colloc":
        loss_str = "Huber" if fd_huber_delta > 0 else "Var"
        print(f"  FD-COLLOC loss: {loss_str}(E_L) via FD Laplacian"
              f"  h={fd_h}  huber_δ={fd_huber_delta}  prox_μ={prox_mu}")
    else:
        print(f"  HYBRID loss: REINFORCE(E_L, no-backprop-lap) + direct(½g², β={direct_weight})"
              f"  clip_el={clip_el}")
    sys.stdout.flush()

    t0 = time.time()
    hist = []
    best_R = float("inf")
    best_vmc_err = float("inf")
    best_state = best_vmc_state = {}
    best_vmc_E = None
    no_imp = 0
    n_ess_reject = 0

    # Save initial state for ESS-gated rollback
    def _save_state():
        st = {"jas_state": {k: v.clone() for k, v in f_net.state_dict().items()}}
        if backflow_net is not None:
            st["bf_state"] = {k: v.clone() for k, v in backflow_net.state_dict().items()}
        if npf_net is not None:
            st["pf_state"] = {k: v.clone() for k, v in npf_net.state_dict().items()}
        return st

    def _restore_state(st):
        f_net.load_state_dict(st["jas_state"])
        if backflow_net is not None and "bf_state" in st:
            backflow_net.load_state_dict(st["bf_state"])
        if npf_net is not None and "pf_state" in st:
            npf_net.load_state_dict(st["pf_state"])

    last_good_state = _save_state()

    for ep in range(n_epochs):
        ept0 = time.time()

        # ── Resample from approximate |Ψ|² ──
        for net in [f_net, backflow_net, npf_net]:
            if net is not None:
                net.eval()

        X, ess = importance_resample(
            psi_log_sample_fn,
            n_coll,
            omega,
            n_cand_mult=oversample,
            sigma_fs=sigma_fs,
            min_pair_cutoff=min_pair_cutoff,
        )

        # ── ESS gate: skip step if ESS too low (revert to last good state) ──
        if min_ess > 0 and ess < min_ess:
            _restore_state(last_good_state)
            n_ess_reject += 1
            if ep % print_every == 0 or ep == 0:
                print(f"  [{ep:4d}] ESS={ess:.0f} < {min_ess} → SKIP (reverted, "
                      f"{n_ess_reject} rejects total)")
                sys.stdout.flush()
            continue

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
                )
            (L / nmb).backward()
            ep_loss += L.item() / nmb
            all_EL.append(EL_det)
            all_e_weak.append(ew_det)

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_trainable, grad_clip)
        opt.step()
        sch.step()

        # Epoch statistics from the hybrid loss
        EL_all = torch.cat(all_EL)
        Em = EL_all.mean().item()  # E_L mean (accurate energy estimate)
        Ev = EL_all.var().item()   # E_L variance
        Es = EL_all.std().item()

        epdt = time.time() - ept0
        entry = dict(ep=ep, E=Em, var_EL=Ev, ess=ess, dt=epdt, loss=ep_loss)

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
                        sampler_steps=40, sampler_step_sigma=0.12, lap_mode="exact",
                        persistent=True, sampler_burn_in=200, sampler_thin=2, progress=False,
                    )
                else:
                    vp = evaluate_energy_vmc(
                        f_net, C_occ, psi_fn=psi_fn,
                        compute_coulomb_interaction=compute_coulomb_interaction,
                        backflow_net=backflow_net, params=params,
                        n_samples=vmc_n, batch_size=512,
                        sampler_steps=40, sampler_step_sigma=0.12, lap_mode="exact",
                        persistent=True, sampler_burn_in=200, sampler_thin=2, progress=False,
                    )
                vE = float(vp["E_mean"])
                vErr = abs(vE - E_ref) / abs(E_ref) if math.isfinite(E_ref) and E_ref != 0 else float("nan")
                entry.update(vmc_E=vE, vmc_err=vErr)
                # When E_ref is NaN (no DMC reference), fall back to minimizing vE directly
                if math.isfinite(E_ref) and E_ref != 0:
                    if math.isfinite(vErr) and vErr < best_vmc_err:
                        best_vmc_err = vErr
                        best_vmc_E = vE
                        best_vmc_state = _save_state()
                else:
                    if best_vmc_E is None or vE < best_vmc_E:
                        best_vmc_E = vE
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
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Do not load default pretrained initialization checkpoints")
    ap.add_argument("--init-jas", type=str, default=None,
                    help="Optional checkpoint path to initialize Jastrow (expects jas_state or state)")
    ap.add_argument("--init-bf", type=str, default=None,
                    help="Optional checkpoint path to initialize backflow (expects bf_state)")
    a = ap.parse_args()

    global N_ELEC, OMEGA, E_DMC
    N_ELEC = int(a.n_elec)
    OMEGA = float(a.omega)
    E_DMC = default_dmc(N_ELEC, OMEGA) if not math.isfinite(a.e_dmc) else float(a.e_dmc)

    sigma_fs = tuple(float(s) for s in a.sigma_fs.split(",") if s.strip())
    if len(sigma_fs) == 0:
        raise ValueError("--sigma-fs must contain at least one value")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C_occ, params, nx, ny = setup(n_elec=N_ELEC, omega=OMEGA, e_dmc=E_DMC)
    n_basis = nx * ny
    n_occ = N_ELEC // 2

    def build_jastrow_model():
        return CTNNJastrowVCycle(
            n_particles=N_ELEC, d=DIM, omega=OMEGA,
            node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
            n_down=1, n_up=1, msg_layers=1, node_layers=1,
            readout_hidden=64, readout_layers=2, act="silu",
        ).to(DEVICE).to(DTYPE)

    def build_default_backflow():
        return CTNNBackflowNet(
            d=DIM,
            msg_hidden=128,
            msg_layers=2,
            hidden=128,
            layers=3,
            act="silu",
            aggregation="sum",
            use_spin=True,
            same_spin_only=False,
            out_bound="tanh",
            bf_scale_init=0.05,
            zero_init_last=True,
            omega=OMEGA,
        ).to(DEVICE).to(DTYPE)

    def try_load_state(module, state, what):
        try:
            module.load_state_dict(state)
            return True
        except RuntimeError as e:
            print(f"    WARNING: could not load {what} (shape mismatch): {e}")
            return False

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

        if not a.no_pretrained:
            bf_ckpt_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
            if bf_ckpt_path.exists():
                print(f"  Trying pretrained BF+Jastrow init from {bf_ckpt_path}")
                ckpt = torch.load(bf_ckpt_path, map_location=DEVICE)
                maybe_load_jas_from_ckpt(f_net, ckpt, "pretrained jas")
                if "bf_state" in ckpt:
                    loaded = try_load_state(backflow_net, ckpt["bf_state"], "pretrained bf")
                    if not loaded and "bf_config" in ckpt:
                        bfc = ckpt["bf_config"]
                        try:
                            backflow_net = CTNNBackflowNet(
                                d=bfc["d"], msg_hidden=bfc["msg_hidden"], msg_layers=bfc["msg_layers"],
                                hidden=bfc["hidden"], layers=bfc["layers"], act=bfc["act"],
                                aggregation=bfc["aggregation"], use_spin=bfc["use_spin"],
                                same_spin_only=bfc["same_spin_only"], out_bound=bfc["out_bound"],
                                bf_scale_init=bfc["bf_scale_init"], zero_init_last=bfc["zero_init_last"],
                                omega=OMEGA,
                            ).to(DEVICE).to(DTYPE)
                            try_load_state(backflow_net, ckpt["bf_state"], "pretrained bf(rebuilt)")
                        except Exception as e:
                            print(f"    WARNING: fallback bf rebuild failed: {e}")

        if a.init_jas:
            print(f"  Init Jastrow from {a.init_jas}")
            jckpt = torch.load(a.init_jas, map_location=DEVICE)
            maybe_load_jas_from_ckpt(f_net, jckpt, "init jas")

        if a.init_bf:
            print(f"  Init Backflow from {a.init_bf}")
            bckpt = torch.load(a.init_bf, map_location=DEVICE)
            if "bf_state" in bckpt:
                try_load_state(backflow_net, bckpt["bf_state"], "init bf")
            else:
                print("    WARNING: init-bf checkpoint has no bf_state")

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
            backflow_net.load_state_dict(rckpt["bf_state"])
            print(f"    Loaded bf_state")
        if "jas_state" in rckpt:
            try_load_state(f_net, rckpt["jas_state"], "resume jas")
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
        loss_type=a.loss_type,
        fd_h=a.fd_h,
        fd_huber_delta=a.fd_huber_delta,
        prox_mu=a.prox_mu,
        min_ess=a.min_ess,
        sigma_fs=sigma_fs,
        min_pair_cutoff=a.min_pair_cutoff,
        n_elec=N_ELEC,
        omega=OMEGA,
        e_ref=E_DMC,
        print_every=10, patience=a.patience,
        vmc_every=a.vmc_every, vmc_n=a.vmc_n, tag=a.tag,
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
                sampler_steps=80, sampler_step_sigma=0.08, lap_mode="exact",
                persistent=True, sampler_burn_in=400, sampler_thin=3, progress=True,
            )
        else:
            vmc = evaluate_energy_vmc(
                f_net, C_occ, psi_fn=psi_fn,
                compute_coulomb_interaction=compute_coulomb_interaction,
                backflow_net=backflow_net, params=params,
                n_samples=a.n_eval, batch_size=512,
                sampler_steps=80, sampler_step_sigma=0.08, lap_mode="exact",
                persistent=True, sampler_burn_in=400, sampler_thin=3, progress=True,
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
