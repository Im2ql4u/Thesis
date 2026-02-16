"""
Hybrid loss + sampling improvements for backflow training.

Key insight from analysis: Our variance-like loss (Huber of E_L residuals)
has gradient ∝ (E_L - Ē)·∂E_L/∂θ — this QUADRATICALLY amplifies outliers,
causing backflow to shrink defensively near nodes.

VMC's mean-energy gradient is ∝ ∂E_L/∂θ (linear) — coherent direction that
doesn't punish near-node volatility. Under collocation, mean(E_L) over
screened points is biased but correlated with the variational energy.

Fix: hybrid loss = β·Huber(E_L - Ē) + (1-β)·TrimmedMean(E_L)
  β anneals from high (variance-stabilizing early) to low (mean-energy later)

Additionally:
  - Gumbel-top-K sampling: weighted without-replacement via Gumbel noise
    β_temp controls exploration. No duplicates (unlike SIR). Trivial ESS = K.
  - Det-conditioning filter: exclude configs with |det S̃| < τ

Experiments:
  1. hybrid:         hybrid loss + standard top-K (isolate loss effect)
  2. gumbel_hybrid:  hybrid loss + Gumbel-top-K (combine both ideas)
  3. hybrid_strong:  strong mean-energy (β_loss=0.2 constant, aggressive)
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import _laplacian_logpsi_exact, psi_fn
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import (
    evaluate_basis_functions_torch_batch_2d,
)
from PINN import PINN, CTNNBackflowNet

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"
LOG_DIR = "/Users/aleksandersekkelsten/thesis/results/logs"

N_PARTICLES, DIM, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)


# ══════════════════════════════════════════════════════════════════
#  Model helpers  (same as previous scripts)
# ══════════════════════════════════════════════════════════════════


def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
    n_occ = N // 2
    nx = {2: 2, 6: 3, 12: 4, 20: 5}.get(N, 4)
    ny = nx
    n_basis = nx * ny
    L = max(8.0, 3.0 / math.sqrt(omega))
    config.update(
        omega=omega,
        n_particles=N,
        d=d,
        L=L,
        n_grid=80,
        nx=nx,
        ny=ny,
        basis="cart",
        device=str(device),
        dtype="float64",
    )
    energies = []
    for ix in range(nx):
        for iy in range(ny):
            energies.append((omega * (ix + iy + 1), ix, iy))
    energies.sort(key=lambda t: t[0])
    C_occ_np = np.zeros((n_basis, n_occ), dtype=np.float64)
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C_occ_np[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C_occ_np, dtype=dtype, device=device)
    params = config.get().as_dict()
    params["device"] = device
    params["torch_dtype"] = dtype
    params["E"] = E_DMC
    occ_str = ", ".join(f"({energies[k][1]},{energies[k][2]})" for k in range(n_occ))
    print(f"N={N}, ω={omega}, basis={nx}×{ny}={n_basis}")
    print(f"  Occupied: {occ_str}   E_DMC={E_DMC}")
    return C_occ, params


def make_nets(bf_scale_init=0.7, zero_init_last=False):
    f_net = (
        PINN(
            n_particles=N_PARTICLES,
            d=DIM,
            omega=OMEGA,
            dL=8,
            hidden_dim=64,
            n_layers=2,
            act="gelu",
            init="xavier",
            use_gate=True,
            use_pair_attn=False,
        )
        .to(DEVICE)
        .to(DTYPE)
    )
    bf_net = (
        CTNNBackflowNet(
            d=DIM,
            msg_hidden=32,
            msg_layers=1,
            hidden=32,
            layers=2,
            act="silu",
            aggregation="sum",
            use_spin=True,
            same_spin_only=False,
            out_bound="tanh",
            bf_scale_init=bf_scale_init,
            zero_init_last=zero_init_last,
            omega=OMEGA,
        )
        .to(DEVICE)
        .to(DTYPE)
    )
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    up = N_PARTICLES // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_PARTICLES - up, dtype=torch.long)]
    ).to(DEVICE)

    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return lp

    return fn, spin


def compute_local_energy(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap_log.view(B) + g2.view(B))
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).view(B)
    return T + V_harm + V_int


def save_model(f_net, bf_net, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"6e_hyb_{name}.pt")
    torch.save({"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()}, path)
    print(f"  Saved -> {path}")


def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    print(f"\n-- VMC eval: {label} --")
    result = evaluate_energy_vmc(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=backflow_net,
        params=params,
        n_samples=n_samples,
        batch_size=512,
        sampler_steps=50,
        sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True,
        sampler_burn_in=300,
        sampler_thin=3,
        progress=True,
    )
    E, E_std = result["E_mean"], result["E_stderr"]
    err = abs(E - E_DMC) / E_DMC * 100
    print(f"  E = {E:.6f} +/- {E_std:.6f}  (target {E_DMC}, err {err:.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════════
#  Smoothness penalty (proven: 0.38% vs 0.59%)
# ══════════════════════════════════════════════════════════════════


def compute_bf_smoothness_penalty(bf_net, x, spin, n_samples=32):
    x_sub = x[:n_samples].detach().requires_grad_(True)
    B, N, d = x_sub.shape
    dx = bf_net(x_sub, spin)
    n_probes = 2
    lap_sq_sum = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    for _ in range(n_probes):
        v = torch.empty_like(x_sub).bernoulli_(0.5).mul_(2).add_(-1)
        for k in range(d):
            dx_k_sum = dx[:, :, k].sum()
            grad1 = torch.autograd.grad(dx_k_sum, x_sub, create_graph=True, retain_graph=True)[0]
            Hv = torch.autograd.grad(
                (grad1 * v).sum(), x_sub, create_graph=True, retain_graph=True
            )[0]
            vTHv = (v * Hv).sum(dim=(1, 2))
            lap_sq_sum = lap_sq_sum + (vTHv**2).mean()
    return lap_sq_sum / (n_probes * d)


# ══════════════════════════════════════════════════════════════════
#  Sampling: standard top-K and Gumbel-top-K
# ══════════════════════════════════════════════════════════════════


@torch.no_grad()
def sample_gaussian(n_samples, sigma):
    x = torch.randn(n_samples, N_PARTICLES, DIM, device=DEVICE, dtype=DTYPE) * sigma
    Nd = N_PARTICLES * DIM
    log_q = -0.5 * Nd * math.log(2 * math.pi * sigma**2) - x.reshape(n_samples, -1).pow(2).sum(
        -1
    ) / (2 * sigma**2)
    return x, log_q


@torch.no_grad()
def topk_collocation(psi_log_fn, n_keep, oversampling, sigma, batch_size=4096):
    """Standard deterministic top-K screening."""
    M = oversampling * n_keep
    x_cand, log_q = sample_gaussian(M, sigma)

    log_psi_parts = []
    for i in range(0, M, batch_size):
        lp = psi_log_fn(x_cand[i : i + batch_size])
        log_psi_parts.append(lp)
    log_psi = torch.cat(log_psi_parts)

    log_w = 2.0 * log_psi - log_q
    valid = torch.isfinite(log_w)
    if not valid.all():
        log_w[~valid] = -1e10

    _, idx = torch.topk(log_w, n_keep)
    X = x_cand[idx].clone()
    return X, {"method": "topk", "n_unique": n_keep, "unique_frac": 1.0}


@torch.no_grad()
def gumbel_topk_collocation(
    psi_log_fn, n_keep, oversampling, sigma, beta_temp=0.5, batch_size=4096
):
    """
    Gumbel-top-K: weighted sampling WITHOUT replacement.

    Instead of deterministic top-K (mode-seeking) or SIR (with duplicates),
    add Gumbel noise to log-weights and take top-K of the perturbed values.

    This is the Gumbel-max trick for sampling without replacement:
      - beta_temp → 0: uniform random (max exploration)
      - beta_temp → ∞: deterministic top-K (mode-seeking)
      - beta_temp ∈ (0, 1): smooth interpolation

    No duplicates by construction. ESS is always n_keep.
    """
    M = oversampling * n_keep
    x_cand, log_q = sample_gaussian(M, sigma)

    log_psi_parts = []
    for i in range(0, M, batch_size):
        lp = psi_log_fn(x_cand[i : i + batch_size])
        log_psi_parts.append(lp)
    log_psi = torch.cat(log_psi_parts)

    log_w = 2.0 * log_psi - log_q
    valid = torch.isfinite(log_w)
    if not valid.all():
        log_w[~valid] = -1e10

    # Gumbel noise for sampling without replacement
    # u ~ Uniform(0,1), gumbel = -log(-log(u))
    u = torch.rand_like(log_w).clamp(1e-30, 1.0 - 1e-30)
    gumbel = -torch.log(-torch.log(u))

    # Perturbed weights: beta_temp controls exploration
    perturbed = beta_temp * log_w + gumbel

    _, idx = torch.topk(perturbed, n_keep)
    X = x_cand[idx].clone()

    # How different from deterministic top-K?
    _, topk_idx = torch.topk(log_w, n_keep)
    topk_set = set(topk_idx.tolist())
    gumbel_set = set(idx.tolist())
    overlap = len(topk_set & gumbel_set) / n_keep

    # Log|Ψ| statistics for selected points
    log_psi_sel = log_psi[idx]

    return X, {
        "method": "gumbel",
        "n_unique": n_keep,
        "unique_frac": 1.0,
        "topk_overlap": overlap,
        "logpsi_mean": log_psi_sel.mean().item(),
        "logpsi_min": log_psi_sel.min().item(),
        "logpsi_std": log_psi_sel.std().item(),
        "beta_temp": beta_temp,
    }


# ══════════════════════════════════════════════════════════════════
#  Det-conditioning filter
# ══════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_log_det_slater(bf_net, x, C_occ, params, spin):
    """
    Compute log|det S̃_up| + log|det S̃_down| for screening.
    Uses the same orbital/Slater machinery as psi_fn.
    """
    B, N, d = x.shape
    n_spin = N // 2

    # Backflow shift
    dx = bf_net(x, spin)
    x_eff = x + dx

    # Evaluate basis functions
    nx, ny = params["nx"], params["ny"]
    Phi = evaluate_basis_functions_torch_batch_2d(x_eff, nx, ny)  # (B, N, n_basis)

    C = C_occ.to(device=x.device, dtype=x.dtype)
    Psi_all = torch.matmul(Phi, C)  # (B, N, n_occ)

    # Split by spin
    if spin.ndim == 1:
        spin_vec = spin.long()
    else:
        spin_vec = spin[0].long()
    idx_up = torch.nonzero(spin_vec == 0, as_tuple=False).squeeze(-1)
    idx_down = torch.nonzero(spin_vec == 1, as_tuple=False).squeeze(-1)

    Psi_up = Psi_all[:, idx_up, :]  # (B, n_spin, n_occ)
    Psi_down = Psi_all[:, idx_down, :]  # (B, n_spin, n_occ)

    _, log_u = torch.linalg.slogdet(Psi_up)
    _, log_d = torch.linalg.slogdet(Psi_down)

    return log_u + log_d  # (B,)  log|det S̃|


def det_filter(bf_net, X, C_occ, params, spin, exclude_frac=0.03, batch_size=512):
    """
    Remove the most ill-conditioned configurations (smallest |det S̃|).
    Returns filtered X and diagnostics.
    """
    B = X.shape[0]
    log_dets = []
    for i in range(0, B, batch_size):
        ld = compute_log_det_slater(bf_net, X[i : i + batch_size], C_occ, params, spin)
        log_dets.append(ld)
    log_det = torch.cat(log_dets)

    # Exclude bottom exclude_frac by log|det|
    threshold = torch.quantile(log_det, exclude_frac)
    keep = log_det >= threshold
    n_removed = (~keep).sum().item()

    return X[keep], {
        "n_removed": n_removed,
        "remove_frac": n_removed / B,
        "logdet_min": log_det.min().item(),
        "logdet_threshold": threshold.item(),
        "logdet_median": log_det.median().item(),
    }


# ══════════════════════════════════════════════════════════════════
#  Hybrid loss function
# ══════════════════════════════════════════════════════════════════


def compute_hybrid_loss(E_L, beta_loss, E_ref, huber_delta=0.5, trim=0.02):
    """
    Hybrid loss: β·Huber(E_L - E_ref) + (1-β)·TrimmedMean(E_L)

    β (beta_loss):
      - high (→1): variance-like, stabilizing, but BF-hostile
      - low (→0): mean-energy, coherent, BF-friendly
      - anneal from high to low during training

    E_ref: stop-gradient'd reference energy

    The mean-energy gradient ∝ mean(∂E_L/∂θ) is a coherent "lower energy"
    signal — it doesn't quadratically amplify near-node outliers.
    """
    # Variance-like term: Huber(E_L - E_ref)
    resid = E_L - E_ref
    var_term = nn.functional.huber_loss(resid, torch.zeros_like(resid), delta=huber_delta)

    # Mean-energy term: trimmed mean of E_L
    # (trimming for robustness against extreme outliers)
    n_trim = int(trim * E_L.numel())
    if n_trim > 0 and E_L.numel() > 2 * n_trim + 10:
        sorted_EL, _ = E_L.sort()
        mean_term = sorted_EL[n_trim:-n_trim].mean()
    else:
        mean_term = E_L.mean()

    return beta_loss * var_term + (1 - beta_loss) * mean_term


def compute_standard_loss(E_L, E_ref, huber_delta=0.5):
    """Standard Huber-variance loss (baseline)."""
    resid = E_L - E_ref
    return nn.functional.huber_loss(resid, torch.zeros_like(resid), delta=huber_delta)


# ══════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════


def train_hybrid(
    f_net,
    bf_net,
    C_occ,
    params,
    *,
    n_epochs=150,
    lr=3e-4,
    lr_min_frac=0.02,
    # Sampling
    n_collocation=2048,
    oversampling=10,
    sigma=None,
    use_gumbel=False,
    gumbel_beta=0.5,
    # Det filter
    use_det_filter=False,
    det_exclude_frac=0.03,
    # Loss
    use_hybrid_loss=True,
    beta_loss_start=0.7,  # variance weight at start
    beta_loss_end=0.15,  # variance weight at end (more mean-energy)
    beta_loss_fixed=None,  # if set, override annealing
    huber_delta=0.5,
    # E_ref schedule
    phase1_frac=0.20,
    alpha_eref_end=0.60,  # how much to anchor E_ref toward E_DMC
    # Smoothness
    smooth_lambda=1e-3,
    smooth_n_samples=32,
    # Robustness
    micro_batch=256,
    grad_clip=0.5,
    quantile_trim=0.02,
    patience=60,
    print_every=10,
    label="",
):
    omega = OMEGA
    if sigma is None:
        sigma = 1.3 * ELL
    psi_log_fn, spin = make_psi_log_fn(f_net, bf_net, C_occ, params)

    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_min = lr * lr_min_frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (
            lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))
        )
        / lr,
    )
    phase1_end = int(phase1_frac * n_epochs)

    n_f = sum(p.numel() for p in f_net.parameters())
    n_bf = sum(p.numel() for p in bf_net.parameters())
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, f={n_f:,} bf={n_bf:,}, lr={lr}->{lr_min:.1e}")
    print(f"  hybrid_loss={use_hybrid_loss}, beta_loss={beta_loss_start}->{beta_loss_end}")
    if beta_loss_fixed is not None:
        print(f"  beta_loss FIXED at {beta_loss_fixed}")
    print(f"  gumbel={use_gumbel} (beta_temp={gumbel_beta})")
    print(f"  det_filter={use_det_filter} (excl={det_exclude_frac})")
    print(f"  smooth_lambda={smooth_lambda:.1e}, Huber delta={huber_delta}")
    print(f"{'='*70}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    diagnostics_log = []

    for epoch in range(n_epochs):
        # ── E_ref schedule: ramp toward E_DMC ──
        if epoch < phase1_end:
            alpha_eref = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha_eref = 0.5 * alpha_eref_end * (1 - math.cos(math.pi * t2))

        # ── Beta_loss schedule: anneal variance weight ──
        if beta_loss_fixed is not None:
            beta_loss = beta_loss_fixed
        else:
            # Cosine anneal from beta_loss_start to beta_loss_end
            t_beta = epoch / max(1, n_epochs - 1)
            beta_loss = beta_loss_end + 0.5 * (beta_loss_start - beta_loss_end) * (
                1 + math.cos(math.pi * t_beta)
            )

        # ── Sample points ──
        f_net.eval()
        bf_net.eval()
        if use_gumbel:
            X, samp_diag = gumbel_topk_collocation(
                psi_log_fn, n_collocation, oversampling, sigma, beta_temp=gumbel_beta
            )
        else:
            X, samp_diag = topk_collocation(psi_log_fn, n_collocation, oversampling, sigma)

        # ── Det-conditioning filter ──
        det_diag = {}
        if use_det_filter:
            X, det_diag = det_filter(bf_net, X, C_occ, params, spin, exclude_frac=det_exclude_frac)

        n_pts = X.shape[0]

        # ── Train step ──
        f_net.train()
        bf_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        n_batches = max(1, math.ceil(n_pts / micro_batch))

        for i in range(0, n_pts, micro_batch):
            x_mb = X[i : i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega).view(-1)

            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue

            # Quantile trim
            if quantile_trim > 0 and E_L.numel() > 20:
                lo = torch.quantile(E_L.detach(), quantile_trim)
                hi = torch.quantile(E_L.detach(), 1.0 - quantile_trim)
                mask = (E_L.detach() >= lo) & (E_L.detach() <= hi)
                E_L = E_L[mask]
                if E_L.numel() == 0:
                    continue

            all_EL.append(E_L.detach())

            # Compute E_ref
            mu = E_L.mean().detach()
            E_ref = alpha_eref * E_DMC + (1.0 - alpha_eref) * mu

            # Loss
            if use_hybrid_loss:
                loss_mb = compute_hybrid_loss(
                    E_L, beta_loss, E_ref, huber_delta=huber_delta, trim=quantile_trim
                )
            else:
                loss_mb = compute_standard_loss(E_L, E_ref, huber_delta=huber_delta)

            # Smoothness penalty
            if smooth_lambda > 0:
                pen = compute_bf_smoothness_penalty(bf_net, x_mb, spin, n_samples=smooth_n_samples)
                loss_mb = loss_mb + smooth_lambda * pen

            (loss_mb / n_batches).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Logging ──
        if len(all_EL) > 0:
            EL_cat = torch.cat(all_EL)
            E_mean = EL_cat.mean().item()
            E_var = EL_cat.var().item()
            E_std = EL_cat.std().item()
        else:
            E_mean, E_var, E_std = float("nan"), float("nan"), float("nan")

        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_f_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            best_bf_state = {k: v.clone() for k, v in bf_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience and epoch > phase1_end + 30:
            print(f"  Early stop at epoch {epoch}  (best var={best_var:.3e})")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / E_DMC * 100

            with torch.no_grad():
                bf_net.eval()
                dx_s = bf_net(X[: min(64, n_pts)], spin)
                bf_mag = dx_s.norm(dim=-1).mean().item()
                bf_net.train()

            extra = ""
            if "topk_overlap" in samp_diag:
                extra += f"  topK_olap={samp_diag['topk_overlap']:.0%}"
            if det_diag.get("n_removed", 0) > 0:
                extra += f"  det_rm={det_diag['n_removed']}"

            print(
                f"[{epoch:4d}] E={E_mean:.5f} +/- {E_std:.4f}  "
                f"var={E_var:.3e}  beta_L={beta_loss:.2f}  "
                f"alpha_E={alpha_eref:.2f}  lr={cur_lr:.1e}  "
                f"|dx|={bf_mag:.3f}  err={err:.2f}%{extra}"
            )
            sys.stdout.flush()

        if epoch % 10 == 0:
            entry = {
                "epoch": epoch,
                "E_mean": E_mean,
                "E_var": E_var,
                "beta_loss": beta_loss,
                **samp_diag,
                **det_diag,
            }
            diagnostics_log.append(entry)

    # Restore best
    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)

    total = time.time() - t0
    print(f"  Best var={best_var:.3e}, {total:.0f}s ({total/60:.1f}min)")
    return f_net, bf_net, diagnostics_log


# ══════════════════════════════════════════════════════════════════
#  Experiments
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    C_occ, params = setup_noninteracting(N_PARTICLES, OMEGA, device=DEVICE, dtype=DTYPE)
    results = {}
    N_EPOCHS = 150

    # ── Experiment 1: Hybrid loss + standard top-K ──
    # The key test: does mean-energy term help backflow?
    # β_loss anneals 0.7 → 0.15 (starts variance-like, shifts to mean-energy)
    print(f"\n{'='*70}")
    print(f"# Exp 1: HYBRID loss (anneal) + top-K  [{N_EPOCHS}ep from scratch]")
    print(f"{'='*70}")

    f1, bf1 = make_nets()
    f1, bf1, diag1 = train_hybrid(
        f1,
        bf1,
        C_occ,
        params,
        n_epochs=N_EPOCHS,
        use_hybrid_loss=True,
        beta_loss_start=0.7,
        beta_loss_end=0.15,
        use_gumbel=False,
        use_det_filter=False,
        smooth_lambda=1e-3,
        huber_delta=0.5,
        label="hybrid_anneal: beta 0.7->0.15 + top-K",
    )
    save_model(f1, bf1, "hybrid_anneal")
    r1 = evaluate(f1, C_occ, params, backflow_net=bf1, label="hybrid_anneal")
    E1, std1 = r1["E_mean"], r1["E_stderr"]
    err1 = abs(E1 - E_DMC) / E_DMC * 100
    print(f"  >>> err = {err1:.2f}%")
    results["hybrid_anneal"] = (E1, std1, err1)

    # ── Experiment 2: Hybrid loss + Gumbel-top-K ──
    # Combines both ideas: better loss + better sampling
    # Gumbel beta_temp=0.5 gives moderate exploration
    print(f"\n{'='*70}")
    print(f"# Exp 2: HYBRID loss + GUMBEL-top-K  [{N_EPOCHS}ep from scratch]")
    print(f"{'='*70}")

    f2, bf2 = make_nets()
    f2, bf2, diag2 = train_hybrid(
        f2,
        bf2,
        C_occ,
        params,
        n_epochs=N_EPOCHS,
        use_hybrid_loss=True,
        beta_loss_start=0.7,
        beta_loss_end=0.15,
        use_gumbel=True,
        gumbel_beta=0.5,  # moderate exploration
        use_det_filter=False,
        smooth_lambda=1e-3,
        huber_delta=0.5,
        label="gumbel_hybrid: beta_loss 0.7->0.15 + Gumbel beta_temp=0.5",
    )
    save_model(f2, bf2, "gumbel_hybrid")
    r2 = evaluate(f2, C_occ, params, backflow_net=bf2, label="gumbel_hybrid")
    E2, std2 = r2["E_mean"], r2["E_stderr"]
    err2 = abs(E2 - E_DMC) / E_DMC * 100
    print(f"  >>> err = {err2:.2f}%")
    results["gumbel_hybrid"] = (E2, std2, err2)

    # ── Experiment 3: Hybrid (strong mean-energy) + det filter ──
    # β_loss=0.2 fixed: strong mean-energy from the start
    # Plus det-conditioning filter (exclude bottom 3% by |det S̃|)
    print(f"\n{'='*70}")
    print(f"# Exp 3: HYBRID strong (beta=0.2) + det filter  [{N_EPOCHS}ep from scratch]")
    print(f"{'='*70}")

    f3, bf3 = make_nets()
    f3, bf3, diag3 = train_hybrid(
        f3,
        bf3,
        C_occ,
        params,
        n_epochs=N_EPOCHS,
        use_hybrid_loss=True,
        beta_loss_fixed=0.2,  # strong mean-energy throughout
        use_gumbel=False,
        use_det_filter=True,
        det_exclude_frac=0.03,
        smooth_lambda=1e-3,
        huber_delta=0.5,
        label="hybrid_strong_detfilt: beta_loss=0.2 + det_filter 3%",
    )
    save_model(f3, bf3, "hybrid_strong_detfilt")
    r3 = evaluate(f3, C_occ, params, backflow_net=bf3, label="hybrid_strong_detfilt")
    E3, std3 = r3["E_mean"], r3["E_stderr"]
    err3 = abs(E3 - E_DMC) / E_DMC * 100
    print(f"  >>> err = {err3:.2f}%")
    results["hybrid_strong_detfilt"] = (E3, std3, err3)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY -- Hybrid loss experiments")
    print(f"{'='*70}")

    # Reference results from previous runs
    results["bf_0.7 joint 300ep (ref)"] = (11.823253, 0.002982, 0.33)
    results["topk_baseline 300ep (SIR)"] = (11.826500, 0.003000, 0.35)
    results["hardsmooth_scratch (ref)"] = (11.828802, 0.002613, 0.38)
    results["PINN only no BF (ref)"] = (11.834691, 0.002782, 0.42)

    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        marker = " <-- NEW" if "hyb" in name.lower() or "gumbel" in name.lower() else ""
        print(f"  {name:40s}  E={E:.6f} +/- {std:.6f}  err={err:.2f}%{marker}")

    print("\nKey question: does mean-energy term let BF contribute more?")
    print("  If hybrid < baseline → mean-energy helps BF avoid shrinkage")
    print("  If hybrid ≈ baseline → loss wasn't the bottleneck")
    print("\nDone.")
