"""
SIR-based residual training: importance resampling instead of top-K screening.

The core argument (from external analysis of our results):

  Top-K screening is a MODE-SEEKING operation that systematically excludes
  near-node and compact configurations where backflow matters most. This is
  NOT importance sampling — it's selection bias that truncates the training
  distribution and creates the train/eval mismatch that limits backflow.

  Fix: replace top-K with Sampling Importance Resampling (SIR). Draw iid
  proposals, compute weights w_m ∝ |Ψ(R_m)|²/q(R_m), then RESAMPLE K
  points from the normalized weight distribution. This gives approximate
  iid samples from |Ψ|² WITHOUT MCMC.

This is still genuinely residual training:
  • No Markov chain (SIR is iid — no burn-in, no autocorrelation)
  • Objective: Var[E_L] (energy variance), NOT ⟨E_L⟩ (VMC energy gradient)
  • Stop-gradient through sampling (no reparameterization)
  • Fresh independent samples each epoch

What's different from standard collocation:
  • Near-node configs (small |Ψ|²) occasionally appear, giving backflow
    gradient signal about *which direction to shift the nodes*
  • All regions of |Ψ|² are represented proportionally, closing the
    train/eval distribution gap

Additionally: mixture proposal with compact + bulk + tail components
to reduce weight degeneracy and improve ESS.

Includes: bf_scale=0.7, smoothness penalty, Huber loss, ESS tracking.
"""

import math, sys, time, os, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, CTNNBackflowNet
from functions.Neural_Networks import psi_fn, _laplacian_logpsi_exact
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"
LOG_DIR = "/Users/aleksandersekkelsten/thesis/results/logs"

N_PARTICLES, DIM, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)


# ══════════════════════════════════════════════════════════════════
#  Model helpers
# ══════════════════════════════════════════════════════════════════

def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
    n_occ = N // 2
    nx = {2: 2, 6: 3, 12: 4, 20: 5}.get(N, 4)
    ny = nx
    n_basis = nx * ny
    L = max(8.0, 3.0 / math.sqrt(omega))
    config.update(
        omega=omega, n_particles=N, d=d,
        L=L, n_grid=80, nx=nx, ny=ny,
        basis="cart", device=str(device), dtype="float64",
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
    f_net = PINN(
        n_particles=N_PARTICLES, d=DIM, omega=OMEGA,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(DEVICE).to(DTYPE)
    bf_net = CTNNBackflowNet(
        d=DIM, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=bf_scale_init,
        zero_init_last=zero_init_last,
        omega=OMEGA,
    ).to(DEVICE).to(DTYPE)
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    up = N_PARTICLES // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N_PARTICLES - up, dtype=torch.long)]).to(DEVICE)
    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net,
                        spin=spin, params=params)
        return lp
    return fn, spin


def compute_local_energy(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap_log.view(B) + g2.view(B))
    V_harm = 0.5 * omega ** 2 * (x ** 2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).view(B)
    return T + V_harm + V_int


def save_model(f_net, bf_net, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"6e_sir_{name}.pt")
    torch.save({"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()}, path)
    print(f"  💾 Saved → {path}")


def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    print(f"\n── VMC eval: {label} ──")
    result = evaluate_energy_vmc(
        f_net, C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=backflow_net, params=params,
        n_samples=n_samples, batch_size=512,
        sampler_steps=50, sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True, sampler_burn_in=300, sampler_thin=3,
        progress=True,
    )
    E, E_std = result["E_mean"], result["E_stderr"]
    err = abs(E - E_DMC) / E_DMC * 100
    print(f"  E = {E:.6f} ± {E_std:.6f}  (target {E_DMC}, err {err:.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════════
#  Systematic resampling  (the key new ingredient)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def systematic_resample(log_weights, n_keep):
    """
    Systematic resampling from log-importance-weights.

    Instead of top-K (mode-seeking, excludes tails and near-node),
    resample proportionally to weights. Near-node configs with small
    |Ψ|² get small but nonzero probability → they occasionally appear
    in the training set.

    Returns (indices, ess):
      indices: (n_keep,) int tensor — indices into the candidate pool
      ess: effective sample size (tracks proposal quality)
    """
    # Stabilize weights in log-space
    log_w = log_weights - log_weights.max()
    w = torch.exp(log_w)
    w_sum = w.sum()
    pi = w / w_sum

    # Effective sample size: ESS = 1 / Σ π_i²
    ess = 1.0 / (pi ** 2).sum()

    # Systematic resampling: equally spaced points + single random offset
    cumsum = torch.cumsum(pi, dim=0)
    u_base = torch.arange(n_keep, dtype=pi.dtype, device=pi.device) / n_keep
    u_offset = torch.rand(1, dtype=pi.dtype, device=pi.device) / n_keep
    u = u_base + u_offset

    indices = torch.searchsorted(cumsum, u)
    indices = indices.clamp(max=len(pi) - 1)

    return indices, ess.item()


# ══════════════════════════════════════════════════════════════════
#  Proposal distributions
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_single_gaussian(n_samples, sigma):
    """Single isotropic Gaussian proposal. Returns (x, log_q)."""
    x = torch.randn(n_samples, N_PARTICLES, DIM, device=DEVICE, dtype=DTYPE) * sigma
    Nd = N_PARTICLES * DIM
    log_q = (
        -0.5 * Nd * math.log(2 * math.pi * sigma ** 2)
        - x.reshape(n_samples, -1).pow(2).sum(-1) / (2 * sigma ** 2)
    )
    return x, log_q


@torch.no_grad()
def sample_mixture_gaussian(n_samples, sigmas, weights):
    """
    Mixture of isotropic Gaussians with different widths.

    sigmas: list of σ values for each component
    weights: list of mixture weights (should sum to 1)

    Returns (x, log_q) where log_q is the log of the full mixture density.

    The mixture targets the three regions our diagnostics identified:
      compact (small σ): close-pair, strongly-interacting configs
      bulk (medium σ): standard sampling region
      tail (large σ): extended configs, rare events
    """
    K = len(sigmas)
    Nd = N_PARTICLES * DIM

    # Decide how many samples from each component
    counts = [int(w * n_samples) for w in weights]
    counts[-1] = n_samples - sum(counts[:-1])  # fix rounding

    # Sample from each component
    x_parts = []
    component_ids = []
    for k, (sigma_k, n_k) in enumerate(zip(sigmas, counts)):
        x_k = torch.randn(n_k, N_PARTICLES, DIM, device=DEVICE, dtype=DTYPE) * sigma_k
        x_parts.append(x_k)
        component_ids.append(torch.full((n_k,), k, dtype=torch.long, device=DEVICE))

    x = torch.cat(x_parts, dim=0)
    component_ids = torch.cat(component_ids)

    # Shuffle to avoid component ordering artifacts
    perm = torch.randperm(n_samples, device=DEVICE)
    x = x[perm]

    # Compute log q(x) = log Σ_k α_k * q_k(x) via logsumexp
    x_flat_sq = x.reshape(n_samples, -1).pow(2).sum(-1)  # |R|²
    log_alpha = torch.log(torch.tensor(weights, dtype=DTYPE, device=DEVICE))

    log_components = []
    for k, sigma_k in enumerate(sigmas):
        log_q_k = (
            -0.5 * Nd * math.log(2 * math.pi * sigma_k ** 2)
            - x_flat_sq / (2 * sigma_k ** 2)
        )
        log_components.append(log_alpha[k] + log_q_k)

    # (n_samples, K) → logsumexp over K
    log_q_stack = torch.stack(log_components, dim=-1)  # (n_samples, K)
    log_q = torch.logsumexp(log_q_stack, dim=-1)       # (n_samples,)

    return x, log_q


# ══════════════════════════════════════════════════════════════════
#  SIR collocation  (replaces screened_collocation)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def sir_collocation(psi_log_fn, n_keep, oversampling, proposal_fn,
                    jitter_sigma=0.0, batch_size=4096):
    """
    Sampling Importance Resampling: approximate iid samples from |Ψ|².

    1. Draw M = oversampling × n_keep proposals from q
    2. Compute log weights = 2 log|Ψ| - log q
    3. Systematic resample K points from normalized weights
    4. Optional jitter to break duplicates

    Returns (X, diagnostics_dict).
    """
    M = oversampling * n_keep

    # Step 1: propose
    x_cand, log_q = proposal_fn(M)

    # Step 2: evaluate log|Ψ| in batches
    log_psi_parts = []
    for i in range(0, M, batch_size):
        lp = psi_log_fn(x_cand[i:i + batch_size])
        log_psi_parts.append(lp)
    log_psi = torch.cat(log_psi_parts)

    # Log importance weights: w ∝ |Ψ|²/q
    log_w = 2.0 * log_psi - log_q

    # Remove non-finite weights
    valid = torch.isfinite(log_w)
    if not valid.all():
        log_w[~valid] = -1e10

    # Step 3: systematic resample
    indices, ess = systematic_resample(log_w, n_keep)

    X = x_cand[indices].clone()

    # Step 4: optional jitter
    if jitter_sigma > 0:
        X = X + torch.randn_like(X) * jitter_sigma

    # Diagnostics
    n_unique = len(torch.unique(indices))

    # Compare with what top-K would have selected
    _, topk_idx = torch.topk(log_w, n_keep)
    log_psi_sir = log_psi[indices]
    log_psi_topk = log_psi[topk_idx]

    # Config features for SIR-selected points
    r2 = (X ** 2).sum(dim=-1).mean(dim=1)                           # (K,)
    diff = X.unsqueeze(2) - X.unsqueeze(1)                          # (K,N,N,d)
    dist = diff.norm(dim=-1)                                         # (K,N,N)
    mask = torch.eye(N_PARTICLES, device=DEVICE, dtype=torch.bool).unsqueeze(0)
    dist_masked = dist.masked_fill(mask, float("inf"))
    dmin = dist_masked.min(dim=-1).values.min(dim=-1).values         # (K,)

    diag = {
        "ess": ess,
        "ess_frac": ess / M,
        "n_unique": n_unique,
        "unique_frac": n_unique / n_keep,
        # Compare |Ψ|² coverage: SIR vs top-K
        "sir_logpsi_mean": log_psi_sir.mean().item(),
        "sir_logpsi_min": log_psi_sir.min().item(),
        "sir_logpsi_std": log_psi_sir.std().item(),
        "topk_logpsi_mean": log_psi_topk.mean().item(),
        "topk_logpsi_min": log_psi_topk.min().item(),
        # Config space coverage
        "sir_mean_r2": r2.mean().item(),
        "sir_min_dmin": dmin.min().item(),
        "sir_mean_dmin": dmin.mean().item(),
    }

    return X, diag


@torch.no_grad()
def topk_collocation(psi_log_fn, n_keep, oversampling, proposal_fn,
                     batch_size=4096):
    """Standard top-K screening (baseline, for comparison)."""
    M = oversampling * n_keep
    x_cand, log_q = proposal_fn(M)

    log_psi_parts = []
    for i in range(0, M, batch_size):
        lp = psi_log_fn(x_cand[i:i + batch_size])
        log_psi_parts.append(lp)
    log_psi = torch.cat(log_psi_parts)

    log_w = 2.0 * log_psi - log_q
    valid = torch.isfinite(log_w)
    if not valid.all():
        log_w[~valid] = -1e10

    _, idx = torch.topk(log_w, n_keep)
    X = x_cand[idx].clone()

    # Diagnostics (for comparison)
    r2 = (X ** 2).sum(dim=-1).mean(dim=1)
    diff = X.unsqueeze(2) - X.unsqueeze(1)
    dist = diff.norm(dim=-1)
    mask = torch.eye(N_PARTICLES, device=DEVICE, dtype=torch.bool).unsqueeze(0)
    dist_masked = dist.masked_fill(mask, float("inf"))
    dmin = dist_masked.min(dim=-1).values.min(dim=-1).values

    diag = {
        "ess": float("nan"),
        "ess_frac": float("nan"),
        "n_unique": n_keep,
        "unique_frac": 1.0,
        "sir_logpsi_mean": log_psi[idx].mean().item(),
        "sir_logpsi_min": log_psi[idx].min().item(),
        "sir_logpsi_std": log_psi[idx].std().item(),
        "topk_logpsi_mean": log_psi[idx].mean().item(),
        "topk_logpsi_min": log_psi[idx].min().item(),
        "sir_mean_r2": r2.mean().item(),
        "sir_min_dmin": dmin.min().item(),
        "sir_mean_dmin": dmin.mean().item(),
    }
    return X, diag


# ══════════════════════════════════════════════════════════════════
#  Smoothness penalty on backflow (proven critical: 0.38% vs 0.59%)
# ══════════════════════════════════════════════════════════════════

def compute_bf_smoothness_penalty(bf_net, x, spin, n_samples=32):
    """‖∇²Δx‖² via Hutchinson trace estimator."""
    x_sub = x[:n_samples].detach().requires_grad_(True)
    B, N, d = x_sub.shape
    dx = bf_net(x_sub, spin)
    n_probes = 2
    lap_sq_sum = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    for _ in range(n_probes):
        v = torch.empty_like(x_sub).bernoulli_(0.5).mul_(2).add_(-1)
        for k in range(d):
            dx_k_sum = dx[:, :, k].sum()
            grad1 = torch.autograd.grad(
                dx_k_sum, x_sub, create_graph=True, retain_graph=True
            )[0]
            Hv = torch.autograd.grad(
                (grad1 * v).sum(), x_sub, create_graph=True, retain_graph=True
            )[0]
            vTHv = (v * Hv).sum(dim=(1, 2))
            lap_sq_sum = lap_sq_sum + (vTHv ** 2).mean()

    return lap_sq_sum / (n_probes * d)


# ══════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════

def train_sir(
    f_net, bf_net, C_occ, params, *,
    n_epochs=300,
    lr=3e-4,
    lr_min_frac=0.02,
    # Sampling
    n_collocation=2048,
    oversampling=10,
    collocation_fn,          # sir_collocation or topk_collocation
    proposal_fn,             # returns (x, log_q) given n_samples
    jitter_sigma=0.0,
    # Schedule
    phase1_frac=0.25,
    alpha_end=0.60,
    # Loss
    huber_delta=0.5,
    # Smoothness
    smooth_lambda=1e-3,
    smooth_n_samples=32,
    # Robustness
    micro_batch=256,
    grad_clip=0.5,
    quantile_trim=0.02,
    patience=60,
    print_every=10,
    diag_every=10,
    label="",
):
    omega = OMEGA
    psi_log_fn, spin = make_psi_log_fn(f_net, bf_net, C_occ, params)

    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_min = lr * lr_min_frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (lr_min + 0.5 * (lr - lr_min) *
                              (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))) / lr,
    )
    phase1_end = int(phase1_frac * n_epochs)

    n_f = sum(p.numel() for p in f_net.parameters())
    n_bf = sum(p.numel() for p in bf_net.parameters())
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, f={n_f:,} bf={n_bf:,}, lr={lr}→{lr_min:.1e}")
    print(f"  smooth_λ={smooth_lambda:.1e}, Huber δ={huber_delta}")
    print(f"  collocation: n={n_collocation}, OS={oversampling}")
    print(f"{'='*65}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    diagnostics_log = []

    for epoch in range(n_epochs):
        # ── Alpha schedule: ramp E_ref toward E_DMC ──
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # ── Sample: SIR or top-K ──
        f_net.eval(); bf_net.eval()
        if collocation_fn == sir_collocation:
            X, diag = collocation_fn(
                psi_log_fn, n_collocation, oversampling, proposal_fn,
                jitter_sigma=jitter_sigma,
            )
        else:
            X, diag = collocation_fn(
                psi_log_fn, n_collocation, oversampling, proposal_fn,
            )
        n_pts = X.shape[0]

        # ── Train step ──
        f_net.train(); bf_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        n_batches = max(1, math.ceil(n_pts / micro_batch))

        for i in range(0, n_pts, micro_batch):
            x_mb = X[i:i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega).view(-1)

            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue

            if quantile_trim > 0 and E_L.numel() > 20:
                lo = torch.quantile(E_L.detach(), quantile_trim)
                hi = torch.quantile(E_L.detach(), 1.0 - quantile_trim)
                mask = (E_L.detach() >= lo) & (E_L.detach() <= hi)
                E_L = E_L[mask]
                if E_L.numel() == 0:
                    continue

            all_EL.append(E_L.detach())
            mu = E_L.mean().detach()
            E_eff = alpha * E_DMC + (1.0 - alpha) * mu
            resid = E_L - E_eff

            loss_mb = nn.functional.huber_loss(
                resid, torch.zeros_like(resid), delta=huber_delta)

            if smooth_lambda > 0:
                pen = compute_bf_smoothness_penalty(
                    bf_net, x_mb, spin, n_samples=smooth_n_samples)
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

        # Save diagnostics
        if epoch % diag_every == 0:
            entry = {"epoch": epoch, **diag,
                     "E_mean": E_mean, "E_var": E_var}
            diagnostics_log.append(entry)

        if epoch % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / E_DMC * 100

            with torch.no_grad():
                bf_net.eval()
                dx_s = bf_net(X[:64], spin)
                bf_mag = dx_s.norm(dim=-1).mean().item()
                bf_net.train()

            ess_str = f"ESS={diag['ess']:.0f}" if math.isfinite(diag.get('ess', float('nan'))) else "top-K"
            uniq_str = f"uniq={diag['unique_frac']:.0%}"

            phase_tag = " [ph1]" if epoch < phase1_end else ""
            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  lr={cur_lr:.1e}  "
                f"|Δx|={bf_mag:.3f}  err={err:.2f}%  "
                f"{ess_str}  {uniq_str}{phase_tag}"
            )
            sys.stdout.flush()

    # Restore best
    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)

    total = time.time() - t0
    print(f"  Best var={best_var:.3e}, {total:.0f}s ({total/60:.1f}min)")

    return f_net, bf_net, diagnostics_log


# ══════════════════════════════════════════════════════════════════
#  Print diagnostic comparison
# ══════════════════════════════════════════════════════════════════

def print_sir_diagnostics(diagnostics_log, label):
    if not diagnostics_log:
        return
    print(f"\n{'─'*75}")
    print(f"  {label} — sampling diagnostics")
    print(f"{'─'*75}")
    print(f"  {'ep':>4s}  {'ESS':>6s}  {'ESS%':>5s}  "
          f"{'uniq%':>5s}  {'⟨r²⟩':>6s}  {'d_min':>6s}  "
          f"{'log|Ψ|':>7s}  {'topK_lΨ':>7s}  {'var':>9s}")

    for d in diagnostics_log:
        ess_str = f"{d['ess']:.0f}" if math.isfinite(d.get('ess', float('nan'))) else "──"
        ess_pct = f"{d['ess_frac']:.1%}" if math.isfinite(d.get('ess_frac', float('nan'))) else "──"
        print(
            f"  {d['epoch']:4d}  {ess_str:>6s}  {ess_pct:>5s}  "
            f"{d['unique_frac']:5.0%}  {d['sir_mean_r2']:6.2f}  "
            f"{d['sir_mean_dmin']:6.3f}  "
            f"{d['sir_logpsi_mean']:7.2f}  {d['topk_logpsi_mean']:7.2f}  "
            f"{d['E_var']:9.3e}"
        )

    # Summary
    ess_vals = [d['ess'] for d in diagnostics_log if math.isfinite(d.get('ess', float('nan')))]
    if ess_vals:
        print(f"\n  ESS: min={min(ess_vals):.0f}, max={max(ess_vals):.0f}, "
              f"mean={sum(ess_vals)/len(ess_vals):.0f}  "
              f"(out of {diagnostics_log[0].get('ess', 0) / max(diagnostics_log[0].get('ess_frac', 1), 1e-10):.0f} candidates)")
        mean_uniq = sum(d['unique_frac'] for d in diagnostics_log) / len(diagnostics_log)
        print(f"  Unique fraction: mean={mean_uniq:.0%}")


# ══════════════════════════════════════════════════════════════════
#  Experiments
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    C_occ, params = setup_noninteracting(N_PARTICLES, OMEGA, device=DEVICE, dtype=DTYPE)
    results = {}

    sigma_bulk = 1.3 * ELL

    # ── Experiment 1: Standard top-K baseline ──
    # (same as bf_0.7 joint, for fair comparison)
    print(f"\n{'═'*65}")
    print(f"# Exp 1: Top-K screening baseline (300ep from scratch)")
    print(f"{'═'*65}")

    f1, bf1 = make_nets()
    proposal_single = lambda n: sample_single_gaussian(n, sigma_bulk)

    f1, bf1, diag1 = train_sir(
        f1, bf1, C_occ, params,
        n_epochs=300,
        collocation_fn=topk_collocation,
        proposal_fn=proposal_single,
        smooth_lambda=1e-3,
        huber_delta=0.5,
        label="Top-K baseline (bf_0.7, smooth, Huber)",
    )
    save_model(f1, bf1, "topk_baseline")
    print_sir_diagnostics(diag1, "Top-K baseline")

    r1 = evaluate(f1, C_occ, params, backflow_net=bf1, label="topk_baseline")
    E1, std1 = r1["E_mean"], r1["E_stderr"]
    err1 = abs(E1 - E_DMC) / E_DMC * 100
    print(f"  ▸ err = {err1:.2f}%")
    results["topk_baseline"] = (E1, std1, err1)

    # ── Experiment 2: SIR with single Gaussian proposal ──
    # Tests resampling alone (same proposal, just SIR instead of top-K)
    print(f"\n{'═'*65}")
    print(f"# Exp 2: SIR resampling, single Gaussian (300ep from scratch)")
    print(f"{'═'*65}")

    f2, bf2 = make_nets()

    f2, bf2, diag2 = train_sir(
        f2, bf2, C_occ, params,
        n_epochs=300,
        collocation_fn=sir_collocation,
        proposal_fn=proposal_single,
        jitter_sigma=0.01 * ELL,     # small jitter to break duplicates
        smooth_lambda=1e-3,
        huber_delta=0.5,
        label="SIR resampling, single Gaussian σ=1.3ℓ",
    )
    save_model(f2, bf2, "sir_single")
    print_sir_diagnostics(diag2, "SIR single-Gaussian")

    r2 = evaluate(f2, C_occ, params, backflow_net=bf2, label="sir_single")
    E2, std2 = r2["E_mean"], r2["E_stderr"]
    err2 = abs(E2 - E_DMC) / E_DMC * 100
    print(f"  ▸ err = {err2:.2f}%")
    results["sir_single"] = (E2, std2, err2)

    # ── Experiment 3: SIR with mixture proposal ──
    # Three components targeting the regions our diagnostics identified:
    #   compact (σ=0.7ℓ): close pairs, strong interactions
    #   bulk    (σ=1.3ℓ): standard sampling region
    #   tail    (σ=2.5ℓ): extended configs, rare events
    print(f"\n{'═'*65}")
    print(f"# Exp 3: SIR resampling, mixture proposal (300ep from scratch)")
    print(f"{'═'*65}")

    f3, bf3 = make_nets()
    sigma_compact = 0.7 * ELL
    sigma_tail = 2.5 * ELL
    mix_sigmas = [sigma_compact, sigma_bulk, sigma_tail]
    mix_weights = [0.2, 0.6, 0.2]
    proposal_mix = lambda n: sample_mixture_gaussian(n, mix_sigmas, mix_weights)

    f3, bf3, diag3 = train_sir(
        f3, bf3, C_occ, params,
        n_epochs=300,
        collocation_fn=sir_collocation,
        proposal_fn=proposal_mix,
        jitter_sigma=0.01 * ELL,
        smooth_lambda=1e-3,
        huber_delta=0.5,
        label="SIR resampling, mixture (compact+bulk+tail)",
    )
    save_model(f3, bf3, "sir_mixture")
    print_sir_diagnostics(diag3, "SIR mixture")

    r3 = evaluate(f3, C_occ, params, backflow_net=bf3, label="sir_mixture")
    E3, std3 = r3["E_mean"], r3["E_stderr"]
    err3 = abs(E3 - E_DMC) / E_DMC * 100
    print(f"  ▸ err = {err3:.2f}%")
    results["sir_mixture"] = (E3, std3, err3)

    # ── Summary ──
    print(f"\n{'═'*65}")
    print(f"SUMMARY — SIR vs Top-K collocation")
    print(f"{'═'*65}")

    # Reference results
    results["bf_0.7 joint (prev ref)"] = (11.823253, 0.002982, 0.33)
    results["hardsmooth_300 (prev best)"] = (11.828802, 0.002613, 0.37)
    results["PINN only (no BF)"] = (11.834691, 0.002782, 0.42)

    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        print(f"  {name:35s}  E={E:.6f} ± {std:.6f}  err={err:.2f}%")

    # Save diagnostics
    os.makedirs(LOG_DIR, exist_ok=True)
    for lbl, diag_data in [("topk_baseline", diag1),
                            ("sir_single", diag2),
                            ("sir_mixture", diag3)]:
        path = os.path.join(LOG_DIR, f"diag_sir_{lbl}.json")
        with open(path, "w") as fp:
            json.dump(diag_data, fp, indent=2)
        print(f"  Diagnostics → {path}")
