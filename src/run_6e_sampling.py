"""
Sampling-strategy experiments for 6e quantum dot (ω=0.5).

Core hypothesis:
  The distribution mismatch between screened-collocation training points
  and the |Ψ|² distribution (where VMC evaluates) is the real bottleneck.
  Better sampling → better VMC energy, even with the same optimizer.

Strategy: Start from TRAINED bf_0.7 models. Short fine-tuning runs (50–80 ep)
  to test which sampling strategy improves the VMC energy.
  Then scale up the winners.

Experiments:
  A. baseline_retrain   — screened collocation (current), 50ep from scratch → reference
  B. mcmc_psi2          — Train on MCMC samples from |Ψ|² directly
  C. mcmc_psi2_finetune — Fine-tune trained model on MCMC |Ψ|² samples
  D. hard_mining        — Find high-|E_L - <E_L>| samples, oversample those regions
  E. mixed_screened_mcmc — 50% screened collocation + 50% MCMC |Ψ|²
  F. adaptive_sigma     — Screened collocation but with wider proposal (σ=2.0*ℓ)
  G. mcmc_varmin_only   — MCMC |Ψ|² with pure variance minimization (no E_DMC targeting)
"""

import math, sys, time, copy, os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, CTNNBackflowNet
from functions.Neural_Networks import (
    psi_fn,
    _laplacian_logpsi_exact,
)
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc

from run_6e_residual import (
    setup_noninteracting,
    compute_local_energy,
    screened_collocation,
    sample_gaussian_proposal,
    evaluate,
)

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"


# ══════════════════════════════════════════════════════════════════
#  Model construction & loading
# ══════════════════════════════════════════════════════════════════

def make_nets(bf_scale_init=0.7, zero_init_last=False):
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(DEVICE).to(DTYPE)
    bf_net = CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=bf_scale_init,
        zero_init_last=zero_init_last,
        omega=0.5,
    ).to(DEVICE).to(DTYPE)
    return f_net, bf_net


def save_model(f_net, bf_net, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"6e_sampling_{name}.pt")
    torch.save({"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()}, path)
    print(f"  💾 Saved → {path}")


def load_model(name, prefix="6e_sampling_"):
    path = os.path.join(CKPT_DIR, f"{prefix}{name}.pt")
    if not os.path.exists(path):
        return None
    f_net, bf_net = make_nets()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    f_net.load_state_dict(ckpt["f_net"])
    bf_net.load_state_dict(ckpt["bf_net"])
    print(f"  📦 Loaded ← {path}")
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    N = params["n_particles"]
    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(DEVICE)
    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net,
                        spin=spin, params=params)
        return lp
    return fn, spin


# ══════════════════════════════════════════════════════════════════
#  MCMC sampler (persistent Metropolis targeting |Ψ|²)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def mcmc_sample_psi2(psi_log_fn, walkers, n_steps, step_sigma,
                     target_accept=0.50, adapt_lr=0.03):
    """
    Persistent random-walk Metropolis targeting |Ψ|².
    walkers: (B, N, d) — current walker positions (mutated in-place).
    Returns updated walkers.
    """
    x = walkers
    sigma = step_sigma
    lp = psi_log_fn(x) * 2.0  # log|Ψ|²

    total_acc = 0.0
    for _ in range(n_steps):
        prop = x + torch.randn_like(x) * sigma
        lp_prop = psi_log_fn(prop) * 2.0
        log_u = torch.rand(x.shape[0], device=x.device, dtype=x.dtype).log()
        acc_mask = (log_u < (lp_prop - lp)).view(-1, 1, 1)
        x = torch.where(acc_mask, prop, x)
        lp = torch.where(acc_mask.view(-1), lp_prop, lp)
        total_acc += acc_mask.float().mean().item()

        # Adapt sigma
        a_hat = acc_mask.float().mean().item()
        sigma = sigma * math.exp(adapt_lr * (a_hat - target_accept))

    avg_acc = total_acc / max(1, n_steps)
    return x, sigma, avg_acc


@torch.no_grad()
def init_walkers(psi_log_fn, n_walkers, N, d, omega):
    """Initialize MCMC walkers via screened collocation (good starting positions)."""
    ell = 1.0 / math.sqrt(omega)
    sigma = 1.3 * ell
    # Use screened collocation to get good initial positions
    x = screened_collocation(
        psi_log_fn, N, d, sigma,
        n_keep=n_walkers, oversampling=10,
        device=DEVICE, dtype=DTYPE,
    )
    return x


# ══════════════════════════════════════════════════════════════════
#  Hard-example mining
# ══════════════════════════════════════════════════════════════════

def find_hard_examples(psi_log_fn, walkers, omega, n_hard, micro_batch=128):
    """
    From current walkers, compute |E_L - <E_L>| and return the hardest ones.
    These are configurations where the wavefunction is least accurate.
    NOTE: compute_local_energy requires grad tracking — no @torch.no_grad() here!
    """
    all_EL = []
    for i in range(0, walkers.shape[0], micro_batch):
        x_mb = walkers[i:i + micro_batch].detach().requires_grad_(True)
        EL = compute_local_energy(psi_log_fn, x_mb, omega).view(-1).detach()
        all_EL.append(EL)
    EL_cat = torch.cat(all_EL)
    good = torch.isfinite(EL_cat)
    if good.any():
        EL_cat[~good] = EL_cat[good].mean()
    else:
        EL_cat[~good] = 0.0

    median = EL_cat.median()
    deviations = (EL_cat - median).abs()

    # Top-K hardest
    _, hard_idx = torch.topk(deviations, min(n_hard, walkers.shape[0]))
    return walkers[hard_idx].clone(), EL_cat, deviations


# ══════════════════════════════════════════════════════════════════
#  Unified trainer with pluggable sampling
# ══════════════════════════════════════════════════════════════════

def train_with_sampling(
    f_net, bf_net, C_occ, params, *,
    sample_fn,           # sample_fn(psi_log_fn, epoch, state) -> (X, state)
    n_epochs=50,
    lr=1e-4,
    lr_min_frac=0.1,
    alpha_schedule="varmin",  # "varmin" | "cosine" | "fixed"
    alpha_fixed=0.0,
    alpha_end=0.60,
    phase1_frac=0.25,
    micro_batch=256,
    grad_clip=0.5,
    quantile_trim=0.02,
    print_every=5,
    label="",
    patience=30,
    loss_type="mse",       # "mse" | "huber"
    huber_delta=1.0,
):
    """
    Unified trainer: the sampling strategy is fully controlled by `sample_fn`.
    sample_fn(psi_log_fn, epoch, state_dict) -> (X: Tensor(B,N,d), state_dict)
    """
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])

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

    n_p = sum(p.numel() for p in all_params)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, {n_p:,} params, lr={lr}→{lr_min:.1e}")
    print(f"  alpha: {alpha_schedule}, loss: {loss_type}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    sample_state = {}  # persistent state for sampler (e.g. MCMC walkers)

    for epoch in range(n_epochs):
        # ── Alpha schedule ──
        if alpha_schedule == "varmin":
            alpha = 0.0
        elif alpha_schedule == "fixed":
            alpha = alpha_fixed
        elif alpha_schedule == "cosine":
            if epoch < phase1_end:
                alpha = 0.0
            else:
                t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
                alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))
        else:
            alpha = 0.0

        # ── Sample training points ──
        f_net.eval(); bf_net.eval()
        X, sample_state = sample_fn(psi_log_fn, epoch, sample_state)
        n_pts = X.shape[0]

        # ── Compute loss ──
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

            if loss_type == "huber":
                loss_mb = nn.functional.huber_loss(resid, torch.zeros_like(resid),
                                                   delta=huber_delta)
            else:
                loss_mb = (resid ** 2).mean()

            (loss_mb / n_batches).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Logging ──
        if len(all_EL) > 0:
            EL_cat = torch.cat(all_EL)
            E_mean = EL_cat.mean().item()
            E_var  = EL_cat.var().item()
            E_std  = EL_cat.std().item()
        else:
            E_mean, E_var, E_std = float("nan"), float("nan"), float("nan")

        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_f_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            best_bf_state = {k: v.clone() for k, v in bf_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience and epoch > 15:
            print(f"  Early stop at epoch {epoch}  (best var={best_var:.3e})")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / E_DMC * 100

            with torch.no_grad():
                bf_net.eval()
                dx_s = bf_net(X[:64], spin)
                bf_mag = dx_s.norm(dim=-1).mean().item()
                bf_net.train()

            extra = ""
            if "accept_rate" in sample_state:
                extra = f"  acc={sample_state['accept_rate']:.2f}"

            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  lr={cur_lr:.1e}  "
                f"|Δx|={bf_mag:.3f}  err={err:.2f}%{extra}"
            )
            sys.stdout.flush()

    # Restore best
    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)

    total = time.time() - t0
    print(f"  Best var={best_var:.3e}, {total:.0f}s ({total/60:.1f}min)")
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  Sampling strategies (pluggable into train_with_sampling)
# ══════════════════════════════════════════════════════════════════

def make_screened_sampler(N, d, omega, n_keep=2048, oversampling=10, sigma_factor=1.3):
    """Standard screened collocation from Gaussian proposal."""
    ell = 1.0 / math.sqrt(omega)
    sigma = sigma_factor * ell
    def sample_fn(psi_log_fn, epoch, state):
        X = screened_collocation(
            psi_log_fn, N, d, sigma,
            n_keep=n_keep, oversampling=oversampling,
            device=DEVICE, dtype=DTYPE,
        )
        return X, state
    return sample_fn


def make_mcmc_sampler(N, d, omega, n_walkers=2048, steps_per_epoch=20,
                      step_sigma_frac=0.12):
    """
    Pure MCMC from |Ψ|² — the golden standard for VMC.
    Walkers are persistent across epochs.
    """
    ell = 1.0 / math.sqrt(omega)
    init_sigma = step_sigma_frac * ell
    def sample_fn(psi_log_fn, epoch, state):
        # Initialize walkers on first call
        if "walkers" not in state:
            state["walkers"] = init_walkers(psi_log_fn, n_walkers, N, d, omega)
            state["sigma"] = init_sigma
            # Burn-in
            print(f"    [MCMC] burn-in 100 steps...")
            state["walkers"], state["sigma"], _ = mcmc_sample_psi2(
                psi_log_fn, state["walkers"], 100, state["sigma"])
        # Evolve walkers
        state["walkers"], state["sigma"], acc = mcmc_sample_psi2(
            psi_log_fn, state["walkers"], steps_per_epoch, state["sigma"])
        state["accept_rate"] = acc
        return state["walkers"].clone(), state
    return sample_fn


def make_hard_mining_sampler(N, d, omega, n_total=2048, hard_frac=0.3,
                             n_walkers=4096, steps_per_epoch=20,
                             step_sigma_frac=0.12):
    """
    Mixed sampling: draw from |Ψ|², find hardest examples (high |E_L - median|),
    oversample the hard region by spawning walkers near them + random-walking.
    """
    ell = 1.0 / math.sqrt(omega)
    init_sigma = step_sigma_frac * ell
    n_hard = int(n_total * hard_frac)
    n_easy = n_total - n_hard

    def sample_fn(psi_log_fn, epoch, state):
        # Initialize walkers
        if "walkers" not in state:
            state["walkers"] = init_walkers(psi_log_fn, n_walkers, N, d, omega)
            state["sigma"] = init_sigma
            print(f"    [HardMine] burn-in 100 steps...")
            state["walkers"], state["sigma"], _ = mcmc_sample_psi2(
                psi_log_fn, state["walkers"], 100, state["sigma"])

        # Evolve walkers
        state["walkers"], state["sigma"], acc = mcmc_sample_psi2(
            psi_log_fn, state["walkers"], steps_per_epoch, state["sigma"])
        state["accept_rate"] = acc

        # Find hard examples
        hard_x, EL_all, devs = find_hard_examples(
            psi_log_fn, state["walkers"], float(omega), n_hard)

        # Easy: random subsample from all walkers
        idx_easy = torch.randperm(state["walkers"].shape[0])[:n_easy]
        easy_x = state["walkers"][idx_easy]

        # Spawn new walkers near hard examples (small perturbation + short MCMC)
        perturbed = hard_x + torch.randn_like(hard_x) * state["sigma"] * 0.5
        hard_evolved, _, _ = mcmc_sample_psi2(
            psi_log_fn, perturbed, 5, state["sigma"] * 0.5)

        X = torch.cat([easy_x, hard_evolved], dim=0)
        return X, state
    return sample_fn


def make_mixed_sampler(N, d, omega, n_total=2048, mcmc_frac=0.5,
                       oversampling=10, sigma_factor=1.3,
                       n_walkers=2048, steps_per_epoch=20,
                       step_sigma_frac=0.12):
    """50% screened collocation + 50% MCMC |Ψ|²."""
    ell = 1.0 / math.sqrt(omega)
    sigma_sc = sigma_factor * ell
    init_sigma = step_sigma_frac * ell
    n_mcmc = int(n_total * mcmc_frac)
    n_sc = n_total - n_mcmc

    def sample_fn(psi_log_fn, epoch, state):
        # Screened collocation portion
        X_sc = screened_collocation(
            psi_log_fn, N, d, sigma_sc,
            n_keep=n_sc, oversampling=oversampling,
            device=DEVICE, dtype=DTYPE,
        )
        # MCMC portion
        if "walkers" not in state:
            state["walkers"] = init_walkers(psi_log_fn, n_walkers, N, d, omega)
            state["sigma"] = init_sigma
            print(f"    [Mixed] MCMC burn-in 100 steps...")
            state["walkers"], state["sigma"], _ = mcmc_sample_psi2(
                psi_log_fn, state["walkers"], 100, state["sigma"])
        state["walkers"], state["sigma"], acc = mcmc_sample_psi2(
            psi_log_fn, state["walkers"], steps_per_epoch, state["sigma"])
        state["accept_rate"] = acc
        idx = torch.randperm(state["walkers"].shape[0])[:n_mcmc]
        X_mcmc = state["walkers"][idx]

        X = torch.cat([X_sc, X_mcmc], dim=0)
        return X, state
    return sample_fn


def make_wide_screened_sampler(N, d, omega, n_keep=2048, oversampling=10,
                               sigma_factor=2.0):
    """Screened collocation with WIDER proposal — covers more of the tails."""
    ell = 1.0 / math.sqrt(omega)
    sigma = sigma_factor * ell
    def sample_fn(psi_log_fn, epoch, state):
        X = screened_collocation(
            psi_log_fn, N, d, sigma,
            n_keep=n_keep, oversampling=oversampling,
            device=DEVICE, dtype=DTYPE,
        )
        return X, state
    return sample_fn


# ══════════════════════════════════════════════════════════════════
#  Experiment definitions
# ══════════════════════════════════════════════════════════════════

N, D, OMEGA = 6, 2, 0.5

# Short run defaults for quick testing
QUICK = dict(n_epochs=50, lr=1e-4, lr_min_frac=0.1, print_every=5, patience=25)
MEDIUM = dict(n_epochs=150, lr=5e-5, lr_min_frac=0.05, print_every=10, patience=40)


def train_bf07_from_scratch(C_occ, params):
    """Reference: standard screened collocation, 50ep from scratch."""
    f_net, bf_net = make_nets()
    sampler = make_screened_sampler(N, D, OMEGA)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="cosine", alpha_end=0.60,
        label="A. screened_collocation (reference, from scratch)",
        **QUICK,
    )


def train_mcmc_from_scratch(C_occ, params):
    """B. MCMC |Ψ|² from scratch — pure VMC-style training."""
    f_net, bf_net = make_nets()
    sampler = make_mcmc_sampler(N, D, OMEGA, n_walkers=2048, steps_per_epoch=20)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="cosine", alpha_end=0.60,
        label="B. MCMC |Ψ|² (from scratch)",
        **QUICK,
    )


def train_mcmc_finetune(C_occ, params):
    """C. Fine-tune trained bf_0.7 model on MCMC |Ψ|² samples."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")  # best available trained model
    if loaded is None:
        print("  ⚠  No trained model found, training from scratch instead")
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_mcmc_sampler(N, D, OMEGA, n_walkers=2048, steps_per_epoch=20)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        label="C. MCMC |Ψ|² fine-tune (from trained)",
        lr=5e-5,
        **{k: v for k, v in QUICK.items() if k != "lr"},
    )


def train_hard_mining(C_occ, params):
    """D. Hard-example mining: oversample high-|E_L - median| regions."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        print("  ⚠  No trained model found, training from scratch instead")
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_hard_mining_sampler(N, D, OMEGA, n_total=2048, hard_frac=0.3)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        label="D. Hard-example mining (from trained)",
        lr=5e-5,
        **{k: v for k, v in QUICK.items() if k != "lr"},
    )


def train_mixed(C_occ, params):
    """E. 50% screened collocation + 50% MCMC."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        print("  ⚠  No trained model found, training from scratch instead")
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_mixed_sampler(N, D, OMEGA, n_total=2048, mcmc_frac=0.5)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        label="E. Mixed screened+MCMC (from trained)",
        lr=5e-5,
        **{k: v for k, v in QUICK.items() if k != "lr"},
    )


def train_wide_sigma(C_occ, params):
    """F. Screened collocation with wider proposal σ=2.0*ℓ."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        print("  ⚠  No trained model found, training from scratch instead")
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_wide_screened_sampler(N, D, OMEGA, n_keep=2048, sigma_factor=2.0)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        label="F. Wide σ=2.0ℓ screened (from trained)",
        lr=5e-5,
        **{k: v for k, v in QUICK.items() if k != "lr"},
    )


def train_mcmc_varmin(C_occ, params):
    """G. MCMC |Ψ|² + pure variance minimization (no E_DMC targeting)."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        print("  ⚠  No trained model found, training from scratch instead")
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_mcmc_sampler(N, D, OMEGA, n_walkers=2048, steps_per_epoch=30)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        label="G. MCMC |Ψ|² + varmin only (from trained)",
        lr=5e-5,
        **{k: v for k, v in QUICK.items() if k != "lr"},
    )


def train_mcmc_huber(C_occ, params):
    """H. MCMC |Ψ|² + Huber loss (robust to E_L outliers near nodes)."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        print("  ⚠  No trained model found, training from scratch instead")
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_mcmc_sampler(N, D, OMEGA, n_walkers=2048, steps_per_epoch=20)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        loss_type="huber",
        huber_delta=0.5,
        label="H. MCMC |Ψ|² + Huber loss (from trained)",
        lr=5e-5,
        **{k: v for k, v in QUICK.items() if k != "lr"},
    )


# ══════════════════════════════════════════════════════════════════
#  VMC evaluation + comparison
# ══════════════════════════════════════════════════════════════════

def run_eval(f_net, bf_net, C_occ, params, label):
    """Shared VMC evaluation (reduced samples for quick tests)."""
    result = evaluate(f_net, C_occ, params, backflow_net=bf_net,
                      n_samples=10_000, label=label)
    E_mean = result["E_mean"]
    E_std  = result["E_stderr"]
    err = abs(E_mean - E_DMC) / E_DMC * 100
    return E_mean, E_std, err


def run_experiment(name, train_fn, C_occ, params):
    """Run experiment, save model, evaluate."""
    # Check if already done
    loaded = load_model(name)
    if loaded is not None:
        f_net, bf_net = loaded
        print(f"  [cached] {name}")
    else:
        print(f"\n{'═'*55}")
        print(f"# {name}")
        print(f"{'═'*55}")
        f_net, bf_net = train_fn(C_occ, params)
        save_model(f_net, bf_net, name)

    E, std, err = run_eval(f_net, bf_net, C_occ, params, name)
    print(f"  E = {E:.6f} ± {std:.6f}  (target {E_DMC}, err {err:.2f}%)")
    return E, std, err


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  LONGER runs for winners
# ══════════════════════════════════════════════════════════════════

def train_mcmc_huber_long(C_occ, params):
    """Best winner: MCMC |Ψ|² + Huber, longer run (150ep)."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_mcmc_sampler(N, D, OMEGA, n_walkers=2048, steps_per_epoch=25)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        loss_type="huber",
        huber_delta=0.5,
        label="I. MCMC |Ψ|² + Huber (150ep, from trained)",
        **MEDIUM,
    )


def train_mcmc_huber_cosine(C_occ, params):
    """MCMC |Ψ|² + Huber + cosine α targeting E_DMC."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_mcmc_sampler(N, D, OMEGA, n_walkers=2048, steps_per_epoch=25)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="cosine",
        alpha_end=0.30,          # gentle targeting
        loss_type="huber",
        huber_delta=0.5,
        label="J. MCMC Huber + cosine α→0.3 (150ep)",
        **MEDIUM,
    )


def train_hard_mining_long(C_occ, params):
    """Hard mining (fixed), longer run."""
    loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
    if loaded is None:
        f_net, bf_net = make_nets()
    else:
        f_net, bf_net = loaded
    sampler = make_hard_mining_sampler(N, D, OMEGA, n_total=2048, hard_frac=0.3)
    return train_with_sampling(
        f_net, bf_net, C_occ, params,
        sample_fn=sampler,
        alpha_schedule="varmin",
        loss_type="huber",
        huber_delta=0.5,
        label="K. Hard mining + Huber (150ep, from trained)",
        **MEDIUM,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=2,
                        help="1=quick screening, 2=longer winners")
    args = parser.parse_args()

    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    results = {}

    if args.phase == 1:
        # ── Phase 1: Quick screening (50 epochs each) ──
        print("\n" + "="*55)
        print("Phase 1: Quick sampling screening (50ep)")
        print("="*55)

        # Eval kfac_sepclip baseline
        loaded = load_model("kfac_sepclip", prefix="6e_kfac_")
        if loaded is not None:
            E, std, err = run_eval(loaded[0], loaded[1], C_occ, params, "kfac_sepclip (baseline)")
            results["kfac_sepclip_ref"] = (E, std, err)

        experiments = [
            ("mcmc_finetune",     train_mcmc_finetune),
            ("hard_mining",       train_hard_mining),
            ("mixed_sc_mcmc",     train_mixed),
            ("wide_sigma",        train_wide_sigma),
            ("mcmc_varmin",       train_mcmc_varmin),
            ("mcmc_huber",        train_mcmc_huber),
        ]

        for name, fn in experiments:
            try:
                E, std, err = run_experiment(name, fn, C_occ, params)
                results[name] = (E, std, err)
            except Exception as e:
                import traceback
                print(f"  ❌ {name} failed: {e}")
                traceback.print_exc()
                results[name] = (float("nan"), float("nan"), float("nan"))

    elif args.phase == 2:
        # ── Phase 2: Longer runs on winners ──
        print("\n" + "="*55)
        print("Phase 2: Longer runs on winners (150ep)")
        print("="*55)

        # Cache quick results
        results["mcmc_huber (50ep)"] = (11.821654, 0.003648, 0.31)
        results["mcmc_varmin (50ep)"] = (11.830054, 0.003499, 0.38)

        experiments = [
            ("mcmc_huber_long",      train_mcmc_huber_long),
            ("mcmc_huber_cosine",    train_mcmc_huber_cosine),
            ("hard_mining_long",     train_hard_mining_long),
        ]

        for name, fn in experiments:
            try:
                E, std, err = run_experiment(name, fn, C_occ, params)
                results[name] = (E, std, err)
            except Exception as e:
                import traceback
                print(f"  ❌ {name} failed: {e}")
                traceback.print_exc()
                results[name] = (float("nan"), float("nan"), float("nan"))

    # ── Summary ──
    print(f"\n{'═'*55}")
    print(f"SUMMARY — Sampling strategy experiments")
    print(f"{'═'*55}")

    results["bf_0.7 joint (300ep ref)"] = (11.823253, 0.002982, 0.33)
    results["PINN only (ref)"] = (11.834691, 0.002782, 0.42)

    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        if math.isnan(E):
            print(f"  {name:30s}  FAILED")
        else:
            print(f"  {name:30s}  E={E:.6f} ± {std:.6f}  err={err:.2f}%")
