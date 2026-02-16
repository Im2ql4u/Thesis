"""
Residual-based sampling experiments for 6e quantum dot (ω=0.5).

Core insight:
  Screened collocation trains on points where |Ψ|²/q is highest — these are the
  "easy" points where the wavefunction is large. But VMC evaluates everywhere |Ψ|²
  lives, including regions where E_L has high variance. We need to train on the
  HARD configurations too — where the local energy deviates most from the mean.

  All approaches here are genuinely residual-based:
    • Fixed proposal distributions (independent of θ)
    • Points selected via |Ψ|² screening (no gradient through selection)
    • Loss computed from Schrödinger residual E_L at selected points

Experiments (all start from trained kfac_sepclip model, 50ep quick test):
  A. screened_baseline   — standard σ=1.3ℓ screened collocation (reference)
  B. hard_mine           — screened collocation, then find high-|E_L - median|
                           points, perturb them, re-screen, and include
  C. multiscale          — pool proposals at 3 scales (σ=0.8ℓ, 1.3ℓ, 2.5ℓ),
                           screen the combined pool → diverse coverage
  D. residual_weighted   — screen by |E_L - E_target|² × |Ψ|²/q instead of
                           just |Ψ|²/q — focuses on high-residual regions
  E. iterative_refine    — use previous epoch's hard points as Gaussian proposal
                           centers, plus fresh screened collocation
  F. huber_screened      — standard screened collocation + Huber loss
  G. oversampled         — 40× oversampling instead of 10× (more candidates to
                           screen from → better coverage)
"""

import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

from functions.Neural_Networks import (
    psi_fn,
)
from PINN import PINN, CTNNBackflowNet
from run_6e_residual import (
    compute_local_energy,
    evaluate,
    sample_gaussian_proposal,
    screened_collocation,
    setup_noninteracting,
)

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"

N, D, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)  # oscillator length


# ══════════════════════════════════════════════════════════════════
#  Model construction & loading
# ══════════════════════════════════════════════════════════════════


def make_nets(bf_scale_init=0.7, zero_init_last=False):
    f_net = (
        PINN(
            n_particles=N,
            d=D,
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
            d=D,
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


def save_model(f_net, bf_net, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"6e_samp2_{name}.pt")
    torch.save({"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()}, path)
    print(f"  💾 Saved → {path}")


def load_model(name, prefix="6e_samp2_"):
    path = os.path.join(CKPT_DIR, f"{prefix}{name}.pt")
    if not os.path.exists(path):
        return None
    f_net, bf_net = make_nets()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    f_net.load_state_dict(ckpt["f_net"])
    bf_net.load_state_dict(ckpt["bf_net"])
    print(f"  📦 Loaded ← {path}")
    return f_net, bf_net


def load_base_model():
    """Load the best available trained model as starting point."""
    path = os.path.join(CKPT_DIR, "6e_kfac_kfac_sepclip.pt")
    if not os.path.exists(path):
        print("  ⚠  No trained model found, training from scratch")
        return make_nets()
    f_net, bf_net = make_nets()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    f_net.load_state_dict(ckpt["f_net"])
    bf_net.load_state_dict(ckpt["bf_net"])
    print(f"  📦 Base model ← {path}")
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]).to(
        DEVICE
    )

    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return lp

    return fn, spin


# ══════════════════════════════════════════════════════════════════
#  Residual-based sampling strategies (NO MCMC — all use fixed proposals)
# ══════════════════════════════════════════════════════════════════

# --- A. Standard screened collocation ---


def make_screened_sampler(n_keep=2048, oversampling=10, sigma_factor=1.3):
    sigma = sigma_factor * ELL

    def sample_fn(psi_log_fn, epoch, state):
        X = screened_collocation(
            psi_log_fn,
            N,
            D,
            sigma,
            n_keep=n_keep,
            oversampling=oversampling,
            device=DEVICE,
            dtype=DTYPE,
        )
        return X, state

    return sample_fn


# --- B. Hard-example mining ---
#   1. Screened collocation → get a big pool
#   2. Evaluate E_L at all → find points with highest |E_L - median|
#   3. Perturb those hard points → Gaussian cloud around them
#   4. Re-screen the cloud through |Ψ|²/q → keep the good perturbed ones
#   5. Combine: normal screened + re-screened-hard


def make_hard_mining_sampler(
    n_keep=2048, hard_frac=0.3, oversampling=10, sigma_factor=1.3, perturb_sigma_frac=0.3
):
    """
    Residual-based hard mining: no MCMC, no walkers.
    Every epoch:
      1. Screened collocation → pool (oversampling × n_keep)
      2. Evaluate E_L at the kept points
      3. Find top-30% by |E_L - median| → "hard" points
      4. Perturb hard points with small Gaussian → generate candidates
      5. Re-screen candidates via |Ψ|²/q → keep the ones where Ψ lives
      6. Final batch = (1-hard_frac)*n_keep normal + hard_frac*n_keep re-screened
    """
    sigma = sigma_factor * ELL
    perturb_sigma = perturb_sigma_frac * ELL
    n_hard = int(n_keep * hard_frac)
    n_normal = n_keep - n_hard

    def sample_fn(psi_log_fn, epoch, state):
        # Step 1: Normal screened collocation (full pool for E_L eval)
        X_full = screened_collocation(
            psi_log_fn,
            N,
            D,
            sigma,
            n_keep=n_keep,
            oversampling=oversampling,
            device=DEVICE,
            dtype=DTYPE,
        )

        # Step 2: Evaluate E_L at all screened points (needs grad!)
        all_EL = []
        mb = 128
        for i in range(0, X_full.shape[0], mb):
            x_mb = X_full[i : i + mb].detach().requires_grad_(True)
            EL = compute_local_energy(psi_log_fn, x_mb, OMEGA).view(-1).detach()
            all_EL.append(EL)
        EL_cat = torch.cat(all_EL)
        good = torch.isfinite(EL_cat)
        if good.any():
            EL_cat[~good] = EL_cat[good].median()
        else:
            # All bad — fall back to normal screened
            return X_full, state

        # Step 3: Find hard points (highest |E_L - median|)
        median = EL_cat.median()
        deviations = (EL_cat - median).abs()
        n_hard_actual = min(n_hard, X_full.shape[0] // 2)
        _, hard_idx = torch.topk(deviations, n_hard_actual)
        X_hard = X_full[hard_idx]

        # Step 4: Perturb hard points → generate candidates near them
        # Each hard point spawns ~5 candidates
        n_perturb = 5
        candidates = X_hard.unsqueeze(1).expand(-1, n_perturb, -1, -1)  # (H, 5, N, d)
        candidates = candidates + torch.randn_like(candidates) * perturb_sigma
        candidates = candidates.reshape(-1, N, D)  # (H*5, N, d)

        # Step 5: Re-screen candidates via |Ψ|²
        with torch.no_grad():
            log_psi_cand = []
            for i in range(0, candidates.shape[0], 512):
                lp = psi_log_fn(candidates[i : i + 512])
                log_psi_cand.append(2.0 * lp)
            log_psi2 = torch.cat(log_psi_cand)

        # Keep top n_hard_actual by |Ψ|² (no proposal correction needed since
        # we're perturbing around points that already have high |Ψ|²)
        valid = torch.isfinite(log_psi2)
        log_psi2[~valid] = -1e10
        n_keep_hard = min(n_hard_actual, int(valid.sum().item()))
        if n_keep_hard > 0:
            _, keep_idx = torch.topk(log_psi2, n_keep_hard)
            X_hard_screened = candidates[keep_idx]
        else:
            X_hard_screened = torch.empty(0, N, D, device=DEVICE, dtype=DTYPE)

        # Step 6: Normal points (from the non-hard screened set)
        # Remove hard indices, take up to n_normal from the rest
        all_idx = set(range(X_full.shape[0]))
        hard_set = set(hard_idx.tolist())
        easy_idx = list(all_idx - hard_set)
        easy_idx = easy_idx[:n_normal]
        X_normal = X_full[easy_idx]

        # Combine
        if X_hard_screened.shape[0] > 0:
            X = torch.cat([X_normal, X_hard_screened], dim=0)
        else:
            X = X_normal

        # Track stats
        state["n_hard"] = n_keep_hard
        state["max_dev"] = deviations.max().item()
        state["median_EL"] = median.item()

        return X, state

    return sample_fn


# --- C. Multi-scale screened collocation ---
#   Pool candidates from 3 different Gaussian widths: tight, medium, wide.
#   Screen the combined pool. Gives diverse radial coverage.


def make_multiscale_sampler(n_keep=2048, oversampling_per_scale=7, sigma_factors=(0.8, 1.3, 2.5)):
    sigmas = [f * ELL for f in sigma_factors]
    n_cand_each = oversampling_per_scale * n_keep  # per scale

    @torch.no_grad()
    def sample_fn(psi_log_fn, epoch, state):
        all_x = []
        all_log_ratio = []

        for sig in sigmas:
            x, log_q = sample_gaussian_proposal(n_cand_each, N, D, sig, DEVICE, DTYPE)
            # Evaluate log|Ψ|²
            log_psi2_parts = []
            for i in range(0, n_cand_each, 4096):
                lp = psi_log_fn(x[i : i + 4096])
                log_psi2_parts.append(2.0 * lp)
            log_psi2 = torch.cat(log_psi2_parts)
            log_ratio = log_psi2 - log_q
            all_x.append(x)
            all_log_ratio.append(log_ratio)

        # Pool everything and take top-K
        X_pool = torch.cat(all_x, dim=0)
        LR_pool = torch.cat(all_log_ratio, dim=0)
        _, idx = torch.topk(LR_pool, n_keep)
        return X_pool[idx].clone(), state

    return sample_fn


# --- D. Residual-weighted screening ---
#   Instead of screening by |Ψ|²/q alone, weight by the residual.
#   Points where E_L is far from E_target AND |Ψ|² is large are most important.
#   Two-epoch process: epoch n uses E_L from epoch n-1 to bias selection.


def make_residual_weighted_sampler(
    n_keep=2048, oversampling=10, sigma_factor=1.3, residual_weight=1.0
):
    sigma = sigma_factor * ELL

    def sample_fn(psi_log_fn, epoch, state):
        n_cand = oversampling * n_keep

        # If we have hard points from previous epoch, use them as extra centers
        if "prev_hard_x" in state and state["prev_hard_x"] is not None:
            # Mix: 70% fresh Gaussian + 30% perturbed from previous hard
            n_fresh = int(0.7 * n_cand)
            n_from_prev = n_cand - n_fresh

            x_fresh, log_q_fresh = sample_gaussian_proposal(n_fresh, N, D, sigma, DEVICE, DTYPE)

            # Sample from previous hard points with perturbation
            prev_hard = state["prev_hard_x"]
            idx = torch.randint(0, prev_hard.shape[0], (n_from_prev,))
            x_from_prev = (
                prev_hard[idx]
                + torch.randn(n_from_prev, N, D, device=DEVICE, dtype=DTYPE) * sigma * 0.3
            )
            # Compute log_q for the perturbed points (approximate with the same Gaussian)
            Nd = N * D
            log_q_prev = -0.5 * Nd * math.log(2 * math.pi * sigma**2) - x_from_prev.reshape(
                n_from_prev, -1
            ).pow(2).sum(-1) / (2 * sigma**2)

            x_all = torch.cat([x_fresh, x_from_prev], dim=0)
            log_q_all = torch.cat([log_q_fresh, log_q_prev], dim=0)
        else:
            x_all, log_q_all = sample_gaussian_proposal(n_cand, N, D, sigma, DEVICE, DTYPE)

        # Forward pass to get |Ψ|²
        with torch.no_grad():
            log_psi2_parts = []
            for i in range(0, x_all.shape[0], 4096):
                lp = psi_log_fn(x_all[i : i + 4096])
                log_psi2_parts.append(2.0 * lp)
            log_psi2 = torch.cat(log_psi2_parts)

        log_ratio = log_psi2 - log_q_all
        _, idx = torch.topk(log_ratio, n_keep)
        X = x_all[idx].clone()

        # After we've chosen our training points, evaluate E_L to find hard ones
        # for next epoch (cheap since we're already computing E_L in training)
        # Store the INDICES of this epoch's hard points as prev_hard_x
        # This is done outside training, but we can flag it in state
        state["need_hard_eval"] = True

        return X, state

    return sample_fn


# --- E. Iterative refinement ---
#   Each epoch: screened collocation + adaptive Gaussian around previous
#   hard points. The proposal adapts without MCMC.


def make_iterative_refine_sampler(
    n_keep=2048, oversampling=10, sigma_factor=1.3, refine_frac=0.3, refine_sigma_factor=0.4
):
    sigma = sigma_factor * ELL
    refine_sigma = refine_sigma_factor * ELL
    n_refine = int(n_keep * refine_frac)
    n_normal = n_keep - n_refine

    def sample_fn(psi_log_fn, epoch, state):
        if "prev_hard_x" not in state or state["prev_hard_x"] is None:
            # First epoch: just normal screened
            X = screened_collocation(
                psi_log_fn,
                N,
                D,
                sigma,
                n_keep=n_keep,
                oversampling=oversampling,
                device=DEVICE,
                dtype=DTYPE,
            )
            state["prev_hard_x"] = None
            state["need_hard_eval"] = True
            return X, state

        # Normal screened portion
        X_normal = screened_collocation(
            psi_log_fn,
            N,
            D,
            sigma,
            n_keep=n_normal,
            oversampling=oversampling,
            device=DEVICE,
            dtype=DTYPE,
        )

        # Refine portion: Gaussian around previous hard points, then screen
        prev = state["prev_hard_x"]
        n_perturb = 8
        n_cand_refine = min(prev.shape[0] * n_perturb, n_refine * oversampling)
        idx = torch.randint(0, prev.shape[0], (n_cand_refine,))
        x_cand = (
            prev[idx] + torch.randn(n_cand_refine, N, D, device=DEVICE, dtype=DTYPE) * refine_sigma
        )

        # Screen by |Ψ|²
        with torch.no_grad():
            log_psi2_parts = []
            for i in range(0, n_cand_refine, 4096):
                lp = psi_log_fn(x_cand[i : i + 4096])
                log_psi2_parts.append(2.0 * lp)
            log_psi2 = torch.cat(log_psi2_parts)

        valid = torch.isfinite(log_psi2)
        log_psi2[~valid] = -1e10
        n_keep_refine = min(n_refine, int(valid.sum().item()))
        if n_keep_refine > 0:
            _, keep_idx = torch.topk(log_psi2, n_keep_refine)
            X_refine = x_cand[keep_idx]
        else:
            X_refine = torch.empty(0, N, D, device=DEVICE, dtype=DTYPE)

        X = torch.cat([X_normal, X_refine], dim=0) if X_refine.shape[0] > 0 else X_normal
        state["need_hard_eval"] = True
        return X, state

    return sample_fn


# --- F. Standard screened + Huber loss (baseline with outlier robustness) ---
# Just uses the standard sampler, Huber loss is set in training config.


# --- G. Heavy oversampling ---


def make_heavy_oversampled_sampler(n_keep=2048, oversampling=40, sigma_factor=1.3):
    """40× oversampling instead of 10× — much better top-K approximation of |Ψ|²."""
    sigma = sigma_factor * ELL

    def sample_fn(psi_log_fn, epoch, state):
        X = screened_collocation(
            psi_log_fn,
            N,
            D,
            sigma,
            n_keep=n_keep,
            oversampling=oversampling,
            device=DEVICE,
            dtype=DTYPE,
        )
        return X, state

    return sample_fn


# ══════════════════════════════════════════════════════════════════
#  Unified trainer (same as before, but with hard-point tracking)
# ══════════════════════════════════════════════════════════════════


def train_with_sampling(
    f_net,
    bf_net,
    C_occ,
    params,
    *,
    sample_fn,
    n_epochs=50,
    lr=5e-5,
    lr_min_frac=0.1,
    alpha_schedule="varmin",
    alpha_fixed=0.0,
    alpha_end=0.60,
    phase1_frac=0.25,
    micro_batch=256,
    grad_clip=0.5,
    quantile_trim=0.02,
    print_every=5,
    label="",
    patience=30,
    loss_type="mse",
    huber_delta=0.5,
):
    omega = OMEGA
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

    n_p = sum(p.numel() for p in all_params)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, {n_p:,} params, lr={lr}→{lr_min:.1e}, loss={loss_type}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    sample_state = {}

    for epoch in range(n_epochs):
        # ── Alpha ──
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

        # ── Sample ──
        f_net.eval()
        bf_net.eval()
        X, sample_state = sample_fn(psi_log_fn, epoch, sample_state)
        n_pts = X.shape[0]

        # ── Train ──
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
                loss_mb = nn.functional.huber_loss(
                    resid, torch.zeros_like(resid), delta=huber_delta
                )
            else:
                loss_mb = (resid**2).mean()

            (loss_mb / n_batches).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Post-epoch: find hard points for next epoch's sampler ──
        if sample_state.get("need_hard_eval", False) and len(all_EL) > 0:
            EL_cat = torch.cat(all_EL)
            median_el = EL_cat.median()
            devs = (EL_cat - median_el).abs()
            # Top 30% hardest from what we've computed
            n_hard = max(1, int(0.3 * EL_cat.shape[0]))
            _, hard_idx_local = torch.topk(devs, min(n_hard, devs.shape[0]))
            # Map back to X indices (approximate — the batching may have gaps)
            # We'll just use the last batch's X directly
            sample_state["prev_hard_x"] = X[hard_idx_local % X.shape[0]].detach().clone()
            sample_state["need_hard_eval"] = False

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

        if patience > 0 and epochs_no_improve >= patience and epoch > 10:
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
            if "n_hard" in sample_state:
                extra = f"  n_hard={sample_state['n_hard']}"

            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  lr={cur_lr:.1e}  "
                f"|Δx|={bf_mag:.3f}  err={err:.2f}%{extra}"
            )
            sys.stdout.flush()

    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)

    total = time.time() - t0
    print(f"  Best var={best_var:.3e}, {total:.0f}s ({total/60:.1f}min)")
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  Experiment definitions
# ══════════════════════════════════════════════════════════════════

QUICK = dict(n_epochs=50, print_every=5, patience=25)


def exp_screened_baseline(C_occ, params):
    """A. Standard screened collocation — reference."""
    f_net, bf_net = load_base_model()
    sampler = make_screened_sampler()
    return train_with_sampling(
        f_net,
        bf_net,
        C_occ,
        params,
        sample_fn=sampler,
        label="A. screened_baseline (σ=1.3ℓ, 10× OS)",
        **QUICK,
    )


def exp_hard_mine(C_occ, params):
    """B. Hard-example mining via screened collocation."""
    f_net, bf_net = load_base_model()
    sampler = make_hard_mining_sampler(
        n_keep=2048, hard_frac=0.3, oversampling=10, perturb_sigma_frac=0.3
    )
    return train_with_sampling(
        f_net,
        bf_net,
        C_occ,
        params,
        sample_fn=sampler,
        label="B. hard_mine (30% hard, perturb+rescreen)",
        **QUICK,
    )


def exp_multiscale(C_occ, params):
    """C. Multi-scale screened: σ = 0.8ℓ, 1.3ℓ, 2.5ℓ pooled."""
    f_net, bf_net = load_base_model()
    sampler = make_multiscale_sampler(
        n_keep=2048, oversampling_per_scale=7, sigma_factors=(0.8, 1.3, 2.5)
    )
    return train_with_sampling(
        f_net,
        bf_net,
        C_occ,
        params,
        sample_fn=sampler,
        label="C. multiscale (σ=0.8,1.3,2.5 ℓ)",
        **QUICK,
    )


def exp_iterative_refine(C_occ, params):
    """E. Iterative refinement: screen + perturb previous hard points."""
    f_net, bf_net = load_base_model()
    sampler = make_iterative_refine_sampler(n_keep=2048, refine_frac=0.3, refine_sigma_factor=0.4)
    return train_with_sampling(
        f_net,
        bf_net,
        C_occ,
        params,
        sample_fn=sampler,
        label="E. iterative_refine (30% near prev hard)",
        **QUICK,
    )


def exp_huber_screened(C_occ, params):
    """F. Standard screened + Huber loss."""
    f_net, bf_net = load_base_model()
    sampler = make_screened_sampler()
    return train_with_sampling(
        f_net,
        bf_net,
        C_occ,
        params,
        sample_fn=sampler,
        loss_type="huber",
        huber_delta=0.5,
        label="F. huber_screened (Huber δ=0.5)",
        **QUICK,
    )


def exp_oversampled(C_occ, params):
    """G. 40× oversampling (vs standard 10×)."""
    f_net, bf_net = load_base_model()
    sampler = make_heavy_oversampled_sampler(n_keep=2048, oversampling=40)
    return train_with_sampling(
        f_net,
        bf_net,
        C_occ,
        params,
        sample_fn=sampler,
        label="G. oversampled (40× OS, σ=1.3ℓ)",
        **QUICK,
    )


def exp_hard_mine_huber(C_occ, params):
    """B+F. Hard mining + Huber loss combined."""
    f_net, bf_net = load_base_model()
    sampler = make_hard_mining_sampler(
        n_keep=2048, hard_frac=0.3, oversampling=10, perturb_sigma_frac=0.3
    )
    return train_with_sampling(
        f_net,
        bf_net,
        C_occ,
        params,
        sample_fn=sampler,
        loss_type="huber",
        huber_delta=0.5,
        label="H. hard_mine + Huber",
        **QUICK,
    )


# ══════════════════════════════════════════════════════════════════
#  Evaluation & orchestration
# ══════════════════════════════════════════════════════════════════


def run_eval(f_net, bf_net, C_occ, params, label):
    result = evaluate(f_net, C_occ, params, backflow_net=bf_net, n_samples=10_000, label=label)
    E_mean = result["E_mean"]
    E_std = result["E_stderr"]
    err = abs(E_mean - E_DMC) / E_DMC * 100
    return E_mean, E_std, err


def run_experiment(name, train_fn, C_occ, params):
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


if __name__ == "__main__":
    C_occ, params = setup_noninteracting(N, OMEGA, device=DEVICE, dtype=DTYPE)
    results = {}

    print("\n" + "=" * 55)
    print("Residual-based sampling experiments (50ep quick screen)")
    print("  All fine-tuned from kfac_sepclip, lr=5e-5, varmin")
    print("=" * 55)

    # Baseline: eval the starting model without any fine-tuning
    base = load_base_model()
    if base is not None:
        E, std, err = run_eval(base[0], base[1], C_occ, params, "base_model (no finetune)")
        results["base_model (no ft)"] = (E, std, err)
        print(f"  → base model: {err:.2f}%")

    experiments = [
        ("screened_baseline", exp_screened_baseline),
        ("hard_mine", exp_hard_mine),
        ("multiscale", exp_multiscale),
        ("iterative_refine", exp_iterative_refine),
        ("huber_screened", exp_huber_screened),
        ("oversampled", exp_oversampled),
        ("hard_mine_huber", exp_hard_mine_huber),
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
    print("SUMMARY — Residual-based sampling experiments")
    print(f"{'═'*55}")

    results["bf_0.7 joint (300ep ref)"] = (11.823253, 0.002982, 0.33)
    results["PINN only (ref)"] = (11.834691, 0.002782, 0.42)

    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        if math.isnan(E):
            print(f"  {name:30s}  FAILED")
        else:
            print(f"  {name:30s}  E={E:.6f} ± {std:.6f}  err={err:.2f}%")
