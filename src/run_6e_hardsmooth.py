"""
Hard mining + gradient smoothing — long run with hard-point diagnostics.

Combines:
  • Hard-example mining (perturb + rescreen high-|E_L - median| configs)
  • Smoothness penalty on backflow (‖∇²Δx‖² via Hutchinson)
  • Huber loss (robust to E_L outliers near nodal surfaces)
  • Diagnostic tracking of WHERE the hard points live across epochs

Starting from trained kfac_sepclip model, 300 epochs.

Diagnostics each epoch:
  • Mean/max radial extent ⟨r²⟩ of hard vs easy points
  • Mean inter-particle distance of hard vs easy
  • Overlap of hard point indices across consecutive epochs
  • Distribution of |E_L - median| for hard vs easy
"""

import math, sys, time, copy, os, json
from collections import defaultdict
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
LOG_DIR = "/Users/aleksandersekkelsten/thesis/results/logs"

N_PARTICLES, DIM, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)


# ══════════════════════════════════════════════════════════════════
#  Model helpers
# ══════════════════════════════════════════════════════════════════

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


def save_model(f_net, bf_net, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"6e_hardsmooth_{name}.pt")
    torch.save({"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()}, path)
    print(f"  💾 Saved → {path}")


def load_base_model():
    path = os.path.join(CKPT_DIR, "6e_kfac_kfac_sepclip.pt")
    if not os.path.exists(path):
        print("  ⚠  No base model, training from scratch")
        return make_nets()
    f_net, bf_net = make_nets()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    f_net.load_state_dict(ckpt["f_net"])
    bf_net.load_state_dict(ckpt["bf_net"])
    print(f"  📦 Base ← {path}")
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


# ══════════════════════════════════════════════════════════════════
#  Smoothness penalty (from kfac experiments)
# ══════════════════════════════════════════════════════════════════

def compute_bf_smoothness_penalty(bf_net, x, spin, n_samples=32):
    """
    ‖∇²Δx‖² via Hutchinson trace estimator.
    Penalises rough backflow displacements that cause Laplacian noise.
    """
    x_sub = x[:n_samples].detach().requires_grad_(True)
    B, N, d = x_sub.shape
    dx = bf_net(x_sub, spin)
    n_probes = 2
    lap_sq_sum = torch.tensor(0.0, device=x.device, dtype=x.dtype)

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
#  Hard-point diagnostics
# ══════════════════════════════════════════════════════════════════

def compute_config_features(x):
    """
    Compute diagnostic features for a batch of configs x: (B, N, d).
    Returns dict of scalar diagnostics.
    """
    B, N, d = x.shape
    # Radial extent: mean ⟨r²⟩ per config
    r2 = (x ** 2).sum(dim=-1)  # (B, N)
    mean_r2 = r2.mean(dim=1)   # (B,)

    # Inter-particle distances
    # (B, N, 1, d) - (B, 1, N, d) → (B, N, N, d)
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    dist = diff.norm(dim=-1)   # (B, N, N)
    # Minimum inter-particle distance (per config, excluding diagonal)
    mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    dist_masked = dist.masked_fill(mask, float("inf"))
    min_dist = dist_masked.min(dim=-1).values.min(dim=-1).values  # (B,)
    mean_dist = dist_masked[~mask.expand(B, -1, -1)].reshape(B, -1).mean(dim=1)  # (B,)

    return {
        "mean_r2": mean_r2,      # (B,)
        "min_dist": min_dist,    # (B,)
        "mean_dist": mean_dist,  # (B,)
    }


def log_hard_diagnostics(epoch, X_hard, X_easy, EL_hard, EL_easy,
                         prev_hard_features, diagnostics_log):
    """
    Log diagnostic info about hard vs easy points.
    Tracks whether hard points move or stay in the same region.
    """
    with torch.no_grad():
        hard_feat = compute_config_features(X_hard)
        easy_feat = compute_config_features(X_easy)

    entry = {
        "epoch": epoch,
        "n_hard": X_hard.shape[0],
        "n_easy": X_easy.shape[0],
        # Hard point stats
        "hard_mean_r2": hard_feat["mean_r2"].mean().item(),
        "hard_max_r2": hard_feat["mean_r2"].max().item(),
        "hard_min_dist": hard_feat["min_dist"].mean().item(),
        "hard_mean_dist": hard_feat["mean_dist"].mean().item(),
        "hard_EL_mean": EL_hard.mean().item() if EL_hard.numel() > 0 else float("nan"),
        "hard_EL_std": EL_hard.std().item() if EL_hard.numel() > 1 else float("nan"),
        "hard_EL_absdev": (EL_hard - EL_hard.median()).abs().mean().item() if EL_hard.numel() > 0 else float("nan"),
        # Easy point stats  
        "easy_mean_r2": easy_feat["mean_r2"].mean().item(),
        "easy_min_dist": easy_feat["min_dist"].mean().item(),
        "easy_mean_dist": easy_feat["mean_dist"].mean().item(),
        "easy_EL_mean": EL_easy.mean().item() if EL_easy.numel() > 0 else float("nan"),
        "easy_EL_std": EL_easy.std().item() if EL_easy.numel() > 1 else float("nan"),
    }

    # Overlap with previous hard points (spatial overlap via mean_r2 distribution)
    if prev_hard_features is not None:
        # Compare distributions: how much do current hard mean_r2 overlap with prev?
        prev_r2 = prev_hard_features["mean_r2"]
        curr_r2 = hard_feat["mean_r2"]
        # Kolmogorov-Smirnov-like: median distance
        entry["r2_shift"] = abs(curr_r2.median().item() - prev_r2.median().item())
        entry["min_dist_shift"] = abs(
            hard_feat["min_dist"].median().item() -
            prev_hard_features["min_dist"].median().item()
        )
    else:
        entry["r2_shift"] = float("nan")
        entry["min_dist_shift"] = float("nan")

    diagnostics_log.append(entry)
    return hard_feat


# ══════════════════════════════════════════════════════════════════
#  Hard mining sampler (with diagnostics)
# ══════════════════════════════════════════════════════════════════

def hard_mine_sample(psi_log_fn, n_keep, oversampling, sigma, hard_frac,
                     perturb_sigma, epoch, state):
    """
    Screened collocation + hard example mining + diagnostics.
    Returns (X_train, X_hard, X_easy, EL_hard, EL_easy, state).
    """
    n_hard = int(n_keep * hard_frac)
    n_normal = n_keep - n_hard

    # Step 1: Screened collocation
    X_full = screened_collocation(
        psi_log_fn, N_PARTICLES, DIM, sigma,
        n_keep=n_keep, oversampling=oversampling,
        device=DEVICE, dtype=DTYPE,
    )

    # Step 2: Evaluate E_L
    all_EL = []
    mb = 128
    for i in range(0, X_full.shape[0], mb):
        x_mb = X_full[i:i + mb].detach().requires_grad_(True)
        EL = compute_local_energy(psi_log_fn, x_mb, OMEGA).view(-1).detach()
        all_EL.append(EL)
    EL_cat = torch.cat(all_EL)
    good = torch.isfinite(EL_cat)
    if good.any():
        EL_cat[~good] = EL_cat[good].median()
    else:
        return X_full, X_full[:0], X_full, torch.tensor([]), EL_cat, state

    # Step 3: Find hard points
    median = EL_cat.median()
    deviations = (EL_cat - median).abs()
    n_hard_actual = min(n_hard, X_full.shape[0] // 2)
    _, hard_idx = torch.topk(deviations, n_hard_actual)
    X_hard_orig = X_full[hard_idx]
    EL_hard = EL_cat[hard_idx]

    # Step 4: Perturb + rescreen
    n_perturb = 5
    candidates = X_hard_orig.unsqueeze(1).expand(-1, n_perturb, -1, -1)
    candidates = candidates + torch.randn_like(candidates) * perturb_sigma
    candidates = candidates.reshape(-1, N_PARTICLES, DIM)

    with torch.no_grad():
        log_psi2_parts = []
        for i in range(0, candidates.shape[0], 512):
            lp = psi_log_fn(candidates[i:i + 512])
            log_psi2_parts.append(2.0 * lp)
        log_psi2 = torch.cat(log_psi2_parts)

    valid = torch.isfinite(log_psi2)
    log_psi2[~valid] = -1e10
    n_keep_hard = min(n_hard_actual, int(valid.sum().item()))

    if n_keep_hard > 0:
        _, keep_idx = torch.topk(log_psi2, n_keep_hard)
        X_hard_screened = candidates[keep_idx]
    else:
        X_hard_screened = torch.empty(0, N_PARTICLES, DIM, device=DEVICE, dtype=DTYPE)

    # Step 5: Easy points
    all_idx = set(range(X_full.shape[0]))
    hard_set = set(hard_idx.tolist())
    easy_idx = sorted(all_idx - hard_set)[:n_normal]
    X_easy = X_full[easy_idx]
    EL_easy = EL_cat[list(easy_idx)]

    # Combine
    if X_hard_screened.shape[0] > 0:
        X_train = torch.cat([X_easy, X_hard_screened], dim=0)
    else:
        X_train = X_easy

    return X_train, X_hard_orig, X_easy, EL_hard, EL_easy, state


# ══════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════

def train_hard_smooth(
    f_net, bf_net, C_occ, params, *,
    n_epochs=300,
    lr=3e-4,
    lr_min_frac=0.02,
    # Sampling
    n_collocation=2048,
    oversampling=10,
    hard_frac=0.3,
    perturb_sigma_frac=0.3,
    sigma_factor=1.3,
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
    # Diagnostics
    diag_every=10,
    label="",
):
    omega = OMEGA
    sigma = sigma_factor * ELL
    perturb_sigma = perturb_sigma_frac * ELL

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
    print(f"  hard_frac={hard_frac}, smooth_λ={smooth_lambda:.1e}, Huber δ={huber_delta}")
    print(f"  perturb_σ={perturb_sigma:.3f}, screening σ={sigma:.3f}")
    print(f"{'='*65}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    sample_state = {}
    diagnostics_log = []
    prev_hard_features = None

    for epoch in range(n_epochs):
        # ── Alpha schedule ──
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # ── Sample with hard mining ──
        f_net.eval(); bf_net.eval()
        X, X_hard, X_easy, EL_hard, EL_easy, sample_state = hard_mine_sample(
            psi_log_fn, n_collocation, oversampling, sigma, hard_frac,
            perturb_sigma, epoch, sample_state,
        )
        n_pts = X.shape[0]

        # ── Diagnostics (every diag_every epochs) ──
        if epoch % diag_every == 0 and X_hard.shape[0] > 0:
            prev_hard_features = log_hard_diagnostics(
                epoch, X_hard, X_easy, EL_hard, EL_easy,
                prev_hard_features, diagnostics_log,
            )

        # ── Train ──
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

            # Huber loss
            loss_mb = nn.functional.huber_loss(
                resid, torch.zeros_like(resid), delta=huber_delta)

            # Smoothness penalty
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

        if patience > 0 and epochs_no_improve >= patience and epoch > phase1_end + 30:
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

            # Hard point diagnostics summary
            diag_str = ""
            if diagnostics_log and epoch % diag_every == 0:
                d_last = diagnostics_log[-1]
                diag_str = (
                    f"  h_r²={d_last['hard_mean_r2']:.2f}"
                    f" e_r²={d_last['easy_mean_r2']:.2f}"
                    f" h_dmin={d_last['hard_min_dist']:.3f}"
                    f" r²Δ={d_last.get('r2_shift', float('nan')):.3f}"
                )

            phase_tag = " [ph1]" if epoch < phase1_end else ""
            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  lr={cur_lr:.1e}  "
                f"|Δx|={bf_mag:.3f}  err={err:.2f}%{phase_tag}{diag_str}"
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
#  Print diagnostics summary
# ══════════════════════════════════════════════════════════════════

def print_diagnostics_summary(diagnostics_log):
    """Print a nice summary of how hard points evolve over training."""
    if not diagnostics_log:
        print("  No diagnostics collected.")
        return

    print(f"\n{'─'*75}")
    print(f"  Hard-point dynamics across training")
    print(f"{'─'*75}")
    print(f"  {'ep':>4s}  {'n_h':>4s}  "
          f"{'h_⟨r²⟩':>7s}  {'e_⟨r²⟩':>7s}  "
          f"{'h_dmin':>7s}  {'e_dmin':>7s}  "
          f"{'h_EL_σ':>7s}  {'e_EL_σ':>7s}  "
          f"{'r²_Δ':>6s}  {'d_Δ':>6s}")

    for d in diagnostics_log:
        print(
            f"  {d['epoch']:4d}  {d['n_hard']:4d}  "
            f"{d['hard_mean_r2']:7.3f}  {d['easy_mean_r2']:7.3f}  "
            f"{d['hard_min_dist']:7.4f}  {d['easy_min_dist']:7.4f}  "
            f"{d['hard_EL_std']:7.3f}  {d['easy_EL_std']:7.3f}  "
            f"{d.get('r2_shift', float('nan')):6.3f}  "
            f"{d.get('min_dist_shift', float('nan')):6.4f}"
        )

    # Summary statistics
    hard_r2s = [d["hard_mean_r2"] for d in diagnostics_log]
    hard_dmins = [d["hard_min_dist"] for d in diagnostics_log]
    r2_shifts = [d["r2_shift"] for d in diagnostics_log if math.isfinite(d.get("r2_shift", float("nan")))]

    print(f"\n  Hard ⟨r²⟩: {min(hard_r2s):.3f} – {max(hard_r2s):.3f}")
    print(f"  Hard d_min: {min(hard_dmins):.4f} – {max(hard_dmins):.4f}")

    if r2_shifts:
        mean_shift = sum(r2_shifts) / len(r2_shifts)
        max_shift = max(r2_shifts)
        print(f"  r² shift: mean={mean_shift:.4f}, max={max_shift:.4f}")
        if mean_shift < 0.05:
            print(f"  → Hard points are STABLE (staying in same region)")
        elif mean_shift < 0.2:
            print(f"  → Hard points DRIFT slowly")
        else:
            print(f"  → Hard points MOVE significantly between epochs")


# ══════════════════════════════════════════════════════════════════
#  Experiments
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    C_occ, params = setup_noninteracting(N_PARTICLES, OMEGA, device=DEVICE, dtype=DTYPE)
    results = {}

    # ── Experiment 1: hard mining + Huber + smoothness (long, from trained) ──
    print(f"\n{'═'*65}")
    print(f"# Exp 1: Hard mine + Huber + smoothness (300ep, from trained)")
    print(f"{'═'*65}")

    f_net, bf_net = load_base_model()
    f_net1, bf_net1, diag1 = train_hard_smooth(
        f_net, bf_net, C_occ, params,
        n_epochs=300,
        lr=3e-4,
        lr_min_frac=0.02,
        hard_frac=0.3,
        smooth_lambda=1e-3,
        huber_delta=0.5,
        perturb_sigma_frac=0.3,
        label="hard_mine + Huber + smooth_λ=1e-3 (300ep from trained)",
        diag_every=10,
    )
    save_model(f_net1, bf_net1, "hardsmooth_300")
    print_diagnostics_summary(diag1)

    # VMC eval
    result1 = evaluate(f_net1, C_occ, params, backflow_net=bf_net1,
                       n_samples=15_000, label="hardsmooth_300")
    E1 = result1["E_mean"]
    std1 = result1["E_stderr"]
    err1 = abs(E1 - E_DMC) / E_DMC * 100
    print(f"  E = {E1:.6f} ± {std1:.6f}  (target {E_DMC}, err {err1:.2f}%)")
    results["hardsmooth_300 (from trained)"] = (E1, std1, err1)

    # ── Experiment 2: same but from SCRATCH (fair comparison to bf_0.7 joint) ──
    print(f"\n{'═'*65}")
    print(f"# Exp 2: Hard mine + Huber + smoothness (300ep, from SCRATCH)")
    print(f"{'═'*65}")

    f_net2, bf_net2 = make_nets()
    f_net2, bf_net2, diag2 = train_hard_smooth(
        f_net2, bf_net2, C_occ, params,
        n_epochs=300,
        lr=3e-4,
        lr_min_frac=0.02,
        hard_frac=0.3,
        smooth_lambda=1e-3,
        huber_delta=0.5,
        perturb_sigma_frac=0.3,
        label="hard_mine + Huber + smooth_λ=1e-3 (300ep from scratch)",
        diag_every=10,
    )
    save_model(f_net2, bf_net2, "hardsmooth_scratch")
    print_diagnostics_summary(diag2)

    result2 = evaluate(f_net2, C_occ, params, backflow_net=bf_net2,
                       n_samples=15_000, label="hardsmooth_scratch")
    E2 = result2["E_mean"]
    std2 = result2["E_stderr"]
    err2 = abs(E2 - E_DMC) / E_DMC * 100
    print(f"  E = {E2:.6f} ± {std2:.6f}  (target {E_DMC}, err {err2:.2f}%)")
    results["hardsmooth_scratch"] = (E2, std2, err2)

    # ── Experiment 3: hard mining + Huber only, NO smoothness (ablation) ──
    print(f"\n{'═'*65}")
    print(f"# Exp 3: Hard mine + Huber, NO smoothness (300ep, from scratch)")
    print(f"{'═'*65}")

    f_net3, bf_net3 = make_nets()
    f_net3, bf_net3, diag3 = train_hard_smooth(
        f_net3, bf_net3, C_occ, params,
        n_epochs=300,
        lr=3e-4,
        lr_min_frac=0.02,
        hard_frac=0.3,
        smooth_lambda=0.0,     # NO smoothness
        huber_delta=0.5,
        perturb_sigma_frac=0.3,
        label="hard_mine + Huber, NO smooth (300ep from scratch)",
        diag_every=10,
    )
    save_model(f_net3, bf_net3, "hardhuber_scratch")
    print_diagnostics_summary(diag3)

    result3 = evaluate(f_net3, C_occ, params, backflow_net=bf_net3,
                       n_samples=15_000, label="hardhuber_scratch")
    E3 = result3["E_mean"]
    std3 = result3["E_stderr"]
    err3 = abs(E3 - E_DMC) / E_DMC * 100
    print(f"  E = {E3:.6f} ± {std3:.6f}  (target {E_DMC}, err {err3:.2f}%)")
    results["hardhuber_scratch"] = (E3, std3, err3)

    # ── Summary ──
    print(f"\n{'═'*65}")
    print(f"SUMMARY — Hard mining + gradient smoothing")
    print(f"{'═'*65}")

    results["bf_0.7 joint (300ep ref)"] = (11.823253, 0.002982, 0.33)
    results["hard_mine_huber 50ep ft"] = (11.833516, 0.003618, 0.41)
    results["PINN only (ref)"] = (11.834691, 0.002782, 0.42)

    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        print(f"  {name:40s}  E={E:.6f} ± {std:.6f}  err={err:.2f}%")

    # Save diagnostics to JSON
    os.makedirs(LOG_DIR, exist_ok=True)
    for label_name, diag_data in [("hardsmooth_300", diag1),
                                   ("hardsmooth_scratch", diag2),
                                   ("hardhuber_scratch", diag3)]:
        json_path = os.path.join(LOG_DIR, f"diag_{label_name}.json")
        with open(json_path, "w") as fp:
            json.dump(diag_data, fp, indent=2)
        print(f"  Diagnostics → {json_path}")
