"""
Fast backflow extension experiments.

Strategy: Train PINN once (300ep), save checkpoint, then try 5 different
ways to add backflow — all starting from the same converged Jastrow.

Experiments (all bf_scale=0.7, zero_init_last=False):
  1. bf_only         — freeze Jastrow, train bf 100ep
  2. cusp+bf         — cusp pre-train 30ep → freeze Jastrow → bf 100ep
  3. fisher_bf       — freeze Jastrow, diagonal Fisher on bf, 100ep
  4. bf_then_joint   — freeze Jastrow → bf 50ep → unfreeze both → joint 80ep (f_lr=5e-5)
  5. cusp+bf+joint   — cusp 30ep → freeze → bf 50ep → unfreeze → joint 80ep

All ~100ep effective bf training.  Total time ~2h instead of ~8h.
"""

import math, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, CTNNBackflowNet
from functions.Neural_Networks import psi_fn, _laplacian_logpsi_exact
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc

from run_6e_residual import (
    setup_noninteracting,
    compute_local_energy,
    screened_collocation,
    evaluate,
    train_residual,
    COMMON,
)

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_PATH = "/Users/aleksandersekkelsten/thesis/results/models/pinn_6e_ckpt.pt"


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def make_bf(bf_scale_init=0.7):
    return CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2, act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=bf_scale_init,
        zero_init_last=False, omega=0.5,
    ).to(DEVICE).to(DTYPE)


def get_spin(N=6):
    up = N // 2
    return torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(DEVICE)


def make_psi_fn(f_net, bf_net, C_occ, params, spin):
    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net,
                        spin=spin, params=params)
        return lp
    return fn


# ══════════════════════════════════════════════════════════════════
#  Compact trainer — fast, prints every 5 epochs, tracks bf diagnostics
# ══════════════════════════════════════════════════════════════════

def train_fast(
    f_net, bf_net, C_occ, params, *,
    optimizer, n_epochs=100,
    lr_sched_fn=None,
    n_collocation=2048, oversampling=10, micro_batch=256,
    grad_clip=0.5, phase1_frac=0.0, alpha_end=0.60,
    print_every=5, patience=40, label="",
):
    device = params["device"]
    dtype  = params.get("torch_dtype", torch.float64)
    omega  = float(params["omega"])
    N      = int(params["n_particles"])
    d      = int(params["d"])
    sigma  = 1.3 / math.sqrt(omega)
    spin   = get_spin(N)

    psi_log_fn = make_psi_fn(f_net, bf_net, C_occ, params, spin)
    scheduler = lr_sched_fn(optimizer, n_epochs) if lr_sched_fn else None
    phase1_end = int(phase1_frac * n_epochs)

    n_f = sum(p.numel() for p in f_net.parameters() if p.requires_grad)
    n_bf = sum(p.numel() for p in bf_net.parameters() if p.requires_grad)
    lrs = [f"{pg['lr']:.1e}" for pg in optimizer.param_groups]

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, f={n_f:,} bf={n_bf:,} trainable, lr={lrs}")
    print(f"{'─'*55}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f = {}; best_bf = {}
    no_imp = 0

    for ep in range(n_epochs):
        if ep < phase1_end:
            alpha = 0.0
        else:
            t2 = (ep - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        f_net.eval(); bf_net.eval()
        X = screened_collocation(psi_log_fn, N, d, sigma,
                                 n_keep=n_collocation, oversampling=oversampling,
                                 device=device, dtype=dtype)

        f_net.train(); bf_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        nb = max(1, math.ceil(n_collocation / micro_batch))
        for i in range(0, n_collocation, micro_batch):
            E_L = compute_local_energy(psi_log_fn, X[i:i+micro_batch], omega).view(-1)
            good = torch.isfinite(E_L)
            if not good.all(): E_L = E_L[good]
            if E_L.numel() == 0: continue
            if E_L.numel() > 20:
                lo, hi = torch.quantile(E_L.detach(), 0.02), torch.quantile(E_L.detach(), 0.98)
                E_L = E_L[(E_L.detach() >= lo) & (E_L.detach() <= hi)]
                if E_L.numel() == 0: continue
            all_EL.append(E_L.detach())
            mu = E_L.mean().detach()
            E_eff = alpha * E_DMC + (1 - alpha) * mu
            loss = ((E_L - E_eff) ** 2).mean()
            (loss / nb).backward()

        all_p = [p for pg in optimizer.param_groups for p in pg["params"] if p.requires_grad]
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_p, grad_clip)
        optimizer.step()
        if scheduler: scheduler.step()

        if all_EL:
            EL = torch.cat(all_EL)
            E_mean, E_var, E_std = EL.mean().item(), EL.var().item(), EL.std().item()
        else:
            E_mean = E_var = E_std = float("nan")

        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_f = {k: v.clone() for k, v in f_net.state_dict().items()}
            best_bf = {k: v.clone() for k, v in bf_net.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if patience > 0 and no_imp >= patience and ep > phase1_end + 10:
            print(f"  Early stop ep {ep} (var={best_var:.3e})")
            break

        if ep % print_every == 0:
            dt = time.time() - t0
            lr0 = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / E_DMC * 100
            bf_s = bf_net.bf_scale.item()
            with torch.no_grad():
                bf_net.eval()
                bf_mag = bf_net(X[:64], spin).norm(dim=-1).mean().item()
                bf_net.train()
            bf_gn = sum(p.grad.data.norm(2).item()**2 for p in bf_net.parameters()
                        if p.grad is not None) ** 0.5
            f_gn = sum(p.grad.data.norm(2).item()**2 for p in f_net.parameters()
                       if p.grad is not None and p.requires_grad) ** 0.5
            print(f"  [{ep:3d}] E={E_mean:.4f}±{E_std:.3f} var={E_var:.2e} "
                  f"α={alpha:.2f} |Δx|={bf_mag:.3f} ‖∇bf‖={bf_gn:.3f} "
                  f"‖∇f‖={f_gn:.3f} err={err:.2f}%  ({dt:.0f}s)")
            sys.stdout.flush()

    if best_f: f_net.load_state_dict(best_f)
    if best_bf: bf_net.load_state_dict(best_bf)
    dt = time.time() - t0
    print(f"  Best var={best_var:.3e}, {dt:.0f}s ({dt/60:.1f}min)")
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  Cusp pre-training
# ══════════════════════════════════════════════════════════════════

def pretrain_cusp(bf_net, *, n_epochs=30, lr=1e-3, n_samples=4096,
                  strength=0.15, sigma_ell=0.5):
    spin = get_spin()
    sigma = 1.3 / math.sqrt(0.5)
    ell = 1.0 / math.sqrt(0.5)
    sig = sigma_ell * ell
    opt = torch.optim.Adam(bf_net.parameters(), lr=lr)

    print(f"\n  Cusp pre-train: {n_epochs}ep, strength={strength}")
    for ep in range(n_epochs):
        x = torch.randn(n_samples, 6, 2, device=DEVICE, dtype=DTYPE) * sigma

        # Target: push same-spin apart
        s = spin.view(1, 6, 1).float()
        si = s.unsqueeze(2).expand(-1, 6, 6, 1)
        sj = s.unsqueeze(1).expand(-1, 6, 6, 1)
        same = (si == sj).float().squeeze(-1)
        eye = torch.eye(6, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        mask = same * (1 - eye)

        r_ij = x.unsqueeze(1) - x.unsqueeze(2)
        r2 = (r_ij**2).sum(-1, keepdim=True)
        r1 = torch.sqrt(r2 + 1e-12)
        r_hat = r_ij / (r1 + 1e-8)
        env = torch.exp(-r2 / (2 * sig**2))
        dx_tgt = strength * (r_hat * env * mask.unsqueeze(-1)).sum(2)
        dx_tgt = dx_tgt - dx_tgt.mean(1, keepdim=True)

        dx_pred = bf_net(x, spin)
        loss = F.mse_loss(dx_pred, dx_tgt)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(bf_net.parameters(), 1.0)
        opt.step()

        if ep % 10 == 0:
            with torch.no_grad():
                mp = dx_pred.norm(dim=-1).mean().item()
                mt = dx_tgt.norm(dim=-1).mean().item()
            print(f"    [{ep:2d}] loss={loss.item():.5f} |pred|={mp:.3f} |tgt|={mt:.3f}")
    return bf_net


# ══════════════════════════════════════════════════════════════════
#  Fisher-preconditioned bf update
# ══════════════════════════════════════════════════════════════════

def train_fisher_bf(
    f_net, bf_net, C_occ, params, *,
    n_epochs=100, lr=5e-4, n_collocation=2048, oversampling=10,
    micro_batch=256, grad_clip=0.5, alpha_end=0.60,
    fisher_samples=256, fisher_damping=1e-3,
    print_every=5, patience=40, label="fisher_bf",
):
    """
    Backflow-only training with diagonal Fisher preconditioning.
    Fisher estimated from ∂log|Ψ|/∂θ_bf at collocation points.
    """
    device = params["device"]
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    sigma = 1.3 / math.sqrt(omega)
    spin = get_spin(N)
    psi_log_fn = make_psi_fn(f_net, bf_net, C_occ, params, spin)

    # Only bf params
    bf_params = [p for p in bf_net.parameters() if p.requires_grad]
    n_bf = sum(p.numel() for p in bf_params)

    optimizer = torch.optim.Adam(bf_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, {n_bf:,} bf trainable, Fisher damping={fisher_damping}")
    print(f"{'─'*55}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf"); best_f = {}; best_bf = {}; no_imp = 0

    for ep in range(n_epochs):
        t2 = ep / max(1, n_epochs - 1)
        alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        f_net.eval(); bf_net.eval()
        X = screened_collocation(psi_log_fn, N, d, sigma,
                                 n_keep=n_collocation, oversampling=oversampling,
                                 device=device, dtype=DTYPE)

        # ── Estimate diagonal Fisher from ∂log|Ψ|/∂θ_bf ──
        if ep % 5 == 0:   # recompute every 5 epochs
            bf_net.eval()
            fisher_diag = [torch.zeros_like(p) for p in bf_params]
            x_fisher = X[:fisher_samples].detach().requires_grad_(False)
            bs = 64
            n_fb = 0
            for i in range(0, min(fisher_samples, x_fisher.shape[0]), bs):
                xb = x_fisher[i:i+bs]
                for j in range(xb.shape[0]):
                    bf_net.zero_grad()
                    lp = psi_log_fn(xb[j:j+1])
                    lp.backward()
                    for k, p in enumerate(bf_params):
                        if p.grad is not None:
                            fisher_diag[k] += p.grad.data ** 2
                    n_fb += 1
            for k in range(len(fisher_diag)):
                fisher_diag[k] /= max(n_fb, 1)

        # ── Standard energy-variance loss ──
        f_net.train(); bf_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        nb = max(1, math.ceil(n_collocation / micro_batch))
        for i in range(0, n_collocation, micro_batch):
            E_L = compute_local_energy(psi_log_fn, X[i:i+micro_batch], omega).view(-1)
            good = torch.isfinite(E_L)
            if not good.all(): E_L = E_L[good]
            if E_L.numel() == 0: continue
            if E_L.numel() > 20:
                lo, hi = torch.quantile(E_L.detach(), 0.02), torch.quantile(E_L.detach(), 0.98)
                E_L = E_L[(E_L.detach() >= lo) & (E_L.detach() <= hi)]
                if E_L.numel() == 0: continue
            all_EL.append(E_L.detach())
            mu = E_L.mean().detach()
            E_eff = alpha * E_DMC + (1 - alpha) * mu
            loss = ((E_L - E_eff) ** 2).mean()
            (loss / nb).backward()

        # ── Apply Fisher preconditioning to gradients ──
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(bf_params, grad_clip)

        for k, p in enumerate(bf_params):
            if p.grad is not None:
                # Natural gradient: F^{-1} g  (diagonal approx)
                p.grad.data /= (fisher_diag[k] + fisher_damping)

        optimizer.step()
        scheduler.step()

        if all_EL:
            EL = torch.cat(all_EL)
            E_mean, E_var, E_std = EL.mean().item(), EL.var().item(), EL.std().item()
        else:
            E_mean = E_var = E_std = float("nan")

        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_f = {k: v.clone() for k, v in f_net.state_dict().items()}
            best_bf = {k: v.clone() for k, v in bf_net.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if patience > 0 and no_imp >= patience and ep > 10:
            print(f"  Early stop ep {ep} (var={best_var:.3e})")
            break

        if ep % print_every == 0:
            dt = time.time() - t0
            err = abs(E_mean - E_DMC) / E_DMC * 100
            bf_s = bf_net.bf_scale.item()
            with torch.no_grad():
                bf_net.eval()
                bf_mag = bf_net(X[:64], spin).norm(dim=-1).mean().item()
                bf_net.train()
            bf_gn = sum(p.grad.data.norm(2).item()**2 for p in bf_params
                        if p.grad is not None) ** 0.5
            print(f"  [{ep:3d}] E={E_mean:.4f}±{E_std:.3f} var={E_var:.2e} "
                  f"α={alpha:.2f} |Δx|={bf_mag:.3f} ‖∇bf‖={bf_gn:.3f} "
                  f"err={err:.2f}%  ({dt:.0f}s)")
            sys.stdout.flush()

    if best_f: f_net.load_state_dict(best_f)
    if best_bf: bf_net.load_state_dict(best_bf)
    dt = time.time() - t0
    print(f"  Best var={best_var:.3e}, {dt:.0f}s ({dt/60:.1f}min)")
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  Phase 0: Train + save PINN checkpoint
# ══════════════════════════════════════════════════════════════════

def train_and_save_pinn(C_occ, params):
    import os
    if os.path.exists(CKPT_PATH):
        print(f"Loading PINN checkpoint from {CKPT_PATH}")
        f_net = PINN(
            n_particles=6, d=2, omega=0.5,
            dL=8, hidden_dim=64, n_layers=2,
            act="gelu", init="xavier",
            use_gate=True, use_pair_attn=False,
        ).to(DEVICE).to(DTYPE)
        f_net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
        return f_net

    print("\n── Training PINN baseline (300ep) ──")
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(DEVICE).to(DTYPE)

    f_net, _, _ = train_residual(
        f_net, C_occ, params,
        n_epochs=300, lr=3e-4,
        n_collocation=2048, oversampling=10, micro_batch=256,
        grad_clip=0.5, print_every=20,
        phase1_frac=0.25, alpha_end=0.60,
        proposal_sigma_factor=1.3,
    )

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(f_net.state_dict(), CKPT_PATH)
    print(f"Saved PINN checkpoint to {CKPT_PATH}")
    return f_net


# ══════════════════════════════════════════════════════════════════
#  Experiments
# ══════════════════════════════════════════════════════════════════

def run_experiment(name, C_occ, params, pinn_state, experiment_fn):
    print(f"\n{'#'*55}")
    print(f"# {name}")
    print(f"{'#'*55}")

    # Fresh copies
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(DEVICE).to(DTYPE)
    f_net.load_state_dict(copy.deepcopy(pinn_state))

    return experiment_fn(f_net, C_occ, params)


def exp_bf_only(f_net, C_occ, params):
    """Freeze Jastrow, train bf only 100ep."""
    bf_net = make_bf()
    for p in f_net.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(bf_net.parameters(), lr=5e-4)
    sched = lambda o, n: torch.optim.lr_scheduler.CosineAnnealingLR(o, n, eta_min=1e-5)
    f_net, bf_net = train_fast(
        f_net, bf_net, C_occ, params,
        optimizer=opt, n_epochs=100, lr_sched_fn=sched,
        alpha_end=0.60, label="bf_only (Jastrow frozen, 100ep)",
    )
    for p in f_net.parameters(): p.requires_grad_(True)
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="bf_only")


def exp_cusp_bf(f_net, C_occ, params):
    """Cusp pre-train → freeze Jastrow → bf 100ep."""
    bf_net = make_bf()
    bf_net = pretrain_cusp(bf_net, n_epochs=30, strength=0.15)
    for p in f_net.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(bf_net.parameters(), lr=5e-4)
    sched = lambda o, n: torch.optim.lr_scheduler.CosineAnnealingLR(o, n, eta_min=1e-5)
    f_net, bf_net = train_fast(
        f_net, bf_net, C_occ, params,
        optimizer=opt, n_epochs=100, lr_sched_fn=sched,
        alpha_end=0.60, label="cusp+bf (cusp 30ep → Jastrow frozen → bf 100ep)",
    )
    for p in f_net.parameters(): p.requires_grad_(True)
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="cusp+bf")


def exp_fisher_bf(f_net, C_occ, params):
    """Freeze Jastrow, train bf with diagonal Fisher preconditioning."""
    bf_net = make_bf()
    for p in f_net.parameters(): p.requires_grad_(False)
    f_net, bf_net = train_fisher_bf(
        f_net, bf_net, C_occ, params,
        n_epochs=100, lr=5e-4,
        fisher_samples=256, fisher_damping=1e-3,
        label="fisher_bf (Jastrow frozen, Fisher precond, 100ep)",
    )
    for p in f_net.parameters(): p.requires_grad_(True)
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="fisher_bf")


def exp_bf_then_joint(f_net, C_occ, params):
    """Freeze Jastrow → bf 50ep → unfreeze → joint 80ep (f_lr low)."""
    bf_net = make_bf()

    # Phase 1: bf only, 50ep
    for p in f_net.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(bf_net.parameters(), lr=5e-4)
    sched = lambda o, n: torch.optim.lr_scheduler.CosineAnnealingLR(o, n, eta_min=5e-5)
    f_net, bf_net = train_fast(
        f_net, bf_net, C_occ, params,
        optimizer=opt, n_epochs=50, lr_sched_fn=sched,
        phase1_frac=0.3, alpha_end=0.40, patience=30,
        label="bf_then_joint phase1: bf-only 50ep",
    )

    # Phase 2: unfreeze both, sep opts, f_lr very low
    for p in f_net.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": f_net.parameters(), "lr": 5e-5},
        {"params": bf_net.parameters(), "lr": 2e-4},
    ])
    sched = lambda o, n: torch.optim.lr_scheduler.CosineAnnealingLR(o, n, eta_min=6e-6)
    f_net, bf_net = train_fast(
        f_net, bf_net, C_occ, params,
        optimizer=opt, n_epochs=80, lr_sched_fn=sched,
        phase1_frac=0.0, alpha_end=0.60,
        label="bf_then_joint phase2: joint 80ep (f_lr=5e-5, bf_lr=2e-4)",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="bf_then_joint")


def exp_cusp_bf_joint(f_net, C_occ, params):
    """Cusp → freeze → bf 50ep → unfreeze → joint 80ep."""
    bf_net = make_bf()
    bf_net = pretrain_cusp(bf_net, n_epochs=30, strength=0.15)

    # Phase 1: bf only, 50ep
    for p in f_net.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(bf_net.parameters(), lr=5e-4)
    sched = lambda o, n: torch.optim.lr_scheduler.CosineAnnealingLR(o, n, eta_min=5e-5)
    f_net, bf_net = train_fast(
        f_net, bf_net, C_occ, params,
        optimizer=opt, n_epochs=50, lr_sched_fn=sched,
        phase1_frac=0.3, alpha_end=0.40, patience=30,
        label="cusp+bf+joint phase1: cusp→bf-only 50ep",
    )

    # Phase 2: unfreeze
    for p in f_net.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": f_net.parameters(), "lr": 5e-5},
        {"params": bf_net.parameters(), "lr": 2e-4},
    ])
    sched = lambda o, n: torch.optim.lr_scheduler.CosineAnnealingLR(o, n, eta_min=6e-6)
    f_net, bf_net = train_fast(
        f_net, bf_net, C_occ, params,
        optimizer=opt, n_epochs=80, lr_sched_fn=sched,
        phase1_frac=0.0, alpha_end=0.60,
        label="cusp+bf+joint phase2: joint 80ep",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="cusp+bf+joint")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)

    # Train or load PINN
    f_net_base = train_and_save_pinn(C_occ, params)

    # Eval baseline
    print("\n── PINN baseline eval ──")
    evaluate(f_net_base, C_occ, params, label="PINN baseline (300ep)")

    pinn_state = copy.deepcopy(f_net_base.state_dict())

    experiments = [
        ("1. bf_only",        exp_bf_only),
        ("2. cusp+bf",        exp_cusp_bf),
        ("3. fisher_bf",      exp_fisher_bf),
        ("4. bf_then_joint",  exp_bf_then_joint),
        ("5. cusp+bf+joint",  exp_cusp_bf_joint),
    ]

    results = {}
    for name, fn in experiments:
        try:
            results[name] = run_experiment(name, C_occ, params, pinn_state, fn)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"E_mean": float("nan"), "E_stderr": float("nan")}

    print(f"\n{'='*65}")
    print("SUMMARY — bf extension from pre-trained PINN")
    print(f"{'='*65}")
    for name, r in results.items():
        E = r.get("E_mean", float("nan"))
        se = r.get("E_stderr", float("nan"))
        err = abs(E - E_DMC) / E_DMC * 100 if math.isfinite(E) else float("nan")
        print(f"  {name:25s}  E={E:.6f} ± {se:.6f}  err={err:.2f}%")
    print(f"  {'bf_0.7 joint (ref)':25s}  E=11.823253 ± 0.002982  err=0.33%")
    print(f"  {'PINN only (ref)':25s}  E=11.834691 ± 0.002782  err=0.42%")
    print(f"  {'DMC target':25s}  E={E_DMC:.6f}")
    print(f"{'='*65}")
