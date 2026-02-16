"""
Quick backflow diagnostic sweeps.

Test 6 theories on why backflow hurts with Adam / residual training:
  1. sep_opt_low_bf   — two optimizers, bf_lr = lr/10
  2. sep_opt_high_bf  — two optimizers, bf_lr = lr*3
  3. freeze_finetune  — train PINN 200ep, freeze it, train only bf 150ep
  4. grad_scale       — scale bf gradients by 0.1× via hooks
  5. large_bf_scale   — bf_scale_init=0.20 (default 0.05)
  6. tiny_bf_scale    — bf_scale_init=0.01

All use screened collocation (same as run_6e_residual.py).
Each run ~150ep to keep total time manageable.
"""

import math, sys, time, copy
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

# ── Import helpers from residual script ──
from run_6e_residual import (
    setup_noninteracting,
    compute_local_energy,
    screened_collocation,
    evaluate,
)

E_DMC = 11.78484
N_PARTICLES = 6
OMEGA = 0.5
DEVICE = "cpu"
DTYPE = torch.float64


def make_nets(bf_scale_init=0.05):
    """Create fresh PINN + CTNNBackflowNet."""
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
        omega=0.5,
    ).to(DEVICE).to(DTYPE)
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params, spin):
    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net,
                        spin=spin, params=params)
        return lp
    return psi_log_fn


# ══════════════════════════════════════════════════════════════════
#  Generic trainer with configurable optimizer setup
# ══════════════════════════════════════════════════════════════════

def train_generic(
    f_net, bf_net, C_occ, params, *,
    optimizer,
    n_epochs=150,
    lr_sched_fn=None,        # callable(optimizer, n_epochs) -> scheduler
    n_collocation=2048,
    oversampling=10,
    micro_batch=256,
    grad_clip=0.5,
    phase1_frac=0.25,
    alpha_end=0.60,
    print_every=10,
    grad_hooks=None,          # list of hooks to register (removed after training)
    label="",
):
    """Screened-collocation trainer with pluggable optimizer/hooks."""
    device = params["device"]
    dtype  = params.get("torch_dtype", torch.float64)
    omega  = float(params["omega"])
    N      = int(params["n_particles"])
    d      = int(params["d"])
    ell    = 1.0 / math.sqrt(omega)
    sigma  = 1.3 * ell

    up   = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(device)
    psi_log_fn = make_psi_log_fn(f_net, bf_net, C_occ, params, spin)

    if lr_sched_fn is not None:
        scheduler = lr_sched_fn(optimizer, n_epochs)
    else:
        scheduler = None

    phase1_end = int(phase1_frac * n_epochs)

    # Count params with grad
    n_p = sum(p.numel() for p in f_net.parameters() if p.requires_grad)
    n_bf = sum(p.numel() for p in bf_net.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"  {n_epochs} ep, {n_collocation} pts, f_net={n_p:,} bf_net={n_bf:,} trainable")
    lrs = [pg.get('lr', '?') for pg in optimizer.param_groups]
    print(f"  optimizer groups: {len(optimizer.param_groups)}, lr={lrs}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    patience = 50

    for epoch in range(n_epochs):
        # Alpha schedule
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # Screened collocation
        f_net.eval(); bf_net.eval()
        X = screened_collocation(
            psi_log_fn, N, d, sigma,
            n_keep=n_collocation, oversampling=oversampling,
            device=device, dtype=dtype,
        )

        f_net.train(); bf_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        n_batches = max(1, math.ceil(n_collocation / micro_batch))
        for i in range(0, n_collocation, micro_batch):
            x_mb = X[i:i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega).view(-1)
            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue
            # Quantile trim
            if E_L.numel() > 20:
                lo = torch.quantile(E_L.detach(), 0.02)
                hi = torch.quantile(E_L.detach(), 0.98)
                mask = (E_L.detach() >= lo) & (E_L.detach() <= hi)
                E_L = E_L[mask]
                if E_L.numel() == 0:
                    continue
            all_EL.append(E_L.detach())
            mu = E_L.mean().detach()
            E_eff = alpha * E_DMC + (1.0 - alpha) * mu
            resid = E_L - E_eff
            loss_mb = (resid ** 2).mean()
            (loss_mb / n_batches).backward()

        if grad_clip > 0:
            all_p = [p for p in list(f_net.parameters()) + list(bf_net.parameters())
                     if p.requires_grad]
            nn.utils.clip_grad_norm_(all_p, grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Logging
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

        if patience > 0 and epochs_no_improve >= patience and epoch > phase1_end + 20:
            print(f"  Early stop at epoch {epoch}  (best var={best_var:.3e})")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / E_DMC * 100
            # Check bf_scale
            bf_s = bf_net.bf_scale.item() if hasattr(bf_net, 'bf_scale') else 0
            # Check bf gradient norm
            bf_gnorm = 0.0
            for p in bf_net.parameters():
                if p.grad is not None:
                    bf_gnorm += p.grad.data.norm(2).item() ** 2
            bf_gnorm = bf_gnorm ** 0.5
            f_gnorm = 0.0
            for p in f_net.parameters():
                if p.grad is not None:
                    f_gnorm += p.grad.data.norm(2).item() ** 2
            f_gnorm = f_gnorm ** 0.5

            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  lr={cur_lr:.1e}  "
                f"bf_s={bf_s:.4f}  ‖∇f‖={f_gnorm:.3f} ‖∇bf‖={bf_gnorm:.3f}  "
                f"err={err:.2f}%"
            )
            sys.stdout.flush()

    # Restore best
    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)
    print(f"Restored best model (var={best_var:.3e})")
    total = time.time() - t0
    print(f"Training done in {total:.0f}s ({total/60:.1f}min)")

    # Remove hooks
    if grad_hooks:
        for h in grad_hooks:
            h.remove()

    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  Experiment 1: Separate optimizers — bf LR = lr/10
# ══════════════════════════════════════════════════════════════════

def run_sep_opt_low_bf():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets()
    lr = 3e-4
    optimizer = torch.optim.Adam([
        {"params": f_net.parameters(), "lr": lr},
        {"params": bf_net.parameters(), "lr": lr / 10},
    ])
    n_epochs = 200
    def sched_fn(opt, ne):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, ne, eta_min=6e-6)
    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=n_epochs, lr_sched_fn=sched_fn,
        label="sep_opt: bf_lr = lr/10",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="sep_opt_low_bf")


# ══════════════════════════════════════════════════════════════════
#  Experiment 2: Separate optimizers — bf LR = lr × 3
# ══════════════════════════════════════════════════════════════════

def run_sep_opt_high_bf():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets()
    lr = 3e-4
    optimizer = torch.optim.Adam([
        {"params": f_net.parameters(), "lr": lr},
        {"params": bf_net.parameters(), "lr": lr * 3},
    ])
    n_epochs = 200
    def sched_fn(opt, ne):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, ne, eta_min=6e-6)
    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=n_epochs, lr_sched_fn=sched_fn,
        label="sep_opt: bf_lr = lr×3",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="sep_opt_high_bf")


# ══════════════════════════════════════════════════════════════════
#  Experiment 3: Freeze Jastrow, fine-tune backflow only
# ══════════════════════════════════════════════════════════════════

def run_freeze_finetune():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)

    # Phase A: Train PINN-only for 200 epochs
    f_net_pinn = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(DEVICE).to(DTYPE)

    from run_6e_residual import train_residual, COMMON
    print("\n── Phase A: PINN-only (200 ep) ──")
    f_net_pinn, _, _ = train_residual(
        f_net_pinn, C_occ, params,
        n_epochs=200, lr=3e-4,
        n_collocation=2048, oversampling=10, micro_batch=256,
        grad_clip=0.5, print_every=20,
        phase1_frac=0.25, alpha_end=0.60,
        proposal_sigma_factor=1.3,
    )

    # Quick eval of PINN-only
    print("\n── PINN-only eval (before backflow) ──")
    evaluate(f_net_pinn, C_occ, params, label="PINN-only baseline")

    # Phase B: Freeze f_net, attach bf_net, train bf only
    print("\n── Phase B: Freeze Jastrow, train backflow only (150 ep) ──")
    bf_net = CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=0.05,
        omega=0.5,
    ).to(DEVICE).to(DTYPE)

    # Freeze all Jastrow params
    for p in f_net_pinn.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(bf_net.parameters(), lr=5e-4)
    def sched_fn(opt, ne):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, ne, eta_min=1e-5)

    f_net_pinn, bf_net = train_generic(
        f_net_pinn, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=150, lr_sched_fn=sched_fn,
        label="freeze_finetune: bf-only (Jastrow frozen)",
    )

    # Unfreeze for eval
    for p in f_net_pinn.parameters():
        p.requires_grad_(True)

    return evaluate(f_net_pinn, C_occ, params, backflow_net=bf_net,
                    label="freeze_finetune")


# ══════════════════════════════════════════════════════════════════
#  Experiment 4: Gradient scaling — bf grads × 0.1
# ══════════════════════════════════════════════════════════════════

def run_grad_scale():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets()

    # Register gradient scaling hooks on backflow parameters
    hooks = []
    scale_factor = 0.1
    for p in bf_net.parameters():
        h = p.register_hook(lambda grad: grad * scale_factor)
        hooks.append(h)

    lr = 3e-4
    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    def sched_fn(opt, ne):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, ne, eta_min=6e-6)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=200, lr_sched_fn=sched_fn,
        grad_hooks=hooks,
        label="grad_scale: bf grads × 0.1",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="grad_scale_0.1")


# ══════════════════════════════════════════════════════════════════
#  Experiment 5: Large bf_scale_init = 0.20
# ══════════════════════════════════════════════════════════════════

def run_large_bf_scale():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets(bf_scale_init=0.20)

    lr = 3e-4
    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    def sched_fn(opt, ne):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, ne, eta_min=6e-6)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=200, lr_sched_fn=sched_fn,
        label="large_bf_scale: bf_scale_init=0.20",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="large_bf_scale_0.20")


# ══════════════════════════════════════════════════════════════════
#  Experiment 6: Tiny bf_scale_init = 0.01
# ══════════════════════════════════════════════════════════════════

def run_tiny_bf_scale():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets(bf_scale_init=0.01)

    lr = 3e-4
    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    def sched_fn(opt, ne):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, ne, eta_min=6e-6)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=200, lr_sched_fn=sched_fn,
        label="tiny_bf_scale: bf_scale_init=0.01",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="tiny_bf_scale_0.01")


# ══════════════════════════════════════════════════════════════════
#  BONUS: Joint baseline (same as run_6e_residual CTNN, shorter)
# ══════════════════════════════════════════════════════════════════

def run_joint_baseline():
    """Standard joint training as control — same total epochs."""
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets()

    lr = 3e-4
    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    def sched_fn(opt, ne):
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, ne, eta_min=6e-6)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=200, lr_sched_fn=sched_fn,
        label="joint_baseline: same optimizer, same lr",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="joint_baseline")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    experiments = [
        ("joint_baseline",     run_joint_baseline),       # control
        ("sep_opt_low_bf",     run_sep_opt_low_bf),       # bf_lr = lr/10
        ("sep_opt_high_bf",    run_sep_opt_high_bf),      # bf_lr = lr*3
        ("grad_scale_0.1",     run_grad_scale),           # bf grads × 0.1
        ("large_bf_scale",     run_large_bf_scale),       # bf_scale_init=0.20
        ("tiny_bf_scale",      run_tiny_bf_scale),        # bf_scale_init=0.01
        ("freeze_finetune",    run_freeze_finetune),      # PINN→freeze→bf only
    ]

    results = {}
    for name, fn in experiments:
        print(f"\n{'#'*60}")
        print(f"# {name.upper()}")
        print(f"{'#'*60}")
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"E_mean": float("nan"), "E_stderr": float("nan")}

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY — 6e ω=0.5   Backflow sweep experiments")
    print(f"{'='*70}")
    for name, r in results.items():
        E = r.get("E_mean", float("nan"))
        se = r.get("E_stderr", float("nan"))
        err = abs(E - E_DMC) / E_DMC * 100 if math.isfinite(E) else float("nan")
        print(f"  {name:22s}  E={E:.6f} ± {se:.6f}  err={err:.2f}%")
    print(f"  {'PINN only (ref)':22s}  E=11.834691 ± 0.002782  err=0.42%")
    print(f"  {'CTNN 300ep (ref)':22s}  E=11.848922 ± 0.002811  err=0.54%")
    print(f"  {'DMC target':22s}  E={E_DMC:.6f}")
    print(f"{'='*70}")
