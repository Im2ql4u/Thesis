"""
Backflow scale + pre-training sweep.

Experiments:
  1. bf_0.7          — bf_scale_init=0.7, zero_init_last=False, joint opt
  2. bf_1.0          — bf_scale_init=1.0, zero_init_last=False, joint opt
  3. sep_opt_0.7     — bf_scale=0.7, separate optimizers (bf 5e-4, f 3e-4)
  4. cusp_pretrain   — pre-train bf to anti-bunch same-spin pairs, then joint
  5. pinn_pretrain   — PINN 300ep → freeze → bf 150ep → unfreeze 100ep

All use screened collocation, zero_init_last=False, 300ep effective.
"""

import math, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    train_residual,
)

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64


def make_nets(bf_scale_init=0.05, zero_init_last=True):
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


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    up = params["n_particles"] // 2
    N = params["n_particles"]
    device = params["device"]
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(device)
    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net,
                        spin=spin, params=params)
        return lp
    return fn, spin


# ══════════════════════════════════════════════════════════════════
#  Generic trainer (from bf_sweeps, extended)
# ══════════════════════════════════════════════════════════════════

def train_generic(
    f_net, bf_net, C_occ, params, *,
    optimizer, n_epochs=300,
    lr_sched_fn=None,
    n_collocation=2048, oversampling=10, micro_batch=256,
    grad_clip=0.5, phase1_frac=0.25, alpha_end=0.60,
    print_every=10, label="",
):
    device = params["device"]
    dtype  = params.get("torch_dtype", torch.float64)
    omega  = float(params["omega"])
    N      = int(params["n_particles"])
    d      = int(params["d"])
    ell    = 1.0 / math.sqrt(omega)
    sigma  = 1.3 * ell

    psi_log_fn, spin = make_psi_log_fn(f_net, bf_net, C_occ, params)

    if lr_sched_fn is not None:
        scheduler = lr_sched_fn(optimizer, n_epochs)
    else:
        scheduler = None

    phase1_end = int(phase1_frac * n_epochs)

    n_f = sum(p.numel() for p in f_net.parameters() if p.requires_grad)
    n_bf = sum(p.numel() for p in bf_net.parameters() if p.requires_grad)
    lrs = [f"{pg['lr']:.1e}" for pg in optimizer.param_groups]

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"  {n_epochs} ep, {n_collocation} pts, f={n_f:,} bf={n_bf:,} trainable")
    print(f"  optimizer groups: {len(optimizer.param_groups)}, lr={lrs}")
    bf_s = bf_net.bf_scale.item() if hasattr(bf_net, 'bf_scale') else 0
    print(f"  bf_scale_init={bf_s:.3f}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    patience = 60

    for epoch in range(n_epochs):
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

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
            all_p = [p for pg in optimizer.param_groups for p in pg["params"]
                     if p.requires_grad]
            nn.utils.clip_grad_norm_(all_p, grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
            bf_s = bf_net.bf_scale.item() if hasattr(bf_net, 'bf_scale') else 0
            bf_gnorm = sum(p.grad.data.norm(2).item()**2
                          for p in bf_net.parameters()
                          if p.grad is not None) ** 0.5
            f_gnorm = sum(p.grad.data.norm(2).item()**2
                         for p in f_net.parameters()
                         if p.grad is not None and p.requires_grad) ** 0.5

            # Measure actual backflow magnitude
            with torch.no_grad():
                bf_net.eval()
                dx_sample = bf_net(X[:64], spin)
                bf_mag = dx_sample.norm(dim=-1).mean().item()
                bf_net.train()

            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  lr={cur_lr:.1e}  "
                f"bf_s={bf_s:.3f} |Δx|={bf_mag:.4f}  "
                f"‖∇f‖={f_gnorm:.3f} ‖∇bf‖={bf_gnorm:.3f}  "
                f"err={err:.2f}%"
            )
            sys.stdout.flush()

    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)
    print(f"Restored best model (var={best_var:.3e})")
    total = time.time() - t0
    print(f"Training done in {total:.0f}s ({total/60:.1f}min)")
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  Cusp pre-training: teach backflow to anti-bunch same-spin pairs
# ══════════════════════════════════════════════════════════════════

def compute_antibunch_target(x, spin, strength=0.15, sigma_ell=0.5, omega=0.5):
    """
    Supervised target for backflow: push same-spin electrons apart.

    Δx_i^target = -c × Σ_{j same spin} (r_i - r_j)/|r_ij| × exp(-|r_ij|²/(2σ²))

    This encodes the exchange-correlation hole: same-spin electrons
    avoid each other, and backflow should learn this displacement.
    """
    B, N, d = x.shape
    ell = 1.0 / math.sqrt(omega)
    sigma = sigma_ell * ell

    # Spin mask: same spin pairs
    s = spin.view(1, N, 1) if spin.ndim == 1 else spin.unsqueeze(-1)
    s_i = s.unsqueeze(2).expand(B, N, N, 1).float()
    s_j = s.unsqueeze(1).expand(B, N, N, 1).float()
    same_spin = (s_i == s_j).float().squeeze(-1)  # (B,N,N)

    # Self-mask
    eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
    mask = same_spin * (1 - eye)  # (B,N,N)

    # Pairwise relative vectors
    r_ij = x.unsqueeze(1) - x.unsqueeze(2)  # (B,N,N,d) : r_i - r_j
    r2 = (r_ij ** 2).sum(-1, keepdim=True)   # (B,N,N,1)
    r1 = torch.sqrt(r2 + 1e-12)              # (B,N,N,1)

    # Direction: unit vector from j toward i (pushes i away from j)
    r_hat = r_ij / (r1 + 1e-8)  # (B,N,N,d)

    # Gaussian envelope: strong push when close, decays with distance
    envelope = torch.exp(-r2 / (2 * sigma ** 2))  # (B,N,N,1)

    # Weighted displacement: sum over same-spin neighbors
    dx_target = strength * (r_hat * envelope * mask.unsqueeze(-1)).sum(dim=2)  # (B,N,d)

    # Zero center-of-mass
    dx_target = dx_target - dx_target.mean(dim=1, keepdim=True)

    return dx_target


def pretrain_backflow_cusp(bf_net, spin, N, d, omega, *,
                           n_epochs=50, lr=1e-3, n_samples=4096,
                           bf_target_strength=0.15, sigma_ell=0.5):
    """
    Supervised pre-training: teach backflow the anti-bunching pattern.
    """
    ell = 1.0 / math.sqrt(omega)
    sigma_proposal = 1.3 * ell
    device = next(bf_net.parameters()).device
    dtype = next(bf_net.parameters()).dtype

    optimizer = torch.optim.Adam(bf_net.parameters(), lr=lr)

    print(f"\n{'─'*50}")
    print(f"Cusp pre-training: {n_epochs} ep, {n_samples} samples")
    print(f"  target: anti-bunch same-spin, strength={bf_target_strength}")
    print(f"{'─'*50}")

    t0 = time.time()
    for epoch in range(n_epochs):
        # Random configurations from Gaussian
        x = torch.randn(n_samples, N, d, device=device, dtype=dtype) * sigma_proposal

        # Target displacement
        dx_target = compute_antibunch_target(
            x, spin, strength=bf_target_strength,
            sigma_ell=sigma_ell, omega=omega,
        )

        # Predicted displacement
        dx_pred = bf_net(x, spin)

        # MSE loss
        loss = F.mse_loss(dx_pred, dx_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(bf_net.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                mag_pred = dx_pred.norm(dim=-1).mean().item()
                mag_tgt = dx_target.norm(dim=-1).mean().item()
            print(f"  [pretrain {epoch:3d}] loss={loss.item():.6f}  "
                  f"|Δx_pred|={mag_pred:.4f}  |Δx_tgt|={mag_tgt:.4f}  "
                  f"bf_scale={bf_net.bf_scale.item():.4f}")

    dt = time.time() - t0
    print(f"  Pre-training done in {dt:.0f}s")
    return bf_net


# ══════════════════════════════════════════════════════════════════
#  Cosine schedule factory
# ══════════════════════════════════════════════════════════════════

def cosine_sched(opt, n_epochs, eta_min=6e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=eta_min)


# ══════════════════════════════════════════════════════════════════
#  Experiment 1: bf_scale=0.7, zero_init_last=False
# ══════════════════════════════════════════════════════════════════

def run_bf_0_7():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets(bf_scale_init=0.7, zero_init_last=False)
    np_total = sum(p.numel() for p in f_net.parameters()) + sum(p.numel() for p in bf_net.parameters())
    print(f"CTNN+PINN params: {np_total:,}")

    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=3e-4)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=300,
        lr_sched_fn=lambda o, n: cosine_sched(o, n),
        label="bf_scale=0.7, zero_init=False",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="bf_0.7")


# ══════════════════════════════════════════════════════════════════
#  Experiment 2: bf_scale=1.0, zero_init_last=False
# ══════════════════════════════════════════════════════════════════

def run_bf_1_0():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets(bf_scale_init=1.0, zero_init_last=False)
    np_total = sum(p.numel() for p in f_net.parameters()) + sum(p.numel() for p in bf_net.parameters())
    print(f"CTNN+PINN params: {np_total:,}")

    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=3e-4)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=300,
        lr_sched_fn=lambda o, n: cosine_sched(o, n),
        label="bf_scale=1.0, zero_init=False",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="bf_1.0")


# ══════════════════════════════════════════════════════════════════
#  Experiment 3: separate opts, bf_scale=0.7
# ══════════════════════════════════════════════════════════════════

def run_sep_opt_0_7():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets(bf_scale_init=0.7, zero_init_last=False)
    np_total = sum(p.numel() for p in f_net.parameters()) + sum(p.numel() for p in bf_net.parameters())
    print(f"CTNN+PINN params: {np_total:,}")

    optimizer = torch.optim.Adam([
        {"params": f_net.parameters(), "lr": 3e-4},
        {"params": bf_net.parameters(), "lr": 5e-4},
    ])

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=300,
        lr_sched_fn=lambda o, n: cosine_sched(o, n),
        label="sep_opt: bf_lr=5e-4, f_lr=3e-4, bf_scale=0.7",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="sep_opt_0.7")


# ══════════════════════════════════════════════════════════════════
#  Experiment 4: cusp pre-training + joint training
# ══════════════════════════════════════════════════════════════════

def run_cusp_pretrain():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)
    f_net, bf_net = make_nets(bf_scale_init=0.7, zero_init_last=False)
    np_total = sum(p.numel() for p in f_net.parameters()) + sum(p.numel() for p in bf_net.parameters())
    print(f"CTNN+PINN params: {np_total:,}")

    N = params["n_particles"]
    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(DEVICE)

    # Phase 1: pre-train backflow on anti-bunching target
    bf_net = pretrain_backflow_cusp(
        bf_net, spin, N=6, d=2, omega=0.5,
        n_epochs=50, lr=1e-3, n_samples=4096,
        bf_target_strength=0.15, sigma_ell=0.5,
    )

    # Phase 2: joint training (300ep)
    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=3e-4)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=300,
        lr_sched_fn=lambda o, n: cosine_sched(o, n),
        label="cusp_pretrain → joint 300ep, bf_scale=0.7",
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="cusp_pretrain")


# ══════════════════════════════════════════════════════════════════
#  Experiment 5: PINN pre-train → freeze → bf → unfreeze
# ══════════════════════════════════════════════════════════════════

def run_pinn_pretrain():
    C_occ, params = setup_noninteracting(6, 0.5, device=DEVICE, dtype=DTYPE)

    # Phase A: full PINN training (300ep)
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(DEVICE).to(DTYPE)

    print("\n── Phase A: PINN-only (300 ep) ──")
    f_net, _, _ = train_residual(
        f_net, C_occ, params,
        n_epochs=300, lr=3e-4,
        n_collocation=2048, oversampling=10, micro_batch=256,
        grad_clip=0.5, print_every=20,
        phase1_frac=0.25, alpha_end=0.60,
        proposal_sigma_factor=1.3,
    )
    print("\n── PINN baseline eval ──")
    evaluate(f_net, C_occ, params, label="PINN-only baseline (300ep)")

    # Phase B: freeze Jastrow, train bf only (150ep)
    print("\n── Phase B: Freeze Jastrow, train bf only (150 ep) ──")
    bf_net = CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=0.7,
        zero_init_last=False,
        omega=0.5,
    ).to(DEVICE).to(DTYPE)

    for p in f_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(bf_net.parameters(), lr=5e-4)
    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=150,
        lr_sched_fn=lambda o, n: cosine_sched(o, n, eta_min=1e-5),
        phase1_frac=0.0, alpha_end=0.60,
        label="pinn_pretrain: bf-only (Jastrow frozen), bf_scale=0.7",
    )

    print("\n── After bf-only eval ──")
    for p in f_net.parameters():
        p.requires_grad_(True)
    evaluate(f_net, C_occ, params, backflow_net=bf_net,
             label="after bf-only phase")

    # Phase C: unfreeze both, fine-tune (100ep)
    print("\n── Phase C: Unfreeze both, fine-tune (100 ep) ──")
    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-4)

    f_net, bf_net = train_generic(
        f_net, bf_net, C_occ, params,
        optimizer=optimizer, n_epochs=100,
        lr_sched_fn=lambda o, n: cosine_sched(o, n, eta_min=6e-6),
        phase1_frac=0.0, alpha_end=0.60,
        grad_clip=0.3,
        label="pinn_pretrain: fine-tune both (100ep)",
    )

    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="pinn_pretrain")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    experiments = [
        ("bf_0.7",          run_bf_0_7),
        ("bf_1.0",          run_bf_1_0),
        ("sep_opt_0.7",     run_sep_opt_0_7),
        ("cusp_pretrain",   run_cusp_pretrain),
        ("pinn_pretrain",   run_pinn_pretrain),
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

    print(f"\n{'='*70}")
    print("SUMMARY — 6e ω=0.5   bf_scale + pre-training sweep")
    print(f"{'='*70}")
    for name, r in results.items():
        E = r.get("E_mean", float("nan"))
        se = r.get("E_stderr", float("nan"))
        err = abs(E - E_DMC) / E_DMC * 100 if math.isfinite(E) else float("nan")
        print(f"  {name:22s}  E={E:.6f} ± {se:.6f}  err={err:.2f}%")
    print(f"  {'bf_0.2 (prev ref)':22s}  E=11.857245 ± 0.003522  err=0.61%")
    print(f"  {'PINN only (ref)':22s}  E=11.834691 ± 0.002782  err=0.42%")
    print(f"  {'CTNN 300ep (ref)':22s}  E=11.848922 ± 0.002811  err=0.54%")
    print(f"  {'DMC target':22s}  E={E_DMC:.6f}")
    print(f"{'='*70}")
