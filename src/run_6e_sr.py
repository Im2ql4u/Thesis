"""
Last-resort SR (Stochastic Reconfiguration) on collocation.

The idea: VMC+SR works brilliantly with backflow because the Fisher matrix
F_ij = Cov[O_i, O_j] (where O_i = ∂logψ/∂θ_i) naturally damps the
ill-conditioned directions near nodes. The Fisher "knows" which parameter
directions cause large changes in ψ near nodes and down-weights them.

Previous K-FAC attempt failed (0.34%) because:
  - K-FAC is a crude Kronecker approximation, only on BF net
  - It doesn't capture the cross-correlations between Jastrow and BF

This test: EXACT low-rank Fisher over ALL parameters.
  - Compute per-sample ∇_θ logψ for B samples → matrix J (B × P)
  - Fisher = J^T J / B  (empirical, rank ≤ B)
  - Preconditioned gradient: (F + εI)^{-1} g  via Woodbury identity
  - Apply to the LOSS gradient (still variance minimization, not VMC energy)

This is SR-preconditioned collocation: the gradient direction is SR-natural,
but the objective is still residual-based. If this works, it means the
Fisher metric was the missing ingredient. If it doesn't, the catch-22 is
confirmed: the problem isn't the gradient direction, it's the objective.

Quick test: 30ep fine-tune from best model.
"""

import math, sys, time, os
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

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"
N_PARTICLES, DIM, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)


def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
    n_occ = N // 2
    nx = {2: 2, 6: 3, 12: 4, 20: 5}.get(N, 4)
    ny = nx
    n_basis = nx * ny
    L = max(8.0, 3.0 / math.sqrt(omega))
    config.update(omega=omega, n_particles=N, d=d, L=L, n_grid=80,
                  nx=nx, ny=ny, basis="cart", device=str(device), dtype="float64")
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
    print(f"N={N}, w={omega}  E_DMC={E_DMC}")
    return C_occ, params


def make_nets(bf_scale_init=0.7, zero_init_last=False):
    f_net = PINN(n_particles=N_PARTICLES, d=DIM, omega=OMEGA,
                 dL=8, hidden_dim=64, n_layers=2, act="gelu", init="xavier",
                 use_gate=True, use_pair_attn=False).to(DEVICE).to(DTYPE)
    bf_net = CTNNBackflowNet(d=DIM, msg_hidden=32, msg_layers=1,
                             hidden=32, layers=2, act="silu", aggregation="sum",
                             use_spin=True, same_spin_only=False,
                             out_bound="tanh", bf_scale_init=bf_scale_init,
                             zero_init_last=zero_init_last, omega=OMEGA).to(DEVICE).to(DTYPE)
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    up = N_PARTICLES // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N_PARTICLES - up, dtype=torch.long)]).to(DEVICE)
    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net, spin=spin, params=params)
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


def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    print(f"\n-- VMC eval: {label} --")
    result = evaluate_energy_vmc(
        f_net, C_occ, psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=backflow_net, params=params,
        n_samples=n_samples, batch_size=512,
        sampler_steps=50, sampler_step_sigma=0.12, lap_mode="exact",
        persistent=True, sampler_burn_in=300, sampler_thin=3, progress=True)
    E, E_std = result["E_mean"], result["E_stderr"]
    err = abs(E - E_DMC) / E_DMC * 100
    print(f"  E = {E:.6f} +/- {E_std:.6f}  (target {E_DMC}, err {err:.2f}%)")
    return result


@torch.no_grad()
def topk_collocation(psi_log_fn, n_keep, oversampling, sigma, batch_size=4096):
    M = oversampling * n_keep
    x = torch.randn(M, N_PARTICLES, DIM, device=DEVICE, dtype=DTYPE) * sigma
    Nd = N_PARTICLES * DIM
    log_q = -0.5 * Nd * math.log(2 * math.pi * sigma**2) - x.reshape(M, -1).pow(2).sum(-1) / (2 * sigma**2)
    lp_parts = []
    for i in range(0, M, batch_size):
        lp_parts.append(psi_log_fn(x[i:i+batch_size]))
    log_psi = torch.cat(lp_parts)
    log_w = 2.0 * log_psi - log_q
    log_w[~torch.isfinite(log_w)] = -1e10
    _, idx = torch.topk(log_w, n_keep)
    return x[idx].clone()


def compute_bf_smoothness_penalty(bf_net, x, spin, n_samples=32):
    x_sub = x[:n_samples].detach().requires_grad_(True)
    dx = bf_net(x_sub, spin)
    lap_sq = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    for _ in range(2):
        v = torch.empty_like(x_sub).bernoulli_(0.5).mul_(2).add_(-1)
        for k in range(DIM):
            g1 = torch.autograd.grad(dx[:,:,k].sum(), x_sub, create_graph=True, retain_graph=True)[0]
            Hv = torch.autograd.grad((g1*v).sum(), x_sub, create_graph=True, retain_graph=True)[0]
            lap_sq = lap_sq + ((v*Hv).sum(dim=(1,2))**2).mean()
    return lap_sq / (2 * DIM)


# ══════════════════════════════════════════════════════════════════
#  SR preconditioner: exact low-rank Fisher via Woodbury
# ══════════════════════════════════════════════════════════════════

def compute_sr_update(all_params_list, psi_log_fn, x_fisher, loss_grad_flat,
                      damping=1e-3, max_fisher_samples=128):
    """
    SR-preconditioned gradient: (F + εI)^{-1} g

    F = J^T J / B  where J_bi = ∂logψ(x_b)/∂θ_i  (B × P matrix)

    Via Woodbury:
      (F + εI)^{-1} g = (1/ε)[g - J^T (B·ε·I + J J^T)^{-1} J g]

    Cost: O(B² P) instead of O(P³). With B=128 and P=24K, this is fast.
    """
    B = min(x_fisher.shape[0], max_fisher_samples)
    x_f = x_fisher[:B]
    P = loss_grad_flat.numel()

    # Compute Jacobian J (B × P): per-sample ∂logψ/∂θ
    J_rows = []
    for b in range(B):
        for p in all_params_list:
            if p.grad is not None:
                p.grad = None
        xb = x_f[b:b+1]
        lp = psi_log_fn(xb)
        lp.backward()
        row = torch.cat([p.grad.flatten() for p in all_params_list if p.grad is not None])
        J_rows.append(row)

    J = torch.stack(J_rows)  # (B, P)

    # Center (remove mean → covariance not just second moment)
    J = J - J.mean(dim=0, keepdim=True)

    g = loss_grad_flat.clone()

    # Woodbury: (J^T J / B + εI)^{-1} g = (1/ε)[g - J^T (B·ε·I + J J^T)^{-1} J g]
    Jg = J @ g                         # (B,)
    K = J @ J.T                         # (B, B)
    K_reg = K + B * damping * torch.eye(B, device=DEVICE, dtype=DTYPE)
    solve = torch.linalg.solve(K_reg, Jg)  # (B,)
    sr_update = (g - J.T @ solve) / damping

    # Diagnostics
    eigs = torch.linalg.eigvalsh(K)
    cond = eigs[-1] / max(eigs[0].item(), 1e-30)

    return sr_update, {
        "fisher_cond": cond.item(),
        "fisher_rank_eff": (eigs > eigs[-1] * 1e-6).sum().item(),
        "fisher_top_eig": eigs[-1].item(),
        "fisher_bot_eig": eigs[0].item(),
        "update_norm": sr_update.norm().item(),
        "grad_norm": g.norm().item(),
        "ratio": sr_update.norm().item() / max(g.norm().item(), 1e-30),
    }


# ══════════════════════════════════════════════════════════════════
#  Training with SR preconditioning
# ══════════════════════════════════════════════════════════════════

def train_sr(f_net, bf_net, C_occ, params, *,
             n_epochs=30, lr=3e-4, n_collocation=2048, oversampling=10,
             sigma=None, damping=1e-3, n_fisher=128,
             huber_delta=0.5, smooth_lambda=1e-3,
             micro_batch=256, grad_clip=0.5, quantile_trim=0.02,
             phase1_frac=0.25, alpha_end=0.60,
             print_every=5, label=""):

    omega = OMEGA
    if sigma is None:
        sigma = 1.3 * ELL
    psi_log_fn, spin = make_psi_log_fn(f_net, bf_net, C_occ, params)

    all_params = list(f_net.parameters()) + list(bf_net.parameters())
    # We use plain SGD since SR provides the preconditioning
    # (Adam would fight with the Fisher)
    n_f = sum(p.numel() for p in f_net.parameters())
    n_bf = sum(p.numel() for p in bf_net.parameters())
    P = sum(p.numel() for p in all_params)
    phase1_end = int(phase1_frac * n_epochs)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, P={P:,} (f={n_f:,} bf={n_bf:,})")
    print(f"  SR damping={damping}, n_fisher={n_fisher}, lr={lr}")
    print(f"{'='*70}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state, best_bf_state = {}, {}

    for epoch in range(n_epochs):
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # Sample
        f_net.eval(); bf_net.eval()
        X = topk_collocation(psi_log_fn, n_collocation, oversampling, sigma)
        n_pts = X.shape[0]

        # === Step 1: compute vanilla loss gradient ===
        f_net.train(); bf_net.train()
        for p in all_params:
            p.grad = None

        all_EL = []
        n_batches = max(1, math.ceil(n_pts / micro_batch))
        for i in range(0, n_pts, micro_batch):
            x_mb = X[i:i+micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega).view(-1)
            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue
            if quantile_trim > 0 and E_L.numel() > 20:
                lo = torch.quantile(E_L.detach(), quantile_trim)
                hi = torch.quantile(E_L.detach(), 1 - quantile_trim)
                m = (E_L.detach() >= lo) & (E_L.detach() <= hi)
                E_L = E_L[m]
                if E_L.numel() == 0:
                    continue
            all_EL.append(E_L.detach())
            mu = E_L.mean().detach()
            E_eff = alpha * E_DMC + (1 - alpha) * mu
            resid = E_L - E_eff
            loss_mb = F.huber_loss(resid, torch.zeros_like(resid), delta=huber_delta)
            if smooth_lambda > 0:
                pen = compute_bf_smoothness_penalty(bf_net, x_mb, spin)
                loss_mb = loss_mb + smooth_lambda * pen
            (loss_mb / n_batches).backward()

        if len(all_EL) == 0:
            continue
        EL_cat = torch.cat(all_EL)
        E_mean = EL_cat.mean().item()
        E_var = EL_cat.var().item()

        # Collect vanilla gradient
        vanilla_grad = torch.cat([p.grad.flatten() for p in all_params if p.grad is not None])

        # === Step 2: SR preconditioning ===
        for p in all_params:
            p.grad = None

        sr_update, sr_diag = compute_sr_update(
            all_params, psi_log_fn, X, vanilla_grad,
            damping=damping, max_fisher_samples=n_fisher)

        # Clip SR update
        sr_norm = sr_update.norm()
        if sr_norm > grad_clip:
            sr_update = sr_update * (grad_clip / sr_norm)

        # === Step 3: apply update ===
        offset = 0
        with torch.no_grad():
            for p in all_params:
                n = p.numel()
                p.add_(sr_update[offset:offset+n].view_as(p), alpha=-lr)
                offset += n

        # Logging
        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_f_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            best_bf_state = {k: v.clone() for k, v in bf_net.state_dict().items()}

        if epoch % print_every == 0:
            err = abs(E_mean - E_DMC) / E_DMC * 100
            with torch.no_grad():
                bf_net.eval()
                bf_mag = bf_net(X[:64], spin).norm(dim=-1).mean().item()
                bf_net.train()
            print(
                f"[{epoch:3d}] E={E_mean:.5f} var={E_var:.3e} "
                f"|dx|={bf_mag:.3f} err={err:.2f}% "
                f"F_cond={sr_diag['fisher_cond']:.1e} "
                f"rank={sr_diag['fisher_rank_eff']} "
                f"|g|={sr_diag['grad_norm']:.2e} "
                f"|sr|={sr_diag['update_norm']:.2e} "
                f"ratio={sr_diag['ratio']:.1f}"
            )
            sys.stdout.flush()

    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)

    total = time.time() - t0
    print(f"  Best var={best_var:.3e}, {total:.0f}s ({total/60:.1f}min)")
    return f_net, bf_net


if __name__ == "__main__":
    C_occ, params = setup_noninteracting(N_PARTICLES, OMEGA, device=DEVICE, dtype=DTYPE)
    results = {}
    BASE = os.path.join(CKPT_DIR, "6e_sir_topk_baseline.pt")

    # ── Exp 1: SR-preconditioned, moderate damping ──
    print(f"\n# SR with damping=1e-3 [30ep fine-tune]")
    f1, bf1 = make_nets()
    ckpt = torch.load(BASE, map_location=DEVICE)
    f1.load_state_dict(ckpt["f_net"]); bf1.load_state_dict(ckpt["bf_net"])
    print(f"  Loaded <- {BASE}")

    f1, bf1 = train_sr(
        f1, bf1, C_occ, params,
        n_epochs=30, lr=1e-4, damping=1e-3, n_fisher=128,
        label="SR damping=1e-3")
    r1 = evaluate(f1, C_occ, params, backflow_net=bf1, label="sr_1e3")
    err1 = abs(r1["E_mean"] - E_DMC) / E_DMC * 100
    print(f"  >>> err = {err1:.2f}%")
    results["sr_1e-3"] = (r1["E_mean"], r1["E_stderr"], err1)

    # ── Exp 2: SR with stronger damping (more conservative) ──
    print(f"\n# SR with damping=1e-1 [30ep fine-tune]")
    f2, bf2 = make_nets()
    f2.load_state_dict(ckpt["f_net"]); bf2.load_state_dict(ckpt["bf_net"])
    print(f"  Loaded <- {BASE}")

    f2, bf2 = train_sr(
        f2, bf2, C_occ, params,
        n_epochs=30, lr=1e-4, damping=1e-1, n_fisher=128,
        label="SR damping=1e-1 (heavy)")
    r2 = evaluate(f2, C_occ, params, backflow_net=bf2, label="sr_1e1")
    err2 = abs(r2["E_mean"] - E_DMC) / E_DMC * 100
    print(f"  >>> err = {err2:.2f}%")
    results["sr_1e-1"] = (r2["E_mean"], r2["E_stderr"], err2)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY -- SR preconditioning")
    print(f"{'='*70}")
    results["topk_baseline (start)"] = (11.826500, 0.003000, 0.35)
    results["bf_0.7 joint 300ep (ref)"] = (11.823253, 0.002982, 0.33)
    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        print(f"  {name:35s}  E={E:.6f} +/- {std:.6f}  err={err:.2f}%")
    print("Done.")
