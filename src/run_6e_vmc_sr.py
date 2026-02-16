"""
VMC + SR: proper stochastic reconfiguration from trained PINN + untrained BF.
Goal: prove the ansatz CAN break 0.33% — the problem is collocation, not capacity.

SR update: δθ = S⁻¹ f  where
  S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩  (covariance of log-derivatives)
  f_i  = ⟨E_L O_i⟩ - ⟨E_L⟩⟨O_i⟩  (energy-log-derivative covariance)

Low-rank Woodbury for efficiency (24K params, B~256 samples).
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import _laplacian_logpsi_exact, psi_fn
from functions.Physics import compute_coulomb_interaction
from PINN import PINN, CTNNBackflowNet

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"
N_P, DIM, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)


def setup():
    n_occ = N_P // 2
    nx = ny = 3
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(
        omega=OMEGA,
        n_particles=N_P,
        d=DIM,
        L=L,
        n_grid=80,
        nx=nx,
        ny=ny,
        basis="cart",
        device="cpu",
        dtype="float64",
    )
    energies = []
    for ix in range(nx):
        for iy in range(ny):
            energies.append((OMEGA * (ix + iy + 1), ix, iy))
    energies.sort(key=lambda t: t[0])
    C = np.zeros((nx * ny, n_occ), dtype=np.float64)
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
    p = config.get().as_dict()
    p["device"] = DEVICE
    p["torch_dtype"] = DTYPE
    p["E"] = E_DMC
    return C_occ, p


def make_nets():
    f = (
        PINN(
            n_particles=N_P,
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
    b = (
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
            bf_scale_init=0.7,
            zero_init_last=False,
            omega=OMEGA,
        )
        .to(DEVICE)
        .to(DTYPE)
    )
    return f, b


def make_psi(f, b, C, p):
    up = N_P // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_P - up, dtype=torch.long)]
    ).to(DEVICE)

    def fn(y):
        lp, _ = psi_fn(f, y, C, backflow_net=b, spin=spin, params=p)
        return lp

    return fn, spin


def local_energy(psi_log_fn, x):
    x = x.detach().requires_grad_(True)
    _, g2, lap = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap.view(B) + g2.view(B))
    V = 0.5 * OMEGA**2 * (x**2).sum(dim=(1, 2)) + compute_coulomb_interaction(x).view(B)
    return T + V


# ══════════════════════════════════════════════════════════════
#  MCMC sampler (Metropolis-Hastings from |ψ|²)
# ══════════════════════════════════════════════════════════════


@torch.no_grad()
def _evolve_walkers(psi_log_fn, x, n_steps, step_sigma, batch_eval=512):
    """Evolve persistent walkers for n_steps MH steps."""
    n = x.shape[0]
    log_p = torch.zeros(n, device=DEVICE, dtype=DTYPE)
    for i in range(0, n, batch_eval):
        log_p[i : i + batch_eval] = 2.0 * psi_log_fn(x[i : i + batch_eval])

    n_acc = 0
    for _ in range(n_steps):
        x_prop = x + torch.randn_like(x) * step_sigma
        log_p_prop = torch.zeros(n, device=DEVICE, dtype=DTYPE)
        for i in range(0, n, batch_eval):
            log_p_prop[i : i + batch_eval] = 2.0 * psi_log_fn(x_prop[i : i + batch_eval])
        log_alpha = log_p_prop - log_p
        accept = torch.log(torch.rand(n, device=DEVICE, dtype=DTYPE)) < log_alpha
        accept = accept & torch.isfinite(log_p_prop)
        x[accept] = x_prop[accept]
        log_p[accept] = log_p_prop[accept]
        n_acc += accept.float().mean().item()

    return x, n_acc / n_steps


# ══════════════════════════════════════════════════════════════
#  SR update via low-rank Woodbury
# ══════════════════════════════════════════════════════════════


def sr_step(all_params, psi_log_fn, x, E_L, damping=1e-3):
    """
    Compute SR update: δθ = S⁻¹ f

    S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩
    f_i  = ⟨E_L O_i⟩ - ⟨E_L⟩⟨O_i⟩

    Using Woodbury for efficiency.
    """
    B = x.shape[0]

    # Compute per-sample O_i = ∂logψ/∂θ_i
    J_rows = []
    for b in range(B):
        for p in all_params:
            if p.grad is not None:
                p.grad = None
        lp = psi_log_fn(x[b : b + 1])
        lp.backward()
        row = torch.cat(
            [
                (
                    p.grad.flatten()
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=DEVICE, dtype=DTYPE)
                )
                for p in all_params
            ]
        )
        J_rows.append(row)

    # Zero grads after
    for p in all_params:
        if p.grad is not None:
            p.grad = None

    O = torch.stack(J_rows)  # (B, P)
    P = O.shape[1]

    # Center: Ō = O - ⟨O⟩
    O_mean = O.mean(dim=0)  # (P,)
    O_centered = O - O_mean.unsqueeze(0)  # (B, P)

    # f_i = ⟨E_L O_i⟩ - ⟨E_L⟩⟨O_i⟩ = ⟨(E_L - ⟨E_L⟩)(O_i - ⟨O_i⟩)⟩
    E_mean = E_L.mean()
    E_centered = (E_L - E_mean).unsqueeze(1)  # (B, 1)
    f = (E_centered * O_centered).mean(dim=0)  # (P,)

    # S = O_centered^T O_centered / B  (rank ≤ B)
    # Solve (S + εI) δ = f via Woodbury:
    # (Oc^T Oc / B + εI)^{-1} f = (1/ε)[f - Oc^T (B·ε·I + Oc Oc^T)^{-1} Oc f]

    Oc_f = O_centered @ f  # (B,)
    K = O_centered @ O_centered.T  # (B, B)
    K_reg = K + B * damping * torch.eye(B, device=DEVICE, dtype=DTYPE)
    solve = torch.linalg.solve(K_reg, Oc_f)  # (B,)
    delta = (f - O_centered.T @ solve) / damping  # (P,)

    # Diagnostics
    eigs = torch.linalg.eigvalsh(K / B)

    return delta, {
        "E_mean": E_mean.item(),
        "E_var": E_L.var().item(),
        "grad_norm": f.norm().item(),
        "update_norm": delta.norm().item(),
        "S_cond": (eigs[-1] / max(eigs[0].item(), 1e-30)).item(),
        "S_top": eigs[-1].item(),
    }


def evaluate(f, C, p, bf, label=""):
    print(f"\n-- VMC eval: {label} --")
    r = evaluate_energy_vmc(
        f,
        C,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=bf,
        params=p,
        n_samples=15000,
        batch_size=512,
        sampler_steps=50,
        sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True,
        sampler_burn_in=300,
        sampler_thin=3,
        progress=True,
    )
    E, std = r["E_mean"], r["E_stderr"]
    err = abs(E - E_DMC) / E_DMC * 100
    print(f"  E = {E:.6f} +/- {std:.6f}  err={err:.2f}%")
    return r


# ══════════════════════════════════════════════════════════════
#  Main: VMC + SR training loop
# ══════════════════════════════════════════════════════════════


def train_vmc_sr(
    f,
    bf,
    C,
    p,
    *,
    n_epochs=50,
    lr=0.01,
    n_walkers=256,
    damping=1e-3,
    step_sigma=0.15,
    clip_delta=1.0,
    E_clip_width=5.0,
    label="",
):
    psi, spin = make_psi(f, bf, C, p)
    all_params = [pp for pp in list(f.parameters()) + list(bf.parameters()) if pp.requires_grad]
    n_f = sum(pp.numel() for pp in f.parameters() if pp.requires_grad)
    n_bf = sum(pp.numel() for pp in bf.parameters() if pp.requires_grad)
    P = n_f + n_bf

    print(f"\n{'='*65}")
    print(f"  VMC + SR: {label}")
    print(f"  {n_epochs}ep, {n_walkers} walkers, P={P:,} (f={n_f:,} bf={n_bf:,})")
    print(f"  lr={lr}, damping={damping}, E_clip={E_clip_width}σ")
    print(f"{'='*65}")
    sys.stdout.flush()

    t0 = time.time()
    best_E = 1e10
    best_fs, best_bs = {}, {}

    # Persistent walkers — initialize once, re-use across epochs
    walkers = torch.randn(n_walkers, N_P, DIM, device=DEVICE, dtype=DTYPE) * ELL

    for ep in range(n_epochs):
        f.eval()
        bf.eval()

        # ── Evolve persistent walkers ──
        n_steps = 50 if ep == 0 else 10
        walkers, acc = _evolve_walkers(psi, walkers, n_steps, step_sigma)

        # ── Compute E_L ──
        x = walkers.clone()
        f.train()
        bf.train()
        E_L = local_energy(psi, x).detach()

        # Clip outlier E_L (standard VMC practice)
        E_med = E_L.median()
        E_std = E_L.std()
        mask = (E_L - E_med).abs() < E_clip_width * E_std
        if mask.sum() < 0.5 * n_walkers:
            mask = torch.ones_like(mask, dtype=torch.bool)
        x_good = x[mask]
        E_good = E_L[mask]

        # ── SR step ──
        delta, diag = sr_step(all_params, psi, x_good, E_good, damping=damping)

        # Clip update norm
        d_norm = delta.norm()
        if d_norm > clip_delta:
            delta = delta * (clip_delta / d_norm)

        # Apply
        offset = 0
        with torch.no_grad():
            for pp in all_params:
                n = pp.numel()
                pp.add_(delta[offset : offset + n].view_as(pp), alpha=-lr)
                offset += n

        E_now = diag["E_mean"]
        if E_now < best_E:
            best_E = E_now
            best_fs = {k: v.clone() for k, v in f.state_dict().items()}
            best_bs = {k: v.clone() for k, v in bf.state_dict().items()}

        if ep % 2 == 0:
            err = abs(E_now - E_DMC) / E_DMC * 100
            with torch.no_grad():
                bf.eval()
                bm = bf(x_good[:64], spin).norm(dim=-1).mean().item()
                bf.train()
            print(
                f"[{ep:3d}] E={E_now:.5f} var={diag['E_var']:.3e} "
                f"|dx|={bm:.3f} acc={acc:.2f} "
                f"err={err:.2f}% "
                f"|f|={diag['grad_norm']:.2e} |δ|={diag['update_norm']:.2e}"
            )
            sys.stdout.flush()

    if best_fs:
        f.load_state_dict(best_fs)
    if best_bs:
        bf.load_state_dict(best_bs)

    dt = time.time() - t0
    err_best = abs(best_E - E_DMC) / E_DMC * 100
    print(f"  Done {dt:.0f}s ({dt/60:.1f}min), best E={best_E:.5f} ({err_best:.2f}%)")
    return f, bf


if __name__ == "__main__":
    C, p = setup()
    PINN_CKPT = os.path.join(CKPT_DIR, "pinn_6e_ckpt.pt")

    # Load trained PINN + fresh CTNN (small bf_scale so starting point ≈ PINN-only)
    f_net, bf_net = make_nets()
    # Override bf_scale to 0.05 so fresh CTNN barely perturbs ψ
    with torch.no_grad():
        bf_net.bf_scale.fill_(0.05)
    ckpt = torch.load(PINN_CKPT, map_location=DEVICE)
    if "f_net" in ckpt:
        f_net.load_state_dict(ckpt["f_net"])
    else:
        f_net.load_state_dict(ckpt)
    print(f"Loaded PINN from {PINN_CKPT}")
    print("CTNN is FRESH, bf_scale=0.05 (will grow via SR)")

    # FREEZE PINN — only train CTNN
    for pp in f_net.parameters():
        pp.requires_grad_(False)
    n_bf = sum(pp.numel() for pp in bf_net.parameters())
    print(f"PINN FROZEN. Training only CTNN ({n_bf:,} params) with SR.")

    # Eval starting point
    r0 = evaluate(f_net, C, p, bf_net, label="start (pinn+fresh_ctnn)")
    err0 = abs(r0["E_mean"] - E_DMC) / E_DMC * 100
    print(f"  Starting err = {err0:.2f}%")

    # VMC + SR training — CTNN only
    f_net, bf_net = train_vmc_sr(
        f_net,
        bf_net,
        C,
        p,
        n_epochs=150,
        lr=0.003,
        n_walkers=512,
        damping=1e-3,
        step_sigma=0.15,
        clip_delta=0.3,
        E_clip_width=5.0,
        label="VMC+SR CTNN-only (PINN frozen)",
    )

    # Save
    torch.save(
        {"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()},
        os.path.join(CKPT_DIR, "6e_vmc_sr.pt"),
    )

    # Final eval
    r1 = evaluate(f_net, C, p, bf_net, label="after VMC+SR")
    err1 = abs(r1["E_mean"] - E_DMC) / E_DMC * 100

    print(f"\n{'='*65}")
    print("SUMMARY — VMC+SR vs collocation")
    print(f"{'='*65}")
    print(f"  DMC target                       E={E_DMC:.6f}            err=0.00%")
    print(
        f"  start (pinn+fresh_ctnn)          E={r0['E_mean']:.6f} +/- {r0['E_stderr']:.6f}  err={err0:.2f}%"
    )
    print(
        f"  VMC+SR 150ep (CTNN only)         E={r1['E_mean']:.6f} +/- {r1['E_stderr']:.6f}  err={err1:.2f}%"
    )
    print("  best collocation (bf_0.7 joint)  E=11.823300 +/- 0.002980  err=0.33%")
    print("  PINN-only collocation            E=~11.834   +/- ~0.003    err=0.42%")
    print("Done.")
