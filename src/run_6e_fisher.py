"""
6e residual training with diagonal Fisher preconditioner for backflow
and close-pair augmented collocation.

Two key additions over run_6e_residual.py:

1. DIAGONAL FISHER PRECONDITIONER (ideas 1 & 5)
   After loss.backward(), rescale backflow gradients by  g_i / (F_ii + λ)
   where  F_ii = (1/M) Σ_k (∂ ln|Ψ(x_k)| / ∂θ_i)²   approximated on
   a subset of screened points.  This is a lightweight natural-gradient
   preconditioner that captures the parameter-space geometry SR exploits,
   without requiring MCMC.

2. CLOSE-PAIR AUGMENTED PROPOSAL (idea 4)
   A fraction of proposal candidates force two random electrons close
   together, ensuring the screened collocation probes the pair-coalescence
   region where backflow matters most.

Both additions apply ONLY to the CTNN backflow component.
The Jastrow (PINN) trains with plain Adam as before (where it already
reaches 0.42%).
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import (
    _laplacian_logpsi_exact,
    psi_fn,
)
from functions.Physics import compute_coulomb_interaction
from PINN import PINN, CTNNBackflowNet

# ══════════════════════════════════════════════════════════════════
#  Helpers  (same as run_6e_residual.py)
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
    E_ni = sum(energies[k][0] for k in range(n_occ)) * 2
    E_DMC = config.DMC_ENERGIES.get(N, {}).get(config._snap_omega(omega), None)
    params = config.get().as_dict()
    params["device"] = device
    params["torch_dtype"] = dtype
    params["E"] = E_DMC
    occ_str = ", ".join(f"({energies[k][1]},{energies[k][2]})" for k in range(n_occ))
    print(f"N={N}, ω={omega}, basis={nx}×{ny}={n_basis}")
    print(f"  Occupied: {occ_str}   E_ni={E_ni:.2f}   E_DMC={E_DMC}")
    return C_occ, params


def compute_local_energy(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap_log.view(B) + g2.view(B))
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).view(B)
    return T + V_harm + V_int


# ══════════════════════════════════════════════════════════════════
#  Screened collocation with close-pair augmentation
# ══════════════════════════════════════════════════════════════════


@torch.no_grad()
def screened_collocation(
    psi_log_fn,
    N,
    d,
    sigma,
    n_keep,
    oversampling,
    device,
    dtype,
    *,
    close_pair_frac=0.0,
    close_pair_sigma=0.4,
    batch_size=4096,
):
    """
    Generate candidates from Gaussian + close-pair proposals,
    rank by |Ψ|², keep top n_keep.
    """
    n_cand = oversampling * n_keep
    n_close = int(close_pair_frac * n_cand)
    n_normal = n_cand - n_close

    # Normal Gaussian candidates
    x_all = torch.randn(n_cand, N, d, device=device, dtype=dtype) * sigma

    # Overwrite last n_close candidates with close-pair configurations
    if n_close > 0:
        idx_start = n_normal
        # Pick random pairs per close-pair candidate
        i_idx = torch.randint(0, N, (n_close,))
        j_offset = torch.randint(1, N, (n_close,))
        j_idx = (i_idx + j_offset) % N
        batch_range = torch.arange(n_close)
        offsets = torch.randn(n_close, d, device=device, dtype=dtype) * close_pair_sigma
        x_all[idx_start + batch_range, j_idx] = x_all[idx_start + batch_range, i_idx] + offsets

    # Rank by |Ψ|²
    log_psi2_parts = []
    for i in range(0, n_cand, batch_size):
        lp = psi_log_fn(x_all[i : i + batch_size])
        log_psi2_parts.append(2.0 * lp)
    log_psi2 = torch.cat(log_psi2_parts)

    _, idx = torch.topk(log_psi2, n_keep)
    return x_all[idx].clone()


# ══════════════════════════════════════════════════════════════════
#  Diagonal Fisher estimator for backflow params
# ══════════════════════════════════════════════════════════════════


def estimate_fisher_diag(psi_log_fn, X, bf_params, n_samples=64, damping=1e-3):
    """
    Estimate diagonal Fisher  F_ii = E[(∂ ln|Ψ| / ∂θ_i)²]
    on a random subset of the screened collocation points.

    Returns a list of tensors parallel to bf_params, each containing
    F_ii + damping  (ready to divide gradients by).
    """
    M = min(n_samples, X.shape[0])
    idx = torch.randperm(X.shape[0])[:M]
    X_sub = X[idx]

    fisher = [torch.zeros_like(p) for p in bf_params]

    for k in range(M):
        x_k = X_sub[k : k + 1]
        log_psi = psi_log_fn(x_k)
        if log_psi.dim() > 0:
            log_psi = log_psi.sum()
        grads = torch.autograd.grad(
            log_psi,
            bf_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for i, g in enumerate(grads):
            if g is not None:
                fisher[i] += g.detach() ** 2

    for i in range(len(fisher)):
        fisher[i] /= M
        fisher[i] += damping

    return fisher


# ══════════════════════════════════════════════════════════════════
#  Training loop with Fisher preconditioning
# ══════════════════════════════════════════════════════════════════


def train_residual_fisher(
    f_net,
    C_occ,
    params,
    *,
    backflow_net,  # required
    # --- schedule ---
    n_epochs=300,
    lr=3e-4,
    lr_min_frac=0.02,
    # --- alpha targeting ---
    phase1_frac=0.25,
    alpha_end=0.60,
    # --- collocation ---
    n_collocation=2048,
    oversampling=10,
    proposal_sigma_factor=1.3,
    micro_batch=256,
    # --- close-pair ---
    close_pair_frac=0.0,
    close_pair_sigma_ell=0.3,
    # --- Fisher preconditioner ---
    fisher_n=64,
    fisher_damping=1e-3,
    fisher_update_every=5,
    # --- robustness ---
    grad_clip=0.5,
    quantile_trim=0.02,
    print_every=10,
):
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    E_DMC = params.get("E", None)
    ell = 1.0 / math.sqrt(omega)
    sigma = proposal_sigma_factor * ell
    cp_sigma = close_pair_sigma_ell * ell

    f_net.to(device).to(dtype)
    backflow_net.to(device).to(dtype)

    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]).to(
        device
    )

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return lp

    f_params = list(f_net.parameters())
    bf_params = list(backflow_net.parameters())
    all_params = f_params + bf_params
    n_p = sum(p.numel() for p in all_params)
    n_bf = sum(p.numel() for p in bf_params)

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

    print(f"\n{'='*60}")
    print("Fisher-preconditioned residual training")
    print(f"  {n_epochs} ep, {n_collocation} coll pts (×{oversampling})")
    print(f"  {n_p:,} total params ({n_bf:,} backflow)")
    print(f"  lr={lr}, cosine → {lr_min:.1e}")
    print(f"  proposal σ = {sigma:.3f}  (ℓ = {ell:.3f})")
    if close_pair_frac > 0:
        print(f"  close-pair: {close_pair_frac*100:.0f}% of candidates, " f"σ_cp = {cp_sigma:.3f}")
    print(
        f"  Fisher: {fisher_n} samples, damping={fisher_damping}, "
        f"update every {fisher_update_every} ep"
    )
    print(f"  phase 1 (var-min): epochs 0–{phase1_end}")
    print(f"  phase 2 (targeting): α 0→{alpha_end}, eps {phase1_end}–{n_epochs}")
    print(f"  E_DMC = {E_DMC}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    history = []
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    patience = 60
    fisher = None  # initialized on first update

    for epoch in range(n_epochs):
        # ── Alpha schedule ──
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # ── Screened collocation ──
        f_net.eval()
        backflow_net.eval()
        X = screened_collocation(
            psi_log_fn,
            N,
            d,
            sigma,
            n_keep=n_collocation,
            oversampling=oversampling,
            device=device,
            dtype=dtype,
            close_pair_frac=close_pair_frac,
            close_pair_sigma=cp_sigma,
        )

        # ── Fisher update (every K epochs) ──
        if epoch % fisher_update_every == 0:
            f_net.train()
            backflow_net.train()
            fisher = estimate_fisher_diag(
                psi_log_fn,
                X,
                bf_params,
                n_samples=fisher_n,
                damping=fisher_damping,
            )

        # ── Compute loss in micro-batches ──
        f_net.train()
        backflow_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        n_batches = max(1, math.ceil(n_collocation / micro_batch))

        for i in range(0, n_collocation, micro_batch):
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
            E_eff = alpha * float(E_DMC) + (1.0 - alpha) * mu
            resid = E_L - E_eff
            loss_mb = (resid**2).mean()
            (loss_mb / n_batches).backward()

        # ── Fisher preconditioning on backflow grads ──
        if fisher is not None:
            for p, f_diag in zip(bf_params, fisher, strict=False):
                if p.grad is not None:
                    p.grad.div_(f_diag)

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

        history.append({"epoch": epoch, "E_mean": E_mean, "E_var": E_var})

        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_f_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            best_bf_state = {k: v.clone() for k, v in backflow_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience and epoch > phase1_end + 30:
            print(f"  Early stop at epoch {epoch}  (best var={best_var:.3e})")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / abs(E_DMC) * 100 if E_DMC else 0
            print(
                f"[{epoch:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  "
                f"lr={cur_lr:.1e}  t={dt:.0f}s  err={err:.2f}%"
            )
            sys.stdout.flush()

    if best_f_state:
        f_net.load_state_dict(best_f_state)
        backflow_net.load_state_dict(best_bf_state)
        print(f"Restored best model (var={best_var:.3e})")

    total = time.time() - t0
    print(f"Training done in {total:.0f}s ({total/60:.1f}min)")
    return f_net, backflow_net, history


# ══════════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════════


def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    print(f"\n── VMC eval: {label} ──")
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
    E_ref = params.get("E")
    err = abs(E - E_ref) / abs(E_ref) * 100 if E_ref else 0
    print(f"  E = {E:.6f} ± {E_std:.6f}  (target {E_ref}, err {err:.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════════
#  Run configs
# ══════════════════════════════════════════════════════════════════


def make_nets(device="cpu", dtype=torch.float64):
    f_net = (
        PINN(
            n_particles=6,
            d=2,
            omega=0.5,
            dL=8,
            hidden_dim=64,
            n_layers=2,
            act="gelu",
            init="xavier",
            use_gate=True,
            use_pair_attn=False,
        )
        .to(device)
        .to(dtype)
    )
    bf_net = (
        CTNNBackflowNet(
            d=2,
            msg_hidden=32,
            msg_layers=1,
            hidden=32,
            layers=2,
            act="silu",
            aggregation="sum",
            use_spin=True,
            same_spin_only=False,
            out_bound="tanh",
            bf_scale_init=0.05,
            omega=0.5,
        )
        .to(device)
        .to(dtype)
    )
    np_total = sum(p.numel() for p in f_net.parameters()) + sum(
        p.numel() for p in bf_net.parameters()
    )
    print(f"CTNN+PINN params: {np_total:,}")
    return f_net, bf_net


COMMON = dict(
    n_epochs=300,
    lr=3e-4,
    n_collocation=2048,
    oversampling=10,
    micro_batch=256,
    grad_clip=0.5,
    print_every=10,
    phase1_frac=0.25,
    alpha_end=0.60,
    proposal_sigma_factor=1.3,
    fisher_n=64,
    fisher_damping=1e-3,
    fisher_update_every=5,
)


def run_fisher_closepair():
    """CTNN+PINN with Fisher preconditioning + close-pair proposal."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net, bf_net = make_nets(device, dtype)
    f_net, bf_net, _ = train_residual_fisher(
        f_net,
        C_occ,
        params,
        backflow_net=bf_net,
        close_pair_frac=0.20,
        close_pair_sigma_ell=0.3,
        **COMMON,
    )
    return evaluate(
        f_net, C_occ, params, backflow_net=bf_net, label="CTNN+PINN  Fisher + close-pair"
    )


def run_fisher_only():
    """CTNN+PINN with Fisher preconditioning, normal proposal."""
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net, bf_net = make_nets(device, dtype)
    f_net, bf_net, _ = train_residual_fisher(
        f_net,
        C_occ,
        params,
        backflow_net=bf_net,
        close_pair_frac=0.0,
        **COMMON,
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net, label="CTNN+PINN  Fisher only")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = {}

    for name, fn in [
        ("fisher+cp", run_fisher_closepair),
        ("fisher", run_fisher_only),
    ]:
        print(f"\n{'#'*60}")
        print(f"# {name.upper()}")
        print(f"{'#'*60}")
        results[name] = fn()

    target = 11.78484
    print(f"\n{'='*60}")
    print("SUMMARY — 6e ω=0.5  CTNN+PINN (Fisher preconditioned)")
    print(f"{'='*60}")
    for name, r in results.items():
        E, se = r["E_mean"], r["E_stderr"]
        err = abs(E - target) / target * 100
        print(f"  {name:20s}  E={E:.6f} ± {se:.6f}  err={err:.2f}%")
    print(f"  {'baseline (no Fisher)':20s}  E=11.848922 ± 0.002811  err=0.54%")
    print(f"  {'PINN only':20s}  E=11.834691 ± 0.002782  err=0.42%")
    print(f"  {'DMC target':20s}  E={target:.6f}")
    print(f"{'='*60}")
