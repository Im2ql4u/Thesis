"""
Regularization experiments: attack the SOURCE of gradient pathology.

The core insight: near-node E_L divergence isn't a sampling problem — it's
a gradient pathology. The Laplacian through ∇²Δx / det(S̃) blows up at
nodes, and no sampling or loss trick fixes that. Instead, we soften the
singularity sources directly.

Three interventions (training only — VMC eval always uses exact physics):

1. SOFT COULOMB: Replace 1/r with 1/√(r² + ε²) during training.
   ε ~ 0.1-0.3 in physical units. The Jastrow already handles the cusp
   analytically, so the NN doesn't need to see the raw singularity.
   The gradient through V_int becomes smooth.

2. REGULARIZED SLOGDET: Floor log|det S̃| so that logψ doesn't → -∞
   near nodes. This prevents the gradient catastrophe in ∇logψ and ∇²logψ.
   Equivalent to: logψ_reg = log(|ψ|² + δ²)^{1/2} = 0.5*log(|ψ|² + δ²)
   which has bounded derivatives everywhere.

3. KINETIC ENERGY CLIPPING: Hard-clip the kinetic energy T = -½(∇²logψ + |∇logψ|²)
   so gradients through extreme T values don't dominate.

4. COMBINED: All three together.

Each experiment: 150 epochs from scratch, bf_scale=0.7, smoothness penalty,
Huber loss, standard top-K screening.

The key test: do regularized gradients let BF contribute constructively
without the defensive shrinkage we see from singularity-driven noise?
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import _laplacian_logpsi_exact, psi_fn
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import slater_determinant_closed_shell
from PINN import PINN, CTNNBackflowNet

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"

N_PARTICLES, DIM, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)


# ══════════════════════════════════════════════════════════════════
#  Standard model setup (same as previous scripts)
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
    params = config.get().as_dict()
    params["device"] = device
    params["torch_dtype"] = dtype
    params["E"] = E_DMC
    occ_str = ", ".join(f"({energies[k][1]},{energies[k][2]})" for k in range(n_occ))
    print(f"N={N}, w={omega}, basis={nx}x{ny}={n_basis}")
    print(f"  Occupied: {occ_str}   E_DMC={E_DMC}")
    return C_occ, params


def make_nets(bf_scale_init=0.7, zero_init_last=False):
    f_net = (
        PINN(
            n_particles=N_PARTICLES,
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
    bf_net = (
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
            bf_scale_init=bf_scale_init,
            zero_init_last=zero_init_last,
            omega=OMEGA,
        )
        .to(DEVICE)
        .to(DTYPE)
    )
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    up = N_PARTICLES // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_PARTICLES - up, dtype=torch.long)]
    ).to(DEVICE)

    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return lp

    return fn, spin


def save_model(f_net, bf_net, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"6e_reg_{name}.pt")
    torch.save({"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()}, path)
    print(f"  Saved -> {path}")


def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    print(f"\n-- VMC eval: {label} --")
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
    err = abs(E - E_DMC) / E_DMC * 100
    print(f"  E = {E:.6f} +/- {E_std:.6f}  (target {E_DMC}, err {err:.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════════
#  Smoothness penalty (proven critical)
# ══════════════════════════════════════════════════════════════════


def compute_bf_smoothness_penalty(bf_net, x, spin, n_samples=32):
    x_sub = x[:n_samples].detach().requires_grad_(True)
    B, N, d = x_sub.shape
    dx = bf_net(x_sub, spin)
    n_probes = 2
    lap_sq_sum = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    for _ in range(n_probes):
        v = torch.empty_like(x_sub).bernoulli_(0.5).mul_(2).add_(-1)
        for k in range(d):
            dx_k_sum = dx[:, :, k].sum()
            grad1 = torch.autograd.grad(dx_k_sum, x_sub, create_graph=True, retain_graph=True)[0]
            Hv = torch.autograd.grad(
                (grad1 * v).sum(), x_sub, create_graph=True, retain_graph=True
            )[0]
            vTHv = (v * Hv).sum(dim=(1, 2))
            lap_sq_sum = lap_sq_sum + (vTHv**2).mean()
    return lap_sq_sum / (n_probes * d)


# ══════════════════════════════════════════════════════════════════
#  INTERVENTION 1: Soft Coulomb (training only)
# ══════════════════════════════════════════════════════════════════


def compute_soft_coulomb(x, eps_phys=0.1):
    """
    Mollified Coulomb: 1/sqrt(r^2 + eps^2).
    eps_phys is in physical units (units of ell = 1/sqrt(omega)).

    The Jastrow already handles the exact e-e cusp analytically via
    gamma * r * exp(-r). So the NN never needs to see the raw 1/r.
    During training, we soften this to remove the gradient singularity.
    VMC evaluation always uses exact Coulomb.
    """
    B, N, d = x.shape
    ii, jj = torch.triu_indices(N, N, 1, device=x.device)
    diff = x[:, ii, :] - x[:, jj, :]  # (B, P, d)
    r2 = (diff**2).sum(-1)  # (B, P)
    eps2 = eps_phys**2
    r_soft = torch.sqrt(r2 + eps2)  # (B, P)
    Vij = 1.0 / r_soft  # (B, P)
    return Vij.sum(dim=1)  # (B,)


# ══════════════════════════════════════════════════════════════════
#  INTERVENTION 2: Regularized slogdet (training only)
# ══════════════════════════════════════════════════════════════════


def make_regularized_psi_log_fn(f_net, bf_net, C_occ, params, det_floor=1e-4):
    """
    Like standard psi_log_fn but with a soft floor on |det S̃|:

      logψ_reg = log(max(|det S̃|, δ)) + f

    Near nodes (|det| < δ), the gradient of logψ w.r.t. x becomes
    bounded instead of → ∞. This removes the ∇²logψ singularity.

    Equivalent to: we pretend ψ never goes below δ·exp(f).
    The wavefunction still changes sign (sign is tracked separately),
    but the log-magnitude is bounded from below.
    """
    up = N_PARTICLES // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_PARTICLES - up, dtype=torch.long)]
    ).to(DEVICE)

    def fn(y):
        B, N, d = y.shape
        spin_bn = spin.unsqueeze(0).expand(B, -1)

        # Backflow shift
        dx = bf_net(y, spin=spin_bn)
        x_eff = y + dx

        # Slater determinant
        sign, logabs = slater_determinant_closed_shell(
            x_config=x_eff,
            C_occ=C_occ,
            params=params,
            spin=spin_bn,
            normalize=True,
        )

        # KEY REGULARIZATION: floor |det| to prevent logabs → -∞
        # logabs = log|det_up| + log|det_down|
        # We floor this sum: logabs_reg = softplus-based smoothing
        log_floor = math.log(det_floor)
        # Smooth max: softmax(logabs, log_floor)
        # = log(exp(logabs) + exp(log_floor))
        # When logabs >> log_floor: → logabs (no change)
        # When logabs << log_floor: → log_floor (capped)
        logabs_reg = torch.logaddexp(
            logabs, torch.tensor(log_floor, dtype=y.dtype, device=y.device)
        )
        # Subtract log(2) bias from logaddexp when both are equal
        # Actually logaddexp(a, b) = log(exp(a) + exp(b)), which is always ≥ max(a,b)
        # This adds a tiny upward bias (~log2 when a≈b), which is fine for training.

        # Jastrow
        f_val = f_net(y, spin=spin_bn).squeeze(-1)  # (B,)

        logpsi = logabs_reg.view(-1) + f_val
        return logpsi

    return fn, spin


# ══════════════════════════════════════════════════════════════════
#  INTERVENTION 3: Kinetic energy clipping
# ══════════════════════════════════════════════════════════════════


def compute_local_energy_soft(psi_log_fn, x, omega, *, eps_coulomb=0.0, T_clip=None):
    """
    Local energy with optional soft Coulomb and kinetic energy clipping.

    T_clip: if set, clips |T| to this value. Prevents extreme kinetic
    energy contributions from propagating gradients.
    Typical E_L ~ 12 for N=6, so T_clip ~ 50 is loose, ~20 is moderate.
    """
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap_log.view(B) + g2.view(B))

    # Kinetic energy clipping
    if T_clip is not None and T_clip > 0:
        T = torch.clamp(T, -T_clip, T_clip)

    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))

    if eps_coulomb > 0:
        V_int = compute_soft_coulomb(x, eps_phys=eps_coulomb)
    else:
        V_int = compute_coulomb_interaction(x).view(B)

    return T + V_harm + V_int


# ══════════════════════════════════════════════════════════════════
#  Standard local energy (baseline)
# ══════════════════════════════════════════════════════════════════


def compute_local_energy_standard(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap_log.view(B) + g2.view(B))
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).view(B)
    return T + V_harm + V_int


# ══════════════════════════════════════════════════════════════════
#  Sampling: standard top-K
# ══════════════════════════════════════════════════════════════════


@torch.no_grad()
def sample_gaussian(n_samples, sigma):
    x = torch.randn(n_samples, N_PARTICLES, DIM, device=DEVICE, dtype=DTYPE) * sigma
    Nd = N_PARTICLES * DIM
    log_q = -0.5 * Nd * math.log(2 * math.pi * sigma**2) - x.reshape(n_samples, -1).pow(2).sum(
        -1
    ) / (2 * sigma**2)
    return x, log_q


@torch.no_grad()
def topk_collocation(psi_log_fn, n_keep, oversampling, sigma, batch_size=4096):
    M = oversampling * n_keep
    x_cand, log_q = sample_gaussian(M, sigma)
    log_psi_parts = []
    for i in range(0, M, batch_size):
        lp = psi_log_fn(x_cand[i : i + batch_size])
        log_psi_parts.append(lp)
    log_psi = torch.cat(log_psi_parts)
    log_w = 2.0 * log_psi - log_q
    valid = torch.isfinite(log_w)
    if not valid.all():
        log_w[~valid] = -1e10
    _, idx = torch.topk(log_w, n_keep)
    return x_cand[idx].clone()


# ══════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════


def train_regularized(
    f_net,
    bf_net,
    C_occ,
    params,
    *,
    n_epochs=150,
    lr=3e-4,
    lr_min_frac=0.02,
    n_collocation=2048,
    oversampling=10,
    sigma=None,
    # Regularization options
    eps_coulomb=0.0,  # soft Coulomb epsilon (0 = exact)
    det_floor=0.0,  # regularized det floor (0 = exact)
    T_clip=0.0,  # kinetic energy clip (0 = no clip)
    # Loss
    huber_delta=0.5,
    smooth_lambda=1e-3,
    smooth_n_samples=32,
    # E_ref schedule
    phase1_frac=0.25,
    alpha_end=0.60,
    # Robustness
    micro_batch=256,
    grad_clip=0.5,
    quantile_trim=0.02,
    patience=60,
    print_every=10,
    label="",
):
    omega = OMEGA
    if sigma is None:
        sigma = 1.3 * ELL

    # Build psi_log_fn — potentially regularized
    if det_floor > 0:
        psi_log_fn_train, spin = make_regularized_psi_log_fn(
            f_net, bf_net, C_occ, params, det_floor=det_floor
        )
    else:
        psi_log_fn_train, spin = make_psi_log_fn(f_net, bf_net, C_occ, params)

    # For screening, always use exact psi (so we screen by real |Ψ|²)
    psi_log_fn_screen, _ = make_psi_log_fn(f_net, bf_net, C_occ, params)

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

    n_f = sum(p.numel() for p in f_net.parameters())
    n_bf = sum(p.numel() for p in bf_net.parameters())
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {n_epochs}ep, f={n_f:,} bf={n_bf:,}, lr={lr}->{lr_min:.1e}")
    print(f"  eps_coulomb={eps_coulomb}, det_floor={det_floor}, T_clip={T_clip}")
    print(f"  smooth_lambda={smooth_lambda:.1e}, Huber delta={huber_delta}")
    print(f"{'='*70}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # E_ref schedule
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # Sample with exact psi
        f_net.eval()
        bf_net.eval()
        X = topk_collocation(psi_log_fn_screen, n_collocation, oversampling, sigma)
        n_pts = X.shape[0]

        # Train step
        f_net.train()
        bf_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        n_batches = max(1, math.ceil(n_pts / micro_batch))

        for i in range(0, n_pts, micro_batch):
            x_mb = X[i : i + micro_batch]

            # Compute E_L with regularizations
            E_L = compute_local_energy_soft(
                psi_log_fn_train,
                x_mb,
                omega,
                eps_coulomb=eps_coulomb,
                T_clip=T_clip if T_clip > 0 else None,
            ).view(-1)

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

            loss_mb = nn.functional.huber_loss(resid, torch.zeros_like(resid), delta=huber_delta)

            if smooth_lambda > 0:
                pen = compute_bf_smoothness_penalty(bf_net, x_mb, spin, n_samples=smooth_n_samples)
                loss_mb = loss_mb + smooth_lambda * pen

            (loss_mb / n_batches).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
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

        if epoch % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / E_DMC * 100

            with torch.no_grad():
                bf_net.eval()
                dx_s = bf_net(X[:64], spin)
                bf_mag = dx_s.norm(dim=-1).mean().item()
                bf_scale = nn.functional.softplus(bf_net.bf_scale_raw).item()
                bf_net.train()

            print(
                f"[{epoch:4d}] E={E_mean:.5f} +/- {E_std:.4f}  "
                f"var={E_var:.3e}  a={alpha:.2f}  lr={cur_lr:.1e}  "
                f"|dx|={bf_mag:.3f}  bf_s={bf_scale:.3f}  err={err:.2f}%"
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
#  Experiments
# ══════════════════════════════════════════════════════════════════


def load_trained(f_net, bf_net, path):
    ckpt = torch.load(path, map_location=DEVICE)
    f_net.load_state_dict(ckpt["f_net"])
    bf_net.load_state_dict(ckpt["bf_net"])
    print(f"  Loaded <- {path}")
    return f_net, bf_net


if __name__ == "__main__":
    C_occ, params = setup_noninteracting(N_PARTICLES, OMEGA, device=DEVICE, dtype=DTYPE)
    results = {}
    N_EP = 50  # quick fine-tune
    BASE_MODEL = os.path.join(CKPT_DIR, "6e_sir_topk_baseline.pt")  # 0.35%

    configs = [
        ("soft_coul", dict(eps_coulomb=0.2, det_floor=0.0, T_clip=0.0)),
        ("reg_det", dict(eps_coulomb=0.0, det_floor=1e-3, T_clip=0.0)),
        ("T_clip", dict(eps_coulomb=0.0, det_floor=0.0, T_clip=30.0)),
        ("all_reg", dict(eps_coulomb=0.2, det_floor=1e-3, T_clip=30.0)),
    ]

    for name, reg_kw in configs:
        print(f"\n{'='*70}")
        print(f"# {name}: {reg_kw}  [{N_EP}ep fine-tune]")
        print(f"{'='*70}")

        f, bf = make_nets()
        f, bf = load_trained(f, bf, BASE_MODEL)

        f, bf = train_regularized(
            f,
            bf,
            C_occ,
            params,
            n_epochs=N_EP,
            lr=1e-4,
            patience=999,  # don't early-stop on short runs
            **reg_kw,
            label=f"{name}: {reg_kw}",
        )
        save_model(f, bf, name)
        r = evaluate(f, C_occ, params, backflow_net=bf, label=name)
        E, std = r["E_mean"], r["E_stderr"]
        err = abs(E - E_DMC) / E_DMC * 100
        print(f"  >>> err = {err:.2f}%")
        results[name] = (E, std, err)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY -- Regularization fine-tune ({N_EP}ep from topk_baseline)")
    print(f"{'='*70}")

    results["topk_baseline 300ep (start)"] = (11.826500, 0.003000, 0.35)
    results["bf_0.7 joint 300ep (ref)"] = (11.823253, 0.002982, 0.33)

    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        print(f"  {name:40s}  E={E:.6f} +/- {std:.6f}  err={err:.2f}%")
    print("Done.")
