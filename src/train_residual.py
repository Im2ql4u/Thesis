"""
Residual-based training with |Ψ|²-sampling (variance minimization VMC).

Key ideas:
  1. Pure PDE residual: minimise Var[E_L] under |Ψ|² → 0 means exact eigenstate
  2. Exact (analytic) Laplacians — no Hutchinson noise
  3. ALL collocation points from persistent MCMC on |Ψ|² — no fixed distribution
  4. Gaussian envelope on Jastrow: architectural guarantee f → 0 at large r
  5. Optimizer: Adam (not SR — simpler, fewer hyperparameters)

This is variance-minimization VMC with Adam, not SR.
"""

import math
import sys
import time

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
from functions.Slater_Determinant import compute_integrals, hartree_fock_closed_shell
from PINN import UnifiedCTNN

# ─────────────────────────────────────────────────────────────────
# System setup
# ─────────────────────────────────────────────────────────────────


def setup_system(N, omega, d=2, device="cpu", dtype=torch.float64):
    L = max(8.0, 3.0 / math.sqrt(omega))
    nx = {2: 2, 6: 4, 12: 5, 20: 6}.get(N, 4)
    ny = nx

    # ── CRITICAL: set the global config so @inject_params picks up
    #    the correct omega, n_particles, etc.  Without this, all basis
    #    functions use the default omega=0.1 from Config().
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
        hf_verbose=False,
        hf_damping=0.5,
    )

    params = config.get().as_dict()
    params["device"] = device
    params["torch_dtype"] = dtype
    params["basis_n_max"] = 5

    Hcore, two_dirac, _ = compute_integrals(params=params)
    C_occ_np, _, E_hf = hartree_fock_closed_shell(Hcore, two_dirac, params=params)
    C_occ = torch.tensor(C_occ_np, dtype=dtype, device=device)

    E_DMC = config.DMC_ENERGIES.get(N, {}).get(config._snap_omega(omega), None)
    params["E"] = E_DMC
    params["E_HF"] = float(E_hf)

    print(f"System: N={N}, ω={omega}  |  E_HF={E_hf:.4f}  E_DMC={E_DMC}")
    sys.stdout.flush()
    return C_occ, params


# ─────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────


def sample_stratified(B, N, d, omega, device, dtype):
    """Gaussian mixture sampler covering the physical domain.

    4 components at different length scales to cover:
      - core region (cusp region)
      - oscillator length region (SD peak)
      - outer region (correlation tails)
      - far region (bound-state boundary)
    """
    ell = 1.0 / math.sqrt(omega)
    x = torch.empty(B, N, d, device=device, dtype=dtype)

    sizes = [B // 4, B // 4, B // 4, B - 3 * (B // 4)]
    sigmas = [0.25 * ell, 0.55 * ell, 0.90 * ell, 1.5 * ell]

    idx = 0
    for n, sig in zip(sizes, sigmas, strict=False):
        x[idx : idx + n] = torch.randn(n, N, d, device=device, dtype=dtype) * sig
        idx += n

    # Random permutation of particles (antisymmetry exploration)
    perm = torch.stack([torch.randperm(N, device=device) for _ in range(B)])
    x = x.gather(1, perm.unsqueeze(-1).expand(B, N, d))

    # Random rotation (2D equivariance)
    if d == 2:
        theta = 2 * math.pi * torch.rand(B, device=device, dtype=dtype)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
        x = torch.einsum("bnc,bcC->bnC", x, R)

    return x
    return x


@torch.no_grad()
def sample_mcmc(
    psi_log_fn,
    N,
    d,
    omega,
    n_samples,
    device,
    dtype,
    burn_in=300,
    thin=5,
    sigma_frac=0.12,
    x_init=None,
):
    """Metropolis MCMC sampling from |Ψ|².  Returns (x, acc_rate)."""
    ell = 1.0 / math.sqrt(omega)
    sig = sigma_frac * ell
    B_chain = min(64, n_samples)  # chain batch size

    if x_init is not None:
        x = x_init[:B_chain].clone()
    else:
        x = torch.randn(B_chain, N, d, device=device, dtype=dtype) * ell

    lp = psi_log_fn(x) * 2.0  # log|Ψ|²

    acc, tot = 0, 0
    samples = []

    total_steps = burn_in + n_samples * thin // B_chain + thin
    for step in range(total_steps):
        prop = x + torch.randn_like(x) * sig
        lp_prop = psi_log_fn(prop) * 2.0
        log_u = torch.log(torch.rand(B_chain, device=device, dtype=dtype))
        accept = log_u < (lp_prop - lp)
        x = torch.where(accept.view(-1, 1, 1), prop, x)
        lp = torch.where(accept, lp_prop, lp)
        acc += accept.sum().item()
        tot += B_chain

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(x.clone())
            if sum(s.shape[0] for s in samples) >= n_samples:
                break

    x_all = torch.cat(samples, dim=0)[:n_samples]
    return x_all, acc / max(tot, 1)


# ─────────────────────────────────────────────────────────────────
# Local energy computation
# ─────────────────────────────────────────────────────────────────


def compute_local_energy(psi_log_fn, x, omega):
    """E_L(x) = T(x) + V(x) using exact analytic Laplacian."""
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)

    T = -0.5 * (lap_log.squeeze(-1) + g2.squeeze(-1))
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).squeeze(-1)
    E_L = T + V_harm + V_int
    return E_L


# ─────────────────────────────────────────────────────────────────
# Training: stratified variance-minimization (i.i.d. samples)
# ─────────────────────────────────────────────────────────────────


def train_stratified(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    params: dict,
    *,
    n_epochs: int = 600,
    lr: float = 5e-4,
    n_collocation: int = 1024,  # samples per epoch
    micro_batch: int = 64,  # for gradient accumulation
    grad_clip: float = 0.5,
    quantile_trim: float = 0.05,
    print_every: int = 25,
    huber_delta: float = 1.0,
    lr_schedule: str = "cosine",
    lr_min_frac: float = 0.01,
    warmup_epochs: int = 20,
    eval_every: int = 100,  # VMC check every N epochs
    patience: int = 150,
):
    """
    Residual PDE training with stratified (i.i.d.) sampling.

    Loss = Var[E_L] under a fixed Gaussian mixture.
    When E_L = const everywhere → Ψ is exact eigenstate.

    Advantages over MCMC:
      - i.i.d. samples → lower gradient noise → stable training
      - Covers broader domain → better generalization
      - No autocorrelation → every sample is independent

    The envelope on f_nn prevents the Jastrow from growing in
    regions not covered by the sampler.
    """
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    E_ref = params.get("E", None)

    f_net.to(device).to(dtype)

    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]).to(
        device
    )

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=None, spin=spin, params=params)
        return lp

    optimizer = torch.optim.Adam(f_net.parameters(), lr=lr)

    if lr_schedule == "cosine":
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=lr * lr_min_frac
        )
    else:
        cosine_sched = None

    def _set_lr(epoch):
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * frac
        elif cosine_sched is not None:
            cosine_sched.step()

    history = []
    best_var = float("inf")
    best_E_mean = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Stratified variance-minimization: {n_epochs} epochs, lr={lr}")
    print(f"  {n_collocation} pts/epoch, exact Laplacian, micro_batch={micro_batch}")
    print(f"  Huber(δ={huber_delta}), grad_clip={grad_clip}, trim={quantile_trim}")
    print(f"  warmup={warmup_epochs}, eval_every={eval_every}, patience={patience}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()

    for epoch in range(n_epochs):
        f_net.train()
        optimizer.zero_grad(set_to_none=True)

        # Fresh i.i.d. samples each epoch
        X = sample_stratified(n_collocation, N, d, omega, device, dtype)

        all_EL = []
        n_batches = max(1, math.ceil(X.shape[0] / micro_batch))

        for i in range(0, X.shape[0], micro_batch):
            x_mb = X[i : i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega)

            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue

            if quantile_trim > 0 and E_L.numel() > 10:
                lo = torch.quantile(E_L.detach(), quantile_trim)
                hi = torch.quantile(E_L.detach(), 1.0 - quantile_trim)
                mask = (E_L >= lo) & (E_L <= hi)
                E_L = E_L[mask]
                if E_L.numel() == 0:
                    continue

            all_EL.append(E_L.detach())

            mu = E_L.mean().detach()
            resid = E_L - mu

            abs_r = resid.abs()
            loss = torch.where(
                abs_r <= huber_delta,
                0.5 * resid**2,
                huber_delta * (abs_r - 0.5 * huber_delta),
            ).mean()

            (loss / n_batches).backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(f_net.parameters(), grad_clip)
        optimizer.step()
        _set_lr(epoch)

        if len(all_EL) > 0:
            EL_all = torch.cat(all_EL)
            E_mean = EL_all.mean().item()
            E_var = EL_all.var().item()
            E_std = EL_all.std().item()
        else:
            E_mean, E_var, E_std = float("nan"), float("nan"), float("nan")

        history.append({"epoch": epoch, "E_mean": E_mean, "E_var": E_var})

        improved = False
        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            improved = True
        if E_ref and math.isfinite(E_mean) and abs(E_mean - E_ref) < abs(best_E_mean - E_ref):
            best_E_mean = E_mean
            improved = True
        if improved:
            best_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience and epoch > warmup_epochs + 50:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            ref_str = f"  (ref={E_ref})" if E_ref else ""
            print(
                f"[{epoch:4d}/{n_epochs}] E_L={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  lr={lr_now:.1e}  t={dt:.0f}s{ref_str}"
            )
            sys.stdout.flush()

        # Periodic VMC evaluation
        if eval_every > 0 and epoch > 0 and epoch % eval_every == 0:
            f_net.eval()
            vmc = evaluate_vmc(f_net, C_occ, params, n_samples=5_000, label=f"checkpoint ep{epoch}")
            f_net.train()

    # Restore best
    if best_state is not None:
        f_net.load_state_dict(best_state)
        print(f"\nRestored best model (var={best_var:.3e})")

    total_time = time.time() - t0
    print(f"Training done in {total_time:.1f}s ({total_time/60:.1f}min)")
    sys.stdout.flush()
    return f_net, history


# ─────────────────────────────────────────────────────────────────
# Training: MCMC variance-minimization (|Ψ|² sampling)
# ─────────────────────────────────────────────────────────────────


def _mcmc_burn_in(psi_log_fn, x_chain, sigma, burn_in, device, dtype):
    """Run burn-in on MCMC chain. Returns (x_chain, lp_chain, acc_rate)."""
    n_walkers = x_chain.shape[0]
    lp_chain = psi_log_fn(x_chain) * 2.0
    acc, tot = 0, 0
    for _ in range(burn_in):
        prop = x_chain + torch.randn_like(x_chain) * sigma
        lp_prop = psi_log_fn(prop) * 2.0
        log_u = torch.log(torch.rand(n_walkers, device=device, dtype=dtype))
        accept = log_u < (lp_prop - lp_chain)
        x_chain = torch.where(accept.view(-1, 1, 1), prop, x_chain)
        lp_chain = torch.where(accept, lp_prop, lp_chain)
        acc += accept.sum().item()
        tot += n_walkers
    return x_chain, lp_chain, acc / max(tot, 1)


def train_residual(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    params: dict,
    *,
    n_epochs: int = 500,
    lr: float = 3e-4,
    n_walkers: int = 512,  # persistent MCMC walkers
    rw_steps: int = 10,  # RW steps between epochs to decorrelate
    burn_in: int = 500,  # initial burn-in steps
    sigma_frac: float = 0.15,  # proposal std in units of ℓ
    micro_batch: int = 64,  # for gradient accumulation with exact Laplacian
    grad_clip: float = 0.5,
    quantile_trim: float = 0.05,
    print_every: int = 25,
    huber_delta: float = 1.0,
    lr_schedule: str = "cosine",
    lr_min_frac: float = 0.01,  # min LR as fraction of max (for cosine)
    warmup_epochs: int = 20,  # linear LR warmup
    rechain_every: int = 100,  # re-burn MCMC chain periodically to avoid drift
    patience: int = 80,  # early stop if no improvement for this many epochs
):
    """
    Pure variance-minimisation VMC with Adam.

    Every epoch:
      1. Advance persistent MCMC chain (|Ψ|²) by `rw_steps` steps
      2. Compute E_L on all walkers using exact Laplacian
      3. Loss = Huber( E_L - <E_L> )  → minimises Var[E_L]
      4. Adam gradient step

    Stability features:
      - LR warmup to avoid early instability
      - Periodic MCMC re-initialization to avoid chain drift
      - Aggressive quantile trimming + Huber loss
      - Gradient clipping
      - Best-model tracking with patience-based early stopping

    When Var[E_L] → 0, E_L = E_exact everywhere.
    """
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    E_ref = params.get("E", None)
    ell = 1.0 / math.sqrt(omega)

    f_net.to(device).to(dtype)

    # Spin
    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]).to(
        device
    )

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=None, spin=spin, params=params)
        return lp

    optimizer = torch.optim.Adam(f_net.parameters(), lr=lr)

    # Cosine schedule (applied after warmup)
    if lr_schedule == "cosine":
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=lr * lr_min_frac
        )
    else:
        cosine_sched = None

    def _set_lr(epoch):
        if epoch < warmup_epochs:
            frac = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * frac
        elif cosine_sched is not None:
            cosine_sched.step()

    # ── Initialize persistent MCMC chain ──
    sigma = sigma_frac * ell
    x_chain = torch.randn(n_walkers, N, d, device=device, dtype=dtype) * ell
    f_net.eval()
    with torch.no_grad():
        print(f"  Burning in MCMC ({burn_in} steps, {n_walkers} walkers)...", end=" ")
        sys.stdout.flush()
        x_chain, lp_chain, acc_burn = _mcmc_burn_in(
            psi_log_fn, x_chain, sigma, burn_in, device, dtype
        )
    print(f"acc={acc_burn:.3f}")
    sys.stdout.flush()

    history = []
    best_var = float("inf")
    best_E_mean = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Variance-minimization VMC: {n_epochs} epochs, lr={lr}")
    print(f"  {n_walkers} walkers, {rw_steps} RW/epoch, exact Laplacian")
    print(f"  Huber(δ={huber_delta}), grad_clip={grad_clip}, trim={quantile_trim}")
    print(f"  warmup={warmup_epochs}, rechain_every={rechain_every}, patience={patience}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()

    for epoch in range(n_epochs):
        # ── Periodic MCMC re-initialization ──
        if rechain_every > 0 and epoch > 0 and epoch % rechain_every == 0:
            f_net.eval()
            with torch.no_grad():
                x_chain = torch.randn(n_walkers, N, d, device=device, dtype=dtype) * ell
                x_chain, lp_chain, acc_re = _mcmc_burn_in(
                    psi_log_fn, x_chain, sigma, burn_in // 2, device, dtype
                )
            if epoch % print_every == 0:
                print(f"  [rechain at ep {epoch}] acc={acc_re:.3f}")

        # ── Step 1: Advance MCMC chain ──
        f_net.eval()
        with torch.no_grad():
            acc_ep, tot_ep = 0, 0
            for _ in range(rw_steps):
                prop = x_chain + torch.randn_like(x_chain) * sigma
                lp_prop = psi_log_fn(prop) * 2.0
                log_u = torch.log(torch.rand(n_walkers, device=device, dtype=dtype))
                accept = log_u < (lp_prop - lp_chain)
                x_chain = torch.where(accept.view(-1, 1, 1), prop, x_chain)
                lp_chain = torch.where(accept, lp_prop, lp_chain)
                acc_ep += accept.sum().item()
                tot_ep += n_walkers
        acc_rate = acc_ep / max(tot_ep, 1)

        # Adapt sigma to target ~50% acceptance
        if acc_rate > 0.6:
            sigma *= 1.03
        elif acc_rate < 0.4:
            sigma *= 0.97
        sigma = max(0.03 * ell, min(sigma, 0.5 * ell))

        X = x_chain.detach().clone()  # (n_walkers, N, d)

        # ── Step 2: Compute loss ──
        f_net.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        all_EL = []
        n_batches = max(1, math.ceil(X.shape[0] / micro_batch))

        for i in range(0, X.shape[0], micro_batch):
            x_mb = X[i : i + micro_batch]
            E_L = compute_local_energy(psi_log_fn, x_mb, omega)

            # Filter non-finite
            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue

            # Quantile trimming — tighter to remove outliers
            if quantile_trim > 0 and E_L.numel() > 10:
                lo = torch.quantile(E_L.detach(), quantile_trim)
                hi = torch.quantile(E_L.detach(), 1.0 - quantile_trim)
                mask = (E_L >= lo) & (E_L <= hi)
                E_L = E_L[mask]
                if E_L.numel() == 0:
                    continue

            all_EL.append(E_L.detach())

            # Variance minimization: (E_L - <E_L>)²
            mu = E_L.mean().detach()
            resid = E_L - mu

            # Huber loss
            abs_r = resid.abs()
            loss = torch.where(
                abs_r <= huber_delta,
                0.5 * resid**2,
                huber_delta * (abs_r - 0.5 * huber_delta),
            ).mean()

            (loss / n_batches).backward()
            total_loss += loss.item()

        # ── Step 3: Gradient step ──
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(f_net.parameters(), grad_clip)
        optimizer.step()
        _set_lr(epoch)

        # ── Refresh log|Ψ|² for chain (parameters changed) ──
        f_net.eval()
        with torch.no_grad():
            lp_chain = psi_log_fn(x_chain) * 2.0

        # ── Logging ──
        if len(all_EL) > 0:
            EL_all = torch.cat(all_EL)
            E_mean = EL_all.mean().item()
            E_var = EL_all.var().item()
            E_std = EL_all.std().item()
        else:
            E_mean, E_var, E_std = float("nan"), float("nan"), float("nan")

        history.append({"epoch": epoch, "E_mean": E_mean, "E_var": E_var, "acc": acc_rate})

        # Track best model by variance (primary) and E proximity to ref
        improved = False
        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            improved = True
        if E_ref and math.isfinite(E_mean) and abs(E_mean - E_ref) < abs(best_E_mean - E_ref):
            best_E_mean = E_mean
            improved = True
        if improved:
            best_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if patience > 0 and epochs_no_improve >= patience and epoch > warmup_epochs + 50:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

        if epoch % print_every == 0:
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            ref_str = f"  (ref={E_ref})" if E_ref else ""
            flag = ""
            if E_ref and E_mean < E_ref:
                flag = " ⚠ BELOW GS"
            r_avg = X.norm(dim=-1).mean().item()
            print(
                f"[{epoch:4d}/{n_epochs}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  acc={acc_rate:.2f}  <r>={r_avg:.2f}  "
                f"lr={lr_now:.1e}  t={dt:.0f}s{ref_str}{flag}"
            )
            sys.stdout.flush()

    # Restore best
    if best_state is not None:
        f_net.load_state_dict(best_state)
        print(f"\nRestored best model (var={best_var:.3e})")

    total_time = time.time() - t0
    print(f"Training done in {total_time:.1f}s ({total_time/60:.1f}min)")
    sys.stdout.flush()
    return f_net, history


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────


def evaluate_vmc(f_net, C_occ, params, n_samples=15_000, label=""):
    """Full VMC evaluation with exact Laplacian."""
    print(f"\n── Evaluating {label} ──")
    sys.stdout.flush()
    result = evaluate_energy_vmc(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=None,
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
    E = result["E_mean"]
    E_std = result["E_stderr"]
    E_ref = params.get("E", None)
    print(f"  VMC E = {E:.6f} ± {E_std:.6f}  (target: {E_ref})")
    if E_ref and E < E_ref:
        print(f"  ⚠ WARNING: E < E_GS ({E:.6f} < {E_ref})")
    sys.stdout.flush()
    return result


# ─────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────


def experiment_2e(device="cpu"):
    """2-electron ω=1.0 → target E=3.0"""
    dtype = torch.float64
    C_occ, params = setup_system(2, 1.0, device=device, dtype=dtype)

    net = (
        UnifiedCTNN(
            d=2,
            n_particles=2,
            omega=1.0,
            node_hidden=64,
            edge_hidden=64,
            msg_layers=2,
            node_layers=2,
            n_mp_steps=1,
            jastrow_hidden=32,
            jastrow_layers=2,
            envelope_width_aho=3.0,
        )
        .to(device)
        .to(dtype)
    )
    n_p = sum(p.numel() for p in net.parameters())
    print(f"Net params: {n_p:,}")

    net, hist = train_stratified(
        net,
        C_occ,
        params,
        n_epochs=600,
        lr=5e-4,
        n_collocation=1024,
        micro_batch=64,
        grad_clip=0.5,
        print_every=25,
        huber_delta=1.0,
        quantile_trim=0.05,
        lr_schedule="cosine",
        warmup_epochs=20,
        eval_every=200,
        patience=200,
    )

    result = evaluate_vmc(net, C_occ, params, n_samples=15_000, label="2e final")

    E = result["E_mean"]
    print(f"\n{'='*60}")
    print(f"2e RESULT: E = {E:.5f}  (target 3.00000, err {abs(E-3.0)/3.0*100:.2f}%)")
    print(f"{'='*60}")
    sys.stdout.flush()
    return net, result, hist


def experiment_6e(device="cpu"):
    """6-electron ω=0.5 → target E=11.78484"""
    dtype = torch.float64
    C_occ, params = setup_system(6, 0.5, device=device, dtype=dtype)

    net = (
        UnifiedCTNN(
            d=2,
            n_particles=6,
            omega=0.5,
            node_hidden=128,
            edge_hidden=128,
            msg_layers=2,
            node_layers=3,
            n_mp_steps=2,
            jastrow_hidden=64,
            jastrow_layers=2,
            envelope_width_aho=3.0,
        )
        .to(device)
        .to(dtype)
    )
    n_p = sum(p.numel() for p in net.parameters())
    print(f"Net params: {n_p:,}")

    net, hist = train_stratified(
        net,
        C_occ,
        params,
        n_epochs=1000,
        lr=3e-4,
        n_collocation=2048,
        micro_batch=64,
        grad_clip=0.5,
        print_every=25,
        huber_delta=1.0,
        quantile_trim=0.05,
        lr_schedule="cosine",
        warmup_epochs=30,
        eval_every=200,
        patience=250,
    )

    result = evaluate_vmc(net, C_occ, params, n_samples=20_000, label="6e final")

    E = result["E_mean"]
    target = 11.78484
    print(f"\n{'='*60}")
    print(f"6e RESULT: E = {E:.5f}  (target {target:.5f}, err {abs(E-target)/target*100:.2f}%)")
    print(f"{'='*60}")
    sys.stdout.flush()
    return net, result, hist


if __name__ == "__main__":
    device = "cpu"
    print(f"Device: {device}\n")

    # ── Step 1: 2e sanity check ──
    print("=" * 60)
    print("STEP 1: 2-electron (ω=1.0, target=3.0)")
    print("=" * 60)
    sys.stdout.flush()
    net2, res2, hist2 = experiment_2e(device=device)

    # ── Step 2: 6e ──
    print("\n" + "=" * 60)
    print("STEP 2: 6-electron (ω=0.5, target=11.78484)")
    print("=" * 60)
    sys.stdout.flush()
    net6, res6, hist6 = experiment_6e(device=device)

    # ── Summary ──
    E2 = res2["E_mean"]
    E6 = res6["E_mean"]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  2e: E={E2:.5f} (target=3.00000, err={abs(E2-3.0)/3.0*100:.2f}%)")
    print(f"  6e: E={E6:.5f} (target=11.78484, err={abs(E6-11.78484)/11.78484*100:.2f}%)")
    print(f"{'='*60}")
