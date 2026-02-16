"""
6e residual-based training with SCREENED COLLOCATION.

KEY IDEA:
  MCMC from |Ψ|² = VMC.  Importance weighting fails in 12D (ESS → 1).
  
  Solution: SCREENED COLLOCATION.
    1. Draw many (10–20×) candidate points from a fixed Gaussian proposal.
    2. Evaluate |Ψ|² at every candidate  (cheap forward-pass, no grads).
    3. Keep only the top-K by |Ψ(x)|²/q(x)  — these are where Ψ lives.
    4. Train on those K points with UNIFORM weights  (no IS weights).
  
  This is genuinely residual-based:
    • The proposal q(x) is a fixed Gaussian — independent of θ.
    • The selection uses |Ψ|² only for screening, with no gradient.
    • The loss & gradient come from the energy residual at the selected points.

Extras:
  • Two-phase training  (phase 1: pure variance-min,  phase 2: E_DMC targeting)
  • Cosine LR annealing
  • Warm-start option  (pre-train PINN, freeze, train backflow, unfreeze)
"""

import math, sys, time, copy
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, CTNNBackflowNet, UnifiedCTNN
from functions.Neural_Networks import (
    psi_fn,
    _laplacian_logpsi_exact,
    _make_closed_shell_spin,
)
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def setup_noninteracting(N, omega, d=2, device="cpu", dtype=torch.float64):
    n_occ = N // 2
    nx = {2: 2, 6: 3, 12: 4, 20: 5}.get(N, 4)
    ny = nx
    n_basis = nx * ny
    L = max(8.0, 3.0 / math.sqrt(omega))
    config.update(
        omega=omega, n_particles=N, d=d,
        L=L, n_grid=80, nx=nx, ny=ny,
        basis="cart", device=str(device), dtype="float64",
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
    """Local energy E_L(x) = T + V_harm + V_int, with graph for backprop."""
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap_log.view(B) + g2.view(B))
    V_harm = 0.5 * omega ** 2 * (x ** 2).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x).view(B)
    return T + V_harm + V_int


# ══════════════════════════════════════════════════════════════════
#  Proposal sampling  &  importance weights
# ══════════════════════════════════════════════════════════════════

def sample_gaussian_proposal(n_samples, N, d, sigma, device, dtype):
    """
    Fixed isotropic Gaussian proposal q(x) = prod_i N(r_i; 0, sigma^2 I_d).
    Returns (x, log_q).
    """
    x = torch.randn(n_samples, N, d, device=device, dtype=dtype) * sigma
    Nd = N * d
    log_q = (
        -0.5 * Nd * math.log(2 * math.pi * sigma ** 2)
        - x.reshape(n_samples, -1).pow(2).sum(-1) / (2 * sigma ** 2)
    )
    return x, log_q


@torch.no_grad()
def screened_collocation(psi_log_fn, N, d, sigma, n_keep, oversampling,
                         device, dtype, batch_size=4096):
    """
    Generate oversampling×n_keep candidate points from a Gaussian proposal,
    evaluate |Ψ|² at each, and return the top n_keep by |Ψ|²/q ratio.

    Returns x of shape (n_keep, N, d) — NO importance weights needed.
    """
    n_cand = oversampling * n_keep
    x_all, log_q_all = sample_gaussian_proposal(n_cand, N, d, sigma, device, dtype)

    # Evaluate log|Ψ|² in batches (forward-only, cheap)
    log_psi2_parts = []
    for i in range(0, n_cand, batch_size):
        lp = psi_log_fn(x_all[i:i + batch_size])
        log_psi2_parts.append(2.0 * lp)
    log_psi2 = torch.cat(log_psi2_parts)

    # Importance ratio
    log_ratio = log_psi2 - log_q_all
    # Top-K selection
    _, idx = torch.topk(log_ratio, n_keep)
    return x_all[idx].clone()


# ══════════════════════════════════════════════════════════════════
#  Residual trainer  (screened collocation)
# ══════════════════════════════════════════════════════════════════

def train_residual(
    f_net, C_occ, params, *,
    backflow_net=None,
    # --- schedule ---
    n_epochs=300,
    lr=3e-4,
    lr_min_frac=0.02,          # min LR as fraction of initial
    # --- phase 1 (pure var-min) / phase 2 (E_DMC targeting) ---
    phase1_frac=0.25,          # fraction of epochs for phase 1
    alpha_end=0.60,            # max alpha at end of phase 2
    # --- collocation ---
    n_collocation=2048,
    oversampling=10,           # generate this × n_collocation candidates
    proposal_sigma_factor=1.3, # sigma = factor * ell
    micro_batch=256,
    # --- robustness ---
    grad_clip=0.5,
    quantile_trim=0.02,
    print_every=10,
):
    """
    Screened-collocation energy-variance minimisation.

    Each epoch:
      1. Draw oversampling × n_collocation points from N(0, σ²I).
      2. Evaluate |Ψ|² at all candidates (forward-only, fast).
      3. Keep top n_collocation by |Ψ|²/q  (= screened collocation).
      4. Compute local energy + loss on those points (uniform weights).

    Phase 1 (epochs 0 … phase1_frac × n_epochs):
        α = 0  →  pure variance minimisation.
    Phase 2 (remaining epochs):
        α cosine-ramps 0 → alpha_end  →  target E_DMC.

    LR follows a cosine schedule over all epochs.
    """
    device = params["device"]
    dtype  = params.get("torch_dtype", torch.float64)
    omega  = float(params["omega"])
    N      = int(params["n_particles"])
    d      = int(params["d"])
    E_DMC  = params.get("E", None)
    ell    = 1.0 / math.sqrt(omega)

    sigma  = proposal_sigma_factor * ell

    f_net.to(device).to(dtype)
    if backflow_net is not None:
        backflow_net.to(device).to(dtype)

    up   = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(device)

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net,
                        spin=spin, params=params)
        return lp

    all_params = list(f_net.parameters())
    if backflow_net is not None:
        all_params += list(backflow_net.parameters())
    n_p = sum(p.numel() for p in all_params)

    optimizer = torch.optim.Adam(all_params, lr=lr)
    lr_min = lr * lr_min_frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (lr_min + 0.5 * (lr - lr_min) *
                              (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))) / lr,
    )

    phase1_end = int(phase1_frac * n_epochs)

    print(f"\n{'='*60}")
    print(f"Screened-collocation training:  {n_epochs} ep, "
          f"{n_collocation} pts (from {oversampling*n_collocation} candidates)")
    print(f"  {n_p:,} params, lr={lr}, cosine → {lr_min:.1e}")
    print(f"  proposal σ = {sigma:.3f}  (ℓ = {ell:.3f})")
    print(f"  phase 1 (var-min): epochs 0–{phase1_end}")
    print(f"  phase 2 (targeting): α 0→{alpha_end}, eps {phase1_end}–{n_epochs}")
    print(f"  E_DMC = {E_DMC}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    history = []
    best_var = float("inf")
    best_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    patience = 60

    for epoch in range(n_epochs):
        # ── Alpha schedule ──
        if epoch < phase1_end:
            alpha = 0.0
        else:
            t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
            alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))

        # ── Screened collocation: select top-K from Gaussian candidates ──
        f_net.eval()
        if backflow_net is not None:
            backflow_net.eval()
        X = screened_collocation(
            psi_log_fn, N, d, sigma,
            n_keep=n_collocation, oversampling=oversampling,
            device=device, dtype=dtype,
        )

        # ── Compute loss in micro-batches (uniform weights) ──
        f_net.train()
        if backflow_net is not None:
            backflow_net.train()
        optimizer.zero_grad(set_to_none=True)

        all_EL = []
        n_batches = max(1, math.ceil(n_collocation / micro_batch))

        for i in range(0, n_collocation, micro_batch):
            x_mb = X[i:i + micro_batch]

            E_L = compute_local_energy(psi_log_fn, x_mb, omega).view(-1)

            # Filter non-finite
            good = torch.isfinite(E_L)
            if not good.all():
                E_L = E_L[good]
            if E_L.numel() == 0:
                continue

            # Quantile trimming
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
            loss_mb = (resid ** 2).mean()      # uniform weights
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

        history.append({"epoch": epoch, "E_mean": E_mean, "E_var": E_var})

        # ── Best model tracking ──
        if math.isfinite(E_var) and E_var < best_var * 0.999:
            best_var = E_var
            best_state = {k: v.clone() for k, v in f_net.state_dict().items()}
            if backflow_net is not None:
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

    # Restore best
    if best_state:
        f_net.load_state_dict(best_state)
        if backflow_net is not None and best_bf_state:
            backflow_net.load_state_dict(best_bf_state)
        print(f"Restored best model (var={best_var:.3e})")

    total = time.time() - t0
    print(f"Training done in {total:.0f}s ({total/60:.1f}min)")
    return f_net, backflow_net, history


# ══════════════════════════════════════════════════════════════════
#  Evaluate  (VMC evaluation — this IS correct VMC for evaluation)
# ══════════════════════════════════════════════════════════════════

def evaluate(f_net, C_occ, params, backflow_net=None, n_samples=15_000, label=""):
    print(f"\n── VMC eval: {label} ──")
    result = evaluate_energy_vmc(
        f_net, C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=backflow_net, params=params,
        n_samples=n_samples, batch_size=512,
        sampler_steps=50, sampler_step_sigma=0.12,
        lap_mode="exact",
        persistent=True, sampler_burn_in=300, sampler_thin=3,
        progress=True,
    )
    E, E_std = result["E_mean"], result["E_stderr"]
    E_ref = params.get("E")
    err = abs(E - E_ref) / abs(E_ref) * 100 if E_ref else 0
    print(f"  E = {E:.6f} ± {E_std:.6f}  (target {E_ref}, err {err:.2f}%)")
    return result


# ══════════════════════════════════════════════════════════════════
#  Run configurations
# ══════════════════════════════════════════════════════════════════

COMMON = dict(
    n_epochs=300, lr=3e-4,
    n_collocation=2048, oversampling=10, micro_batch=256,
    grad_clip=0.5, print_every=10,
    phase1_frac=0.25, alpha_end=0.60,
    proposal_sigma_factor=1.3,
)


def run_pinn():
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(device).to(dtype)
    print(f"PINN params: {sum(p.numel() for p in f_net.parameters()):,}")
    f_net, _, hist = train_residual(f_net, C_occ, params, **COMMON)
    return evaluate(f_net, C_occ, params, label="PINN dL=8 (residual)")


def run_ctnn_pinn():
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(device).to(dtype)
    bf_net = CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=0.05,
        omega=0.5,
    ).to(device).to(dtype)
    np_total = (sum(p.numel() for p in f_net.parameters())
              + sum(p.numel() for p in bf_net.parameters()))
    print(f"CTNN+PINN params: {np_total:,}")
    f_net, bf_net, hist = train_residual(
        f_net, C_occ, params, backflow_net=bf_net, **COMMON,
    )
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="CTNN+PINN (residual)")


def run_warmstart_ctnn():
    """
    Warm-start:
      1) Train PINN alone for 60% of epochs
      2) Freeze PINN, attach CTNN backflow, train backflow for 20%
      3) Unfreeze both, fine-tune for remaining 20%
    """
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)

    f_net = PINN(
        n_particles=6, d=2, omega=0.5,
        dL=8, hidden_dim=64, n_layers=2,
        act="gelu", init="xavier",
        use_gate=True, use_pair_attn=False,
    ).to(device).to(dtype)

    total_epochs = COMMON["n_epochs"]
    ep1 = int(0.60 * total_epochs)  # PINN only
    ep2 = int(0.20 * total_epochs)  # backflow only (PINN frozen)
    ep3 = total_epochs - ep1 - ep2  # fine-tune both

    # ── Phase 1: PINN only ──
    print(f"\n{'─'*40}")
    print(f"Warm-start phase 1: PINN only ({ep1} ep)")
    print(f"{'─'*40}")
    f_net, _, _ = train_residual(
        f_net, C_occ, params,
        n_epochs=ep1, lr=3e-4,
        n_collocation=2048, oversampling=10, micro_batch=256,
        grad_clip=0.5, print_every=10,
        phase1_frac=0.30, alpha_end=0.50,
        proposal_sigma_factor=1.3,
    )

    # ── Phase 2: freeze PINN, train backflow ──
    print(f"\n{'─'*40}")
    print(f"Warm-start phase 2: backflow only, PINN frozen ({ep2} ep)")
    print(f"{'─'*40}")
    bf_net = CTNNBackflowNet(
        d=2, msg_hidden=32, msg_layers=1,
        hidden=32, layers=2,
        act="silu", aggregation="sum",
        use_spin=True, same_spin_only=False,
        out_bound="tanh", bf_scale_init=0.05,
        omega=0.5,
    ).to(device).to(dtype)

    # Freeze PINN
    for p in f_net.parameters():
        p.requires_grad_(False)

    f_net, bf_net, _ = train_residual(
        f_net, C_occ, params, backflow_net=bf_net,
        n_epochs=ep2, lr=5e-4,      # slightly larger LR for backflow alone
        n_collocation=2048, oversampling=10, micro_batch=256,
        grad_clip=0.5, print_every=10,
        phase1_frac=0.0, alpha_end=0.50,   # skip phase 1, start targeting
        proposal_sigma_factor=1.3,
    )

    # ── Phase 3: unfreeze both, fine-tune ──
    print(f"\n{'─'*40}")
    print(f"Warm-start phase 3: fine-tune both ({ep3} ep)")
    print(f"{'─'*40}")
    for p in f_net.parameters():
        p.requires_grad_(True)

    f_net, bf_net, _ = train_residual(
        f_net, C_occ, params, backflow_net=bf_net,
        n_epochs=ep3, lr=1e-4,      # lower LR for fine-tuning
        n_collocation=2048, oversampling=10, micro_batch=256,
        grad_clip=0.3, print_every=10,
        phase1_frac=0.0, alpha_end=0.60,
        proposal_sigma_factor=1.3,
    )

    np_total = (sum(p.numel() for p in f_net.parameters())
              + sum(p.numel() for p in bf_net.parameters()))
    print(f"Warm-start CTNN+PINN total params: {np_total:,}")
    return evaluate(f_net, C_occ, params, backflow_net=bf_net,
                    label="Warm-start CTNN+PINN (residual)")


def run_unified():
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_noninteracting(6, 0.5, device=device, dtype=dtype)
    net = UnifiedCTNN(
        d=2, n_particles=6, omega=0.5,
        node_hidden=64, edge_hidden=64,
        msg_layers=1, node_layers=2, n_mp_steps=1,
        jastrow_hidden=32, jastrow_layers=2,
        envelope_width_aho=3.0,
    ).to(device).to(dtype)
    print(f"UnifiedCTNN params: {sum(p.numel() for p in net.parameters()):,}")
    net, _, hist = train_residual(net, C_occ, params, lr=2e-4, **{
        k: v for k, v in COMMON.items() if k != "lr"
    })
    return evaluate(net, C_occ, params, label="UnifiedCTNN (residual)")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = {}

    for name, fn in [
        ("pinn",       run_pinn),
        ("ctnn",       run_ctnn_pinn),
        ("warmstart",  run_warmstart_ctnn),
        ("unified",    run_unified),
    ]:
        print(f"\n{'#'*60}")
        print(f"# {name.upper()}")
        print(f"{'#'*60}")
        results[name] = fn()

    target = 11.78484
    print(f"\n{'='*60}")
    print("SUMMARY — 6e ω=0.5  (importance-weighted residual)")
    print(f"{'='*60}")
    for name, r in results.items():
        E, se = r["E_mean"], r["E_stderr"]
        err = abs(E - target) / target * 100
        print(f"  {name:15s}  E={E:.6f} ± {se:.6f}  err={err:.2f}%")
    print(f"  {'DMC target':15s}  E={target:.6f}")
    print(f"{'='*60}")
