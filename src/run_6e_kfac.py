"""
K-FAC + gradient conditioning experiments for backflow.

Diagnoses:
  The SD inverse M^{-T} creates ill-conditioned gradients for BF params.
  Adam rescales per-parameter but cannot rotate the gradient direction.

Experiments:
  1. kfac_bf         — K-FAC preconditioner on BF linear layers, Adam on Jastrow
  2. smooth_bf       — Smooth BF + penalty on ‖∇²Δx‖² to reduce Laplacian noise
  3. varmin_pretrain — Variance-minimization pre-training (α=0 always) for BF,
                       then switch to full energy-targeting joint training
  4. damped_sdinv    — Damped SD inverse in backward pass only (forward unchanged)
  5. kfac_smooth     — Combine K-FAC + smoothness penalty (the full treatment)

All use bf_scale=0.7, zero_init_last=False, 300ep effective, joint training.
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

from functions.Neural_Networks import (
    psi_fn,
)
from PINN import PINN, CTNNBackflowNet
from run_6e_residual import (
    compute_local_energy,
    evaluate,
    screened_collocation,
    setup_noninteracting,
)

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64


# ══════════════════════════════════════════════════════════════════
#                        Helpers
# ══════════════════════════════════════════════════════════════════


def make_nets(bf_scale_init=0.7, zero_init_last=False):
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
        .to(DEVICE)
        .to(DTYPE)
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
            bf_scale_init=bf_scale_init,
            zero_init_last=zero_init_last,
            omega=0.5,
        )
        .to(DEVICE)
        .to(DTYPE)
    )
    return f_net, bf_net


def make_psi_log_fn(f_net, bf_net, C_occ, params):
    up = params["n_particles"] // 2
    N = params["n_particles"]
    device = params["device"]
    spin = torch.cat([torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]).to(
        device
    )

    def fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return lp

    return fn, spin


def cosine_sched(optimizer, n_epochs, lr_min_frac=0.02):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=optimizer.param_groups[0]["lr"] * lr_min_frac,
    )


# ══════════════════════════════════════════════════════════════════
#  1.  K-FAC PRECONDITIONER FOR BACKFLOW LINEAR LAYERS
# ══════════════════════════════════════════════════════════════════
# K-FAC approximates the Fisher for a linear layer y = Wx + b as:
#     F ≈ (E[aa^T]) ⊗ (E[gg^T])   =  A ⊗ G
# where a = input activation (augmented with 1 for bias),
#       g = output-side gradient (from loss through this layer).
#
# The preconditioned gradient is:  A^{-1} ∇_W G^{-1}
# (reshaped back from matrix form).
#
# We hook into each Linear layer to capture activations and output grads.
# After backward(), we update running averages of A and G, invert them,
# and replace each parameter's .grad with the preconditioned version.


class KFACPreconditioner:
    """
    Layer-wise K-FAC for nn.Linear modules.

    Usage:
        kfac = KFACPreconditioner(bf_net, damping=1e-3, ema=0.95)
        kfac.register_hooks()
        ...
        loss.backward()
        kfac.step()  # preconditions .grad in-place
        optimizer.step()
    """

    def __init__(
        self, module: nn.Module, damping: float = 1e-3, ema: float = 0.95, update_freq: int = 1
    ):
        self.damping = damping
        self.ema = ema
        self.update_freq = update_freq
        self._step_counter = 0

        # Collect all nn.Linear layers
        self.linear_layers = {}  # name -> (module, {A, G, ...})
        self.hooks = []

        for name, mod in module.named_modules():
            if isinstance(mod, nn.Linear):
                self.linear_layers[name] = {
                    "module": mod,
                    "A": None,  # running E[aa^T]
                    "G": None,  # running E[gg^T]
                    "a_save": None,  # saved input activation
                }

    def register_hooks(self):
        """Register forward/backward hooks on each Linear layer."""
        for name, info in self.linear_layers.items():
            mod = info["module"]

            def make_fwd_hook(n):
                def hook(module, inp, out):
                    a = inp[0].detach()  # input activations
                    # Flatten all batch-like dimensions: (*, d_in) -> (M, d_in)
                    a = a.reshape(-1, a.shape[-1])
                    # Augment with 1 for bias
                    if module.bias is not None:
                        ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
                        a = torch.cat([a, ones], dim=-1)
                    self.linear_layers[n]["a_save"] = a

                return hook

            def make_bwd_hook(n):
                def hook(module, grad_in, grad_out):
                    g = grad_out[0].detach()  # output gradient
                    g = g.reshape(-1, g.shape[-1])
                    self.linear_layers[n]["g_save"] = g

                return hook

            h1 = mod.register_forward_hook(make_fwd_hook(name))
            h2 = mod.register_full_backward_hook(make_bwd_hook(name))
            self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def step(self):
        """Update running A,G estimates and precondition .grad in-place."""
        self._step_counter += 1

        for name, info in self.linear_layers.items():
            mod = info["module"]
            a = info.get("a_save")
            g = info.get("g_save")
            if a is None or g is None:
                continue

            M = a.shape[0]
            # Compute batch A and G Kronecker factors
            A_batch = (a.T @ a) / M  # (d_in+1, d_in+1) or (d_in, d_in)
            G_batch = (g.T @ g) / M  # (d_out, d_out)

            # EMA update
            if info["A"] is None:
                info["A"] = A_batch
                info["G"] = G_batch
            else:
                ema = self.ema
                info["A"] = ema * info["A"] + (1 - ema) * A_batch
                info["G"] = ema * info["G"] + (1 - ema) * G_batch

            # Precondition gradients every update_freq steps
            if self._step_counter % self.update_freq != 0:
                continue

            A = info["A"]
            G = info["G"]

            # Damped inverse:  (A + λI)^{-1}  and  (G + λI)^{-1}
            d_A = A.shape[0]
            d_G = G.shape[0]
            A_inv = torch.linalg.solve(
                A + self.damping * torch.eye(d_A, device=A.device, dtype=A.dtype),
                torch.eye(d_A, device=A.device, dtype=A.dtype),
            )
            G_inv = torch.linalg.solve(
                G + self.damping * torch.eye(d_G, device=G.device, dtype=G.dtype),
                torch.eye(d_G, device=G.device, dtype=G.dtype),
            )

            # Precondition: reshape grad_W to (d_out, d_in+1), apply G_inv @ grad @ A_inv
            grad_W = mod.weight.grad  # (d_out, d_in)
            if grad_W is None:
                continue

            if mod.bias is not None and mod.bias.grad is not None:
                # Augment: stack [grad_W, grad_b] → (d_out, d_in+1)
                grad_aug = torch.cat([grad_W, mod.bias.grad.unsqueeze(1)], dim=1)
            else:
                grad_aug = grad_W

            # K-FAC update: G^{-1} @ grad @ A^{-1}
            precond = G_inv @ grad_aug @ A_inv

            # Write back
            mod.weight.grad = precond[:, : mod.weight.shape[1]].contiguous()
            if mod.bias is not None and mod.bias.grad is not None:
                mod.bias.grad = precond[:, -1].contiguous()

            # Clear saved activations
            info["a_save"] = None
            info.pop("g_save", None)


# ══════════════════════════════════════════════════════════════════
#  2.  SMOOTHNESS REGULARIZATION  ON  ∇²Δx
# ══════════════════════════════════════════════════════════════════
# Penalise the Laplacian of Δx to ensure smoother displacement fields.
# This reduces noise in the kinetic energy Laplacian which compounds
# the SD^{-1} singularity.


def compute_bf_laplacian_penalty(bf_net, x, spin, n_samples=64):
    """
    Estimate ‖∇²Δx‖² via Hutchinson trace estimator.

    Returns scalar penalty = E[ Σ_i Σ_j (∂²Δx_i / ∂x_j²)² ] / (N*d)

    Uses Hutchinson: Tr(H) ≈ v^T H v  with v ~ Rademacher
    """
    x_sub = x[:n_samples].detach().requires_grad_(True)
    B, N, d = x_sub.shape

    dx = bf_net(x_sub, spin)  # (B, N, d)

    # First derivatives of each component of dx w.r.t. x
    # dx has shape (B, N, d), x_sub has shape (B, N, d)
    # We want ∂²(dx_{n,k}) / ∂x² summed

    # Efficient Hutchinson: pick random v, compute v^T H v
    n_probes = 2
    lap_sq_sum = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    for _ in range(n_probes):
        v = torch.empty_like(x_sub).bernoulli_(0.5).mul_(2).add_(-1)  # ±1

        # Compute Jv for each output component of dx
        for k in range(d):
            # Sum over particles and batch for this displacement component
            dx_k_sum = dx[:, :, k].sum()
            grad1 = torch.autograd.grad(dx_k_sum, x_sub, create_graph=True, retain_graph=True)[
                0
            ]  # (B, N, d)

            # Directional second derivative: v^T H v
            Hv = torch.autograd.grad(
                (grad1 * v).sum(), x_sub, create_graph=True, retain_graph=True
            )[
                0
            ]  # (B, N, d)

            # Tr(H²) ≈ ‖Hv‖² gives us the Frobenius norm of the Hessian
            # But we really want sum of diagonal second derivs squared
            # Hutchinson: v^T H v ≈ Tr(H) per component
            vTHv = (v * Hv).sum(dim=(1, 2))  # (B,)
            lap_sq_sum = lap_sq_sum + (vTHv**2).mean()

    penalty = lap_sq_sum / (n_probes * d)
    return penalty


# ══════════════════════════════════════════════════════════════════
#  3.  DAMPED SD INVERSE IN BACKWARD PASS
# ══════════════════════════════════════════════════════════════════
# Replace the standard slogdet backward (which uses M^{-T}) with a
# damped version that caps the condition number.
# The forward pass is UNCHANGED — only the gradient is modified.
# This is a custom autograd function.

# Save the ORIGINAL slogdet before any monkey-patching
_ORIGINAL_SLOGDET = torch.linalg.slogdet


class DampedSlogdet(torch.autograd.Function):
    """
    Forward:  sign, logabs = slogdet(M)     (exact, no change)
    Backward: ∂logabs/∂M = (M^T M + λI)^{-1} M^T  instead of M^{-T}

    This caps the gradient magnitude near nodes where M is singular.
    The wavefunction itself is unaffected.
    """

    DAMPING = 1e-4  # class-level default

    @staticmethod
    def forward(ctx, M):
        sign, logabs = _ORIGINAL_SLOGDET(M)  # use the saved original
        ctx.save_for_backward(M)
        return sign, logabs

    @staticmethod
    def backward(ctx, grad_sign, grad_logabs):
        (M,) = ctx.saved_tensors
        # Standard: grad_M = grad_logabs * M^{-T}
        # Damped:   grad_M = grad_logabs * (M^T M + λI)^{-1} M^T
        lam = DampedSlogdet.DAMPING

        # (M^T M + λI)^{-1} M^T is the Tikhonov-regularized pseudo-inverse
        # For small matrices (3×3), this is cheap
        MtM = M.transpose(-2, -1) @ M  # (B, n, n)
        n = MtM.shape[-1]
        eye = lam * torch.eye(n, device=M.device, dtype=M.dtype)
        # Solve (MtM + λI) X = M^T → X = (MtM + λI)^{-1} M^T
        rhs = M.transpose(-2, -1)  # (B, n, n)
        # X has shape (B, n, n)
        X = torch.linalg.solve(MtM + eye, rhs)

        # grad_M = grad_logabs.unsqueeze(-1).unsqueeze(-1) * X.transpose(-2, -1)
        gl = grad_logabs
        while gl.ndim < M.ndim:
            gl = gl.unsqueeze(-1)
        grad_M = gl * X.transpose(-2, -1)

        return grad_M


def damped_slogdet(M, damping=1e-4):
    """Drop-in replacement for torch.linalg.slogdet with damped backward."""
    DampedSlogdet.DAMPING = damping
    return DampedSlogdet.apply(M)


# ══════════════════════════════════════════════════════════════════
#  GENERIC TRAINER with all options
# ══════════════════════════════════════════════════════════════════


def train_kfac(
    f_net,
    bf_net,
    C_occ,
    params,
    *,
    optimizer,
    n_epochs=300,
    n_collocation=2048,
    oversampling=10,
    micro_batch=256,
    grad_clip=0.5,
    phase1_frac=0.25,
    alpha_end=0.60,
    print_every=10,
    label="",
    # New options:
    kfac_precond=None,  # KFACPreconditioner instance or None
    smooth_lambda=0.0,  # smoothness penalty weight
    varmin_epochs=0,  # extra var-min (α=0) epochs at start
    use_damped_sd=False,  # whether to monkey-patch slogdet
    damping_sd=1e-4,  # damping for SD inverse
):
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    ell = 1.0 / math.sqrt(omega)
    sigma = 1.3 * ell

    psi_log_fn, spin = make_psi_log_fn(f_net, bf_net, C_occ, params)

    scheduler = cosine_sched(optimizer, n_epochs)

    phase1_end = int(phase1_frac * n_epochs)

    # If damped SD, monkey-patch torch.linalg.slogdet for backward only
    _orig_slogdet = None
    if use_damped_sd:
        _orig_slogdet = torch.linalg.slogdet
        torch.linalg.slogdet = lambda M: damped_slogdet(M, damping_sd)
        print(f"  [damped SD⁻¹] damping={damping_sd:.1e}")

    n_f = sum(p.numel() for p in f_net.parameters() if p.requires_grad)
    n_bf = sum(p.numel() for p in bf_net.parameters() if p.requires_grad)
    lrs = [f"{pg['lr']:.1e}" for pg in optimizer.param_groups]

    print(f"\n{'─'*55}")
    print(f"  {label}")
    hdr_parts = [f"{n_epochs}ep", f"f={n_f:,}", f"bf={n_bf:,}"]
    if kfac_precond:
        hdr_parts.append(f"K-FAC({len(kfac_precond.linear_layers)} layers)")
    if smooth_lambda > 0:
        hdr_parts.append(f"smooth_λ={smooth_lambda:.1e}")
    if varmin_epochs > 0:
        hdr_parts.append(f"varmin_pre={varmin_epochs}")
    if use_damped_sd:
        hdr_parts.append(f"damped_sd={damping_sd:.1e}")
    print(f"  {', '.join(hdr_parts)}, lr={lrs}")
    print(f"{'─'*55}")
    sys.stdout.flush()

    t0 = time.time()
    best_var = float("inf")
    best_f_state = {}
    best_bf_state = {}
    epochs_no_improve = 0
    patience = 60

    total_epochs = varmin_epochs + n_epochs

    for epoch_global in range(total_epochs):
        # Phase logic: varmin pre-training uses α=0 always
        if epoch_global < varmin_epochs:
            alpha = 0.0
            epoch_label = f"varmin {epoch_global}/{varmin_epochs}"
        else:
            epoch = epoch_global - varmin_epochs
            if epoch < phase1_end:
                alpha = 0.0
            else:
                t2 = (epoch - phase1_end) / max(1, n_epochs - phase1_end - 1)
                alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * t2))
            epoch_label = None

        # --- screened collocation sampling ---
        f_net.eval()
        bf_net.eval()
        X = screened_collocation(
            psi_log_fn,
            N,
            d,
            sigma,
            n_keep=n_collocation,
            oversampling=oversampling,
            device=device,
            dtype=dtype,
        )

        f_net.train()
        bf_net.train()
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
            loss_mb = (resid**2).mean()

            # --- smoothness penalty ---
            if smooth_lambda > 0:
                pen = compute_bf_laplacian_penalty(
                    bf_net, x_mb, spin, n_samples=min(32, x_mb.shape[0])
                )
                loss_mb = loss_mb + smooth_lambda * pen

            (loss_mb / n_batches).backward()

        # --- K-FAC preconditioning of BF gradients ---
        if kfac_precond is not None:
            kfac_precond.step()

        if grad_clip > 0:
            if kfac_precond is not None:
                # Clip BF and Jastrow SEPARATELY so K-FAC-inflated BF grads
                # don't suppress the Jastrow learning
                bf_p = [p for p in bf_net.parameters() if p.requires_grad and p.grad is not None]
                f_p = [p for p in f_net.parameters() if p.requires_grad and p.grad is not None]
                if bf_p:
                    nn.utils.clip_grad_norm_(bf_p, grad_clip)
                if f_p:
                    nn.utils.clip_grad_norm_(f_p, grad_clip)
            else:
                all_p = [
                    p for pg in optimizer.param_groups for p in pg["params"] if p.requires_grad
                ]
                nn.utils.clip_grad_norm_(all_p, grad_clip)

        optimizer.step()
        scheduler.step()

        # --- logging ---
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

        real_epoch = epoch_global - varmin_epochs if epoch_global >= varmin_epochs else epoch_global
        if patience > 0 and epochs_no_improve >= patience and real_epoch > phase1_end + 30:
            print(f"  Early stop at epoch {epoch_global}  (best var={best_var:.3e})")
            break

        if epoch_global % print_every == 0:
            dt = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            err = abs(E_mean - E_DMC) / E_DMC * 100
            bf_s = bf_net.bf_scale.item()

            bf_gnorm = (
                sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in bf_net.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            f_gnorm = (
                sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in f_net.parameters()
                    if p.grad is not None and p.requires_grad
                )
                ** 0.5
            )

            with torch.no_grad():
                bf_net.eval()
                dx_sample = bf_net(X[:64], spin)
                bf_mag = dx_sample.norm(dim=-1).mean().item()
                bf_net.train()

            phase_tag = ""
            if epoch_global < varmin_epochs:
                phase_tag = " [varmin]"
            elif epoch_global - varmin_epochs < phase1_end:
                phase_tag = " [ph1]"

            print(
                f"[{epoch_global:4d}] E={E_mean:.5f} ± {E_std:.4f}  "
                f"var={E_var:.3e}  α={alpha:.3f}  lr={cur_lr:.1e}  "
                f"|Δx|={bf_mag:.3f} ‖∇bf‖={bf_gnorm:.3f} ‖∇f‖={f_gnorm:.3f}  "
                f"err={err:.2f}%{phase_tag}"
            )
            sys.stdout.flush()

    if best_f_state:
        f_net.load_state_dict(best_f_state)
    if best_bf_state:
        bf_net.load_state_dict(best_bf_state)

    # Restore slogdet
    if _orig_slogdet is not None:
        torch.linalg.slogdet = _orig_slogdet

    total = time.time() - t0
    print(f"  Best var={best_var:.3e}, {total:.0f}s ({total/60:.1f}min)")
    sys.stdout.flush()
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  MODEL SAVE / LOAD
# ══════════════════════════════════════════════════════════════════

import os

CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"


def save_model(f_net, bf_net, name):
    """Save f_net + bf_net after training."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"6e_kfac_{name}.pt")
    torch.save(
        {
            "f_net": f_net.state_dict(),
            "bf_net": bf_net.state_dict(),
        },
        path,
    )
    print(f"  💾 Saved model → {path}")


def load_model(name, bf_scale_init=0.7, zero_init_last=False):
    """Load f_net + bf_net from checkpoint. Returns (f_net, bf_net) or None."""
    path = os.path.join(CKPT_DIR, f"6e_kfac_{name}.pt")
    if not os.path.exists(path):
        return None
    f_net, bf_net = make_nets(bf_scale_init, zero_init_last)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    f_net.load_state_dict(ckpt["f_net"])
    bf_net.load_state_dict(ckpt["bf_net"])
    print(f"  📦 Loaded model ← {path}")
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════


def run_eval(f_net, bf_net, C_occ, params, label):
    """Shared VMC evaluation."""
    result = evaluate(f_net, C_occ, params, backflow_net=bf_net, label=label)
    E_mean = result["E_mean"]
    E_std = result["E_stderr"]
    err = abs(E_mean - E_DMC) / E_DMC * 100
    return E_mean, E_std, err


def run_experiment(name, train_fn, C_occ, params, force=False):
    """
    Run an experiment: check for saved model first.
    train_fn(C_occ, params) -> (f_net, bf_net)
    Returns (E_mean, E_std, err).
    """
    if not force:
        loaded = load_model(name)
        if loaded is not None:
            f_net, bf_net = loaded
            print("  Skipping training — using saved model")
            return run_eval(f_net, bf_net, C_occ, params, name)

    f_net, bf_net = train_fn(C_occ, params)
    save_model(f_net, bf_net, name)
    return run_eval(f_net, bf_net, C_occ, params, name)


# ────────────────────────────────────────────────────
#  1. kfac_sepclip — K-FAC with separate BF/Jastrow grad clipping
#     (Previous kfac_bf had ‖∇f‖=0 because joint clip was dominated
#      by K-FAC-inflated BF grads. Now they clip independently.)
# ────────────────────────────────────────────────────
def train_kfac_sepclip(C_occ, params):
    f_net, bf_net = make_nets(bf_scale_init=0.7, zero_init_last=False)

    optimizer = torch.optim.Adam(
        [
            {"params": list(f_net.parameters()), "lr": 3e-4},
            {"params": list(bf_net.parameters()), "lr": 3e-4},
        ]
    )

    kfac = KFACPreconditioner(bf_net, damping=1e-3, ema=0.95, update_freq=1)
    kfac.register_hooks()

    f_net, bf_net = train_kfac(
        f_net,
        bf_net,
        C_occ,
        params,
        optimizer=optimizer,
        n_epochs=300,
        kfac_precond=kfac,
        label="kfac_sepclip (K-FAC on BF, separate grad clipping)",
    )

    kfac.remove_hooks()
    return f_net, bf_net


# ────────────────────────────────────────────────────
#  2. smooth_bf — smoothness penalty on ‖∇²Δx‖²
# ────────────────────────────────────────────────────
def train_smooth(C_occ, params):
    f_net, bf_net = make_nets(bf_scale_init=0.7, zero_init_last=False)

    optimizer = torch.optim.Adam(
        [
            {"params": list(f_net.parameters()), "lr": 3e-4},
            {"params": list(bf_net.parameters()), "lr": 3e-4},
        ]
    )

    f_net, bf_net = train_kfac(
        f_net,
        bf_net,
        C_occ,
        params,
        optimizer=optimizer,
        n_epochs=300,
        smooth_lambda=1e-3,
        label="smooth_bf (‖∇²Δx‖² penalty, λ=1e-3)",
    )
    return f_net, bf_net


# ────────────────────────────────────────────────────
#  3. varmin_pretrain — 50 extra α=0 epochs then normal
# ────────────────────────────────────────────────────
def train_varmin(C_occ, params):
    f_net, bf_net = make_nets(bf_scale_init=0.7, zero_init_last=False)

    optimizer = torch.optim.Adam(
        [
            {"params": list(f_net.parameters()), "lr": 3e-4},
            {"params": list(bf_net.parameters()), "lr": 3e-4},
        ]
    )

    f_net, bf_net = train_kfac(
        f_net,
        bf_net,
        C_occ,
        params,
        optimizer=optimizer,
        n_epochs=300,
        varmin_epochs=50,
        label="varmin_pretrain (50ep α=0 pre-train → 300ep normal)",
    )
    return f_net, bf_net


# ────────────────────────────────────────────────────
#  4. kfac_smooth — K-FAC + smoothness penalty
# ────────────────────────────────────────────────────
def train_kfac_smooth(C_occ, params):
    f_net, bf_net = make_nets(bf_scale_init=0.7, zero_init_last=False)

    optimizer = torch.optim.Adam(
        [
            {"params": list(f_net.parameters()), "lr": 3e-4},
            {"params": list(bf_net.parameters()), "lr": 3e-4},
        ]
    )

    kfac = KFACPreconditioner(bf_net, damping=1e-3, ema=0.95, update_freq=1)
    kfac.register_hooks()

    f_net, bf_net = train_kfac(
        f_net,
        bf_net,
        C_occ,
        params,
        optimizer=optimizer,
        n_epochs=300,
        kfac_precond=kfac,
        smooth_lambda=1e-3,
        label="kfac+smooth (K-FAC + ‖∇²Δx‖² penalty)",
    )

    kfac.remove_hooks()
    return f_net, bf_net


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 55)
    print(" K-FAC + Gradient Conditioning for Backflow")
    print("=" * 55)

    C_occ, params = setup_noninteracting(6, 0.5, d=2, device=DEVICE, dtype=DTYPE)

    results = {}

    # Already completed from round 1 (no saved models, cache results only):
    results["kfac_bf_oldclip"] = (11.825228, 0.003025, 0.34)
    results["smooth_bf"] = (11.844429, 0.002916, 0.51)
    results["varmin_pretrain"] = (11.848661, 0.002936, 0.54)
    print("\n  [cached] kfac_bf (old joint clip, ‖∇f‖=0) → 0.34%")
    print("  [cached] smooth_bf                         → 0.51%")
    print("  [cached] varmin_pretrain                   → 0.54%")
    print("  [skip]   damped_sdinv — impractical (50× slower)")

    # New experiments to run:
    experiments = [
        ("kfac_sepclip", train_kfac_sepclip),
        ("kfac_smooth", train_kfac_smooth),
    ]

    for name, train_fn in experiments:
        torch.manual_seed(42)
        np.random.seed(42)
        print(f"\n{'═'*55}")
        print(f"# {name}")
        print(f"{'═'*55}")
        E_mean, E_std, err = run_experiment(name, train_fn, C_occ, params)
        results[name] = (E_mean, E_std, err)

    # ── Summary ──
    print("\n" + "═" * 55)
    print("SUMMARY — K-FAC & gradient conditioning")
    print("═" * 55)
    for name, (E, std, err) in results.items():
        print(f"  {name:25s}  E={E:.6f} ± {std:.6f}  err={err:.2f}%")
    print(f"  {'bf_0.7 joint (ref)':25s}  E=11.823253 ± 0.002982  err=0.33%")
    print(f"  {'PINN only (ref)':25s}  E=11.834691 ± 0.002782  err=0.42%")
