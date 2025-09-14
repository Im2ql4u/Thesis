# training_sr.py — Stable Stochastic Reconfiguration (SR)
# Most important fixes:
#   • HVP Hutchinson with FD fallback + nan guards
#   • Per-row filtering before building S and g
#   • Eval-mode MCMC sampling
#   • Diagonal Fisher preconditioner + trust-region scaled step
#   • Adaptive FD epsilon
#
# Expects:
#   - psi_fn(f_net, x, C_occ, backflow_net=None, spin=None, params=None) -> (logpsi:(B,), psi:(B,))
#   - compute_coulomb_interaction(x) -> (B,1) or (B,) [will be reshaped]
#
# Tip: Favor increasing batch_size over sampler_steps for SR accuracy.

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from utils import inject_params

# ============================================================
# Utilities: params vectorization
# ============================================================


def _gather_trainable_params(
    modules: list[nn.Module | None],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Flatten all trainable params across modules into a single vector."""
    params: list[torch.Tensor] = []
    for m in modules:
        if m is None:
            continue
        params.extend([p for p in m.parameters() if p.requires_grad])
    if params:
        flat = parameters_to_vector(params)
    else:
        # safe dummy
        dev = "cpu"
        if modules and hasattr(modules[0], "parameters"):
            try:
                dev = next(modules[0].parameters()).device  # type: ignore
            except StopIteration:
                pass
        flat = torch.tensor([], device=dev)
    return params, flat


# ============================================================
# Local energies (3 variants) — stable guards
# ============================================================


def _hvp_hutch_grad_laplogpsi(psi_log_fn, x: torch.Tensor, probes: int = 2, fd_eps: float = 1e-4):
    """
    Hutchinson estimate for Δ logψ using HVPs with FD fallback:
       grad_logpsi = ∇ logψ
       lap_logpsi  ≈ E_v [ vᵀ H_{logψ} v ], v ∈ {±1}^{B×N×d}
    Returns: grad_logpsi (B,N,d), lap_logpsi (B,1)
    """
    x = x.requires_grad_(True)
    logpsi = psi_log_fn(x)  # (B,)
    grad_logpsi = torch.autograd.grad(logpsi.sum(), x, create_graph=True, retain_graph=True)[0]

    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)

    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)

        # Preferred: H·v via vJP
        hv = torch.autograd.grad(
            grad_logpsi, x, grad_outputs=v, retain_graph=True, create_graph=True
        )[0]

        if not torch.isfinite(hv).all():
            # Fallback: central FD on ∇ logψ in direction v
            xp = (x + fd_eps * v).requires_grad_(True)
            xm = (x - fd_eps * v).requires_grad_(True)
            gp = torch.autograd.grad(
                psi_log_fn(xp).sum(), xp, retain_graph=True, create_graph=True
            )[0]
            gm = torch.autograd.grad(
                psi_log_fn(xm).sum(), xm, retain_graph=True, create_graph=True
            )[0]
            hv = (gp - gm) / (2.0 * fd_eps)

        hv = torch.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0)
        acc += (v * hv).sum(dim=(1, 2))

    lap_logpsi = (acc / max(1, probes)).unsqueeze(1)  # (B,1)
    return grad_logpsi, lap_logpsi


def _local_energy_fd(
    psi_log_fn,
    x: torch.Tensor,
    compute_coulomb_interaction,
    omega: float,
    probes: int = 2,
    eps: float = 1e-3,
):
    """
    E_L = -1/2 * (Δ logψ + ||∇ logψ||^2) + V(x)
    FD-Hutchinson for Δ logψ (first-order only). Adaptive ε is recommended.
    """
    x = x.requires_grad_(True)

    # ∇ logψ
    logpsi = psi_log_fn(x)  # (B,)
    g = torch.autograd.grad(logpsi.sum(), x, create_graph=True)[0]  # (B,N,d)
    g2 = (g**2).sum(dim=(1, 2))  # (B,)

    # Δ logψ via FD-Hutch
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)

        x_p = (x + eps * v).requires_grad_(True)
        gp = torch.autograd.grad(psi_log_fn(x_p).sum(), x_p, create_graph=True)[0]

        x_m = (x - eps * v).requires_grad_(True)
        gm = torch.autograd.grad(psi_log_fn(x_m).sum(), x_m, create_graph=True)[0]

        dir_p = (gp * v).sum(dim=(1, 2))
        dir_m = (gm * v).sum(dim=(1, 2))
        acc += (dir_p - dir_m) / (2.0 * eps)

    lap_log = acc / max(1, probes)  # (B,)

    # Potentials in physical coords
    V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
    V_int = compute_coulomb_interaction(x)
    V_int = V_int.view(-1) if V_int.ndim > 1 else V_int  # (B,)
    V = V_harm + V_int

    E_L = -0.5 * (lap_log + g2) + V  # (B,)
    return E_L.detach(), logpsi.detach()


def _local_energy_hvp(
    psi_log_fn,
    x: torch.Tensor,
    compute_coulomb_interaction,
    omega: float,
    probes: int = 2,
    fd_eps: float = 1e-4,
):
    """
    E_L via Hutchinson HVP on Δ logψ with FD fallback inside HVP helper.
      E_L = -1/2 * (Δ logψ + ||∇ logψ||^2) + V(x)
    """
    x = x.requires_grad_(True)
    logpsi = psi_log_fn(x)  # (B,)

    grad_logpsi, lap_logpsi = _hvp_hutch_grad_laplogpsi(
        psi_log_fn, x, probes=probes, fd_eps=fd_eps
    )  # grad: (B,N,d), lap: (B,1)
    g2 = (grad_logpsi**2).sum(dim=(1, 2))  # (B,)

    V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
    V_int = compute_coulomb_interaction(x)
    V_int = V_int.view(-1) if V_int.ndim > 1 else V_int
    V = V_harm + V_int

    E_L = -0.5 * (lap_logpsi.view(-1) + g2) + V  # (B,)
    return E_L.detach(), logpsi.detach()


def _local_energy_exact(
    psi_fn,
    f_net: nn.Module,
    C_occ: torch.Tensor,
    x: torch.Tensor,
    compute_coulomb_interaction,
    omega: float,
    *,
    backflow_net=None,
    spin=None,
    params=None,
):
    """
    Exact local energy using nested autograd on ψ:
      E_L = -1/2 * (Δψ / ψ) + V(x)
    """
    x = x.requires_grad_(True)
    logpsi, psi = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
    psi = psi.view(-1, 1)  # (B,1)

    grads = torch.autograd.grad(psi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)
    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            g_ij = grads[:, i, j]  # (B,)
            second = torch.autograd.grad(g_ij.sum(), x, create_graph=True, retain_graph=True)[0]
            lap += second[:, i, j]  # accumulate ∂²ψ/∂x_{i,j}²

    # Δψ/ψ with guards
    psi_safe = psi.clamp(min=1e-30)
    delta_over_psi = (lap / psi_safe.view(-1)).view(-1)  # (B,)

    V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
    V_int = compute_coulomb_interaction(x)
    V_int = V_int.view(-1) if V_int.ndim > 1 else V_int
    V = V_harm + V_int

    E_L = -0.5 * delta_over_psi + V  # (B,)
    E_L = torch.nan_to_num(E_L, nan=1e6, posinf=1e6, neginf=-1e6)
    return E_L.detach(), logpsi.detach()


# ============================================================
# Metropolis sampler targeting |Ψ|^2 ∝ exp(2 logψ) (eval-mode)
# ============================================================


@torch.no_grad()
def _metropolis_psi2(psi_log_fn, x0, n_steps: int = 30, step_sigma: float = 0.2):
    """
    Simple independent-proposal Metropolis for |Ψ|^2.
    Returns x with requires_grad=True.
    """
    x = x0.clone().requires_grad_(True)
    lp = psi_log_fn(x) * 2.0  # (B,)

    for _ in range(n_steps):
        prop = (x + torch.randn_like(x) * step_sigma).requires_grad_(True)
        lp_prop = psi_log_fn(prop) * 2.0
        accept = (torch.rand_like(lp_prop).log() < (lp_prop - lp)).view(-1, 1, 1)
        x = torch.where(accept, prop, x)
        lp = torch.where(accept.view(-1), lp_prop, lp)
    return x


# ============================================================
# Per-sample score matrix S = ∂logψ/∂θ (flattened per sample)
# ============================================================


def _compute_score_matrix(psi_log_fn, x: torch.Tensor, modules: list[nn.Module | None]):
    """
    Returns:
      score_mat : (B, P)
      params_list : list of parameters in update order
    Robust to unused params; zero-fills Nones.
    """
    params_list, flat = _gather_trainable_params(modules)
    P = flat.numel()
    B = x.shape[0]
    if P == 0:
        return torch.zeros(B, 0, device=x.device, dtype=x.dtype), params_list

    score_mat = torch.zeros(B, P, device=flat.device, dtype=flat.dtype)

    for i in range(B):
        xi = x[i : i + 1].requires_grad_(True)
        logpsi_i = psi_log_fn(xi)  # (1,)
        # skip non-finite rows gracefully
        if not torch.isfinite(logpsi_i).all():
            continue
        grads = torch.autograd.grad(logpsi_i, params_list, retain_graph=False, allow_unused=True)
        grads_filled = [
            (g if g is not None else torch.zeros_like(p))
            for g, p in zip(grads, params_list, strict=False)
        ]
        score_mat[i].copy_(parameters_to_vector(grads_filled))
    return score_mat, params_list


# ============================================================
# Conjugate Gradient solver for (A + λ I) x = b
# ============================================================


def _cg(matvec, b, lam: float = 1e-3, tol: float = 1e-6, max_iter: int = 100):
    """
    Solve (A + λ I)x = b given implicit matvec A(v). Returns (x, iters).
    """
    x = torch.zeros_like(b)
    r = b - (matvec(x) + lam * x)
    p = r.clone()
    rs = r @ r
    k = 0
    for _k in range(1, max_iter + 1):
        k = _k  # track last iteration number
        Ap = matvec(p) + lam * p
        alpha = rs / (p @ Ap + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / (rs + 1e-20)) * p
        rs = rs_new
    return x, k



# ============================================================
# One SR step (energy-based) with selectable Laplacian
# ============================================================


def sr_step_energy(
    f_net,
    C_occ,
    *,
    psi_fn,
    backflow_net=None,
    spin=None,
    params=None,
    compute_coulomb_interaction=None,
    batch_size=1024,
    sampler_steps=30,
    sampler_step_sigma=0.2,
    fd_probes=8,
    fd_eps_scale=1e-3,
    damping=1e-2,
    cg_tol=1e-6,
    cg_iters=150,
    step_size=0.04,
    center_O=True,
    lap_mode="hvp-hutch",
    restart_every=40,
    drop_frac=0.02,  # drop lowest-variance *elements* of flat param vector
    max_damping=5e-1,
):
    assert compute_coulomb_interaction is not None
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float32)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    # ---- spin / device ----
    if spin is None:
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)
    f_net.to(device).to(dtype).train()
    if backflow_net is not None:
        backflow_net.to(device).to(dtype).train()

    # ---- logψ closure ----
    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi

    # ---- sample |ψ|^2 (eval nets for MCMC, then restore train) ----
    x0 = torch.randn(batch_size, N, d, device=device, dtype=dtype)
    was_train = f_net.training
    f_net.eval()
    back_state = backflow_net.training if backflow_net is not None else False
    if backflow_net is not None:
        backflow_net.eval()
    x = _metropolis_psi2(psi_log_fn, x0, n_steps=sampler_steps, step_sigma=sampler_step_sigma)
    if was_train:
        f_net.train()
    if backflow_net is not None and back_state:
        backflow_net.train()

    # ---- local energy ----
    with torch.no_grad():
        x_scale = torch.quantile(x.abs().reshape(x.shape[0], -1), 0.5).item() + 1e-6
        eps_adapt = fd_eps_scale * x_scale
    if lap_mode == "hvp-hutch":
        E_L, _ = _local_energy_hvp(
            psi_log_fn, x, compute_coulomb_interaction, omega, probes=fd_probes
        )
    elif lap_mode == "fd-hutch":
        E_L, _ = _local_energy_fd(
            psi_log_fn, x, compute_coulomb_interaction, omega, probes=fd_probes, eps=eps_adapt
        )
    elif lap_mode == "exact":
        E_L, _ = _local_energy_exact(
            psi_fn,
            f_net,
            C_occ,
            x,
            compute_coulomb_interaction,
            omega,
            backflow_net=backflow_net,
            spin=spin,
            params=params,
        )
    else:
        raise ValueError(lap_mode)

    # filter any NaN/inf rows
    good = torch.isfinite(E_L)
    if good.sum() < E_L.numel():
        x, E_L = x[good], E_L[good]
        if x.numel() == 0:
            return {
                "E_mean": float("nan"),
                "E_std": float("nan"),
                "g_norm": 0.0,
                "step_norm": 0.0,
                "filtered": int(good.numel()),
                "cg_iters": 0,
            }

    # ---- score matrix in full flat space ----
    modules = [f_net, backflow_net] if backflow_net is not None else [f_net]
    O_full, params_list = _compute_score_matrix(psi_log_fn, x, modules)  # (B, P_full)
    B, P_full = O_full.shape
    mu_E = E_L.mean()

    # center rows if requested
    if center_O:
        O_full = O_full - O_full.mean(dim=0)

    # ---- elementwise variance for whitening & dropping (still full space) ----
    var_full = (O_full**2).mean(dim=0) + 1e-12

    # choose kept indices in flat space
    if drop_frac > 0.0 and P_full > 10:
        k_drop = int(drop_frac * P_full)
        if k_drop > 0:
            # threshold at kth smallest variance
            thresh = torch.kthvalue(var_full, k_drop).values
            keep_mask = var_full > thresh
        else:
            keep_mask = torch.ones(P_full, dtype=torch.bool, device=device)
    else:
        keep_mask = torch.ones(P_full, dtype=torch.bool, device=device)

    keep_idx = keep_mask.nonzero(as_tuple=False).flatten()  # (P_kept,)
    O = O_full[:, keep_idx]  # (B, P_kept)
    var = var_full[keep_idx]  # (P_kept,)

    # ---- whitening (Jacobi) ----
    D_inv_sqrt = var.rsqrt()
    Ow = O * D_inv_sqrt

    # SR gradient in whitened coords
    g = 2.0 * ((Ow * (E_L - mu_E).view(-1, 1)).mean(dim=0))  # (P_kept,)

    # implicit matvec A v = (Ow^T Ow) v / B
    def A_mv(v):
        return (Ow.t() @ (Ow @ v)) / B

    # ---- preconditioned CG with relative tolerance + restarts ----
    lam = float(damping)
    b = -g
    xk = torch.zeros_like(b)
    r = b - (A_mv(xk) + lam * xk)
    r0 = r.norm()
    rel_floor = max(cg_tol, 3.0 / (B**0.5))  # noise floor
    p = r.clone()
    its = 0
    while its < cg_iters:
        Ap = A_mv(p) + lam * p
        alpha = (r @ r) / (p @ Ap + 1e-20)
        xk = xk + alpha * p
        r_new = r - alpha * Ap
        its += 1
        if r_new.norm() <= rel_floor * r0:
            r = r_new
            break
        if (its % restart_every) == 0:
            r = r_new
            p = r.clone()
        else:
            beta = (r_new @ r_new) / (r @ r + 1e-20)
            p = r_new + beta * p
            r = r_new

    # ---- unwhiten and SCATTER back to full flat vector ----
    delta_kept = D_inv_sqrt * xk  # (P_kept,)
    delta_full = torch.zeros(P_full, device=device, dtype=dtype)
    delta_full[keep_idx] = delta_kept  # expand to full size

    # ---- trust region + adaptive damping in full space ----
    def S_mv_full(v):  # uses *full* scores
        tmp = O_full @ v
        return (O_full.t() @ tmp) / B

    Sd = S_mv_full(delta_full)
    quad = (delta_full @ Sd) + lam * (delta_full @ delta_full)
    if not torch.isfinite(quad) or quad <= 1e-12:
        lam = min(max_damping, max(1e-2, 10 * lam))
        quad = (delta_full @ S_mv_full(delta_full)) + lam * (delta_full @ delta_full) + 1e-12
    scale = step_size / torch.sqrt(quad)

    with torch.no_grad():
        theta_flat = parameters_to_vector(params_list)
        theta_new = theta_flat + scale * delta_full
        # simple safeguard: if predicted decrease is positive, shrink & bump λ
        model_red = 0.5 * scale**2 * quad + scale * (
            delta_full @ (2.0 * ((O_full * (E_L - mu_E).view(-1, 1)).mean(dim=0)))
        )
        if model_red > 0:
            lam = min(max_damping, max(2 * lam, 1e-2))
            scale = 0.5 * scale
            theta_new = theta_flat + scale * delta_full
        vector_to_parameters(theta_new, params_list)

    return {
        "E_mean": float(mu_E.item()),
        "E_std": float(E_L.std().item()),
        "g_norm": float(g.norm().item()),
        "step_norm": float((scale * delta_kept).norm().item()),  # step measured in kept subspace
        "filtered": int((~good).sum().item()),
        "cg_iters": int(its),
        "kept_params": int(keep_idx.numel()),
        "damping": float(lam),
    }


# ============================================================
# Full SR trainer (energy-based)
# ============================================================


@inject_params
def train_model_sr_energy(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    *,
    psi_fn,
    backflow_net: nn.Module | None = None,
    spin=None,
    params=None,
    compute_coulomb_interaction=None,
    n_sr_steps: int = 200,
    batch_size: int = 1024,
    sampler_steps: int = 30,
    sampler_step_sigma: float = 0.2,
    fd_probes: int = 2,
    fd_eps_scale: float = 1e-3,
    damping: float = 1e-3,
    cg_tol: float = 1e-6,
    cg_iters: int = 100,
    step_size: float = 0.05,
    center_O: bool = True,
    log_every: int = 10,
    lap_mode: str = "fd-hutch",
):
    """
    Standalone SR trainer (energy-based). Returns: f_net, backflow_net.
    lap_mode: "fd-hutch" | "hvp-hutch" | "exact"
    """
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float32)

    f_net.to(device).to(dtype)
    if backflow_net is not None:
        backflow_net.to(device).to(dtype)

    for t in range(n_sr_steps):
        info = sr_step_energy(
            f_net,
            C_occ,
            psi_fn=psi_fn,
            backflow_net=backflow_net,
            spin=spin,
            params=params,
            compute_coulomb_interaction=compute_coulomb_interaction,
            batch_size=batch_size,
            sampler_steps=sampler_steps,
            sampler_step_sigma=sampler_step_sigma,
            fd_probes=fd_probes,
            fd_eps_scale=fd_eps_scale,
            damping=damping,
            cg_tol=cg_tol,
            cg_iters=cg_iters,
            step_size=step_size,
            center_O=center_O,
            lap_mode=lap_mode,
        )
        if (t % log_every) == 0:
            print(
                f"[SR {t:04d}]  E={info['E_mean']:.8f}  σ(E)={info['E_std']:.6f}  "
                f"‖g‖={info['g_norm']:.3e}  ‖Δθ‖={info['step_norm']:.3e}  "
                f"filtered={info['filtered']}  cg={info['cg_iters']}  "
                f"[lap={lap_mode.replace('fd-hutch','fd').replace('hvp-hutch','hvp')}]"
            )

    return f_net, backflow_net
