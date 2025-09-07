# training_sr.py
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from utils import inject_params

# ============================================================
# Utilities: params vectorization
# ============================================================


def _gather_trainable_params(modules):
    """Flatten all trainable params across a list of modules into a single vector."""
    params = []
    for m in modules:
        if m is None:
            continue
        params.extend([p for p in m.parameters() if p.requires_grad])
    flat = (
        parameters_to_vector(params)
        if params
        else torch.tensor(
            [],
            device=(
                modules[0].parameters().__iter__().__next__().device
                if modules and hasattr(modules[0], "parameters")
                else "cpu"
            ),
        )
    )
    return params, flat


# ============================================================
# Local energies (3 variants)
# ============================================================


def _local_energy_fd(
    psi_log_fn,
    x,
    compute_coulomb_interaction,
    omega: float,
    probes: int = 2,
    eps: float = 1e-3,
):
    """
    E_L = -1/2 * (Δ logψ + ||∇ logψ||^2) + V(x)
    FD-Hutchinson for Δ logψ (first-order only).
    Returns: E_L (B,), logψ (B,)
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
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)  # Rademacher ±1

        x_p = (x + eps * v).requires_grad_(True)
        lp = psi_log_fn(x_p)
        gp = torch.autograd.grad(lp.sum(), x_p, create_graph=True)[0]

        x_m = (x - eps * v).requires_grad_(True)
        lm = psi_log_fn(x_m)
        gm = torch.autograd.grad(lm.sum(), x_m, create_graph=True)[0]

        acc += ((gp * v).sum(dim=(1, 2)) - (gm * v).sum(dim=(1, 2))) / (2.0 * eps)

    lap_log = acc / probes  # (B,)

    # V(x) at physical coords
    V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
    V_int = compute_coulomb_interaction(x).view(-1)  # (B,)
    V = V_harm + V_int

    E_L = -0.5 * (lap_log + g2) + V  # (B,)
    return E_L.detach(), logpsi.detach()


def _hvp_hutch_grad_laplogpsi(logpsi_scalar, x, probes: int = 2):
    """
    Hutchinson estimate for Δ logψ using HVPs:
      lap_logpsi ≈ E_v [ v^T H_{logψ} v ], v ∈ {±1}^{B×N×d}.
    Returns: grad_logpsi (B,N,d), lap_logpsi (B,1)
    """
    grad_logpsi = torch.autograd.grad(logpsi_scalar, x, create_graph=True, retain_graph=True)[0]

    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        hv = torch.autograd.grad((grad_logpsi * v).sum(), x, create_graph=True, retain_graph=True)[
            0
        ]
        acc += (v * hv).sum(dim=(1, 2))  # v^T H v
    lap_logpsi = (acc / probes).unsqueeze(1)  # (B,1)
    return grad_logpsi, lap_logpsi


def _local_energy_hvp(
    psi_log_fn,
    x,
    compute_coulomb_interaction,
    omega: float,
    probes: int = 2,
):
    """
    E_L via Hutchinson HVP on Δ logψ:
      E_L = -1/2 * (Δ logψ + ||∇ logψ||^2) + V(x)
    Returns: E_L (B,), logψ (B,)
    """
    x = x.requires_grad_(True)
    logpsi = psi_log_fn(x)  # (B,)

    grad_logpsi, lap_logpsi = _hvp_hutch_grad_laplogpsi(
        logpsi.sum(), x, probes=probes
    )  # grad: (B,N,d), lap: (B,1)
    g2 = (grad_logpsi**2).sum(dim=(1, 2))  # (B,)

    V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
    V_int = compute_coulomb_interaction(x).view(-1)  # (B,)
    V = V_harm + V_int

    E_L = -0.5 * (lap_logpsi.view(-1) + g2) + V  # (B,)
    return E_L.detach(), logpsi.detach()


def _local_energy_exact(
    psi_fn,
    f_net,
    C_occ,
    x,
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
    Returns: E_L (B,), logψ (B,)
    """
    x = x.requires_grad_(True)
    logpsi, psi = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
    psi = psi.view(-1, 1)  # (B,1)

    # Δψ via nested autograd (exact)
    grads = torch.autograd.grad(psi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)
    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            g_ij = grads[:, i, j]  # (B,)
            second = torch.autograd.grad(g_ij.sum(), x, create_graph=True, retain_graph=True)[0]
            lap += second[:, i, j]  # accumulate ∂²ψ/∂x_{i,j}²

    # Safe division Δψ / ψ
    psi_safe = psi.clamp(min=1e-12)
    delta_over_psi = (lap / psi_safe.view(-1)).view(-1)  # (B,)

    V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
    V_int = compute_coulomb_interaction(x).view(-1)  # (B,)
    V = V_harm + V_int

    E_L = -0.5 * delta_over_psi + V  # (B,)
    return E_L.detach(), logpsi.detach()


# ============================================================
# Metropolis sampler targeting |Ψ|^2 ∝ exp(2 logψ)
# ============================================================


@torch.no_grad()
def _metropolis_psi2(psi_log_fn, x0, n_steps: int = 30, step_sigma: float = 0.2):
    """
    Simple independent-proposal Metropolis for |Ψ|^2.
    Ensures x passed to psi_fn has requires_grad=True (to satisfy your assert),
    while preventing graph creation via torch.no_grad().
    """
    # start state
    x = x0.clone().requires_grad_(True)

    with torch.no_grad():
        lp = psi_log_fn(x) * 2.0  # (B,)

        for _ in range(n_steps):
            prop = (x + torch.randn_like(x) * step_sigma).requires_grad_(True)
            lp_prop = psi_log_fn(prop) * 2.0

            accept_logprob = lp_prop - lp
            accept = (torch.rand_like(accept_logprob).log() < accept_logprob).view(-1, 1, 1).float()

            x = accept * prop + (1.0 - accept) * x
            lp = accept.view(-1) * lp_prop + (1.0 - accept.view(-1)) * lp

    # Return with requires_grad=True so later code can build graphs if needed
    return x


# ============================================================
# Per-sample score matrix S = ∂logψ/∂θ (flattened per sample)
# ============================================================


def _compute_score_matrix(psi_log_fn, x, modules):
    """
    Compute the per-sample parameter score matrix:
      score[b, j] = ∂ logψ(x_b) / ∂ θ_j
    Robust to unused parameters: allow_unused=True and zero-fill Nones.

    Returns:
      score_mat : (B, P)  (P = total #trainable params)
      params_list : list of tensors in update order
    """
    params_list, flat = _gather_trainable_params(modules)
    P = flat.numel()
    B = x.shape[0]

    # If nothing is trainable (rare), return an empty matrix gracefully
    if P == 0:
        return torch.zeros(B, 0, device=x.device, dtype=x.dtype), params_list

    score_mat = torch.zeros(B, P, device=flat.device, dtype=flat.dtype)

    for i in range(B):
        xi = x[i : i + 1].requires_grad_(True)
        logpsi_i = psi_log_fn(xi)  # (1,)
        grads = torch.autograd.grad(
            logpsi_i, params_list, retain_graph=False, allow_unused=True  # <-- important
        )
        # zero-fill any None grads to keep shapes consistent
        grads_filled = [
            (g if g is not None else torch.zeros_like(p))
            for g, p in zip(grads, params_list, strict=False)
        ]
        score_mat[i].copy_(parameters_to_vector(grads_filled))
    return score_mat, params_list


# ============================================================
# Conjugate Gradient solver for (S + λ I) Δ = -g
# ============================================================


def _cg(matvec, b, lam: float = 1e-3, tol: float = 1e-6, max_iter: int = 100):
    """
    Solve (A + λ I)x = b given implicit matvec A(v).
    """
    x = torch.zeros_like(b)
    r = b - (matvec(x) + lam * x)
    p = r.clone()
    rs = r @ r
    for _ in range(max_iter):
        Ap = matvec(p) + lam * p
        alpha = rs / (p @ Ap + 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / (rs + 1e-20)) * p
        rs = rs_new
    return x


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
    fd_probes=2,
    fd_eps_scale=1e-3,
    damping=1e-3,
    cg_tol=1e-6,
    cg_iters=100,
    step_size=0.05,
    center_O=True,  # kept for backwards-compat; refers to center_scores
    lap_mode: str = "fd-hutch",  # "fd-hutch" | "hvp-hutch" | "exact"
):
    assert compute_coulomb_interaction is not None, "Need compute_coulomb_interaction."

    device = params["device"]
    dtype = params.get("torch_dtype", torch.float32)
    omega = float(params["omega"])
    n_particles = int(params["n_particles"])
    d = int(params["d"])

    f_net.to(device).to(dtype).train()
    if backflow_net is not None:
        backflow_net.to(device).to(dtype).train()

    if spin is None:
        up = n_particles // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(n_particles - up, dtype=torch.long)]
        ).to(device)
    else:
        spin = spin.to(device)

    # --- logψ wrapper (ensures requires_grad=True on x) ---
    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _psi = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi  # (B,)

    # --------- Sample x ~ |Ψ|^2 -----------
    x0 = torch.randn(batch_size, n_particles, d, device=device, dtype=dtype)
    x = _metropolis_psi2(psi_log_fn, x0, n_steps=sampler_steps, step_sigma=sampler_step_sigma)

    # --------- Local energy on samples ----
    if lap_mode == "fd-hutch":
        E_L, _logpsi = _local_energy_fd(
            psi_log_fn, x, compute_coulomb_interaction, omega, probes=fd_probes, eps=fd_eps_scale
        )
    elif lap_mode == "hvp-hutch":
        E_L, _logpsi = _local_energy_hvp(
            psi_log_fn, x, compute_coulomb_interaction, omega, probes=fd_probes
        )
    elif lap_mode == "exact":
        E_L, _logpsi = _local_energy_exact(
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
        raise ValueError(f"Unknown lap_mode={lap_mode!r}")

    # --------- Per-sample scores S = ∂logψ/∂θ ----
    modules = [f_net, backflow_net] if backflow_net is not None else [f_net]
    score_mat, params_list = _compute_score_matrix(psi_log_fn, x, modules)  # (B,P)
    B, P = score_mat.shape

    # --------- Build SR system ------------
    score_mean = score_mat.mean(dim=0) if center_O else torch.zeros(P, device=device, dtype=dtype)
    score_centered = score_mat - score_mean
    mu_E = E_L.mean()
    g_vec = 2.0 * ((score_centered * (E_L - mu_E).view(-1, 1)).mean(dim=0))  # (P,)

    def S_matvec(v):
        tmp = score_centered @ v
        return (score_centered.t() @ tmp) / B

    delta = _cg(S_matvec, -g_vec, lam=damping, tol=cg_tol, max_iter=cg_iters)

    with torch.no_grad():
        theta = parameters_to_vector(params_list)
        theta.add_(step_size * delta)
        vector_to_parameters(theta, params_list)

    return {
        "E_mean": float(mu_E.item()),
        "E_std": float(E_L.std().item()),
        "g_norm": float(g_vec.norm().item()),
        "step_norm": float(delta.norm().item()),
    }


# ============================================================
# Full SR trainer (separate from your residual trainer)
# ============================================================


@inject_params
def train_model_sr_energy(
    f_net,
    C_occ,
    *,
    psi_fn,  # your existing psi_fn
    backflow_net=None,
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
    center_O: bool = True,  # kept for compatibility
    log_every: int = 10,
    lap_mode: str = "fd-hutch",  # NEW
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
            lap_mode=lap_mode,  # pass through
        )
        if (t % log_every) == 0:
            print(
                f"[SR {t:04d}]  E={info['E_mean']:.8f}  σ(E)={info['E_std']:.6f}  "
                f"‖g‖={info['g_norm']:.3e}  ‖Δθ‖={info['step_norm']:.3e}  "
                f"[lap={lap_mode.replace('fd-hutch','fd').replace('hvp-hutch','hvp')}]"
            )

    return f_net, backflow_net
