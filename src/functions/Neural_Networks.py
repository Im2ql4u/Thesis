from typing import Literal

import torch

from utils import inject_params

from .Physics import compute_coulomb_interaction
from .Slater_Determinant import slater_determinant_closed_shell


@inject_params
def psi_fn(
    f_net,
    x_batch: torch.Tensor,
    C_occ: torch.Tensor,
    *,
    backflow_net=None,
    spin: torch.Tensor | None = None,
    params=None,
):
    """
    ψ(x) = det(Slater(x+Δx; C_occ)) * exp(f_net(x+Δx)) with optional backflow Δx.
    Basis selection is handled inside slater_determinant_closed_shell via params['basis'].
    """
    # Device / dtype hygiene
    x_batch = x_batch.contiguous()
    C_occ = C_occ.to(device=x_batch.device, dtype=x_batch.dtype).contiguous()

    # Optional backflow: Δx = backflow_net(x, spin)
    if backflow_net is not None:
        # spin can be (N,) or (B,N); both are supported by the provided BackflowNet
        dx = backflow_net(x_batch, spin=spin)
        x_eff = x_batch + dx
    else:
        x_eff = x_batch
    assert x_eff.requires_grad, "x_eff must have requires_grad=True for autograd"
    # Slater determinant at x_eff; basis dispatch is inside this call
    sign, logabs = slater_determinant_closed_shell(
        x_config=x_eff,
        C_occ=C_occ,
        params=params,
        normalize=True,
    )  # (B,1)

    # Jastrow/log-amplitude from f_net — clamp to keep exp stable early in training
    f = f_net(x_eff).squeeze(-1)
    logpsi = logabs + f  # (B,)
    psi = sign * torch.exp(logpsi)  # (B,)
    return logpsi, psi  # SD.squeeze(-1)  # (B,)


def grad_and_laplace_logpsi(logpsi_scalar, x, probes=4):
    """
    logpsi_scalar: scalar = logpsi.sum()
    returns: grad_logpsi (B,N,d), lap_logpsi (B,1)
    """
    # First gradient of logpsi
    grad_logpsi = torch.autograd.grad(logpsi_scalar, x, create_graph=True, retain_graph=True)[
        0
    ]  # (B,N,d)

    # Hutchinson trace estimation for Δ logψ = tr(H_{logψ})
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)  # Rademacher ±1
        # HVP: (H_{logψ} v) = ∂/∂x [ <grad_logpsi, v> ]
        hv = torch.autograd.grad((grad_logpsi * v).sum(), x, create_graph=True, retain_graph=True)[
            0
        ]
        acc += (v * hv).sum(dim=(1, 2))  # v^T H v
    lap_logpsi = (acc / probes).unsqueeze(1)  # (B,1)

    return grad_logpsi, lap_logpsi


def compute_laplacian_fast(psi_fn, f_net, x, C_occ, **psi_kwargs):
    """
    Exact Laplacian via nested autograd (no torch.func), avoiding in-place ops.

    Args:
      psi_fn : callable(f_net, x, C_occ) -> (B,)
      f_net  : NN mapping x -> log factor
      x      : (B, N, d) with requires_grad=True (will be set here)
      C_occ  : (n_basis, n_occ)

    Returns:
      Psi        : (B,1)
      Laplacian  : (B,1)  (Δψ)
    """
    x = x.requires_grad_(True)
    B, N, d = x.shape

    Psi = psi_fn(f_net, x, C_occ, **psi_kwargs)  # (B,)
    grads = torch.autograd.grad(Psi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)

    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    # accumulate ∂²ψ/∂x_{i,j}²
    for i in range(N):
        for j in range(d):
            g_ij = grads[:, i, j]  # (B,)
            # sum over batch to get a scalar, then differentiate wrt x and pick the same coord
            gsum = g_ij.sum()
            second = torch.autograd.grad(gsum, x, create_graph=True, retain_graph=True)[
                0
            ]  # (B,N,d)
            lap = lap + second[:, i, j]

    return Psi.unsqueeze(1), lap.unsqueeze(1)  # (B,1), (B,1)


# --- Fast Δ log ψ via finite-difference Hutchinson (first-order only) ---
def _laplacian_logpsi_fd(psi_log_fn, x_eff, eps, probes=2):
    """
    psi_log_fn: closure (x_eff: (B,N,d) with requires_grad=True) -> logpsi (B,)
    x_eff      : (B,N,d) requires_grad=True
    eps        : finite-difference step (float)
    probes     : # of Hutchinson probes (int)

    Returns:
      grad_logpsi : (B,N,d)
      g2          : (B,1)   = ||∇ log ψ||^2
      lap_logpsi  : (B,1)   = Δ log ψ (FD-Hutch estimate)
    """
    # center gradient (for ||∇ log ψ||^2 term)
    logpsi = psi_log_fn(x_eff)  # (B,)
    grad_logpsi = torch.autograd.grad(logpsi.sum(), x_eff, create_graph=True)[0]  # (B,N,d)
    g2 = (grad_logpsi**2).sum(dim=(1, 2), keepdim=True)  # (B,1)

    B = x_eff.shape[0]
    acc = torch.zeros(B, device=x_eff.device, dtype=x_eff.dtype)

    for _ in range(probes):
        v = torch.empty_like(x_eff).bernoulli_(0.5).mul_(2).add_(-1)  # Rademacher ±1

        x_plus = (x_eff + eps * v).requires_grad_(True)
        lp_plus = psi_log_fn(x_plus)
        g_plus = torch.autograd.grad(lp_plus.sum(), x_plus, create_graph=True)[0]

        x_minus = (x_eff - eps * v).requires_grad_(True)
        lp_minus = psi_log_fn(x_minus)
        g_minus = torch.autograd.grad(lp_minus.sum(), x_minus, create_graph=True)[0]

        # directional second derivative estimate along v
        dir_plus = (g_plus * v).sum(dim=(1, 2))
        dir_minus = (g_minus * v).sum(dim=(1, 2))
        acc += (dir_plus - dir_minus) / (2.0 * eps)

    lap_logpsi = (acc / probes).unsqueeze(1)  # (B,1)
    return grad_logpsi, g2, lap_logpsi


@inject_params
def train_model(
    f_net,
    optimizer,
    C_occ,
    mapper=None,  # kept for interface parity
    *,
    backflow_net=None,
    spin: torch.Tensor | None = None,
    params=None,
    std: float = 2.5,
    norm_penalty: float = 1e-5,
    probes: int = 2,
    eps_scale: float = 1e-3,
    print_e: int = 50,
    lap_mode: Literal["fd-hutch", "hvp-hutch", "exact"] = "fd-hutch",
):
    """
    Residual-based PINN training with selectable Laplacian:
      - 'fd-hutch'  : FD Hutchinson on Δ log ψ (1st-order only)
      - 'hvp-hutch' : Hutchinson via HVPs on log ψ (2nd-order graph, no coord loops)
      - 'exact'     : nested-autograd exact Δψ (slowest)
    Assumes psi_fn(f_net, x, C_occ, ...) -> (logpsi, psi).
    """
    device = params["device"]
    w = params["omega"]
    n_particles = params["n_particles"]
    n_epochs = params["n_epochs"]
    E_target = params["E"]
    N_collocation = params["N_collocation"]
    d = params["d"]
    dtype = params.get("torch_dtype", None)
    QHO_const = 0.5 * (w**2)

    f_net.to(device)
    if backflow_net is not None:
        backflow_net.to(device)

    if spin is None:
        up = n_particles // 2
        down = n_particles - up
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(down, dtype=torch.long)]
        ).to(device)

    # ψ-only wrapper for exact Laplacian helper
    def psi_only(_f, _x, _C, **kw):
        _logpsi, _psi = psi_fn(_f, _x, _C, **kw)
        return _psi.view(-1)

    # log ψ closure for Hutchinson variants
    def psi_log_fn(y):
        logpsi_y, _psi_y = psi_fn(
            f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params
        )
        return logpsi_y  # (B,)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Sample collocation points (physical coords)
        x_kwargs = dict(device=device)
        if dtype is not None:
            x_kwargs["dtype"] = dtype
        x = (
            torch.normal(0, std, size=(N_collocation, n_particles, d), **x_kwargs)
            .clamp(min=-9, max=9)
            .requires_grad_(True)
        )

        # Choose Laplacian path
        if lap_mode == "exact":
            # Exact Δψ via nested autograd on ψ
            Psi, Delta_psi = compute_laplacian_fast(
                psi_only, f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )  # (B,1), (B,1)
            psi = Psi

        elif lap_mode == "hvp-hutch":
            # Hutchinson on Δ log ψ using HVPs
            logpsi, psi = psi_fn(
                f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )
            psi = psi.view(-1, 1)
            grad_logpsi, lap_logpsi = grad_and_laplace_logpsi(
                logpsi.sum(), x, probes=probes
            )  # grad: (B,N,d), lap_logpsi: (B,1)
            g2 = (grad_logpsi**2).sum(dim=(1, 2), keepdim=True)  # (B,1)
            Delta_psi = psi * (lap_logpsi + g2)

        elif lap_mode == "fd-hutch":
            # Finite-difference Hutchinson on Δ log ψ (first-order only)
            logpsi, psi = psi_fn(
                f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )
            psi = psi.view(-1, 1)
            eps = eps_scale * float(std)
            grad_logpsi, g2, lap_logpsi = _laplacian_logpsi_fd(
                psi_log_fn, x, eps=eps, probes=probes
            )
            Delta_psi = psi * (lap_logpsi + g2)

        else:
            raise ValueError(f"Unknown lap_mode={lap_mode!r}")

        # Potentials at physical coordinates x
        V_harmonic = QHO_const * (x**2).sum(dim=(1, 2), keepdim=True)  # (B,1)
        V_int = compute_coulomb_interaction(x)
        if V_int.dim() != 2:
            V_int = V_int.view(-1, 1)
        V_total = V_harmonic + V_int  # (B,1)

        # Hamiltonian and residual
        H_psi = -0.5 * Delta_psi + V_total * psi  # (B,1)
        residual = H_psi - E_target * psi  # (B,1)

        # Loss
        loss_pde = (residual**2).mean()
        norm = torch.linalg.vector_norm(psi)
        loss_norm = norm_penalty * (norm - 1.0) ** 2
        loss = loss_pde + loss_norm
        loss.backward()
        optimizer.step()

        if epoch % print_e == 0:
            with torch.no_grad():
                local_E = (H_psi / psi).clamp(min=-1e6, max=1e6)
                var_E = torch.var(local_E)
            bf_scale_str = ""
            if backflow_net is not None and hasattr(backflow_net, "bf_scale"):
                try:
                    bf_val = backflow_net.bf_scale
                    bf_val = bf_val.item() if torch.is_tensor(bf_val) else float(bf_val)
                    bf_scale_str = f"  bf_scale={bf_val:.3e}"
                except Exception:
                    pass
            print(
                f"Epoch {epoch:05d}: PDE={loss_pde.item():.3e}  "
                f"Norm={norm.item():.3e}  Var(EL)={var_E.item():.3e}  "
                f"[lap={lap_mode.replace('fd-hutch','fd').replace('hvp-hutch','hvp')}]"
                + bf_scale_str
            )

        # free big tensors
        del (loss, loss_pde, loss_norm, residual, H_psi, V_total, V_int, V_harmonic, psi, Delta_psi)

    return f_net, backflow_net
