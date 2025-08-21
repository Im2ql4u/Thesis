import torch

from utils import inject_params

from .Normalizing_Flow import sample_with_flow
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

    # Slater determinant at x_eff; basis dispatch is inside this call
    SD = slater_determinant_closed_shell(
        x_config=x_eff,
        C_occ=C_occ,
        params=params,
        normalize=True,
    )  # (B,1)

    # Jastrow/log-amplitude from f_net — clamp to keep exp stable early in training
    f = f_net(x_eff)
    if f.ndim == 1:
        f = f.unsqueeze(-1)
    f = torch.clamp(f, max=30)  # .exp_()  # in-place exp to avoid extra allocs
    f = torch.exp(f)

    # SD.mul_(f)  # in-place multiply, keeps grad
    psi = SD * f  # (B,1)
    return psi  # SD.squeeze(-1)  # (B,)


def compute_laplacian_fast2(psi_fn, f_net, x, C_occ, probes=4, **psi_kwargs):
    """
    Unbiased stochastic Laplacian via Hutchinson: E_v[v^T H v] with v∈{±1}^{B×N×d}.
    Returns (Psi, Δψ_est) both (B,1).
    """
    x = x.requires_grad_(True)
    Psi = psi_fn(f_net, x, C_occ, **psi_kwargs)  # (B,1)
    grad = torch.autograd.grad(Psi.sum(), x, create_graph=True)[0]  # (B,N,d)

    B, N, d = x.shape
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)

    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)  # Rademacher ±1
        Hv = torch.autograd.grad(grad, x, grad_outputs=v, create_graph=True)[0]  # (B,N,d)
        acc = acc + (v * Hv).sum(dim=(1, 2))  # vᵀH v

    lap = (acc / probes).unsqueeze(1)  # (B,1)
    return Psi, lap


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


@inject_params
def train_model(
    f_net,
    optimizer,
    C_occ,
    mapper,
    *,
    backflow_net=None,
    spin: torch.Tensor | None = None,
    params=None,
    std=2.5,
    factor=0.1,
    print_e=50,
):
    """
    Train the PINN by minimizing the PDE residual and enforcing normalization.

    Reads: omega, n_particles, n_epochs, d, N_collocation, E,
           device, (optional) torch_dtype from params.
    """
    device = params["device"]
    w = params["omega"]
    n_particles = params["n_particles"]
    n_epochs = params["n_epochs"]
    E = params["E"]
    QHO_const = 0.5 * w**2

    # Move nets
    f_net.to(device)
    if backflow_net is not None:
        backflow_net.to(device)

    # Prepare a default closed-shell spin pattern if not supplied
    if spin is None:
        up = n_particles // 2
        down = n_particles - up
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(down, dtype=torch.long)]
        ).to(device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # sample collocation points
        # x = torch.normal(
        #     0,
        #     std,
        #     size=(N_collocation, n_particles, d),
        #     device=device,
        #     dtype=dtype if dtype is not None else None,
        # ).clamp(min=-9, max=9)
        x = sample_with_flow(mapper) * std
        # ψ and ∇²ψ at x (+ backflow if provided)
        psi, laplacian = compute_laplacian_fast(
            psi_fn, f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
        )

        # normalize ψ
        norm = torch.norm(psi, p=2)
        psi = psi / norm
        laplacian = laplacian / norm  # scale Laplacian by the same factor
        # potentials
        V_harmonic = QHO_const * (x**2).sum(dim=(1, 2)).view(-1, 1)
        V_int = compute_coulomb_interaction(x)  # Coulomb from Physics.py
        V_total = V_harmonic + V_int

        # Hamiltonian on ψ and residual
        H_psi = -0.5 * laplacian + V_total * psi
        residual = H_psi - E * psi

        # losses
        loss_pde = torch.mean(residual**2)
        variance = torch.var(H_psi / psi)
        loss_norm = factor * (norm - 1) ** 2
        loss = loss_pde + loss_norm

        loss.backward()
        optimizer.step()

        if epoch % print_e == 0:
            # Try to report the *effective* scale if present
            bf_scale_str = ""
            if backflow_net is not None:
                try:
                    # Prefer a property that returns softplus(raw)
                    bf_val = backflow_net.bf_scale
                    if torch.is_tensor(bf_val):
                        bf_val = bf_val.item()
                    bf_scale_str = f"  bf_scale={bf_val:.3e}"
                except Exception:
                    bf_scale_str = ""
            print(
                f"Epoch {epoch:05d}: PDE={loss_pde.item():.3e}  "
                f"Norm={norm.item():.3e}  Var={variance.item():.3e}{bf_scale_str}"
            )

        # free graph memory each iter
        del (
            loss,
            variance,
            loss_pde,
            residual,
            H_psi,
            V_total,
            V_int,
            V_harmonic,
            psi,
            laplacian,
        )

    return f_net, backflow_net
