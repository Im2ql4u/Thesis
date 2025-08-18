import torch

from utils import inject_params
from .Slater_Determinant import slater_determinant_closed_shell
from .Physics import compute_coulomb_interaction


@inject_params
def psi_fn(f_net, x_batch, C_occ, backflow_net=None, spin=None, *, params=None):
    """
    ψ(x) = det(Slater(x)) * exp(f_net(x)).
    NOTE:
      - backflow/spin are ignored (kept in signature for compatibility).
      - no in-place ops; clone Slater output to avoid version-bump issues.
    """
    nx = int(params.get("nx", 0))
    ny = int(params.get("ny", 0))

    x_batch = x_batch  # no backflow: use coords as-is
    C_occ = C_occ.to(device=x_batch.device, dtype=x_batch.dtype)

    SD = slater_determinant_closed_shell(
        x_config=x_batch,
        C_occ=C_occ,
        n_basis_x=nx,
        n_basis_y=ny,
        params=params,
        normalize=True,
    )  # (B,1)
    SD = SD.clone()  # break potential alias to internal masked-fill buffers

    f = f_net(x_batch)
    if f.ndim == 1:
        f = f.unsqueeze(-1)
    f = torch.exp(torch.clamp(f, max=30))  # out-of-place

    psi = SD * f  # out-of-place
    return psi.squeeze(-1)  # (B,)


def compute_laplacian_fast(psi_fn, f_net, x, C_occ):
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

    Psi = psi_fn(f_net, x, C_occ)  # (B,)
    grads = torch.autograd.grad(Psi.sum(), x, create_graph=True, retain_graph=True)[
        0
    ]  # (B,N,d)

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
    backflow_net=None,
    spin=None,
    *,
    params=None,
    std=2.5,
    factor=0.1,
    print_e=50,
):
    """
    Training with FIXED E. No backflow/spin usage. Avoids in-place ops.
    Keeps your normalization step, and scales Δψ by the same norm to keep Hψ consistent.
    Prints energy estimate (mean local energy) and its variance every print_e steps.
    """
    device = params["device"]
    w = params["omega"]
    n_particles = int(params["n_particles"])
    n_epochs = int(params["n_epochs"])
    d = int(params["d"])
    N_collocation = int(params["N_collocation"])
    E = params["E"]  # fixed, per your request
    dtype = params["torch_dtype"]

    QHO_const = 0.5 * (w**2)
    f_net.to(device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        x = torch.normal(
            0, std, size=(N_collocation, n_particles, d), device=device, dtype=dtype
        ).clamp(min=-5, max=5)

        # Exact Laplacian (nested autograd); psi_fn ignores backflow/spin
        psi, laplacian = compute_laplacian_fast(
            lambda fnet, xb, C: psi_fn(fnet, xb, C, params=params), f_net, x, C_occ
        )  # (B,1), (B,1)

        # Normalize ψ and scale Δψ by the same factor (no in-place)
        norm = torch.linalg.norm(psi) + 1e-30
        psi = psi / norm
        laplacian = laplacian / norm

        # Potentials
        V_harmonic = QHO_const * (x**2).sum(dim=(1, 2)).view(-1, 1)
        V_int = compute_coulomb_interaction(x).view(-1, 1)
        V_total = V_harmonic + V_int

        # Hamiltonian application and residual with FIXED E
        H_psi = -0.5 * laplacian + V_total * psi
        residual = H_psi - E * psi

        # Diagnostics
        E_loc = H_psi / (psi + 1e-30)
        E_est = E_loc.mean().detach()
        variance = torch.var(E_loc.detach())

        # Loss
        loss_pde = torch.mean(residual**2)
        loss_norm = factor * (norm - 1) ** 2
        loss = loss_pde + loss_norm

        loss.backward()
        optimizer.step()

        if epoch % print_e == 0:
            print(
                f"Epoch {epoch:05d}: PDE={loss_pde.item():.3e}  Norm={norm.item():.3e}  "
                f"E_est={E_est.item():.6f}  Var(E_loc)={variance.item():.3e}"
            )

        # release graph refs
        del (
            loss,
            loss_norm,
            loss_pde,
            residual,
            H_psi,
            V_total,
            V_int,
            V_harmonic,
            E_loc,
            psi,
            laplacian,
            x,
        )

    return f_net
