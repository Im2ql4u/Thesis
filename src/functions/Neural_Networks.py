import torch

from utils import inject_params
from .Slater_Determinant import slater_determinant_closed_shell
from .Physics import compute_coulomb_interaction


@inject_params
def psi_fn(f_net, x_batch, C_occ, *, params=None):
    """
    ψ(x) = det(Slater) * exp(f_net). Works for both bases:
      - FD: Slater uses params['emax']; nx,ny are ignored.
      - Cartesian: Slater uses (nx, ny).
    Accepts NN outputs shaped (B,1), (B,), (B,N,1), or (B,N).
    """
    nx, ny = params.get("nx", 1), params.get("ny", 1)
    SD = slater_determinant_closed_shell(x_batch, C_occ, nx, ny)  # (B,1)

    f = f_net(x_batch)
    if f.dim() == 3:  # (B,N,1) or (B,N)
        f = f.sum(dim=1, keepdim=False)  # -> (B,1) or (B,)
    if f.dim() == 1:  # (B,)
        f = f.unsqueeze(-1)  # -> (B,1)

    f = f.to(x_batch.dtype)
    return (SD * torch.exp(f)).squeeze(-1)  # (B,)


def compute_laplacian_fast(psi_fn, f_net, x, C_occ):
    """
    Exact Laplacian via nested autograd.

    Args:
      psi_fn : callable(f_net, x, C_occ) -> (B,)
      f_net  : NN mapping x -> log factor
      x      : (B, N, d) with requires_grad=True
      C_occ  : (n_basis, n_spin)
    """
    x = x.requires_grad_(True)
    B, N, d = x.shape

    Psi = psi_fn(f_net, x, C_occ)  # (B,)
    grads = torch.autograd.grad(Psi.sum(), x, create_graph=True)[0]  # (B, N, d)

    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            g_ij = grads[:, i, j].sum()
            second = torch.autograd.grad(g_ij, x, create_graph=True)[0]
            lap = lap + second[:, i, j]

    return Psi.unsqueeze(1), lap.unsqueeze(1)  # both (B, 1)


@inject_params
def train_model(
    f_net, optimizer, C_occ, *, params=None, std=2.5, factor=0.1, print_e=50
):
    """
    Train the PINN by minimizing the PDE residual and enforcing normalization.

    Notes:
    - Reads V and E from params["V"] and params["E"] (no longer passed as args).
    - Uses omega, n_particles, n_epochs, d, N_collocation, device from params.
    """
    device = params["device"]  # string like "cpu" / "cuda" / "mps"
    w = params["omega"]
    n_particles = params["n_particles"]
    n_epochs = params["n_epochs"]
    d = params["d"]
    N_collocation = params["N_collocation"]
    E = params["E"]

    QHO_const = 0.5 * w**2
    f_net.to(device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # sample collocation points
        x = torch.normal(0, std, size=(N_collocation, n_particles, d), device=device)
        x = x.clamp(min=-5, max=5)

        # ψ and ∇²ψ
        psi, laplacian = compute_laplacian_fast(psi_fn, f_net, x, C_occ)

        # normalize ψ
        norm = torch.norm(psi, p=2)
        psi = psi / norm

        # potentials
        V_harmonic = QHO_const * (x**2).sum(dim=(1, 2)).view(-1, 1)
        V_int = compute_coulomb_interaction(x)  # V comes from params
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
            print(
                f"Epoch {epoch:05d}: PDE={loss_pde.item():.3e}  "
                f"Norm={norm.item():.3e}  Var={variance.item():.3e}"
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

    return f_net
