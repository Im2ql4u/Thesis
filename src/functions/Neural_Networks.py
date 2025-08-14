import torch
from torch.func import jacfwd, jacrev

from utils import inject_params
from .Slater_Determinant import slater_determinant_closed_shell
from .Physics import compute_coulomb_interaction


def compute_laplacian_fastt(psi_fn, f_net, x, C_occ):
    """
    Compute ψ and ∇²ψ via a single Hessian-trace for batch x of shape (B,N,d).
    """
    x = x.requires_grad_(True)  # (B, N, d)
    B, N, d = x.shape

    def psi_sum(x_batch):
        # scalar: sum of ψ over the batch
        return psi_fn(f_net, x_batch, C_occ).sum()

    # Hessian w.r.t. flattened coordinates
    H = jacfwd(jacrev(psi_sum))(x)  # (B, N, d, N, d)
    lap = H.reshape(B, N * d, N * d).diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B,)

    psi = psi_fn(f_net, x, C_occ).unsqueeze(1)  # (B, 1)
    return psi, lap.unsqueeze(1)  # (B, 1), (B, 1)


@inject_params
def psi_fn(f_net, x_batch, C_occ, *, params=None):
    """
    Wavefunction ψ = det(Slater) * exp(f_net).
    Uses params['nx'], params['ny'] for basis sizes; ω is injected inside Slater.
    """
    SD_val = slater_determinant_closed_shell(
        x_batch, C_occ, params["nx"], params["ny"]
    )  # (B, 1)
    f_val = torch.exp(f_net(x_batch))  # broadcastable to (B, 1)
    psi_val = SD_val * f_val
    return psi_val.squeeze(-1)  # (B,)


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


"""
import torch
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# ---------- 1. Build configurations ---------- #
def construct_grid2_random(
    n_particles: int,
    dims: int,
    probe_idx: int,          # which particle we draw on the grid
    Ngrid: int,
    L: float,
    n_random: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    Parameters
    ----------
    probe_idx : int
        Index of the particle whose (x,y) coordinates are scanned.
    Ngrid, L :
        Grid resolution and half-width; the grid is [-L, L]².
    n_random : int
        Number of Monte-Carlo samples for every grid node.
    Returns
    -------
    x_configs : (n_random*Ngrid², n_particles, dims) tensor
    X, Y      : (Ngrid, Ngrid) tensors with grid coordinates (useful for pcolormesh)

    assert dims == 2, "This routine assumes 2-D particles."

    # Square mesh for (x,y) of the probe particle
    lin = torch.linspace(-L, L, Ngrid, device=device)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")                 # (Ngrid, Ngrid)

    # Random block: shape (n_random, Ngrid, Ngrid, n_particles, dims)
    rand = torch.randn(n_random, Ngrid, Ngrid, n_particles, dims, device=device)

    # Overwrite probe coordinates with the mesh (broadcasting happens automatically)
    rand[..., probe_idx, 0] = X           # x-coordinate
    rand[..., probe_idx, 1] = Y           # y-coordinate

    # Collapse the first three axes → batch dimension
    x_configs = rand.reshape(-1, n_particles, dims)                # (n_random*Ngrid², …)
    return x_configs, X, Y

# ---------- 2. Evaluate and plot ---------- #
def plot_marginal_2d(
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    n_particles: int,
    probe_idx: int,
    Ngrid: int,
    L: float,
    n_random: int,
    device: str = "cpu",
    batch_size: int = 4096,
    cmap: str = "viridis",
):

    Averages |Ψ|² over random companions and shows a 2-D density for particle `probe_idx`.

    # Build configurations
    configs, X, Y = construct_grid2_random(
        n_particles, 2, probe_idx,
        Ngrid, L, n_random, device
    )

    # Evaluate psi in manageable chunks
    psi_sq = torch.empty(configs.shape[0], device=device)
    for i in range(0, configs.shape[0], batch_size):
        out = psi_fn_with_fnet(f_net, configs[i:i+batch_size],C_occ)                # (batch, 1) or (batch,)
        psi_sq[i:i+batch_size] = out.squeeze().abs()**2

    # Reshape and Monte-Carlo average
    psi_sq = psi_sq.view(n_random, Ngrid, Ngrid)             # (n_random, Ngrid, Ngrid)
    density = psi_sq.mean(dim=0).cpu()                       # (Ngrid, Ngrid)

    # Plot
    plt.figure(figsize=(7,5))
    plt.pcolormesh(X.cpu(), Y.cpu(), density.detach(), shading="auto", cmap=cmap)
    plt.xlabel(f"x (particle {probe_idx})")
    plt.ylabel(f"y (particle {probe_idx})")
    plt.title(r"fakk off")
    plt.colorbar(label="Probability density")
    plt.tight_layout()
    plt.show()

"""
