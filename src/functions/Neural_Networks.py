import torch
from .Slater_Determinant import slater_determinant_closed_shell
from utils import inject_params
from .Physics import compute_coulomb_interaction
from torch.func import jacfwd, jacrev


def compute_laplacian_fastt(psi_fn, f_net, x, C_occ):
    """
    Compute ψ and ∇²ψ via a single Hessian‐trace, for batch x of shape (B,N,d).
    """
    # 1) make sure x has grad enabled
    x = x.requires_grad_(True)  # (B, N, d)
    B, N, d = x.shape

    # 2) define a sum‐over‐batch wrapper for functorch
    def psi_sum(x_batch):
        # returns a scalar: sum of ψ over the batch
        return psi_fn(f_net, x_batch, C_occ).sum()

    # 3) compute Hessian H_{αβ} = ∂²(∑ψ)/∂x_α∂x_β in one shot
    H = jacfwd(jacrev(psi_sum))(x)  # shape (B, N, d, N, d)

    # 4) trace and reshape: sum over the diagonal of the flattened (N·d)x(N·d)
    lap = (
        H.reshape(B, N * d, N * d).diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # shape (B,)
    )

    # 5) return ψ (as column) and ∇²ψ
    psi = psi_fn(f_net, x, C_occ).unsqueeze(1)  # (B,1)
    return psi, lap.unsqueeze(1)  # both (B,1)


@inject_params
def psi_fn(f_net, x_batch, C_occ, device=None, params=None):
    """
    Compute the wavefunction for a single configuration.
    Args:
        x_single (torch.Tensor): (n_particles, d) with requires_grad=True.
    Returns:
        Scalar tensor for psi.
    """
    SD_val = slater_determinant_closed_shell(x_batch, C_occ, params["nx"], params["ny"])
    f_val = torch.exp(f_net(x_batch))
    psi_val = SD_val * f_val
    return psi_val


def compute_laplacian_fast(psi_fn, f_net, x, C_occ):
    """
    Exact Laplacian via nested torch.autograd.grad calls.

    Args:
      psi_fn : callable(f_net, x_batch, C_occ, nx, ny) -> psi of shape (batch,)
      f_net  : your neural network
      x      : Tensor (batch, n_particles, d) with requires_grad=True
      C_occ, nx, ny : parameters for slater_determinant_closed_shell

    Returns:
      lap : Tensor of shape (batch, 1), containing ∇²ψ for each sample.
    """
    # ensure we have a fresh graph on x
    x = x.requires_grad_(True)
    batch, n, d = x.shape

    # 1) compute psi (batch,)
    Psi = psi_fn(f_net, x, C_occ)
    psi = Psi  # .squeeze()
    # 2) first derivatives ∂ψ/∂x  -> shape (batch, n, d)
    grads = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]

    # 3) accumulate second derivatives per coordinate
    lap = torch.zeros(batch, device=x.device)
    for i in range(n):
        for j in range(d):
            # sum over batch to get scalar for this coordinate
            g_ij = grads[:, i, j].sum()
            # second derivative ∂²ψ/∂x_{i,j}²
            second = torch.autograd.grad(g_ij, x, create_graph=True)[0]
            # add the diagonal piece
            lap = lap + second[:, i, j]

    return Psi, lap.unsqueeze(1)  # shape (batch,1)


@inject_params
def train_model(
    V, E, f_net, optimizer, C_occ, std=2.5, factor=0.1, params=None, print_e=50
):
    """
    Train the model as a PINN by minimizing the PDE residual and enforcing normalization.
    (This function is kept unchanged.)
    """
    device = params["device"]
    w = params["omega"]
    n_particles = params["n_particles"]
    n_epochs = params["n_epochs"]
    d = params["d"]
    N_collocation = params["N_collocation"]
    QHO_const = 0.5 * w**2
    f_net.to(device)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        x = torch.normal(0, std, size=(N_collocation, n_particles, d), device=device)
        x = x.clamp(min=-5, max=5)
        psi, laplacian = compute_laplacian_fast(psi_fn, f_net, x, C_occ)
        norm = torch.norm(psi, p=2)
        psi = psi / norm

        V_harmonic = QHO_const * (x**2).sum(dim=(1, 2)).view(-1, 1)
        V_int = compute_coulomb_interaction(x, V)
        V_total = V_harmonic + V_int
        H_psi = -0.5 * laplacian + V_total * psi

        residual = H_psi - E * psi
        loss_pde = torch.mean((residual) ** 2)
        variance = torch.var(H_psi / psi)

        loss_norm = factor * (norm - 1) ** 2
        loss = loss_pde + loss_norm
        loss.backward()
        optimizer.step()
        if epoch % print_e == 0:
            print(
                f"Epoch {epoch:05d}: PDE Loss = {loss_pde.item():.3e},Norm = {norm.item():.3e},  Variance = {variance.item():.3e}"
            )
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
    return f_net  # , metrics


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
