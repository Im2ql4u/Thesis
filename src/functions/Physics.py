import numpy as np
import torch
from Slater_Determinant import laplacian_2d


def compute_coulomb_interaction(x, V, eps=1e-18):
    """
    Computes the Coulomb interaction potential for a system of particles.

    Parameters:
    - x: Tensor of shape (batch_size, n_particles, d), where d is the spatial dimension.
    - V: Scaling factor for the interaction strength.

    Returns:
    - V_int: Coulomb interaction energy as a tensor of shape (batch_size, 1).
    """
    batch_size, n_particles, d = x.shape
    V_int = torch.zeros(batch_size, device=x.device)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r = x[:, i, :] - x[:, j, :]  # Pairwise distance vectors
            r_norm = torch.norm(r, dim=1) + eps  # Compute Euclidean norm (distance)
            V_int += V / r_norm  # Coulomb interaction 1/r

    return V_int.view(-1, 1)


def gaussian_interaction_2d(x, V: float, eps: float = 1e-12):
    """
    Coulomb interaction energy for a batch of electronic configurations.

    Parameters
    ----------
    x : Tensor, shape (batch, n_particles, d)
        Particle coordinates.
    V : float
        Prefactor in atomic units (e.g. e² / (4πϵ₀)).
    eps : float
        Softening term to avoid division by zero.

    Returns
    -------
    Tensor, shape (batch, 1)
        Total Coulomb energy per configuration.
    """
    batch, n_particles, _ = x.shape
    # Build pair-index tensors once (reuse globally if you like)
    idx_i, idx_j = torch.triu_indices(
        n_particles, n_particles, offset=1, device=x.device
    )
    rij = torch.norm(x[:, idx_i] - x[:, idx_j], dim=-1).clamp_min_(
        eps
    )  # (batch, n_pairs)
    V_int = (V / rij).sum(dim=1, keepdim=True)  # (batch,1)
    return V_int


# ---------------------------------------------------------------------
# 2-D Coulomb potential matrix on a grid  ( NumPy, no explicit double loop )
# ---------------------------------------------------------------------
def gaussian_interaction_potential_2d(xgrid, ygrid, V, eps=1e-12):
    """
    Build V_ij = V / |r_i − r_j| on a 2-D tensor-product grid.

    Returns
    -------
    Vmat : ndarray, shape (n_points, n_points)
        Symmetric interaction matrix with zero self-interaction on the diagonal.
    """
    X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")  # each (nx, ny)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # (n_points, 2)
    diff = coords[:, None, :] - coords[None, :, :]  # (n_points, n_points, 2)
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)  # avoid 1/0
    Vmat = V / (dist + eps)
    np.fill_diagonal(Vmat, 0.0)
    return Vmat


def compute_two_body_integrals_2d(basis, V_interaction, xgrid, ygrid):
    """
    Compute the two-electron integrals:

      ⟨pq|V|rs⟩ = ∫ d²r ∫ d²r' φ_p(r) φ_q(r) V(r, r') φ_r(r') φ_s(r')

    Parameters:
      basis         : array of shape (n_points, n_basis)
      V_interaction : interaction matrix of shape (n_points, n_points)
      xgrid, ygrid  : 1D spatial grids.
    """
    n_points, n_basis = basis.shape
    two_body = np.zeros((n_basis, n_basis, n_basis, n_basis))

    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dA = dx * dy

    for p in range(n_basis):
        print(f"Computing integrals for basis p={p}")
        for q in range(n_basis):
            pq = basis[:, p] * basis[:, q]
            for r in range(n_basis):
                for s in range(n_basis):
                    rs = basis[:, r] * basis[:, s]
                    integrand = np.outer(pq, rs) * V_interaction
                    val = np.sum(integrand) * (dA * dA)
                    two_body[p, q, r, s] = val
    return two_body


def one_electron_integral_2d(basis, xgrid, ygrid, omega):
    """
    Compute the one-electron integrals:

      Hcore_{pq} = ∫ d²r φ_p(r)[ -½∇² + ½ ω² (x²+y²) ]φ_q(r)

    Parameters:
      basis : array of shape (n_points, n_basis) with basis functions
      xgrid, ygrid : 1D spatial grids
      omega : harmonic oscillator frequency, which modifies the potential term

    Returns:
      Hcore : the one-electron Hamiltonian matrix.
    """
    n_points, n_basis = basis.shape
    nx = len(xgrid)
    ny = len(ygrid)
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    # Create 2D grid for the potential term with omega
    X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")
    V_ho = 0.5 * (omega**2) * (X**2 + Y**2)

    Hcore = np.zeros((n_basis, n_basis))

    for p in range(n_basis):
        phi_p = basis[:, p].reshape(nx, ny)
        for q in range(n_basis):
            phi_q = basis[:, q].reshape(nx, ny)
            d2phi_q = laplacian_2d(phi_q, dx, dy)
            kinetic = -0.5 * np.sum(phi_p * d2phi_q) * dx * dy
            potential = np.sum(phi_p * V_ho * phi_q) * dx * dy
            Hcore[p, q] = kinetic + potential
    return Hcore
