import math

import numpy as np
import torch

from utils import inject_params

from .Slater_Determinant import laplacian_2d


@inject_params
def compute_coulomb_interaction(
    x: torch.Tensor,
    *,
    params=None,
    eps_rel: float = 1e-6,  # ε = eps_rel * 1/sqrt(ω)  (physical units)
    cap: float = 1e6,  # clamp absurd contributions
) -> torch.Tensor:
    """
    Vectorized Coulomb with C² soft-core radius r̃ = sqrt(r^2 + ε^2).
    Returns V_int of shape (B,1).
    """
    kappa = float(params["V"])
    omega = float(params["omega"])
    B, N, d = x.shape
    dev = x.device

    # pairwise diffs and distances (physical coords)
    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
    ii, jj = torch.triu_indices(N, N, 1, device=dev)
    r2 = (diff[:, ii, jj, :] ** 2).sum(-1)  # (B,P)

    # soft-core ε in physical units
    l0 = 1.0 / math.sqrt(max(omega, 1e-12))
    eps2 = (eps_rel * l0) ** 2

    r_soft = torch.sqrt(r2 + eps2)  # (B,P)
    Vij = kappa / r_soft  # (B,P)

    # clamp and guard
    Vij = torch.clamp(Vij, max=cap)
    Vij = torch.nan_to_num(Vij, nan=cap, posinf=cap, neginf=-cap)

    V = Vij.sum(dim=1, keepdim=True)  # (B,1)
    bad = ~torch.isfinite(V).squeeze(1)
    if bad.any():
        V[bad] = 0.0  # optional: or resample those rows upstream
    return V


@inject_params
def gaussian_interaction_2d(x, eps: float = 1e-12, *, params=None):
    """
    Coulomb interaction energy for a batch of electronic configurations (vectorized).

    Parameters
    ----------
    x : Tensor, shape (batch, n_particles, d)
        Particle coordinates.
    V : float (taken from params["V"])
        Prefactor in atomic units (e.g. e² / (4πϵ₀)).
    eps : float
        Softening term to avoid division by zero.

    Returns
    -------
    Tensor, shape (batch, 1)
        Total Coulomb energy per configuration.
    """
    V = params["V"]
    batch, n_particles, _ = x.shape
    idx_i, idx_j = torch.triu_indices(n_particles, n_particles, offset=1, device=x.device)
    rij = torch.norm(x[:, idx_i] - x[:, idx_j], dim=-1).clamp_min_(eps)  # (batch, n_pairs)
    V_int = (V / rij).sum(dim=1, keepdim=True)  # (batch, 1)
    return V_int


@inject_params
def gaussian_interaction_potential_2d(xgrid, ygrid, eps: float = 1e-12, *, params=None):
    """
    Build V_ij = V / |r_i − r_j| on a 2-D tensor-product grid.

    Returns
    -------
    Vmat : ndarray, shape (n_points, n_points)
        Symmetric interaction matrix with zero self-interaction on the diagonal.
    """
    V = params["V"]
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


@inject_params
def one_electron_integral_2d(basis, xgrid, ygrid, *, params=None):
    """
    Compute the one-electron integrals:

      Hcore_{pq} = ∫ d²r φ_p(r)[ -½∇² + ½ ω² (x²+y²) ]φ_q(r)

    Parameters:
      basis : array of shape (n_points, n_basis) with basis functions
      xgrid, ygrid : 1D spatial grids
      ω : taken from params["omega"]

    Returns:
      Hcore : the one-electron Hamiltonian matrix.
    """
    omega = params["omega"]
    n_points, n_basis = basis.shape
    nx = len(xgrid)
    ny = len(ygrid)
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    # 2D grid for the harmonic oscillator potential
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
