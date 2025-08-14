import numpy as np
import math
from math import factorial
from scipy.integrate import simps
import torch

from utils import inject_params

###############################################################################
# 1) 2D Harmonic Oscillator Basis Functions (with harmonic oscillator constant omega)
###############################################################################


def hermite_polynomial(n, x):
    """
    Evaluate the physicists' Hermite polynomial H_n(x).
    """
    coeffs = [0] * n + [1]
    return np.polynomial.hermite.hermval(x, coeffs)


def harmonic_oscillator_wavefunction_1d(n, x, omega):
    """
    1D harmonic oscillator eigenfunction ψ_n(x) with harmonic oscillator constant omega:

      ψ_n(x) = (ω/π)^(1/4)/sqrt(2^n n!) * exp(-ω x²/2) * H_n(√ω x)

    Parameters:
      n     : quantum number
      x     : spatial coordinate (can be a numpy array)
      omega : harmonic oscillator frequency
    """
    norm = (omega / np.pi) ** (1 / 4) / np.sqrt((2**n) * factorial(n))
    xi = np.sqrt(omega) * x
    Hn = hermite_polynomial(n, xi)
    return norm * np.exp(-0.5 * omega * x**2) * Hn


def harmonic_oscillator_wavefunction_2d(n_x, n_y, X, Y, omega):
    """
    2D product harmonic oscillator eigenfunction:

      ψ_{n_x,n_y}(x,y) = ψ_{n_x}(x) * ψ_{n_y}(y)

    with harmonic oscillator constant omega.

    Parameters:
      n_x   : quantum number in the x-direction
      n_y   : quantum number in the y-direction
      X, Y  : 2D arrays (e.g., from np.meshgrid) for the spatial grid
      omega : harmonic oscillator frequency
    """
    psi_x = harmonic_oscillator_wavefunction_1d(n_x, X, omega)
    psi_y = harmonic_oscillator_wavefunction_1d(n_y, Y, omega)
    return psi_x * psi_y


def initialize_harmonic_basis_2d(n_x_max, n_y_max, xgrid, ygrid, omega):
    """
    Build a 2D harmonic oscillator basis set for quantum numbers:
      0 <= n_x < n_x_max and 0 <= n_y < n_y_max,
    using the harmonic oscillator constant omega.

    Returns:
      basis: array of shape (n_points, n_basis) where
             n_points = len(xgrid) * len(ygrid)
             n_basis  = n_x_max * n_y_max.
    """
    X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")  # shape (nx, ny)
    basis_list = []
    for nx_ho in range(n_x_max):
        for ny_ho in range(n_y_max):
            psi2d = harmonic_oscillator_wavefunction_2d(nx_ho, ny_ho, X, Y, omega)
            # Flatten 2D grid to 1D vector (n_points,)
            psi2d_flat = psi2d.ravel()
            # Normalize numerically (should be ~1 if the 1D functions are normalized)
            norm_val = np.sqrt(simps(simps(psi2d**2, ygrid), xgrid))
            psi2d_flat /= norm_val
            basis_list.append(psi2d_flat)

    # Stack basis functions as columns: shape (n_points, n_basis)
    basis_array = np.column_stack(basis_list)
    return basis_array


###############################################################################
# 2) One-Electron Integrals in 2D (with modified potential including omega)
###############################################################################


def laplacian_2d(phi2d, dx, dy):
    """
    Compute the 2D Laplacian using a 5-point finite difference.
    phi2d has shape (nx, ny).
    """
    nx, ny = phi2d.shape
    lap = np.zeros_like(phi2d)

    # Second derivative in x-direction
    lap[1:-1, :] += (phi2d[:-2, :] - 2 * phi2d[1:-1, :] + phi2d[2:, :]) / (dx**2)
    # Second derivative in y-direction
    lap[:, 1:-1] += (phi2d[:, :-2] - 2 * phi2d[:, 1:-1] + phi2d[:, 2:]) / (dy**2)
    return lap


###############################################################################
# 4) Closed-Shell Hartree-Fock in 2D (Two Electrons in Ground State)
###############################################################################


def hartree_fock_2d(
    n_electrons, basis, xgrid, ygrid, Hcore, two_body, max_iter=100, tol=1e-6
):
    """
    Perform a closed-shell Hartree-Fock calculation in 2D.

    For a closed-shell system, the number of occupied spatial orbitals is:
         n_occ = n_electrons // 2
    The density matrix is constructed as:
         D_{pq} = 2 * Σ_i C_{p,i} C_{q,i}
    where the factor of 2 accounts for opposite spins.

    Returns:
      C_occ: array of shape (n_basis, n_occ) containing the occupied orbital coefficients.
      orbital_energies: array of occupied orbital energies.
    """
    n_basis = Hcore.shape[0]
    n_occ = n_electrons // 2  # For closed-shell systems

    # Initial guess: diagonalize Hcore
    eigvals, C = np.linalg.eigh(Hcore)
    C_occ = C[:, :n_occ].copy()

    def density_matrix(C_occ):
        return 2 * np.dot(C_occ, C_occ.T)

    D = density_matrix(C_occ)

    for iteration in range(max_iter):
        F = np.copy(Hcore)
        for p in range(n_basis):
            for q in range(n_basis):
                coulomb = 0.0
                exchange = 0.0
                for r in range(n_basis):
                    for s in range(n_basis):
                        coulomb += D[r, s] * two_body[p, r, q, s]
                        exchange += D[r, s] * two_body[p, r, s, q]
                F[p, q] += coulomb - 0.5 * exchange

        eigvals_new, C_new = np.linalg.eigh(F)
        C_occ_new = C_new[:, :n_occ]
        D_new = density_matrix(C_occ_new)

        delta = np.linalg.norm(D_new - D)
        print(f"Iteration {iteration}: ΔD = {delta:.3e}")
        if delta < tol:
            break
        C_occ = C_occ_new
        D = D_new

    orbital_energies = eigvals_new[:n_occ]
    E_hf = 0.5 * np.sum(D * (Hcore + F))
    print(f"Final HF Energy = {E_hf:.6f}")

    return C_occ, orbital_energies


@inject_params
def slater_determinant_closed_shell(
    x_config, C_occ, n_basis_x, n_basis_y, params=None, normalize=True
):
    """
    Compute the Slater determinant for a closed-shell system with harmonic oscillator basis.

    Parameters:
        x_config  : Tensor (B, n_particles, d), electron positions.
        C_occ     : Tensor (n_basis_x * n_basis_y, n_spin), orbital coefficients.
        n_basis_x : Number of basis functions in x.
        n_basis_y : Number of basis functions in y.
        omega     : Harmonic oscillator frequency (from params).
        normalize : Whether to normalize by 1/(n_spin!).

    Returns:
        Tensor of shape (B, 1), the (possibly normalized) product of spin-up and spin-down determinants.
    """
    device = x_config.device
    dtype = x_config.dtype
    omega = params["omega"]

    B, n_particles, d = x_config.shape
    n_spin = n_particles // 2  # Assume closed-shell

    # Ensure orbital coefficients are on the correct device
    C_occ = C_occ.to(device=device, dtype=dtype)

    # Evaluate the basis functions: must return tensor on same device
    phi_vals = evaluate_basis_functions_torch_batch_2d(
        x_config, n_basis_x, n_basis_y, omega
    )
    # shape: (B, n_particles, n_basis)

    # Split spin-up and spin-down
    phi_vals_up = phi_vals[:, :n_spin, :]  # (B, n_spin, n_basis)
    phi_vals_down = phi_vals[:, n_spin:, :]  # (B, n_spin, n_basis)

    # Contract with coefficients: result (B, n_spin, n_spin)
    psi_mat_up = torch.matmul(phi_vals_up, C_occ)  # (B, n_spin, n_spin)
    psi_mat_down = torch.matmul(phi_vals_down, C_occ)  # (B, n_spin, n_spin)

    # Compute log-determinants on CPU (MPS does not support det)
    sign_up, logdet_up = torch.linalg.slogdet(psi_mat_up.cpu())
    sign_down, logdet_down = torch.linalg.slogdet(psi_mat_down.cpu())

    # Combine results, move back to device
    logdet_full = logdet_up + logdet_down
    sign_full = sign_up * sign_down
    det_full = sign_full * torch.exp(logdet_full)  # shape: (B,)
    det_full = det_full.to(device=device, dtype=dtype)

    if normalize:
        det_full = det_full / math.factorial(n_spin)

    return det_full.view(B, 1)


# --- 1D Harmonic Oscillator Basis Functions with omega ---
def evaluate_basis_functions_torch(x, n_basis, omega):
    """
    Evaluate 1D harmonic oscillator basis functions using a Gaussian times a Hermite polynomial,
    with the harmonic oscillator parameter omega incorporated.

    The eigenfunction is:
      ψ_n(x) = (ω/π)^(1/4) / √(2ⁿ n!)  H_n(√ω x) exp(-ω x²/2)

    Parameters:
        x      : Tensor of shape (B, N) with spatial coordinates.
        n_basis: Number of basis functions.
        omega  : Harmonic oscillator frequency.

    Returns:
        A tensor of shape (B, N, n_basis) with the evaluated basis functions.
    """
    B, N = x.shape
    sqrt_omega = math.sqrt(omega)
    gauss = torch.exp(-0.5 * omega * x**2)
    # Normalization for the n=0 basis function.
    norm0 = (omega / math.pi) ** 0.25
    phi_list = [norm0 * gauss]  # n = 0

    if n_basis > 1:
        # n = 1: H_1(√ω x) = 2√ω x and normalization factor √2 in the denominator.
        norm1 = norm0 / math.sqrt(2)
        phi_list.append(norm1 * (2 * sqrt_omega * x) * gauss)

    if n_basis > 2:
        # Use the recurrence relation for Hermite polynomials:
        # H_{n+1}(√ω x) = 2√ω x * H_n(√ω x) - 2n * H_{n-1}(√ω x)
        H_prev_prev = torch.ones_like(x)  # H_0(√ω x)
        H_prev = 2 * sqrt_omega * x  # H_1(√ω x)
        for n in range(1, n_basis - 1):
            H_curr = 2 * sqrt_omega * x * H_prev - 2 * n * H_prev_prev
            norm = norm0 / math.sqrt((2 ** (n + 1)) * math.factorial(n + 1))
            phi_list.append(norm * H_curr * gauss)
            H_prev_prev, H_prev = H_prev, H_curr
    return torch.stack(phi_list, dim=-1)


# --- 2D Basis Functions from 1D ---
def evaluate_basis_functions_torch_batch_2d(x, n_basis_x, n_basis_y, omega):
    """
    Evaluate 2D harmonic oscillator basis functions as a separable product of 1D basis functions.

    Parameters:
        x        : Tensor of shape (B, N, d) with spatial coordinates,
                   where d >= 2 and the first two components correspond to x and y.
        n_basis_x: Number of 1D basis functions in the x direction.
        n_basis_y: Number of 1D basis functions in the y direction.
        omega    : Harmonic oscillator frequency.

    Returns:
        A tensor of shape (B, N, n_basis_x * n_basis_y) with the evaluated 2D basis functions.
    """
    B, N, d = x.shape
    x_coord = x[..., 0]  # shape: (B, N)
    y_coord = x[..., 1]  # shape: (B, N)
    phi_x = evaluate_basis_functions_torch(
        x_coord, n_basis_x, omega
    )  # shape: (B, N, n_basis_x)
    phi_y = evaluate_basis_functions_torch(
        y_coord, n_basis_y, omega
    )  # shape: (B, N, n_basis_y)

    # Compute the outer product of phi_x and phi_y for each electron.
    product = phi_x.unsqueeze(-1) * phi_y.unsqueeze(
        -2
    )  # shape: (B, N, n_basis_x, n_basis_y)
    return product.reshape(B, N, n_basis_x * n_basis_y)
