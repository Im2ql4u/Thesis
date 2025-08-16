# functions/Slater_Determinant.py
# ---------------------------------------------------------------
# Clean, basis-agnostic engines for 2D HO (Cartesian + Fock–Darwin)
# ---------------------------------------------------------------
from __future__ import annotations

import math
from math import factorial
from typing import List, Tuple, Optional

import numpy as np

# SciPy generalized symmetric eig (preferred), with graceful fallback
try:
    from scipy.linalg import eigh as eigh_generalized  # solves A v = S v w
except Exception:
    eigh_generalized = None

import torch

from utils import inject_params

# ===============================================================
# Utilities: grids, Simpson weights, centered FFT convolution
# ===============================================================


def simpson_weights(x: np.ndarray) -> np.ndarray:
    """Classic Simpson weights on a uniform 1D grid x."""
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points for Simpson integration.")
    h = (x[-1] - x[0]) / (n - 1)
    w = np.ones(n)
    if n >= 3:
        w[1:-1:2] = 4
        w[2:-1:2] = 2
    return w * (h / 3.0)


def mesh_xy(
    L: float, n_grid: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-L, L, n_grid)
    y = np.linspace(-L, L, n_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def _zero_pad_center(
    f: np.ndarray, pad_factor: int = 2
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Center f into a larger complex array for linear convolution."""
    nx, ny = f.shape
    Nx, Ny = pad_factor * nx, pad_factor * ny
    F = np.zeros((Nx, Ny), dtype=np.complex128)
    i0 = (Nx - nx) // 2
    j0 = (Ny - ny) // 2
    F[i0 : i0 + nx, j0 : j0 + ny] = f.astype(np.complex128, copy=False)
    return F, (i0, j0), (Nx, Ny)


def _crop_center(
    h: np.ndarray, out_shape: Tuple[int, int], origin: Tuple[int, int]
) -> np.ndarray:
    """Crop the centered physical region after linear convolution."""
    i0, j0 = origin
    nx, ny = out_shape
    return h[i0 : i0 + nx, j0 : j0 + ny]


@inject_params
def coulomb_kernel_fft(
    x: np.ndarray,
    y: np.ndarray,
    pad_factor: int = 2,
    eps: float = 1e-9,
    kappa: float = 1.0,
    *,
    params=None,
) -> np.ndarray:
    """
    FFT of the *centered* 1/(kappa*r) kernel for linear convolution.
    We build V with its singularity at the center pixel and then ifftshift
    so that np.fft.fftn sees the origin at index (0,0).
    """
    if params is not None:
        if kappa is None:
            kappa = params.get("kappa", 1.0)
        if pad_factor is None:
            pad_factor = params.get("pad_factor", 2)

    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    Nx, Ny = pad_factor * nx, pad_factor * ny

    ix = np.arange(Nx) - (Nx // 2)
    iy = np.arange(Ny) - (Ny // 2)
    RX, RY = np.meshgrid(ix * dx, iy * dy, indexing="ij")
    R = np.hypot(RX, RY)

    V = 1.0 / (kappa * np.maximum(R, eps))
    V[Nx // 2, Ny // 2] = 0.0  # remove self term

    V0 = np.fft.ifftshift(V.astype(np.complex128))
    return np.fft.fftn(V0)  # g_fft


# ===============================================================
# Cartesian HO basis (numpy grids + torch evaluators)
# ===============================================================


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """Physicists' Hermite polynomial H_n(x)."""
    coeffs = [0] * n + [1]
    return np.polynomial.hermite.hermval(x, coeffs)


@inject_params
def harmonic_oscillator_wavefunction_1d(
    n: int, x: np.ndarray, *, params=None
) -> np.ndarray:
    r"""
    1D HO eigenfunction ψ_n(x) with frequency ω from params:
      ψ_n(x) = (ω/π)^(1/4) / sqrt(2^n n!) * exp(-ω x²/2) * H_n(√ω x)
    """
    omega = float(params["omega"])
    xi = np.sqrt(omega) * x
    Hn = hermite_polynomial(n, xi)
    norm = (omega / np.pi) ** 0.25 / np.sqrt((2.0**n) * factorial(n))
    return norm * np.exp(-0.5 * omega * x**2) * Hn


@inject_params
def harmonic_oscillator_wavefunction_2d(
    n_x: int, n_y: int, X: np.ndarray, Y: np.ndarray, *, params=None
) -> np.ndarray:
    """Separable 2D HO eigenfunction ψ_{n_x}(x) ψ_{n_y}(y), ω via params."""
    psi_x = harmonic_oscillator_wavefunction_1d(n_x, X)
    psi_y = harmonic_oscillator_wavefunction_1d(n_y, Y)
    return psi_x * psi_y


@inject_params
def initialize_harmonic_basis_2d(
    n_x_max: int, n_y_max: int, xgrid: np.ndarray, ygrid: np.ndarray, *, params=None
) -> np.ndarray:
    """
    Cartesian product basis: 0 <= n_x < n_x_max, 0 <= n_y < n_y_max.
    Each basis function normalized on the (xgrid, ygrid) box with Simpson.
    Returns (n_points, n_basis) with basis as columns.
    """
    X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")  # (nx, ny)
    nx, ny = len(xgrid), len(ygrid)
    wx = simpson_weights(xgrid)[:, None]
    wy = simpson_weights(ygrid)[None, :]

    cols = []
    for nx_ho in range(n_x_max):
        for ny_ho in range(n_y_max):
            psi2d = harmonic_oscillator_wavefunction_2d(nx_ho, ny_ho, X, Y)
            dens = psi2d**2
            norm2 = float(np.sum(wx * dens * wy))
            if norm2 <= 0.0:
                raise RuntimeError("Normalization failed: non-positive norm.")
            psi2d /= np.sqrt(norm2)
            cols.append(psi2d.reshape(nx * ny))

    return np.column_stack(cols)  # (nx*ny, n_basis)


def laplacian_2d(phi2d: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Five-point FD Laplacian on a regular 2D grid."""
    lap = np.zeros_like(phi2d)
    lap[1:-1, :] += (phi2d[:-2, :] - 2.0 * phi2d[1:-1, :] + phi2d[2:, :]) / (dx * dx)
    lap[:, 1:-1] += (phi2d[:, :-2] - 2.0 * phi2d[:, 1:-1] + phi2d[:, 2:]) / (dy * dy)
    return lap


# Torch evaluators (Cartesian)
@inject_params
def evaluate_basis_functions_torch(
    x: torch.Tensor, n_basis: int, *, params=None
) -> torch.Tensor:
    """1D HO basis in torch with ω from params, returns (B,N,n_basis)."""
    omega = float(params["omega"])
    dtype = x.dtype
    sqrt_omega = math.sqrt(omega)
    gauss = torch.exp(-0.5 * omega * x * x)

    norm0 = (omega / math.pi) ** 0.25
    cols = [torch.as_tensor(norm0, dtype=dtype, device=x.device) * gauss]

    if n_basis > 1:
        norm1 = norm0 / math.sqrt(2.0)
        cols.append(norm1 * (2.0 * sqrt_omega * x) * gauss)

    if n_basis > 2:
        H_nm1 = 2.0 * sqrt_omega * x
        H_nm2 = torch.ones_like(x)
        for n in range(1, n_basis - 1):
            H_n = 2.0 * sqrt_omega * x * H_nm1 - 2.0 * n * H_nm2
            norm = norm0 / math.sqrt((2.0 ** (n + 1)) * math.factorial(n + 1))
            cols.append(norm * H_n * gauss)
            H_nm2, H_nm1 = H_nm1, H_n

    return torch.stack(cols, dim=-1)


@inject_params
def evaluate_basis_functions_torch_batch_2d(
    x: torch.Tensor, n_basis_x: int, n_basis_y: int, *, params=None
) -> torch.Tensor:
    """Separable 2D HO basis in torch (Cartesian). Returns (B,N,n_basis_x*n_basis_y)."""
    if x.shape[-1] < 2:
        raise ValueError("x must have at least 2 spatial dims (x,y).")
    xcoord = x[..., 0]
    ycoord = x[..., 1]
    phi_x = evaluate_basis_functions_torch(xcoord, n_basis_x)
    phi_y = evaluate_basis_functions_torch(ycoord, n_basis_y)
    prod = phi_x.unsqueeze(-1) * phi_y.unsqueeze(-2)  # (B,N,nx,ny)
    return prod.reshape(x.shape[0], x.shape[1], n_basis_x * n_basis_y)


# ===============================================================
# FD (Fock–Darwin) basis: numpy grid builder + torch evaluator
# ===============================================================


def _fd_indices(emax: int) -> List[Tuple[int, int]]:
    """List of (n,m) with 2n+|m| <= emax, ordered by shell then m."""
    out = []
    for e in range(emax + 1):
        for m in range(-e, e + 1):
            if (e - abs(m)) % 2 == 0:
                out.append(((e - abs(m)) // 2, m))
    return out


def _genlaguerre_torch(n: int, alpha: int, z: torch.Tensor) -> torch.Tensor:
    """Differentiable generalized Laguerre L_n^{(alpha)}(z) via series."""
    k = torch.arange(n + 1, device=z.device, dtype=z.dtype)
    log_num = (
        torch.lgamma(torch.tensor(n + alpha + 1.0, dtype=z.dtype, device=z.device))
        - torch.lgamma((n - k) + 1.0)
        - torch.lgamma((alpha + k) + 1.0)
    )
    log_den = torch.lgamma(k + 1.0)
    coeff = torch.exp(log_num - log_den).view(*([1] * z.ndim), -1)
    zpow = ((-z)[..., None]) ** k
    return (coeff * zpow).sum(dim=-1)


def _fd_orbital_torch(
    n: int, m: int, x: torch.Tensor, y: torch.Tensor, omega: float, make_real: bool
) -> torch.Tensor:
    """FD orbital (torch); real cos/sin combos if make_real=True, else complex e^{imθ}."""
    l0 = 1.0 / math.sqrt(omega)
    rho2 = (x * x + y * y) / (l0 * l0)
    rho = torch.sqrt(rho2 + 1e-40)
    theta = torch.atan2(y, x)
    am = abs(m)
    # log normalization constant to avoid overflow
    logN = -math.log(l0) + 0.5 * (
        torch.lgamma(torch.tensor(float(n) + 1, dtype=x.dtype, device=x.device))
        - torch.lgamma(torch.tensor(float(n + am) + 1, dtype=x.dtype, device=x.device))
        - math.log(math.pi)
    )
    N = torch.exp(logN)
    radial = (rho**am) * torch.exp(-0.5 * rho2) * _genlaguerre_torch(n, am, rho2)
    if make_real:
        if m == 0:
            ang = torch.ones_like(theta)
        elif m > 0:
            ang = math.sqrt(2.0) * torch.cos(m * theta)
        else:
            ang = math.sqrt(2.0) * torch.sin(abs(m) * theta)
        return N * radial * ang
    else:
        # complex e^{imθ}
        ang = torch.complex(torch.cos(m * theta), torch.sin(m * theta))
        return torch.complex(N * radial, torch.zeros_like(radial)) * ang


# Global FD index order, set after compute_integrals() so Slater matches HF exactly.
_fd_idx_global: Optional[List[Tuple[int, int]]] = None


def set_fd_idx(idx_list: List[Tuple[int, int]]) -> None:
    """Set the global FD (n,m) order to the integrals' order."""
    global _fd_idx_global
    _fd_idx_global = list(idx_list)


@inject_params
def _evaluate_fd_basis_torch_batch(
    x: torch.Tensor,
    emax: int = 3,
    omega: float = 1.0,
    make_real: bool = True,
    *,
    params=None,
) -> torch.Tensor:
    """
    Evaluate FD basis on a batch x: (B,N,2). Uses the exact (n,m) order set by set_fd_idx(...)
    if available; otherwise rebuilds via _fd_indices(emax).
    """
    if params is not None:
        if emax is None:
            emax = params["emax"]
        if omega is None:
            omega = params["omega"]
        if make_real is None:
            make_real = params.get("fd_make_real", True)

    idx = _fd_idx_global if _fd_idx_global is not None else _fd_indices(emax)
    cols = [
        _fd_orbital_torch(n, m, x[..., 0], x[..., 1], float(omega), bool(make_real))
        for (n, m) in idx
    ]
    return torch.stack(cols, dim=-1)  # (B, N, n_basis_fd)


@inject_params
def fd_wavefunctions_on_grid(
    e_max: int, omega: float, x: np.ndarray, y: np.ndarray, *, params=None
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Fock–Darwin orbitals on a Cartesian grid (NumPy).
    If params['fd_make_real'] is True (default), returns REAL cos/sin combos matching the Torch evaluator:
      - m = 0:             φ_{n,0}(r)                           (1 column)
      - m > 0:             √2 φ_{n,|m|}(r) cos(mθ)  -> index (n, +m)
      - m < 0:             √2 φ_{n,|m|}(r) sin(|m|θ) -> index (n, -|m|)
    If False, returns complex e^{imθ} (then keep fd_make_real=False in Torch too).
    Returns: Phi (nx*ny, nb), idx list [(n,m)].
    """
    if params is not None:
        if omega is None:
            omega = params["omega"]
        if e_max is None:
            e_max = params.get("emax", e_max)
    make_real = True if params is None else params.get("fd_make_real", True)

    X, Y = np.meshgrid(x, y, indexing="ij")
    nx, ny = len(x), len(y)
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    l0 = 1.0 / math.sqrt(omega)
    rho = r / l0

    idx_raw = _fd_indices(e_max)  # pairs (n,m) with 2n+|m|<=emax
    cols: List[np.ndarray] = []
    idx: List[Tuple[int, int]] = []

    # Use generalized Laguerre with α=|m|
    try:
        from scipy.special import eval_genlaguerre as Lgen

        def Rnm(n, mabs, rho2):
            return (rho**mabs) * np.exp(-0.5 * rho2) * Lgen(n, mabs, rho2)

    except Exception:

        def Rnm(n, mabs, rho2):
            # Fallback: L_n^{(α)} ~ lagval with α=0 (rough fallback); prefer SciPy when available
            return (
                (rho**mabs)
                * np.exp(-0.5 * rho2)
                * np.polynomial.laguerre.lagval(rho2, [0] * n + [1])
            )

    rho2 = rho**2

    for n, m in idx_raw:
        am = abs(m)
        # analytic normalization
        N = (1.0 / l0) * math.sqrt(
            math.factorial(n) / (math.pi * math.factorial(n + am))
        )
        radial = Rnm(n, am, rho2)

        if not make_real:
            ang = np.exp(1j * m * theta)
            phi = (N * radial * ang).reshape(nx * ny)
            cols.append(phi.astype(np.complex128))
            idx.append((n, m))
            continue

        # REAL basis:
        if m == 0:
            ang = np.ones_like(theta)
            phi = (N * radial * ang).reshape(nx * ny)
            cols.append(phi.astype(np.float64))
            idx.append((n, 0))
        elif m > 0:
            # cos(mθ) column -> (n, +m)
            ang_c = math.sqrt(2.0) * np.cos(m * theta)
            phi_c = (N * radial * ang_c).reshape(nx * ny)
            cols.append(phi_c.astype(np.float64))
            idx.append((n, +m))
            # sin(mθ) column -> (n, -m)
            ang_s = math.sqrt(2.0) * np.sin(m * theta)
            phi_s = (N * radial * ang_s).reshape(nx * ny)
            cols.append(phi_s.astype(np.float64))
            idx.append((n, -m))
        else:
            # We handle negative m via the m>0 branch, so skip here
            pass

    Phi = np.column_stack(cols)
    return Phi, idx


# ===============================================================
# Integrals builder (basis-agnostic)
# ===============================================================
@inject_params
def compute_integrals(
    *,
    params=None,
):
    """
    Build basis on a grid, compute one-electron core and two-electron Dirac integrals.
    Returns:
      Hcore (nb,nb), two_dirac (nb,nb,nb,nb), basis_info dict with:
        name ('fd' or 'cart'), idx (list), grid (x,y), Phi (npts,nb), S (nb,nb)
    """
    # ---- defaults from params
    if params is not None:
        basis_type = params.get("basis", "cart")
        omega = params["omega"]
        L = params["L"]
        n_grid = params["n_grid"]
        kappa = params.get("kappa", 1.0)
        pad_factor = params.get("pad_factor", 2)
        e_max = params.get("emax")
        n_x_max = params.get("nx")
        n_y_max = params.get("ny")

    # ---- grids and weights
    x, y, X, Y = mesh_xy(L, n_grid)
    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    wx = simpson_weights(x)[:, None]
    wy = simpson_weights(y)[None, :]
    W = (wx * wy).reshape(-1)  # (npts,)

    # ---- build basis matrix Phi, index list, and flags
    basis_type_l = basis_type.lower()
    if basis_type_l.startswith("fd"):
        Phi, idx = fd_wavefunctions_on_grid(
            e_max, omega, x, y
        )  # (npts, nb), real if fd_make_real=True
        nb = Phi.shape[1]
        basis_name = "fd"
        # disable m-selection when using real combos
        if params is not None and params.get("fd_make_real", True):
            mvals = None
        else:
            mvals = [m for (n, m) in idx]
    elif basis_type_l.startswith("cart"):
        Phi_real = initialize_harmonic_basis_2d(
            n_x_max, n_y_max, x, y
        )  # (npts, nb), real
        Phi = Phi_real.astype(np.complex128)
        nb = Phi.shape[1]
        basis_name = "cart"
        idx = [(ix, iy) for ix in range(n_x_max) for iy in range(n_y_max)]
        mvals = None
    else:
        raise ValueError("basis_type must be 'fd' or 'cart'.")

    # ---- overlap S (MUST be defined before return)
    S = (Phi.conj().T * W) @ Phi  # (nb, nb) Hermitian

    # ---- one-electron core Hcore
    if basis_name == "fd":
        # analytic diagonal FD core: E_{n,m} = (2n + |m| + 1) * ω
        Hcore = np.zeros((nb, nb), dtype=float)
        for p, (n, m) in enumerate(idx):
            Hcore[p, p] = (2 * n + abs(m) + 1) * omega
        # we still reuse phi_grid below for Coulomb
        phi_grid = [Phi[:, k].reshape(nx, ny) for k in range(nb)]
    else:
        # numeric (Cartesian) core via Laplacian + trap
        Vtrap = 0.5 * (omega**2) * (X * X + Y * Y)
        phi_grid = [Phi[:, k].reshape(nx, ny) for k in range(nb)]
        Hcore = np.zeros((nb, nb), dtype=float)
        lap_list = [-0.5 * laplacian_2d(phi_grid[p].real, dx, dy) for p in range(nb)]
        for p in range(nb):
            Tp = lap_list[p]  # (nx,ny)
            for q in range(p, nb):
                Tpq = np.real(np.sum(wx * (phi_grid[q].real * Tp) * wy))
                Vpq = np.real(
                    np.sum(wx * (phi_grid[p] * (Vtrap * phi_grid[q])).real * wy)
                )
                Hcore[p, q] = Tpq + Vpq
                Hcore[q, p] = Hcore[p, q]

        # ---- two-electron Dirac via centered FFT convolution (correct formulation)
    V_fft = coulomb_kernel_fft(x, y, pad_factor=pad_factor, kappa=kappa)
    rho = [
        [(phi_grid[p] * np.conjugate(phi_grid[r])) for r in range(nb)]
        for p in range(nb)
    ]

    def allow(p, r, q, s) -> bool:
        if mvals is None:
            return True
        return (mvals[p] - mvals[r]) == (mvals[q] - mvals[s])

    two_dirac = np.zeros((nb, nb, nb, nb), dtype=float)

    # Precompute FFT(ρ_pr) and the convolved G_pr once, reuse across (q,s)
    for p in range(nb):
        for r in range(nb):
            rho_pr = rho[p][r]  # (nx,ny)
            rho_pr_pad, origin_pr, (Nx, Ny) = _zero_pad_center(rho_pr, pad_factor)
            F_pr = np.fft.fftn(rho_pr_pad)  # FFT(ρ_pr)
            # G_pr = V * ρ_pr  (linear convolution)
            G_pad = np.fft.ifftn(F_pr * V_fft).real  # (Nx,Ny)
            G_phys = _crop_center(G_pad, (nx, ny), origin_pr)  # back to (nx,ny)
            # scale by dr1 = dx*dy to approximate the inner integral
            G_phys *= dx * dy

            # Now (pr|qs) = ∫ ρ_qs(r2) * G_pr(r2) dr2
            for q in range(nb):
                for s in range(nb):
                    if not allow(p, r, q, s):
                        continue
                    rho_qs = rho[q][s]  # (nx,ny)
                    val = float(np.sum(wx * (G_phys * rho_qs).real * wy))
                    two_dirac[p, r, q, s] = val

    basis_info = {
        "name": basis_name,
        "idx": idx,
        "grid": (x, y),
        "Phi": Phi,
        "S": S,
    }
    return Hcore, two_dirac, basis_info


# ===============================================================
# RHF (closed shell) with generalized eigensolver
# ===============================================================


def _eigh_generalized_or_lowdin(
    A: np.ndarray, S: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve A v = S v w for Hermitian A,S. Uses SciPy if available, else Löwdin orthogonalization."""
    if eigh_generalized is not None:
        eps, C = eigh_generalized(A, S)
        return eps.real, C.real if np.isrealobj(A) and np.isrealobj(S) else (eps, C)
    # Löwdin orthogonalization fallback
    evals, U = np.linalg.eigh(S)
    evals[evals < 1e-14] = 1e-14
    Sinvhalf = U @ (np.diag(1.0 / np.sqrt(evals)) @ U.T)
    A_ortho = Sinvhalf.T @ A @ Sinvhalf
    eps, Ctil = np.linalg.eigh(A_ortho)
    C = Sinvhalf @ Ctil
    return eps, C


@inject_params
def hartree_fock_closed_shell(
    Hcore: np.ndarray,
    two_dirac: np.ndarray,  # (pr|qs)
    *,
    S: Optional[np.ndarray] = None,
    params=None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Basis-agnostic closed-shell RHF (Roothaan):
      F C = S C ε,  with C^T S C = I  (or Löwdin fallback).
    Returns: C_occ (nb,n_occ), eps_occ (n_occ,), E_HF
    """
    if params is not None:
        n_electrons = params["n_particles"]
        max_iter = params.get("hf_max_iter", 100)
        tol = params.get("hf_tol", 1e-8)
        damping = params.get("hf_damping", 0.0)
        verbose = params.get("hf_verbose", True)

    nb = Hcore.shape[0]
    n_occ = n_electrons // 2
    if S is None:
        S = np.eye(nb)

    # Initial generalized eig from Hcore
    eps, C = _eigh_generalized_or_lowdin(Hcore, S)
    C_occ = C[:, :n_occ].copy()

    def density(Cocc: np.ndarray) -> np.ndarray:
        # With C^T S C = I, the usual D = 2 C_occ C_occ^T holds
        return 2.0 * (Cocc @ Cocc.T)

    D = density(C_occ)

    for it in range(1, max_iter + 1):
        J = np.einsum("rs,prqs->pq", D, two_dirac, optimize=True)
        K = np.einsum("rs,prsq->pq", D, two_dirac, optimize=True)
        F = Hcore + J - 0.5 * K

        eps_new, C_new = _eigh_generalized_or_lowdin(F, S)
        C_occ_new = C_new[:, :n_occ]
        D_new = density(C_occ_new)

        D_mix = (
            (1 - damping) * D_new + damping * D if damping and damping > 0 else D_new
        )
        dD = np.linalg.norm(D_mix - D)
        if verbose:
            print(f"HF iter {it:3d}: ||ΔD|| = {dD:.3e}")

        D, C_occ, eps = D_mix, C_occ_new, eps_new
        if dD < tol:
            break

    # Final RHF energy
    E_HF = 0.5 * np.sum(D * (Hcore + F))

    # Enforce S-orthonormality on occupied block
    S_occ = C_occ.T @ S @ C_occ
    # Robust symmetric normalization
    U, svals, _ = np.linalg.svd(S_occ)
    S_occ_inv_sqrt = U @ (np.diag(1.0 / np.sqrt(np.maximum(svals, 1e-15))) @ U.T)
    C_occ = C_occ @ S_occ_inv_sqrt

    return C_occ, eps[:n_occ], float(E_HF)


# ===============================================================
# Slater determinant (torch), basis-agnostic front-end
# ===============================================================


def _slogdet_batched(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safe slogdet on batch; falls back to CPU if needed."""
    try:
        return torch.linalg.slogdet(M)
    except RuntimeError:
        S, L = torch.linalg.slogdet(M.cpu())
        return S.to(M.device), L.to(M.device)


@inject_params
def slater_determinant_closed_shell(
    x_config: torch.Tensor,
    C_occ: torch.Tensor,
    n_basis_x: int,
    n_basis_y: int,
    *,
    params=None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Closed-shell Slater determinant (basis-agnostic).
      - If params['basis'] starts with "fd", use FD evaluator (@inject_params).
      - Else use Cartesian evaluator with (n_basis_x, n_basis_y).
      - For N=2 and n_occ=1, uses product shortcut φ_occ(r1) φ_occ(r2).
    Returns: (B,1) real/float tensor.
    """
    device = x_config.device
    B, N, d = x_config.shape
    if d < 2:
        raise ValueError("x_config must contain (x,y) coordinates.")

    basis = params.get("basis", "cart").lower()
    omega = params["omega"]

    if basis.startswith("fd"):
        emax = params["emax"]
        make_real = params.get("fd_make_real", True)
        Phi = _evaluate_fd_basis_torch_batch(
            x_config, emax=emax, omega=omega, make_real=make_real
        )  # (B,N,nb_fd)
    else:
        Phi = evaluate_basis_functions_torch_batch_2d(x_config, n_basis_x, n_basis_y)

    C_occ = C_occ.to(device=device, dtype=Phi.dtype)  # (n_basis, n_occ)
    if Phi.shape[-1] != C_occ.shape[0]:
        raise RuntimeError(
            f"Basis size mismatch: Phi has {Phi.shape[-1]} cols, C_occ has {C_occ.shape[0]} rows."
        )

    Psi = torch.matmul(Phi, C_occ)  # (B,N,n_occ)

    # N=2 closed-shell shortcut (more stable)
    if N == 2 and C_occ.shape[1] == 1:
        mo = Psi.squeeze(-1)  # (B,2)
        sd = (mo[:, 0] * mo[:, 1]).view(B, 1)
        return sd if not normalize else sd  # factorial(1)=1

    # general closed-shell determinant with spin split [↑...][↓...]
    n_spin = N // 2
    Psi_up = Psi[:, :n_spin, :]
    Psi_down = Psi[:, n_spin:, :]

    sign_u, log_u = _slogdet_batched(Psi_up)
    sign_d, log_d = _slogdet_batched(Psi_down)
    det_full = (sign_u * sign_d) * torch.exp(log_u + log_d)  # (B,)

    if normalize and n_spin > 1:
        det_full = det_full / math.factorial(n_spin)

    return det_full.view(B, 1)


# ===============================================================
# Backward-compatible simple RHF (orthonormal assumption)
# ===============================================================


def hartree_fock_2d(
    n_electrons: int,
    basis: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    Hcore: np.ndarray,
    two_body: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy RHF assuming orthonormal basis. Kept for compatibility.
    Prefer hartree_fock_closed_shell with S from compute_integrals().
    """
    n_occ = n_electrons // 2

    eigvals, C = np.linalg.eigh(Hcore)
    C_occ = C[:, :n_occ].copy()

    def density_matrix(Cocc: np.ndarray) -> np.ndarray:
        return 2.0 * (Cocc @ Cocc.T)

    D = density_matrix(C_occ)

    for iteration in range(max_iter):
        F = Hcore.copy()
        J = np.einsum("rs,prqs->pq", D, two_body, optimize=True)
        K = np.einsum("rs,prsq->pq", D, two_body, optimize=True)
        F += J - 0.5 * K

        eigvals_new, C_new = np.linalg.eigh(F)
        C_occ_new = C_new[:, :n_occ]
        D_new = density_matrix(C_occ_new)

        delta = np.linalg.norm(D_new - D)
        print(f"Iteration {iteration:3d}: ΔD = {delta:.3e}")
        if delta < tol:
            break
        C_occ = C_occ_new
        D = D_new

    orbital_energies = eigvals_new[:n_occ]
    E_hf = 0.5 * np.sum(D * (Hcore + F))
    print(f"Final HF Energy = {E_hf:.6f}")
    return C_occ, orbital_energies
