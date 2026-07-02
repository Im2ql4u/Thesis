"""Exact / reference wavefunctions for the 2-electron parabolic quantum dot.

For two electrons in a 2D harmonic trap with Coulomb interaction the Hamiltonian
separates into centre-of-mass (CM) and relative coordinates:

    H = sum_i [ -1/2 nabla_i^2 + 1/2 omega^2 r_i^2 ] + 1/r_12
      = H_CM(R) + H_rel(r),   R = (r1+r2)/2,  r = r1 - r2.

H_CM is a 2D harmonic oscillator (mass M=2, frequency omega) with ground energy
E_CM = omega and ground state psi_CM(R) ~ exp(-omega R^2).

H_rel = -nabla_r^2 + 1/4 omega^2 r^2 + 1/r (reduced mass mu=1/2). Its lowest
s-wave (m=0) state is found here by finite-difference diagonalisation of the
radial equation, valid for *any* omega. For the special value omega=1 the
problem closes analytically:

    E = 3.0,  u_rel(r) = (1 + r) exp(-r^2/4),  psi ~ exp(-(r1^2+r2^2)/2) (1 + r12),

which is used to validate the numerical solver.

The "exact correlation factor" (the Jastrow a network must learn on top of the
non-interacting HO core exp(-omega(r1^2+r2^2)/2)) is

    J_exact(r) = log u_rel(r) + omega r^2 / 4,

with J_exact(r) = log(1 + r) at omega=1 and cusp slope dJ/dr|_{r->0} = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.linalg import eigh_tridiagonal

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fallback if scipy missing
    _HAVE_SCIPY = False


@dataclass
class TwoElectronExact:
    """Exact ground state of the 2-electron 2D parabolic dot for a given omega.

    Attributes are populated on construction by solving the relative radial ODE.
    All quantities are in oscillator (Hartree) units consistent with the repo
    Hamiltonian H = sum -1/2 nabla^2 + 1/2 omega^2 r^2 + sum 1/r_ij.
    """

    omega: float
    lam: float = 1.0  # Coulomb strength (1.0 = physical); enables finite-difference d/dlam responses
    r_max: float = 0.0  # 0 -> auto
    n_grid: int = 40000
    n_states: int = 1  # number of relative s-wave states to solve (>=1; excited states enable overlaps)

    energy: float = 0.0
    energy_cm: float = 0.0
    energy_rel: float = 0.0
    energies_rel_all: np.ndarray = None  # type: ignore[assignment]  # (n_states,) relative eigenvalues
    _r: np.ndarray = None  # type: ignore[assignment]
    _u: np.ndarray = None  # type: ignore[assignment]  # relative radial wavefn u_0(r)
    _U: np.ndarray = None  # type: ignore[assignment]  # (n_grid, n_states) relative radial states
    _logu: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        omega = float(self.omega)
        if self.r_max <= 0.0:
            # relative oscillator length ~ sqrt(2/omega); Coulomb widens it a bit.
            self.r_max = max(25.0, 18.0 / np.sqrt(omega))
        r, U, evals = _solve_relative_states(
            omega, float(self.lam), self.r_max, self.n_grid, self.n_states
        )
        self._r = r
        self._U = U
        self._u = U[:, 0]
        self._logu = np.log(np.clip(self._u, 1e-300, None))
        self.energies_rel_all = np.asarray(evals, dtype=np.float64)
        self.energy_rel = float(evals[0])
        self.energy_cm = float(omega)  # 2D HO ground of CM (mass 2, freq omega)
        self.energy = self.energy_cm + self.energy_rel

    def relative_excitation_ratio(self, r12: np.ndarray, n: int) -> np.ndarray:
        """Psi_n / Psi_0 as a function of pair distance (the CM factor cancels).

        This is the log-space tangent direction that admixing a bit of the n-th relative
        excitation adds to log|Psi_0| (delta log|Psi| ~ eps * Psi_n/Psi_0), used to name the
        network's tangent modes as physical excitations."""
        if self._U is None or n >= self._U.shape[1]:
            raise ValueError(f"excited state {n} not solved (n_states={self.n_states})")
        r12 = np.asarray(r12, dtype=np.float64)
        un = np.interp(r12, self._r, self._U[:, n])
        u0 = np.interp(r12, self._r, self._u)
        return un / np.clip(np.abs(u0), 1e-300, None) * np.sign(u0 + 1e-300)

    # ---- relative correlation factor (the "exact Jastrow") ----
    def jastrow_log(self, r12: np.ndarray) -> np.ndarray:
        """J_exact(r) = log u_rel(r) + omega r^2/4, the correlation beyond the HO core."""
        r12 = np.asarray(r12, dtype=np.float64)
        logu = np.interp(r12, self._r, self._logu)
        return logu + 0.25 * float(self.omega) * r12**2

    def jastrow_cusp_slope(self) -> float:
        """dJ/dr as r->0 (Kato cusp coefficient), read off the radial grid.

        Near r=0, J(r) = log u(r) + omega r^2/4 with d/dr(omega r^2/4)|_0 = 0,
        so the cusp is d(log u)/dr at the origin. Exact value is 1 at omega=1.
        """
        return float((self._logu[1] - self._logu[0]) / (self._r[1] - self._r[0]))

    # ---- full exact log|Psi| on particle configurations ----
    def log_psi(self, x: np.ndarray) -> np.ndarray:
        """log|Psi_exact| (up to an additive constant) for configs x: (B, 2, d).

        log|Psi| = -omega R^2 + log u_rel(r),  R^2 = |r1+r2|^2/4, r = |r1-r2|.
        Constant (normalisation) is irrelevant for overlaps / shapes.
        """
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[1] == 2, "TwoElectronExact.log_psi expects N=2 configurations"
        r1, r2 = x[:, 0, :], x[:, 1, :]
        R2 = ((r1 + r2) ** 2).sum(-1) / 4.0
        r = np.sqrt(((r1 - r2) ** 2).sum(-1) + 1e-300)
        logu = np.interp(r, self._r, self._logu)
        return -float(self.omega) * R2 + logu


def _solve_relative_states(
    omega: float, lam: float, r_max: float, n: int, k: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The k lowest s-wave (m=0) eigenpairs of the relative Hamiltonian.

    Solves the 2D radial eigenproblem directly in u(r),

        -(1/r) d/dr( r du/dr ) + V(r) u = E_rel u,   V = 1/4 omega^2 r^2 + lam/r,

    under the radial measure int |u|^2 r dr, via a symmetric finite-volume scheme
    on the midpoint grid r_j = (j+1/2) h. This avoids the spurious 1/(4 r^2)
    singularity of the chi = sqrt(r) u substitution and represents the finite
    Coulomb cusp u'(0) correctly. `lam` scales the Coulomb term (1.0 = physical),
    enabling finite-difference d/dlam responses. Returns (r_grid, U (n,k) each
    normalised to max|.|=1, E_rel (k,)).
    """
    h = r_max / n
    j = np.arange(n, dtype=np.float64)
    r = (j + 0.5) * h
    v = 0.25 * omega**2 * r**2 + lam / r
    m = (j + 0.5) * h**2  # diagonal mass (cell measure ~ r_j h * h)

    # K u = E M u with K symmetric tridiagonal (finite-volume fluxes).
    # diag_j = (r_{j+1/2}+r_{j-1/2})/h + V_j m_j ; r_{j+1/2}=(j+1)h, r_{j-1/2}=j h.
    k_diag = (2.0 * j + 1.0) + v * m
    k_off = -(j[:-1] + 1.0)  # K_{j,j+1} = -r_{j+1/2}/h = -(j+1)

    # Symmetrise: A = M^{-1/2} K M^{-1/2}
    a_diag = k_diag / m
    a_off = k_off / np.sqrt(m[:-1] * m[1:])

    if _HAVE_SCIPY:
        evals, evecs = eigh_tridiagonal(a_diag, a_off, select="i", select_range=(0, k - 1))
    else:  # pragma: no cover
        A = np.diag(a_diag) + np.diag(a_off, 1) + np.diag(a_off, -1)
        evals_all, evecs_all = np.linalg.eigh(A)
        evals, evecs = evals_all[:k], evecs_all[:, :k]

    U = evecs / np.sqrt(m)[:, None]  # undo the symmetrising transform, (n, k)
    for c in range(U.shape[1]):  # sign + shape normalisation per state
        col = U[:, c]
        if col[np.argmax(np.abs(col))] < 0:
            col = -col
        U[:, c] = col / np.max(np.abs(col))
    return r, U, np.asarray(evals[:k], dtype=np.float64)


def _solve_relative_ground(
    omega: float, r_max: float, n: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Backward-compatible wrapper: lowest relative s-wave state at physical Coulomb (lam=1)."""
    r, U, evals = _solve_relative_states(omega, 1.0, r_max, n, 1)
    return r, U[:, 0], float(evals[0])
