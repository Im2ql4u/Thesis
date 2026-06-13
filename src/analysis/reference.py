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
    r_max: float = 0.0  # 0 -> auto
    n_grid: int = 40000

    energy: float = 0.0
    energy_cm: float = 0.0
    energy_rel: float = 0.0
    _r: np.ndarray = None  # type: ignore[assignment]
    _u: np.ndarray = None  # type: ignore[assignment]  # relative radial wavefn u(r)
    _logu: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        omega = float(self.omega)
        if self.r_max <= 0.0:
            # relative oscillator length ~ sqrt(2/omega); Coulomb widens it a bit.
            self.r_max = max(25.0, 18.0 / np.sqrt(omega))
        r, u, e_rel = _solve_relative_ground(omega, self.r_max, self.n_grid)
        self._r = r
        self._u = u
        self._logu = np.log(np.clip(u, 1e-300, None))
        self.energy_rel = float(e_rel)
        self.energy_cm = float(omega)  # 2D HO ground of CM (mass 2, freq omega)
        self.energy = self.energy_cm + self.energy_rel

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


def _solve_relative_ground(
    omega: float, r_max: float, n: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lowest s-wave (m=0) eigenpair of the relative Hamiltonian.

    Solves the 2D radial eigenproblem directly in u(r),

        -(1/r) d/dr( r du/dr ) + V(r) u = E_rel u,   V = 1/4 omega^2 r^2 + 1/r,

    under the radial measure int |u|^2 r dr, via a symmetric finite-volume scheme
    on the midpoint grid r_j = (j+1/2) h. This avoids the spurious 1/(4 r^2)
    singularity of the chi = sqrt(r) u substitution and represents the finite
    Coulomb cusp u'(0) correctly. Returns (r_grid, u(r) normalised to max=1, E_rel).
    """
    h = r_max / n
    j = np.arange(n, dtype=np.float64)
    r = (j + 0.5) * h
    v = 0.25 * omega**2 * r**2 + 1.0 / r
    m = (j + 0.5) * h**2  # diagonal mass (cell measure ~ r_j h * h)

    # K u = E M u with K symmetric tridiagonal (finite-volume fluxes).
    # diag_j = (r_{j+1/2}+r_{j-1/2})/h + V_j m_j ; r_{j+1/2}=(j+1)h, r_{j-1/2}=j h.
    k_diag = (2.0 * j + 1.0) + v * m
    k_off = -(j[:-1] + 1.0)  # K_{j,j+1} = -r_{j+1/2}/h = -(j+1)

    # Symmetrise: A = M^{-1/2} K M^{-1/2}
    a_diag = k_diag / m
    a_off = k_off / np.sqrt(m[:-1] * m[1:])

    if _HAVE_SCIPY:
        evals, evecs = eigh_tridiagonal(a_diag, a_off, select="i", select_range=(0, 0))
        e_rel = float(evals[0])
        w = evecs[:, 0]
    else:  # pragma: no cover
        A = np.diag(a_diag) + np.diag(a_off, 1) + np.diag(a_off, -1)
        evals, evecs = np.linalg.eigh(A)
        e_rel = float(evals[0])
        w = evecs[:, 0]

    u = w / np.sqrt(m)  # undo the symmetrising transform
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    u = u / np.max(u)  # convenience normalisation (shape only)
    return r, u, e_rel
