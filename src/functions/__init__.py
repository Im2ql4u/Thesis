# src/functions/__init__.py

# --- Slater_Determinant (Cartesian + FD + engines) ---
from .Slater_Determinant import (
    # Cartesian HO + helpers
    hermite_polynomial,
    harmonic_oscillator_wavefunction_1d,
    harmonic_oscillator_wavefunction_2d,
    initialize_harmonic_basis_2d,
    laplacian_2d,
    slater_determinant_closed_shell,  # basis-agnostic under params['basis']
    evaluate_basis_functions_torch,
    evaluate_basis_functions_torch_batch_2d,
    # Engines (basis-agnostic) + FD helper
    compute_integrals,  # build Hcore & two-electron Dirac
    hartree_fock_closed_shell,  # basis-agnostic RHF
    fd_wavefunctions_on_grid,  # FD orbitals on grid (complex)
)

# --- Neural_Networks ---
from .Neural_Networks import (
    psi_fn,
    compute_laplacian_fast,
    train_model,
)

# --- Physics ---
from .Physics import (
    compute_coulomb_interaction,
    gaussian_interaction_2d,
    gaussian_interaction_potential_2d,
    compute_two_body_integrals_2d,
    one_electron_integral_2d,
)

__all__ = [
    # Slater_Determinant (existing)
    "hermite_polynomial",
    "harmonic_oscillator_wavefunction_1d",
    "harmonic_oscillator_wavefunction_2d",
    "initialize_harmonic_basis_2d",
    "laplacian_2d",
    "slater_determinant_closed_shell",
    "evaluate_basis_functions_torch",
    "evaluate_basis_functions_torch_batch_2d",
    # Slater_Determinant (engines)
    "compute_integrals",
    "hartree_fock_closed_shell",
    "fd_wavefunctions_on_grid",
    # Neural_Networks
    "psi_fn",
    "compute_laplacian_fast",
    "train_model",
    # Physics
    "compute_coulomb_interaction",
    "gaussian_interaction_2d",
    "gaussian_interaction_potential_2d",
    "compute_two_body_integrals_2d",
    "one_electron_integral_2d",
]
