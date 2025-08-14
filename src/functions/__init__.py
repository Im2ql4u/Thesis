# src/functions/__init__.py

from .Slater_Determinant import (
    hermite_polynomial,
    harmonic_oscillator_wavefunction_1d,
    harmonic_oscillator_wavefunction_2d,
    initialize_harmonic_basis_2d,
    laplacian_2d,
    hartree_fock_2d,
    slater_determinant_closed_shell,
    evaluate_basis_functions_torch,
    evaluate_basis_functions_torch_batch_2d,
)

from .Neural_Networks import (
    psi_fn,
    compute_laplacian_fast,
    train_model,
)


from .Physics import (
    compute_coulomb_interaction,
    gaussian_interaction_2d,
    gaussian_interaction_potential_2d,
    compute_two_body_integrals_2d,
    one_electron_integral_2d,
)

__all__ = [
    # Slater_Determinant
    "hermite_polynomial",
    "harmonic_oscillator_wavefunction_1d",
    "harmonic_oscillator_wavefunction_2d",
    "initialize_harmonic_basis_2d",
    "laplacian_2d",
    "hartree_fock_2d",
    "slater_determinant_closed_shell",
    "evaluate_basis_functions_torch",
    "evaluate_basis_functions_torch_batch_2d",
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
