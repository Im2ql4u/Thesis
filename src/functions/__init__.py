# src/functions/__init__.py

# --- Slater_Determinant (Cartesian + FD + engines) ---
from .Energy import (
    estimate_energy_vmc,
    local_energy_autograd,
    potential_qdot_2d,
)

# --- Neural_Networks ---
from .Neural_Networks import (
    compute_laplacian_fast,
    psi_fn,
    train_model,
)

# --- Physics ---
from .Physics import (
    compute_coulomb_interaction,
    compute_two_body_integrals_2d,
    gaussian_interaction_2d,
    gaussian_interaction_potential_2d,
    one_electron_integral_2d,
)

# --- Plotting ---
from .Plotting import (
    construct_grid_configurations,
    make_mala_sample_fn,
    mirror_quadrants,
    plot_f_psi_sd_with_backflow,
    radial_two_body_density_2d,
    run_radial_map,
)
from .Slater_Determinant import (
    # Engines (basis-agnostic) + FD helper
    compute_integrals,  # build Hcore & two-electron Dirac
    evaluate_basis_functions_torch,
    evaluate_basis_functions_torch_batch_2d,
    fd_wavefunctions_on_grid,  # FD orbitals on grid (complex)
    harmonic_oscillator_wavefunction_1d,
    harmonic_oscillator_wavefunction_2d,
    hartree_fock_closed_shell,  # basis-agnostic RHF
    # Cartesian HO + helpers
    hermite_polynomial,
    initialize_harmonic_basis_2d,
    laplacian_2d,
    slater_determinant_closed_shell,  # basis-agnostic under params['basis']
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
    # Plotting
    "make_mala_sample_fn",
    "radial_two_body_density_2d",
    "run_radial_map",
    "mirror_quadrants",
    "construct_grid_configurations",
    "plot_f_psi_sd_with_backflow",
    # Energy
    "local_energy_autograd",
    "estimate_energy_vmc",
    "potential_qdot_2d",
]
