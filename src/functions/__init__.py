# src/functions/__init__.py

# --- Slater_Determinant (Cartesian + FD + engines) ---
from .Analysis import analyze_model_all, render_analysis_report
from .Energy import (
    evaluate_energy_vmc,
)

# --- Neural_Networks ---
from .Neural_Networks import (
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
    make_mala_sample_fn,
    mirror_quadrants,
    plot_f_psi_sd_with_backflow,
    run_radial_map,
)
from .Save_Model import (
    load_model_into,
    load_object,
    save_model,
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
from .Stochastic_Reconfiguration import train_model_sr_energy

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
    # Physics
    "compute_coulomb_interaction",
    "gaussian_interaction_2d",
    "gaussian_interaction_potential_2d",
    "compute_two_body_integrals_2d",
    "one_electron_integral_2d",
    # Plotting
    "make_mala_sample_fn",
    "radial_two_body_density_2d_fast",
    "mirror_quadrants",
    "plot_f_psi_sd_with_backflow",
    "run_radial_map",
    # Energy
    "evaluate_energy_vmc",
    # Save_Model
    "save_model",
    "load_object",
    "load_model_into",
    # Stochastic_Reconfiguration
    "train_model_sr_energy",
    # Analysis
    "analyze_model_all",
    "render_analysis_report",
    # Neural Networks
    "psi_fn",
    "train_model",
]
