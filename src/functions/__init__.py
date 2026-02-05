# src/functions/__init__.py

# --- Slater_Determinant (Cartesian + FD + engines) ---
from .Analysis import run_compact_analysis

# --- Double Well System ---
from .DoubleWell import (
    double_well_coulomb_interaction,
    double_well_local_potential,
    double_well_potential,
    double_well_total_potential,
    evaluate_double_well_energy,
    local_energy_double_well,
    plot_energy_vs_separation,
    sample_double_well,
    scan_energy_vs_separation,
    time_evolve_double_well,
    train_double_well,
    two_well_basis_1d_torch,
    two_well_basis_2d_torch,
)
from .Energy import (
    evaluate_energy_vmc,
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
    make_mala_sample_fn,
    mirror_quadrants,
    plot_f_psi_sd_with_backflow,
    radial_two_body_density_2d,
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

# train_model_sr_energy might not exist in current Stochastic_Reconfiguration.py
try:
    from .Stochastic_Reconfiguration import train_model_sr_energy
except ImportError:
    train_model_sr_energy = None

from .analyze_shells import ShellAnalysisConfig, analyze_case, run_many

# from .qd_structure import (AnalyzeParams, ShellSplitParams, SingleRingParams, StructureFactorParams,
#     analyze_case, analyze_grid, summarize_results_table, pretty_top_transitions)

try:
    from .qd_structure import print_report, scan_cases
except ImportError:
    scan_cases = None
    print_report = None
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
    "radial_two_body_density_2d",
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
    "run_compact_analysis",
    # Neural Networks
    "psi_fn",
    "train_model",
    # Shell analysis
    "ShellAnalysisConfig",
    "analyze_case",
    "run_many",
    # QD structure
    "scan_cases",
    "print_report",
    # Double Well System
    "double_well_potential",
    "double_well_coulomb_interaction",
    "double_well_total_potential",
    "double_well_local_potential",
    "sample_double_well",
    "evaluate_double_well_energy",
    "train_double_well",
    "scan_energy_vs_separation",
    "plot_energy_vs_separation",
    "time_evolve_double_well",
    "local_energy_double_well",
    "two_well_basis_1d_torch",
    "two_well_basis_2d_torch",
]
