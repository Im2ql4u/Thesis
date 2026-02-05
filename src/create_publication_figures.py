"""
Generate Individual Publication-Quality Plots
=============================================

Creates separate high-quality figures for each analysis:
1. Energy decay curves (individual separations)
2. VMC energy comparison
3. Decay rate analysis
4. Summary figure

Uses thesis_style.mplstyle for consistent appearance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import json

# Setup path and style
STYLE_PATH = Path(__file__).parent / "Thesis_style.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))

# Override for individual figures
rcParams.update({
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.grid': True,
})

RESULTS_DIR = Path(__file__).parent.parent / "results" / "double_well"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(RESULTS_DIR / "imaginary_time_final.json") as f:
    imag_time_results = json.load(f)

try:
    with open(RESULTS_DIR / "comparison_results.json") as f:
        comparison_results = json.load(f)
except:
    comparison_results = None


def expected_ground_energy(d):
    """Expected E₀ based on well separation."""
    if d == 0:
        return 3.0
    r_eff = np.sqrt(d**2 + 2.0)
    return 2.0 + 1.0 / r_eff


# ========================================
# Figure 1: Individual E(τ) decay curves
# ========================================

def create_individual_decay_plots():
    """Create separate figure for each well separation showing E(τ) decay."""
    
    separations = [float(d) for d in imag_time_results["separations"]]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(separations)))
    
    for i, d in enumerate(separations):
        d_str = str(d)
        if d_str not in imag_time_results["time_evolution"]:
            continue
            
        data = imag_time_results["time_evolution"][d_str]
        fit = imag_time_results["fits"].get(d_str, {})
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Extract data
        taus = np.array([r["tau"] for r in data["trajectory"]])
        Es = np.array([r["E"] for r in data["trajectory"]])
        errs = np.array([r["E_err"] for r in data["trajectory"]])
        
        # Plot data
        ax.errorbar(taus, Es, yerr=errs, fmt='o', markersize=6, color=colors[i],
                    alpha=0.7, capsize=3, label='VMC data')
        
        # Plot fit
        if fit.get("success"):
            E0 = fit["E0"]
            dE = fit["dE"]
            gamma = fit["gamma"]
            
            tau_fit = np.linspace(0, taus.max(), 200)
            E_fit = E0 + dE * np.exp(-gamma * tau_fit)
            
            ax.plot(tau_fit, E_fit, '--', linewidth=2.5, color='darkred',
                    label=f'Fit: $E_0$ + {dE:.3f}·exp(-{gamma:.2f}τ)')
            
            # Reference line
            ax.axhline(E0, color='gray', linestyle=':', alpha=0.6,
                       label=f'E₀(fit) = {E0:.4f}')
        
        E_ref = expected_ground_energy(d)
        ax.axhline(E_ref, color='black', linestyle='--', alpha=0.5,
                   label=f'E₀(ref) = {E_ref:.4f}')
        
        ax.set_xlabel('Imaginary time τ', fontsize=14)
        ax.set_ylabel('Energy E(τ)', fontsize=14)
        ax.set_title(f'Imaginary Time Evolution: d = {d:.1f}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        
        # Set y limits
        y_min = min(Es.min(), E_ref) - 0.05
        y_max = Es.max() + 0.05
        ax.set_ylim(y_min, y_max)
        
        # Add annotation
        if fit.get("success"):
            text = f"γ = {gamma:.3f} → Δε = {gamma/2:.3f}"
            ax.text(0.95, 0.5, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        path = FIGURES_DIR / f"energy_decay_d{d:.0f}.png"
        plt.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
        print(f"📊 Saved: {path}")
        plt.close()


# ========================================
# Figure 2: All E(τ) curves together
# ========================================

def create_combined_decay_plot():
    """Create figure showing all decay curves together."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    separations = [float(d) for d in imag_time_results["separations"]]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(separations)))
    
    for i, d in enumerate(separations):
        d_str = str(d)
        if d_str not in imag_time_results["time_evolution"]:
            continue
            
        data = imag_time_results["time_evolution"][d_str]
        fit = imag_time_results["fits"].get(d_str, {})
        
        taus = np.array([r["tau"] for r in data["trajectory"]])
        Es = np.array([r["E"] for r in data["trajectory"]])
        errs = np.array([r["E_err"] for r in data["trajectory"]])
        
        # Normalize tau for comparison
        tau_norm = taus / data["tau_max"]
        
        ax.errorbar(tau_norm, Es, yerr=errs, fmt='o-', markersize=4, 
                    color=colors[i], linewidth=1.5, alpha=0.8,
                    label=f'd={d:.0f}', capsize=2)
        
        # Add fit curve
        if fit.get("success"):
            tau_fit = np.linspace(0, 1, 100) * data["tau_max"]
            E_fit = fit["E0"] + fit["dE"] * np.exp(-fit["gamma"] * tau_fit)
            tau_fit_norm = tau_fit / data["tau_max"]
            ax.plot(tau_fit_norm, E_fit, '--', color=colors[i], alpha=0.5, linewidth=1)
    
    ax.axhline(2.0, color='gray', linestyle=':', alpha=0.5, label='E=2 (separated)')
    ax.axhline(3.0, color='gray', linestyle=':', alpha=0.5, label='E=3 (d=0)')
    
    ax.set_xlabel('Normalized imaginary time τ/τ_max', fontsize=14)
    ax.set_ylabel('Energy E(τ)', fontsize=14)
    ax.set_title('Imaginary Time Evolution: All Well Separations', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=10)
    ax.set_ylim(1.95, 3.3)
    
    plt.tight_layout()
    
    path = FIGURES_DIR / "energy_decay_all.png"
    plt.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"📊 Saved: {path}")
    plt.close()


# ========================================
# Figure 3: Ground state energy vs d
# ========================================

def create_ground_state_plot():
    """Create figure showing ground state energy vs well separation."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    separations = [float(d) for d in imag_time_results["separations"]]
    
    # Left: Absolute energies
    ax = axes[0]
    
    E_ref = [expected_ground_energy(d) for d in separations]
    E_vmc = [imag_time_results["vmc"][str(d)]["E"] for d in separations if str(d) in imag_time_results["vmc"]]
    E_fit = [imag_time_results["fits"][str(d)]["E0"] for d in separations 
             if str(d) in imag_time_results["fits"] and imag_time_results["fits"][str(d)].get("success")]
    d_fit = [d for d in separations 
             if str(d) in imag_time_results["fits"] and imag_time_results["fits"][str(d)].get("success")]
    
    d_dense = np.linspace(0, max(separations), 100)
    E_dense = [expected_ground_energy(d) for d in d_dense]
    
    ax.plot(d_dense, E_dense, 'k-', linewidth=2, label='Theory', alpha=0.7)
    ax.scatter(separations, E_vmc, s=100, marker='o', color='blue', 
               label='VMC', zorder=5, edgecolors='black', linewidths=1)
    ax.scatter(d_fit, E_fit, s=80, marker='s', color='red', 
               label='Imag. time fit', zorder=4, edgecolors='black', linewidths=1)
    
    ax.axhline(2.0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(3.0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Well Separation d', fontsize=14)
    ax.set_ylabel('Ground State Energy E₀', fontsize=14)
    ax.set_title('Ground State Energy vs Separation', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(1.9, 3.2)
    
    # Right: Error
    ax = axes[1]
    
    errors = [(E_vmc[i] - E_ref[i]) / E_ref[i] * 100 for i in range(len(separations))]
    
    colors = ['green' if abs(e) < 1 else 'orange' if abs(e) < 2 else 'red' for e in errors]
    ax.bar(range(len(separations)), errors, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(0, color='black', linewidth=1)
    ax.axhspan(-1, 1, alpha=0.2, color='green')
    
    ax.set_xticks(range(len(separations)))
    ax.set_xticklabels([f'{d:.0f}' for d in separations])
    ax.set_xlabel('Well Separation d', fontsize=14)
    ax.set_ylabel('Relative Error (%)', fontsize=14)
    ax.set_title('VMC Energy Accuracy', fontsize=16, fontweight='bold')
    ax.set_ylim(-3, 5)
    
    plt.tight_layout()
    
    path = FIGURES_DIR / "ground_state_energy.png"
    plt.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"📊 Saved: {path}")
    plt.close()


# ========================================
# Figure 4: Decay rate and energy gap
# ========================================

def create_decay_rate_plot():
    """Create figure showing decay rate and energy gap vs separation."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    separations = [float(d) for d in imag_time_results["separations"]]
    
    # Extract successful fits
    d_vals = []
    gamma_vals = []
    gamma_errs = []
    gap_vals = []
    
    for d in separations:
        d_str = str(d)
        fit = imag_time_results["fits"].get(d_str, {})
        if fit.get("success"):
            d_vals.append(d)
            gamma_vals.append(fit["gamma"])
            gamma_errs.append(fit.get("gamma_err", 0))
            gap_vals.append(fit["gap"])
    
    # Left: Decay rate
    ax = axes[0]
    ax.errorbar(d_vals, gamma_vals, yerr=gamma_errs, fmt='o-', markersize=10,
                linewidth=2, color='blue', capsize=4, elinewidth=2)
    ax.axhline(4.0, color='gray', linestyle='--', alpha=0.6, label='γ = 4')
    
    ax.set_xlabel('Well Separation d', fontsize=14)
    ax.set_ylabel('Decay Rate γ', fontsize=14)
    ax.set_title('Imaginary Time Decay Rate', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(3, 6)
    
    # Right: Energy gap
    ax = axes[1]
    ax.plot(d_vals, gap_vals, 'o-', markersize=10, linewidth=2, color='red')
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.6, label='Δε = 2')
    
    ax.set_xlabel('Well Separation d', fontsize=14)
    ax.set_ylabel('Energy Gap Δε = γ/2', fontsize=14)
    ax.set_title('First Excitation Energy Gap', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(1.5, 3)
    
    plt.tight_layout()
    
    path = FIGURES_DIR / "decay_rate_gap.png"
    plt.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"📊 Saved: {path}")
    plt.close()


# ========================================
# Figure 5: Physics summary
# ========================================

def create_summary_figure():
    """Create comprehensive summary figure."""
    
    fig = plt.figure(figsize=(16, 12))
    
    separations = [float(d) for d in imag_time_results["separations"]]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(separations)))
    
    # Top left: E(τ) curves
    ax1 = fig.add_subplot(2, 2, 1)
    for i, d in enumerate(separations[:4]):  # Show first 4
        d_str = str(d)
        if d_str not in imag_time_results["time_evolution"]:
            continue
        data = imag_time_results["time_evolution"][d_str]
        taus = np.array([r["tau"] for r in data["trajectory"]])
        Es = np.array([r["E"] for r in data["trajectory"]])
        ax1.plot(taus, Es, 'o-', markersize=4, linewidth=1.5, color=colors[i],
                 alpha=0.8, label=f'd={d:.0f}')
    
    ax1.set_xlabel('τ (imaginary time)')
    ax1.set_ylabel('E(τ)')
    ax1.set_title('(a) Energy Decay Curves')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(2.0, 3.3)
    
    # Top right: E₀ vs d
    ax2 = fig.add_subplot(2, 2, 2)
    E_vmc = [imag_time_results["vmc"][str(d)]["E"] for d in separations]
    d_dense = np.linspace(0, max(separations), 100)
    E_dense = [expected_ground_energy(d) for d in d_dense]
    
    ax2.plot(d_dense, E_dense, 'k-', linewidth=2, label='Theory')
    ax2.scatter(separations, E_vmc, s=80, c=colors, marker='o', zorder=5, edgecolors='black')
    ax2.set_xlabel('Well Separation d')
    ax2.set_ylabel('E₀')
    ax2.set_title('(b) Ground State Energy')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Bottom left: γ vs d
    ax3 = fig.add_subplot(2, 2, 3)
    d_fit = []
    gamma_fit = []
    for d in separations:
        fit = imag_time_results["fits"].get(str(d), {})
        if fit.get("success"):
            d_fit.append(d)
            gamma_fit.append(fit["gamma"])
    
    ax3.plot(d_fit, gamma_fit, 'o-', markersize=10, linewidth=2, color='blue')
    ax3.axhline(4.0, color='gray', linestyle='--', alpha=0.6)
    ax3.set_xlabel('Well Separation d')
    ax3.set_ylabel('Decay Rate γ')
    ax3.set_title('(c) Decay Rate')
    
    # Bottom right: Summary text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary = """
    IMAGINARY TIME EVOLUTION
    ========================
    
    Physics:
    • ψ(τ) = Σₙ cₙ exp(-Eₙ τ) φₙ
    • E(τ) = E₀ + ΔE × exp(-γτ)
    • γ = 2×(E₁-E₀) = 2×Δε
    
    Key Results:
    • d=0: E₀ = 3.0 (Taut exact)
    • d→∞: E₀ → 2.0 (separated)
    • γ ≈ 4 → Δε ≈ 2
    
    Ansätze Comparison:
    • Slater-Jastrow: Fast, <1% error
    • CTNN+PINN: Neural network
    """
    
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    path = FIGURES_DIR / "physics_summary.png"
    plt.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"📊 Saved: {path}")
    plt.close()


# ========================================
# Main
# ========================================

if __name__ == "__main__":
    print("Creating individual publication-quality figures...\n")
    
    create_individual_decay_plots()
    create_combined_decay_plot()
    create_ground_state_plot()
    create_decay_rate_plot()
    create_summary_figure()
    
    print(f"\n✓ All figures saved to {FIGURES_DIR}")
