"""
Proper Imaginary Time Evolution with Slater-Jastrow Wavefunction
================================================================

Focus: Get the CORRECT exponential decay E(τ) = E₀ + ΔE·exp(-Δε·τ)

Physics:
- ψ(τ) = Σₙ cₙ exp(-Eₙ τ) φₙ
- For large τ: ψ(τ) → c₀ exp(-E₀ τ) φ₀ + c₁ exp(-E₁ τ) φ₁
- Energy: E(τ) ≈ E₀ + (E₁ - E₀)|c₁/c₀|² exp(-2(E₁-E₀)τ)
- Exponential decay rate = 2×(energy gap)

Wavefunction: Slater Determinant × Jastrow Factor
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")
DTYPE = torch.float64
OMEGA = 1.0

RESULTS_DIR = Path(__file__).parent.parent / "results" / "double_well"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Slater-Jastrow Wavefunction
# ============================================================


class SlaterJastrow(nn.Module):
    """
    ψ = Det[φᵢ(rⱼ)] × exp(J(r₁₂))

    For 2 electrons in double well:
    - d > 0: φ₁ centered at left, φ₂ centered at right
    - d = 0: Use s and p orbitals instead (different symmetry)

    Slater: Det = φ₁(r₁)φ₂(r₂) - φ₁(r₂)φ₂(r₁) (antisymmetric)
    Jastrow: J = a·r₁₂/(1 + b·r₁₂) (correlation)
    """

    name = "SlaterJastrow"

    def __init__(self, well_centers: torch.Tensor):
        super().__init__()
        self.register_buffer("well_centers", well_centers)

        # Check if single well (d=0)
        self.single_well = torch.allclose(well_centers[0], well_centers[1])

        # Orbital width (learnable)
        self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))

        # Jastrow parameters
        self.jastrow_a = nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
        self.jastrow_b = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))

    def orbital_s(self, r: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """s-orbital: exp(-α|r - R|²)"""
        alpha = torch.exp(self.log_alpha) * OMEGA / 2
        disp = r - center
        return torch.exp(-alpha * (disp**2).sum(dim=-1))

    def orbital_p(self, r: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """p-orbital: x × exp(-α|r - R|²) for antisymmetry"""
        alpha = torch.exp(self.log_alpha) * OMEGA / 2
        disp = r - center
        return disp[..., 0] * torch.exp(-alpha * (disp**2).sum(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, 2) - B samples, 2 particles, 2D
        Returns: log|ψ|
        """
        B = x.shape[0]
        r1, r2 = x[:, 0], x[:, 1]  # (B, 2)
        R1, R2 = self.well_centers[0], self.well_centers[1]  # (2,)

        if self.single_well:
            # d=0: Use s and p orbitals for proper antisymmetry
            center = R1
            phi_s_r1 = self.orbital_s(r1, center)
            phi_s_r2 = self.orbital_s(r2, center)
            phi_p_r1 = self.orbital_p(r1, center)
            phi_p_r2 = self.orbital_p(r2, center)

            # Slater: |s(r1) p(r1)| = s(r1)p(r2) - s(r2)p(r1)
            #         |s(r2) p(r2)|
            det = phi_s_r1 * phi_p_r2 - phi_s_r2 * phi_p_r1
        else:
            # d > 0: Left and right orbitals
            phi1_r1 = self.orbital_s(r1, R1)
            phi2_r2 = self.orbital_s(r2, R2)
            phi1_r2 = self.orbital_s(r2, R1)
            phi2_r1 = self.orbital_s(r1, R2)

            det = phi1_r1 * phi2_r2 - phi1_r2 * phi2_r1

        # Handle sign (log of absolute value)
        log_det = torch.log(torch.abs(det) + 1e-10)

        # Jastrow factor
        r12 = torch.norm(r1 - r2, dim=-1)
        a = torch.abs(self.jastrow_a)
        b = torch.abs(self.jastrow_b) + 0.1
        jastrow = a * r12 / (1 + b * r12)

        return log_det + jastrow


class SlaterJastrowNN(nn.Module):
    """Slater determinant with neural network Jastrow."""

    name = "SlaterJastrow-NN"

    def __init__(self, well_centers: torch.Tensor, hidden: int = 32):
        super().__init__()
        self.register_buffer("well_centers", well_centers)
        self.single_well = torch.allclose(well_centers[0], well_centers[1])
        self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))

        # Neural Jastrow
        self.jastrow_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.jastrow_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def orbital_s(self, r: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha) * OMEGA / 2
        disp = r - center
        return torch.exp(-alpha * (disp**2).sum(dim=-1))

    def orbital_p(self, r: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha) * OMEGA / 2
        disp = r - center
        return disp[..., 0] * torch.exp(-alpha * (disp**2).sum(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        r1, r2 = x[:, 0], x[:, 1]
        R1, R2 = self.well_centers[0], self.well_centers[1]

        if self.single_well:
            center = R1
            phi_s_r1 = self.orbital_s(r1, center)
            phi_s_r2 = self.orbital_s(r2, center)
            phi_p_r1 = self.orbital_p(r1, center)
            phi_p_r2 = self.orbital_p(r2, center)
            det = phi_s_r1 * phi_p_r2 - phi_s_r2 * phi_p_r1
        else:
            phi1_r1 = self.orbital_s(r1, R1)
            phi2_r2 = self.orbital_s(r2, R2)
            phi1_r2 = self.orbital_s(r2, R1)
            phi2_r1 = self.orbital_s(r1, R2)
            det = phi1_r1 * phi2_r2 - phi1_r2 * phi2_r1

        log_det = torch.log(torch.abs(det) + 1e-10)

        r12 = torch.norm(r1 - r2, dim=-1, keepdim=True)
        jastrow = self.jastrow_net(r12).squeeze(-1)

        return log_det + jastrow


class TimeEvolvingSlaterJastrow(nn.Module):
    """Slater-Jastrow with τ-dependent parameters for imaginary time."""

    name = "TimeEvolvingSlaterJastrow"

    def __init__(self, well_centers: torch.Tensor):
        super().__init__()
        self.register_buffer("well_centers", well_centers)

        # Check if single well (d=0)
        self.single_well = torch.allclose(well_centers[0], well_centers[1])

        # τ-dependent orbital width: starts wide (excited), narrows (ground)
        self.log_alpha_0 = nn.Parameter(torch.tensor(-0.3, dtype=DTYPE))  # τ=0
        self.log_alpha_inf = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))  # τ→∞
        self.alpha_rate = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))

        # Jastrow (also τ-dependent)
        self.jastrow_a_0 = nn.Parameter(torch.tensor(0.3, dtype=DTYPE))
        self.jastrow_a_inf = nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
        self.jastrow_b = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))

    def orbital_s(self, r: torch.Tensor, center: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        disp = r - center
        return torch.exp(-alpha.unsqueeze(-1) * (disp**2).sum(dim=-1))

    def orbital_p(self, r: torch.Tensor, center: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        disp = r - center
        return disp[..., 0] * torch.exp(-alpha.unsqueeze(-1) * (disp**2).sum(dim=-1))

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tau_val = tau.expand(B) if tau.dim() == 0 else tau.view(-1)

        # Interpolate parameters
        rate = torch.abs(self.alpha_rate)
        blend = 1 - torch.exp(-rate * tau_val)

        alpha_0 = torch.exp(self.log_alpha_0) * OMEGA / 2
        alpha_inf = torch.exp(self.log_alpha_inf) * OMEGA / 2
        alpha = alpha_0 + blend * (alpha_inf - alpha_0)

        a_0 = torch.abs(self.jastrow_a_0)
        a_inf = torch.abs(self.jastrow_a_inf)
        a = a_0 + blend * (a_inf - a_0)
        b = torch.abs(self.jastrow_b) + 0.1

        r1, r2 = x[:, 0], x[:, 1]
        R1, R2 = self.well_centers[0], self.well_centers[1]

        if self.single_well:
            # d=0: Use s and p orbitals
            center = R1
            phi_s_r1 = self.orbital_s(r1, center, alpha)
            phi_s_r2 = self.orbital_s(r2, center, alpha)
            phi_p_r1 = self.orbital_p(r1, center, alpha)
            phi_p_r2 = self.orbital_p(r2, center, alpha)
            det = phi_s_r1 * phi_p_r2 - phi_s_r2 * phi_p_r1
        else:
            # Slater determinant
            phi1_r1 = self.orbital_s(r1, R1, alpha)
            phi2_r2 = self.orbital_s(r2, R2, alpha)
            phi1_r2 = self.orbital_s(r2, R1, alpha)
            phi2_r1 = self.orbital_s(r1, R2, alpha)
            det = phi1_r1 * phi2_r2 - phi1_r2 * phi2_r1

        log_det = torch.log(torch.abs(det) + 1e-10)

        # Jastrow
        r12 = torch.norm(r1 - r2, dim=-1)
        jastrow = a * r12 / (1 + b * r12)

        return log_det + jastrow


# ============================================================
# MCMC Sampling
# ============================================================


def mcmc_sample(
    log_psi_fn,
    n_samples: int,
    well_centers: torch.Tensor,
    n_warmup: int = 500,
    step_size: float = 0.5,
) -> torch.Tensor:
    """Metropolis-Hastings sampling from |ψ|²."""
    # Initialize near well centers
    x = torch.randn(n_samples, 2, 2, dtype=DTYPE, device=DEVICE) * 0.5
    x = x + well_centers.unsqueeze(0)

    log_prob = 2 * log_psi_fn(x)
    n_accept = 0

    for i in range(n_warmup):
        x_new = x + step_size * torch.randn_like(x)
        log_prob_new = 2 * log_psi_fn(x_new)

        log_ratio = log_prob_new - log_prob
        accept = torch.log(torch.rand(n_samples, dtype=DTYPE)) < log_ratio

        x = torch.where(accept.view(-1, 1, 1), x_new, x)
        log_prob = torch.where(accept, log_prob_new, log_prob)
        n_accept += accept.sum().item()

    return x


# ============================================================
# Local Energy
# ============================================================


def local_energy(x: torch.Tensor, log_psi_fn, well_centers: torch.Tensor) -> dict:
    """Compute local energy E_L = Hψ/ψ."""
    B, N, dim = x.shape
    x = x.requires_grad_(True)

    log_psi = log_psi_fn(x)
    grad = torch.autograd.grad(log_psi.sum(), x, create_graph=True)[0]

    # Laplacian via trace of Hessian
    laplacian = torch.zeros(B, dtype=DTYPE, device=DEVICE)
    for i in range(N):
        for j in range(dim):
            g2 = torch.autograd.grad(grad[:, i, j].sum(), x, retain_graph=True, create_graph=True)[
                0
            ]
            laplacian = laplacian + g2[:, i, j]

    grad_sq = (grad**2).sum(dim=(1, 2))
    T = -0.5 * (laplacian + grad_sq)

    # Harmonic potential
    disp = x - well_centers.unsqueeze(0)
    r2 = (disp**2).sum(dim=(1, 2))
    V_harm = 0.5 * OMEGA**2 * r2

    # Coulomb
    r12 = torch.norm(x[:, 0] - x[:, 1], dim=-1)
    V_coul = 1.0 / (r12 + 1e-8)

    E_L = T + V_harm + V_coul

    return {
        "E_L": E_L.detach(),
        "T": T.detach(),
        "V_harm": V_harm.detach(),
        "V_coul": V_coul.detach(),
        "r12": r12.detach(),
    }


# ============================================================
# VMC Ground State Training
# ============================================================


def train_vmc(
    well_sep: float, wf_class, n_epochs: int = 2000, n_samples: int = 500, verbose: bool = True
):
    """Train VMC to find ground state."""
    well_centers = torch.zeros(2, 2, dtype=DTYPE, device=DEVICE)
    if well_sep > 0:
        well_centers[0, 0] = -well_sep / 2
        well_centers[1, 0] = +well_sep / 2

    wf = wf_class(well_centers).to(DTYPE)
    optimizer = torch.optim.Adam(wf.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    energies = []
    best_E, best_state = float("inf"), None

    for epoch in range(n_epochs):
        with torch.no_grad():
            x = mcmc_sample(wf, n_samples, well_centers, n_warmup=200)

        obs = local_energy(x, wf, well_centers)
        E_mean = obs["E_L"].mean()
        E_var = obs["E_L"].var()

        # Score function gradient
        x_grad = x.clone().requires_grad_(True)
        log_psi = wf(x_grad)
        baseline = E_mean.detach()
        loss = ((obs["E_L"] - baseline) * log_psi).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        E_val = E_mean.item()
        energies.append(E_val)

        if E_val < best_E:
            best_E = E_val
            best_state = {k: v.clone() for k, v in wf.state_dict().items()}

        if verbose and epoch % 400 == 0:
            err = np.sqrt(E_var.item() / n_samples)
            print(f"    Epoch {epoch:4d}: E = {E_val:.4f} ± {err:.4f}")

    if best_state:
        wf.load_state_dict(best_state)

    return wf, well_centers, np.array(energies)


def evaluate_vmc(wf, well_centers, n_samples: int = 5000) -> dict:
    """Final evaluation of converged wavefunction."""
    with torch.no_grad():
        x = mcmc_sample(wf, n_samples, well_centers, n_warmup=800)

    obs = local_energy(x, wf, well_centers)
    N = np.sqrt(n_samples)

    return {
        "E": obs["E_L"].mean().item(),
        "E_err": obs["E_L"].std().item() / N,
        "T": obs["T"].mean().item(),
        "V_harm": obs["V_harm"].mean().item(),
        "V_coul": obs["V_coul"].mean().item(),
        "r12": obs["r12"].mean().item(),
    }


# ============================================================
# Imaginary Time Evolution
# ============================================================


def train_imaginary_time(
    well_sep: float,
    n_epochs: int = 1500,
    n_samples: int = 400,
    tau_max: float = 5.0,
    verbose: bool = True,
):
    """Train imaginary time evolution."""
    well_centers = torch.zeros(2, 2, dtype=DTYPE, device=DEVICE)
    if well_sep > 0:
        well_centers[0, 0] = -well_sep / 2
        well_centers[1, 0] = +well_sep / 2

    wf = TimeEvolvingSlaterJastrow(well_centers).to(DTYPE)
    optimizer = torch.optim.Adam(wf.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Expected ground state energy
    E_target = expected_ground_energy(well_sep)

    if verbose:
        print(f"  Training imag. time for d={well_sep}, E₀≈{E_target:.4f}")

    losses = []

    for epoch in range(n_epochs):
        # Sample τ values (more weight on early times for decay)
        tau = torch.rand(n_samples, dtype=DTYPE) ** 1.2 * tau_max

        with torch.no_grad():

            def log_psi_mid(x, _tau=tau_max / 2):
                return wf(x, torch.tensor(_tau, dtype=DTYPE))

            x = mcmc_sample(log_psi_mid, n_samples, well_centers, n_warmup=150)

        x_grad = x.requires_grad_(True)
        tau_grad = tau.requires_grad_(True)

        log_psi = wf(x_grad, tau_grad)
        d_tau = torch.autograd.grad(log_psi.sum(), tau_grad, create_graph=True)[0]

        def log_psi_fn(x_in, _tau=tau_grad):
            return wf(x_in, _tau)

        obs = local_energy(x_grad, log_psi_fn, well_centers)

        # Imaginary time Schrödinger: ∂log ψ/∂τ = -(E_L - E₀)
        residual = d_tau + (obs["E_L"] - E_target)
        loss = (residual**2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if verbose and epoch % 300 == 0:
            print(f"    Epoch {epoch:4d}: loss = {loss.item():.6f}")

    return wf, well_centers, np.array(losses), tau_max


def evaluate_time_evolution(
    wf, well_centers, n_tau: int = 30, tau_max: float = 5.0, n_samples: int = 2000
) -> list:
    """Evaluate E(τ) at many τ values."""
    tau_values = np.linspace(0, tau_max, n_tau)
    results = []

    for tau in tau_values:
        tau_t = torch.tensor(tau, dtype=DTYPE)

        with torch.no_grad():

            def log_psi_fn(x, _tau=tau_t):
                return wf(x, _tau)

            x = mcmc_sample(log_psi_fn, n_samples, well_centers, n_warmup=400)

        obs = local_energy(x, log_psi_fn, well_centers)

        results.append(
            {
                "tau": tau,
                "E": obs["E_L"].mean().item(),
                "E_err": obs["E_L"].std().item() / np.sqrt(n_samples),
                "T": obs["T"].mean().item(),
                "V_coul": obs["V_coul"].mean().item(),
                "r12": obs["r12"].mean().item(),
            }
        )

    return results


def fit_exponential_decay(tau_values, E_values, E_err, E0_guess):
    """
    Fit E(τ) = E₀ + ΔE × exp(-γτ)

    γ = 2×(E₁ - E₀) = 2×Δε (energy gap)
    """

    def model(tau, E0, dE, gamma):
        return E0 + dE * np.exp(-gamma * tau)

    try:
        # Initial guesses
        p0 = [E0_guess, E_values[0] - E0_guess, 1.0]

        # Bounds: E0 > 0, dE > 0, gamma > 0
        bounds = ([1.5, 0.0, 0.01], [5.0, 5.0, 10.0])

        popt, pcov = curve_fit(
            model, tau_values, E_values, p0=p0, sigma=E_err, bounds=bounds, maxfev=5000
        )

        E0_fit, dE_fit, gamma_fit = popt
        perr = np.sqrt(np.diag(pcov))

        # Energy gap: Δε = γ/2
        gap = gamma_fit / 2
        gap_err = perr[2] / 2

        return {
            "E0": E0_fit,
            "E0_err": perr[0],
            "dE": dE_fit,
            "gamma": gamma_fit,
            "gamma_err": perr[2],
            "gap": gap,
            "gap_err": gap_err,
            "success": True,
        }
    except Exception as e:
        print(f"    Fit failed: {e}")
        return {"success": False}


def expected_ground_energy(d: float) -> float:
    """Expected ground state energy based on separation."""
    if d == 0:
        return 3.0  # Taut 1993 exact
    # Coulomb ~ 1/sqrt(d² + <r²>)
    r_eff = np.sqrt(d**2 + 2.0)
    return 2.0 + 1.0 / r_eff


def expected_energy_gap(d: float) -> float:
    """Expected energy gap E₁ - E₀ based on separation."""
    if d == 0:
        return 1.0  # Approximate for single well
    # For large d: gap decreases exponentially (tunneling)
    return 1.0 * np.exp(-0.3 * d)


# ============================================================
# Main Analysis
# ============================================================


def run_full_analysis():
    """Run complete analysis with proper exponential decay fitting."""

    print("=" * 80)
    print("   IMAGINARY TIME EVOLUTION: SLATER-JASTROW WAVEFUNCTION")
    print("   Focus: Correct exponential decay E(τ) = E₀ + ΔE×exp(-γτ)")
    print("=" * 80)

    # Extended separation range
    separations = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]

    all_results = {
        "separations": separations,
        "vmc": {},
        "time_evolution": {},
        "fits": {},
    }

    # ========================================
    # Part 1: VMC Ground States
    # ========================================
    print("\n" + "=" * 80)
    print("PART 1: VMC GROUND STATE (Slater-Jastrow)")
    print("=" * 80)

    for d in separations:
        print(f"\n--- d = {d:.1f} ---")
        E_ref = expected_ground_energy(d)
        print(f"  Expected E₀ ≈ {E_ref:.4f}")

        wf, wc, energies = train_vmc(d, SlaterJastrow, n_epochs=2000)
        result = evaluate_vmc(wf, wc)

        error = (result["E"] - E_ref) / E_ref * 100
        status = "✓" if abs(error) < 2 else "~" if abs(error) < 5 else "?"

        print(f"  VMC: E = {result['E']:.4f} ± {result['E_err']:.4f} " f"({error:+.2f}%) {status}")
        print(
            f"  ⟨T⟩={result['T']:.3f}, ⟨V_c⟩={result['V_coul']:.4f}, " f"⟨r₁₂⟩={result['r12']:.3f}"
        )

        all_results["vmc"][d] = {
            "E_ref": E_ref,
            "E": result["E"],
            "E_err": result["E_err"],
            "error_pct": error,
            "components": result,
        }

        # Save model
        torch.save(wf.state_dict(), RESULTS_DIR / f"slater_jastrow_d{d:.1f}.pt")

    # ========================================
    # Part 2: Imaginary Time Evolution
    # ========================================
    print("\n" + "=" * 80)
    print("PART 2: IMAGINARY TIME EVOLUTION")
    print("=" * 80)

    for d in separations:
        print(f"\n--- d = {d:.1f} ---")
        E_ref = expected_ground_energy(d)
        gap_ref = expected_energy_gap(d)

        # Adjust tau_max based on expected gap
        tau_max = min(8.0, 4.0 / max(gap_ref, 0.1))

        wf, wc, losses, tau_max = train_imaginary_time(d, n_epochs=1500, tau_max=tau_max)

        # Evaluate at many τ points
        results = evaluate_time_evolution(wf, wc, n_tau=30, tau_max=tau_max)

        tau_vals = np.array([r["tau"] for r in results])
        E_vals = np.array([r["E"] for r in results])
        E_errs = np.array([r["E_err"] for r in results])

        # Fit exponential decay
        fit = fit_exponential_decay(tau_vals, E_vals, E_errs, E_ref)

        print("\n  E(τ) trajectory:")
        print(f"    τ=0.0: E = {results[0]['E']:.4f}")
        print(f"    τ={tau_max/2:.1f}: E = {results[len(results)//2]['E']:.4f}")
        print(f"    τ={tau_max:.1f}: E = {results[-1]['E']:.4f}")

        if fit["success"]:
            print(
                f"\n  Exponential fit: E(τ) = {fit['E0']:.4f} + "
                f"{fit['dE']:.4f}×exp(-{fit['gamma']:.3f}τ)"
            )
            print(f"    E₀ (fit) = {fit['E0']:.4f} ± {fit['E0_err']:.4f}")
            print(f"    E₀ (ref) = {E_ref:.4f}")
            print(f"    Energy gap Δε = {fit['gap']:.4f} ± {fit['gap_err']:.4f}")
            print(f"    Expected gap ≈ {gap_ref:.4f}")

            # Check if decay is correct
            if fit["gamma"] > 0.05:
                print("  ✓ Exponential decay observed")
            else:
                print("  ⚠ Very slow decay - may need more τ range")
        else:
            print("  ⚠ Exponential fit failed")

        all_results["time_evolution"][d] = {
            "E_ref": E_ref,
            "gap_ref": gap_ref,
            "tau_max": tau_max,
            "trajectory": results,
        }
        all_results["fits"][d] = fit

    # ========================================
    # Create Plots
    # ========================================
    create_decay_plots(all_results, separations)

    # ========================================
    # Save Results
    # ========================================
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(RESULTS_DIR / "imaginary_time_proper.json", "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n📁 Results saved to {RESULTS_DIR}/imaginary_time_proper.json")

    # ========================================
    # Summary
    # ========================================
    print_summary(all_results, separations)

    return all_results


def create_decay_plots(all_results: dict, separations: list):
    """Create comprehensive decay plots."""

    fig = plt.figure(figsize=(18, 12))

    # Colors for different separations
    colors = plt.cm.viridis(np.linspace(0, 1, len(separations)))

    # 1. E(τ) for all separations
    ax1 = fig.add_subplot(2, 3, 1)
    for i, d in enumerate(separations):
        if d in all_results["time_evolution"]:
            data = all_results["time_evolution"][d]
            taus = [r["tau"] for r in data["trajectory"]]
            Es = [r["E"] for r in data["trajectory"]]
            errs = [r["E_err"] for r in data["trajectory"]]

            ax1.errorbar(
                taus,
                Es,
                yerr=errs,
                color=colors[i],
                marker="o",
                markersize=3,
                label=f"d={d:.0f}",
                linewidth=1.5,
            )
            ax1.axhline(data["E_ref"], color=colors[i], linestyle="--", alpha=0.3)

    ax1.axhline(2.0, color="gray", linestyle=":", alpha=0.5, label="E=2.0")
    ax1.set_xlabel("Imaginary time τ", fontsize=12)
    ax1.set_ylabel("Energy E(τ)", fontsize=12)
    ax1.set_title("Energy vs Imaginary Time", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=1.9)

    # 2. Decay rate vs separation
    ax2 = fig.add_subplot(2, 3, 2)
    d_vals, gamma_vals, gamma_errs = [], [], []
    for d in separations:
        fit = all_results["fits"].get(d, {})
        if fit.get("success"):
            d_vals.append(d)
            gamma_vals.append(fit["gamma"])
            gamma_errs.append(fit.get("gamma_err", 0))

    if d_vals:
        ax2.errorbar(
            d_vals, gamma_vals, yerr=gamma_errs, marker="o", color="blue", linewidth=2, markersize=8
        )
        ax2.set_xlabel("Well separation d", fontsize=12)
        ax2.set_ylabel("Decay rate γ", fontsize=12)
        ax2.set_title("Decay Rate vs Separation", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

    # 3. Energy gap from fit
    ax3 = fig.add_subplot(2, 3, 3)
    d_vals, gap_vals, gap_errs, gap_refs = [], [], [], []
    for d in separations:
        fit = all_results["fits"].get(d, {})
        if fit.get("success"):
            d_vals.append(d)
            gap_vals.append(fit["gap"])
            gap_errs.append(fit.get("gap_err", 0))
            gap_refs.append(expected_energy_gap(d))

    if d_vals:
        ax3.errorbar(
            d_vals,
            gap_vals,
            yerr=gap_errs,
            marker="o",
            color="blue",
            label="Fit",
            linewidth=2,
            markersize=8,
        )
        ax3.plot(d_vals, gap_refs, "r--", label="Expected", linewidth=2)
        ax3.set_xlabel("Well separation d", fontsize=12)
        ax3.set_ylabel("Energy gap Δε", fontsize=12)
        ax3.set_title("Energy Gap vs Separation", fontsize=13, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

    # 4. Ground state energy
    ax4 = fig.add_subplot(2, 3, 4)
    d_vmc = [d for d in separations if d in all_results["vmc"]]
    E_vmc = [all_results["vmc"][d]["E"] for d in d_vmc]
    E_ref = [all_results["vmc"][d]["E_ref"] for d in d_vmc]

    ax4.plot(d_vmc, E_ref, "k--", label="Expected", linewidth=2)
    ax4.plot(d_vmc, E_vmc, "bo-", label="VMC", linewidth=2, markersize=8)
    ax4.axhline(2.0, color="gray", linestyle=":", alpha=0.5)
    ax4.set_xlabel("Well separation d", fontsize=12)
    ax4.set_ylabel("Ground state energy E₀", fontsize=12)
    ax4.set_title("Ground State Energy", fontsize=13, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Individual decay curves with fits
    ax5 = fig.add_subplot(2, 3, 5)
    for i, d in enumerate([0.0, 4.0, 8.0, 12.0]):
        if d in all_results["time_evolution"] and d in all_results["fits"]:
            data = all_results["time_evolution"][d]
            fit = all_results["fits"][d]

            taus = np.array([r["tau"] for r in data["trajectory"]])
            Es = np.array([r["E"] for r in data["trajectory"]])

            ax5.scatter(taus, Es, color=colors[separations.index(d)], s=20, alpha=0.7)

            if fit.get("success"):
                tau_fit = np.linspace(0, taus.max(), 100)
                E_fit = fit["E0"] + fit["dE"] * np.exp(-fit["gamma"] * tau_fit)
                ax5.plot(
                    tau_fit,
                    E_fit,
                    color=colors[separations.index(d)],
                    linewidth=2,
                    label=f"d={d:.0f}: γ={fit['gamma']:.2f}",
                )

    ax5.set_xlabel("τ", fontsize=12)
    ax5.set_ylabel("E(τ)", fontsize=12)
    ax5.set_title("Exponential Fits", fontsize=13, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    summary_lines = ["EXPONENTIAL DECAY ANALYSIS", "-" * 35, ""]
    summary_lines.append("E(τ) = E₀ + ΔE × exp(-γτ)")
    summary_lines.append("γ = 2 × (E₁ - E₀) = 2 × Δε")
    summary_lines.append("")
    summary_lines.append("d      E₀(fit)   γ        Δε")
    summary_lines.append("-" * 35)

    for d in separations[:6]:
        fit = all_results["fits"].get(d, {})
        if fit.get("success"):
            summary_lines.append(
                f"{d:4.1f}   {fit['E0']:.4f}    {fit['gamma']:.3f}    {fit['gap']:.4f}"
            )

    summary_text = "\n".join(summary_lines)
    ax6.text(
        0.1,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()

    path = RESULTS_DIR / "imaginary_time_decay.png"
    plt.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"\n📊 Decay plot saved: {path}")
    plt.close()


def print_summary(all_results: dict, separations: list):
    """Print final summary."""

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: EXPONENTIAL DECAY ANALYSIS")
    print("=" * 80)

    print("\n┌───────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│   d   │  E₀(ref) │  E₀(VMC) │  E₀(fit) │    γ     │    Δε    │")
    print("├───────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    for d in separations:
        vmc = all_results["vmc"].get(d, {})
        fit = all_results["fits"].get(d, {})

        E_ref = vmc.get("E_ref", "-")
        E_vmc = vmc.get("E", "-")
        E_fit = fit.get("E0", "-") if fit.get("success") else "-"
        gamma = fit.get("gamma", "-") if fit.get("success") else "-"
        gap = fit.get("gap", "-") if fit.get("success") else "-"

        if isinstance(E_ref, float):
            E_ref = f"{E_ref:.4f}"
        if isinstance(E_vmc, float):
            E_vmc = f"{E_vmc:.4f}"
        if isinstance(E_fit, float):
            E_fit = f"{E_fit:.4f}"
        if isinstance(gamma, float):
            gamma = f"{gamma:.4f}"
        if isinstance(gap, float):
            gap = f"{gap:.4f}"

        print(f"│ {d:5.1f} │ {E_ref:>8} │ {E_vmc:>8} │ " f"{E_fit:>8} │ {gamma:>8} │ {gap:>8} │")

    print("└───────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    print("\nKey Physics:")
    print("  • E(τ) = E₀ + ΔE × exp(-γτ)  [exponential decay to ground state]")
    print("  • γ = 2×Δε where Δε = E₁ - E₀ is the energy gap")
    print("  • Larger separation → smaller gap → slower decay")


if __name__ == "__main__":
    start = time.time()
    run_full_analysis()
    elapsed = time.time() - start
    print(f"\n⏱ Total time: {elapsed:.1f}s")
