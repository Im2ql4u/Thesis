"""
Quick Energy Analysis - Simple VMC evaluation
==============================================

Just compute VMC energy for the trained models.
No imaginary time evolution - that was wrong anyway (VMC doesn't do that).
"""

# Import from project
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import config as config_module
from functions.DoubleWell import double_well_total_potential
from functions.Neural_Networks import psi_fn as psi_fn_raw
from PINN import PINN, CTNNBackflowNet

DEVICE = torch.device("cpu")
DTYPE = torch.float64
N_PARTICLES = 2
D = 2
OMEGA = 1.0

RESULTS_DIR = Path(__file__).parent.parent / "results" / "double_well"


def make_cartesian_C_occ(nx, ny, n_occ):
    """Create occupation matrix for Cartesian HO basis.

    For closed shell: n_occ should be N_particles // 2 (orbitals per spin)
    """
    pairs = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    pairs.sort(key=lambda t: (t[0] + t[1], t[0]))
    sel = pairs[:n_occ]
    cols = [ix * ny + iy for (ix, iy) in sel]
    C = torch.zeros(nx * ny, n_occ, dtype=DTYPE, device=DEVICE)
    for j, c in enumerate(cols):
        C[c, j] = 1.0
    return C


def setup_config():
    config_module.update(
        device=DEVICE,
        omega=OMEGA,
        n_particles=N_PARTICLES,
        d=D,
        basis="cart",
        nx=2,
        ny=2,
    )
    return config_module.get().as_dict()


def load_model(well_separation: float):
    """Load trained model."""
    model_path = RESULTS_DIR / f"model_d{well_separation:.1f}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Infer architecture from state_dict:
    # phi.0.weight: (128, 2) -> hidden_dim=128, input=d=2
    # phi.4.weight: (5, 128) -> dL=5
    f_net = (
        PINN(
            n_particles=N_PARTICLES,
            d=D,
            omega=OMEGA,
            dL=5,
            hidden_dim=128,
            n_layers=2,
            act="gelu",
            init="xavier",
        )
        .to(DEVICE)
        .to(DTYPE)
    )

    # CTNNBackflowNet: d=2, msg_hidden=128, hidden=128
    backflow_net = (
        CTNNBackflowNet(
            d=D, msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu", omega=OMEGA
        )
        .to(DEVICE)
        .to(DTYPE)
    )

    # Keys are "f_net" and "backflow_net", not "*_state_dict"
    f_net.load_state_dict(checkpoint["f_net"])
    backflow_net.load_state_dict(checkpoint["backflow_net"])
    f_net.eval()
    backflow_net.eval()

    return f_net, backflow_net, checkpoint.get("energy", None)


def compute_vmc_energy(f_net, backflow_net, well_sep, n_samples=1000, n_equil=200, n_measure=500):
    """
    Compute VMC energy with MCMC sampling.
    """
    params = setup_config()
    # For closed-shell with N_PARTICLES, we need N_PARTICLES//2 orbitals (1 per spin)
    n_occ = N_PARTICLES // 2  # =1 for 2 particles
    C_occ = make_cartesian_C_occ(2, 2, n_occ)
    spin = torch.tensor([0, 1], device=DEVICE, dtype=torch.long)

    # Physical separation (coordinate transform)
    sep_phys = well_sep / np.sqrt(OMEGA)

    def log_psi(r):
        x_shifted = r.clone()
        x_shifted[:, 0, 0] += sep_phys / 2  # Shift particle 0 to left well
        x_shifted[:, 1, 0] -= sep_phys / 2  # Shift particle 1 to right well
        with torch.no_grad():
            log_psi_val, _ = psi_fn_raw(
                f_net, x_shifted, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )
            return log_psi_val

    def local_energy(r):
        """Compute local energy E_L = Hψ/ψ"""
        r_grad = r.clone().requires_grad_(True)
        x_shifted = r_grad.clone()
        x_shifted[:, 0, 0] = r_grad[:, 0, 0] + sep_phys / 2
        x_shifted[:, 1, 0] = r_grad[:, 1, 0] - sep_phys / 2

        with torch.enable_grad():
            log_psi_val, _ = psi_fn_raw(
                f_net, x_shifted, C_occ, backflow_net=backflow_net, spin=spin, params=params
            )

            # Kinetic energy via Laplacian
            grad_log_psi = torch.autograd.grad(log_psi_val.sum(), r_grad, create_graph=True)[0]

            laplacian = 0.0
            for i in range(N_PARTICLES):
                for j in range(D):
                    grad2 = torch.autograd.grad(
                        grad_log_psi[:, i, j].sum(), r_grad, retain_graph=True
                    )[0][:, i, j]
                    laplacian = laplacian + grad2

            grad_sq = (grad_log_psi**2).sum(dim=(1, 2))
            T = -0.5 * (laplacian + grad_sq)

            # Potential energy: harmonic trap + Coulomb
            V = double_well_total_potential(r_grad, well_sep, params={"omega": OMEGA})

            return (T + V).detach()

    # Initialize positions
    r = torch.randn(n_samples, N_PARTICLES, D, device=DEVICE, dtype=DTYPE) * 0.5
    r[:, 0, 0] -= sep_phys / 2  # Near left well
    r[:, 1, 0] += sep_phys / 2  # Near right well

    dt = 0.3  # MCMC step size
    accept_count = 0
    total_count = 0

    # Equilibrate
    print(f"  Equilibrating {n_equil} steps...", end=" ", flush=True)
    for _ in range(n_equil):
        log_p = 2.0 * log_psi(r)
        r_prop = r + dt * torch.randn_like(r)
        log_p_prop = 2.0 * log_psi(r_prop)
        accept = torch.rand(n_samples, device=DEVICE, dtype=DTYPE).log() < (log_p_prop - log_p)
        r = torch.where(accept.view(-1, 1, 1), r_prop, r)
    print("done")

    # Measure energy
    print(f"  Measuring {n_measure} steps...", end=" ", flush=True)
    energies = []
    for step in range(n_measure):
        # MCMC step
        log_p = 2.0 * log_psi(r)
        r_prop = r + dt * torch.randn_like(r)
        log_p_prop = 2.0 * log_psi(r_prop)
        accept = torch.rand(n_samples, device=DEVICE, dtype=DTYPE).log() < (log_p_prop - log_p)
        r = torch.where(accept.view(-1, 1, 1), r_prop, r)

        accept_count += accept.sum().item()
        total_count += n_samples

        # Measure every 5 steps (reduce autocorrelation)
        if step % 5 == 0:
            E_L = local_energy(r)
            energies.append(E_L.mean().item())

    print("done")

    E_mean = np.mean(energies)
    E_std = np.std(energies) / np.sqrt(len(energies))
    accept_rate = accept_count / total_count

    return E_mean, E_std, accept_rate


def main():
    print("=" * 60)
    print("VMC ENERGY EVALUATION FOR DOUBLE-WELL MODELS")
    print("=" * 60)

    print("\nExpected energies:")
    print("  d=0 (single well): E ≈ 3.0 (includes Coulomb repulsion)")
    print("  d→∞ (separate wells): E → 2.0 (each particle in its own well)")
    print()

    separations = [0.0, 4.0, 8.0]
    results = []

    for d in separations:
        print(f"\n{'='*60}")
        print(f"WELL SEPARATION d = {d}")
        print(f"{'='*60}")

        try:
            f_net, backflow_net, train_energy = load_model(d)
            print(
                f"  Training reported energy: {train_energy:.4f}"
                if train_energy
                else "  Training energy: N/A"
            )

            E_vmc, E_err, acc = compute_vmc_energy(
                f_net, backflow_net, d, n_samples=500, n_equil=100, n_measure=200
            )

            print(f"\n  VMC Energy: E = {E_vmc:.4f} ± {E_err:.4f}")
            print(f"  Acceptance rate: {acc:.1%}")

            results.append({"d": d, "E_vmc": E_vmc, "E_err": E_err, "E_train": train_energy})

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"d": d, "E_vmc": None, "E_err": None, "E_train": None})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'d':>6} | {'E_VMC':>12} | {'E_train':>12} | {'Expected':>10}")
    print("-" * 50)
    for r in results:
        E_exp = 3.0 if r["d"] == 0 else 2.0
        E_vmc_str = f"{r['E_vmc']:.4f} ± {r['E_err']:.4f}" if r["E_vmc"] else "N/A"
        E_train_str = f"{r['E_train']:.4f}" if r.get("E_train") else "N/A"
        print(f"{r['d']:>6.1f} | {E_vmc_str:>12} | {E_train_str:>12} | {E_exp:>10.1f}")


if __name__ == "__main__":
    main()
