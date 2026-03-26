#!/usr/bin/env python3
"""
Simple Layer 1 Diagnostic: Low-Omega Transfer Failure Root Cause
=================================================================

Test hypothesis: Does importance-resampling ESS collapse when transferring 
from ω=0.01 checkpoint to ω=0.001 evaluation?

Run: python scripts/diag_lowomega_simple.py --checkpoint PATH_TO_CKPT --n-elec 2
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
from functions.Neural_Networks import (
    sample_mixture,
    lookup_dmc_energy,
    psi_fn,
)
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DIM = 2


def main():
    ap = argparse.ArgumentParser(description="Simple Layer 1 diagnostic for low-omega")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-elec", type=int, default=2)
    ap.add_argument("--training-omega", type=float, default=0.01, help="Omega where checkpoint was trained")
    ap.add_argument("--transfer-omega", type=float, default=0.001, help="Omega we want to transfer to")
    ap.add_argument("--n-test-samples", type=int, default=100000)
    a = ap.parse_args()

    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC: Low-Omega ESS Collapse")
    print(f"Checkpoint trained at ω={a.training_omega}")
    print(f"Test transfer to ω={a.transfer_omega}")
    print("=" * 80)

    # 1. Verify DMC reference
    e_dmc_train = lookup_dmc_energy(a.n_elec, a.training_omega)
    e_dmc_target = lookup_dmc_energy(a.n_elec, a.transfer_omega)
    print(f"\n✓ DMC References:")
    print(f"  ω={a.training_omega}: E_DMC={e_dmc_train:.6f} {'✓' if math.isfinite(e_dmc_train) else '✗ MISSING'}")
    print(f"  ω={a.transfer_omega}: E_DMC={e_dmc_target:.6f} {'✓' if math.isfinite(e_dmc_target) else '✗ MISSING'}")

    # 2. Load checkpoint
    try:
        ckpt = torch.load(a.checkpoint, map_location=DEVICE)
        bf_state = ckpt.get("bf_state")
        jas_state = ckpt.get("jas_state")
        print(f"\n✓ Checkpoint loaded: bf={bf_state is not None}, jas={jas_state is not None}")
    except Exception as e:
        print(f"\n✗ Failed to load checkpoint: {e}")
        return

    # 3. Build networks at transfer omega (this is the key test)
    n_occ = a.n_elec // 2
    nx = ny = max(3, int(math.ceil(math.sqrt(float(n_occ)))))

    print(f"\n✓ Building networks for ω={a.transfer_omega}:")

    f_net = CTNNJastrowVCycle(
        n_particles=a.n_elec,
        d=DIM,
        omega=a.transfer_omega,  # KEY: using *transfer* omega
        node_hidden=24,
        edge_hidden=24,
        bottleneck_hidden=12,
        n_down=1,
        n_up=1,
        msg_layers=1,
        node_layers=1,
        readout_hidden=64,
        readout_layers=2,
        act="silu",
    ).to(DEVICE).to(DTYPE)

    bf_net = CTNNBackflowNet(
        d=DIM,
        msg_hidden=64,
        msg_layers=2,
        hidden=64,
        layers=2,
        act="silu",
        omega=a.transfer_omega,  # KEY: using *transfer* omega
    ).to(DEVICE).to(DTYPE)

    # Load checkpoint weights
    try:
        if jas_state is not None:
            f_net.load_state_dict(jas_state)
        if bf_state is not None:
            bf_net.load_state_dict(bf_state)
        print(f"  ✓ Weights loaded from checkpoint")
    except RuntimeError as e:
        print(f"  ✗ State dict mismatch: {e}")
        return

    # 4. Build psi_log_fn
    spin = torch.cat(
        [
            torch.zeros(a.n_elec // 2, dtype=torch.long),
            torch.ones(a.n_elec - a.n_elec // 2, dtype=torch.long),
        ]
    ).to(DEVICE)

    # C_occ from basis functions
    energies = sorted(
        [(a.transfer_omega * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)]
    )
    C = np.zeros((nx * ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)

    params_dict = {
        "omega": a.transfer_omega,
        "n_particles": a.n_elec,
        "d": DIM,
        "nx": nx,
        "ny": ny,
        "basis": "cart",
        "L": max(8.0, 3.0 / math.sqrt(a.transfer_omega)),
    }

    def psi_log_fn(x):
        lp, _ = psi_fn(f_net, x, C_occ, backflow_net=bf_net, spin=spin, params=params_dict)
        return lp

    # 5. Test proposal overlap at transfer omega
    print(f"\n✓ Testing proposal-wavefunction overlap at ω={a.transfer_omega}:")
    print(f"  Sampling {a.n_test_samples:,} points from Gaussian mixture...")

    sigma_fs = (0.8, 1.3, 2.0)
    with torch.no_grad():
        x_samples, lq_samples = sample_mixture(
            a.n_test_samples,
            a.n_elec,
            DIM,
            a.transfer_omega,
            device=DEVICE,
            dtype=DTYPE,
            sigma_fs=sigma_fs,
        )

        # Evaluate log|ψ|² at samples
        lp2_list = []
        chunk = 2048
        for i in range(0, len(x_samples), chunk):
            lp2_i = 2.0 * psi_log_fn(x_samples[i : i + chunk])
            lp2_list.append(lp2_i)
        lp2_samples = torch.cat(lp2_list)

        # Importance weights
        log_w_raw = lp2_samples - lq_samples
        log_w_clipped = torch.clamp(log_w_raw, min=-20, max=20)
        w_samples = torch.exp(log_w_clipped)

        # ESS
        ESS = (w_samples.sum() ** 2) / (w_samples.pow(2).sum())
        ESS_frac = ESS.item() / len(w_samples)

    print(f"  ESS = {ESS.item():.0f} out of {len(w_samples):,} ({ESS_frac*100:.2f}%)")
    print(f"  Weight stats:")
    print(f"    mean = {w_samples.mean():.4e}")
    print(f"    std  = {w_samples.std():.4e}")
    print(f"    min  = {w_samples.min():.4e}")
    print(f"    max  = {w_samples.max():.4e}")
    print(f"    max/min = {(w_samples.max() / w_samples.min()).item():.2e}")

    # 6. Diagnosis
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    if ESS_frac < 0.05:
        print(f"✗ CRITICAL: ESS ratio {ESS_frac*100:.2f}% < 5%")
        print(f"   Interpretation: Proposal (Gaussian mixture σ_fs={sigma_fs})")
        print(f"                   has negligible overlap with wavefunction at ω={a.transfer_omega}")
        print(f"   Root cause: This is a Layer 1 (sampling) failure.")
        print(f"\n   Action: Adaptive sigma_fs or wider mixture needed for ω<0.01 transfer.")
    elif ESS_frac < 0.1:
        print(f"⚠ WARNING: ESS ratio {ESS_frac*100:.2f}% is borderline")
        print(f"   Interpretation: Proposal overlap is marginal.")
        print(f"   Effect: Training will be very noisy, gradients unreliable.")
        print(f"\n   Action: Consider bias/reliability diagnostics before training.")
    else:
        print(f"✓ ESS ratio {ESS_frac*100:.2f}% is acceptable (≥10%)")
        print(f"   Interpretation: Sampling is NOT the primary failure mode.")
        print(f"   Next step: Investigate Layers 2-4 (implementation, architecture, training).")

    if (w_samples.max() / w_samples.min()).item() > 1e6:
        print(f"\n⚠ Weight distribution is extremely skewed (max/min > 1e6)")
        print(f"   This causes high variance in gradient estimates even with good ESS.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
