#!/usr/bin/env python3
"""
Diagnostic Update: Test Adaptive Sigma_fs Fix
============================================

Extend the simple diagnostic to compare:
1. FIXED sigma_fs = (0.8, 1.3, 2.0)
2. ADAPTIVE sigma_fs from adapt_sigma_fs()

Usage: python scripts/diag_adaptive_sigma_update.py --checkpoint PATH
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
from run_weak_form import adapt_sigma_fs

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DIM = 2


def main():
    ap = argparse.ArgumentParser(description="Test adaptive sigma_fs fix")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--n-elec", type=int, default=2)
    ap.add_argument("--training-omega", type=float, default=0.01)
    ap.add_argument("--transfer-omega", type=float, default=0.001)
    ap.add_argument("--n-test-samples", type=int, default=80000)
    a = ap.parse_args()

    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC: Adaptive Sigma_fs Fix Validation")
    print(f"Checkpoint trained at ω={a.training_omega}")
    print(f"Test transfer to ω={a.transfer_omega}")
    print("=" * 80)

    # 1. Verify DMC reference
    e_dmc_train = lookup_dmc_energy(a.n_elec, a.training_omega)
    e_dmc_target = lookup_dmc_energy(a.n_elec, a.transfer_omega)
    print(f"\n✓ DMC References:")
    print(f"  ω={a.training_omega}: E_DMC={e_dmc_train:.6f}")
    print(f"  ω={a.transfer_omega}: E_DMC={e_dmc_target:.6f}")

    # 2. Load checkpoint
    try:
        ckpt = torch.load(a.checkpoint, map_location=DEVICE)
        bf_state = ckpt.get("bf_state")
        jas_state = ckpt.get("jas_state")
        print(f"\n✓ Checkpoint loaded: bf={bf_state is not None}, jas={jas_state is not None}")
    except Exception as e:
        print(f"\n✗ Failed to load checkpoint: {e}")
        return

    # 3. Build networks at transfer omega
    n_occ = a.n_elec // 2
    nx = ny = max(3, int(math.ceil(math.sqrt(float(n_occ)))))

    print(f"\n✓ Building networks for ω={a.transfer_omega}:")

    f_net = CTNNJastrowVCycle(
        n_particles=a.n_elec,
        d=DIM,
        omega=a.transfer_omega,
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
        omega=a.transfer_omega,
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

    # ------- PHASE 1: BASELINE (Fixed Sigma_fs) -------
    print(f"\n{'='*80}")
    print("PHASE 1: Baseline Test (Fixed sigma_fs)")
    print(f"{'='*80}")

    sigma_fs_fixed = (0.8, 1.3, 2.0)
    print(f"\nProposal widths (FIXED): {sigma_fs_fixed}")
    print(f"Sampling {a.n_test_samples:,} points...")

    with torch.no_grad():
        x_fixed, lq_fixed = sample_mixture(
            a.n_test_samples,
            a.n_elec,
            DIM,
            a.transfer_omega,
            device=DEVICE,
            dtype=DTYPE,
            sigma_fs=sigma_fs_fixed,
        )
        psi_fixed = psi_log_fn(x_fixed)
        log_w_fixed = 2.0 * psi_fixed - lq_fixed
        log_w_fixed_norm = log_w_fixed - log_w_fixed.max()
        w_fixed = torch.exp(log_w_fixed_norm)

    ess_fixed = (w_fixed.sum() ** 2) / (w_fixed ** 2).sum()
    ess_ratio_fixed = float(ess_fixed) / a.n_test_samples * 100.0

    print(f"\nResults (FIXED σ_fs):")
    print(f"  ESS: {int(ess_fixed):,} / {a.n_test_samples:,} ({ess_ratio_fixed:.4f}%)")
    print(f"  Weight range: [{w_fixed.min():.3e}, {w_fixed.max():.3e}]")
    print(f"  Max/min ratio: {w_fixed.max() / w_fixed.min():.3e}")

    if ess_ratio_fixed < 5.0:
        print(f"  ✗ CRITICAL: ESS ratio {ess_ratio_fixed:.4f}% < 5% (sampling fails)")
    else:
        print(f"  ✓ ESS ratio {ess_ratio_fixed:.4f}% >= 5% (sampling OK)")

    # ------- PHASE 2: ADAPTIVE SigmaFS -------
    print(f"\n{'='*80}")
    print("PHASE 2: Proposed Fix (Adaptive sigma_fs)")
    print(f"{'='*80}")

    sigma_fs_adaptive = adapt_sigma_fs(a.transfer_omega, sigma_fs_fixed)
    print(f"\nProposal widths (ADAPTIVE): {sigma_fs_adaptive}")
    print(f"  {len(sigma_fs_adaptive)} components (vs {len(sigma_fs_fixed)} baseline)")
    print(f"Sampling {a.n_test_samples:,} points...")

    with torch.no_grad():
        x_adaptive, lq_adaptive = sample_mixture(
            a.n_test_samples,
            a.n_elec,
            DIM,
            a.transfer_omega,
            device=DEVICE,
            dtype=DTYPE,
            sigma_fs=sigma_fs_adaptive,
        )
        psi_adaptive = psi_log_fn(x_adaptive)
        log_w_adaptive = 2.0 * psi_adaptive - lq_adaptive
        log_w_adaptive_norm = log_w_adaptive - log_w_adaptive.max()
        w_adaptive = torch.exp(log_w_adaptive_norm)

    ess_adaptive = (w_adaptive.sum() ** 2) / (w_adaptive ** 2).sum()
    ess_ratio_adaptive = float(ess_adaptive) / a.n_test_samples * 100.0

    print(f"\nResults (ADAPTIVE σ_fs):")
    print(f"  ESS: {int(ess_adaptive):,} / {a.n_test_samples:,} ({ess_ratio_adaptive:.4f}%)")
    print(f"  Weight range: [{w_adaptive.min():.3e}, {w_adaptive.max():.3e}]")
    print(f"  Max/min ratio: {w_adaptive.max() / w_adaptive.min():.3e}")

    if ess_ratio_adaptive >= 5.0:
        print(f"  ✓ SUCCESS: ESS ratio {ess_ratio_adaptive:.4f}% >= 5% (sampling works!)")
    else:
        print(f"  ✗ INSUFFICIENT: ESS ratio still {ess_ratio_adaptive:.4f}% < 5%")

    # ------- SUMMARY -------
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    ess_improvement = float(ess_adaptive) / float(ess_fixed) if ess_fixed > 0 else 0
    ratio_improvement = ess_ratio_adaptive / ess_ratio_fixed if ess_ratio_fixed > 0 else 0

    print(f"\nESS Improvement:")
    print(f"  Baseline: {int(ess_fixed):,} samples (ratio={ess_ratio_fixed:.4f}%)")
    print(f"  Adaptive: {int(ess_adaptive):,} samples (ratio={ess_ratio_adaptive:.4f}%)")
    print(f"  Factor: {ess_improvement:.1f}x improvement")

    weight_improvement = (w_fixed.max() / w_fixed.min()) / (w_adaptive.max() / w_adaptive.min()) if w_adaptive.max() / w_adaptive.min() > 0 else 0
    print(f"\nWeight Distribution Stability:")
    print(f"  Baseline max/min: {w_fixed.max() / w_fixed.min():.3e}")
    print(f"  Adaptive max/min: {w_adaptive.max() / w_adaptive.min():.3e}")
    print(f"  Stability gain: {weight_improvement:.1f}x")

    print(f"\n{'='*80}")
    if ess_ratio_adaptive >= 5.0 and ess_improvement > 5:
        print("✓✓✓ ADAPTIVE SIGMA_FS FIX VALIDATED ✓✓✓")
        print(f"{'='*80}")
        print("\nConclusion:")
        print(f"  • ESS collapse at ω={a.transfer_omega} is FIXED by adaptive widening")
        print(f"  • Ready to retrain N=2,6,12,20 grid with fix applied")
        print(f"  • Use: adapt_sigma_fs() in run_weak_form.py training loop")
        return 0
    else:
        print("⚠ PARTIAL SUCCESS - Further investigation may be needed")
        print(f"{'='*80}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
