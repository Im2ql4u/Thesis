#!/usr/bin/env python3
"""
Test: Does adaptive sigma_fs fix the ESS collapse at ω=0.001?
========================================================================

Compare importance resampling ESS with:
1. FIXED sigma_fs = (0.8, 1.3, 2.0)  [baseline: broken]
2. ADAPTIVE sigma_fs (auto-widened)  [proposed fix]

If adaptive σ_fs works, ESS should jump from 0.02% to ≥5% at ω=0.001
"""

import sys
import torch
import math
import argparse
from pathlib import Path

# Setup path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from functions.Neural_Networks import (
    eval_mixture_logq
)
from run_weak_form import adapt_sigma_fs, sample_mixture

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DIM = 2
N_ELEC = 2


def eval_mixture_logq_custom(x, omega, sigma_fs):
    """Local mixture log-density eval (copy of neural_networks version)."""
    nc = len(sigma_fs)
    s_eff = torch.tensor([sf / math.sqrt(omega) for sf in sigma_fs], 
                         device=x.device, dtype=x.dtype)
    nd = N_ELEC * DIM
    lq_sum = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    for s in s_eff:
        var = s ** 2
        lq_sum_exp = -0.5 * (x ** 2).sum(dim=(1, 2)) / var - 0.5 * nd * torch.log(var)
        lq_sum = torch.logaddexp(lq_sum, lq_sum_exp)
    lq_sum -= math.log(nc)
    return lq_sum


def measure_ess(psi_log_fn, n_samples, omega, sigma_fs, label=""):
    """Measure ESS when resampling with given sigma_fs."""
    # Sample from proposal
    x_samples, lq_samples = sample_mixture(n_samples, omega, sigma_fs)
    
    # Compute log Psi at samples
    with torch.no_grad():
        psi_log_samples = psi_log_fn(x_samples)
    
    # Compute importance weights: w = exp(2*log_psi - log_q)
    with torch.no_grad():
        log_w = 2.0 * psi_log_samples - lq_samples
        
        # Clamp to prevent overflow
        log_w_max = log_w.max()
        log_w = log_w - log_w_max
        
        w = torch.exp(log_w)
        
    # ESS = (sum w)^2 / sum w^2
    w_sum = w.sum()
    w2_sum = (w ** 2).sum()
    
    ess = (w_sum ** 2) / w2_sum if w2_sum > 0 else 0.0
    ess_ratio = float(ess) / n_samples * 100.0
    
    w_mean = float(w.mean())
    w_std = float(w.std())
    w_max = float(w.max())
    w_min = float(w.min())
    w_range = w_max / w_min if w_min > 0 else float('inf')
    
    return {
        'ess': float(ess),
        'ess_ratio': ess_ratio,
        'w_mean': w_mean,
        'w_std': w_std,
        'w_min': w_min,
        'w_max': w_max,
        'w_range': w_range,
        'sigma_fs': sigma_fs,
        'label': label,
    }


def main():
    parser = argparse.ArgumentParser(description="Test adaptive sigma_fs fix")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--n-elec", type=int, default=2)
    parser.add_argument("--training-omega", type=float, default=0.01)
    parser.add_argument("--transfer-omega", type=float, default=0.001)
    parser.add_argument("--n-test-samples", type=int, default=80000)
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"\n{'='*70}")
    print("ADAPTIVE SIGMA_FS FIX VALIDATION")
    print(f"{'='*70}")
    print(f"\nLoading checkpoint: {args.checkpoint}")
    
    try:
        from jastrow_architectures import CTNNJastrowVCycle
        from PINN import CTNNBackflowNet
        from functions.Neural_Networks import psi_fn
        import numpy as np
        
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        
        # Build networks at transfer omega
        n_occ = args.n_elec // 2
        nx = ny = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
        
        f_net = CTNNJastrowVCycle(
            n_particles=args.n_elec,
            d=DIM,
            omega=args.transfer_omega,
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
            omega=args.transfer_omega,
        ).to(DEVICE).to(DTYPE)
        
        # Load checkpoint state
        if "jas_state" in ckpt:
            f_net.load_state_dict(ckpt["jas_state"])
        if "bf_state" in ckpt and bf_net is not None:
            bf_net.load_state_dict(ckpt["bf_state"])
        
        # Build C_occ (occupancy matrix for Slater determinant)
        spin = torch.cat(
            [
                torch.zeros(args.n_elec // 2, dtype=torch.long),
                torch.ones(args.n_elec - args.n_elec // 2, dtype=torch.long),
            ]
        ).to(DEVICE)
        
        energies = sorted(
            [(args.transfer_omega * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)]
        )
        C = np.zeros((nx * ny, n_occ))
        for k in range(n_occ):
            _, ix, iy = energies[k]
            C[ix * ny + iy, k] = 1.0
        C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
        
        params_dict = {
            "omega": args.transfer_omega,
            "n_particles": args.n_elec,
            "d": DIM,
            "nx": nx,
            "ny": ny,
            "basis": "cart",
            "L": max(8.0, 3.0 / math.sqrt(args.transfer_omega)),
        }
        
        # Define psi function for closure
        def psi_fn_closure(x):
            with torch.no_grad():
                lp, _ = psi_fn(f_net, x, C_occ, backflow_net=bf_net,
                            spin=spin, params=params_dict)
                return lp
        
        # Test conditions
        print(f"\nTest Setup:")
        print(f"  Checkpoint trained at: ω = {args.training_omega}")
        print(f"  Transfer to: ω = {args.transfer_omega}")
        print(f"  N electrons: {args.n_elec}")
        print(f"  Test samples: {args.n_test_samples:,}")

        
        print(f"\n{'-'*70}")
        print("PHASE 1: Baseline (Fixed sigma_fs)")
        print(f"{'-'*70}")
        
        # Baseline: fixed sigma_fs
        sigma_fs_fixed = (0.8, 1.3, 2.0)
        result_fixed = measure_ess(
            psi_fn_closure, args.n_test_samples, args.transfer_omega, 
            sigma_fs_fixed, "Fixed σ_fs"
        )
        
        print(f"\nProposal widths: {sigma_fs_fixed}")
        print(f"  ESS: {result_fixed['ess']:.0f} / {args.n_test_samples:,} ({result_fixed['ess_ratio']:.3f}%)")
        print(f"  Weight stats:")
        print(f"    mean = {result_fixed['w_mean']:.3e}")
        print(f"    std  = {result_fixed['w_std']:.3e}")
        print(f"    min  = {result_fixed['w_min']:.3e}")
        print(f"    max  = {result_fixed['w_max']:.3e}")
        print(f"    max/min ratio = {result_fixed['w_range']:.3e}")
        
        if result_fixed['ess_ratio'] < 5:
            print(f"\n✗ BASELINE FAILURE CONFIRMED: ESS ratio {result_fixed['ess_ratio']:.3f}% < 5%")
        
        # Proposed: adaptive sigma_fs
        print(f"\n{'-'*70}")
        print("PHASE 2: Proposed Fix (Adaptive sigma_fs)")
        print(f"{'-'*70}")
        
        sigma_fs_adaptive = adapt_sigma_fs(args.transfer_omega, sigma_fs_fixed)
        result_adaptive = measure_ess(
            psi_fn_closure, args.n_test_samples, args.transfer_omega,
            sigma_fs_adaptive, "Adaptive σ_fs"
        )
        
        print(f"\nAdapted proposal widths: {sigma_fs_adaptive}")
        print(f"  ({len(sigma_fs_adaptive)} components vs {len(sigma_fs_fixed)} baseline)")
        print(f"  ESS: {result_adaptive['ess']:.0f} / {args.n_test_samples:,} ({result_adaptive['ess_ratio']:.3f}%)")
        print(f"  Weight stats:")
        print(f"    mean = {result_adaptive['w_mean']:.3e}")
        print(f"    std  = {result_adaptive['w_std']:.3e}")
        print(f"    min  = {result_adaptive['w_min']:.3e}")
        print(f"    max  = {result_adaptive['w_max']:.3e}")
        print(f"    max/min ratio = {result_adaptive['w_range']:.3e}")
        
        if result_adaptive['ess_ratio'] >= 5:
            print(f"\n✓ FIX SUCCESSFUL: ESS ratio improved to {result_adaptive['ess_ratio']:.3f}%")
        else:
            print(f"\n✗ FIX INSUFFICIENT: ESS ratio still {result_adaptive['ess_ratio']:.3f}% < 5%")
        
        # Summary
        print(f"\n{'-'*70}")
        print("SUMMARY")
        print(f"{'-'*70}")
        
        ess_improvement = result_adaptive['ess'] / result_fixed['ess'] if result_fixed['ess'] > 0 else 0
        print(f"\nESS Improvement Factor: {ess_improvement:.1f}x")
        print(f"  Baseline ESS: {result_fixed['ess']:.0f}")
        print(f"  Adaptive ESS: {result_adaptive['ess']:.0f}")
        
        ratio_improvement = result_adaptive['ess_ratio'] / result_fixed['ess_ratio'] * 100 if result_fixed['ess_ratio'] > 0 else 0
        print(f"\nESS Ratio Improvement: {ratio_improvement:.0f}x")
        print(f"  Baseline ratio: {result_fixed['ess_ratio']:.3f}%")
        print(f"  Adaptive ratio: {result_adaptive['ess_ratio']:.3f}%")
        
        weight_range_ratio = result_fixed['w_range'] / result_adaptive['w_range'] if result_adaptive['w_range'] > 0 else float('inf')
        print(f"\nWeight Distribution Stability: {weight_range_ratio:.1f}x better")
        print(f"  Baseline max/min: {result_fixed['w_range']:.3e}")
        print(f"  Adaptive max/min: {result_adaptive['w_range']:.3e}")
        
        if result_adaptive['ess_ratio'] >= 5 and ess_improvement > 10:
            print("\n" + "="*70)
            print("✓✓✓ FIX VALIDATED ✓✓✓")
            print("="*70)
            print("Adaptive sigma_fs resolves the ESS collapse at low omega.")
            print("Ready for production training runs on N=2,6,12,20 grid.")
            return 0
        else:
            print("\n" + "="*70)
            print("⚠ PARTIAL FIX - Further investigation needed")
            print("="*70)
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
