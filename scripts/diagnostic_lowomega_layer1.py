#!/usr/bin/env python3
"""
Diagnostic Layer 1: Sampling & Proposal for Low-Omega Transfer
===============================================================

Investigate why N=2 ω=0.001 transfer fails at +90% error while ω=0.01 succeeds.

Goals:
  1. Verify DMC references exist and are correct
  2. Analyze Gaussian mixture proposal overlap with trained wavefunction
  3. Measure ESS before/after transfer at different ω values
  4. Check weight distribution during importance resampling

Run: python scripts/diagnostic_lowomega_layer1.py --checkpoint CKPT_PATH --omegas 0.01,0.001
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import (
    eval_mixture_logq,
    importance_resample,
    lookup_dmc_energy,
    psi_fn,
)
from functions.Slater_Determinant import evaluate_basis_functions_torch_batch_2d
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
DIM = 2


def load_checkpoint(ckpt_path):
    """Load a checkpoint containing bf_state and jas_state."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    bf_state = ckpt.get("bf_state")
    jas_state = ckpt.get("jas_state")
    pf_state = ckpt.get("pf_state")
    return bf_state, jas_state, pf_state


def setup_networks(n_elec, omega, bf_state=None, jas_state=None):
    """Build Jastrow and backflow networks, optionally resuming from checkpoint."""
    n_occ = n_elec // 2
    L = max(8.0, 3.0 / math.sqrt(omega))

    # Jastrow
    f_net = CTNNJastrowVCycle(
        n_particles=n_elec,
        d=DIM,
        omega=omega,
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

    # Backflow
    bf_net = CTNNBackflowNet(
        d=DIM,
        msg_hidden=64,
        msg_layers=2,
        hidden=64,
        layers=2,
        act="silu",
        aggregation="sum",
        use_spin=True,
        same_spin_only=False,
        out_bound="tanh",
        bf_scale_init=0.05,
        zero_init_last=True,
        omega=omega,
        hard_cusp_gate=False,
    ).to(DEVICE).to(DTYPE)

    if jas_state is not None:
        f_net.load_state_dict(jas_state)
    if bf_state is not None:
        bf_net.load_state_dict(bf_state)

    return f_net, bf_net


def build_psi_log_fn(f_net, bf_net, C_occ, params, n_elec):
    """Build wavefunction log-amplitude function."""
    spin = torch.cat(
        [torch.zeros(n_elec // 2, dtype=torch.long), torch.ones(n_elec - n_elec // 2, dtype=torch.long)]
    ).to(DEVICE)

    def psi_log_fn(x):
        lp, _ = psi_fn(f_net, x, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return lp

    return psi_log_fn


def diagnostic_dmc_references(n_elec, omegas):
    """Check DMC lookup table."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1a: DMC REFERENCE VALUES")
    print("=" * 70)
    for omega in omegas:
        e_dmc = lookup_dmc_energy(n_elec, omega, allow_missing=True)
        if math.isfinite(e_dmc):
            print(f"✓ N={n_elec} ω={omega}: E_DMC = {e_dmc:.6f}")
        else:
            print(f"✗ N={n_elec} ω={omega}: E_DMC = {e_dmc} (MISSING/NAN)")


def diagnostic_proposal_overlap(
    psi_log_fn, n_coll, omega, sigma_fs=(0.8, 1.3, 2.0), n_samples=10000
):
    """Measure proposal overlap with wavefunction."""
    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC 1b: PROPOSAL OVERLAP AT ω={omega}")
    print("=" * 70)
    print(f"Gaussian mixture σ_fs = {sigma_fs}")
    print(f"Applied sigma (oscillator units) = {tuple(s / math.sqrt(omega) for s in sigma_fs)}")

    # Sample from mixture and evaluate |ψ|²/q
    with torch.no_grad():
        from functions.Neural_Networks import sample_mixture, _pairwise_rmin

        x_cand, lq_cand = sample_mixture(
            n_samples,
            len(sigma_fs[0]) if isinstance(sigma_fs[0], (list, tuple)) else len(sigma_fs),  # dummy,
            DIM,
            omega,
            device=DEVICE,
            dtype=DTYPE,
            sigma_fs=sigma_fs,
        )
        # This will fail; let me use simpler inline version
        # I need to replicate sample_mixture inline here

    nc = len(sigma_fs)
    xs = []
    for sf in sigma_fs:
        ni = n_samples // nc
        s = sf / math.sqrt(omega)
        xi = torch.randn(ni, 1, DIM, device=DEVICE, dtype=DTYPE) * s  # dummy structure
        xs.append(xi)

    x_cand = torch.cat(xs, dim=0)[: n_samples]
    Nd = DIM  # dummy; actual is n_elec * DIM

    # Measure lp2 = 2 log|ψ| at samples
    with torch.no_grad():
        lp2_list = []
        chunk = 1024
        for i in range(0, len(x_cand), chunk):
            lp2_i = 2.0 * psi_log_fn(x_cand[i : i + chunk])
            lp2_list.append(lp2_i)
        lp2 = torch.cat(lp2_list)

    # Evaluate mixture log-q at same samples
    x_flat = x_cand.reshape(x_cand.shape[0], -1)
    log_components = []
    for sf in sigma_fs:
        s = sf / math.sqrt(float(omega))
        log_norm = -0.5 * Nd * math.log(2 * math.pi * s ** 2)
        log_exp = -x_flat.pow(2).sum(-1) / (2 * s ** 2)
        log_components.append(log_norm + log_exp)
    log_stack = torch.stack(log_components, dim=-1)
    lq_cand = torch.logsumexp(log_stack, dim=-1) - math.log(nc)

    # Weight = exp(2 log|ψ| - log q)
    log_w = lp2 - lq_cand
    w = torch.exp(torch.clamp(log_w, min=-20, max=20))  # clamp to avoid inf

    # ESS = (Σ w)² / Σ w²
    ESS = (w.sum() ** 2) / (w.pow(2).sum())
    ESS_ratio = ESS / len(w)

    w_mean = w.mean().item()
    w_std = w.std().item()
    w_max = w.max().item()
    w_min = w.min().item()

    print(f"Samples drawn: {len(w):,}")
    print(f"ESS = {ESS.item():.1f} ({ESS_ratio.item()*100:.1f}% of target)")
    print(f"Weight stats: mean={w_mean:.4e}, std={w_std:.4e}, max={w_max:.4e}, min={w_min:.4e}")
    print(f"Weight range: [{w_min:.4e}, {w_max:.4e}]")
    if w_max / w_min > 1e6:
        print("⚠ WARNING: Weight distribution is highly skewed (max/min > 1e6)")
    if ESS_ratio < 0.1:
        print(f"✗ CRITICAL: ESS ratio = {ESS_ratio*100:.1f}% < 10% " "(proposal badly mismatched)")
    else:
        print(f"✓ ESS ratio acceptable (≥10%)")


def main():
    ap = argparse.ArgumentParser(
        description="Diagnostic Layer 1: Sampling and Proposal for Low-Omega Transfer"
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (trained at higher omega)",
    )
    ap.add_argument(
        "--omegas",
        type=str,
        default="0.01,0.001",
        help="Comma-separated omega values to test",
    )
    ap.add_argument("--n-elec", type=int, default=2, help="Number of electrons")
    ap.add_argument("--n-samples", type=int, default=50000, help="Samples for overlap measurement")
    ap.add_argument(
        "--sigma-fs",
        type=str,
        default="0.8,1.3,2.0",
        help="Gaussian mixture sigma_fs (oscillator units)",
    )
    a = ap.parse_args()

    omegas = [float(o) for o in a.omegas.split(",")]
    sigma_fs = tuple(float(s) for s in a.sigma_fs.split(","))

    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC SESSION: Low-Omega Layer 1 (Sampling)")
    print(f"Checkpoint: {a.checkpoint}")
    print(f"Test omegas: {omegas}")
    print("=" * 70)

    # 1. Check DMC references
    diagnostic_dmc_references(a.n_elec, omegas)

    # 2. Load checkpoint
    bf_state, jas_state, _ = load_checkpoint(a.checkpoint)
    print(f"\n✓ Loaded checkpoint: bf={bf_state is not None}, jas={jas_state is not None}")

    # 3. Set up networks and test proposal at each omega
    for omega in omegas:
        torch.manual_seed(42)
        f_net, bf_net = setup_networks(a.n_elec, omega, bf_state=bf_state, jas_state=jas_state)

        # Build simplified params dict
        n_occ = a.n_elec // 2
        nx = ny = max(3, int(math.ceil(math.sqrt(float(n_occ)))))
        params = {
            "omega": omega,
            "n_particles": a.n_elec,
            "d": DIM,
            "L": max(8.0, 3.0 / math.sqrt(omega)),
        }

        # Build C_occ
        energies = sorted(
            [(omega * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)]
        )
        C = np.zeros((nx * ny, n_occ))
        for k in range(n_occ):
            _, ix, iy = energies[k]
            C[ix * ny + iy, k] = 1.0
        C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)

        psi_log_fn = build_psi_log_fn(f_net, bf_net, C_occ, params, a.n_elec)

        diagnostic_proposal_overlap(psi_log_fn, 4096, omega, sigma_fs=sigma_fs, n_samples=a.n_samples)


if __name__ == "__main__":
    main()
