#!/usr/bin/env python3
"""
run_jastrow_diagnostics.py
==========================
Run run_compact_analysis on the Jastrow-only (CTNNJastrowVCycle) and BF+CTNN
checkpoints from the arch_colloc comparison, and save the results for thesis
representation analysis comparison.

Target: N=6, omega=1.0 (the arch_colloc experiment conditions).

Models:
  Jastrow-only: results/arch_colloc/ctnn_vcycle.pt
  BF+CTNN:      results/arch_colloc/bf_ctnn_vcycle2.pt

Outputs saved to: results/tables/jastrow_diag_N6_w1p0.json
                   results/tables/bf_ctnn_diag_N6_w1p0.json

Run from the thesis root:
  python scripts/run_jastrow_diagnostics.py [--jastrow-only] [--device cuda:0]
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functions.Analysis import make_psi_log_fn, run_compact_analysis, sample_psi2_batch
from functions.Neural_Networks import psi_fn
from functions.Physics import coulomb_interaction_2d_vectorized
from functions.Slater_Determinant import evaluate_basis_functions_torch_batch_2d
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet

# ── constants ──────────────────────────────────────────────────────────────────
N_ELEC = 6
OMEGA = 1.0
DIM = 2
N_OCC = N_ELEC // 2

CKPT_JAS = Path("results/arch_colloc/ctnn_vcycle.pt")
CKPT_BF = Path("results/arch_colloc/bf_ctnn_vcycle2.pt")


def build_basis(n_elec: int, omega: float, device, dtype):
    n_occ = n_elec // 2
    x_dummy = torch.zeros(1, n_elec, DIM, device=device, dtype=dtype)
    C_occ = evaluate_basis_functions_torch_batch_2d(
        x_dummy, omega, n_occ, device=device, dtype=dtype
    )
    return C_occ


def build_jastrow(n_elec: int, omega: float, device, dtype) -> CTNNJastrowVCycle:
    return (
        CTNNJastrowVCycle(
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
        )
        .to(device)
        .to(dtype)
    )


def build_backflow(omega: float, device, dtype) -> CTNNBackflowNet:
    return (
        CTNNBackflowNet(
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
        )
        .to(device)
        .to(dtype)
    )


def load_states(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt.get("jas_state"), ckpt.get("bf_state")


def run_analysis(label: str, f_net, bf_net, C_occ, params, device, dtype):
    spin = torch.cat(
        [
            torch.zeros(N_OCC, dtype=torch.long),
            torch.ones(N_OCC, dtype=torch.long),
        ]
    ).to(device)

    std = 1.5 / math.sqrt(OMEGA)

    print(f"\n{'='*60}")
    print(f"  Running analysis: {label}")
    print(f"  backflow_net: {'yes' if bf_net is not None else 'None (Jastrow-only)'}")
    print(f"{'='*60}")

    out = run_compact_analysis(
        f_net=f_net,
        psi_fn=psi_fn,
        C_occ=C_occ,
        params=params,
        std=std,
        make_psi_log_fn=make_psi_log_fn,
        sample_psi2_batch=sample_psi2_batch,
        compute_coulomb_interaction=coulomb_interaction_2d_vectorized,
        spin=spin,
        backflow_net=bf_net,
        save_json=f"jastrow_diag_{label}_N{N_ELEC}_w1p0",
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--jastrow-only", action="store_true", help="Run only the Jastrow-only model (skip BF)"
    )
    parser.add_argument(
        "--bf-only", action="store_true", help="Run only the BF+CTNN model (skip Jastrow-only)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64

    params = {
        "omega": OMEGA,
        "N": N_ELEC,
        "dim": DIM,
    }

    C_occ = build_basis(N_ELEC, OMEGA, device, dtype)

    # ── Jastrow-only ───────────────────────────────────────────────────────────
    if not args.bf_only:
        print(f"\nLoading Jastrow checkpoint: {CKPT_JAS}")
        jas_state, _ = load_states(CKPT_JAS, device)
        f_net_jas = build_jastrow(N_ELEC, OMEGA, device, dtype)
        if jas_state is not None:
            f_net_jas.load_state_dict(jas_state)
        else:
            # ctnn_vcycle.pt may store the full model state directly
            ckpt = torch.load(CKPT_JAS, map_location=device)
            # Try direct state dict
            if isinstance(ckpt, dict) and not any(k in ckpt for k in ("jas_state", "bf_state")):
                f_net_jas.load_state_dict(ckpt)
            else:
                print("[WARN] Could not find jas_state; using random init for Jastrow.")

        run_analysis("jastrow", f_net_jas, None, C_occ, params, device, dtype)

    # ── BF+CTNN ────────────────────────────────────────────────────────────────
    if not args.jastrow_only:
        print(f"\nLoading BF+CTNN checkpoint: {CKPT_BF}")
        jas_state, bf_state = load_states(CKPT_BF, device)
        f_net_bf = build_jastrow(N_ELEC, OMEGA, device, dtype)
        bf_net = build_backflow(OMEGA, device, dtype)
        if jas_state is not None:
            f_net_bf.load_state_dict(jas_state)
        if bf_state is not None:
            bf_net.load_state_dict(bf_state)

        run_analysis("bf_ctnn", f_net_bf, bf_net, C_occ, params, device, dtype)

    print("\nDone. Results saved to results/tables/.")


if __name__ == "__main__":
    main()
