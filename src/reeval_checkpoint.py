"""
Re-evaluate a saved checkpoint with much more careful VMC sampling.
Uses longer burn-in, more samples, and multiple independent chains.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import psi_fn
from functions.Physics import compute_coulomb_interaction
from jastrow_architectures import (
    CTNNBackflowStyleJastrow,
    CTNNJastrow,
    CTNNJastrowVCycle,
)

DEVICE = "cpu"
DTYPE = torch.float64
N_ELEC = 6
DIM = 2
OMEGA = 1.0
E_DMC = 20.15932

ARCHS = {
    "ctnn_vcycle": lambda: CTNNJastrowVCycle(
        n_particles=N_ELEC,
        d=DIM,
        omega=OMEGA,
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
    ),
    "ctnn_bfstyle": lambda: CTNNBackflowStyleJastrow(
        n_particles=N_ELEC,
        d=DIM,
        omega=OMEGA,
        msg_hidden=32,
        msg_layers=2,
        hidden=32,
        layers=2,
        n_steps=2,
        act="silu",
        aggregation="sum",
        use_spin=True,
        same_spin_only=False,
        readout_hidden=64,
        readout_layers=2,
    ),
    "ctnn": lambda: CTNNJastrow(
        n_particles=N_ELEC,
        d=DIM,
        omega=OMEGA,
        node_hidden=32,
        edge_hidden=32,
        n_mp_steps=2,
        msg_layers=2,
        node_layers=2,
        readout_hidden=32,
        readout_layers=2,
        act="silu",
    ),
}


def setup():
    N, d = N_ELEC, DIM
    n_occ = N // 2
    nx = ny = 3
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(
        omega=OMEGA,
        n_particles=N,
        d=d,
        L=L,
        n_grid=80,
        nx=nx,
        ny=ny,
        basis="cart",
        device="cpu",
        dtype="float64",
    )
    energies = sorted([(OMEGA * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)])
    C = np.zeros((nx * ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
    p = config.get().as_dict()
    p.update(device=DEVICE, torch_dtype=DTYPE, E=E_DMC)
    return C_occ, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--arch", type=str, required=True, choices=list(ARCHS))
    ap.add_argument("--n-samples", type=int, default=16000)
    ap.add_argument("--n-chains", type=int, default=3, help="Independent eval chains to average")
    ap.add_argument("--burn-in", type=int, default=2000)
    ap.add_argument("--sampler-steps", type=int, default=120)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--step-sigma", type=float, default=0.08)
    a = ap.parse_args()

    C_occ, params = setup()
    net = ARCHS[a.arch]().to(DEVICE).to(DTYPE)  # type: ignore[attr-defined]
    ckpt = torch.load(a.ckpt, map_location=DEVICE)
    net.load_state_dict(ckpt["state"])
    net.eval()
    np_ = sum(p.numel() for p in net.parameters())
    print(f"Loaded {a.arch} ({np_:,} params) from {a.ckpt}")
    if "E" in ckpt:
        print(f"  Original checkpoint E = {ckpt['E']:.6f}")
    print(f"  Re-evaluating with {a.n_chains} independent chains x {a.n_samples} samples")
    print(
        f"  burn_in={a.burn_in}, sampler_steps={a.sampler_steps}, thin={a.thin}, "
        f"step_sigma={a.step_sigma}"
    )
    sys.stdout.flush()

    chain_Es = []
    chain_ses = []
    for i in range(a.n_chains):
        print(f"\n  Chain {i+1}/{a.n_chains}:")
        sys.stdout.flush()
        t0 = time.time()
        vmc = evaluate_energy_vmc(
            net,
            C_occ,
            psi_fn=psi_fn,
            compute_coulomb_interaction=compute_coulomb_interaction,
            backflow_net=None,
            params=params,
            n_samples=a.n_samples,
            batch_size=512,
            sampler_steps=a.sampler_steps,
            sampler_step_sigma=a.step_sigma,
            lap_mode="exact",
            persistent=True,
            sampler_burn_in=a.burn_in,
            sampler_thin=a.thin,
            progress=True,
        )
        E, se = vmc["E_mean"], vmc["E_stderr"]
        dt = time.time() - t0
        err = (E - E_DMC) / abs(E_DMC) * 100
        print(f"    E = {E:.6f} ± {se:.6f}  err={err:+.3f}%  ({dt/60:.1f}min)")
        sys.stdout.flush()
        chain_Es.append(E)
        chain_ses.append(se)

    # Aggregate: weighted mean of independent chains
    Es = np.array(chain_Es)
    ses = np.array(chain_ses)
    w = 1.0 / ses**2
    E_final = np.sum(w * Es) / np.sum(w)
    se_final = 1.0 / np.sqrt(np.sum(w))
    err_final = (E_final - E_DMC) / abs(E_DMC) * 100

    print(f"\n{'='*60}")
    print(f"FINAL (weighted mean of {a.n_chains} chains):")
    print(f"  E = {E_final:.6f} ± {se_final:.6f}")
    print(f"  err = {err_final:+.4f}%")
    print(
        f"  Target: 20.17  →  "
        f"{'REACHED' if E_final <= 20.17 else f'gap = {E_final - 20.17:.6f}'}"
    )
    print(f"  E_DMC = {E_DMC}")
    print(f"{'='*60}")
    sys.stdout.flush()
