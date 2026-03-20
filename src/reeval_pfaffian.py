"""
Re-evaluate a Pfaffian checkpoint with careful VMC sampling.
Supports both pure Pfaffian and Pfaffian+Backflow checkpoints.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from functions.Energy import evaluate_energy_vmc
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import evaluate_basis_functions_torch_batch_2d
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet

# Import Pfaffian components from run_pfaffian
from run_pfaffian import PfaffianNet, psi_fn_pfaffian, pfaffian_6x6

_manual = os.environ.get("CUDA_MANUAL_DEVICE")
if _manual is not None and torch.cuda.is_available():
    DEVICE = f"cuda:{_manual}" if _manual.isdigit() else _manual
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
DTYPE = torch.float64
N_ELEC = 6
DIM = 2
OMEGA = 1.0
E_DMC = 20.15932

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"


def setup():
    N, d = N_ELEC, DIM
    n_occ = N // 2
    nx = ny = 3
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(
        omega=OMEGA, n_particles=N, d=d, L=L, n_grid=80,
        nx=nx, ny=ny, basis="cart", device=DEVICE, dtype="float64",
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
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--burn-in", type=int, default=3000)
    ap.add_argument("--sampler-steps", type=int, default=150)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--step-sigma", type=float, default=0.06)
    a = ap.parse_args()

    C_occ, params = setup()

    # Load checkpoint
    ckpt = torch.load(a.ckpt, map_location=DEVICE)
    print(f"Loaded checkpoint: {a.ckpt}")
    print(f"  Original E = {ckpt.get('E', '?')}, err = {ckpt.get('err', '?')}%")

    # Reconstruct PfaffianNet
    pfc = ckpt.get("pf_config", {})
    nx = pfc.get("nx", 3)
    ny = pfc.get("ny", 3)
    n_basis = pfc.get("n_basis", 9)
    n_occ = pfc.get("n_occ", 3)

    pfaffian_net = PfaffianNet(
        n_basis, n_occ, C_occ, nx, ny,
        use_mlp=("pair_mlp.layers.0.weight" in ckpt["pf_state"]),
    ).to(DEVICE).to(DTYPE)
    pfaffian_net.load_state_dict(ckpt["pf_state"])
    pfaffian_net.eval()
    n_pf = sum(p.numel() for p in pfaffian_net.parameters())
    print(f"  PfaffianNet: {n_pf:,} params")

    # Reconstruct Jastrow
    f_net = CTNNJastrowVCycle(
        n_particles=N_ELEC, d=DIM, omega=OMEGA,
        node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
        n_down=1, n_up=1, msg_layers=1, node_layers=1,
        readout_hidden=64, readout_layers=2, act="silu",
    ).to(DEVICE).to(DTYPE)
    f_net.load_state_dict(ckpt["jas_state"])
    f_net.eval()

    # Reconstruct backflow if present (check ckpt first, fallback to bf_ctnn_vcycle.pt)
    bf_net = None
    bf_source = None
    if "bf_state" in ckpt and ckpt["bf_state"]:
        bf_source = ckpt
    elif "pfaffian_bf" in a.ckpt or "_bf" in Path(a.ckpt).stem:
        # Checkpoint was trained with backflow but didn't save it — load from original
        bf_orig_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
        if bf_orig_path.exists():
            print(f"  (Loading backflow from original: {bf_orig_path})")
            bf_source = torch.load(bf_orig_path, map_location=DEVICE)

    if bf_source is not None:
        bfc = bf_source.get("bf_config", {})
        bf_net = CTNNBackflowNet(
            d=bfc.get("d", 2),
            msg_hidden=bfc.get("msg_hidden", 48),
            msg_layers=bfc.get("msg_layers", 2),
            hidden=bfc.get("hidden", 48),
            layers=bfc.get("layers", 2),
            act=bfc.get("act", "silu"),
            aggregation=bfc.get("aggregation", "sum"),
            use_spin=bfc.get("use_spin", True),
            same_spin_only=bfc.get("same_spin_only", False),
            out_bound=bfc.get("out_bound", "tanh"),
            bf_scale_init=bfc.get("bf_scale_init", 0.15),
            zero_init_last=bfc.get("zero_init_last", True),
            omega=bfc.get("omega", 1.0),
        ).to(DEVICE).to(DTYPE)
        bf_net.load_state_dict(bf_source["bf_state"])
        bf_net.eval()
        n_bf = sum(p.numel() for p in bf_net.parameters())
        print(f"  Backflow: {n_bf:,} params  (frozen)")
    else:
        print(f"  Backflow: NONE (pure Pfaffian)")

    # VMC evaluation wrapper
    up = N_ELEC // 2
    spin = torch.cat([
        torch.zeros(up, dtype=torch.long),
        torch.ones(N_ELEC - up, dtype=torch.long),
    ]).to(DEVICE)

    def _psi_fn_wrap(f_net_wrap, x, C_occ, *, backflow_net=None, orbital_bf_net=None, spin=None, params=None):
        lp, ps = psi_fn_pfaffian(pfaffian_net, f_net, x, params, spin=spin, bf_net=bf_net)
        return lp, ps

    C_dummy = torch.eye(9, 3, device=DEVICE, dtype=DTYPE)

    print(f"\n  Re-evaluating with {a.n_chains} chains x {a.n_samples} samples")
    print(f"  burn_in={a.burn_in}, steps={a.sampler_steps}, thin={a.thin}, sigma={a.step_sigma}")
    sys.stdout.flush()

    chain_Es = []
    chain_ses = []
    for i in range(a.n_chains):
        print(f"\n  Chain {i+1}/{a.n_chains}:")
        sys.stdout.flush()
        t0 = time.time()
        vmc = evaluate_energy_vmc(
            f_net, C_dummy,
            psi_fn=_psi_fn_wrap,
            compute_coulomb_interaction=compute_coulomb_interaction,
            params=params,
            n_samples=a.n_samples, batch_size=512,
            sampler_steps=a.sampler_steps,
            sampler_step_sigma=a.step_sigma,
            lap_mode="exact", persistent=True,
            sampler_burn_in=a.burn_in,
            sampler_thin=a.thin, progress=True,
        )
        E = float(vmc["E_mean"])
        se = float(vmc["E_stderr"])
        dt = time.time() - t0
        err = (E - E_DMC) / abs(E_DMC) * 100
        print(f"    E = {E:.6f} +- {se:.6f}  err={err:+.3f}%  ({dt/60:.1f}min)")
        sys.stdout.flush()
        chain_Es.append(E)
        chain_ses.append(se)

    Es = np.array(chain_Es)
    ses = np.array(chain_ses)
    w = 1.0 / ses**2
    E_final = np.sum(w * Es) / np.sum(w)
    se_final = 1.0 / np.sqrt(np.sum(w))
    err_final = (E_final - E_DMC) / abs(E_DMC) * 100

    print(f"\n{'='*60}")
    print(f"FINAL (weighted mean of {a.n_chains} chains):")
    print(f"  E = {E_final:.6f} +- {se_final:.6f}")
    print(f"  err = {err_final:+.4f}%")
    print(f"  DMC = {E_DMC}")
    print(f"{'='*60}")
