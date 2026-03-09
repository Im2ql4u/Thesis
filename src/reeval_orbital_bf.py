"""Re-evaluate orbital BF checkpoint with wider MCMC settings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import psi_fn
from functions.Physics import compute_coulomb_interaction
from jastrow_architectures import CTNNJastrowVCycle
from PINN import OrbitalBackflowNet

DEVICE = "cpu"
DTYPE = torch.float64
N_ELEC = 6
DIM = 2
OMEGA = 1.0
E_DMC = 20.15932
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"

# Setup
n_occ = N_ELEC // 2
config.update(
    omega=OMEGA,
    n_particles=N_ELEC,
    d=DIM,
    L=8.0,
    n_grid=80,
    nx=3,
    ny=3,
    basis="cart",
    device="cpu",
    dtype="float64",
)
energies = sorted([(OMEGA * (ix + iy + 1), ix, iy) for ix in range(3) for iy in range(3)])
C = np.zeros((9, n_occ))
for k in range(n_occ):
    _, ix, iy = energies[k]
    C[ix * 3 + iy, k] = 1.0
C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
params = config.get().as_dict()
params.update(device=DEVICE, torch_dtype=DTYPE, E=E_DMC)

# Load checkpoint
ckpt = torch.load(RESULTS_DIR / "orb_bf_vcycle.pt", map_location=DEVICE)
cfg = ckpt["bf_config"]

f_net = (
    CTNNJastrowVCycle(
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
    )
    .to(DEVICE)
    .to(DTYPE)
)
f_net.load_state_dict(ckpt["jas_state"])

orbital_bf_net = (
    OrbitalBackflowNet(
        d=cfg["d"],
        n_occ=cfg["n_occ"],
        msg_hidden=cfg["msg_hidden"],
        msg_layers=cfg["msg_layers"],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        act=cfg["act"],
        aggregation=cfg["aggregation"],
        use_spin=cfg["use_spin"],
        same_spin_only=cfg["same_spin_only"],
        out_bound=cfg["out_bound"],
        bf_scale_init=cfg["bf_scale_init"],
        zero_init_last=cfg["zero_init_last"],
        omega=cfg["omega"],
    )
    .to(DEVICE)
    .to(DTYPE)
)
orbital_bf_net.load_state_dict(ckpt["bf_state"])
print(f"Loaded orbital BF: bf_scale={orbital_bf_net.bf_scale.item():.6f}")

# Try 3 different MCMC settings
for label, ss, steps, burn, thin, ns in [
    ("probe-like", 0.12, 40, 200, 2, 8000),
    ("medium", 0.15, 60, 500, 3, 10000),
    ("wide", 0.20, 80, 800, 4, 10000),
]:
    print(f"\n--- {label}: sigma={ss}, steps={steps}, burn={burn}, thin={thin}, n={ns} ---")
    sys.stdout.flush()
    vmc = evaluate_energy_vmc(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        orbital_bf_net=orbital_bf_net,
        params=params,
        n_samples=ns,
        batch_size=512,
        sampler_steps=steps,
        sampler_step_sigma=ss,
        lap_mode="exact",
        persistent=True,
        sampler_burn_in=burn,
        sampler_thin=thin,
        progress=True,
    )
    E, se = vmc["E_mean"], vmc["E_stderr"]
    err = (E - E_DMC) / abs(E_DMC) * 100
    acc = vmc.get("accept_rate", vmc.get("accept_avg", "?"))
    print(f"  E = {E:.6f} ± {se:.6f}   err = {err:+.3f}%   accept = {acc}")
    sys.stdout.flush()
