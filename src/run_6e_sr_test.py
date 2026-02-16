"""
Quick VMC+SR test using the EXISTING train_model_sr_energy.
Trained PINN + fresh CTNNBackflowNet. 500 SR steps to verify the ansatz works.
"""
import math, sys, time, os
import numpy as np
import torch

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, CTNNBackflowNet
from functions.Neural_Networks import psi_fn
from functions.Physics import compute_coulomb_interaction
from functions.Stochastic_Reconfiguration import train_model_sr_energy
from functions.Energy import evaluate_energy_vmc

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"
N_P, DIM, OMEGA = 6, 2, 0.5


def setup():
    n_occ = N_P // 2
    nx = ny = 3
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(omega=OMEGA, n_particles=N_P, d=DIM, L=L, n_grid=80,
                  nx=nx, ny=ny, basis="cart", device="cpu", dtype="float64")
    energies = []
    for ix in range(nx):
        for iy in range(ny):
            energies.append((OMEGA * (ix + iy + 1), ix, iy))
    energies.sort(key=lambda t: t[0])
    C = np.zeros((nx * ny, n_occ), dtype=np.float64)
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C, dtype=DTYPE, device=DEVICE)
    p = config.get().as_dict()
    p["device"] = DEVICE; p["torch_dtype"] = DTYPE; p["E"] = E_DMC
    return C_occ, p


def make_nets(bf_scale=0.7):
    f = PINN(n_particles=N_P, d=DIM, omega=OMEGA, dL=8, hidden_dim=64,
             n_layers=2, act="gelu", init="xavier", use_gate=True,
             use_pair_attn=False).to(DEVICE).to(DTYPE)
    b = CTNNBackflowNet(d=DIM, msg_hidden=32, msg_layers=1, hidden=32,
                        layers=2, act="silu", aggregation="sum", use_spin=True,
                        same_spin_only=False, out_bound="tanh",
                        bf_scale_init=bf_scale, zero_init_last=False,
                        omega=OMEGA).to(DEVICE).to(DTYPE)
    return f, b


def evaluate(f, C, p, bf, label=""):
    print(f"\n-- VMC eval: {label} --")
    r = evaluate_energy_vmc(
        f, C, psi_fn=psi_fn, compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=bf, params=p, n_samples=15000, batch_size=512,
        sampler_steps=50, sampler_step_sigma=0.12, lap_mode="exact",
        persistent=True, sampler_burn_in=300, sampler_thin=3, progress=True)
    E, std = r["E_mean"], r["E_stderr"]
    err = abs(E - E_DMC) / E_DMC * 100
    print(f"  E = {E:.6f} +/- {std:.6f}  err={err:.2f}%")
    return r


if __name__ == "__main__":
    C_occ, params = setup()
    PINN_CKPT = os.path.join(CKPT_DIR, "pinn_6e_ckpt.pt")

    # Load trained PINN + fresh CTNN
    f_net, bf_net = make_nets(bf_scale=0.7)
    ckpt = torch.load(PINN_CKPT, map_location=DEVICE)
    if "f_net" in ckpt:
        f_net.load_state_dict(ckpt["f_net"])
    else:
        f_net.load_state_dict(ckpt)
    print(f"Loaded PINN from {PINN_CKPT}")
    print(f"CTNN is FRESH (untrained), bf_scale=0.7")

    # Set up spin
    up = N_P // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N_P - up, dtype=torch.long)]).to(DEVICE)

    # Eval starting point
    r0 = evaluate(f_net, C_occ, params, bf_net, label="start (pinn+fresh_ctnn)")
    err0 = abs(r0["E_mean"] - E_DMC) / E_DMC * 100
    print(f"  Starting err = {err0:.2f}%")

    # SR training using YOUR existing code
    print(f"\n{'='*65}")
    print("  Running train_model_sr_energy (your existing SR)")
    print(f"  500 steps, step_size=0.01, damping=1e-2")
    print(f"{'='*65}")
    sys.stdout.flush()

    t0 = time.time()
    f_net, bf_net = train_model_sr_energy(
        f_net,
        C_occ,
        psi_fn=psi_fn,
        compute_coulomb_interaction=compute_coulomb_interaction,
        backflow_net=bf_net,
        spin=spin,
        params=params,
        n_sr_steps=500,
        log_every=10,
        step_size=0.01,
        max_param_step=0.01,
        damping=1e-2,
        total_rows=3000,
        micro_batch=600,
        sampler_steps=50,
        sampler_step_sigma=0.10,
        lap_mode="exact",
        store_dtype=torch.float64,
    )
    dt = time.time() - t0
    print(f"  SR done in {dt:.1f}s ({dt/60:.1f}min)")

    # Save
    torch.save({"f_net": f_net.state_dict(), "bf_net": bf_net.state_dict()},
               os.path.join(CKPT_DIR, "6e_vmc_sr_existing.pt"))

    # Final eval
    r1 = evaluate(f_net, C_occ, params, bf_net, label="after SR 500 steps")
    err1 = abs(r1["E_mean"] - E_DMC) / E_DMC * 100

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  DMC target                       E={E_DMC:.6f}            err=0.00%")
    print(f"  start (pinn+fresh_ctnn)          E={r0['E_mean']:.6f} +/- {r0['E_stderr']:.6f}  err={err0:.2f}%")
    print(f"  after SR 500 steps               E={r1['E_mean']:.6f} +/- {r1['E_stderr']:.6f}  err={err1:.2f}%")
    print(f"  best collocation (bf_0.7 joint)  E=11.823300 +/- 0.002980  err=0.33%")
    print("Done.")
