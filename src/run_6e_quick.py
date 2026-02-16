"""
Quick plain residual (collocation) fine-tune — baseline sanity check.
30 epochs, Adam, from best checkpoint. No SR, no Fisher, no tricks.
Just the standard recipe: top-K screened collocation + Huber var loss + smoothness.
"""

import math, sys, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

import config
from PINN import PINN, CTNNBackflowNet
from functions.Neural_Networks import psi_fn, _laplacian_logpsi_exact
from functions.Physics import compute_coulomb_interaction
from functions.Energy import evaluate_energy_vmc

E_DMC = 11.78484
DEVICE = "cpu"
DTYPE = torch.float64
CKPT_DIR = "/Users/aleksandersekkelsten/thesis/results/models"
N, DIM, OMEGA = 6, 2, 0.5
ELL = 1.0 / math.sqrt(OMEGA)


def setup():
    n_occ = N // 2
    nx = ny = 3
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(omega=OMEGA, n_particles=N, d=DIM, L=L, n_grid=80,
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


def make_nets():
    f = PINN(n_particles=N, d=DIM, omega=OMEGA, dL=8, hidden_dim=64,
             n_layers=2, act="gelu", init="xavier", use_gate=True,
             use_pair_attn=False).to(DEVICE).to(DTYPE)
    b = CTNNBackflowNet(d=DIM, msg_hidden=32, msg_layers=1, hidden=32,
                        layers=2, act="silu", aggregation="sum", use_spin=True,
                        same_spin_only=False, out_bound="tanh",
                        bf_scale_init=0.7, zero_init_last=False,
                        omega=OMEGA).to(DEVICE).to(DTYPE)
    return f, b


def make_psi(f, b, C, p):
    up = N // 2
    spin = torch.cat([torch.zeros(up, dtype=torch.long),
                      torch.ones(N - up, dtype=torch.long)]).to(DEVICE)
    def fn(y):
        lp, _ = psi_fn(f, y, C, backflow_net=b, spin=spin, params=p)
        return lp
    return fn, spin


@torch.no_grad()
def topk(psi_fn, n_keep, oversample, sigma, bs=4096):
    M = oversample * n_keep
    x = torch.randn(M, N, DIM, device=DEVICE, dtype=DTYPE) * sigma
    Nd = N * DIM
    log_q = -0.5 * Nd * math.log(2 * math.pi * sigma**2) - x.reshape(M, -1).pow(2).sum(-1) / (2 * sigma**2)
    lp = torch.cat([psi_fn(x[i:i+bs]) for i in range(0, M, bs)])
    log_w = 2.0 * lp - log_q
    log_w[~torch.isfinite(log_w)] = -1e10
    _, idx = torch.topk(log_w, n_keep)
    return x[idx].clone()


def local_energy(psi_fn, x):
    x = x.detach().requires_grad_(True)
    _, g2, lap = _laplacian_logpsi_exact(psi_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap.view(B) + g2.view(B))
    V = 0.5 * OMEGA**2 * (x**2).sum(dim=(1, 2)) + compute_coulomb_interaction(x).view(B)
    return T + V


def smooth_pen(bf, x, spin, ns=32):
    xs = x[:ns].detach().requires_grad_(True)
    dx = bf(xs, spin)
    s = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    for _ in range(2):
        v = torch.empty_like(xs).bernoulli_(0.5).mul_(2).add_(-1)
        for k in range(DIM):
            g1 = torch.autograd.grad(dx[:,:,k].sum(), xs, create_graph=True, retain_graph=True)[0]
            Hv = torch.autograd.grad((g1*v).sum(), xs, create_graph=True, retain_graph=True)[0]
            s = s + ((v*Hv).sum(dim=(1,2))**2).mean()
    return s / (2 * DIM)


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


def train(f, bf, C, p, n_ep=30, lr=3e-4, n_coll=2048, oversample=10,
          huber_d=0.5, smooth_l=1e-3, mb=256, clip=0.5, qtrim=0.02,
          label=""):
    sigma = 1.3 * ELL
    psi, spin = make_psi(f, bf, C, p)
    opt = torch.optim.Adam(list(f.parameters()) + list(bf.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep, eta_min=lr/50)

    print(f"\n{'='*60}\n  {label}\n  {n_ep}ep lr={lr}\n{'='*60}")
    t0 = time.time()
    best_var = float("inf")
    best_fs, best_bs = {}, {}

    for ep in range(n_ep):
        f.eval(); bf.eval()
        X = topk(psi, n_coll, oversample, sigma)
        f.train(); bf.train()
        opt.zero_grad()

        all_EL = []
        nb = max(1, math.ceil(X.shape[0] / mb))
        for i in range(0, X.shape[0], mb):
            xm = X[i:i+mb]
            EL = local_energy(psi, xm).view(-1)
            good = torch.isfinite(EL)
            if not good.all(): EL = EL[good]
            if EL.numel() == 0: continue
            if qtrim > 0 and EL.numel() > 20:
                lo = torch.quantile(EL.detach(), qtrim)
                hi = torch.quantile(EL.detach(), 1 - qtrim)
                EL = EL[(EL.detach() >= lo) & (EL.detach() <= hi)]
                if EL.numel() == 0: continue
            all_EL.append(EL.detach())
            mu = EL.mean().detach()
            res = EL - mu
            loss = F.huber_loss(res, torch.zeros_like(res), delta=huber_d)
            if smooth_l > 0:
                loss = loss + smooth_l * smooth_pen(bf, xm, spin)
            (loss / nb).backward()

        if not all_EL: continue
        ELc = torch.cat(all_EL)
        Em, Ev = ELc.mean().item(), ELc.var().item()

        nn.utils.clip_grad_norm_(list(f.parameters()) + list(bf.parameters()), clip)
        opt.step(); sched.step()

        if math.isfinite(Ev) and Ev < best_var * 0.999:
            best_var = Ev
            best_fs = {k: v.clone() for k, v in f.state_dict().items()}
            best_bs = {k: v.clone() for k, v in bf.state_dict().items()}

        if ep % 5 == 0:
            err = abs(Em - E_DMC) / E_DMC * 100
            with torch.no_grad():
                bf.eval()
                bm = bf(X[:64], spin).norm(dim=-1).mean().item()
                bf.train()
            print(f"[{ep:3d}] E={Em:.5f} var={Ev:.3e} |dx|={bm:.3f} err={err:.2f}%")
            sys.stdout.flush()

    if best_fs: f.load_state_dict(best_fs)
    if best_bs: bf.load_state_dict(best_bs)
    dt = time.time() - t0
    print(f"  Done {dt:.0f}s ({dt/60:.1f}min), best var={best_var:.3e}")
    return f, bf


if __name__ == "__main__":
    C, p = setup()
    BASE = os.path.join(CKPT_DIR, "6e_sir_topk_baseline.pt")
    ckpt = torch.load(BASE, map_location=DEVICE)
    results = {}

    # ── 1: plain Adam fine-tune (standard recipe, 30ep) ──
    f1, bf1 = make_nets()
    f1.load_state_dict(ckpt["f_net"]); bf1.load_state_dict(ckpt["bf_net"])
    print(f"Loaded {BASE}")
    f1, bf1 = train(f1, bf1, C, p, n_ep=30, lr=1e-4, label="adam_finetune_30ep")
    r1 = evaluate(f1, C, p, bf1, label="adam_finetune")
    err1 = abs(r1["E_mean"] - E_DMC) / E_DMC * 100
    results["adam_finetune"] = (r1["E_mean"], r1["E_stderr"], err1)

    # ── 2: higher LR burst (3e-4, 15ep — can it escape?) ──
    f2, bf2 = make_nets()
    f2.load_state_dict(ckpt["f_net"]); bf2.load_state_dict(ckpt["bf_net"])
    f2, bf2 = train(f2, bf2, C, p, n_ep=15, lr=3e-4, label="adam_burst_15ep")
    r2 = evaluate(f2, C, p, bf2, label="adam_burst")
    err2 = abs(r2["E_mean"] - E_DMC) / E_DMC * 100
    results["adam_burst"] = (r2["E_mean"], r2["E_stderr"], err2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    results["start (topk_baseline)"] = (11.8265, 0.003, 0.35)
    results["ref (bf_0.7 joint)"] = (11.8233, 0.00298, 0.33)
    for name, (E, std, err) in sorted(results.items(), key=lambda x: x[1][2]):
        print(f"  {name:30s}  E={E:.6f} +/- {std:.6f}  err={err:.2f}%")
    print("Done.")
