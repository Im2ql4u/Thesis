"""
Fast collocation comparison — new Jastrow architectures only
=============================================================
N=6, ω=1.0, E_DMC=20.15932

Screened collocation + Huber(E_L − E_eff) residual loss.
Three new architectures (no baseline PINN — already known: 20.210 ± 0.003).

Target: < 30 min total wall time.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import _laplacian_logpsi_exact, psi_fn
from functions.Physics import compute_coulomb_interaction
from jastrow_architectures import (
    CTNNBackflowStyleJastrow,
    CTNNJastrow,
    CTNNJastrowAttnGlobal,
    CTNNJastrowVCycle,
    DeepSetJastrow,
    TriadicDeepSetJastrow,
)

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
E_TARGET = 20.17

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"


# ─── System ───
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
        device=DEVICE,
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


# ─── Models ───
ARCHS = {
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
    "ctnn_vcycle_tiny": lambda: CTNNJastrowVCycle(
        n_particles=N_ELEC,
        d=DIM,
        omega=OMEGA,
        node_hidden=16,
        edge_hidden=16,
        bottleneck_hidden=8,
        n_down=1,
        n_up=1,
        msg_layers=1,
        node_layers=1,
        readout_hidden=32,
        readout_layers=1,
        act="silu",
    ),
    "ctnn_attn_global": lambda: CTNNJastrowAttnGlobal(
        n_particles=N_ELEC,
        d=DIM,
        omega=OMEGA,
        node_hidden=32,
        edge_hidden=32,
        n_mp_steps=3,
        msg_layers=2,
        node_layers=2,
        readout_hidden=64,
        readout_layers=2,
        act="silu",
    ),
    "deepset": lambda: DeepSetJastrow(
        n_particles=N_ELEC,
        d=DIM,
        omega=OMEGA,
        pair_hidden=64,
        pair_layers=3,
        pair_out=16,
        readout_hidden=64,
        readout_layers=2,
        act="gelu",
    ),
    "triadic_deepset": lambda: TriadicDeepSetJastrow(
        n_particles=N_ELEC,
        d=DIM,
        omega=OMEGA,
        pair_hidden=64,
        pair_layers=3,
        pair_out=24,
        triad_hidden=32,
        readout_hidden=64,
        readout_layers=2,
        act="gelu",
    ),
}


# ─── Helpers ───
def compute_EL(psi_log_fn, x, omega):
    x = x.detach().requires_grad_(True)
    _, g2, lap = _laplacian_logpsi_exact(psi_log_fn, x)
    B = x.shape[0]
    T = -0.5 * (lap.view(B) + g2.view(B))
    V = 0.5 * omega**2 * (x**2).sum(dim=(1, 2)) + compute_coulomb_interaction(x).view(B)
    return T + V


def huber(r, d):
    a = r.abs()
    return torch.where(a <= d, 0.5 * r**2, d * (a - 0.5 * d))


def sample_gauss(n, omega, sigma_f=1.3):
    s = sigma_f / math.sqrt(omega)
    x = torch.randn(n, N_ELEC, DIM, device=DEVICE, dtype=DTYPE) * s
    Nd = N_ELEC * DIM
    lq = -0.5 * Nd * math.log(2 * math.pi * s**2) - x.reshape(n, -1).pow(2).sum(-1) / (2 * s**2)
    return x, lq


@torch.no_grad()
def screened_colloc(
    psi_log_fn, n_keep, omega, oversample=8, sigma_fs=(0.8, 1.3, 2.0), explore=0.10
):
    n_cand = oversample * n_keep
    nc = len(sigma_fs)
    xs, lqs = [], []
    for i, sf in enumerate(sigma_fs):
        ni = n_cand // nc if i < nc - 1 else n_cand - sum(n_cand // nc for _ in range(i))
        xi, lqi = sample_gauss(ni, omega, sf)
        xs.append(xi)
        lqs.append(lqi)
    x_all = torch.cat(xs)
    lq_all = torch.cat(lqs)
    lp2 = []
    for i in range(0, len(x_all), 4096):
        lp2.append(2.0 * psi_log_fn(x_all[i : i + 4096]))
    lr = torch.cat(lp2) - lq_all
    n_exp = int(max(0, min(n_keep - 1, round(explore * n_keep))))
    n_top = n_keep - n_exp
    _, idx = torch.sort(lr, descending=True)
    sel = idx[:n_top]
    if n_exp > 0 and idx[n_top:].numel() > 0:
        rest = idx[n_top:]
        sel = torch.cat([sel, rest[torch.randperm(len(rest))[:n_exp]]])
    return x_all[sel[:n_keep]].clone()


# ─── Collocation trainer ───
def train_colloc(
    f_net,
    C_occ,
    params,
    *,
    n_epochs=250,
    lr=3e-4,
    lr_min_frac=0.02,
    phase1_frac=0.25,
    alpha_end=0.70,
    n_coll=2048,
    oversample=8,
    micro_batch=256,
    grad_clip=0.5,
    replay_frac=0.25,
    qtrim=0.02,
    huber_d=1.0,
    print_every=10,
    patience=60,
    vmc_every=40,
    vmc_n=6000,
    tag="",
):
    omega = OMEGA
    E_ref = E_DMC
    up = N_ELEC // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_ELEC - up, dtype=torch.long)]
    ).to(DEVICE)

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=None, spin=spin, params=params)
        return lp

    prms = [p for p in f_net.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in prms)
    opt = torch.optim.Adam(prms, lr=lr)
    lr_min = lr * lr_min_frac
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda ep: (
            lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))
        )
        / lr,
    )
    p1end = int(phase1_frac * n_epochs)

    print(f"  Training: {n_epochs} ep, {n_coll} colloc pts, {ntr:,} params")
    sys.stdout.flush()

    t0 = time.time()
    hist = []
    best_var = best_vmc_err = float("inf")
    best_st = best_vmc_st = {}
    best_vmc_E = None
    no_imp = 0
    replay_X = None

    for ep in range(n_epochs):
        ept0 = time.time()
        alpha = (
            0.0
            if ep < p1end
            else 0.5
            * alpha_end
            * (1 - math.cos(math.pi * (ep - p1end) / max(1, n_epochs - p1end - 1)))
        )

        f_net.eval()
        X = screened_colloc(psi_log_fn, n_coll, omega, oversample=oversample)

        if replay_frac > 0 and replay_X is not None and replay_X.numel() > 0:
            n_rep = int(min(n_coll - 1, round(replay_frac * n_coll)))
            if n_rep > 0:
                n_new = n_coll - n_rep
                if replay_X.shape[0] >= n_rep:
                    idx = torch.randperm(replay_X.shape[0], device=replay_X.device)[:n_rep]
                    rep = replay_X[idx]
                else:
                    rep_idx = torch.randint(0, replay_X.shape[0], (n_rep,), device=replay_X.device)
                    rep = replay_X[rep_idx]
                X = torch.cat([X[:n_new], rep], dim=0)

        f_net.train()
        opt.zero_grad(set_to_none=True)
        all_EL = []
        nmb = max(1, math.ceil(n_coll / micro_batch))

        for i in range(0, n_coll, micro_batch):
            xb = X[i : i + micro_batch]
            EL = compute_EL(psi_log_fn, xb, omega).view(-1)
            ok = torch.isfinite(EL)
            if not ok.all():
                EL = EL[ok]
            if EL.numel() == 0:
                continue
            if qtrim > 0 and EL.numel() > 20:
                lo = torch.quantile(EL.detach(), qtrim)
                hi = torch.quantile(EL.detach(), 1.0 - qtrim)
                m = (EL.detach() >= lo) & (EL.detach() <= hi)
                EL = EL[m]
                if EL.numel() == 0:
                    continue
            all_EL.append(EL.detach())
            mu = EL.mean().detach()
            E_eff = alpha * E_ref + (1 - alpha) * mu if alpha > 0 else mu
            loss = huber(EL - E_eff, huber_d).mean()
            (loss / nmb).backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(prms, grad_clip)
        opt.step()
        sch.step()

        if all_EL:
            ELc = torch.cat(all_EL)
            Em, Ev, Es = ELc.mean().item(), ELc.var().item(), ELc.std().item()

            probe_n = min(128, X.shape[0])
            probe_x = X[:probe_n]
            probe_EL = compute_EL(psi_log_fn, probe_x, omega).view(-1)
            if torch.isfinite(probe_EL).any():
                keep = torch.isfinite(probe_EL)
                probe_x = probe_x[keep]
                probe_EL = probe_EL[keep]
                if probe_x.shape[0] > 4:
                    resid_abs = (probe_EL - probe_EL.mean()).abs()
                    topk = min(max(8, probe_x.shape[0] // 4), probe_x.shape[0])
                    idx_top = torch.topk(resid_abs, k=topk).indices
                    replay_X = probe_x[idx_top].detach().clone()
        else:
            Em = Ev = Es = float("nan")

        epdt = time.time() - ept0
        hist.append(dict(ep=ep, E=Em, var=Ev, alpha=alpha, dt=epdt))

        if math.isfinite(Ev) and Ev < best_var * 0.999:
            best_var = Ev
            best_st = {k: v.clone() for k, v in f_net.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if vmc_every > 0 and ep > 0 and ep % vmc_every == 0:
            vp = evaluate_energy_vmc(
                f_net,
                C_occ,
                psi_fn=psi_fn,
                compute_coulomb_interaction=compute_coulomb_interaction,
                backflow_net=None,
                params=params,
                n_samples=vmc_n,
                batch_size=512,
                sampler_steps=40,
                sampler_step_sigma=0.12,
                lap_mode="exact",
                persistent=True,
                sampler_burn_in=200,
                sampler_thin=2,
                progress=False,
            )
            vE = float(vp["E_mean"])
            vErr = abs(vE - E_ref) / abs(E_ref)
            hist[-1].update(vmc_E=vE, vmc_err=vErr)
            if vErr < best_vmc_err:
                best_vmc_err = vErr
                best_vmc_E = vE
                best_vmc_st = {k: v.clone() for k, v in f_net.state_dict().items()}

        if patience > 0 and no_imp >= patience and ep > 30:
            print(f"  Early stop ep {ep}")
            sys.stdout.flush()
            break

        if ep % print_every == 0:
            dt = time.time() - t0  # noqa: F841
            err = (Em - E_ref) / abs(E_ref) * 100 if math.isfinite(Em) else float("nan")
            vs = ""
            if "vmc_E" in hist[-1]:
                vs = f"  vmc={hist[-1]['vmc_E']:.4f}({hist[-1]['vmc_err']*100:.2f}%)"
            eta = epdt * (n_epochs - ep - 1) / 60
            print(
                f"  [{ep:3d}] E={Em:.4f}±{Es:.3f} var={Ev:.2e} α={alpha:.2f} "
                f"{epdt:.1f}s err={err:+.2f}% eta={eta:.0f}m{vs}"
            )
            sys.stdout.flush()

    if best_vmc_st:
        f_net.load_state_dict(best_vmc_st)
        print(f"  Restored VMC-best E={best_vmc_E:.5f} err={best_vmc_err*100:.3f}%")
    elif best_st:
        f_net.load_state_dict(best_st)
        print(f"  Restored var-best var={best_var:.3e}")
    tot = time.time() - t0
    print(f"  Done {tot:.0f}s ({tot/60:.1f}min)")
    sys.stdout.flush()
    return f_net, hist


# ─── Run one arch ───
def run_one(
    name,
    *,
    n_epochs=80,
    n_coll=320,
    eval_samples=8000,
    patience=30,
    vmc_every=40,
    vmc_n=3000,
    lr=5e-4,
    alpha_end=0.70,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C_occ, params = setup()
    print(f"\n{'#'*55}\n# {name}\n{'#'*55}")
    sys.stdout.flush()
    net = ARCHS[name]().to(DEVICE).to(DTYPE)
    np_ = sum(p.numel() for p in net.parameters())
    print(f"  {type(net).__name__}  {np_:,} params")
    sys.stdout.flush()

    if "vcycle" in name:
        micro_batch = 32
    elif "ctnn" in name:
        micro_batch = 128
    else:
        micro_batch = 256

    t0 = time.time()
    net, hist = train_colloc(
        net,
        C_occ,
        params,
        n_epochs=n_epochs,
        lr=lr,
        alpha_end=alpha_end,
        n_coll=n_coll,
        oversample=8,
        micro_batch=micro_batch,
        print_every=10,
        replay_frac=0.25,
        patience=patience,
        vmc_every=vmc_every,
        vmc_n=vmc_n,
        tag=name,
    )

    if eval_samples > 0:
        vmc = evaluate_energy_vmc(
            net,
            C_occ,
            psi_fn=psi_fn,
            compute_coulomb_interaction=compute_coulomb_interaction,
            backflow_net=None,
            params=params,
            n_samples=eval_samples,
            batch_size=512,
            sampler_steps=80,
            sampler_step_sigma=0.08,
            lap_mode="exact",
            persistent=True,
            sampler_burn_in=1000,
            sampler_thin=4,
            progress=True,
        )
        E, se = vmc["E_mean"], vmc["E_stderr"]
        eval_tag = "final_vmc"
    else:
        vmc_hist = [h for h in hist if "vmc_E" in h]
        if vmc_hist:
            best = min(vmc_hist, key=lambda h: h["vmc_err"])
            E = float(best["vmc_E"])
            se = float("nan")
            eval_tag = "best_vmc_probe"
        else:
            E = float(hist[-1]["E"])
            se = float("nan")
            eval_tag = "train_EL_last"
    err = (E - E_DMC) / abs(E_DMC) * 100
    wall = time.time() - t0
    print(f"\n  FINAL[{eval_tag}]: E = {E:.6f} ± {se:.6f}  err={err:+.3f}%  wall={wall/60:.1f}min")
    sys.stdout.flush()

    torch.save(
        dict(arch=name, state=net.state_dict(), n_params=np_, E=E, se=se, hist=hist, wall=wall),
        RESULTS_DIR / f"{name}.pt",
    )
    return dict(arch=name, cls=type(net).__name__, params=np_, E=E, se=se, err=err, wall=wall)


# ─── Main ───
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS) + ["all"], default="all")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-coll", type=int, default=320)
    ap.add_argument("--n-eval", type=int, default=8000)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--vmc-every", type=int, default=40)
    ap.add_argument("--vmc-n", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha-end", type=float, default=0.70)
    a = ap.parse_args()
    names = list(ARCHS) if a.arch == "all" else [a.arch]

    print(f"Collocation comparison (structural search) — {names}")
    print("Baseline PINN result: 20.210 ± 0.003 (+0.25%)\n")
    sys.stdout.flush()

    results = [
        run_one(
            n,
            n_epochs=a.epochs,
            n_coll=a.n_coll,
            eval_samples=a.n_eval,
            patience=a.patience,
            vmc_every=a.vmc_every,
            vmc_n=a.vmc_n,
            lr=a.lr,
            alpha_end=a.alpha_end,
        )
        for n in names
    ]

    print(f"\n{'='*70}")
    print(f"RESULTS  N=6  ω=1.0  E_DMC={E_DMC}")
    print(f"{'='*70}")
    print(f"{'Arch':<14s} {'Class':<16s} {'Params':>7s}  {'E':>14s}  {'err%':>7s} {'Wall':>6s}")
    print("-" * 70)
    print(
        f"  {'pinn_base':<12s} {'PINN':<16s} {'15,493':>7s}  "
        f"{'20.210±0.003':>14s}  {'+0.25%':>7s} {'prev':>6s}"
    )
    for r in results:
        print(
            f"  {r['arch']:<12s} {r['cls']:<16s} {r['params']:>7,d}  "
            f"{r['E']:.6f}±{r['se']:.6f}  {r['err']:+.3f}% {r['wall']/60:5.1f}m"
        )
    print(f"  {'E_DMC':<12s} {'':16s} {'':>8s}  {E_DMC:.6f}")
    print(f"{'='*70}")
    sys.stdout.flush()

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
