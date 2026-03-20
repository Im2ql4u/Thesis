"""
Node-targeted collocation fine-tune
====================================
N=6, ω=1.0, E_DMC=20.15932

Loads the Phase 2 BF+Jastrow checkpoint (bf_ctnn_vcycle.pt),
freezes the Jastrow, and fine-tunes the backflow with a sampling
strategy that over-represents configurations near the nodal surface
(where |det Slater| is small).

Diagnostic: if this closes the gap to DMC → the error is in the
node *positions* (backflow can fix them with better data).  If not →
the error is *topological* (wrong nodal structure).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from functions.Energy import evaluate_energy_vmc
from functions.Neural_Networks import _laplacian_logpsi_exact, psi_fn
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import (
    slater_determinant_closed_shell,
    evaluate_basis_functions_torch_batch_2d,
)
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet

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
def screened_colloc(psi_log_fn, n_keep, omega, oversample=8, sigma_fs=(0.8, 1.3, 2.0), explore=0.10):
    n_cand = oversample * n_keep
    nc = len(sigma_fs)
    xs, lqs = [], []
    for i, sf in enumerate(sigma_fs):
        ni = n_cand // nc if i < nc - 1 else n_cand - sum(n_cand // nc for _ in range(i))
        xi, lqi = sample_gauss(ni, omega, sf)
        xs.append(xi); lqs.append(lqi)
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


@torch.no_grad()
def node_targeted_colloc(
    psi_log_fn, backflow_net, f_net, C_occ, params, spin,
    n_keep, omega, node_frac=0.4, oversample=10, sigma_fs=(0.8, 1.3, 2.0),
):
    """
    Sample collocation points with a fraction biased toward the nodal surface.

    Strategy:
      1. Draw a large pool of candidate points (Gaussian mixture).
      2. Compute |det Slater(x_eff)| for each (the backflow-shifted Slater det).
      3. Build a node-proximity weight  w_i = 1 / (|det| + eps).
      4. Split the budget: (1-node_frac)*n_keep from standard |ψ|²-screened,
         node_frac*n_keep from node-proximity weighted resampling.
    """
    n_node = int(node_frac * n_keep)
    n_bulk = n_keep - n_node

    # --- bulk: standard screened collocation ---
    X_bulk = screened_colloc(psi_log_fn, n_bulk, omega, oversample=oversample, sigma_fs=sigma_fs)

    # --- node-targeted: oversample then pick near-node points ---
    n_cand = oversample * n_node
    nc = len(sigma_fs)
    xs = []
    for i, sf in enumerate(sigma_fs):
        ni = n_cand // nc if i < nc - 1 else n_cand - sum(n_cand // nc for _ in range(i))
        xi, _ = sample_gauss(ni, omega, sf)
        xs.append(xi)
    x_all = torch.cat(xs)

    # Compute |det Slater(x_eff)| in chunks
    det_abs_list = []
    for i in range(0, len(x_all), 2048):
        xb = x_all[i : i + 2048]
        B_cur = xb.shape[0]
        spin_bn = spin.unsqueeze(0).expand(B_cur, -1)
        x_eff = xb + backflow_net(xb, spin=spin_bn)
        sign, logabs = slater_determinant_closed_shell(
            x_config=x_eff, C_occ=C_occ, params=params, spin=spin_bn, normalize=True,
        )
        # |det| = exp(logabs)
        det_abs_list.append(logabs.view(-1))
    logdet_all = torch.cat(det_abs_list)

    # Node proximity weight: smaller |det| → higher weight
    # Use 1/(|det|+eps) in log space: log_w = -logabs (choose points with smallest logabs)
    # Simply pick the n_node points with the smallest |det|
    _, idx_sorted = torch.sort(logdet_all, descending=False)  # ascending: smallest det first
    sel_node = idx_sorted[:n_node]
    X_node = x_all[sel_node].clone()

    X = torch.cat([X_bulk, X_node], dim=0)
    # Shuffle
    perm = torch.randperm(X.shape[0], device=X.device)
    return X[perm]


def train_node_targeted(
    f_net, backflow_net, C_occ, params,
    *, n_epochs=300, lr_bf=2e-4, lr_min_frac=0.02,
    alpha_end=0.80, n_coll=512, oversample=10, node_frac=0.4,
    micro_batch=32, grad_clip=0.5, replay_frac=0.25,
    qtrim=0.02, huber_d=1.0, print_every=10, patience=100,
    vmc_every=30, vmc_n=6000, tag="node_ft",
):
    omega = OMEGA
    E_ref = E_DMC
    up = N_ELEC // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_ELEC - up, dtype=torch.long)]
    ).to(DEVICE)

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return lp

    # Only backflow params are trainable (Jastrow is frozen)
    bf_params = [p for p in backflow_net.parameters() if p.requires_grad]
    n_bf = sum(p.numel() for p in bf_params)
    n_jas = sum(p.numel() for p in f_net.parameters())
    print(f"  Backflow params (trainable): {n_bf:,}")
    print(f"  Jastrow params (frozen):     {n_jas:,}")
    print(f"  Node fraction: {node_frac:.0%}")

    opt = torch.optim.Adam(bf_params, lr=lr_bf)

    def lr_lambda(ep):
        lr_min = lr_bf * lr_min_frac
        return (lr_min + 0.5 * (lr_bf - lr_min) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))) / lr_bf

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"  Training: {n_epochs} ep, {n_coll} colloc pts, LR={lr_bf}")
    sys.stdout.flush()

    t0 = time.time()
    hist = []
    best_var = best_vmc_err = float("inf")
    best_state = best_vmc_state = {}
    best_vmc_E = None
    no_imp = 0
    replay_X = None

    for ep in range(n_epochs):
        ept0 = time.time()
        # Alpha ramp: start from 0.5 (already partially trained) → alpha_end
        alpha = 0.5 * alpha_end * (1 - math.cos(math.pi * ep / max(1, n_epochs - 1)))

        f_net.eval()
        backflow_net.eval()
        X = node_targeted_colloc(
            psi_log_fn, backflow_net, f_net, C_occ, params, spin,
            n_keep=n_coll, omega=omega, node_frac=node_frac,
            oversample=oversample,
        )

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

        f_net.eval()  # Jastrow stays in eval
        backflow_net.train()
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
            nn.utils.clip_grad_norm_(bf_params, grad_clip)
        opt.step()
        sch.step()

        if all_EL:
            ELc = torch.cat(all_EL)
            Em, Ev, Es = ELc.mean().item(), ELc.var().item(), ELc.std().item()
            # Replay buffer: high-residual points
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

        def _save_state():
            return {
                "bf_state": {k: v.clone() for k, v in backflow_net.state_dict().items()},
                "jas_state": {k: v.clone() for k, v in f_net.state_dict().items()},
            }

        if math.isfinite(Ev) and Ev < best_var * 0.999:
            best_var = Ev
            best_state = _save_state()
            no_imp = 0
        else:
            no_imp += 1

        if vmc_every > 0 and ep > 0 and ep % vmc_every == 0:
            vp = evaluate_energy_vmc(
                f_net, C_occ,
                psi_fn=psi_fn, compute_coulomb_interaction=compute_coulomb_interaction,
                backflow_net=backflow_net, params=params,
                n_samples=vmc_n, batch_size=512, sampler_steps=40,
                sampler_step_sigma=0.12, lap_mode="exact",
                persistent=True, sampler_burn_in=200, sampler_thin=2, progress=False,
            )
            vE = float(vp["E_mean"])
            vErr = abs(vE - E_ref) / abs(E_ref)
            bf_sc = backflow_net.bf_scale.item()
            hist[-1].update(vmc_E=vE, vmc_err=vErr, bf_scale=bf_sc)
            if vErr < best_vmc_err:
                best_vmc_err = vErr
                best_vmc_E = vE
                best_vmc_state = _save_state()

        if patience > 0 and no_imp >= patience and ep > 30:
            print(f"  Early stop ep {ep}")
            sys.stdout.flush()
            break

        if ep % print_every == 0:
            err = (Em - E_ref) / abs(E_ref) * 100 if math.isfinite(Em) else float("nan")
            bf_sc = backflow_net.bf_scale.item()
            vs = ""
            if "vmc_E" in hist[-1]:
                vs = f"  vmc={hist[-1]['vmc_E']:.4f}({hist[-1]['vmc_err'] * 100:.2f}%)"
            eta = epdt * (n_epochs - ep - 1) / 60
            print(
                f"  [{ep:3d}] E={Em:.4f}±{Es:.3f} var={Ev:.2e} α={alpha:.2f} "
                f"bf_sc={bf_sc:.4f} {epdt:.1f}s err={err:+.2f}% eta={eta:.0f}m{vs}"
            )
            sys.stdout.flush()

    # Restore best checkpoint
    if best_vmc_state:
        backflow_net.load_state_dict(best_vmc_state["bf_state"])
        f_net.load_state_dict(best_vmc_state["jas_state"])
        print(f"  Restored VMC-best E={best_vmc_E:.5f} err={best_vmc_err * 100:.3f}%")
    elif best_state:
        backflow_net.load_state_dict(best_state["bf_state"])
        f_net.load_state_dict(best_state["jas_state"])
        print(f"  Restored var-best var={best_var:.3e}")
    tot = time.time() - t0
    print(f"  Done {tot:.0f}s ({tot / 60:.1f}min)")
    sys.stdout.flush()
    return f_net, backflow_net, hist


def main():
    ap = argparse.ArgumentParser(description="Node-targeted BF fine-tune (Jastrow frozen)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-coll", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--vmc-every", type=int, default=30)
    ap.add_argument("--vmc-n", type=int, default=6000)
    ap.add_argument("--lr-bf", type=float, default=2e-4, help="LR for backflow (lower for fine-tune)")
    ap.add_argument("--alpha-end", type=float, default=0.80)
    ap.add_argument("--node-frac", type=float, default=0.40, help="Fraction of colloc points near nodes")
    ap.add_argument("--tag", type=str, default="node_ft")
    a = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C_occ, params = setup()

    # ─── Load Phase 2 BF+Jastrow checkpoint ───
    bf_ckpt_path = RESULTS_DIR / "bf_ctnn_vcycle.pt"
    print(f"Loading BF+Jastrow checkpoint from {bf_ckpt_path}")
    ckpt = torch.load(bf_ckpt_path, map_location=DEVICE)

    # Reconstruct CTNNJastrowVCycle (Jastrow)
    f_net = CTNNJastrowVCycle(
        n_particles=N_ELEC, d=DIM, omega=OMEGA,
        node_hidden=24, edge_hidden=24, bottleneck_hidden=12,
        n_down=1, n_up=1, msg_layers=1, node_layers=1,
        readout_hidden=64, readout_layers=2, act="silu",
    ).to(DEVICE).to(DTYPE)
    f_net.load_state_dict(ckpt["jas_state"])
    # FREEZE Jastrow
    for p in f_net.parameters():
        p.requires_grad = False
    n_jas = sum(p.numel() for p in f_net.parameters())
    print(f"  Jastrow: CTNNJastrowVCycle  {n_jas:,} params  (FROZEN)")

    # Reconstruct CTNNBackflowNet
    bfc = ckpt["bf_config"]
    backflow_net = CTNNBackflowNet(
        d=bfc["d"], msg_hidden=bfc["msg_hidden"], msg_layers=bfc["msg_layers"],
        hidden=bfc["hidden"], layers=bfc["layers"], act=bfc["act"],
        aggregation=bfc["aggregation"], use_spin=bfc["use_spin"],
        same_spin_only=bfc["same_spin_only"], out_bound=bfc["out_bound"],
        bf_scale_init=bfc["bf_scale_init"], zero_init_last=bfc["zero_init_last"],
        omega=bfc["omega"],
    ).to(DEVICE).to(DTYPE)
    backflow_net.load_state_dict(ckpt["bf_state"])
    n_bf = sum(p.numel() for p in backflow_net.parameters())
    bf_sc = backflow_net.bf_scale.item()
    print(f"  Backflow: CTNNBackflowNet  {n_bf:,} params  bf_scale={bf_sc:.4f}")
    print(f"  Checkpoint energy: E={ckpt.get('E', '?')}")
    sys.stdout.flush()

    # ─── Train ───
    print(f"\n{'#' * 60}")
    print("# Node-targeted collocation fine-tune (Jastrow frozen)")
    print(f"# {a.epochs} epochs, {a.n_coll} colloc pts, node_frac={a.node_frac}")
    print(f"# α_end={a.alpha_end}, LR_bf={a.lr_bf}")
    print(f"{'#' * 60}\n")
    sys.stdout.flush()

    t0 = time.time()
    f_net, backflow_net, hist = train_node_targeted(
        f_net, backflow_net, C_occ, params,
        n_epochs=a.epochs, lr_bf=a.lr_bf, alpha_end=a.alpha_end,
        n_coll=a.n_coll, oversample=10, node_frac=a.node_frac,
        micro_batch=32, print_every=10, replay_frac=0.25,
        patience=a.patience, vmc_every=a.vmc_every, vmc_n=a.vmc_n,
        tag=a.tag,
    )

    # ─── Final heavy VMC evaluation ───
    if a.n_eval > 0:
        print(f"\n  Final VMC eval: {a.n_eval} samples ...")
        sys.stdout.flush()
        up = N_ELEC // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N_ELEC - up, dtype=torch.long)]
        ).to(DEVICE)
        vmc = evaluate_energy_vmc(
            f_net, C_occ,
            psi_fn=psi_fn, compute_coulomb_interaction=compute_coulomb_interaction,
            backflow_net=backflow_net, params=params,
            n_samples=a.n_eval, batch_size=512,
            sampler_steps=80, sampler_step_sigma=0.08,
            lap_mode="exact", persistent=True,
            sampler_burn_in=400, sampler_thin=3, progress=True,
        )
        E = float(vmc["E_mean"])
        se = float(vmc["E_stderr"])
        err = (E - E_DMC) / abs(E_DMC) * 100
        wall = time.time() - t0
        print(f"\n  *** Final: E = {E:.5f} ± {se:.5f}   err = {err:+.3f}%  ({wall/60:.1f} min)")
    else:
        E = se = err = wall = float("nan")

    # ─── Save checkpoint ───
    save_path = RESULTS_DIR / f"{a.tag}.pt"
    torch.save(
        dict(
            tag=a.tag,
            bf_state=backflow_net.state_dict(),
            jas_state=f_net.state_dict(),
            bf_class="CTNNBackflowNet",
            jas_class="CTNNJastrowVCycle",
            n_bf_params=n_bf,
            n_jas_params=n_jas,
            bf_config=bfc,
            E=E, se=se, err=err, hist=hist, wall=wall,
        ),
        save_path,
    )
    print(f"  Saved → {save_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
