"""
Orbital Backflow + Jastrow collocation training
================================================
N=6, ω=1.0, E_DMC=20.15932

Loads the pre-trained CTNNJastrowVCycle (best Jastrow-only checkpoint)
and trains it jointly with a fresh OrbitalBackflowNet.

Orbital backflow path (NEW):
  δΨ = orbital_bf_net(x, spin)             -- orbital perturbations (B,N,n_occ)
  Psi = Phi·C_occ + δΨ                     -- perturbed orbital matrix
  logpsi = log|det(Psi_up)·det(Psi_down)| + f(x)  -- Jastrow on original coords
"""

import argparse
import math
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
from jastrow_architectures import CTNNJastrowVCycle
from PINN import OrbitalBackflowNet

DEVICE = "cpu"
DTYPE = torch.float64
N_ELEC = 6
DIM = 2
OMEGA = 1.0
E_DMC = 20.15932
E_TARGET = 20.17

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"


# ─── System setup ───
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


# ─── Combined Orbital BF + Jastrow collocation trainer ───
def train_colloc_orbital_bf(
    f_net,
    orbital_bf_net,
    C_occ,
    params,
    *,
    n_epochs=600,
    lr_bf=5e-4,
    lr_jas=5e-5,
    lr_min_frac=0.02,
    phase1_frac=0.25,
    alpha_end=0.70,
    n_coll=512,
    oversample=8,
    micro_batch=32,
    grad_clip=0.2,
    replay_frac=0.25,
    qtrim=0.02,
    huber_d=1.0,
    print_every=10,
    patience=200,
    vmc_every=40,
    vmc_n=6000,
    bf_warmup=900,
    bf_scale_start=0.001,
    bf_scale_end=0.05,
    bf_ramp_epochs=300,
    tag="orb_bf_vcycle",
):
    omega = OMEGA
    E_ref = E_DMC
    up = N_ELEC // 2
    spin = torch.cat(
        [torch.zeros(up, dtype=torch.long), torch.ones(N_ELEC - up, dtype=torch.long)]
    ).to(DEVICE)

    def get_bf_scale(ep):
        """Linear ramp from bf_scale_start to bf_scale_end over bf_ramp_epochs."""
        if ep >= bf_ramp_epochs:
            return bf_scale_end
        t = ep / max(1, bf_ramp_epochs)
        return bf_scale_start + t * (bf_scale_end - bf_scale_start)

    def psi_log_fn(y):
        lp, _ = psi_fn(f_net, y, C_occ, orbital_bf_net=orbital_bf_net, spin=spin, params=params)
        return lp

    bf_params = [p for p in orbital_bf_net.parameters() if p.requires_grad]
    jas_params = [p for p in f_net.parameters() if p.requires_grad]
    n_bf = sum(p.numel() for p in bf_params)
    n_jas = sum(p.numel() for p in jas_params)
    print(f"  Orbital BF params: {n_bf:,}   Jastrow params: {n_jas:,}   Total: {n_bf + n_jas:,}")
    print(f"  Jastrow frozen for all {bf_warmup} warmup epochs")
    print(f"  bf_scale ramp: {bf_scale_start} → {bf_scale_end} over {bf_ramp_epochs} epochs")

    opt = torch.optim.Adam(
        [
            {"params": bf_params, "lr": lr_bf},
            {"params": jas_params, "lr": lr_jas},
        ]
    )

    def lr_lambda_bf(ep):
        lr_min = lr_bf * lr_min_frac
        return (
            lr_min + 0.5 * (lr_bf - lr_min) * (1 + math.cos(math.pi * ep / max(1, n_epochs - 1)))
        ) / lr_bf

    def lr_lambda_jas(ep):
        if ep < bf_warmup:
            return 0.0
        t = (ep - bf_warmup) / max(1, n_epochs - bf_warmup - 1)
        lr_min = lr_jas * lr_min_frac
        return (lr_min + 0.5 * (lr_jas - lr_min) * (1 + math.cos(math.pi * t))) / lr_jas

    sch = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda_bf, lr_lambda_jas])
    p1end = int(phase1_frac * n_epochs)

    print(f"  Training: {n_epochs} ep, {n_coll} colloc pts, grad_clip={grad_clip}")
    print(f"  LR: orbital_bf={lr_bf}, jastrow={lr_jas}")
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

        # Set scheduled bf_scale for this epoch
        cur_scale = get_bf_scale(ep)
        orbital_bf_net._scale_override = cur_scale

        alpha = (
            0.0
            if ep < p1end
            else 0.5
            * alpha_end
            * (1 - math.cos(math.pi * (ep - p1end) / max(1, n_epochs - p1end - 1)))
        )

        f_net.eval()
        orbital_bf_net.eval()
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
        orbital_bf_net.train()
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
            all_prms = list(bf_params) + list(jas_params)
            nn.utils.clip_grad_norm_(all_prms, grad_clip)
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

        def _save_state():
            return {
                "bf_state": {k: v.clone() for k, v in orbital_bf_net.state_dict().items()},
                "jas_state": {k: v.clone() for k, v in f_net.state_dict().items()},
            }

        if math.isfinite(Ev) and Ev < best_var * 0.999:
            best_var = Ev
            best_state = _save_state()
            no_imp = 0
        else:
            no_imp += 1

        if vmc_every > 0 and ep > 0 and ep % vmc_every == 0:
            # Use schedule scale for VMC probe too
            vp = evaluate_energy_vmc(
                f_net,
                C_occ,
                psi_fn=psi_fn,
                compute_coulomb_interaction=compute_coulomb_interaction,
                orbital_bf_net=orbital_bf_net,
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
            bf_sc = cur_scale
            hist[-1].update(vmc_E=vE, vmc_err=vErr, bf_scale=bf_sc)
            if vErr < best_vmc_err:
                best_vmc_err = vErr
                best_vmc_E = vE
                best_vmc_state = _save_state()

        if patience > 0 and no_imp >= patience and ep > 60:
            print(f"  Early stop ep {ep}")
            sys.stdout.flush()
            break

        if ep % print_every == 0:
            dt = time.time() - t0  # noqa: F841
            err = (Em - E_ref) / abs(E_ref) * 100 if math.isfinite(Em) else float("nan")
            bf_sc = cur_scale
            vs = ""
            if "vmc_E" in hist[-1]:
                vs = f"  vmc={hist[-1]['vmc_E']:.4f}({hist[-1]['vmc_err'] * 100:.2f}%)"
            eta = epdt * (n_epochs - ep - 1) / 60
            phase = "WRM" if ep < bf_warmup else "JNT"
            print(
                f"  [{ep:3d}|{phase}] E={Em:.4f}±{Es:.3f} var={Ev:.2e} α={alpha:.2f} "
                f"bf_sc={bf_sc:.4f} {epdt:.1f}s err={err:+.2f}% eta={eta:.0f}m{vs}"
            )
            sys.stdout.flush()

    # Restore best checkpoint
    if best_vmc_state:
        orbital_bf_net.load_state_dict(best_vmc_state["bf_state"])
        f_net.load_state_dict(best_vmc_state["jas_state"])
        print(f"  Restored VMC-best E={best_vmc_E:.5f} err={best_vmc_err * 100:.3f}%")
    elif best_state:
        orbital_bf_net.load_state_dict(best_state["bf_state"])
        f_net.load_state_dict(best_state["jas_state"])
        print(f"  Restored var-best var={best_var:.3e}")
    tot = time.time() - t0
    print(f"  Done {tot:.0f}s ({tot / 60:.1f}min)")
    sys.stdout.flush()
    return f_net, orbital_bf_net, hist


# ─── Main ───
def main():
    ap = argparse.ArgumentParser(description="Orbital BF + Jastrow collocation training")
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-coll", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--vmc-every", type=int, default=40)
    ap.add_argument("--vmc-n", type=int, default=6000)
    ap.add_argument("--lr-bf", type=float, default=3e-4, help="LR for orbital BF (fresh)")
    ap.add_argument("--lr-jas", type=float, default=5e-5, help="LR for Jastrow (warm)")
    ap.add_argument("--alpha-end", type=float, default=0.70)
    ap.add_argument("--bf-hidden", type=int, default=64, help="Orbital BF hidden dim")
    ap.add_argument("--bf-msg-hidden", type=int, default=64, help="Orbital BF edge/msg dim")
    ap.add_argument(
        "--bf-warmup", type=int, default=900, help="Epochs of orbital BF-only (Jastrow frozen)"
    )
    ap.add_argument(
        "--bf-scale-init", type=float, default=0.05, help="Learned bf_scale init (fallback)"
    )
    ap.add_argument("--bf-scale-start", type=float, default=0.001, help="Schedule: start scale")
    ap.add_argument("--bf-scale-end", type=float, default=0.05, help="Schedule: end scale")
    ap.add_argument("--bf-ramp-epochs", type=int, default=300, help="Schedule: epochs to ramp over")
    ap.add_argument("--grad-clip", type=float, default=0.2, help="Gradient clip norm")
    ap.add_argument("--tag", type=str, default="orb_bf_vcycle")
    a = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C_occ, params = setup()
    n_occ = N_ELEC // 2

    # ─── Load pre-trained V-cycle Jastrow ───
    jas_ckpt_path = RESULTS_DIR / "ctnn_vcycle.pt"
    print(f"Loading Jastrow from {jas_ckpt_path}")
    jas_ckpt = torch.load(jas_ckpt_path, map_location=DEVICE)
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
    f_net.load_state_dict(jas_ckpt["state"])
    n_jas = sum(p.numel() for p in f_net.parameters())
    print(f"  Jastrow: CTNNJastrowVCycle  {n_jas:,} params  (E={jas_ckpt.get('E', '?')})")

    # ─── Fresh OrbitalBackflowNet ───
    orbital_bf_net = (
        OrbitalBackflowNet(
            d=DIM,
            n_occ=n_occ,
            msg_hidden=a.bf_msg_hidden,
            msg_layers=2,
            hidden=a.bf_hidden,
            layers=2,
            act="silu",
            aggregation="sum",
            use_spin=True,
            same_spin_only=False,
            out_bound="tanh",
            bf_scale_init=a.bf_scale_init,
            zero_init_last=True,  # start δΨ ≈ 0 for stable identity start
            omega=OMEGA,
        )
        .to(DEVICE)
        .to(DTYPE)
    )
    n_bf = sum(p.numel() for p in orbital_bf_net.parameters())
    bf_sc = orbital_bf_net.bf_scale.item()
    print(f"  Orbital BF: OrbitalBackflowNet  {n_bf:,} params  bf_scale={bf_sc:.4f}")
    print(f"  Output: δΨ shape (B, {N_ELEC}, {n_occ}) — orbital perturbations")
    sys.stdout.flush()

    # ─── Train ───
    print(f"\n{'#' * 60}")
    print("# Orbital Backflow + Jastrow V-cycle collocation training")
    print(f"# {a.epochs} epochs, {a.n_coll} colloc pts, α_end={a.alpha_end}")
    print(f"# LR: orbital_bf={a.lr_bf}, jas={a.lr_jas}, grad_clip={a.grad_clip}")
    print(f"# Jastrow frozen: {a.bf_warmup} ep")
    print(f"# bf_scale ramp: {a.bf_scale_start} → {a.bf_scale_end} over {a.bf_ramp_epochs} ep")
    print(f"{'#' * 60}\n")
    sys.stdout.flush()

    t0 = time.time()
    f_net, orbital_bf_net, hist = train_colloc_orbital_bf(
        f_net,
        orbital_bf_net,
        C_occ,
        params,
        n_epochs=a.epochs,
        lr_bf=a.lr_bf,
        lr_jas=a.lr_jas,
        alpha_end=a.alpha_end,
        n_coll=a.n_coll,
        oversample=8,
        micro_batch=32,
        grad_clip=a.grad_clip,
        print_every=10,
        replay_frac=0.25,
        patience=a.patience,
        vmc_every=a.vmc_every,
        vmc_n=a.vmc_n,
        bf_warmup=a.bf_warmup,
        bf_scale_start=a.bf_scale_start,
        bf_scale_end=a.bf_scale_end,
        bf_ramp_epochs=a.bf_ramp_epochs,
        tag=a.tag,
    )

    # ─── Final heavy VMC evaluation ───
    # Use the final scheduled scale for evaluation
    orbital_bf_net._scale_override = a.bf_scale_end

    if a.n_eval > 0:
        print(
            f"\n  Final VMC eval: {a.n_eval} samples"
            f" (bf_scale={orbital_bf_net._scale_override}) ..."
        )
        sys.stdout.flush()
        vmc = evaluate_energy_vmc(
            f_net,
            C_occ,
            psi_fn=psi_fn,
            compute_coulomb_interaction=compute_coulomb_interaction,
            orbital_bf_net=orbital_bf_net,
            params=params,
            n_samples=a.n_eval,
            batch_size=512,
            sampler_steps=60,
            sampler_step_sigma=0.12,
            lap_mode="exact",
            persistent=True,
            sampler_burn_in=500,
            sampler_thin=3,
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
    print(
        f"\n  FINAL[{eval_tag}]: E = {E:.6f} ± {se:.6f}"
        f"  err={err:+.3f}%  wall={wall / 60:.1f}min"
    )
    sys.stdout.flush()

    # ─── Save checkpoint ───
    save_path = RESULTS_DIR / f"{a.tag}.pt"
    torch.save(
        dict(
            tag=a.tag,
            bf_state=orbital_bf_net.state_dict(),
            jas_state=f_net.state_dict(),
            bf_class="OrbitalBackflowNet",
            jas_class="CTNNJastrowVCycle",
            n_bf_params=n_bf,
            n_jas_params=n_jas,
            bf_config=dict(
                d=DIM,
                n_occ=n_occ,
                msg_hidden=a.bf_msg_hidden,
                msg_layers=2,
                hidden=a.bf_hidden,
                layers=2,
                act="silu",
                aggregation="sum",
                use_spin=True,
                same_spin_only=False,
                out_bound="tanh",
                bf_scale_init=a.bf_scale_init,
                zero_init_last=True,
                omega=OMEGA,
            ),
            E=E,
            se=se,
            err=err,
            hist=hist,
            wall=wall,
        ),
        save_path,
    )
    print(f"  Saved → {save_path}")

    # ─── Summary ───
    print(f"\n{'=' * 60}")
    print(f"RESULT  N={N_ELEC}  ω={OMEGA}  E_DMC={E_DMC}")
    print(f"{'=' * 60}")
    print(f"  Orbital BF: OrbitalBackflowNet  {n_bf:,} params")
    print(f"  Jastrow:    CTNNJastrowVCycle {n_jas:,} params")
    print(f"  Total:      {n_bf + n_jas:,} params")
    print(f"  E = {E:.6f} ± {se:.6f}   err = {err:+.3f}%")
    print(f"  Wall:  {wall / 60:.1f} min")
    print(f"  Target: {E_TARGET}  ({'+' if E > E_TARGET else ''}{(E - E_TARGET):.4f})")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
