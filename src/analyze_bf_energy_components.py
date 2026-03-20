"""
Analyze BF+Jastrow checkpoint failures by decomposing local energy components.

Focus:
- Sample from |Psi|^2 (persistent Metropolis)
- Compute per-sample local energy decomposition:
    E_L = T + V_int + V_trap
- Identify worst-energy quantile samples
- Analyze geometry (pair spacing, radial concentration) of hard samples

Outputs JSON + NPZ in a dated diagnostics folder.
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import torch

import config
from functions.Energy import _local_energy_multi, _metropolis_psi2_persistent
from functions.Neural_Networks import psi_fn
from functions.Physics import compute_coulomb_interaction
from jastrow_architectures import CTNNJastrowVCycle
from PINN import CTNNBackflowNet


DTYPE = torch.float64
N_ELEC = 6
DIM = 2
OMEGA = 1.0
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "arch_colloc"


def _device_from_arg(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def setup(device: str):
    n_occ = N_ELEC // 2
    nx = ny = 3
    L = max(8.0, 3.0 / math.sqrt(OMEGA))
    config.update(
        omega=OMEGA,
        n_particles=N_ELEC,
        d=DIM,
        L=L,
        n_grid=80,
        nx=nx,
        ny=ny,
        basis="cart",
        device=device,
        dtype="float64",
    )

    energies = sorted([(OMEGA * (ix + iy + 1), ix, iy) for ix in range(nx) for iy in range(ny)])
    C = np.zeros((nx * ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    C_occ = torch.tensor(C, dtype=DTYPE, device=device)

    params = config.get().as_dict()
    params.update(device=device, torch_dtype=DTYPE)
    return C_occ, params


def load_bf_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    f_net = CTNNJastrowVCycle(
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
    ).to(device).to(DTYPE)
    f_net.load_state_dict(ckpt["jas_state"])
    f_net.eval()

    bfc = ckpt.get("bf_config")
    if bfc is None:
        # Resume checkpoints from run_weak_form may omit bf_config.
        base_ckpt = torch.load(RESULTS_DIR / "bf_ctnn_vcycle.pt", map_location=device)
        bfc = base_ckpt["bf_config"]
    bf_net = CTNNBackflowNet(
        d=bfc["d"],
        msg_hidden=bfc["msg_hidden"],
        msg_layers=bfc["msg_layers"],
        hidden=bfc["hidden"],
        layers=bfc["layers"],
        act=bfc["act"],
        aggregation=bfc["aggregation"],
        use_spin=bfc["use_spin"],
        same_spin_only=bfc["same_spin_only"],
        out_bound=bfc["out_bound"],
        bf_scale_init=bfc["bf_scale_init"],
        zero_init_last=bfc["zero_init_last"],
        omega=bfc["omega"],
    ).to(device).to(DTYPE)
    bf_net.load_state_dict(ckpt["bf_state"])
    bf_net.eval()
    return f_net, bf_net


def pairwise_geometry(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """x: (B,N,2). Returns per-sample geometry stats."""
    B, N, _ = x.shape
    dmat = torch.cdist(x, x, p=2.0)  # (B,N,N)
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    d_upper = dmat.masked_fill(eye, float("inf"))
    min_pair = d_upper.amin(dim=(1, 2))

    # Unique pairs i<j
    iu = torch.triu_indices(N, N, offset=1, device=x.device)
    pair_vals = dmat[:, iu[0], iu[1]]
    mean_pair = pair_vals.mean(dim=1)
    std_pair = pair_vals.std(dim=1)

    # Radial structure
    r = torch.linalg.norm(x, dim=-1)  # (B,N)
    r_mean = r.mean(dim=1)
    r_max = r.max(dim=1).values
    com = x.mean(dim=1)
    com_r = torch.linalg.norm(com, dim=-1)

    return {
        "min_pair": min_pair,
        "mean_pair": mean_pair,
        "std_pair": std_pair,
        "r_mean": r_mean,
        "r_max": r_max,
        "com_r": com_r,
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-14 or sb < 1e-14:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description="Analyze BF local-energy components and hard configurations")
    ap.add_argument("--ckpt", type=str, default="results/arch_colloc/bf_resume_lr_v1.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n-samples", type=int, default=24000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--burn-in", type=int, default=300)
    ap.add_argument("--thin", type=int, default=3)
    ap.add_argument("--step-sigma", type=float, default=0.12)
    ap.add_argument("--hard-quantile", type=float, default=0.95)
    ap.add_argument("--lap-mode", type=str, default="exact", choices=["exact", "hvp-hutch", "fd-hutch", "fd-central-hutch"])
    ap.add_argument("--lap-probes", type=int, default=24)
    ap.add_argument("--fd-eps", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    device = _device_from_arg(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path

    C_occ, params = setup(device)
    # config.update() applies its own seed policy; reseed here to honor CLI seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    f_net, bf_net = load_bf_checkpoint(ckpt_path, device)

    up = N_ELEC // 2
    spin = torch.cat([
        torch.zeros(up, dtype=torch.long),
        torch.ones(N_ELEC - up, dtype=torch.long),
    ]).to(device)

    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        lp, _ = psi_fn(f_net, x, C_occ, backflow_net=bf_net, spin=spin, params=params)
        return lp

    n_left = args.n_samples
    prev_x = torch.randn(args.batch_size, N_ELEC, DIM, device=device, dtype=DTYPE) * (1.0 / math.sqrt(OMEGA))
    did_burn = False

    E_all = []
    T_all = []
    Vi_all = []
    Vh_all = []
    geo_all = {k: [] for k in ["min_pair", "mean_pair", "std_pair", "r_mean", "r_max", "com_r"]}

    accepted = 0
    proposals = 0

    while n_left > 0:
        bsz = min(args.batch_size, n_left)
        x0 = prev_x[:bsz]
        burn = args.burn_in if not did_burn else 0
        samples, x_last, acc, prop = _metropolis_psi2_persistent(
            psi_log_fn,
            x0,
            burn_in=burn,
            thin=args.thin,
            n_keep=1,
            step_sigma=args.step_sigma,
            target_accept=0.45,
            adapt_lr=0.05,
        )
        did_burn = True
        prev_x = x_last.detach()
        accepted += acc
        proposals += prop

        x = samples.reshape(bsz, N_ELEC, DIM).detach().requires_grad_(True)
        E, T, Vi, Vh, _ = _local_energy_multi(
            psi_log_fn,
            x,
            compute_coulomb_interaction,
            OMEGA,
            lap_mode=args.lap_mode,
            lap_probes=args.lap_probes,
            fd_eps=args.fd_eps,
        )

        geo = pairwise_geometry(x.detach())

        E_all.append(E.detach().cpu().numpy())
        T_all.append(T.detach().cpu().numpy())
        Vi_all.append(Vi.detach().cpu().numpy())
        Vh_all.append(Vh.detach().cpu().numpy())
        for k, v in geo.items():
            geo_all[k].append(v.detach().cpu().numpy())

        n_left -= bsz

    E = np.concatenate(E_all)
    T = np.concatenate(T_all)
    Vi = np.concatenate(Vi_all)
    Vh = np.concatenate(Vh_all)
    geo_np = {k: np.concatenate(v) for k, v in geo_all.items()}

    q = float(args.hard_quantile)
    thr = float(np.quantile(E, q))
    hard = E >= thr

    cov = np.cov(np.stack([T, Vi, Vh], axis=0), bias=False)
    var_sum = float(np.var(E, ddof=1))
    decomp = {
        "Var_E": var_sum,
        "Var_T": float(np.var(T, ddof=1)),
        "Var_Vint": float(np.var(Vi, ddof=1)),
        "Var_Vtrap": float(np.var(Vh, ddof=1)),
        "Cov_T_Vint": float(cov[0, 1]),
        "Cov_T_Vtrap": float(cov[0, 2]),
        "Cov_Vint_Vtrap": float(cov[1, 2]),
    }

    summary = {
        "checkpoint": str(ckpt_path),
        "device": device,
        "seed": int(args.seed),
        "n_samples": int(E.shape[0]),
        "accept_rate": float(accepted / max(1, proposals)),
        "hard_quantile": q,
        "hard_threshold_E": thr,
        "means": {
            "E": float(np.mean(E)),
            "T": float(np.mean(T)),
            "V_int": float(np.mean(Vi)),
            "V_trap": float(np.mean(Vh)),
        },
        "stds": {
            "E": float(np.std(E, ddof=1)),
            "T": float(np.std(T, ddof=1)),
            "V_int": float(np.std(Vi, ddof=1)),
            "V_trap": float(np.std(Vh, ddof=1)),
        },
        "variance_decomposition": decomp,
        "corr_with_E": {
            "corr_E_T": corr(E, T),
            "corr_E_Vint": corr(E, Vi),
            "corr_E_Vtrap": corr(E, Vh),
            "corr_E_min_pair": corr(E, geo_np["min_pair"]),
            "corr_E_mean_pair": corr(E, geo_np["mean_pair"]),
            "corr_E_r_mean": corr(E, geo_np["r_mean"]),
            "corr_E_com_r": corr(E, geo_np["com_r"]),
        },
        "geometry_all": {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)),
                "q10": float(np.quantile(v, 0.10)),
                "q50": float(np.quantile(v, 0.50)),
                "q90": float(np.quantile(v, 0.90)),
            }
            for k, v in geo_np.items()
        },
        "geometry_hard": {
            k: {
                "mean": float(np.mean(v[hard])),
                "std": float(np.std(v[hard], ddof=1)),
                "q10": float(np.quantile(v[hard], 0.10)),
                "q50": float(np.quantile(v[hard], 0.50)),
                "q90": float(np.quantile(v[hard], 0.90)),
            }
            for k, v in geo_np.items()
        },
        "component_hard_vs_all": {
            "E_hard_mean": float(np.mean(E[hard])),
            "T_hard_mean": float(np.mean(T[hard])),
            "Vint_hard_mean": float(np.mean(Vi[hard])),
            "Vtrap_hard_mean": float(np.mean(Vh[hard])),
            "E_all_mean": float(np.mean(E)),
            "T_all_mean": float(np.mean(T)),
            "Vint_all_mean": float(np.mean(Vi)),
            "Vtrap_all_mean": float(np.mean(Vh)),
        },
    }

    ckpt_tag = ckpt_path.stem
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_dir = RESULTS_DIR / "diagnostics" / f"{date.today().isoformat()}_bf_components_{ckpt_tag}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "summary.json"
    npz_path = out_dir / "samples.npz"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        npz_path,
        E=E,
        T=T,
        V_int=Vi,
        V_trap=Vh,
        min_pair=geo_np["min_pair"],
        mean_pair=geo_np["mean_pair"],
        std_pair=geo_np["std_pair"],
        r_mean=geo_np["r_mean"],
        r_max=geo_np["r_max"],
        com_r=geo_np["com_r"],
        hard_mask=hard,
    )

    print(f"Wrote: {json_path}")
    print(f"Wrote: {npz_path}")
    print(
        "Topline: "
        f"E={summary['means']['E']:.6f}, "
        f"std(E)={summary['stds']['E']:.6f}, "
        f"acc={summary['accept_rate']:.3f}, "
        f"hard_q={summary['hard_quantile']:.2f}, "
        f"thr={summary['hard_threshold_E']:.6f}"
    )


if __name__ == "__main__":
    main()
