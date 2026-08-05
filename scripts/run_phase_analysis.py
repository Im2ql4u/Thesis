"""Phase analysis driver: train (VMC/CG-SR) a closed-shell NQS ground state for a
given (N, omega), verify it is a genuine ground state, and run the kernel-picture
diagnostics. Generalizable to any even N and any omega.

Outputs (results/analysis/<date>_N{N}_w{tag}_{arch}/):
  * energy convergence, learned Jastrow vs exact (N=2), S/K spectrum,
    SR-vs-plain (NTK whitening) figures
  * diagnostics.npz with all arrays
  * REPORT.md with the verdict and key numbers

Usage:
  python scripts/run_phase_analysis.py --N 2 --omega 1.0
  python scripts/run_phase_analysis.py --N 6 --omega 1.0 --steps 600 --arch ctnn_vcycle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from functions.Neural_Networks import psi_fn  # noqa: E402
from functions.Physics import compute_coulomb_interaction  # noqa: E402
from functions.Stochastic_Reconfiguration import train_model_sr_energy  # noqa: E402

from analysis import diagnostics as dg  # noqa: E402
from analysis.reference import TwoElectronExact  # noqa: E402
from analysis.system import System  # noqa: E402
from analysis.train import (  # noqa: E402
    train_vmc_adam, train_collocation_weak, train_collocation_sr, train_staged_backflow,
)
from analysis.fast_sr import train_sr  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Small CTNN V-cycle configuration (kept modest; analysis is the point, not capacity).
ARCH_KWARGS = {
    "ctnn_vcycle": dict(
        node_hidden=16, edge_hidden=16, bottleneck_hidden=8,
        n_down=1, n_up=1, msg_layers=1, node_layers=1,
        readout_hidden=32, readout_layers=2, act="silu",
    ),
    "ctnn_vcycle_big": dict(
        node_hidden=32, edge_hidden=32, bottleneck_hidden=16,
        n_down=2, n_up=2, msg_layers=2, node_layers=2,
        readout_hidden=64, readout_layers=3, act="silu",
    ),
    "ctnn": dict(node_hidden=16, edge_hidden=16, n_mp_steps=2, msg_layers=1,
                 node_layers=1, readout_hidden=32, readout_layers=2, act="silu"),
    "deepset": dict(pair_hidden=32, pair_layers=3, pair_out=16, readout_hidden=32,
                    readout_layers=2, act="silu"),
    "deepset_big": dict(pair_hidden=64, pair_layers=4, pair_out=32, readout_hidden=64,
                        readout_layers=3, act="silu"),
    # FFNN-equivalence sweep (params: ~7k, ~34k, ~76k(matched to CTNN-big), ~151k)
    "deepset_s": dict(pair_hidden=32, pair_layers=3, pair_out=16, readout_hidden=32,
                      readout_layers=2, act="silu"),
    "deepset_m": dict(pair_hidden=64, pair_layers=4, pair_out=32, readout_hidden=64,
                      readout_layers=3, act="silu"),
    "deepset_match": dict(pair_hidden=96, pair_layers=4, pair_out=48, readout_hidden=96,
                          readout_layers=3, act="silu"),
    "deepset_xl": dict(pair_hidden=128, pair_layers=5, pair_out=64, readout_hidden=128,
                       readout_layers=3, act="silu"),
    "pinn": dict(dL=5, hidden_dim=128, n_layers=2, act="gelu"),
}


def _arch_builder(name: str) -> str:
    if name.startswith("ctnn_vcycle"):
        return "ctnn_vcycle"
    if name.startswith("deepset"):
        return "deepset"
    return name


def build_segments(total: int, n_seg: int) -> list[int]:
    """Denser sampling early (where the kernel changes fastest)."""
    raw = np.geomspace(max(2, total // (4 * n_seg)), total, n_seg)
    cum = np.unique(np.clip(np.round(raw).astype(int), 1, total))
    segs = np.diff(np.concatenate([[0], cum]))
    return [int(s) for s in segs if s > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--arch", type=str, default="ctnn_vcycle", choices=list(ARCH_KWARGS))
    ap.add_argument("--backflow", action="store_true",
                    help="add coordinate backflow (needed for nodes at N>=6)")
    ap.add_argument("--backflow-arch", type=str, default="conv", choices=["conv", "ctnn"],
                    help="conv = per-particle BackflowNet; ctnn = message-passing CTNNBackflowNet (thesis ansatz)")
    ap.add_argument("--staged", action="store_true",
                    help="thesis curriculum (run_6e_bf_extend 'cusp+bf+joint'): Jastrow -> cusp "
                         "pre-train -> backflow with Jastrow frozen -> joint. Without this the "
                         "Jastrow wins the race and the backflow stays near-trivial (rank ~1).")
    ap.add_argument("--bf-scale-init", type=float, default=0.05,
                    help="thesis PINN+backflow runs used 0.7 (run_weak_form's default is 0.05)")
    ap.add_argument("--bf-zero-init-last", type=int, default=1, choices=[0, 1],
                    help="0 = dx_head starts non-zero (thesis PINN+backflow used zero_init_last=False)")
    ap.add_argument("--cusp-steps", type=int, default=300,
                    help="0 skips the cusp pre-train (the thesis 'bf_only'/'bf_then_joint' variants do). "
                         "It MSE-fits Delta_x to a target of magnitude ~0.02 while the backflow lives at "
                         "~0.3, so it can shrink the displacement instead of priming it.")
    ap.add_argument("--cusp-attractive", action="store_true",
                    help="flip the cusp target to point TOWARD same-spin neighbours (the sign the "
                         "original snippet's index convention implies); default is the repulsive "
                         "Pauli-hole sign its comment describes")
    ap.add_argument("--init", type=str, default="",
                    help="warm-start from a previous checkpoint.pt (cascade across omega)")
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sr"])
    ap.add_argument("--paradigm", type=str, default="vmc", choices=["vmc", "colloc"],
                    help="vmc = |Psi|^2-MCMC energy minimisation; colloc = weak-form importance-sampled")
    ap.add_argument("--sigma-q", type=float, default=1.3, help="collocation Gaussian proposal width (units of ell)")
    ap.add_argument("--steps", type=int, default=2500, help="total training steps")
    ap.add_argument("--polish-steps", type=int, default=500,
                    help="final low-lr Adam settle (one optimizer call) for a clean endpoint")
    ap.add_argument("--sr-polish-steps", type=int, default=0,
                    help="natural-gradient (SR) polish after Adam, for near-DMC accuracy")
    ap.add_argument("--sr-lr", type=float, default=0.2)
    ap.add_argument("--sr-damping", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="Adam lr for the backflow; the Jastrow group trains at lr*0.1 (thesis split)")
    ap.add_argument("--batch", type=int, default=2048, help="VMC batch (Adam)")
    ap.add_argument("--n-seg", type=int, default=12, help="diagnostic checkpoints")
    ap.add_argument("--eval-samples", type=int, default=1024)
    ap.add_argument("--final-samples", type=int, default=8192)
    ap.add_argument("--align-samples", type=int, default=2048,
                    help="samples for the final O/alignment (use > score-matrix rank)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-size", type=float, default=0.03)
    ap.add_argument("--damping", type=float, default=1e-2)
    ap.add_argument("--sampler-steps", type=int, default=20)
    ap.add_argument("--sampler-sigma", type=float, default=0.4, help="proposal sigma in units of ell")
    ap.add_argument("--micro-batch", type=int, default=512)
    ap.add_argument("--total-rows", type=int, default=3000)
    ap.add_argument("--outdir", type=str, default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    wtag = f"{a.omega:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    out = Path(a.outdir) if a.outdir else Path(
        f"results/analysis/{date.today().isoformat()}_N{a.N}_w{wtag}_{a.arch}"
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"[phase] N={a.N} omega={a.omega} arch={a.arch}  ->  {out}")

    arch_builder = _arch_builder(a.arch)
    # thesis backflow sizing (run_weak_form.py: --bf-hidden 128, --bf-layers 3, msg_hidden=bf_hidden)
    bf_kwargs = (dict(msg_hidden=128, msg_layers=2, hidden=128, layers=3, act="silu",
                      out_bound="tanh", bf_scale_init=a.bf_scale_init,
                      zero_init_last=bool(a.bf_zero_init_last)) if a.backflow
                 else dict(msg_hidden=32, msg_layers=2, hidden=32, layers=2, act="silu",
                           out_bound="tanh", bf_scale_init=a.bf_scale_init,
                           zero_init_last=bool(a.bf_zero_init_last)))
    sysm = System(N=a.N, omega=a.omega, d=2, arch=arch_builder,
                  arch_kwargs=ARCH_KWARGS[a.arch], use_backflow=a.backflow,
                  backflow_kwargs=bf_kwargs, backflow_arch=a.backflow_arch, seed=a.seed)
    if a.init:
        ck = torch.load(a.init, map_location=sysm.device)
        sysm.f_net.load_state_dict(ck["f_net"])
        if sysm.backflow_net is not None and ck.get("backflow") is not None:
            sysm.backflow_net.load_state_dict(ck["backflow"])
        print(f"[phase] warm-started from {a.init} (omega {ck.get('omega')} -> {a.omega})")

    P = sysm.n_params()
    dev = sysm.device
    ref_energy = config.get().E if np.isfinite(config.get().E) else None
    print(f"[phase] params={P:,} device={dev} ref_energy={ref_energy}")

    exact = TwoElectronExact(omega=a.omega) if a.N == 2 else None

    # ---- train in segments, diagnosing the kernel along the way ----
    traj = {"step": [], "E": [], "E_raw": [], "se": [], "var": [], "cond_S": [], "eff_rank_S": [],
            "num_rank_S": [], "cos_sr": [], "cos_plain": [], "ntk_cond": []}
    segs = [] if a.staged else build_segments(a.steps, a.n_seg)
    done = 0
    t0 = time.time()
    if a.staged:
        # Thesis curriculum. Budget split: Jastrow 30% / backflow 30% / joint 40% of --steps.
        sysm.train()
        train_staged_backflow(
            sysm, jastrow_steps=int(0.3 * a.steps), cusp_steps=a.cusp_steps,
            backflow_steps=int(0.3 * a.steps), joint_steps=int(0.4 * a.steps),
            lr=a.lr, batch=a.batch, cusp_repulsive=not a.cusp_attractive,
        )
        done = a.steps
    for si, seg in enumerate(segs):
        sysm.train()
        if a.paradigm == "colloc":
            if a.optimizer == "sr":
                train_collocation_sr(sysm, steps=seg, batch=min(a.batch, 1024), sigma_q=a.sigma_q,
                                     lr=0.1, damping=1e-3, max_step=0.05, log_every=max(1, seg // 2))
            else:
                train_collocation_weak(sysm, steps=seg, lr=a.lr, batch=a.batch, sigma_q=a.sigma_q,
                                       log_every=max(1, seg // 2))
        elif a.optimizer == "adam":
            train_vmc_adam(
                sysm, steps=seg, lr=a.lr, batch=a.batch,
                sampler_steps=a.sampler_steps, sampler_sigma=a.sampler_sigma,
                lap_mode="exact", log_every=max(1, seg // 2),
            )
        else:
            train_model_sr_energy(
                sysm.f_net, sysm.C_occ, psi_fn=psi_fn,
                compute_coulomb_interaction=compute_coulomb_interaction,
                backflow_net=sysm.backflow_net, spin=sysm.spin, params=sysm.params,
                n_sr_steps=seg, log_every=max(1, seg // 2),
                micro_batch=a.micro_batch, total_rows=a.total_rows,
                sampler_steps=a.sampler_steps, sampler_step_sigma=a.sampler_sigma,
                sampler_sigma_bounds=(0.05, 1.5), lap_mode="exact",
                step_size=a.step_size, max_param_step=a.step_size, damping=a.damping,
            )
        done += seg
        sysm.eval()
        x = sysm.sample(a.eval_samples, steps=200, burn_in=400)
        E_L = dg.local_energy(sysm.log_psi, x, a.omega, sysm.params, lap_mode="exact", chunk=256)
        finite = torch.isfinite(E_L)
        x, E_L = x[finite], E_L[finite]
        q = dg.gs_quality(E_L, ref_energy=ref_energy)
        O = dg.build_O(sysm.log_psi, x, sysm.modules(), center=True)
        sp = dg.kernel_spectrum(O)
        al = dg.sr_vs_plain_alignment(O, E_L)
        traj["step"].append(done)
        traj["E"].append(q["E_mean"]); traj["E_raw"].append(q["E_mean_raw"])
        traj["se"].append(q["E_stderr"])
        traj["var"].append(q["var_EL"]); traj["cond_S"].append(sp["condition_number"])
        traj["eff_rank_S"].append(sp["effective_rank"])
        traj["num_rank_S"].append(sp["numerical_rank"])
        traj["cos_sr"].append(al["cos_sr"]); traj["cos_plain"].append(al["cos_plain"])
        traj["ntk_cond"].append(al["ntk_condition"])
        err_str = "" if ref_energy is None else f" ({q['error_pct']:+.3f}%)"
        print(f"[seg {si+1}/{len(segs)}] step={done} E_raw={q['E_mean_raw']:.5f}{err_str}"
              f" var={q['var_EL']:.3e} kappa(S)={sp['condition_number']:.2e} rank={sp['numerical_rank']}"
              f" cos_sr={al['cos_sr']:.3f} cos_plain={al['cos_plain']:.3f}")

    print(f"[phase] training done in {(time.time()-t0)/60:.1f} min")

    # ---- final polish: one settled, lower-lr, larger-batch run for a clean endpoint ----
    # (driver-only; the trainer is unchanged. A single call => no per-segment momentum resets.)
    # Cap the polish batch by N: the CTNN backflow's (B, N, N, 3*edge_hidden) forward is ~5 GB at
    # N=20, B=4096, which OOM'd every N=20 CTNN chain. The 4096/2048 inflation is a clean-endpoint
    # nicety, not a correctness requirement, so drop it once N^2 makes it too large to fit.
    polish_cap = max(512, 600_000 // (a.N * a.N))   # N=6:16667->4096, N=12:4166->2048+, N=20:1500
    if a.polish_steps > 0:
        sysm.train()
        if a.paradigm == "colloc":
            if a.optimizer == "sr":
                train_collocation_sr(sysm, steps=a.polish_steps, batch=min(a.batch, 1024),
                                     sigma_q=a.sigma_q, lr=0.05, damping=5e-4, max_step=0.02,
                                     log_every=max(1, a.polish_steps // 4))
            else:
                train_collocation_weak(sysm, steps=a.polish_steps, lr=a.lr * 0.3,
                                       batch=min(max(a.batch, 2048), polish_cap), sigma_q=a.sigma_q,
                                       log_every=max(1, a.polish_steps // 4))
        else:
            train_vmc_adam(
                sysm, steps=a.polish_steps, lr=a.lr * 0.2, batch=min(max(a.batch, 4096), polish_cap),
                sampler_steps=a.sampler_steps, sampler_sigma=a.sampler_sigma,
                lap_mode="exact", log_every=max(1, a.polish_steps // 4),
            )
        done += a.polish_steps

    # ---- SR (natural-gradient) polish: the last approach to ~DMC (Adam plateaus above it) ----
    # (VMC-only; a collocation run stays pure collocation -- no VMC/SR polish.)
    if a.sr_polish_steps > 0 and a.paradigm != "colloc":
        sysm.train()
        train_sr(sysm, steps=a.sr_polish_steps, batch=a.batch, lr=a.sr_lr,
                 lr_final=a.sr_lr * 0.05, damping=a.sr_damping,
                 damping_final=max(1e-4, a.sr_damping * 0.05),
                 max_step=0.05, max_step_final=0.005,
                 lap_mode="exact", log_every=max(1, a.sr_polish_steps // 10),
                 ref_energy=ref_energy)
        done += a.sr_polish_steps

    # ---- save the trained checkpoint NOW, before the memory-heavy final verification, so a
    #      large-N OOM in build_O/two-body does not throw away a trained model ----
    torch.save({"f_net": sysm.f_net.state_dict(),
                "backflow": None if sysm.backflow_net is None else sysm.backflow_net.state_dict(),
                "arch": a.arch, "arch_kwargs": ARCH_KWARGS[a.arch],
                "backflow_arch": a.backflow_arch, "backflow_kwargs": bf_kwargs,
                "N": a.N, "omega": a.omega}, out / "checkpoint.pt")

    # ---- final verification on a large sample ----
    sysm.eval()
    xf = sysm.sample(a.final_samples, steps=400, burn_in=800)
    E_Lf = dg.local_energy(sysm.log_psi, xf, a.omega, sysm.params, lap_mode="exact", chunk=256)
    finite = torch.isfinite(E_Lf)
    xf, E_Lf = xf[finite], E_Lf[finite]
    qf = dg.gs_quality(E_Lf, ref_energy=ref_energy)
    # Use more samples than the score-matrix rank so the representable fraction is meaningful.
    n_align = min(a.align_samples, xf.shape[0])
    Of = dg.build_O(sysm.log_psi, xf[:n_align], sysm.modules(), center=True)
    spf = dg.kernel_spectrum(Of)
    alf = dg.sr_vs_plain_alignment(Of, E_Lf[:n_align])
    tb = dg.two_body_correlation(sysm.log_psi, sysm.log_slater, xf)

    # Fold the settled endpoint into the trajectory so the curves end on the polished state.
    if a.polish_steps > 0:
        traj["step"].append(done)
        traj["E"].append(qf["E_mean"]); traj["E_raw"].append(qf["E_mean_raw"])
        traj["se"].append(qf["E_stderr"]); traj["var"].append(qf["var_EL"])
        traj["cond_S"].append(spf["condition_number"])
        traj["eff_rank_S"].append(spf["effective_rank"])
        traj["num_rank_S"].append(spf["numerical_rank"])
        traj["cos_sr"].append(alf["cos_sr"]); traj["cos_plain"].append(alf["cos_plain"])
        traj["ntk_cond"].append(alf["ntk_condition"])

    zv = dg.zero_variance_extrapolation(np.array(traj["E_raw"]), np.array(traj["var"]))

    err_raw = (None if ref_energy is None
               else (qf["E_mean_raw"] - ref_energy) / abs(ref_energy) * 100.0)
    err_zv = (None if ref_energy is None
              else (zv["E_zv"] - ref_energy) / abs(ref_energy) * 100.0)
    summary = {
        "N": a.N, "omega": a.omega, "arch": a.arch, "n_params": P,
        "final": qf, "energy_raw": qf["E_mean_raw"], "error_pct_raw": err_raw,
        "energy_zv": zv["E_zv"], "error_pct_zv": err_zv, "zv_n_points": zv["n_points"],
        "spectrum": {k: spf[k] for k in spf if k != "eigenvalues"},
        "alignment": {k: alf[k] for k in alf if not isinstance(alf[k], np.ndarray)},
        "n_align_samples": int(n_align),
    }
    if exact is not None:
        ov = dg.overlap_with_exact(sysm.log_psi, exact.log_psi, xf)
        summary["exact"] = {
            "energy": exact.energy, "cusp_slope": exact.jastrow_cusp_slope(),
            "overlap_sq": ov["overlap_sq"],
        }

    # ---- assemble ALL plot-source data (compute once, save, then plot from it) ----
    pd_ = _assemble_plot_data(traj, spf, alf, tb, exact, ref_energy, E_Lf, zv)
    pd_["scalars"] = summary

    # ---- save everything: npz (all arrays) + CSVs (key curves) + checkpoint + summary ----
    np.savez(out / "plot_data.npz", **{k: v for k, v in pd_.items() if k != "scalars"})
    _save_csvs(out, pd_)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    torch.save({"f_net": sysm.f_net.state_dict(),
                "backflow": None if sysm.backflow_net is None else sysm.backflow_net.state_dict(),
                "arch": a.arch, "arch_kwargs": ARCH_KWARGS[a.arch],
                "backflow_arch": a.backflow_arch, "backflow_kwargs": bf_kwargs,
                "N": a.N, "omega": a.omega}, out / "checkpoint.pt")

    _make_figures(out, a, pd_)
    _write_report(out, a, summary)
    print("[phase] summary:\n" + json.dumps(summary, indent=2, default=float))


def _bin_curve(r: np.ndarray, y: np.ndarray, nb: int = 40):
    """Quantile-binned mean curve y(r). Returns (r_centers, y_means, y_std, counts)."""
    edges = np.quantile(r, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(r, edges[1:-1]), 0, nb - 1)
    rc, yc, ys, cnt = [], [], [], []
    for b in range(nb):
        m = idx == b
        if np.any(m):
            rc.append(r[m].mean()); yc.append(y[m].mean())
            ys.append(y[m].std()); cnt.append(int(m.sum()))
    return (np.array(rc), np.array(yc), np.array(ys), np.array(cnt))


def _assemble_plot_data(traj, spf, alf, tb, exact, ref_energy, E_Lf, zv) -> dict:
    """Every array behind every figure, gathered so plots and saved data are identical."""
    r = tb["r12"].reshape(-1)
    J = np.repeat(tb["J"], tb["n_pairs"])
    rc, Jc, Jc_std, Jc_cnt = _bin_curve(r, J, nb=40)
    r0 = float(np.nanmedian(rc))
    J0 = float(np.interp(r0, rc, Jc))
    pd_ = {
        # energy convergence
        "traj_step": np.array(traj["step"]), "traj_E_clip": np.array(traj["E"]),
        "traj_E_raw": np.array(traj["E_raw"]), "traj_se": np.array(traj["se"]),
        "traj_var": np.array(traj["var"]),
        # kernel trajectory
        "traj_cond_S": np.array(traj["cond_S"]), "traj_eff_rank_S": np.array(traj["eff_rank_S"]),
        "traj_num_rank_S": np.array(traj["num_rank_S"]),
        "traj_cos_sr": np.array(traj["cos_sr"]), "traj_cos_plain": np.array(traj["cos_plain"]),
        "traj_ntk_cond": np.array(traj["ntk_cond"]),
        "ref_energy": np.array([np.nan if ref_energy is None else ref_energy]),
        "E_zv": np.array([zv["E_zv"]]),
        # final spectrum + whitening
        "eig_S": spf["eigenvalues"], "mu_desc": alf["mu_desc"],
        "residual_power_desc": alf["residual_power_desc"],
        "sr_mode_weight": alf["sr_mode_weight"], "plain_mode_weight": alf["plain_mode_weight"],
        # learned correlation: raw scatter + binned + reference
        "r12_raw": r, "J_raw": J,
        "jastrow_r": rc, "jastrow_J_learned": Jc, "jastrow_J_learned_std": Jc_std,
        "jastrow_J_count": Jc_cnt, "jastrow_r0": np.array([r0]), "jastrow_J0": np.array([J0]),
        # local energies behind the final energy/var
        "final_E_L": E_Lf.detach().cpu().double().numpy(),
    }
    if exact is not None:
        pd_["jastrow_J_exact"] = exact.jastrow_log(rc) - exact.jastrow_log(np.array([r0]))[0]
        r_fine = np.linspace(max(1e-3, rc.min()), rc.max(), 300)
        pd_["jastrow_r_fine"] = r_fine
        pd_["jastrow_J_exact_fine"] = exact.jastrow_log(r_fine) - exact.jastrow_log(np.array([r0]))[0]
    return pd_


def _save_csvs(out, pd_) -> None:
    """Human-readable CSVs for the headline curves."""
    def w(name, cols, arrs):
        n = min(len(a) for a in arrs)
        rows = [",".join(cols)] + [",".join(f"{arrs[c][i]:.8g}" for c in range(len(cols)))
                                   for i in range(n)]
        (out / name).write_text("\n".join(rows) + "\n")

    w("data_energy_convergence.csv", ["step", "E_raw", "E_clip", "stderr", "var"],
      [pd_["traj_step"], pd_["traj_E_raw"], pd_["traj_E_clip"], pd_["traj_se"], pd_["traj_var"]])
    w("data_alignment_trajectory.csv", ["step", "cos_sr", "cos_plain", "kappa_S", "eff_rank_S",
                                        "num_rank_S"],
      [pd_["traj_step"], pd_["traj_cos_sr"], pd_["traj_cos_plain"], pd_["traj_cond_S"],
       pd_["traj_eff_rank_S"], pd_["traj_num_rank_S"]])
    w("data_S_spectrum.csv", ["index", "eigenvalue"],
      [np.arange(1, pd_["eig_S"].size + 1), pd_["eig_S"]])
    w("data_ntk_whitening.csv", ["mode", "mu", "sr_weight", "plain_weight", "residual_power"],
      [np.arange(1, pd_["mu_desc"].size + 1), pd_["mu_desc"], pd_["sr_mode_weight"],
       pd_["plain_mode_weight"], pd_["residual_power_desc"]])
    jcols = ["r", "J_learned", "J_learned_std", "count"]
    jarr = [pd_["jastrow_r"], pd_["jastrow_J_learned"], pd_["jastrow_J_learned_std"],
            pd_["jastrow_J_count"]]
    if "jastrow_J_exact" in pd_:
        jcols.append("J_exact"); jarr.append(pd_["jastrow_J_exact"])
    w("data_jastrow.csv", jcols, jarr)


def _make_figures(out, a, pd_):
    """Render figures strictly from the saved plot_data dict."""
    step = pd_["traj_step"]
    ref = float(pd_["ref_energy"][0])
    has_ref = np.isfinite(ref)

    # (a) energy convergence (unclipped, with zero-variance extrapolation line)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(step, pd_["traj_E_raw"], yerr=pd_["traj_se"], marker="o", lw=1.5, capsize=2,
                label="E (unclipped)")
    ax.plot(step, pd_["traj_E_clip"], "x--", alpha=0.5, label="E (clipped est.)")
    if has_ref:
        ax.axhline(ref, color="k", ls="--", label=f"exact {ref:.5f}")
    ax.axhline(float(pd_["E_zv"][0]), color="g", ls=":", label=f"zero-var extrap {float(pd_['E_zv'][0]):.5f}")
    ax.set_xlabel("training step"); ax.set_ylabel("E (Ha)")
    ax.set_title(f"N={a.N}, omega={a.omega}: energy convergence"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_energy_convergence.png", dpi=140); plt.close(fig)

    # (b) learned Jastrow vs exact
    rc = pd_["jastrow_r"]; Jc = pd_["jastrow_J_learned"]; J0 = float(pd_["jastrow_J0"][0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc, Jc - J0, "o", label="learned  J_net")
    if "jastrow_J_exact_fine" in pd_:
        ax.plot(pd_["jastrow_r_fine"], pd_["jastrow_J_exact_fine"], "k--",
                label="exact  log u + omega r^2/4")
    ax.set_xlabel("pair distance r12"); ax.set_ylabel("J(r) - J(r0)")
    ax.set_title("learned correlation vs exact"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_jastrow_vs_exact.png", dpi=140); plt.close(fig)

    # (c) S spectrum
    lam = pd_["eig_S"]; lam = lam[lam > 0]
    sc = pd_["scalars"]["spectrum"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, lam.size + 1), lam, "o-", ms=3)
    ax.set_xlabel("index"); ax.set_ylabel("eigenvalue of S = O^T O / B")
    ax.set_title(f"QGT/NTK spectrum  (eff_rank={sc['effective_rank']:.1f}, "
                 f"kappa={sc['condition_number']:.1e}, P={sc['n_params']})")
    fig.tight_layout(); fig.savefig(out / "fig_S_spectrum.png", dpi=140); plt.close(fig)

    # (d) NTK whitening + alignment trajectory
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(1, pd_["mu_desc"].size + 1)
    axs[0].semilogy(k, pd_["plain_mode_weight"], label="plain grad weight (mu_a/mu_max)")
    axs[0].semilogy(k, np.clip(pd_["sr_mode_weight"], 1e-12, None), label="SR weight (=1)")
    rp = pd_["residual_power_desc"]; rp = rp / (rp.max() + 1e-30)
    axs[0].semilogy(k, np.clip(rp, 1e-12, None), alpha=0.5, label="residual power (norm.)")
    axs[0].set_xlabel("NTK mode (sorted by mu)"); axs[0].set_ylabel("weight")
    axs[0].set_title("NTK whitening (final)"); axs[0].legend(fontsize=8)
    axs[1].plot(step, pd_["traj_cos_sr"], "o-", label="cos(SR, imag-time)")
    axs[1].plot(step, pd_["traj_cos_plain"], "s-", label="cos(plain, imag-time)")
    axs[1].set_xlabel("training step"); axs[1].set_ylabel("alignment with Hamiltonian flow")
    axs[1].set_ylim(-0.05, 1.05); axs[1].legend(fontsize=8)
    axs[1].set_title("SR vs plain gradient")
    fig.tight_layout(); fig.savefig(out / "fig_ntk_whitening.png", dpi=140); plt.close(fig)


def _write_report(out, a, summary):
    f = summary["final"]
    er_raw = summary.get("error_pct_raw")
    er_zv = summary.get("error_pct_zv")
    lines = [
        f"# Phase analysis: N={a.N}, omega={a.omega}, arch={a.arch}",
        "",
        f"- params: {summary['n_params']:,}",
        f"- **E (unclipped) = {summary['energy_raw']:.6f} +/- {f['E_stderr']:.6f} Ha**"
        + ("" if er_raw is None else f"  (exact {summary['final'].get('ref_energy')}, err {er_raw:+.3f}%)"),
        f"- E (zero-variance extrap, {summary['zv_n_points']} pts) = {summary['energy_zv']:.6f} Ha"
        + ("" if er_zv is None else f"  (err {er_zv:+.3f}%)"),
        f"- E (clipped est.) = {f['E_mean']:.6f} Ha; var(E_L) = {f['var_EL']:.4e}",
        f"- QGT/NTK: eff_rank = {summary['spectrum']['effective_rank']:.2f}, "
        f"kappa(S) = {summary['spectrum']['condition_number']:.3e}, "
        f"numerical rank = {summary['spectrum']['numerical_rank']}/{summary['n_params']} "
        f"(alignment on {summary['n_align_samples']} samples)",
        f"- alignment (final): cos(SR)={summary['alignment']['cos_sr']:.3f}, "
        f"cos(plain)={summary['alignment']['cos_plain']:.3f}, "
        f"NTK kappa={summary['alignment']['ntk_condition']:.3e}",
    ]
    if "exact" in summary:
        e = summary["exact"]
        lines += [
            "",
            "## Exact-truth checks (N=2)",
            f"- exact energy = {e['energy']:.6f} Ha; exact Jastrow cusp dJ/dr|0 = {e['cusp_slope']:.4f}",
            f"- **|<Psi_net|Psi_exact>|^2 = {e['overlap_sq']:.6f}**",
        ]
    lines += ["", "## Data files (all plot inputs saved)",
              "- plot_data.npz : every array behind every figure",
              "- data_energy_convergence.csv, data_alignment_trajectory.csv, data_S_spectrum.csv, "
              "data_ntk_whitening.csv, data_jastrow.csv",
              "- summary.json : all scalar metrics; checkpoint.pt : trained weights",
              "",
              "Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, fig_ntk_whitening."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
