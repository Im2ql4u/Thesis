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
from analysis.train import train_vmc_adam  # noqa: E402

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
    "ctnn": dict(node_hidden=16, edge_hidden=16, n_mp_steps=2, msg_layers=1,
                 node_layers=1, readout_hidden=32, readout_layers=2, act="silu"),
    "deepset": dict(hidden=32, layers=2, readout_hidden=32, readout_layers=2, act="silu"),
    "pinn": dict(hidden_dim=64, n_layers=2, act="silu"),
}


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
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sr"])
    ap.add_argument("--steps", type=int, default=2500, help="total training steps")
    ap.add_argument("--lr", type=float, default=3e-3, help="Adam learning rate")
    ap.add_argument("--batch", type=int, default=2048, help="VMC batch (Adam)")
    ap.add_argument("--n-seg", type=int, default=12, help="diagnostic checkpoints")
    ap.add_argument("--eval-samples", type=int, default=512)
    ap.add_argument("--final-samples", type=int, default=4096)
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

    sysm = System(N=a.N, omega=a.omega, d=2, arch=a.arch,
                  arch_kwargs=ARCH_KWARGS[a.arch], seed=a.seed)
    P = sysm.n_params()
    dev = sysm.device
    ref_energy = config.get().E if np.isfinite(config.get().E) else None
    print(f"[phase] params={P:,} device={dev} ref_energy={ref_energy}")

    exact = TwoElectronExact(omega=a.omega) if a.N == 2 else None

    # ---- train in segments, diagnosing the kernel along the way ----
    traj = {"step": [], "E": [], "se": [], "var": [], "cond_S": [], "eff_rank_S": [],
            "cos_sr": [], "cos_plain": [], "ntk_cond": []}
    segs = build_segments(a.steps, a.n_seg)
    done = 0
    t0 = time.time()
    for si, seg in enumerate(segs):
        sysm.train()
        if a.optimizer == "adam":
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
        E_L = dg.local_energy(sysm.log_psi, x, a.omega, sysm.params, lap_mode="exact")
        finite = torch.isfinite(E_L)
        x, E_L = x[finite], E_L[finite]
        q = dg.gs_quality(E_L, ref_energy=ref_energy)
        O = dg.build_O(sysm.log_psi, x, sysm.modules(), center=True)
        sp = dg.kernel_spectrum(O)
        al = dg.sr_vs_plain_alignment(O, E_L)
        traj["step"].append(done)
        traj["E"].append(q["E_mean"]); traj["se"].append(q["E_stderr"])
        traj["var"].append(q["var_EL"]); traj["cond_S"].append(sp["condition_number"])
        traj["eff_rank_S"].append(sp["effective_rank"])
        traj["cos_sr"].append(al["cos_sr"]); traj["cos_plain"].append(al["cos_plain"])
        traj["ntk_cond"].append(al["ntk_condition"])
        err_str = "" if ref_energy is None else f" ({q['error_pct']:+.3f}%)"
        print(f"[seg {si+1}/{len(segs)}] step={done} E={q['E_mean']:.5f}{err_str}"
              f" var={q['var_EL']:.3e} kappa(S)={sp['condition_number']:.2e}"
              f" cos_sr={al['cos_sr']:.3f} cos_plain={al['cos_plain']:.3f}")

    print(f"[phase] training done in {(time.time()-t0)/60:.1f} min")

    # ---- final verification on a large sample ----
    sysm.eval()
    xf = sysm.sample(a.final_samples, steps=400, burn_in=800)
    E_Lf = dg.local_energy(sysm.log_psi, xf, a.omega, sysm.params, lap_mode="exact")
    finite = torch.isfinite(E_Lf)
    xf, E_Lf = xf[finite], E_Lf[finite]
    qf = dg.gs_quality(E_Lf, ref_energy=ref_energy)
    Of = dg.build_O(sysm.log_psi, xf[: min(1024, a.final_samples)], sysm.modules(), center=True)
    spf = dg.kernel_spectrum(Of)
    alf = dg.sr_vs_plain_alignment(Of, E_Lf[: Of.shape[0]])
    tb = dg.two_body_correlation(sysm.log_psi, sysm.log_slater, xf)

    summary = {
        "N": a.N, "omega": a.omega, "arch": a.arch, "n_params": P,
        "final": qf, "spectrum": {k: spf[k] for k in spf if k != "eigenvalues"},
        "alignment": {k: alf[k] for k in alf if not isinstance(alf[k], np.ndarray)},
    }
    if exact is not None:
        ov = dg.overlap_with_exact(sysm.log_psi, exact.log_psi, xf)
        summary["exact"] = {
            "energy": exact.energy, "cusp_slope": exact.jastrow_cusp_slope(),
            "overlap_sq": ov["overlap_sq"],
        }

    # ---- save arrays + checkpoint ----
    np.savez(
        out / "diagnostics.npz",
        **{f"traj_{k}": np.array(v) for k, v in traj.items()},
        eig_S=spf["eigenvalues"], mu_desc=alf["mu_desc"],
        residual_power=alf["residual_power_desc"],
        sr_weight=alf["sr_mode_weight"], plain_weight=alf["plain_mode_weight"],
        r12=tb["r12"], J=tb["J"],
    )
    torch.save({"f_net": sysm.f_net.state_dict(),
                "backflow": None if sysm.backflow_net is None else sysm.backflow_net.state_dict(),
                "arch": a.arch, "arch_kwargs": ARCH_KWARGS[a.arch],
                "N": a.N, "omega": a.omega}, out / "checkpoint.pt")

    _make_figures(out, a, traj, spf, alf, tb, exact, ref_energy)
    _write_report(out, a, summary, traj)
    print("[phase] summary:\n" + json.dumps(summary, indent=2, default=float))


def _make_figures(out, a, traj, spf, alf, tb, exact, ref_energy):
    step = np.array(traj["step"])

    # (a) energy convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(step, traj["E"], yerr=traj["se"], marker="o", lw=1.5, capsize=2)
    if ref_energy is not None:
        ax.axhline(ref_energy, color="k", ls="--", label=f"reference {ref_energy:.5f}")
        ax.legend()
    ax.set_xlabel("CG-SR step"); ax.set_ylabel("E (Ha)")
    ax.set_title(f"N={a.N}, omega={a.omega}: energy convergence")
    fig.tight_layout(); fig.savefig(out / "fig_energy_convergence.png", dpi=140); plt.close(fig)

    # (b) learned Jastrow vs exact (N=2 exact; else just learned)
    r = tb["r12"].reshape(-1); J = np.repeat(tb["J"], tb["n_pairs"])
    order = np.argsort(r)
    nb = 40
    edges = np.quantile(r, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(r, edges[1:-1]), 0, nb - 1)
    rc = np.array([r[idx == b].mean() if np.any(idx == b) else np.nan for b in range(nb)])
    Jc = np.array([J[idx == b].mean() if np.any(idx == b) else np.nan for b in range(nb)])
    r0 = np.nanmedian(rc)
    J0 = np.interp(r0, rc[~np.isnan(rc)], Jc[~np.isnan(rc)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc, Jc - J0, "o-", label="learned  J_net")
    if exact is not None:
        Je = exact.jastrow_log(rc) - exact.jastrow_log(np.array([r0]))[0]
        ax.plot(rc, Je, "k--", label="exact  log u + omega r^2/4")
        ax.text(0.05, 0.9, f"exact cusp dJ/dr|0 = {exact.jastrow_cusp_slope():.3f}",
                transform=ax.transAxes)
    ax.set_xlabel("pair distance r12"); ax.set_ylabel("J(r) - J(r0)")
    ax.set_title("learned correlation vs exact"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_jastrow_vs_exact.png", dpi=140); plt.close(fig)

    # (c) S spectrum
    lam = spf["eigenvalues"]; lam = lam[lam > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, lam.size + 1), lam, "o-", ms=3)
    ax.set_xlabel("index"); ax.set_ylabel("eigenvalue of S = O^T O / B")
    ax.set_title(f"QGT/NTK spectrum  (eff_rank={spf['effective_rank']:.1f}, "
                 f"kappa={spf['condition_number']:.1e}, P={spf['n_params']})")
    fig.tight_layout(); fig.savefig(out / "fig_S_spectrum.png", dpi=140); plt.close(fig)

    # (d) NTK whitening: per-mode weights + residual power; and alignment trajectory
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(1, alf["mu_desc"].size + 1)
    axs[0].semilogy(k, alf["plain_mode_weight"], label="plain grad weight (mu_a/mu_max)")
    axs[0].semilogy(k, np.clip(alf["sr_mode_weight"], 1e-12, None), label="SR weight (=1)")
    rp = alf["residual_power_desc"]; rp = rp / (rp.max() + 1e-30)
    axs[0].semilogy(k, np.clip(rp, 1e-12, None), alpha=0.5, label="residual power (norm.)")
    axs[0].set_xlabel("NTK mode (sorted by mu)"); axs[0].set_ylabel("weight")
    axs[0].set_title("NTK whitening (final)"); axs[0].legend(fontsize=8)
    axs[1].plot(step, traj["cos_sr"], "o-", label="cos(SR, imag-time)")
    axs[1].plot(step, traj["cos_plain"], "s-", label="cos(plain, imag-time)")
    axs[1].set_xlabel("CG-SR step"); axs[1].set_ylabel("alignment with Hamiltonian flow")
    axs[1].set_ylim(-0.05, 1.05); axs[1].legend(fontsize=8)
    axs[1].set_title("SR vs plain gradient")
    fig.tight_layout(); fig.savefig(out / "fig_ntk_whitening.png", dpi=140); plt.close(fig)


def _write_report(out, a, summary, traj):
    f = summary["final"]
    lines = [
        f"# Phase analysis: N={a.N}, omega={a.omega}, arch={a.arch}",
        "",
        f"- params: {summary['n_params']:,}",
        f"- **E = {f['E_mean']:.6f} +/- {f['E_stderr']:.6f} Ha**"
        + (f"  (ref {f.get('ref_energy')}, err {f.get('error_pct'):+.3f}%, "
           f"{f.get('error_sigma'):+.1f} sigma)" if "ref_energy" in f else ""),
        f"- var(E_L) = {f['var_EL']:.4e}",
        f"- QGT/NTK: eff_rank = {summary['spectrum']['effective_rank']:.2f}, "
        f"kappa(S) = {summary['spectrum']['condition_number']:.3e}, "
        f"numerical rank = {summary['spectrum']['numerical_rank']}/{summary['n_params']}",
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
    lines += ["", "Figures: fig_energy_convergence, fig_jastrow_vs_exact, fig_S_spectrum, "
              "fig_ntk_whitening. Arrays: diagnostics.npz. Checkpoint: checkpoint.pt."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
