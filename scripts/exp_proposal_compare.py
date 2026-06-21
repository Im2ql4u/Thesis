"""How much does the smart (adaptive multi-width) proposal buy over a simple Gaussian?

For each proposal family and each omega we measure the coverage-vs-efficiency tradeoff that governs
collocation training:
  * ESS fraction  = (sum w)^2 / sum w^2 / n,  w = |Psi|^2/q   -- estimator efficiency (higher better)
  * coalescence coverage = fraction of points with min pair distance < 0.5/sqrt(omega)
        -- does the proposal actually visit the cusp/correlation-hole region the residual lives in?
  * tail reach    = 95th pct of max single-particle radius (in oscillator lengths)
  * kappa(A_strong), kappa(S) under that measure (optional; --kappa)  -- conditioning

Simple Gaussians (one width) are efficient but blind to the hard regions; the smart mixture buys
coverage at an ESS cost. This quantifies that exchange-rate.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import diagnostics as dg  # noqa: E402
from analysis.system import load_system  # noqa: E402

import importlib.util as _u  # noqa: E402
_spec = _u.spec_from_file_location("ec", str(Path(__file__).resolve().parent / "exp_conditioning_A.py"))
_ec = _u.module_from_spec(_spec); _spec.loader.exec_module(_ec)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CKPTS = [
    (1.0, "results/analysis/2026-06-15_N6_w1_ctnn_big_bf_acc/checkpoint.pt"),
    (0.5, "results/analysis/2026-06-15_N6_w05_ctnn_big_bf_casc/checkpoint.pt"),
    (0.1, "results/analysis/2026-06-15_N6_w01_ctnn_big_bf_casc/checkpoint.pt"),
    (0.01, "results/analysis/2026-06-15_N6_w001_ctnn_big_bf_casc/checkpoint.pt"),
]


def proposals(omega: float) -> dict:
    return {
        "gauss_narrow": (1.0,),
        "gauss_matched": (1.3,),          # run_weak_form.sample_gauss default
        "gauss_broad": (2.5,),
        "mixture_smart": _ec._mixture_sigma_fs(omega),  # adaptive multi-width
    }


@torch.no_grad()
def ess_and_coverage(system, sigma_fs, n: int) -> dict:
    ell = 1.0 / math.sqrt(float(system.omega))
    x = _ec._sample_proposal(system, n, sigma_fs)
    lw = 2.0 * system.log_psi(x).double() - _ec._proposal_logq(system, x, sigma_fs).double()
    fin = torch.isfinite(lw)
    lw = lw[fin] - lw[fin].max()
    w = torch.exp(lw); w = w / w.sum()
    ess = float(1.0 / (w**2).sum()) / lw.numel()
    # coalescence coverage + tail reach
    dmat = torch.cdist(x, x)
    eye = torch.eye(system.N, device=x.device, dtype=torch.bool).unsqueeze(0)
    min_pair = dmat.masked_fill(eye, float("inf")).amin(dim=(1, 2))
    coal = float((min_pair < 0.5 * ell).float().mean())
    rmax = x.norm(dim=2).amax(dim=1)               # max single-particle radius per config
    tail = float(torch.quantile(rmax, 0.95) / ell)
    return {"ess_frac": ess, "coalescence_cov": coal, "tail_reach_ell": tail}


def kappa_under(system, sigma_fs, n: int, chunk: int) -> dict:
    x = _ec._sample_proposal(system, n, sigma_fs)
    out = {}
    for name, M in (("S", dg.build_O(system.log_psi, x, system.modules(), center=True)),
                    ("A_strong", dg.residual_jacobian(system, x, form="strong", chunk=chunk))):
        out[name] = _ec._kappa(dg.kernel_spectrum(M)["eigenvalues"])
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--kappa", action="store_true", help="also measure kappa (slow)")
    ap.add_argument("--kappa-samples", type=int, default=96)
    ap.add_argument("--kappa-omegas", type=str, default="1.0,0.1")
    a = ap.parse_args()

    out = Path(f"results/analysis/{date.today().isoformat()}_proposal_compare")
    out.mkdir(parents=True, exist_ok=True)
    kap_om = set(float(s) for s in a.kappa_omegas.split(","))
    rec = []
    for omega, ck in CKPTS:
        if not Path(ck).exists():
            print(f"[w={omega}] MISSING {ck}"); continue
        s = load_system(ck); s.eval()
        for pname, sfs in proposals(omega).items():
            stats = [ess_and_coverage(s, sfs, a.draw) for _ in range(a.repeat)]
            row = {"omega": omega, "proposal": pname, "n_components": len(sfs)}
            for k in ("ess_frac", "coalescence_cov", "tail_reach_ell"):
                row[k] = float(np.mean([d[k] for d in stats]))
                row[k + "_std"] = float(np.std([d[k] for d in stats]))
            if a.kappa and omega in kap_om:
                row.update({f"kappa_{k}": v for k, v in
                            kappa_under(s, sfs, a.kappa_samples, 16).items()})
            rec.append(row)
            msg = (f"[w={omega:5.2f}] {pname:14s} ESS={100*row['ess_frac']:5.2f}%  "
                   f"coal_cov={100*row['coalescence_cov']:5.1f}%  tail={row['tail_reach_ell']:.1f}l")
            if "kappa_A_strong" in row:
                msg += f"  kS={row['kappa_S']:.1e} kAs={row['kappa_A_strong']:.1e}"
            print(msg)
        (out / "compare.json").write_text(json.dumps(rec, indent=2) + "\n")

    # figure: ESS and coalescence coverage vs omega, per proposal
    pnames = list(proposals(1.0).keys())
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for pname in pnames:
        rows = sorted([r for r in rec if r["proposal"] == pname], key=lambda r: r["omega"])
        w = [r["omega"] for r in rows]
        axs[0].plot(w, [100 * r["ess_frac"] for r in rows], "-o", label=pname)
        axs[1].plot(w, [100 * r["coalescence_cov"] for r in rows], "-o", label=pname)
    for ax, ttl, yl in zip(axs, ("ESS fraction", "coalescence coverage"), ("ESS %", "coverage %")):
        ax.set_xscale("log"); ax.set_xlabel("omega"); ax.set_ylabel(yl); ax.set_title(ttl); ax.legend(fontsize=8)
    axs[0].set_yscale("log")
    fig.tight_layout(); fig.savefig(out / "fig_proposal_compare.png", dpi=140); plt.close(fig)
    print(f"[proposal_compare] wrote {out}")


if __name__ == "__main__":
    main()
