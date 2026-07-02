"""How much conditioning does the THESIS sampling (importance resampling) buy over the primitive?

The reported PINN/collocation energies used importance resampling (functions.Neural_Networks.
importance_resample): draw n_cand = mult x n_keep candidates from the proposal q, weight by
w = |Psi|^2/q, then MULTINOMIALLY RESAMPLE n_keep points -> they are approximately |Psi|^2-distributed
WITHOUT MCMC. The primitive approach uses the raw proposal points directly (q-distributed, weighted).

We compare, on the SAME number of operator points and the SAME proposal q:
  * primitive : raw points ~ q                       (the ill-conditioned, ESS-collapsed measure)
  * smart     : importance_resample(q)               (points ~ |Psi|^2, the well-conditioned measure)
and report kappa(S), kappa(A_strong) for each, plus the realised sample quality (ESS_raw vs ESS_eff,
PSIS k-hat, number of UNIQUE resampled points -> the diversity the technique can actually deliver).

Prediction (honest): resampling restores the well-conditioned |Psi|^2 operator where ESS_raw is
healthy (high omega) but cannot create diversity that is not there (low omega -> few unique points ->
still ill-conditioned), which is exactly why the technique helped but did not fix the hardest regimes.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import diagnostics as dg  # noqa: E402
from analysis.system import load_system  # noqa: E402
from functions.Neural_Networks import importance_resample as nn_ir  # noqa: E402

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


def _n_unique(x: torch.Tensor) -> int:
    return int(torch.unique(x.reshape(x.shape[0], -1), dim=0).shape[0])


def _kappas(system, x: torch.Tensor, chunk: int) -> dict:
    out = {}
    for name, M in (("S", dg.build_O(system.log_psi, x, system.modules(), center=True)),
                    ("A_strong", dg.residual_jacobian(system, x, form="strong", chunk=chunk))):
        sp = dg.kernel_spectrum(M)
        out[f"kappa_{name}"] = _ec._kappa(sp["eigenvalues"])
        out[f"rank_{name}"] = int(sp["numerical_rank"])
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-op", type=int, default=96, help="operator points (same for both arms)")
    ap.add_argument("--cand-mult", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()

    out = Path(f"results/analysis/{date.today().isoformat()}_resample_vs_primitive")
    out.mkdir(parents=True, exist_ok=True)
    rec = []
    for omega, ck in CKPTS:
        if not Path(ck).exists():
            print(f"[w={omega}] MISSING {ck}"); continue
        s = load_system(ck); s.eval()
        sfs = _ec._mixture_sigma_fs(omega)  # same proposal q for both arms

        # primitive: raw proposal points
        xp = _ec._sample_proposal(s, a.n_op, sfs)
        with torch.no_grad():
            lw = 2 * s.log_psi(xp).double() - _ec._proposal_logq(s, xp, sfs).double()
            lw = lw[torch.isfinite(lw)]; lw -= lw.max(); wq = torch.exp(lw); wq /= wq.sum()
            ess_raw = float(1.0 / (wq**2).sum()) / lw.numel()
        prim = {"arm": "primitive", "omega": omega, "ess_frac": ess_raw,
                "n_unique": a.n_op, **_kappas(s, xp, a.chunk)}

        # smart: importance-resampled points (~|Psi|^2), the actual thesis technique
        xs, _ess, stats = nn_ir(s.log_psi, a.n_op, s.N, s.d, omega, device=s.device, dtype=s.dtype,
                                n_cand_mult=a.cand_mult, sigma_fs=sfs, return_stats=True)
        smart = {"arm": "smart_resample", "omega": omega,
                 "ess_frac_raw": stats["ess_raw"] / (a.cand_mult * a.n_op),
                 "ess_frac_eff": stats["ess_eff"] / (a.cand_mult * a.n_op),
                 "psis_khat": stats["psis_khat"], "top1_mass": stats["top1_mass"],
                 "n_unique": _n_unique(xs), **_kappas(s, xs, a.chunk)}
        rec += [prim, smart]
        print(f"[w={omega:5.2f}] primitive: kS={prim['kappa_S']:.1e} kAs={prim['kappa_A_strong']:.1e} "
              f"(rank {prim['rank_A_strong']}, ESS_raw {100*ess_raw:.2f}%)")
        print(f"          smart    : kS={smart['kappa_S']:.1e} kAs={smart['kappa_A_strong']:.1e} "
              f"(rank {smart['rank_A_strong']}, n_unique {smart['n_unique']}/{a.n_op}, "
              f"khat {smart['psis_khat']:.2f}, top1 {100*smart['top1_mass']:.1f}%)")
        (out / "compare.json").write_text(json.dumps(rec, indent=2) + "\n")

    # figure: kappa(A_strong) primitive vs smart, and unique-point diversity, vs omega
    ws = sorted({r["omega"] for r in rec}, reverse=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, col in (("primitive", "C1"), ("smart_resample", "C0")):
        rs = sorted([r for r in rec if r["arm"] == arm], key=lambda r: -r["omega"])
        axs[0].plot([r["omega"] for r in rs], [r["kappa_A_strong"] for r in rs], "-o", color=col, label=arm)
        axs[1].plot([r["omega"] for r in rs], [r["n_unique"] for r in rs], "-o", color=col, label=arm)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("omega"); ax.legend(fontsize=8)
    axs[0].set_yscale("log"); axs[0].set_ylabel("kappa(A_strong)"); axs[0].set_title("conditioning")
    axs[1].set_ylabel("unique operator points"); axs[1].set_title("realised diversity")
    fig.tight_layout(); fig.savefig(out / "fig_resample_vs_primitive.png", dpi=140); plt.close(fig)
    print(f"[resample_vs_primitive] wrote {out}")


if __name__ == "__main__":
    main()
