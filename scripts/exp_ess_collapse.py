"""Phase 0 (decisive, threshold-free): effective sample size of the collocation mixture.

The collocation trainer corrects its broad Gaussian mixture q back to |Psi|^2 with importance
weights w = |Psi|^2/q. ESS = (sum w)^2 / sum w^2 is the effective number of points actually
contributing -- a clean, threshold-free number (unlike kappa, which floors). Low ESS means the
gradient / Fisher is built from a handful of points: that is the real collocation bottleneck and
it directly explains (i) why VMC (|Psi|^2 sampling, weights=1, ESS=N) is easy, (ii) why low-omega /
large-N collocation fails, (iii) why Adam+ESS beats natural-gradient at low omega (SR needs a stable
Fisher, which needs ESS).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.system import load_system  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reuse the mixture sampler / log-density already written for the conditioning probe
import importlib.util as _u  # noqa: E402
_spec = _u.spec_from_file_location("ec", str(Path(__file__).resolve().parent / "exp_conditioning_A.py"))
_ec = _u.module_from_spec(_spec); _spec.loader.exec_module(_ec)

CKPTS = [
    (1.0, "results/analysis/2026-06-15_N6_w1_ctnn_big_bf_acc/checkpoint.pt"),
    (0.5, "results/analysis/2026-06-15_N6_w05_ctnn_big_bf_casc/checkpoint.pt"),
    (0.1, "results/analysis/2026-06-15_N6_w01_ctnn_big_bf_casc/checkpoint.pt"),
    (0.01, "results/analysis/2026-06-15_N6_w001_ctnn_big_bf_casc/checkpoint.pt"),
]
N_DRAW = 4096   # large draw so the ESS-fraction estimate is itself stable
N_REPEAT = 5


@torch.no_grad()
def ess_fraction(system, n: int) -> float:
    x = _ec._sample_mixture(system, n)
    lw = 2.0 * system.log_psi(x).double() - _ec._mixture_logq(system, x).double()
    lw = lw[torch.isfinite(lw)]
    lw = lw - lw.max()
    w = torch.exp(lw); w = w / w.sum()
    return float(1.0 / (w**2).sum()) / lw.numel()


def main() -> None:
    out = Path(f"results/analysis/{date.today().isoformat()}_ess_collapse")
    out.mkdir(parents=True, exist_ok=True)
    rec = []
    for omega, ck in CKPTS:
        if not Path(ck).exists():
            print(f"[w={omega}] MISSING {ck}"); continue
        s = load_system(ck); s.eval()
        fr = [ess_fraction(s, N_DRAW) for _ in range(N_REPEAT)]
        m, sd = float(np.mean(fr)), float(np.std(fr))
        rec.append({"omega": omega, "ess_frac_mean": m, "ess_frac_std": sd,
                    "ess_eff_mean": m * N_DRAW, "n_draw": N_DRAW})
        print(f"[w={omega:5.2f}] ESS fraction = {100*m:.2f}% +/- {100*sd:.2f}%  "
              f"(~{m*N_DRAW:.0f} / {N_DRAW} effective points)")
    (out / "ess.json").write_text(json.dumps(rec, indent=2) + "\n")

    w = [r["omega"] for r in rec]; f = [100 * r["ess_frac_mean"] for r in rec]
    e = [100 * r["ess_frac_std"] for r in rec]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(w, f, yerr=e, fmt="-o", color="C3")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("omega"); ax.set_ylabel("ESS fraction (%)")
    ax.set_title("Collocation mixture ESS collapse vs omega (N=6)")
    fig.tight_layout(); fig.savefig(out / "fig_ess_collapse.png", dpi=140); plt.close(fig)
    print(f"[ess_collapse] wrote {out}")


if __name__ == "__main__":
    main()
