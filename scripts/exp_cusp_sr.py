"""A1.2 cusp x SR 2x2 (N=2): where does SR actually help?

Trains {cusp on/off} x {Adam / Adam+SR-polish} and measures final energy, var(E_L), the NTK
condition number, and the exact-GS alignment. Hypotheses:
  - cusp-off adds a stiff (cusp) NTK direction -> worse conditioning; SR should help there.
  - cusp-on leaves a smooth residual -> SR's whitening gives little / hurts alignment.
Writes results to results/analysis/<date>_cuspSR_N2/summary.json.
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
from analysis.exact_align import exact_alignment  # noqa: E402
from analysis.fast_sr import train_sr  # noqa: E402
from analysis.system import System  # noqa: E402
from analysis.train import train_vmc_adam  # noqa: E402

AKW = dict(node_hidden=16, edge_hidden=16, bottleneck_hidden=8, n_down=1, n_up=1,
           msg_layers=1, node_layers=1, readout_hidden=32, readout_layers=2, act="silu")


def run(cusp: bool, use_sr: bool, seed: int = 0) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    akw = dict(AKW); akw["use_analytic_cusp"] = cusp
    s = System(N=2, omega=1.0, arch="ctnn_vcycle", arch_kwargs=akw, seed=seed)
    train_vmc_adam(s, steps=600, lr=5e-3, batch=2048, log_every=600)
    train_vmc_adam(s, steps=400, lr=1e-3, batch=4096, log_every=400)  # Adam settle
    if use_sr:
        train_sr(s, steps=200, batch=1024, lr=0.2, lr_final=0.01, damping=1e-3,
                 damping_final=1e-4, max_step=0.05, max_step_final=0.005, log_every=200, ref_energy=3.0)
    s.eval()
    x = s.sample(1024, steps=300, burn_in=600)
    E_L = dg.local_energy(s.log_psi, x, 1.0, s.params, lap_mode="exact")
    E_L = E_L[torch.isfinite(E_L)]
    O = dg.build_O(s.log_psi, x, s.modules(), center=True)
    sp = dg.kernel_spectrum(O)
    al = exact_alignment(s, x)
    return {"cusp": cusp, "sr": use_sr, "E": float(E_L.mean()),
            "err_pct": float((E_L.mean() - 3.0) / 3.0 * 100), "var_EL": float(E_L.var()),
            "kappa_S": sp["condition_number"], "eff_rank_S": sp["effective_rank"],
            "cos_plain_exact": al["cos_plain_toward_exact"],
            "cos_sr_exact": al["cos_sr_toward_exact"], "rep_fraction": al["rep_fraction_exact_dir"]}


def main():
    out = Path(f"results/analysis/{date.today().isoformat()}_cuspSR_N2")
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for cusp in (True, False):
        for use_sr in (False, True):
            r = run(cusp, use_sr)
            res.append(r)
            print(f"cusp={cusp} sr={use_sr}: E={r['E']:.5f} ({r['err_pct']:+.3f}%) "
                  f"var={r['var_EL']:.2e} kappa={r['kappa_S']:.2e} "
                  f"cos_plain={r['cos_plain_exact']:.3f} cos_sr={r['cos_sr_exact']:.3f}")
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print("[cuspSR] wrote", out / "summary.json")


if __name__ == "__main__":
    main()
