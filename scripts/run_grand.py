"""Grand probe set on a checkpoint: backflow/nodes [1], entanglement [2], conditioning/SNR [3],
bootstrap CIs [4]. Writes grand.json next to the checkpoint.

  python scripts/run_grand.py --load <dir>/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from analysis import physics_probes as pp  # noqa: E402
from analysis.ablation import ablate_messages, has_messages  # noqa: E402
from analysis.backflow_probes import backflow_analysis, backflow_ablation_energy  # noqa: E402
from analysis.system import load_system  # noqa: E402


def _try(name, fn, store, errs):
    try:
        store[name] = fn(); print(f"  [ok] {name}")
    except Exception as e:  # noqa: BLE001
        errs[name] = repr(e); print(f"  [FAIL] {name}: {e}"); traceback.print_exc()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


@torch.no_grad()
def _manybody_perconfig(system, x):
    logf = system.log_psi(x).double()
    with ablate_messages(system.f_net, "zero"):
        loga = system.log_psi(x).double()
    return (logf - loga).abs().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--samples", type=int, default=2048)
    a = ap.parse_args()
    sysm = load_system(a.load)
    out = Path(a.load).parent
    ref = config.get().E if np.isfinite(config.get().E) else None
    is_ctnn = has_messages(sysm.f_net)
    print(f"[grand] N={sysm.N} omega={sysm.omega} ctnn={is_ctnn} bf={sysm.backflow_net is not None} -> {out}")
    x = sysm.sample(a.samples, steps=400, burn_in=800)
    E_L = dg.local_energy(sysm.log_psi, x, sysm.omega, sysm.params, lap_mode="exact")
    E_L = E_L[torch.isfinite(E_L)]
    res, errs = {}, {}
    res["base"] = {"N": sysm.N, "omega": sysm.omega, "is_ctnn": is_ctnn,
                   "E": float(E_L.mean()), "var_EL": float(E_L.var()),
                   "err_pct": None if ref is None else float((E_L.mean() - ref) / abs(ref) * 100)}

    # [1] backflow / nodes
    if sysm.backflow_net is not None:
        _try("backflow", lambda: backflow_analysis(sysm, x), res, errs)
        _try("backflow_ablation", lambda: backflow_ablation_energy(sysm, n_samples=a.samples // 2), res, errs)
    # [2] entanglement (occupation-spectrum entropies)
    _try("entanglement", lambda: {k: v for k, v in
         pp.natural_orbital_occupations(sysm, x[:640], n_grid=28).items()
         if k != "occupations"}, res, errs)
    # [3] conditioning / SNR
    Oc = dg.build_O(sysm.log_psi, x[: min(1024, a.samples)], sysm.modules(), center=True)
    _try("conditioning", lambda: {**{k: v for k, v in dg.kernel_spectrum(Oc).items() if k != "eigenvalues"},
                                  **dg.gradient_snr(Oc, E_L[: Oc.shape[0]])}, res, errs)
    # [4] bootstrap CIs on headline sample-based quantities
    _try("ci_energy", lambda: dg.bootstrap_ci(E_L.cpu().numpy()), res, errs)
    if is_ctnn:
        _try("ci_manybody", lambda: dg.bootstrap_ci(_manybody_perconfig(sysm, x[:1024])), res, errs)

    res["errors"] = errs
    (out / "grand.json").write_text(json.dumps(_clean(res), indent=2) + "\n")
    print("[grand] " + json.dumps(_clean(res), indent=2))


if __name__ == "__main__":
    main()
