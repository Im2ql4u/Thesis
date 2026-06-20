"""Run the over-smoothing / non-separability / collectivity battery on a checkpoint.

Measurements 1,2,3,5,7 (collectivity.py) + 4 (kappa(S)). 6 (training-speed) is a separate 2x2.
Writes collectivity.json next to the checkpoint.

  python scripts/run_collectivity.py --load <dir>/checkpoint.pt
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
from analysis import collectivity as col  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from analysis.ablation import has_messages  # noqa: E402
from analysis.system import load_system  # noqa: E402


def _try(name, fn, store, errs):
    try:
        store[name] = fn()
        print(f"  [ok] {name}")
    except Exception as e:  # noqa: BLE001
        errs[name] = repr(e)
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--samples", type=int, default=1024)
    a = ap.parse_args()
    sysm = load_system(a.load)
    out = Path(a.load).parent
    is_ctnn = has_messages(sysm.f_net)
    print(f"[collectivity] N={sysm.N} omega={sysm.omega} ctnn={is_ctnn} -> {out}")
    x = sysm.sample(a.samples, steps=400, burn_in=800)

    res, errs = {}, {}
    res["base"] = {"N": sysm.N, "omega": sysm.omega, "is_ctnn": is_ctnn,
                   "n_params": sysm.n_params()}
    # [4] Fisher/QGT conditioning
    Oc = dg.build_O(sysm.log_psi, x[: min(1024, a.samples)], sysm.modules(), center=True)
    _try("kappa_S", lambda: {k: v for k, v in dg.kernel_spectrum(Oc).items() if k != "eigenvalues"},
         res, errs)
    # [3] spectral content (works for any arch)
    _try("spectral", lambda: {k: v for k, v in col.spectral_content(sysm, x).items()
                              if k in ("k_centroid", "k95")}, res, errs)
    if is_ctnn:
        _try("dirichlet", lambda: col.dirichlet_smoothing(sysm, x[:512]), res, errs)        # [1,5]
        _try("non_separability", lambda: col.non_separability(sysm, x[:256]), res, errs)    # [2]
        _try("meanfield_alignment", lambda: col.meanfield_alignment(sysm, x[:512]), res, errs)  # [7]
    res["errors"] = errs
    (out / "collectivity.json").write_text(json.dumps(_clean(res), indent=2) + "\n")
    print("[collectivity] " + json.dumps(_clean(res), indent=2))


if __name__ == "__main__":
    main()
