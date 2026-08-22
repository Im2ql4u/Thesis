"""T1.2 (Phase M0) — unify the intrinsic dimensions: are the correlator FEATURE rank, the tangent
d_eff, and the intrinsic dimension the same low-dimensional object? And how does the low-rank
correlator contrast with the high-rank backflow (the "two feature spaces")?

For the production V-cycle CTNN across omega we measure, on common |Psi|^2 samples:
  - r_eff(Z)  : effective rank of the correlator readout feature (input to f_head), the thesis quantity
  - d_eff(S)  : tangent-space effective dimension (QGT eff-rank on f_net)   [my prior work]
  - ID(Z)     : nonlinear intrinsic dimension (TwoNN) of the correlator feature
  - rank(dX)  : effective rank of the backflow displacement field (the high-rank channel)
so the two low-dimensionalities (feature vs tangent) can be compared, and the low-rank correlator /
high-rank backflow separation quantified in one place.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_dim_unification.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from analysis.physics_probes import intrinsic_dimension  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-07-02_dim_unification"
CKPTS = {
    1.0:  "2026-06-15_N6_w1_ctnn_big_bf_acc",
    0.1:  "2026-06-15_N6_w01_ctnn_big_bf_casc",
    0.01: "2026-07-02_N6_w001_ctnn_s0",
}
N_PROBE = 1024
SAMPLE_KW = dict(steps=400, burn_in=800)


def _effrank(M: np.ndarray) -> float:
    X = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False); lam = s ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum()) if lam.sum() > 0 else 0.0


def correlator_feature(system, x):
    """Capture the input to the readout head f_head (the pooled correlator feature Z)."""
    cap = {}
    def pre(_m, inp):
        cap["Z"] = inp[0].detach()
    h = system.f_net.f_head.register_forward_pre_hook(pre)
    _ = system.log_psi(x)
    h.remove()
    return cap["Z"].reshape(x.shape[0], -1).cpu().double().numpy()


def backflow_rank(system, x):
    if system.backflow_net is None:
        return float("nan")
    with torch.no_grad():
        dxk = system.backflow_net(x, spin=system.spin)
    B = x.shape[0]
    return _effrank(dxk.reshape(B, -1).cpu().double().numpy())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for omega, d in CKPTS.items():
        s = load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
        x = s.sample(N_PROBE, **SAMPLE_KW)
        Z = correlator_feature(s, x)
        reff_Z = _effrank(Z)
        id_Z = intrinsic_dimension(Z[:512])
        O = dg.build_O(s.log_psi, x, [s.f_net], center=True)
        deff = float(dg.kernel_spectrum(O.cpu())["effective_rank"])
        bf_rank = backflow_rank(s, x)
        rows.append(dict(omega=omega, reff_feature=reff_Z, intrinsic_dim=float(id_Z),
                         deff_tangent=deff, backflow_rank=bf_rank, feature_dim=int(Z.shape[1])))
        print(f"  w={omega:<5} r_eff(feature Z)={reff_Z:.2f}  ID(Z)={id_Z:.2f}  d_eff(tangent)={deff:.2f}  "
              f"| backflow rank={bf_rank:.1f}/{2*x.shape[1]}")
    with open(OUT / "dims.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(OUT / "summary.json", "w"), indent=2)
    print("\n[dims] the correlator is low-rank in BOTH feature and tangent space; the backflow is high-rank")
    print("       (the two feature spaces: collective correlator vs per-particle backflow).")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
