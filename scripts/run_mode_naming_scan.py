"""Mode-naming scan: extend the N=2/N=6 operator-decomposition analysis to a full
omega sweep, multiple seeds, and N=12.

For each trained checkpoint we take the correlator's top NTK eigenfunctions (the
effective tangent directions) and ask how much of the LEADING one lies in the span of a
physical-operator dictionary (breathing/monopole, quartic, quadrupole, correlation-hole,
pair-linear, pair-Coulomb). r2_leading close to 1 => the network's dominant tangent
direction is a nameable physical collective mode. No training; existing checkpoints only.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_mode_naming_scan.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
# reuse the exact operator / eigenvector machinery
from run_mode_naming_N6 import operators, top_eigvecs, _center_unit, KTOP  # noqa: E402

OUT = ROOT / "results/analysis/2026-08-28_mode_naming_scan"
AN = ROOT / "results/analysis"
N_PROBE = 1536
SAMPLE_KW = dict(steps=400, burn_in=800)

# Reference ground-state energies (from tab:energies) for the training-quality gate.
# A checkpoint is only used if its measured energy is within GATE_PCT of the reference;
# omega values with no published reference are gated on local-energy variance instead.
REF: dict[tuple[int, float], float] = {
    (6, 1.0): 20.15932, (6, 0.5): 11.78484, (6, 0.1): 3.55385, (6, 0.01): 0.69036,
    (12, 1.0): 65.7001, (12, 0.5): 39.1596, (12, 0.1): 12.2698, (12, 0.01): 2.47363,
}
GATE_PCT = 1.5   # max |energy error| vs reference to count as well-trained
GATE_RELVAR = 0.05  # max Var(E_L)/E^2 when no reference is available

# (N, omega, arch, seed, checkpoint_dir_relative_to_results/analysis)
CKPTS: list[tuple[int, float, str, int, str]] = []

# ---- N=6, CTNN vs DeepSet, full omega sweep (single seed except w=0.01) ----
N6 = {
    1.0:  ("2026-06-15_N6_w1_ctnn_big_bf_acc",    "2026-06-15_N6_w1_deepset_big_bf_casc"),
    0.5:  ("2026-06-15_N6_w05_ctnn_big_bf_casc",  "2026-06-15_N6_w05_deepset_big_bf_casc"),
    0.28: ("2026-06-23_N6_w028_ctnn_xfill",       "2026-06-23_N6_w028_deepset_xfill"),
    0.1:  ("2026-06-15_N6_w01_ctnn_big_bf_casc",  "2026-06-15_N6_w01_deepset_big_bf_casc"),
    0.05: ("2026-06-23_N6_w005_ctnn_xfill",       "2026-06-23_N6_w005_deepset_xfill"),
    0.03: ("2026-06-23_N6_w003_ctnn_xfill",       "2026-06-23_N6_w003_deepset_xfill"),
}
for w, (c, d) in N6.items():
    CKPTS.append((6, w, "CTNN", 0, c))
    CKPTS.append((6, w, "DeepSet", 0, d))
# w=0.01 with three seeds each -> seed robustness at the critical Wigner point
for sd in (0, 1, 2):
    CKPTS.append((6, 0.01, "CTNN", sd, f"2026-07-02_N6_w001_ctnn_s{sd}"))
    CKPTS.append((6, 0.01, "DeepSet", sd, f"2026-07-02_N6_w001_deepset_s{sd}"))

# ---- N=12: CTNN vs DeepSet at w=1 (clean correlator comparison) ----
CKPTS.append((12, 1.0, "CTNN", 0, "2026-07-04_N12scaling/ctnn_w1"))
CKPTS.append((12, 1.0, "DeepSet", 0, "2026-07-04_N12scaling/deepset_w1"))
# ---- N=12: CTNN correlator across omega, two seeds (no DeepSet foil below w=1) ----
WTAG = {1.0: "w1", 0.5: "w0p5", 0.1: "w0p1", 0.01: "w0p01"}
for w, tag in WTAG.items():
    for sd in (0, 1):
        CKPTS.append((12, w, "CTNN", sd, f"2026-07-16_scaling/N12_ctnnbf_s{sd}_{tag}"))


def analyse(ckpt_dir: str, N: int, omega: float, dev: str) -> dict | None:
    path = AN / ckpt_dir / "checkpoint.pt"
    if not path.exists():
        return None
    s = load_system(str(path), device=dev, seed=0)
    x = s.sample(N_PROBE, **SAMPLE_KW)

    # --- training-quality gate: measure energy and compare to reference ---
    ref = REF.get((N, round(omega, 3)))
    EL = dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256)
    q = dg.gs_quality(EL, ref_energy=ref)
    if ref is not None:
        trained_ok = abs(q.get("error_pct", 1e9)) <= GATE_PCT
    else:  # no reference: use the zero-variance principle as a training-quality proxy
        rel_var = q["var_EL"] / max(q["E_mean"] ** 2, 1e-12)
        trained_ok = rel_var <= GATE_RELVAR

    # --- mode naming on the correlator tangent kernel ---
    M, names = operators(x, omega)
    Mu = _center_unit(M)
    U = top_eigvecs(s, x, KTOP)
    coef, *_ = np.linalg.lstsq(Mu, U[:, 0], rcond=None)
    resid = U[:, 0] - Mu @ coef
    r2_top = float(1 - (resid @ resid) / (U[:, 0] @ U[:, 0]))
    corr = {names[j]: float(abs(Mu[:, j] @ U[:, 0])) for j in range(len(names))}
    top_op = max(corr, key=corr.get)
    return dict(r2_leading=r2_top, leading_operator=top_op, top_corr=corr[top_op],
                E_mean=q["E_mean"], error_pct=q.get("error_pct", float("nan")),
                var_EL=q["var_EL"], trained_ok=bool(trained_ok))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for (N, w, arch, sd, d) in CKPTS:
        try:
            r = analyse(d, N, w, dev)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] N={N} w={w} {arch} s{sd} {d}: {e!r}", flush=True)
            continue
        if r is None:
            print(f"[MISS] N={N} w={w} {arch} s{sd} {d}", flush=True)
            continue
        row = dict(N=N, omega=w, arch=arch, seed=sd, **r)
        rows.append(row)
        flag = "OK " if r["trained_ok"] else "REJECT"
        print(f"[{flag}] N={N:2d} w={w:<5} {arch:8} s{sd}  E={r['E_mean']:.4f} "
              f"err%={r['error_pct']:+.3f}  R2_leading={r['r2_leading']:.3f}  "
              f"top_op={r['leading_operator']:13}", flush=True)

    with open(OUT / "scan.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["N", "omega", "arch", "seed", "trained_ok",
                                             "E_mean", "error_pct", "var_EL", "r2_leading",
                                             "leading_operator", "top_corr"])
        wtr.writeheader()
        wtr.writerows(rows)
    # seed-aggregated summary per (N, omega, arch) -- WELL-TRAINED checkpoints only
    agg = {}
    for row in rows:
        if not row["trained_ok"]:
            continue
        key = f"N{row['N']}_w{row['omega']}_{row['arch']}"
        agg.setdefault(key, []).append(row["r2_leading"])
    summary = {k: dict(r2_mean=float(np.mean(v)), r2_std=float(np.std(v)), n_seeds=len(v))
               for k, v in agg.items()}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n-> {OUT}  ({len(rows)} checkpoints analysed)")


if __name__ == "__main__":
    main()
