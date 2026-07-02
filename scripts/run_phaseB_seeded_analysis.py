"""Phase B seeded analysis — error bars on the Q1 dimension gap, two regimes.

Reports, per architecture (mean +/- seed s.d.) on a COMMON pooled probe:
  - GS quality: energy error vs reference, var(E_L)   (acceptance gate)
  - fair tangent d_eff (f_net)
  - natural-orbital participation ratio (physical mode count)

Two configs:
  crossover : omega=0.01, the 3-seed matched-grade GS trained in Phase B1
              (launch_phaseB_crossover.sh). Puts error bars on the (retracted) inversion.
  anchor    : omega=1.0, the EXISTING independent Phase-0/1 runs (5 CTNN-big, 3 DeepSet-big) --
              no retraining; these are genuinely independent basins. Confirms the weak-coupling
              compression gap (CTNN ~1.4 vs DeepSet ~3.3) with seeds.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_phaseB_seeded_analysis.py [crossover|anchor]
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
from analysis import fair_dimension as fd  # noqa: E402
from analysis.physics_probes import natural_orbital_occupations  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A = "results/analysis"

CONFIGS = {
    "crossover": dict(
        omega=0.01, ref_E=0.69036, out="2026-07-02_phaseB_crossover_reanalysis",
        ckpts=[("ctnn", f"s{s}", f"{A}/2026-07-02_N6_w001_ctnn_s{s}") for s in (0, 1, 2)]
              + [("deepset", f"s{s}", f"{A}/2026-07-02_N6_w001_deepset_s{s}") for s in (0, 1, 2)],
    ),
    "anchor": dict(
        omega=1.0, ref_E=20.15932, out="2026-07-02_phaseB_anchor",
        ckpts=[
            ("ctnn", "seed1", f"{A}/2026-06-15_eq_N6w1_ctnn_seed1"),
            ("ctnn", "seed2", f"{A}/2026-06-15_eq_N6w1_ctnn_seed2"),
            ("ctnn", "2x2adam", f"{A}/2026-06-15_2x2_N6w1_ctnn_adam"),
            ("ctnn", "2x2sr", f"{A}/2026-06-15_2x2_N6w1_ctnn_sr"),
            ("ctnn", "acc", f"{A}/2026-06-15_N6_w1_ctnn_big_bf_acc"),
            ("deepset", "2x2adam", f"{A}/2026-06-15_2x2_N6w1_deepset_adam"),
            ("deepset", "2x2sr", f"{A}/2026-06-15_2x2_N6w1_deepset_sr"),
            ("deepset", "casc", f"{A}/2026-06-15_N6_w1_deepset_big_bf_casc"),
        ],
    ),
}
N_POOL = 512     # per net; kept modest so the pooled probe stays small with many systems
N_NO = 384
SAMPLE_KW = dict(steps=400, burn_in=800)


def _no_pr(system):
    x = system.sample(N_NO, **SAMPLE_KW)
    r98 = float(torch.quantile(x.norm(dim=-1).reshape(-1).double(), 0.98).cpu())
    ell = 1.0 / np.sqrt(system.omega)
    no = natural_orbital_occupations(system, x, grid_half=max(4.0 * ell, 1.25 * r98), n_grid=26)
    return float(no["participation_ratio"]), float(no["trace"])


def main(which: str) -> None:
    cfg = CONFIGS[which]
    out = ROOT / A / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    systems = {}
    for arch, label, d in cfg["ckpts"]:
        ck = ROOT / d / "checkpoint.pt"
        if not ck.exists():
            print(f"[warn] missing {ck} — skipping"); continue
        systems[(arch, label)] = load_system(str(ck), device=dev, seed=0)
    if not systems:
        print("[abort] no checkpoints"); return

    probe = fd.pooled_probe_set(list(systems.values()), N_POOL, **SAMPLE_KW)
    rows = []
    for (arch, label), sysm in systems.items():
        x = sysm.sample(2048, **SAMPLE_KW)
        q = dg.gs_quality(dg.local_energy(sysm.log_psi, x, sysm.omega, sysm.params, chunk=256),
                          ref_energy=cfg["ref_E"])
        O = dg.build_O(sysm.log_psi, probe, [sysm.f_net], center=True)
        sp = dg.kernel_spectrum(O.cpu())  # SVD on CPU: safe with many nets on a shared GPU
        del O
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        no_pr, no_tr = _no_pr(sysm)
        rows.append(dict(arch=arch, label=label, error_pct=float(q["error_pct"]), var_EL=float(q["var_EL"]),
                         deff=float(sp["effective_rank"]), no_pr=no_pr, no_trace=no_tr))
        print(f"  {arch:8} {label:8} err={q['error_pct']:+.3f}%  var={q['var_EL']:.2e}  "
              f"d_eff={sp['effective_rank']:.2f}  NO_PR={no_pr:.2f}")

    agg = {}
    for arch in ("ctnn", "deepset"):
        rs = [r for r in rows if r["arch"] == arch]
        if not rs:
            continue
        agg[arch] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])), len(rs)]
                     for k in ("error_pct", "var_EL", "deff", "no_pr")}
        m = agg[arch]
        print(f"[agg] {arch:8} n={m['deff'][2]}  d_eff={m['deff'][0]:.2f}+/-{m['deff'][1]:.2f}  "
              f"var={m['var_EL'][0]:.2e}  err={m['error_pct'][0]:+.3f}%  NO={m['no_pr'][0]:.2f}")

    with open(out / "seeded.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"which": which, "omega": cfg["omega"], "rows": rows, "agg": agg},
              open(out / "summary.json", "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "anchor")
