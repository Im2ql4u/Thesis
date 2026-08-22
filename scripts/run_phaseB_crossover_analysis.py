"""Phase B1 analysis — error bars on the omega=0.01 d_eff crossover.

Loads the 3-seed x {CTNN, DeepSet} analysis-grade GS trained at omega=0.01
(scripts/launch_phaseB_crossover.sh) and reports, per architecture (mean +/- seed s.d.):
  - GS quality: energy error vs reference, var(E_L)   (the acceptance gate)
  - fair tangent d_eff on a COMMON pooled probe (all 6 nets' |Psi|^2)
  - natural-orbital participation ratio (physical mode count)
so the headline crossover (CTNN ~5.2 vs DeepSet ~3.24) gets seed error bars and the DeepSet
low value is confirmed on genuine ground states (not an under-converged checkpoint).

Run (HPC): source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u scripts/run_phaseB_crossover_analysis.py
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
OUT = ROOT / "results/analysis/2026-07-02_phaseB_crossover"
STAMP = "2026-07-02"
OMEGA = 0.01
REF_E = 0.69036  # N=6 omega=0.01 reference energy (from config)
SEEDS = [0, 1, 2]
N_POOL = 768     # per net -> 6*768 pooled probe points
N_NO = 384
SAMPLE_KW = dict(steps=400, burn_in=800)


def _no_pr(system):
    x = system.sample(N_NO, **SAMPLE_KW)
    r98 = float(torch.quantile(x.norm(dim=-1).reshape(-1).double(), 0.98).cpu())
    ell = 1.0 / np.sqrt(system.omega)
    no = natural_orbital_occupations(system, x, grid_half=max(4.0 * ell, 1.25 * r98), n_grid=26)
    return float(no["participation_ratio"]), float(no["trace"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    systems, rows = {}, []
    for arch in ("ctnn", "deepset"):
        for s in SEEDS:
            ck = ROOT / f"results/analysis/{STAMP}_N6_w001_{arch}_s{s}/checkpoint.pt"
            if not ck.exists():
                print(f"[warn] missing {ck} — skipping")
                continue
            systems[(arch, s)] = load_system(str(ck), device=dev, seed=0)

    if not systems:
        print("[abort] no checkpoints found yet"); return

    # common pooled probe across ALL available nets (fair d_eff)
    probe = fd.pooled_probe_set(list(systems.values()), N_POOL, **SAMPLE_KW)

    for (arch, s), sysm in systems.items():
        x = sysm.sample(2048, **SAMPLE_KW)
        E_L = dg.local_energy(sysm.log_psi, x, sysm.omega, sysm.params, chunk=256)
        q = dg.gs_quality(E_L, ref_energy=REF_E)
        big = sum(p.numel() for p in sysm.f_net.parameters()) > 1e5
        O = dg.build_O(sysm.log_psi, probe, [sysm.f_net], center=True)
        sp = dg.kernel_spectrum(O.cpu() if big else O)
        no_pr, no_tr = _no_pr(sysm)
        row = dict(arch=arch, seed=s, error_pct=float(q["error_pct"]), var_EL=float(q["var_EL"]),
                   deff=float(sp["effective_rank"]), kappa=float(sp["condition_number"]),
                   no_pr=no_pr, no_trace=no_tr)
        rows.append(row)
        print(f"  {arch:8} s{s}  err={q['error_pct']:+.3f}%  var(E_L)={q['var_EL']:.2e}  "
              f"d_eff={row['deff']:.2f}  NO_PR={no_pr:.2f}  (trace {no_tr:.2f})")

    # aggregate per arch
    agg = {}
    for arch in ("ctnn", "deepset"):
        rs = [r for r in rows if r["arch"] == arch]
        if not rs:
            continue
        agg[arch] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))]
                     for k in ("error_pct", "var_EL", "deff", "no_pr")}
        m = agg[arch]
        print(f"[agg] {arch:8} n={len(rs)}  d_eff={m['deff'][0]:.2f}+/-{m['deff'][1]:.2f}  "
              f"var(E_L)={m['var_EL'][0]:.2e}  err={m['error_pct'][0]:+.3f}%  NO_PR={m['no_pr'][0]:.2f}")

    with open(OUT / "crossover.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"rows": rows, "agg": agg, "omega": OMEGA, "ref_E": REF_E}, open(OUT / "summary.json", "w"), indent=2)
    _figure(rows, agg)
    print(f"-> {OUT}")


def _figure(rows, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    for i, arch in enumerate(("ctnn", "deepset")):
        rs = [r for r in rows if r["arch"] == arch]
        if not rs:
            continue
        col = "C0" if arch == "ctnn" else "C3"
        a1.errorbar(i, agg[arch]["deff"][0], yerr=agg[arch]["deff"][1], fmt="o", color=col,
                    capsize=5, ms=9, label=arch)
        a1.scatter([i] * len(rs), [r["deff"] for r in rs], color=col, alpha=0.4, zorder=3)
        a2.errorbar(i, agg[arch]["var_EL"][0], yerr=agg[arch]["var_EL"][1], fmt="s", color=col, capsize=5, ms=9)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["CTNN", "DeepSet"])
    a1.set_ylabel("d_eff(S)"); a1.set_title("omega=0.01 crossover: tangent dimension (seeded)")
    a1.axhline(0, color="k", lw=0.4); a1.legend()
    a2.set_xticks([0, 1]); a2.set_xticklabels(["CTNN", "DeepSet"])
    a2.set_ylabel("var(E_L)"); a2.set_yscale("log"); a2.set_title("omega=0.01: wavefunction quality")
    fig.tight_layout(); fig.savefig(OUT / "fig_phaseB_crossover.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
