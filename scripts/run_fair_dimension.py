"""Phase 1 / Q1 — fair cross-architecture effective dimension at fixed (N, omega).

Re-measures eff-rank(S) of the f_net (Jastrow) tangent space for CTNN seeds and the DeepSet
equivalence ladder on a COMMON probe set (mixture of CTNN + DeepSet |Psi|^2), with a
sample-convergence sweep and a measure-sensitivity check (pooled / CTNN-density / DeepSet-density).
This turns the own-density depth numbers into a fair comparison (same points, same estimator).

Run (HPC): source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
  PYTHONUNBUFFERED=1 python3 -u scripts/run_fair_dimension.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import fair_dimension as fd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-06-22_fair_dimension_N6w1"

# (label, family, checkpoint dir under results/analysis/)
MODELS = [
    ("ctnn_seed1", "CTNN", "2026-06-15_eq_N6w1_ctnn_seed1"),
    ("ctnn_seed2", "CTNN", "2026-06-15_eq_N6w1_ctnn_seed2"),
    ("ctnn_2x2sr", "CTNN", "2026-06-15_2x2_N6w1_ctnn_sr"),
    ("deepset_s_20k", "DeepSet", "2026-06-15_eq_N6w1_ds_s"),
    ("deepset_m_48k", "DeepSet", "2026-06-15_eq_N6w1_ds_m"),
    ("deepset_match_89k", "DeepSet", "2026-06-15_eq_N6w1_dsmatch_sr"),
    ("deepset_xl_164k", "DeepSet", "2026-06-15_eq_N6w1_ds_xl"),
]
N_GRID = [256, 512, 1024, 2048]
N_PER_POOL = 1536  # per family -> pooled probe of ~3072
SAMPLE_KW = dict(steps=400, burn_in=800)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    systems = {lab: load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
               for lab, fam, d in MODELS}
    fam = {lab: f for lab, f, _ in MODELS}
    nparams = {lab: sum(p.numel() for p in systems[lab].f_net.parameters()) for lab in systems}

    # --- common probe set: mixture of one CTNN and one DeepSet density (covers both families) ---
    ref_ctnn = systems["ctnn_seed1"]
    ref_ds = systems["deepset_match_89k"]
    probes = {
        "pooled": fd.pooled_probe_set([ref_ctnn, ref_ds], N_PER_POOL, **SAMPLE_KW),
        "ctnn_density": ref_ctnn.sample(2 * N_PER_POOL, **SAMPLE_KW),
        "ds_density": ref_ds.sample(2 * N_PER_POOL, **SAMPLE_KW),
    }
    print(f"[fair-dim] device={dev} probe sizes: " + ", ".join(f"{k}={v.shape[0]}" for k, v in probes.items()))

    conv_rows, fair_rows = [], []
    for lab in systems:
        s = systems[lab]
        big = nparams[lab] > 1.0e5  # xl -> SVD on CPU to avoid OOM
        # convergence + final fair value under the pooled measure
        conv = fd.dimension_convergence(s, probes["pooled"], N_GRID, svd_on_cpu=big)
        for r in conv:
            conv_rows.append(dict(model=lab, family=fam[lab], measure="pooled", **r))
        eff_final = conv[-1]["eff_rank"]
        # measure sensitivity at the largest n
        sens = {}
        for mk in ("ctnn_density", "ds_density"):
            sp = fd.effective_dimension(s, probes[mk][:N_GRID[-1]], svd_on_cpu=big)
            sens[mk] = float(sp["effective_rank"])
            conv_rows.append(dict(model=lab, family=fam[lab], measure=mk, n_samples=N_GRID[-1],
                                  eff_rank=sens[mk], kappa=float(sp["condition_number"]),
                                  num_rank=int(sp["numerical_rank"]), n_params=nparams[lab]))
        fair_rows.append(dict(model=lab, family=fam[lab], n_params=nparams[lab],
                              eff_rank_pooled=eff_final, eff_rank_ctnnpts=sens["ctnn_density"],
                              eff_rank_dspts=sens["ds_density"], kappa=conv[-1]["kappa"]))
        print(f"  {lab:20} {fam[lab]:8} P={nparams[lab]:>6}  eff-rank: pooled={eff_final:.2f} "
              f"ctnn-pts={sens['ctnn_density']:.2f} ds-pts={sens['ds_density']:.2f}")

    # --- save ---
    with open(OUT / "convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conv_rows[0].keys())); w.writeheader(); w.writerows(conv_rows)
    with open(OUT / "fair_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fair_rows[0].keys())); w.writeheader(); w.writerows(fair_rows)
    ct = [r["eff_rank_pooled"] for r in fair_rows if r["family"] == "CTNN"]
    ds = [r["eff_rank_pooled"] for r in fair_rows if r["family"] == "DeepSet"]
    summary = dict(N=6, omega=1.0, n_grid=N_GRID, probe_pooled_size=int(probes["pooled"].shape[0]),
                   ctnn_eff_rank_mean=sum(ct) / len(ct), ctnn_eff_rank_range=[min(ct), max(ct)],
                   deepset_eff_rank_range=[min(ds), max(ds)], fair_rows=fair_rows)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)

    _figure(conv_rows, fair_rows)
    print(f"[fair-dim] CTNN eff-rank {min(ct):.2f}-{max(ct):.2f} vs DeepSet {min(ds):.2f}-{max(ds):.2f} "
          f"(pooled measure) -> {OUT}")


def _figure(conv_rows, fair_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(11, 4))
    # convergence under pooled measure
    models = sorted({r["model"] for r in conv_rows})
    for m in models:
        pts = [r for r in conv_rows if r["model"] == m and r["measure"] == "pooled"]
        pts.sort(key=lambda r: r["n_samples"])
        col = "C0" if pts and pts[0]["family"] == "CTNN" else "C3"
        axc.plot([p["n_samples"] for p in pts], [p["eff_rank"] for p in pts], "o-", color=col, alpha=0.7, label=m)
    axc.set_xlabel("n_samples (common pooled probe)"); axc.set_ylabel("eff-rank(S)")
    axc.set_title("Sample-convergence of eff-rank (N=6, w=1)"); axc.legend(fontsize=6, ncol=2)
    # fair bar chart
    fair_rows = sorted(fair_rows, key=lambda r: (r["family"], r["n_params"]))
    labs = [r["model"] for r in fair_rows]
    vals = [r["eff_rank_pooled"] for r in fair_rows]
    cols = ["C0" if r["family"] == "CTNN" else "C3" for r in fair_rows]
    axb.bar(range(len(labs)), vals, color=cols)
    axb.set_xticks(range(len(labs))); axb.set_xticklabels(labs, rotation=45, ha="right", fontsize=6)
    axb.set_ylabel("eff-rank(S) (pooled measure, n=2048)")
    axb.set_title("Fair effective dimension: CTNN (blue) vs DeepSet (red)")
    fig.tight_layout(); fig.savefig(OUT / "fig_fair_dimension.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
