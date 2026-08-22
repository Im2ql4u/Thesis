"""Phase A (no-training) closers for Q1 — run on the existing N=6 cascade checkpoints.

Closes the interpretation of the CTNN-vs-DeepSet effective-dimension story without training:

  A2  physical mode count.  Natural-orbital participation ratio (1-RDM) for BOTH architectures
      across omega, paired with the fair tangent d_eff. Tests whether each net's tangent dimension
      TRACKS the physical collective-mode count. Grid half-width is scaled to the sampled density so
      the low-omega (spread-out Wigner) 1-RDM is not truncated (trace ~ N_up is the QC).

  A3  cross-architecture tangent projection.  On a COMMON pooled probe set, how many DeepSet NTK
      eigen-directions are needed to represent CTNN's leading collective mode, and vice versa? This
      quantifies "the pairwise net spends k directions on what message passing carries in one" —
      the mechanism for why a separable FFNN cannot form a single collective coordinate.

  A5  sample-convergence of d_eff at low omega (the fragile crossover point).

Run (HPC): source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u scripts/run_phaseA_closers.py
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
OUT = ROOT / "results/analysis/2026-07-02_phaseA_closers"

CASCADE = {
    1.0:  {"CTNN": "2026-06-15_N6_w1_ctnn_big_bf_acc",   "DeepSet": "2026-06-15_N6_w1_deepset_big_bf_casc"},
    0.5:  {"CTNN": "2026-06-15_N6_w05_ctnn_big_bf_casc",  "DeepSet": "2026-06-15_N6_w05_deepset_big_bf_casc"},
    0.1:  {"CTNN": "2026-06-15_N6_w01_ctnn_big_bf_casc",  "DeepSet": "2026-06-15_N6_w01_deepset_big_bf_casc"},
    0.01: {"CTNN": "2026-06-15_N6_w001_ctnn_big_bf_casc", "DeepSet": "2026-06-15_N6_w001_deepset_big_bf_casc"},
}
N_POOL = 1024            # per arch -> 2048 pooled probe points (matches the omega-sweep)
N_NO = 384               # samples for the natural-orbital 1-RDM estimator
SAMPLE_KW = dict(steps=400, burn_in=800)
KMAX = 6                 # cross-projection: report capture in top-1..KMAX modes
CONV_OMEGA = {1.0, 0.01}  # A5 sample-convergence only where it matters
CONV_GRID = [256, 512, 1024, 2048]


def _no_pr(system, n=N_NO):
    """Natural-orbital participation ratio with grid half-width scaled to the sampled density."""
    x = system.sample(n, **SAMPLE_KW)
    r98 = float(torch.quantile(x.norm(dim=-1).reshape(-1).double(), 0.98).cpu())
    ell = 1.0 / np.sqrt(system.omega)
    grid_half = max(4.0 * ell, 1.25 * r98)   # cover the Wigner ring at low omega
    no = natural_orbital_occupations(system, x, grid_half=grid_half, n_grid=26)
    no["grid_half"] = grid_half
    no["r98"] = r98
    return no


def _top_eigvecs(O: torch.Tensor, k: int):
    """Top-k NTK eigenvectors (function-space, in R^B) and the full eigenvalue spectrum."""
    B = O.shape[0]
    K = (O.double() @ O.double().t()) / B          # (B,B) == NTK/B, same space for both arches
    evals, evecs = torch.linalg.eigh(K)            # ascending
    order = torch.argsort(evals, descending=True)
    return evecs[:, order[:k]], evals[order]


def _cross_projection(O_c, O_d):
    """Subspace overlap between the two architectures' effective tangent spaces (common probe)."""
    k = min(KMAX, O_c.shape[0] - 1)
    Uc, lam_c = _top_eigvecs(O_c, k)
    Ud, lam_d = _top_eigvecs(O_d, k)
    u, v = Uc[:, 0], Ud[:, 0]
    cap_c_in_d = [float((Ud[:, :j].t() @ u).pow(2).sum()) for j in range(1, k + 1)]   # CTNN top mode in DS top-j
    cap_d_in_c = [float((Uc[:, :j].t() @ v).pow(2).sum()) for j in range(1, k + 1)]   # DS top mode in CTNN top-j
    def k90(caps):
        for j, c in enumerate(caps, start=1):
            if c >= 0.9:
                return j
        return None
    # principal angles between the top-3 subspaces (singular values of Uc^T Ud = cos of angles)
    K0 = min(3, k)
    sv = torch.linalg.svdvals(Uc[:, :K0].t() @ Ud[:, :K0]).cpu().numpy()
    return {"cap_ctnn_top1_in_ds": cap_c_in_d, "cap_ds_top1_in_ctnn": cap_d_in_c,
            "k90_ctnn_in_ds": k90(cap_c_in_d), "k90_ds_in_ctnn": k90(cap_d_in_c),
            "principal_cos_top3": sv.tolist(),
            "subspace_overlap_top3": float((sv ** 2).mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mode_rows, proj_rows, conv_rows = [], [], []

    for omega in sorted(CASCADE, reverse=True):
        sysm = {a: load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
                for a, d in CASCADE[omega].items()}

        # common pooled probe (same points for both arches)
        probe = fd.pooled_probe_set([sysm["CTNN"], sysm["DeepSet"]], N_POOL, **SAMPLE_KW)

        # tangent d_eff (fair) + natural-orbital PR, per arch
        O = {}
        for a in sysm:
            big = sum(p.numel() for p in sysm[a].f_net.parameters()) > 1e5
            O[a] = dg.build_O(sysm[a].log_psi, probe, [sysm[a].f_net], center=True)
            sp = dg.kernel_spectrum(O[a].cpu() if big else O[a])
            no = _no_pr(sysm[a])
            deff, no_pr = float(sp["effective_rank"]), float(no["participation_ratio"])
            mode_rows.append(dict(omega=omega, arch=a, n_params=int(sp["n_params"]),
                                  deff_tangent=deff, no_pr=no_pr, ratio=deff / no_pr,
                                  no_trace=float(no["trace"]), no_lead_occ=float(no["leading_occ"]),
                                  grid_half=float(no["grid_half"])))
            print(f"  w={omega:<5} {a:8} d_eff={deff:.2f}  NO_PR={no_pr:.2f}  "
                  f"ratio(d_eff/NO)={deff/no_pr:.2f}  trace={no['trace']:.2f}")

        # A3: cross-architecture tangent projection on the common probe
        proj = _cross_projection(O["CTNN"], O["DeepSet"])
        proj_rows.append(dict(omega=omega, **proj))
        print(f"    A3 w={omega}: CTNN top mode captured by DeepSet top-k = "
              f"{[round(c,2) for c in proj['cap_ctnn_top1_in_ds']]} (k90={proj['k90_ctnn_in_ds']}); "
              f"DS top mode in CTNN top-k = {[round(c,2) for c in proj['cap_ds_top1_in_ctnn']]} "
              f"(k90={proj['k90_ds_in_ctnn']}); top-3 overlap={proj['subspace_overlap_top3']:.2f}")

        # A5: sample-convergence at the endpoints
        if omega in CONV_OMEGA:
            for a in sysm:
                big = sum(p.numel() for p in sysm[a].f_net.parameters()) > 1e5
                for rec in fd.dimension_convergence(sysm[a], probe, CONV_GRID, svd_on_cpu=big):
                    conv_rows.append(dict(omega=omega, arch=a, **rec))
            print(f"    A5 w={omega}: convergence "
                  + " | ".join(f"{r['arch']} n={r['n_samples']}:{r['eff_rank']:.2f}"
                               for r in conv_rows if r["omega"] == omega))

    _write(mode_rows, proj_rows, conv_rows)
    _figure(mode_rows, proj_rows)
    print(f"\n-> {OUT}")


def _write(mode_rows, proj_rows, conv_rows):
    with open(OUT / "modecount.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mode_rows[0].keys())); w.writeheader(); w.writerows(mode_rows)
    with open(OUT / "convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conv_rows[0].keys())); w.writeheader(); w.writerows(conv_rows)
    json.dump({"modecount": mode_rows, "crossproj": proj_rows, "convergence": conv_rows},
              open(OUT / "summary.json", "w"), indent=2)


def _figure(mode_rows, proj_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    omegas = sorted({r["omega"] for r in mode_rows}, reverse=True)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for arch, col in (("CTNN", "C0"), ("DeepSet", "C3")):
        deff = [next(r for r in mode_rows if r["omega"] == w and r["arch"] == arch)["deff_tangent"] for w in omegas]
        nopr = [next(r for r in mode_rows if r["omega"] == w and r["arch"] == arch)["no_pr"] for w in omegas]
        ratio = [next(r for r in mode_rows if r["omega"] == w and r["arch"] == arch)["ratio"] for w in omegas]
        a1.plot(omegas, deff, "o-", color=col, label=f"{arch} d_eff(tangent)")
        a1.plot(omegas, nopr, "s--", color=col, alpha=0.5, label=f"{arch} NO count")
        a2.plot(omegas, ratio, "o-", color=col, label=arch)
    a1.set_xscale("log"); a1.set_xlabel("omega"); a1.set_ylabel("dimension")
    a1.set_title("A2: tangent d_eff vs physical mode count"); a1.legend(fontsize=7)
    a2.set_xscale("log"); a2.axhline(1.0, color="k", lw=0.6, ls=":")
    a2.set_xlabel("omega"); a2.set_ylabel("d_eff(tangent) / NO count")
    a2.set_title("A2: tangent tracks physics? (=1 ideal)"); a2.legend(fontsize=8)
    # A3: capture of CTNN top mode by DeepSet top-k, per omega
    for r in proj_rows:
        ks = list(range(1, len(r["cap_ctnn_top1_in_ds"]) + 1))
        a3.plot(ks, r["cap_ctnn_top1_in_ds"], "o-", label=f"w={r['omega']}")
    a3.axhline(0.9, color="k", lw=0.6, ls=":")
    a3.set_xlabel("# DeepSet NTK directions (k)"); a3.set_ylabel("captured fraction of CTNN top mode")
    a3.set_title("A3: DeepSet directions to represent\nCTNN's leading collective coordinate")
    a3.legend(fontsize=8); a3.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(OUT / "fig_phaseA_closers.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
