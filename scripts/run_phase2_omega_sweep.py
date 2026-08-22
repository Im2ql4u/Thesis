"""Phase 2 (no-training half) — omega-sweep of Q1 (effective dimension) and Q3 (estimator variance)
on the existing N=6 CTNN/DeepSet backflow cascade checkpoints.

Tests the two headline Wigner predictions without any training:
  Q1: does the CTNN-vs-DeepSet eff-rank(S) gap GROW toward the Wigner regime (omega -> 0.01)?
  Q3: does var(weak)/var(strong) (the zero-variance lost by dropping the Laplacian) track omega?

Per omega, eff-rank is measured fairly: a COMMON pooled probe set (CTNN+DeepSet |Psi|^2 at that omega)
evaluated for both architectures (f_net tangent space). Reuses fair_dimension + diagnostics.

Run (HPC): source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u scripts/run_phase2_omega_sweep.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from analysis import fair_dimension as fd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-06-23_phase2_omega_sweep"

# omega -> {arch: checkpoint dir}  (cascade + 2026-06-23 crossover-fill points)
CASCADE = {
    1.0:  {"CTNN": "2026-06-15_N6_w1_ctnn_big_bf_acc",   "DeepSet": "2026-06-15_N6_w1_deepset_big_bf_casc"},
    0.5:  {"CTNN": "2026-06-15_N6_w05_ctnn_big_bf_casc",  "DeepSet": "2026-06-15_N6_w05_deepset_big_bf_casc"},
    0.28: {"CTNN": "2026-06-23_N6_w028_ctnn_xfill",       "DeepSet": "2026-06-23_N6_w028_deepset_xfill"},
    0.1:  {"CTNN": "2026-06-15_N6_w01_ctnn_big_bf_casc",  "DeepSet": "2026-06-15_N6_w01_deepset_big_bf_casc"},
    0.05: {"CTNN": "2026-06-23_N6_w005_ctnn_xfill",       "DeepSet": "2026-06-23_N6_w005_deepset_xfill"},
    0.03: {"CTNN": "2026-06-23_N6_w003_ctnn_xfill",       "DeepSet": "2026-06-23_N6_w003_deepset_xfill"},
    0.01: {"CTNN": "2026-06-15_N6_w001_ctnn_big_bf_casc", "DeepSet": "2026-06-15_N6_w001_deepset_big_bf_casc"},
}
N_POOL = 1024   # per arch -> 2048 pooled probe points
N_VAR = 2048    # samples for the variance estimators
SAMPLE_KW = dict(steps=400, burn_in=800)


def _var_stats(R: torch.Tensor) -> float:
    return float(R.detach().double().var(unbiased=True))


def _weak_chunked(system, x, chunk=256):
    outs = [dg.residual_local_energy(system, x[s:s + chunk], form="weak").detach()
            for s in range(0, x.shape[0], chunk)]
    return torch.cat(outs, dim=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for omega in sorted(CASCADE, reverse=True):
        sysm = {a: load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
                for a, d in CASCADE[omega].items()}
        # --- Q1: fair eff-rank on a common pooled probe set at this omega ---
        probe = fd.pooled_probe_set([sysm["CTNN"], sysm["DeepSet"]], N_POOL, **SAMPLE_KW)
        deff = {a: fd.effective_dimension(sysm[a], probe,
                                          svd_on_cpu=sum(p.numel() for p in sysm[a].f_net.parameters()) > 1e5)
                for a in sysm}
        # --- Q3: var(strong)=var(E_L) and var(weak) per arch on own |Psi|^2 ---
        for a in sysm:
            x = sysm[a].sample(N_VAR, **SAMPLE_KW)
            v_s = _var_stats(dg.local_energy(sysm[a].log_psi, x, sysm[a].omega, sysm[a].params, chunk=256))
            v_w = _var_stats(_weak_chunked(sysm[a], x))
            row = dict(omega=omega, arch=a, n_params=int(sum(p.numel() for p in sysm[a].f_net.parameters())),
                       eff_rank=float(deff[a]["effective_rank"]), kappa=float(deff[a]["condition_number"]),
                       var_strong=v_s, var_weak=v_w, var_ratio=(v_w / v_s if v_s > 0 else float("inf")))
            rows.append(row)
            print(f"  w={omega:<5} {a:8} eff_rank={row['eff_rank']:.2f} kappa={row['kappa']:.1e} "
                  f"var(E_L)={v_s:.2e} var(e_w)={v_w:.2e} ratio={row['var_ratio']:.0f}x")
        # gap at this omega
        g = deff["DeepSet"]["effective_rank"] / max(deff["CTNN"]["effective_rank"], 1e-9)
        print(f"  --> w={omega}: eff-rank gap DeepSet/CTNN = {g:.2f}x")

    with open(OUT / "omega_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(OUT / "omega_sweep.json", "w"), indent=2)
    _figure(rows)
    # summary lines
    print("\n[phase2] eff-rank gap (DeepSet/CTNN) vs omega:")
    for omega in sorted(CASCADE, reverse=True):
        ct = next(r for r in rows if r["omega"] == omega and r["arch"] == "CTNN")
        ds = next(r for r in rows if r["omega"] == omega and r["arch"] == "DeepSet")
        print(f"  w={omega:<5} CTNN {ct['eff_rank']:.2f} vs DeepSet {ds['eff_rank']:.2f} "
              f"(gap {ds['eff_rank']/ct['eff_rank']:.2f}x) | var-ratio CTNN {ct['var_ratio']:.0f}x DeepSet {ds['var_ratio']:.0f}x")
    print(f"-> {OUT}")


def _figure(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    omegas = sorted({r["omega"] for r in rows}, reverse=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for arch, col in (("CTNN", "C0"), ("DeepSet", "C3")):
        e = [next(r for r in rows if r["omega"] == w and r["arch"] == arch)["eff_rank"] for w in omegas]
        vr = [next(r for r in rows if r["omega"] == w and r["arch"] == arch)["var_ratio"] for w in omegas]
        a1.plot(omegas, e, "o-", color=col, label=arch)
        a2.plot(omegas, vr, "o-", color=col, label=arch)
    a1.set_xscale("log"); a1.set_xlabel("omega"); a1.set_ylabel("eff-rank(S)")
    a1.set_title("Q1: effective dimension vs omega (-> Wigner)"); a1.legend()
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("omega")
    a2.set_ylabel("var(weak)/var(strong)"); a2.set_title("Q3: zero-variance gap vs omega"); a2.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_phase2_omega_sweep.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
