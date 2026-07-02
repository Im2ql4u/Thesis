"""Q1 scaling (step 4, tractable form): the CTNN-vs-DeepSet effective-dimension gap at
INITIALISATION vs N.

Trained-state d_eff at N>=12 is blocked by (a) from-scratch training instability and (b) exact-Laplacian
OOM -- a separate infrastructure sub-project. But the Phase-0 finding was that the architectural gap is
already present at initialisation (CTNN ~1.1 vs DeepSet ~3.9 at N=6). We can therefore probe the
*architectural inductive bias* and its N-scaling cheaply: build the untrained f_net tangent space and
measure its fair d_eff. This needs only first derivatives of log|Psi| (no Laplacian -> no OOM, no
training). It is the inductive-bias dimension, NOT the trained-solution dimension (that remains the
deferred training sub-project) -- reported as such.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_dinit_scaling.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import System  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-07-02_dinit_scaling"

ARCH_KW = {  # base arch name + exact _big kwargs from run_phase_analysis.ARCH_KWARGS
    "ctnn": ("ctnn_vcycle", dict(node_hidden=32, edge_hidden=32, bottleneck_hidden=16,
             n_down=2, n_up=2, msg_layers=2, node_layers=2, readout_hidden=64, readout_layers=3, act="silu")),
    "deepset": ("deepset", dict(pair_hidden=64, pair_layers=4, pair_out=32,
                readout_hidden=64, readout_layers=3, act="silu")),
}
BF_KW = dict(msg_hidden=64, msg_layers=2, hidden=64, layers=3, act="silu",
             out_bound="tanh", bf_scale_init=0.05, zero_init_last=True)
NS = [6, 12, 20]
OMEGA = 1.0
SEEDS = [0, 1, 2]
B = 1024


def d_eff_at_init(N, arch_name, arch_kwargs, seed, dev):
    s = System(N=N, omega=OMEGA, d=2, arch=arch_name, arch_kwargs=arch_kwargs,
               use_backflow=True, backflow_kwargs=BF_KW, device=dev, seed=seed)
    s.eval()
    ell = 1.0 / np.sqrt(OMEGA)
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    probe = (torch.randn(B, N, 2, generator=g) * ell).to(dev, dtype=s.dtype)  # common Gaussian probe
    O = dg.build_O(s.log_psi, probe, [s.f_net], center=True, chunk_size=128)
    sp = dg.kernel_spectrum(O.cpu())  # SVD on CPU (safe at large N)
    n_par = int(sum(p.numel() for p in s.f_net.parameters()))
    return float(sp["effective_rank"]), n_par


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for N in NS:
        for arch, (aname, akw) in ARCH_KW.items():
            vals = []
            for seed in SEEDS:
                try:
                    de, npar = d_eff_at_init(N, aname, akw, seed, dev)
                    vals.append(de)
                except torch.cuda.OutOfMemoryError:
                    print(f"[oom] N={N} {arch} seed{seed} — skipping"); torch.cuda.empty_cache(); continue
            if not vals:
                continue
            rows.append(dict(N=N, arch=arch, n_params=npar,
                             deff_mean=float(np.mean(vals)), deff_std=float(np.std(vals)), n_seeds=len(vals)))
            print(f"  N={N:<3} {arch:8} d_eff(init) = {np.mean(vals):.2f} +/- {np.std(vals):.2f}  "
                  f"({[round(v,2) for v in vals]})")
        c = next((r for r in rows if r["N"] == N and r["arch"] == "ctnn"), None)
        d = next((r for r in rows if r["N"] == N and r["arch"] == "deepset"), None)
        if c and d:
            print(f"  --> N={N}: gap DeepSet/CTNN = {d['deff_mean']/c['deff_mean']:.2f}x")
    with open(OUT / "dinit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(OUT / "summary.json", "w"), indent=2)
    _figure(rows)
    print(f"-> {OUT}")


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, col in (("ctnn", "C0"), ("deepset", "C3")):
        rs = sorted([r for r in rows if r["arch"] == arch], key=lambda r: r["N"])
        if rs:
            ax.errorbar([r["N"] for r in rs], [r["deff_mean"] for r in rs],
                        yerr=[r["deff_std"] for r in rs], fmt="o-", color=col, capsize=4, label=arch.upper())
    ax.set_xlabel("N"); ax.set_ylabel("d_eff(S) at initialisation")
    ax.set_title("Architectural inductive-bias dimension vs N (omega=1, init)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_dinit_scaling.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
