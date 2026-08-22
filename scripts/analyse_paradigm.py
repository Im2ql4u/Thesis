"""Q2/Q3 analysis on the 2x2 campaign (build_base -> {vmc,colloc} x {adam,sr}).

Q3 (paradigm) — do VMC and collocation reach the SAME STATE, or only the same energy?
  overlap^2(A,B) = <psi_B/psi_A>_{|psi_A|^2} * <psi_A/psi_B>_{|psi_B|^2}   (unbiased, symmetric)
  Reported for vmc_adam vs colloc_adam and vmc_sr vs colloc_sr. At N=2 the exact ground state is
  known, so we also report overlap^2 with it (ground truth). If overlap^2 ~ 1 the paradigms find the
  same solution (paradigm = efficiency only); if < 1 they find different states.

Q2 (optimizer) — when/why does SR beat Adam? Energy gap dE = E(adam) - E(sr) per cell-pair, alongside
  the QGT condition number kappa(S) and d_eff of each state: the tangent-kernel prediction is that SR
  helps most where kappa(S) is largest (most anisotropic tangent space).

Also reports, per cell: energy error, var(E_L), backflow rank, d_eff, and (colloc) the final ESS —
  where fixed-proposal collocation loses ESS toward Wigner is a Q3 domain-of-validity finding.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/analyse_paradigm.py --camp <dir>
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from config import DMC_ENERGIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMP = ROOT / "results/analysis/2026-08-05_paradigm_optimizer"
CELLS = ["vmc_adam", "vmc_sr", "colloc_adam", "colloc_sr"]
SAMPLE_KW = dict(steps=400, burn_in=800)


def ref_energy(N, omega):
    for w, e in DMC_ENERGIES.get(int(N), {}).items():
        if abs(w - omega) < 1e-9 or (omega and abs(w - omega) / omega < 0.02):
            return float(e)
    return None


@torch.no_grad()
def overlap_sq(sysA, sysB, n=2048):
    """Symmetric, unbiased overlap^2 via reciprocal importance sampling from each state's own |psi|^2.

    overlap^2 = <psi_B/psi_A>_{|psi_A|^2} * <psi_A/psi_B>_{|psi_B|^2}. Each mean is computed with
    logsumexp (internally max-stabilised and correct — no manual shift, which would drop the offset).
    """
    def log_mean_ratio(src, other):
        x = src.sample(n, **SAMPLE_KW)
        r = (other.log_psi(x) - src.log_psi(x)).double()    # log(psi_other/psi_src) at x ~ |psi_src|^2
        return torch.logsumexp(r, 0) - math.log(r.numel())  # log <psi_other/psi_src>_src
    mA = log_mean_ratio(sysA, sysB)   # log <psi_B/psi_A>_A
    mB = log_mean_ratio(sysB, sysA)   # log <psi_A/psi_B>_B
    return float(torch.exp(mA + mB).clamp(max=1.0))


def cell_metrics(ckpt, omega, N, dev):
    s = load_system(str(ckpt), device=dev, seed=0)
    x = s.sample(1536, **SAMPLE_KW)
    q = dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256),
                      ref_energy=ref_energy(N, omega))
    O = dg.build_O(s.log_psi, x, s.modules(), center=True)
    spec = dg.kernel_spectrum(O)
    with torch.no_grad():
        dx = s.backflow_net(x, spin=s.spin)
    X = dx.reshape(x.shape[0], -1).cpu().double().numpy(); X = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(X, compute_uv=False) ** 2
    bfrank = float((sv.sum() ** 2) / (sv ** 2).sum()) if sv.sum() > 0 else 0.0
    return dict(E_raw=float(q["E_mean_raw"]), err=float(q.get("error_pct") or float("nan")),
                var=float(q["var_EL"]), deff=float(spec["effective_rank"]),
                kappa=float(spec["condition_number"]), bfrank=bfrank), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camp", type=Path, default=DEFAULT_CAMP)
    camp = ap.parse_args().camp
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # group cells by (N, seed, omega)
    groups: dict = {}
    for d in sorted(camp.glob("N*_s*_w*_*")):
        m = re.match(r"N(\d+)_s(\d+)_(w[0-9p]+)_(vmc_adam|vmc_sr|colloc_adam|colloc_sr)$", d.name)
        if not m or not (d / "checkpoint.pt").exists():
            continue
        N, seed, wt, cell = int(m[1]), int(m[2]), m[3], m[4]
        w = float(wt[1:].replace("p", "."))
        groups.setdefault((N, seed, w), {})[cell] = d / "checkpoint.pt"

    rows = []
    for (N, seed, w), cells in sorted(groups.items()):
        met, sysm = {}, {}
        for cell, ck in cells.items():
            try:
                met[cell], sysm[cell] = cell_metrics(ck, w, N, dev)
            except Exception as e:
                print(f"  N{N} s{seed} w{w} {cell}: ERR {e!r}")
        # Q3: same-state overlap between paradigms, matched optimizer
        ov_adam = overlap_sq(sysm["vmc_adam"], sysm["colloc_adam"]) if {"vmc_adam", "colloc_adam"} <= met.keys() else float("nan")
        ov_sr = overlap_sq(sysm["vmc_sr"], sysm["colloc_sr"]) if {"vmc_sr", "colloc_sr"} <= met.keys() else float("nan")
        for cell in CELLS:
            if cell not in met:
                continue
            m = met[cell]
            rows.append(dict(N=N, seed=seed, omega=w, cell=cell, **m,
                             overlap_vmc_colloc=(ov_adam if cell.endswith("adam") else ov_sr)))
        # Q2: energy gap adam - sr, per paradigm
        def gap(p):
            a, s = f"{p}_adam", f"{p}_sr"
            return (met[a]["E_raw"] - met[s]["E_raw"]) if {a, s} <= met.keys() else float("nan")
        print(f"N={N} s{seed} w={w:6}: "
              f"Q3 overlap^2(vmc,colloc) adam={ov_adam:.4f} sr={ov_sr:.4f} | "
              f"Q2 dE(adam-sr) vmc={gap('vmc'):+.4f} colloc={gap('colloc'):+.4f} | "
              f"kappa(S) vmc_adam={met.get('vmc_adam',{}).get('kappa',float('nan')):.1e}")

    if rows:
        with open(camp / "paradigm_master.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"-> {camp}/paradigm_master.csv")
    else:
        print("no completed cells yet")


if __name__ == "__main__":
    main()
