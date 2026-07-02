"""Phase 1 / Q3 — the Laplacian / zero-variance anatomy of the two estimators.

On converged checkpoints (no training), compare the strong-form local energy
  E_L  = -1/2 (lap logPsi + |grad logPsi|^2) + V      (needs the Laplacian; zero-variance)
against the weak-form Rayleigh integrand
  e_w  =  1/2 |grad logPsi|^2 + V                      (no Laplacian; first derivatives only)
under each model's |Psi|^2.

Two facts to demonstrate:
  (1) Integration by parts: <E_L> = <e_w>  (the Laplacian term has zero mean) -> removing it still
      trains, because the variational energy is preserved in the mean.
  (2) Zero-variance: var(E_L) -> 0 at an eigenstate, var(e_w) does NOT. So removing the Laplacian
      (the weak form) sacrifices the zero-variance property -- the cost of "training without it".

Run (HPC): source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
  PYTHONUNBUFFERED=1 python3 -u scripts/run_estimator_variance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-06-22_estimator_variance"

MODELS = [
    ("N2_ctnn_exact", 2, 1.0, "2026-06-15_N2_w1_ctnn_acc"),
    ("N6_ctnn", 6, 1.0, "2026-06-15_2x2_N6w1_ctnn_sr"),
    ("N6_deepset", 6, 1.0, "2026-06-15_2x2_N6w1_deepset_sr"),
]
N_SAMPLES = 2048


def _stats(R: torch.Tensor) -> tuple[float, float]:
    R = R.detach().double()
    return float(R.mean()), float(R.var(unbiased=True))


def _weak_chunked(system, x: torch.Tensor, *, chunk: int = 256) -> torch.Tensor:
    """Weak-form Rayleigh integrand 1/2|grad logPsi|^2 + V, values only (no param graph)."""
    outs = []
    for st in range(0, x.shape[0], chunk):
        r = dg.residual_local_energy(system, x[st : st + chunk], form="weak").detach()
        outs.append(r)
    return torch.cat(outs, dim=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for lab, N, w, d in MODELS:
        s = load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
        x = s.sample(N_SAMPLES, steps=400, burn_in=800)
        # strong = E_L via the chunked, detached exact-Laplacian path (memory-safe)
        m_s, v_s = _stats(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256))
        m_w, v_w = _stats(_weak_chunked(s, x))
        row = dict(model=lab, N=N, omega=w,
                   mean_strong=m_s, var_strong=v_s, mean_weak=m_w, var_weak=v_w,
                   mean_gap=m_w - m_s, var_ratio_weak_over_strong=(v_w / v_s if v_s > 0 else float("inf")))
        rows.append(row)
        print(f"  {lab:16} <E_L>={m_s:.5f} var(E_L)={v_s:.3e} | <e_w>={m_w:.5f} var(e_w)={v_w:.3e} "
              f"| mean-gap={m_w - m_s:+.2e} var-ratio(w/s)={row['var_ratio_weak_over_strong']:.1f}x")
    json.dump(rows, open(OUT / "estimator_variance.json", "w"), indent=2)
    print(f"[estimator-var] -> {OUT}/estimator_variance.json")
    print("Interpretation: mean-gap ~ 0 confirms integration-by-parts (weak trains in the mean); "
          "var-ratio >> 1 (esp. as var(E_L)->0) is the zero-variance lost by dropping the Laplacian.")


if __name__ == "__main__":
    main()
