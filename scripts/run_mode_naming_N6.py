"""N=6 mode-naming by operator decomposition (step 3 of the Q1 program).

At N=2 the tangent modes were named exactly (breathing / correlation-hole / relative excitation) against
the analytic solution. There is no exact N=6 solution, so we name the N=6 tangent modes against a basis
of PHYSICAL OPERATOR generators evaluated on the probe (functions of the configuration):

  monopole/breathing : sum_i r_i^2         quartic : sum_i r_i^4
  quadrupole         : sum_i (x_i^2-y_i^2), sum_i 2 x_i y_i
  correlation-hole   : sum_{i<j} exp(-r_ij^2 / 2 sigma^2)  (short-range, two sigma)
  pair-Coulomb       : sum_{i<j} 1/r_ij     pair-linear : sum_{i<j} r_ij

We project the network's top NTK eigenfunctions (the effective tangent directions) onto the span of
these operators: how much of each tangent mode is a named physical collective mode, and which operator
dominates the leading mode. Runs on existing checkpoints (no training); CTNN and DeepSet, per omega.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_mode_naming_N6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-07-02_mode_naming_N6"
KTOP = 6
N_PROBE = 1536
SAMPLE_KW = dict(steps=400, burn_in=800)

# good analysis-grade checkpoints per omega
CKPTS = {
    1.0:  {"CTNN": "2026-06-15_N6_w1_ctnn_big_bf_acc",    "DeepSet": "2026-06-15_N6_w1_deepset_big_bf_casc"},
    0.01: {"CTNN": "2026-07-02_N6_w001_ctnn_s0",          "DeepSet": "2026-07-02_N6_w001_deepset_s0"},
}


def operators(x: torch.Tensor, omega: float) -> tuple[np.ndarray, list[str]]:
    """Physical operator generators on configs x:(B,N,2). Returns (B, n_ops) matrix + names."""
    B, N, _ = x.shape
    ell = 1.0 / np.sqrt(omega)
    xf = x.double()
    r2 = (xf ** 2).sum(-1)                                  # (B,N)
    cols, names = [], []
    cols.append(r2.sum(-1));                    names.append("monopole_r2")
    cols.append((r2 ** 2).sum(-1));             names.append("quartic_r4")
    cols.append((xf[..., 0] ** 2 - xf[..., 1] ** 2).sum(-1)); names.append("quadrupole_x2y2")
    cols.append((2 * xf[..., 0] * xf[..., 1]).sum(-1));       names.append("quadrupole_xy")
    ii, jj = torch.triu_indices(N, N, offset=1)
    rij = (xf[:, ii, :] - xf[:, jj, :]).norm(dim=-1).clamp_min(1e-6)  # (B, npairs)
    for s in (0.5, 1.0):
        cols.append(torch.exp(-(rij ** 2) / (2 * (s * ell) ** 2)).sum(-1)); names.append(f"hole_sig{s}")
    cols.append((1.0 / rij).sum(-1));           names.append("pair_coulomb")
    cols.append(rij.sum(-1));                   names.append("pair_linear")
    M = torch.stack(cols, dim=1).cpu().numpy()             # (B, n_ops)
    return M, names


def top_eigvecs(system, x, k):
    O = dg.build_O(system.log_psi, x, [system.f_net], center=True)
    K = (O.double() @ O.double().t()).cpu() / O.shape[0]
    ev, V = torch.linalg.eigh(K)
    order = torch.argsort(ev, descending=True)
    return V[:, order[:k]].numpy()


def _center_unit(M):
    M = M - M.mean(0, keepdims=True)
    return M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-30)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    result = {}
    for omega, pair in CKPTS.items():
        result[str(omega)] = {}
        for arch, d in pair.items():
            s = load_system(str(ROOT / "results/analysis" / d / "checkpoint.pt"), device=dev, seed=0)
            x = s.sample(N_PROBE, **SAMPLE_KW)
            M, names = operators(x, omega)
            Mu = _center_unit(M)
            Q, _ = np.linalg.qr(Mu)                         # orthonormal basis of the operator span
            U = top_eigvecs(s, x, KTOP)                     # (B, k) net tangent modes
            cap_mode_in_ops = [float((Q.T @ U[:, a]) @ (Q.T @ U[:, a])) for a in range(KTOP)]
            # name the leading mode: least-squares onto the operators
            coef, *_ = np.linalg.lstsq(Mu, U[:, 0], rcond=None)
            resid = U[:, 0] - Mu @ coef
            r2_top = float(1 - (resid @ resid) / (U[:, 0] @ U[:, 0]))
            # per-operator |correlation| with the leading mode (which operator it is)
            corr = {names[j]: float(abs(Mu[:, j] @ U[:, 0])) for j in range(len(names))}
            top_op = max(corr, key=corr.get)
            result[str(omega)][arch] = dict(cap_mode_in_ops=cap_mode_in_ops, r2_leading=r2_top,
                                            leading_operator=top_op, corr=corr)
            print(f"[name] w={omega:<5} {arch:8} leading mode captured by operators R^2={r2_top:.2f} "
                  f"(top op: {top_op} |corr|={corr[top_op]:.2f}); "
                  f"per-mode capture={[round(c,2) for c in cap_mode_in_ops]}")
    json.dump(result, open(OUT / "summary.json", "w"), indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
