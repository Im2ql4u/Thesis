"""A4 — Name the manifold: project the N=2 network's tangent (Jastrow) modes onto the exact
physical response and excitation modes.

At N=2, omega=1 the ansatz is exact (overlap^2 = 1), so we can ask what the ~1-dimensional
variational tangent space actually IS, in physical terms. The Jastrow tangent modes are functions
of the pair distance r12; we compare them against the derivatives of the EXACT Jastrow J*(r12) and
the exact relative excitations, all in the same function space (functions of r12 on a common probe):

  d_omega J*   breathing / trap-scale response      (finite difference in omega)
  d_lambda J*  correlation-hole response            (finite difference in Coulomb strength lambda)
  phi_n = u_n/u_0   n-th relative s-wave excitation  (exact excited states of the relative H)

Pre-registered: the tangent space is "named" if the network's leading mode u_1 (and its top-2
subspace) is captured > 0.8 by span{d_omega, d_lambda, phi_1}.

Run (HPC): source /etc/profile.d/lmod.sh; module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u scripts/run_response_projection.py
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
from analysis.reference import TwoElectronExact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/analysis/2026-07-02_response_projection"
CKPT = ROOT / "results/analysis/2026-06-14_N2_w1_ctnn_vcycle_polished/checkpoint.pt"
OMEGA, LAM = 1.0, 1.0
DELTA = 0.02          # finite-difference step for d/domega, d/dlambda
B = 1536             # probe points
KTOP = 6             # network tangent modes to inspect
SAMPLE_KW = dict(steps=400, burn_in=800)


def _pair_dist(x: torch.Tensor) -> np.ndarray:
    return (x[:, 0, :] - x[:, 1, :]).norm(dim=-1).cpu().double().numpy()


def _center(v: np.ndarray) -> np.ndarray:
    return v - v.mean()


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _net_tangent_modes(system, x: torch.Tensor, k: int) -> np.ndarray:
    """Top-k NTK eigenvectors of the Jastrow tangent space, as functions on the probe (B,k)."""
    O = dg.build_O(system.log_psi, x, [system.f_net], center=True)  # (B,P)
    K = (O.double() @ O.double().t()) / O.shape[0]                  # (B,B)
    evals, evecs = torch.linalg.eigh(K)
    order = torch.argsort(evals, descending=True)
    U = evecs[:, order[:k]].cpu().double().numpy()                 # (B,k), orthonormal columns
    return U, evals[order].cpu().double().numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- network + GS-quality gate (overlap with exact) ---
    net = load_system(str(CKPT), device=dev, seed=0)
    x = net.sample(B, **SAMPLE_KW)
    r12 = _pair_dist(x)

    base = TwoElectronExact(omega=OMEGA, lam=LAM, n_states=3)
    # overlap^2 on |Psi_net|^2 samples: <Psi_ex/Psi_net>^2 / <(Psi_ex/Psi_net)^2>
    lp_net = net.log_psi(x).detach().cpu().double().numpy()
    lp_ex = base.log_psi(x.cpu().double().numpy())
    d = lp_ex - lp_net
    d -= d.mean()
    w = np.exp(d)
    overlap2 = (w.mean() ** 2) / (w ** 2).mean()

    # --- physical modes on the probe (functions of r12), centred + unit-normalised ---
    ref_wp = TwoElectronExact(omega=OMEGA + DELTA, lam=LAM)
    ref_wm = TwoElectronExact(omega=OMEGA - DELTA, lam=LAM)
    ref_lp = TwoElectronExact(omega=OMEGA, lam=LAM + DELTA)
    ref_lm = TwoElectronExact(omega=OMEGA, lam=LAM - DELTA)
    modes = {
        "d_omega":  (ref_wp.jastrow_log(r12) - ref_wm.jastrow_log(r12)) / (2 * DELTA),
        "d_lambda": (ref_lp.jastrow_log(r12) - ref_lm.jastrow_log(r12)) / (2 * DELTA),
        "phi_1":    base.relative_excitation_ratio(r12, 1),
        "phi_2":    base.relative_excitation_ratio(r12, 2),
    }
    names = list(modes)
    M = np.stack([_unit(_center(modes[m])) for m in names], axis=1)  # (B,4)
    Q, _ = np.linalg.qr(M)  # orthonormal basis of the physical-mode span

    # --- network tangent modes ---
    U, lam_spec = _net_tangent_modes(net, x, KTOP)  # (B,k)

    # (1) fraction of each network mode captured by the physical-mode span
    cap_net_in_phys = [float((Q.T @ U[:, a]) @ (Q.T @ U[:, a])) for a in range(KTOP)]
    # (2) fraction of each physical mode captured by the network's top-KTOP subspace
    cap_phys_in_net = {names[j]: float((U.T @ M[:, j]) @ (U.T @ M[:, j])) for j in range(len(names))}
    # (3) named decomposition of the leading network mode u_1: least squares onto the physical modes
    u1 = U[:, 0]
    coef, *_ = np.linalg.lstsq(M, u1, rcond=None)
    resid = u1 - M @ coef
    r2_u1 = float(1.0 - (resid @ resid) / (u1 @ u1))
    contrib = (M * coef) .sum(0)  # not used; report normalised |coef| instead
    coef_norm = {names[j]: float(coef[j]) for j in range(len(names))}
    # capture of u_1 by the reduced span {d_omega, d_lambda, phi_1} (the pre-registered set)
    M3 = M[:, [0, 1, 2]]
    Q3, _ = np.linalg.qr(M3)
    cap_u1_in_3 = float((Q3.T @ u1) @ (Q3.T @ u1))
    cap_top2_in_3 = float(np.mean([(Q3.T @ U[:, a]) @ (Q3.T @ U[:, a]) for a in range(2)]))

    out = {
        "checkpoint": str(CKPT), "omega": OMEGA, "lam": LAM, "B": B, "delta": DELTA,
        "gs_gate": {"overlap2_on_net_samples": overlap2, "net_energy_ref": base.energy,
                    "energies_rel_all": base.energies_rel_all.tolist()},
        "cap_net_in_phys_span4": cap_net_in_phys,
        "cap_phys_in_net_top6": cap_phys_in_net,
        "u1_named": {"R2_on_4modes": r2_u1, "coef": coef_norm,
                     "cap_u1_in_{domega,dlambda,phi1}": cap_u1_in_3,
                     "cap_top2_in_{domega,dlambda,phi1}": cap_top2_in_3},
        "net_tangent_eigenvalues_top6": lam_spec[:6].tolist(),
    }
    json.dump(out, open(OUT / "summary.json", "w"), indent=2)

    print(f"[A4] overlap^2(net,exact) = {overlap2:.5f}  (E_ref={base.energy:.4f}, "
          f"E_rel excited = {np.round(base.energies_rel_all,3).tolist()})")
    print(f"[A4] network leading mode u_1 captured by span{{d_omega,d_lambda,phi_1}} = {cap_u1_in_3:.3f}")
    print(f"[A4] top-2 network subspace captured by that 3-mode span (mean) = {cap_top2_in_3:.3f}")
    print(f"[A4] u_1 named: R^2 on 4 physical modes = {r2_u1:.3f}; coefs = "
          + ", ".join(f"{k}={v:+.2f}" for k, v in coef_norm.items()))
    print(f"[A4] each network mode captured by physical span (top-{KTOP}): "
          + ", ".join(f"{c:.2f}" for c in cap_net_in_phys))
    print(f"[A4] each physical mode captured by network top-{KTOP}: "
          + ", ".join(f"{k}={v:.2f}" for k, v in cap_phys_in_net.items()))
    _figure(r12, modes, U, names)
    print(f"-> {OUT}")


def _figure(r12, modes, U, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(r12)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in names:
        a1.plot(r12[order], _unit(_center(modes[m]))[order], label=m, lw=1.4)
    a1.set_xlabel("pair distance r12"); a1.set_ylabel("mode (centred, unit)")
    a1.set_title("A4: exact physical response/excitation modes"); a1.legend(fontsize=8)
    for a in range(min(3, U.shape[1])):
        a2.plot(r12[order], _unit(U[:, a])[order], label=f"net mode {a+1}", lw=1.4)
    a2.set_xlabel("pair distance r12"); a2.set_ylabel("net tangent mode (unit)")
    a2.set_title("A4: network's leading tangent modes"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_response_projection.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
