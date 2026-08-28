"""Correlation-structure probe: is the DeepSet's intrinsic limitation detectable at HIGH omega?

The mode-naming leading-R2 metric is tied at high omega (DeepSet ~ CTNN), so it cannot see
the limitation there. This probe uses correlation-SPECIFIC diagnostics that should expose a
deficiency present at all omega, disguised at high omega where correlation is a small energy
fraction:

  (1) hole_capture   -- fraction of the correlation-hole response direction lying in the span
                        of the network's top-k tangent modes; "pure" first removes the
                        single-particle part, isolating genuine correlation.
  (2) nonphys_mass   -- eigenvalue-weighted fraction of the top-k tangent variance OUTSIDE the
                        physical-operator span (physical "filler").
  (3) causal ablation-- for the CTNN, zero the copresheaf transport maps and recompute (1)-(2):
                        does physicality degrade toward the DeepSet?
  (4) backflow_rank  -- participation ratio of the backflow displacement (unify Q1a/Q1b).

Well-trained checkpoints only (energy gate reused from run_mode_naming_scan).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from run_mode_naming_N6 import operators, _center_unit, KTOP  # noqa: E402
from run_mode_naming_scan import CKPTS, REF, GATE_PCT, GATE_RELVAR, AN, N_PROBE, SAMPLE_KW  # noqa: E402

OUT = ROOT / "results/analysis/2026-08-28_correlation_probe"
SINGLE_PARTICLE = ["monopole_r2", "quartic_r4", "quadrupole_x2y2", "quadrupole_xy"]


def top_modes(system, x, k):
    O = dg.build_O(system.log_psi, x, [system.f_net], center=True)
    K = (O.double() @ O.double().t()).cpu() / O.shape[0]
    ev, V = torch.linalg.eigh(K)
    order = torch.argsort(ev, descending=True)
    return V[:, order[:k]].numpy(), ev[order[:k]].clamp_min(0).numpy()


def _capture(direction, U):
    d = direction / (np.linalg.norm(direction) + 1e-30)
    return float(np.sum((U.T @ d) ** 2))


def probe_physicality(U, evals, M, names):
    Mu = _center_unit(M)
    Q, _ = np.linalg.qr(Mu)
    hole = Mu[:, names.index("hole_sig1.0")]
    hole_capture = _capture(hole, U)
    sp_idx = [names.index(n) for n in SINGLE_PARTICLE]
    Qsp, _ = np.linalg.qr(Mu[:, sp_idx])
    hole_pure = hole - Qsp @ (Qsp.T @ hole)
    hole_capture_pure = _capture(hole_pure, U) if np.linalg.norm(hole_pure) > 1e-8 else float("nan")
    phys = np.array([float((Q.T @ U[:, a]) @ (Q.T @ U[:, a])) for a in range(U.shape[1])])
    w = evals / (evals.sum() + 1e-30)
    nonphys_mass = float(np.sum(w * (1.0 - phys)))
    return dict(r2_leading=float(phys[0]), hole_capture=hole_capture,
                hole_capture_pure=hole_capture_pure, nonphys_mass=nonphys_mass)


def ablate_transport(system):
    f = system.f_net
    found = False
    with torch.no_grad():
        for nm in ["rho_v_to_e_down", "rho_e_to_v_down", "rho_v_to_e_up", "rho_e_to_v_up"]:
            ml = getattr(f, nm, None)
            if ml is None:
                continue
            for lin in ml:
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
                found = True
    return found


def backflow_rank(system, x):
    if system.backflow_net is None:
        return float("nan")
    with torch.no_grad():
        dxk = system.backflow_net(x, spin=system.spin)
    M = dxk.reshape(x.shape[0], -1).cpu().double().numpy()
    M = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    lam = s ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum()) if lam.sum() > 0 else 0.0


def analyse(ckpt_dir, N, omega, arch, dev):
    path = AN / ckpt_dir / "checkpoint.pt"
    if not path.exists():
        return None
    s = load_system(str(path), device=dev, seed=0)
    x = s.sample(N_PROBE, **SAMPLE_KW)
    ref = REF.get((N, round(omega, 3)))
    EL = dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256)
    q = dg.gs_quality(EL, ref_energy=ref)
    if ref is not None:
        trained_ok = abs(q.get("error_pct", 1e9)) <= GATE_PCT
    else:
        trained_ok = q["var_EL"] / max(q["E_mean"] ** 2, 1e-12) <= GATE_RELVAR
    M, names = operators(x, omega)
    U, evals = top_modes(s, x, KTOP)
    out = probe_physicality(U, evals, M, names)
    out["bf_rank"] = backflow_rank(s, x)
    out["E_mean"] = q["E_mean"]
    out["error_pct"] = q.get("error_pct", float("nan"))
    out["trained_ok"] = bool(trained_ok)
    if arch == "CTNN" and ablate_transport(s):
        Ua, evala = top_modes(s, x, KTOP)
        pa = probe_physicality(Ua, evala, M, names)
        out["r2_leading_abl"] = pa["r2_leading"]
        out["hole_capture_abl"] = pa["hole_capture"]
        out["hole_capture_pure_abl"] = pa["hole_capture_pure"]
        out["nonphys_mass_abl"] = pa["nonphys_mass"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for (N, w, arch, sd, d) in CKPTS:
        try:
            r = analyse(d, N, w, arch, dev)
        except Exception as e:
            print(f"[FAIL] N={N} w={w} {arch} s{sd} {d}: {e!r}", flush=True)
            continue
        if r is None:
            print(f"[MISS] N={N} w={w} {arch} s{sd} {d}", flush=True)
            continue
        rows.append(dict(N=N, omega=w, arch=arch, seed=sd, **r))
        tag = "OK " if r["trained_ok"] else "REJECT"
        abl = f" abl_hole={r.get('hole_capture_pure_abl', float('nan')):.3f}" if arch == "CTNN" else ""
        print(f"[{tag}] N={N:2d} w={w:<5} {arch:8} s{sd}  err%={r['error_pct']:+.3f}  "
              f"hole_pure={r['hole_capture_pure']:.3f}  nonphys={r['nonphys_mass']:.3f}  "
              f"bf_rank={r['bf_rank']:.1f}{abl}", flush=True)
    keys = sorted({k for row in rows for k in row})
    with open(OUT / "probe.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    json.dump(rows, open(OUT / "probe.json", "w"), indent=2)
    print(f"\n-> {OUT}  ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
