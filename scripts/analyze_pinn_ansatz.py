"""Mechanism analysis on the REAL ansatz: Slater x PINN Jastrow x backflow (CTNN vs conventional).

The message passing lives in the CTNN BACKFLOW (rho_v_to_e / rho_e_to_v), not a CTNN Jastrow. So:
  - message ablation = zero the CTNN backflow's rho maps -> what coordinated messages in the backflow buy
  - PINN+CTNN-bf vs PINN+conv-bf = the capacity contrast (message-passing vs per-particle backflow)
Per checkpoint: energy error, var(E_L), full-tangent d_eff, backflow displacement rank, kinetic T,
and (CTNN only) the message-ablation kinetic dT. Writes master.csv + prints the contrast.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/analyze_pinn_ansatz.py
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMP = ROOT / "results/analysis/2026-07-11_pinn_ansatz_v3"
REF_E = {1.0: 20.15932, 0.1: 3.55164, 0.01: 0.69036}
RHO = ["rho_v_to_e", "rho_e_to_v"]  # CTNNBackflowNet inter-particle maps
SAMPLE_KW = dict(steps=400, burn_in=800)
# A backflow whose |dx| ~ 0 contributes nothing: any rank/ablation number read off it is meaningless.
# v1 (2026-07-04) died exactly this way (tanh saturation x COM projection), so the analyzer now
# refuses to report on a dead backflow rather than printing a confident number about nothing.
DEAD_DX = 1e-4


@contextlib.contextmanager
def no_bf_messages(system):
    """Zero the CTNN backflow's inter-particle rho maps (single Linear or ModuleList)."""
    bf = system.backflow_net
    saved = []
    for nm in RHO:
        m = getattr(bf, nm, None)
        if m is None:
            continue
        mods = list(m) if isinstance(m, torch.nn.ModuleList) else [m]
        for lin in mods:
            if hasattr(lin, "weight"):
                saved.append((lin, lin.weight.data.clone())); lin.weight.data.zero_()
    try:
        yield
    finally:
        for lin, w in saved:
            lin.weight.data.copy_(w)


def _effrank(M):
    X = M - M.mean(0, keepdims=True); s = np.linalg.svd(X, compute_uv=False); l = s ** 2
    return float((l.sum() ** 2) / (l ** 2).sum()) if l.sum() > 0 else 0.0


def kinetic(system, x):
    xg = x.detach().requires_grad_(True)
    g = torch.autograd.grad(system.log_psi(xg).sum(), xg)[0]
    return float((0.5 * (g ** 2).sum(dim=(1, 2))).double().mean())


def analyze(ckpt, bfarch, seed, omega, dev):
    s = load_system(str(ckpt), device=dev, seed=0)
    x = s.sample(1536, **SAMPLE_KW)
    q = dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256), ref_energy=REF_E.get(round(omega, 4)))
    O = dg.build_O(s.log_psi, x, s.modules(), center=True)         # FULL tangent (Jastrow + backflow)
    deff = float(dg.kernel_spectrum(O.cpu())["effective_rank"])
    with torch.no_grad():
        dx = s.backflow_net(x, spin=s.spin)
    dx_mag = float(dx.norm(dim=-1).mean()) * math.sqrt(s.omega)    # |dx| in oscillator lengths
    alive = dx_mag > DEAD_DX
    bf_rank = _effrank(dx.reshape(x.shape[0], -1).cpu().double().numpy()) if alive else float("nan")
    T_full = kinetic(s, x)
    dT_msg = float("nan")
    if bfarch == "ctnn" and alive:
        with no_bf_messages(s):
            dT_msg = kinetic(s, x) - T_full
    return dict(bfarch=bfarch, seed=seed, omega=omega, error_pct=float(q.get("error_pct") or float("nan")),
                var_EL=float(q["var_EL"]), deff=deff, dx_mag=dx_mag, alive=bool(alive),
                bf_rank=bf_rank, T_full=T_full, dT_msg=dT_msg)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--camp", type=Path, default=DEFAULT_CAMP, help="campaign dir of checkpoints")
    CAMP = ap.parse_args().camp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for d in sorted(CAMP.glob("pinn_*bf_s*_w*")):
        if not (d / "checkpoint.pt").exists():
            continue
        m = re.match(r"pinn_(ctnn|conv)bf_s(\d+)_w([0-9p.]+)$", d.name)
        if not m:
            continue
        bfarch, seed, wtag = m.group(1), int(m.group(2)), float(m.group(3).replace("p", "."))
        try:
            r = analyze(d / "checkpoint.pt", bfarch, seed, wtag, dev); rows.append(r)
            flag = "" if r["alive"] else "   <<< DEAD BACKFLOW — rank/ablation NOT reported"
            print(f"  PINN+{bfarch}-bf s{seed} w{wtag:5}  err={r['error_pct']:+.3f}% var={r['var_EL']:.2e} "
                  f"d_eff={r['deff']:.2f} |dx|={r['dx_mag']:.4f} BFrank={r['bf_rank']:.1f} "
                  f"dT_msg={r['dT_msg']:+.3f}{flag}")
        except Exception as e:
            print(f"  {d.name}: ERR {e!r}")
    if not rows:
        print("no checkpoints yet"); return
    with open(CAMP / "master.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(CAMP / "master.json", "w"), indent=2)
    # contrast: CTNN vs conv backflow, seed-averaged, by omega
    print("\n[contrast] PINN + CTNN-backflow vs PINN + conventional-backflow (seed-avg):")
    print(f"{'w':>6} | {'err% ctnn/conv':>16} | {'var ctnn/conv':>18} | {'BFrank ctnn/conv':>16} | {'dT_msg(ctnn)':>12}")
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['bfarch'], r['omega'])].append(r)
    def m(bfarch, w, k):
        rs = by.get((bfarch, w), [])
        vs = [x[k] for x in rs if x[k] == x[k]]
        return sum(vs) / len(vs) if vs else float('nan')
    for w in sorted({r['omega'] for r in rows}, reverse=True):
        print(f"{w:>6} | {m('ctnn',w,'error_pct'):+.3f}/{m('conv',w,'error_pct'):+.3f} "
              f"| {m('ctnn',w,'var_EL'):.2e}/{m('conv',w,'var_EL'):.2e} "
              f"| {m('ctnn',w,'bf_rank'):.1f}/{m('conv',w,'bf_rank'):.1f} | {m('ctnn',w,'dT_msg'):+.3f}")
    print(f"-> {CAMP}/master.csv")


if __name__ == "__main__":
    main()
