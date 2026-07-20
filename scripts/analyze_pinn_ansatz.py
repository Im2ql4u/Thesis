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
from config import DMC_ENERGIES  # noqa: E402  (thesis Table 5.2 references, keyed by N then omega)


def ref_energy(N: int, omega: float) -> float | None:
    tab = DMC_ENERGIES.get(int(N), {})
    for w, e in tab.items():
        if abs(w - omega) < 1e-9 or (omega and abs(w - omega) / omega < 0.02):
            return float(e)
    return None
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


@contextlib.contextmanager
def no_backflow(system):
    """Remove the backflow entirely (Delta_x = 0) — how much energy does it actually buy?

    |dx| alone cannot answer this: a small displacement may still be doing essential work, and a
    large one may be redundant with the Jastrow. The energy cost of deleting it is the honest test.
    """
    bf = system.backflow_net
    system.backflow_net = None
    try:
        yield
    finally:
        system.backflow_net = bf


def _effrank(M):
    X = M - M.mean(0, keepdims=True); s = np.linalg.svd(X, compute_uv=False); l = s ** 2
    return float((l.sum() ** 2) / (l ** 2).sum()) if l.sum() > 0 else 0.0


def kinetic(system, x):
    xg = x.detach().requires_grad_(True)
    g = torch.autograd.grad(system.log_psi(xg).sum(), xg)[0]
    return float((0.5 * (g ** 2).sum(dim=(1, 2))).double().mean())


def potential(system, x):
    """<V> = trap + Coulomb. Splitting E into T and V is what decides whether message passing
    buys kinetic energy (its high-omega story) or something else at Wigner."""
    xd = x.detach().double()
    trap = 0.5 * (system.omega ** 2) * (xd ** 2).sum(dim=(1, 2))
    r = torch.cdist(xd, xd)
    iu = torch.triu_indices(system.N, system.N, offset=1)
    coul = (1.0 / r[:, iu[0], iu[1]].clamp_min(1e-12)).sum(dim=1)
    return float(trap.mean()), float(coul.mean())


def energy_TV(system, n=1024):
    """Resample from THIS ansatz and return (err-free) E, T, V_trap, V_coul.

    Resampling matters: evaluating an ablated wavefunction on configurations drawn from the
    un-ablated |Psi|^2 is not a variational estimate — it produced energies BELOW the exact
    ground state (-1.6%), which is impossible. Each ablation must sample its own distribution.
    """
    x = system.sample(n, **SAMPLE_KW)
    E = float(dg.gs_quality(dg.local_energy(system.log_psi, x, system.omega,
                                            system.params, chunk=256))["E_mean_raw"])
    T = kinetic(system, x)
    Vt, Vc = potential(system, x)
    return E, T, Vt, Vc


def analyze(ckpt, bfarch, seed, omega, dev):
    s = load_system(str(ckpt), device=dev, seed=0)
    x = s.sample(1536, **SAMPLE_KW)
    q = dg.gs_quality(dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256), ref_energy=ref_energy(int(s.N), omega))
    O = dg.build_O(s.log_psi, x, s.modules(), center=True)         # FULL tangent (Jastrow + backflow)
    deff = float(dg.kernel_spectrum(O.cpu())["effective_rank"])
    with torch.no_grad():
        dx = s.backflow_net(x, spin=s.spin)
    dx_mag = float(dx.norm(dim=-1).mean()) * math.sqrt(s.omega)    # |dx| in oscillator lengths
    alive = dx_mag > DEAD_DX
    bf_rank = _effrank(dx.reshape(x.shape[0], -1).cpu().double().numpy()) if alive else float("nan")
    # ---- ablations, each RESAMPLED from its own |Psi|^2 and split into T and V ----
    E_f, T_f, Vt_f, Vc_f = energy_TV(s)
    with no_backflow(s):
        E_nb, T_nb, Vt_nb, Vc_nb = energy_TV(s)
    dE_bf = E_nb - E_f                    # >0 => the backflow lowers the energy (it is worth this much)
    dT_bf, dVc_bf = T_nb - T_f, Vc_nb - Vc_f
    dE_msg = dT_msg = dVc_msg = float("nan")
    if bfarch == "ctnn" and alive:
        with no_bf_messages(s):
            E_nm, T_nm, Vt_nm, Vc_nm = energy_TV(s)
        dE_msg, dT_msg, dVc_msg = E_nm - E_f, T_nm - T_f, Vc_nm - Vc_f
    T_full = T_f
    # The MAD-clipped mean drops coalescence spikes and biases E LOW — it can even read below the exact
    # energy, which the variational principle forbids. Report the unclipped error as the honest accuracy.
    ref = ref_energy(int(s.N), omega)
    err_raw = float((q["E_mean_raw"] - ref) / abs(ref) * 100.0) if ref else float("nan")
    err_nobf = float((E_nb - ref) / abs(ref) * 100.0) if ref else float("nan")
    return dict(bfarch=bfarch, seed=seed, omega=omega, error_pct=float(q.get("error_pct") or float("nan")),
                error_pct_raw=err_raw, var_EL=float(q["var_EL"]), deff=deff, dx_mag=dx_mag,
                alive=bool(alive), bf_rank=bf_rank, T_full=T_full, Vc_full=Vc_f,
                dE_backflow=dE_bf, dT_backflow=dT_bf, dVc_backflow=dVc_bf, err_pct_nobf=err_nobf,
                dE_msg=dE_msg, dT_msg=dT_msg, dVc_msg=dVc_msg,
                # in units of omega: is the mechanism scale-free or does it change character?
                dE_bf_over_w=dE_bf / omega, dE_msg_over_w=dE_msg / omega)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--camp", type=Path, default=DEFAULT_CAMP, help="campaign dir of checkpoints")
    CAMP = ap.parse_args().camp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for d in sorted(CAMP.glob("*bf_s*_w*")):
        if not (d / "checkpoint.pt").exists():
            continue
        # accepts both layouts: pinn_ctnnbf_s0_w1p0 (v3) and N6_ctnnbf_s0_w1 (scaling campaign)
        m = re.match(r"(?:pinn|N\d+)_(ctnn|conv)bf_s(\d+)_w([0-9p.]+)$", d.name)
        if not m:
            continue
        bfarch, seed, wtag = m.group(1), int(m.group(2)), float(m.group(3).replace("p", "."))
        try:
            r = analyze(d / "checkpoint.pt", bfarch, seed, wtag, dev); rows.append(r)
            flag = "" if r["alive"] else "   <<< DEAD BACKFLOW — rank/ablation NOT reported"
            print(f"  PINN+{bfarch}-bf s{seed} w{wtag:<5} err={r['error_pct_raw']:+.3f}% "
                  f"| bf buys dE={r['dE_backflow']:+.4f} (dT={r['dT_backflow']:+.4f} "
                  f"dVc={r['dVc_backflow']:+.4f}, {r['dE_bf_over_w']:+.2f}w) "
                  f"| msg dE={r['dE_msg']:+.4f} (dT={r['dT_msg']:+.4f} dVc={r['dVc_msg']:+.4f}) "
                  f"| d_eff={r['deff']:.2f} BFrank={r['bf_rank']:.1f} |dx|={r['dx_mag']:.3f}{flag}")
        except Exception as e:
            print(f"  {d.name}: ERR {e!r}")
    if not rows:
        print("no checkpoints yet"); return
    with open(CAMP / "master.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(CAMP / "master.json", "w"), indent=2)
    # contrast: CTNN vs conv backflow, seed-averaged, by omega
    print("\n[contrast] PINN + CTNN-backflow vs PINN + conventional-backflow (seed-avg):")
    print(f"{'w':>6} | {'err% ctnn/conv':>16} | {'d_eff ctnn/conv':>16} | {'BFrank ctnn/conv':>16} "
          f"| {'msg dE/dT/dVc (ctnn)':>24}")
    _ = None
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['bfarch'], r['omega'])].append(r)
    def m(bfarch, w, k):
        rs = by.get((bfarch, w), [])
        vs = [x[k] for x in rs if x[k] == x[k]]
        return sum(vs) / len(vs) if vs else float('nan')
    for w in sorted({r['omega'] for r in rows}, reverse=True):
        print(f"{w:>6} | {m('ctnn',w,'error_pct_raw'):+.3f}/{m('conv',w,'error_pct_raw'):+.3f} "
              f"| {m('ctnn',w,'deff'):.2f}/{m('conv',w,'deff'):.2f} "
              f"| {m('ctnn',w,'bf_rank'):.1f}/{m('conv',w,'bf_rank'):.1f} "
              f"| {m('ctnn',w,'dE_msg'):+.4f}/{m('ctnn',w,'dT_msg'):+.4f}/{m('ctnn',w,'dVc_msg'):+.4f}")
    print(f"-> {CAMP}/master.csv")


if __name__ == "__main__":
    main()
