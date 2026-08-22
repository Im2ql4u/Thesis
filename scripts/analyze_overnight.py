"""Post-run analyzer for the overnight campaign — turns every trained checkpoint into one master
mechanism table so the comparisons (what-is-MP-worth, paradigm-internal-structure) fall right out.

For each results/analysis/2026-07-02_overnight/<tag>_s<seed>_w<omega>/checkpoint.pt it computes:
  energy error, var(E_L), fair d_eff, backflow displacement rank, and the message-ablation kinetic
  gain dT (zeroing rho_* -> pairwise), all on common |Psi|^2 samples. Parses (group, arch, backflow,
  paradigm, seed, omega) from the tag. Writes master.csv.

Run: CUDA_VISIBLE_DEVICES=0 python3 -u scripts/analyze_overnight.py
"""
from __future__ import annotations

import contextlib
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.system import load_system  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAMP = ROOT / "results/analysis/2026-07-02_overnight"
REF_E = {1.0: 20.15932, 0.1: 3.55164, 0.01: 0.69036}  # N=6 references; N=12 handled by config lookup
RHO = ["rho_v_to_e_down", "rho_e_to_v_down", "rho_v_to_e_up", "rho_e_to_v_up"]
SAMPLE_KW = dict(steps=400, burn_in=800)


@contextlib.contextmanager
def no_messages(system):
    saved = []
    for nm in RHO:
        ml = getattr(system.f_net, nm, None)
        if ml:
            for lin in ml:
                saved.append((lin, lin.weight.data.clone())); lin.weight.data.zero_()
    try:
        yield
    finally:
        for lin, w in saved:
            lin.weight.data.copy_(w)


def _effrank(M):
    X = M - M.mean(0, keepdims=True); s = np.linalg.svd(X, compute_uv=False); l = s ** 2
    return float((l.sum() ** 2) / (l ** 2).sum()) if l.sum() > 0 else 0.0


def kinetic(system, x, omega):
    xg = x.detach().requires_grad_(True)
    g = torch.autograd.grad(system.log_psi(xg).sum(), xg)[0]
    return float((0.5 * (g ** 2).sum(dim=(1, 2))).double().mean())


def analyze(ckpt: Path, tag: str, seed: int, omega: float, dev: str) -> dict:
    s = load_system(str(ckpt), device=dev, seed=0)
    x = s.sample(1536, **SAMPLE_KW)
    EL = dg.local_energy(s.log_psi, x, s.omega, s.params, chunk=256)
    q = dg.gs_quality(EL, ref_energy=REF_E.get(round(omega, 4)))
    big = sum(p.numel() for p in s.f_net.parameters()) > 1e5
    O = dg.build_O(s.log_psi, x, [s.f_net], center=True)
    deff = float(dg.kernel_spectrum(O.cpu() if big else O)["effective_rank"])
    bf_rank = float("nan")
    if s.backflow_net is not None:
        with torch.no_grad():
            dx = s.backflow_net(x, spin=s.spin)
        bf_rank = _effrank(dx.reshape(x.shape[0], -1).cpu().double().numpy())
    T_full = kinetic(s, x, omega)
    dT_msg = float("nan")
    if any(getattr(s.f_net, nm, None) for nm in RHO):
        with no_messages(s):
            dT_msg = kinetic(s, x, omega) - T_full
    m = re.match(r"([A-Z])_(.+?)_(bf|nobf)_?(colloc|vmcadam|collocsr)?", tag)
    group = m.group(1) if m else "?"
    arch = "ctnn" if "ctnn" in tag else ("deepset" if "deepset" in tag else "?")
    backflow = "bf" if "_bf" in tag else "nobf"
    paradigm = ("collocsr" if "collocsr" in tag else "colloc" if "colloc" in tag
                else "vmcadam" if "vmcadam" in tag else "vmcsr")
    return dict(tag=tag, group=group, arch=arch, backflow=backflow, paradigm=paradigm, seed=seed,
                omega=omega, N=s.N, error_pct=float(q.get("error_pct") or float("nan")),
                var_EL=float(q["var_EL"]), deff=deff, bf_rank=bf_rank, T_full=T_full, dT_msg=dT_msg)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for d in sorted(CAMP.glob("*_s*_w*")):
        ck = d / "checkpoint.pt"
        if not ck.exists():
            continue
        m = re.match(r"(.+)_s(\d+)_w([0-9p.]+)$", d.name)
        if not m:
            continue
        tag, seed, wtag = m.group(1), int(m.group(2)), m.group(3).replace("p", ".")
        try:
            rows.append(analyze(ck, tag, seed, float(wtag), dev))
            r = rows[-1]
            print(f"  {tag:24} s{seed} w{wtag:5}  err={r['error_pct']:+.3f}% var={r['var_EL']:.2e} "
                  f"d_eff={r['deff']:.2f} BFrank={r['bf_rank']:.1f} dT_msg={r['dT_msg']:+.3f}")
        except Exception as e:
            print(f"  {d.name}: ERR {e!r}")
    if rows:
        with open(CAMP / "master.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        json.dump(rows, open(CAMP / "master.json", "w"), indent=2)
        print(f"\n-> {CAMP}/master.csv  ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
