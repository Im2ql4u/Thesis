"""Full analysis battery on a trained checkpoint (Angles 1-3 depth layer).

Loads an existing checkpoint (no retraining) and runs every load-time probe:
  Angle 1 (N=2): exact-truth SR-vs-plain alignment + distance sweep
  Angle 2: within-model message ablation (energy + variance), many-body signature,
           message decode (with target-variance controls), NTK spectrum
  Angle 3: natural-orbital occupations, intrinsic dimension, rotational invariance,
           pair correlation u(r)+cusp, internal order parameters
Saves battery.npz + battery_summary.json + REPORT_battery.md. Each probe is guarded.

Usage:
  python scripts/run_full_battery.py --load <dir>/checkpoint.pt [--tag w1]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from analysis import diagnostics as dg  # noqa: E402
from analysis import representation as rp  # noqa: E402
from analysis import physics_probes as pp  # noqa: E402
from analysis.ablation import has_messages, message_ablation_energy, manybody_signature  # noqa: E402
from analysis.system import load_system  # noqa: E402


def _try(name, fn, store, errors):
    try:
        store[name] = fn()
        print(f"  [ok] {name}")
    except Exception as e:  # noqa: BLE001
        errors[name] = repr(e)
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--align-samples", type=int, default=1024)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    sysm = load_system(a.load, device=a.device)
    N, omega = sysm.N, sysm.omega
    ref = config.get().E if np.isfinite(config.get().E) else None
    out = Path(a.load).parent
    tag = a.tag or out.name
    print(f"[battery] N={N} omega={omega} params={sysm.n_params():,} ctnn={has_messages(sysm.f_net)} "
          f"dev={sysm.device} -> {out}")

    x = sysm.sample(a.samples, steps=400, burn_in=800)
    E_L = dg.local_energy(sysm.log_psi, x, omega, sysm.params, lap_mode="exact")
    E_L = E_L[torch.isfinite(E_L)]
    res, errs = {}, {}
    res["base"] = {"N": N, "omega": omega, "n_params": sysm.n_params(),
                   "is_ctnn": has_messages(sysm.f_net),
                   "E": float(E_L.mean()), "var_EL": float(E_L.var()),
                   "err_pct": (None if ref is None else float((E_L.mean() - ref) / abs(ref) * 100))}

    # ---- Angle 2: message-passing (within-model, no capacity confound) ----
    if has_messages(sysm.f_net):
        _try("ablation", lambda: message_ablation_energy(sysm, n_samples=a.samples // 2), res, errs)
        _try("manybody", lambda: manybody_signature(sysm, x), res, errs)
        _try("message_decode", lambda: rp.decode_message(sysm, x[: a.align_samples]), res, errs)
        _try("message_target_var",
             lambda: _target_variances(sysm, x[: a.align_samples]), res, errs)

    # ---- NTK / SR diagnostics ----
    Oc = dg.build_O(sysm.log_psi, x[: a.align_samples], sysm.modules(), center=True)
    _try("spectrum", lambda: {k: v for k, v in dg.kernel_spectrum(Oc).items() if k != "eigenvalues"},
         res, errs)
    _try("sr_vs_plain",
         lambda: {k: v for k, v in dg.sr_vs_plain_alignment(Oc, E_L[: Oc.shape[0]]).items()
                  if not isinstance(v, np.ndarray)}, res, errs)

    # ---- Angle 1: exact-truth alignment (N=2 only) ----
    if N == 2:
        from analysis.exact_align import exact_alignment, alignment_vs_distance
        _try("exact_align", lambda: exact_alignment(sysm, x[: a.align_samples]), res, errs)
        _try("align_vs_distance",
             lambda: {k: v.tolist() for k, v in
                      alignment_vs_distance(sysm, x[: a.align_samples]).items()}, res, errs)

    # ---- Angle 3: physics extraction ----
    _try("natural_orbitals",
         lambda: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in pp.natural_orbital_occupations(sysm, x[: 512]).items()}, res, errs)
    _try("intrinsic_dim", lambda: _intrinsic_dim(sysm, x[: 1024]), res, errs)
    _try("rotational_invariance", lambda: pp.rotational_invariance(sysm, x[: 1024]), res, errs)
    _try("pair_correlation",
         lambda: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in pp.pair_correlation(sysm, x).items()}, res, errs)
    _try("order_params", lambda: pp.internal_order_params(sysm, x), res, errs)

    res["errors"] = errs
    clean = _clean(res)
    (out / "battery_summary.json").write_text(json.dumps(clean, indent=2) + "\n")
    print("[battery] summary:\n" + json.dumps(clean, indent=2))


def _clean(o):
    """Recursively convert numpy arrays/scalars to JSON-serialisable types."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (float, int, str, bool)) or o is None:
        return o
    try:
        return float(o)
    except Exception:
        return str(o)


@torch.no_grad()
def _target_variances(sysm, x):
    """Variances of the message-decode targets (control: low target variance => R^2 unreliable)."""
    t = rp.physical_local_targets(x, sysm.spin, sysm.omega)
    return {k: float(v.double().var()) for k, v in t.items()}


@torch.no_grad()
def _intrinsic_dim(sysm, x):
    """Intrinsic dimension of the per-configuration latent (pooled node features)."""
    cap = {}
    h = None
    for name in ["node_skip_fuse", "node_embed"]:
        m = getattr(sysm.f_net, name, None)
        if m is not None:
            h = m.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", o.detach()))
            break
    if h is None:
        return {"id_node": float("nan")}
    _ = sysm.log_psi(x)
    h.remove()
    feat = cap["h"]
    per_cfg = feat.reshape(feat.shape[0], -1, feat.shape[-1]).mean(1).cpu().numpy()  # (B,H)
    return {"id_node": pp.intrinsic_dimension(per_cfg), "linear_rank_node": _eff_rank(per_cfg)}


def _eff_rank(X):
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s**2
    return float((lam.sum() ** 2) / (lam**2).sum()) if lam.sum() > 0 else 0.0


if __name__ == "__main__":
    main()
