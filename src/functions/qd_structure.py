# qd_structure.py
from __future__ import annotations

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# ---------------------------
# Small utilities
# ---------------------------

def _omega_dirname(omega: float) -> str:
    # matches your folders like omega_0.00100
    return f"omega_{omega:0.5f}"

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else -1e30

def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts.astype(np.float64) / float(total)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _angles(xy: np.ndarray) -> np.ndarray:
    # xy: (..., 2)
    return np.arctan2(xy[..., 1], xy[..., 0])

def _radii(xy: np.ndarray) -> np.ndarray:
    return np.sqrt((xy[..., 0] ** 2) + (xy[..., 1] ** 2))

def _phi_m(angles: np.ndarray, m: int) -> float:
    # angles shape (n,)
    z = np.exp(1j * m * angles)
    return float(np.abs(z.mean()))

def _angular_lindemann_for_ring(angles: np.ndarray, radii: Optional[np.ndarray] = None) -> float:
    """
    A simple, robust 'angular Lindemann' for a ring-like set of points:

    1) sort angles on [0, 2π)
    2) compute neighbor angular spacings Δθ
    3) map to approximate chord lengths ~ 2 r sin(Δθ/2) (if radii given)
    4) return CV = std/mean of chord lengths (or of Δθ if no radii)

    This can exceed 1 if spacings are extremely uneven (clumping), which is fine.
    """
    th = np.mod(angles, 2*np.pi)
    th = np.sort(th)
    dth = np.diff(np.concatenate([th, th[:1] + 2*np.pi]))
    if radii is None:
        mean = dth.mean()
        std = dth.std()
        return float(std / mean) if mean > 0 else float("nan")

    r = float(np.mean(radii))
    chords = 2.0 * r * np.sin(0.5 * dth)
    mean = chords.mean()
    std = chords.std()
    return float(std / mean) if mean > 0 else float("nan")


# ---------------------------
# File discovery + loading
# ---------------------------

def find_latest_pt_file(
    root: Union[str, Path],
    N: int,
    omega: float,
) -> Path:
    """
    Searches:
      root / {N} / omega_{omega:0.5f} / {run_dir}/ *.pt

    Picks:
      - latest run_dir by directory name (timestamp-like) AND fallback to mtime
      - inside that: latest .pt by mtime
    """
    root = Path(root)
    base = root / str(N) / _omega_dirname(omega)
    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories in: {base}")

    # Prefer timestamp-like lexical order, but also stable via mtime fallback
    run_dirs_sorted = sorted(
        run_dirs,
        key=lambda p: (p.name, p.stat().st_mtime),
        reverse=True,
    )
    latest_run = run_dirs_sorted[0]

    pt_files = sorted(latest_run.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in: {latest_run}")

    return pt_files[0]


def load_samples_from_pt(
    pt_path: Union[str, Path],
    key_preference: Sequence[str] = ("samples_X_trap", "samples_X_bohr", "X", "samples"),
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns:
      X: (K, N, 2) float64 numpy
      meta: dict with 'units' if present, and whatever else is useful

    Notes:
      - loads with map_location="cpu"
      - tries keys in key_preference
      - supports subsampling via stride and max_frames
    """
    pt_path = Path(pt_path)
    bundle = torch.load(pt_path, map_location="cpu")

    meta: Dict[str, Any] = {}
    if isinstance(bundle, dict):
        if "units" in bundle:
            meta["units"] = bundle["units"]
    else:
        # If your .pt is literally a tensor
        bundle = {"_tensor": bundle}

    X = None
    chosen_key = None
    for k in key_preference:
        if isinstance(bundle, dict) and k in bundle:
            X = bundle[k]
            chosen_key = k
            break

    if X is None:
        # Fall back: find first tensor-ish value with shape (..., N, 2)
        for k, v in bundle.items():
            if torch.is_tensor(v) and v.ndim == 3 and v.shape[-1] == 2:
                X = v
                chosen_key = k
                break
    if X is None:
        raise KeyError(
            f"Could not find samples in {pt_path}. "
            f"Tried keys {list(key_preference)} and tensor fallback."
        )

    Xn = _to_numpy(X).astype(np.float64)  # (K,N,2)
    if Xn.ndim != 3 or Xn.shape[-1] != 2:
        raise ValueError(f"Expected (K,N,2), got {Xn.shape} from key={chosen_key}")

    meta["pt_path"] = str(pt_path)
    meta["samples_key"] = chosen_key
    meta["K_total"] = int(Xn.shape[0])
    meta["N"] = int(Xn.shape[1])

    # stride first, then max_frames cap
    if frame_stride > 1:
        Xn = Xn[::frame_stride]
    if max_frames is not None and Xn.shape[0] > max_frames:
        Xn = Xn[:max_frames]

    meta["K_used"] = int(Xn.shape[0])
    return Xn, meta


# ---------------------------
# Shell splitting + occupancy labels
# ---------------------------

@dataclass
class RingStats:
    ring_frac: float
    ring_count: int
    phiN_mean: float
    phiN_se: float
    lind_mean: float
    lind_se: float
    gapcv_mean: float
    gapcv_se: float
    rcv_mean: float
    rcv_se: float

@dataclass
class ShellModalStats:
    S: int
    modal_occ: Tuple[int, ...]
    p_modal_within_S: float
    count_S: int
    # per-shell metrics (for modal frames only; n<3 => None)
    phi_by_shell: List[Optional[Tuple[float, float]]]   # (mean, se) or None
    lind_by_shell: List[Optional[Tuple[float, float]]]

@dataclass
class CaseResult:
    N: int
    omega: float
    pt_path: str
    samples_key: str
    K_used: int

    # ring candidates (0,N)
    ring: RingStats

    # shell split summary
    split_frac: float
    split_count: int
    mean_S: float
    H_occ: float
    p_modal: float
    modal_occ: Tuple[int, ...]
    modal_count: int

    # per-S modal details (e.g. S=2, S=3)
    modal_by_S: List[ShellModalStats]

    # transitions
    top_transitions: List[Tuple[Tuple[int, ...], Tuple[int, ...], int]]


def _frame_occupancy_from_gaps(
    r_sorted: np.ndarray,
    gaps: np.ndarray,
    tau: float,
) -> Tuple[Tuple[int, ...], float, float, float]:
    """
    For a single frame:
      - identify "strong gaps" where gap > tau * median(gaps)
      - occupancy is counts between those cuts
      - returns (occ_tuple, margin, gapCV, rCV)

    margin: max_strong_gap / (median_gap + eps)   (0 if no strong gap)
    gapCV : std(g)/mean(g)
    rCV   : std(r)/mean(r)
    """
    eps = 1e-12
    g = gaps
    med = float(np.median(g)) if g.size > 0 else 0.0
    thr = tau * med

    strong = (g > thr)
    cut_idxs = np.where(strong)[0]  # indices in gaps, cut after that index

    # margin for confidence
    if cut_idxs.size > 0:
        max_strong = float(g[cut_idxs].max())
        margin = max_strong / (med + eps)
    else:
        margin = 0.0

    # occupancy
    # boundaries are [0] + (cut+1) + [N]
    Np = r_sorted.size
    bounds = [0] + [int(i + 1) for i in cut_idxs.tolist()] + [Np]
    occ = tuple(bounds[i+1] - bounds[i] for i in range(len(bounds) - 1))

    # CV diagnostics
    gmean = float(g.mean()) if g.size > 0 else 0.0
    gapcv = float(g.std() / (gmean + eps)) if gmean > 0 else float("nan")

    rmean = float(r_sorted.mean()) if r_sorted.size > 0 else 0.0
    rcv = float(r_sorted.std() / (rmean + eps)) if rmean > 0 else float("nan")

    return occ, margin, gapcv, rcv


def analyze_case(
    X: np.ndarray,                       # (K,N,2)
    N: int,
    omega: float,
    *,
    tau: float = 3.0,
    min_shells: int = 2,
    max_shells: int = 6,
    # ring candidate controls (0,N)
    ring_gap_tau: Optional[float] = None,   # if None, uses same tau
    ring_r_cv_max: float = 0.12,
    ring_max_gap_over_med: float = 1.2,     # "no strong gap" condition
    # sampling
    max_modal_frames: int = 5000,
    # transitions
    transition_stride: int = 1,
    # margin filter (0 => no filter)
    min_margin: float = 0.0,
) -> Dict[str, Any]:
    """
    Returns a dict of raw results; scan_cases() wraps into CaseResult and printing.
    """
    K = X.shape[0]
    assert X.shape[1] == N and X.shape[2] == 2

    ring_tau = ring_gap_tau if ring_gap_tau is not None else tau

    # Per-frame computed labels
    occ_labels: List[Tuple[int, ...]] = []
    S_list = np.empty(K, dtype=np.int32)
    margin_list = np.empty(K, dtype=np.float64)
    gapcv_list = np.empty(K, dtype=np.float64)
    rcv_list = np.empty(K, dtype=np.float64)

    # ring candidates
    ring_mask = np.zeros(K, dtype=bool)
    ring_phi = np.full(K, np.nan, dtype=np.float64)
    ring_lind = np.full(K, np.nan, dtype=np.float64)

    # We'll loop frames (N is small; this is OK and stable).
    for t in range(K):
        xy = X[t]  # (N,2)
        r = _radii(xy)
        idx = np.argsort(r)
        r_sorted = r[idx]
        g = np.diff(r_sorted)

        occ, margin, gapcv, rcv = _frame_occupancy_from_gaps(r_sorted, g, tau=tau)

        occ_labels.append(occ)
        S_list[t] = len(occ)
        margin_list[t] = margin
        gapcv_list[t] = gapcv
        rcv_list[t] = rcv

        # ring candidates: "no strong gap" + "tight radii"
        # no-strong-gap: max_gap <= ring_max_gap_over_med * median_gap (using ring_tau scale)
        # Here we recompute threshold notion with ring_tau by comparing max_gap/median_gap
        if g.size > 0:
            medg = float(np.median(g)) + 1e-12
            max_over_med = float(g.max() / medg)
        else:
            max_over_med = 0.0

        if (rcv <= ring_r_cv_max) and (max_over_med <= ring_max_gap_over_med):
            # treat as (0,N) candidate
            ring_mask[t] = True
            th = _angles(xy)
            ring_phi[t] = _phi_m(th, m=N)  # N-fold order parameter
            ring_lind[t] = _angular_lindemann_for_ring(th, radii=r)

    # Apply margin filter to shell-splittable statistics (NOT to ring candidates by default)
    margin_ok = (margin_list >= min_margin) if (min_margin and min_margin > 0) else np.ones(K, dtype=bool)

    split_mask = (
        margin_ok &
        (S_list >= min_shells) &
        (S_list <= max_shells)
    )
    split_count = int(split_mask.sum())
    split_frac = float(split_count / K)

    # Occupancy distribution among split frames
    occs_split = [occ_labels[i] for i in range(K) if split_mask[i]]
    if split_count > 0:
        # count occupancies
        from collections import Counter
        C = Counter(occs_split)
        occ_items = list(C.items())
        occ_items.sort(key=lambda kv: kv[1], reverse=True)

        modal_occ, modal_count = occ_items[0][0], int(occ_items[0][1])
        p_modal = float(modal_count / split_count)

        counts = np.array([v for _, v in occ_items], dtype=np.int64)
        H_occ = _shannon_entropy_from_counts(counts)
        mean_S = float(S_list[split_mask].mean())
    else:
        modal_occ, modal_count, p_modal, H_occ, mean_S = tuple(), 0, float("nan"), float("nan"), float("nan")
        occ_items = []

    # Build per-S modal details (only for S in [min_shells, max_shells])
    modal_by_S: List[ShellModalStats] = []
    for S in range(min_shells, max_shells + 1):
        S_mask = split_mask & (S_list == S)
        count_S = int(S_mask.sum())
        if count_S == 0:
            continue

        occs_S = [occ_labels[i] for i in range(K) if S_mask[i]]
        from collections import Counter
        C_S = Counter(occs_S)
        items = sorted(C_S.items(), key=lambda kv: kv[1], reverse=True)
        modal_occ_S, modal_count_S = items[0][0], int(items[0][1])
        p_modal_S = float(modal_count_S / count_S)

        # Compute per-shell phi/lind on frames matching modal_occ_S
        idxs = [i for i in range(K) if S_mask[i] and occ_labels[i] == modal_occ_S]
        if len(idxs) > max_modal_frames:
            idxs = idxs[:max_modal_frames]

        phi_by_shell: List[Optional[Tuple[float, float]]] = []
        lind_by_shell: List[Optional[Tuple[float, float]]] = []

        # To assign shells: use sorted radii; shells are consecutive chunks.
        for shell_j, n_shell in enumerate(modal_occ_S):
            if n_shell < 3:
                phi_by_shell.append(None)
                lind_by_shell.append(None)
                continue

            phis = []
            linds = []
            for i in idxs:
                xy = X[i]
                r = _radii(xy)
                order = np.argsort(r)
                xy_sorted = xy[order]
                r_sorted = r[order]

                # shell slice
                start = sum(modal_occ_S[:shell_j])
                end = start + n_shell
                pts = xy_sorted[start:end]
                th = _angles(pts)
                phis.append(_phi_m(th, m=n_shell))
                linds.append(_angular_lindemann_for_ring(th, radii=r_sorted[start:end]))

            phis = np.asarray(phis, dtype=np.float64)
            linds = np.asarray(linds, dtype=np.float64)
            phi_mean = float(np.nanmean(phis))
            lind_mean = float(np.nanmean(linds))

            # standard error
            phi_se = float(np.nanstd(phis, ddof=1) / math.sqrt(max(1, np.isfinite(phis).sum()))) if np.isfinite(phis).sum() > 1 else 0.0
            lind_se = float(np.nanstd(linds, ddof=1) / math.sqrt(max(1, np.isfinite(linds).sum()))) if np.isfinite(linds).sum() > 1 else 0.0

            phi_by_shell.append((phi_mean, phi_se))
            lind_by_shell.append((lind_mean, lind_se))

        modal_by_S.append(
            ShellModalStats(
                S=S,
                modal_occ=modal_occ_S,
                p_modal_within_S=p_modal_S,
                count_S=count_S,
                phi_by_shell=phi_by_shell,
                lind_by_shell=lind_by_shell,
            )
        )

    # Ring stats summary
    ring_count = int(ring_mask.sum())
    ring_frac = float(ring_count / K)

    def _mean_se(arr: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        x = arr[mask]
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        mean = float(x.mean())
        if x.size == 1:
            return mean, 0.0
        se = float(x.std(ddof=1) / math.sqrt(x.size))
        return mean, se

    phiN_mean, phiN_se = _mean_se(ring_phi, ring_mask)
    lind_mean, lind_se = _mean_se(ring_lind, ring_mask)
    gapcv_mean, gapcv_se = _mean_se(gapcv_list, ring_mask)
    rcv_mean, rcv_se = _mean_se(rcv_list, ring_mask)

    ring_stats = RingStats(
        ring_frac=ring_frac,
        ring_count=ring_count,
        phiN_mean=phiN_mean,
        phiN_se=phiN_se,
        lind_mean=lind_mean,
        lind_se=lind_se,
        gapcv_mean=gapcv_mean,
        gapcv_se=gapcv_se,
        rcv_mean=rcv_mean,
        rcv_se=rcv_se,
    )

    # Transitions (simple; use stride to reduce autocorrelation)
    trans_counts: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
    idxs_for_chain = [i for i in range(0, K, max(1, transition_stride)) if split_mask[i]]
    for a, b in zip(idxs_for_chain[:-1], idxs_for_chain[1:]):
        s1 = occ_labels[a]
        s2 = occ_labels[b]
        if s1 == s2:
            continue
        key = (s1, s2)
        trans_counts[key] = trans_counts.get(key, 0) + 1

    top_transitions = sorted(
        [(k[0], k[1], v) for k, v in trans_counts.items()],
        key=lambda t: t[2],
        reverse=True,
    )[:25]

    return dict(
        N=N,
        omega=omega,
        K=K,
        ring=ring_stats,
        split_frac=split_frac,
        split_count=split_count,
        mean_S=mean_S,
        H_occ=H_occ,
        p_modal=p_modal,
        modal_occ=modal_occ,
        modal_count=modal_count,
        modal_by_S=modal_by_S,
        top_transitions=top_transitions,
    )


# ---------------------------
# Public API: scan + report
# ---------------------------

def scan_cases(
    Ns: Sequence[int],
    omegas: Sequence[float],
    *,
    root: Union[str, Path] = "../results/tables",
    key_preference: Sequence[str] = ("samples_X_trap", "samples_X_bohr"),
    tau0: float = 3.0,
    min_shells: int = 2,
    max_shells: int = 6,
    # ring
    ring_r_cv_max: float = 0.12,
    ring_max_gap_over_med: float = 1.2,
    # sampling controls
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    max_modal_frames: int = 5000,
    # margin confidence filter for shell statistics
    min_margin: float = 0.0,
    # transitions
    transition_stride: int = 1,
    # output
    save_json: Optional[Union[str, Path]] = None,
) -> List[CaseResult]:
    """
    Main entrypoint: runs analysis for all (N, omega).

    Returns a list of CaseResult (dataclasses).
    """
    results: List[CaseResult] = []

    for N in Ns:
        tau = tau0*(N/6)**0.5
        for omega in omegas:
            pt_path = find_latest_pt_file(root=root, N=N, omega=omega)
            X, meta = load_samples_from_pt(
                pt_path,
                key_preference=key_preference,
                max_frames=max_frames,
                frame_stride=frame_stride,
            )

            raw = analyze_case(
                X, N=N, omega=omega,
                tau=tau,
                min_shells=min_shells,
                max_shells=max_shells,
                ring_r_cv_max=ring_r_cv_max,
                ring_max_gap_over_med=ring_max_gap_over_med,
                max_modal_frames=max_modal_frames,
                min_margin=min_margin,
                transition_stride=transition_stride,
            )

            # Build CaseResult
            r = CaseResult(
                N=N,
                omega=float(omega),
                pt_path=str(meta["pt_path"]),
                samples_key=str(meta["samples_key"]),
                K_used=int(meta["K_used"]),

                ring=raw["ring"],

                split_frac=float(raw["split_frac"]),
                split_count=int(raw["split_count"]),
                mean_S=float(raw["mean_S"]),
                H_occ=float(raw["H_occ"]),
                p_modal=float(raw["p_modal"]) if not (raw["p_modal"] is None) else float("nan"),
                modal_occ=tuple(raw["modal_occ"]),
                modal_count=int(raw["modal_count"]),

                modal_by_S=list(raw["modal_by_S"]),
                top_transitions=list(raw["top_transitions"]),
            )
            results.append(r)

    if save_json is not None:
        save_json = Path(save_json)
        payload = [asdict(r) for r in results]
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(payload, indent=2, default=str))

    return results


def print_report(r: CaseResult, *, max_transitions: int = 12) -> None:
    print("=" * 96)
    print(f"CASE: N={r.N}  omega={r.omega:g}")
    print(f"Loaded: {r.pt_path}")
    print(f"Key: {r.samples_key} | K={r.K_used}")
    print("-" * 96)

    # Ring
    if r.ring.ring_count > 0:
        print(f"S=1 '(0,{r.N})' candidate frames: {r.ring.ring_frac:.3f}  ({r.ring.ring_count}/{r.K_used})")
        print(f"  |Phi_{r.N}|={r.ring.phiN_mean:.3f}±{r.ring.phiN_se:.3f}   (random baseline ~ 1/sqrt(N)={1/math.sqrt(r.N):.3f})")
        print(f"  Lind={r.ring.lind_mean:.3f}±{r.ring.lind_se:.3f}")
        print(f"  rCV={r.ring.rcv_mean:.3f}±{r.ring.rcv_se:.3f}  gapCV={r.ring.gapcv_mean:.3f}±{r.ring.gapcv_se:.3f}")
        print("")
    else:
        print(f"S=1 '(0,{r.N})' candidate frames: 0.000  (0/{r.K_used})\n")

    # Shell split summary
    print(f"Shell-splittable frames (S={2}..{max([m.S for m in r.modal_by_S], default=2)}): {r.split_frac:.3f}  ({r.split_count}/{r.K_used})")
    print(f"  mean_S={r.mean_S:.3f}   H_occ={r.H_occ:.3f}   p_modal={r.p_modal:.3f}   modal={r.modal_occ} ({r.modal_count})")
    print("")

    # Per-S modals
    for m in r.modal_by_S:
        print(f"--- MODAL ANALYSIS for S={m.S} ---")
        print(f"S={m.S} frames: {m.count_S} | modal occ {m.modal_occ} occurs {m.p_modal_within_S*100:.2f}% of S={m.S} frames")
        for j, n_shell in enumerate(m.modal_occ):
            if n_shell < 3:
                print(f"  shell{j} (n={n_shell}): skipped (n<3)")
            else:
                phi = m.phi_by_shell[j]
                lind = m.lind_by_shell[j]
                if phi is None or lind is None:
                    print(f"  shell{j} (n={n_shell}): (missing)")
                else:
                    print(f"  shell{j} (n={n_shell}): |Phi_{n_shell}|={phi[0]:.3f}±{phi[1]:.3f}, Lind={lind[0]:.3f}±{lind[1]:.3f}")
        print("")

    # Transitions
    if r.top_transitions:
        print("Top transitions:")
        for (a, b, c) in r.top_transitions[:max_transitions]:
            print(f"  {a} -> {b}: {c}")
    print("=" * 96)
