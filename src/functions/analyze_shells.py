# shell_analysis.py
# ------------------------------------------------------------
# Shell / ring structure analysis for 2D quantum-dot samples
# (supports S=1 "single-ring / (0,N)" detection + S>=2 shells)
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch


# =========================
# Config
# =========================

@dataclass
class ShellAnalysisConfig:
    # Where your results live: ../results/tables/{N}/omega_{omega:0.5f}/{run}/...pt
    base: Path = Path("../results/tables")

    # Gap threshold in "normalized gap units" (same idea you used)
    tau: float = 3.0

    # Maximum shells to consider (including S=1)
    max_shells: int = 6

    # For S>=2 shell splits: minimum shell size for non-center shells
    # (keeping this mild by default; set 1 to match your earlier behavior exactly)
    min_shell_size: int = 1

    # Allow singleton center shell (useful for (1,5), (1,7,12), etc.)
    allow_singleton_center: bool = True

    # --- S=1 detection ("(0,N)" candidate frames) ---
    # Require: no strong gap AND radii not too spread (coefficient of variation)
    single_shell_cv_max: float = 0.12

    # Pair-hist parameters (kept from your approach)
    nbins: int = 240
    r_q: float = 0.997
    chunk: int = 64

    # Reporting
    topk_occ: int = 12
    bootstrap_B: int = 300
    seed_frac: int = 42
    seed_mean: int = 123


# =========================
# Loading helpers
# =========================

def ensure_float32_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32).contiguous()

def load_bundle_any(path: Path) -> dict:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("Bundle isn't a dict; please save as a dict.")
    return obj

def bundle_to_trap_samples(obj: dict) -> Tuple[torch.Tensor, Optional[float], dict, str]:
    """
    Returns:
      y_trap : (K,N,2) in TRAP units (dimensionless)
      omega  : float or None
      meta   : dict (original bundle)
      used_key: which key we loaded from
    """
    omega = obj.get("omega", None)
    units = obj.get("units", None)

    if "samples_X_trap" in obj:
        X = obj["samples_X_trap"]
        y = X
        used = "samples_X_trap"
    elif "samples_X_bohr" in obj:
        X = obj["samples_X_bohr"]
        if omega is None:
            raise KeyError("Found samples_X_bohr but no 'omega' in bundle; cannot convert to trap units.")
        y = X * math.sqrt(float(omega))
        used = "samples_X_bohr"
    elif "X" in obj:
        X = obj["X"]
        if units == "trap":
            y = X
            used = "X (trap)"
        else:
            if omega is None:
                raise KeyError("Found 'X' without units==trap and no 'omega'—cannot convert to trap units.")
            y = X * math.sqrt(float(omega))
            used = "X (bohr→trap)"
    else:
        raise KeyError(f"No samples found. Available keys: {list(obj.keys())}. "
                       "Expected one of samples_X_trap / samples_X_bohr / X.")

    y = ensure_float32_cpu(y)
    if y.dim() != 3 or y.shape[-1] != 2:
        raise ValueError(f"Samples must be (K,N,2); got {tuple(y.shape)} from key '{used}'.")
    return y, (float(omega) if omega is not None else None), obj, used

def _omega_dirname(omega: Union[float, str]) -> str:
    w = float(omega)
    return f"omega_{w:0.5f}"

def resolve_latest_pt(base: Path, N: int, omega: Union[float, str]) -> Path:
    """
    Go to: base/N/omega_xxxxx/, pick most recent run folder (lexicographic timestamp),
    then pick a .pt inside (most recent by mtime).
    """
    root = (base / f"{N}" / _omega_dirname(omega))
    if not root.exists():
        raise FileNotFoundError(f"Missing folder: {root}")

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs under {root}")

    # timestamped dirs sort lexicographically if format is YYYYMMDD_HHMMSS
    run = run_dirs[-1]

    pts = list(run.rglob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt under {run}")

    # choose most recent pt by modification time
    pts.sort(key=lambda p: p.stat().st_mtime)
    return pts[-1]


# =========================
# Core geometry / stats
# =========================

def radial_sorted(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = torch.linalg.norm(y, dim=-1)  # (K,N)
    r_sorted, idx = torch.sort(r, dim=1)
    return r_sorted, idx, r

def _norm_gaps_from_rsorted(r_sorted: torch.Tensor) -> torch.Tensor:
    diffs = r_sorted[:, 1:] - r_sorted[:, :-1]      # (K,N-1)
    med = torch.median(diffs, dim=1).values.clamp_min(1e-12)  # (K,)
    return diffs / med[:, None]                      # (K,N-1)

def single_shell_mask(X: torch.Tensor, tau: float, cv_max: float) -> torch.Tensor:
    """
    S=1 candidate frames: no strong gap AND tight radial band.
    """
    r_sorted, _, r = radial_sorted(X)
    norm_gaps = _norm_gaps_from_rsorted(r_sorted)
    maxgap = torch.max(norm_gaps, dim=1).values
    no_gap = maxgap < tau

    r_mean = r.mean(dim=1).clamp_min(1e-12)
    r_std = r.std(dim=1, unbiased=False)
    cv = r_std / r_mean
    tight = cv < cv_max
    return no_gap & tight

def complex_order_phin(thetas: torch.Tensor, n: int) -> torch.Tensor:
    # thetas: (M,n)
    e = torch.view_as_complex(torch.stack([torch.cos(n * thetas), torch.sin(n * thetas)], dim=-1))
    return torch.abs(e.mean(dim=1))

def ring_lindemann_theta(thetas: torch.Tensor) -> torch.Tensor:
    """
    thetas: (M,n). Compute std(dphi)/target where target = 2π/n.
    """
    M, n = thetas.shape
    thetas_sorted, _ = torch.sort(thetas, dim=1)
    dphi = torch.diff(thetas_sorted, dim=1)
    wrap = (thetas_sorted[:, :1] + 2 * math.pi) - thetas_sorted[:, -1:]
    dphi = torch.cat([dphi, wrap], dim=1)
    target = 2 * math.pi / n
    std = torch.std(dphi, dim=1, unbiased=False)
    return std / target

def bootstrap_frac(count: int, total: int, B: int, seed: int) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = count / total
    rng = np.random.default_rng(seed)
    bs = rng.binomial(total, p, size=B) / total
    return float(p), float(bs.std(ddof=1))

def bootstrap_mean_std(x: torch.Tensor, B: int, seed: int) -> Tuple[float, float]:
    if x.numel() == 0:
        return float("nan"), float("nan")
    x_np = x.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x_np), size=(B, len(x_np)))
    means = x_np[idx].mean(axis=1)
    return float(x_np.mean()), float(means.std(ddof=1))


# =========================
# Shell splitting (S>=2)
# =========================

def split_shells_by_gaps(
    X: torch.Tensor,
    tau: float,
    max_shells: int,
    min_shell_size: int,
    allow_singleton_center: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      shell_id: (K,N) int64 in [0..S-1] for each frame (S varies per frame)
      S:        (K,)  int64 number of shells per frame (1..max_shells)

    Mechanism:
      - Sort radii.
      - Compute normalized gaps.
      - Choose up to (max_shells-1) largest gaps that exceed tau.
      - Those gaps define shell boundaries.
      - Optional mild filtering via min_shell_size.
    """
    K, N, _ = X.shape
    r_sorted, idx_sorted, _ = radial_sorted(X)
    norm_gaps = _norm_gaps_from_rsorted(r_sorted)   # (K,N-1)

    # For each frame: candidate cut positions where norm_gap > tau
    shell_id = torch.zeros((K, N), dtype=torch.long)
    S = torch.ones((K,), dtype=torch.long)

    for k in range(K):
        g = norm_gaps[k]  # (N-1,)
        cand = torch.where(g > tau)[0]
        if cand.numel() == 0 or max_shells <= 1:
            continue  # S=1, everything stays shell 0

        # pick top (max_shells-1) candidates by gap size
        vals = g[cand]
        top = torch.argsort(vals, descending=True)[: max_shells - 1]
        cuts = torch.sort(cand[top])[0].tolist()  # positions in sorted radii (between i and i+1)

        # Convert cuts into segment sizes in sorted index space
        # segments: [0..cut0], [cut0+1..cut1], ..., [last+1..N-1]
        boundaries = [-1] + cuts + [N - 1]
        seg_sizes = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            seg_sizes.append(b - a)

        # Filter tiny shells if requested (except allow singleton center)
        # We interpret "center" as the innermost shell (first segment).
        if min_shell_size > 1:
            ok = True
            for si, sz in enumerate(seg_sizes):
                if si == 0 and allow_singleton_center and sz == 1:
                    continue
                if sz < min_shell_size:
                    ok = False
                    break
            if not ok:
                continue  # fallback to S=1 for this frame

        # Assign shell ids to particles (in original indexing)
        # idx_sorted[k] maps sorted positions -> original particle index
        shell_k = torch.zeros((N,), dtype=torch.long)
        start = 0
        for si, sz in enumerate(seg_sizes):
            end = start + sz
            orig_idx = idx_sorted[k, start:end]
            shell_k[orig_idx] = si
            start = end

        shell_id[k] = shell_k
        S[k] = len(seg_sizes)

    return shell_id, S

def occupancy_tuple(shell_id_row: torch.Tensor, S: int) -> Tuple[int, ...]:
    return tuple(int((shell_id_row == s).sum().item()) for s in range(S))


# =========================
# Pair-hist helpers (global + per pair-type)
# =========================

def make_edges_from_r(X: torch.Tensor, q: float, nbins: int) -> torch.Tensor:
    r = torch.linalg.norm(X, dim=-1).reshape(-1)
    Rq = torch.quantile(r, q)
    rmax = float(2.0 * Rq)
    return torch.linspace(0.0, rmax, nbins + 1)

def global_pair_hist(X: torch.Tensor, edges: torch.Tensor, chunk: int) -> torch.Tensor:
    N = X.shape[1]
    iu = torch.triu_indices(N, N, 1)
    H = torch.zeros(len(edges) - 1)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        d = torch.cdist(X[s:e], X[s:e])[:, iu[0], iu[1]].reshape(-1)
        idx = torch.bucketize(d.clamp_max(edges[-1] - 1e-12), edges) - 1
        H += torch.bincount(idx, minlength=H.numel()).to(H.dtype)
    return H / (H.sum() + 1e-30)

def pair_hist_by_shell_types(
    X: torch.Tensor,
    shell_id: torch.Tensor,
    edges: torch.Tensor,
    chunk: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Accumulate histograms for each pair-type (a,b) with a<=b
    over the provided frames.
    """
    K, N, _ = X.shape
    iu = torch.triu_indices(N, N, 1)
    H: Dict[Tuple[int, int], torch.Tensor] = {}

    for s in range(0, K, chunk):
        e = min(s + chunk, K)
        Xm = X[s:e]
        sm = shell_id[s:e]  # (k,N)
        d = torch.cdist(Xm, Xm)[:, iu[0], iu[1]]      # (k,P)
        a = sm[:, iu[0]]
        b = sm[:, iu[1]]
        lo = torch.minimum(a, b)
        hi = torch.maximum(a, b)

        flat_d = d.reshape(-1)
        flat_lo = lo.reshape(-1)
        flat_hi = hi.reshape(-1)

        # process unique types in this chunk
        types = torch.stack([flat_lo, flat_hi], dim=1)
        uniq = torch.unique(types, dim=0)
        for u in uniq:
            aa = int(u[0].item()); bb = int(u[1].item())
            mask = (flat_lo == aa) & (flat_hi == bb)
            if mask.sum().item() == 0:
                continue
            idx = torch.bucketize(flat_d[mask].clamp_max(edges[-1] - 1e-12), edges) - 1
            if (aa, bb) not in H:
                H[(aa, bb)] = torch.zeros(len(edges) - 1)
            H[(aa, bb)] += torch.bincount(idx, minlength=len(edges) - 1).to(H[(aa, bb)].dtype)

    # normalize each histogram
    for k in list(H.keys()):
        H[k] = H[k] / (H[k].sum() + 1e-30)
    return H

def combinatorial_weights(occ: Tuple[int, ...]) -> Dict[Tuple[int, int], float]:
    """
    For shell occupancies occ=(n0,n1,...,n_{S-1}),
    return weights for pair-types (a,b) with a<=b as fractions of total pairs.
    """
    S = len(occ)
    N = sum(occ)
    total_pairs = N * (N - 1) // 2
    w: Dict[Tuple[int, int], float] = {}

    for a in range(S):
        na = occ[a]
        if na <= 0:
            continue
        # (a,a)
        w[(a, a)] = w.get((a, a), 0.0) + (na * (na - 1) // 2) / total_pairs
        # (a,b)
        for b in range(a + 1, S):
            nb = occ[b]
            if nb <= 0:
                continue
            w[(a, b)] = w.get((a, b), 0.0) + (na * nb) / total_pairs

    return w

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    an = a.detach().cpu().numpy()
    bn = b.detach().cpu().numpy()
    return float(np.dot(an, bn) / (np.linalg.norm(an) * np.linalg.norm(bn) + 1e-30))


# =========================
# Ring order summaries
# =========================

def summarize_single_shell(
    X: torch.Tensor,
    mask_1shell: torch.Tensor,
    cfg: ShellAnalysisConfig
) -> dict:
    """
    Summarize S=1 "(0,N)" candidate frames:
      - fraction among all frames
      - |Phi_N| and Lindemann on N-ring
    """
    K, N, _ = X.shape
    M = int(mask_1shell.sum().item())
    frac, frac_se = bootstrap_frac(M, K, B=cfg.bootstrap_B, seed=cfg.seed_frac)

    if M == 0:
        return dict(frac=frac, frac_se=frac_se, M=0, K=K,
                    phi=(float("nan"), float("nan")), lind=(float("nan"), float("nan")))

    Y = X[mask_1shell]  # (M,N,2)
    thetas = torch.atan2(Y[..., 1], Y[..., 0]).view(M, N)
    phi_vals = complex_order_phin(thetas, N)
    lind_vals = ring_lindemann_theta(thetas)

    return dict(
        frac=frac, frac_se=frac_se, M=M, K=K,
        phi=bootstrap_mean_std(phi_vals, B=cfg.bootstrap_B, seed=cfg.seed_mean),
        lind=bootstrap_mean_std(lind_vals, B=cfg.bootstrap_B, seed=cfg.seed_mean),
    )

def summarize_modal_shell_structure(
    X: torch.Tensor,
    shell_id: torch.Tensor,
    S: int,
    mask_S: torch.Tensor,
    cfg: ShellAnalysisConfig
) -> dict:
    """
    For frames with fixed shell count S:
      - modal occupancy
      - pair-hist mix cosine similarity vs global
      - ring order per shell (for shells with n>=3)
    """
    K, N, _ = X.shape
    idx = torch.where(mask_S)[0]
    if idx.numel() == 0:
        return {}

    # occupancy histogram
    occs = [occupancy_tuple(shell_id[i], S) for i in idx.tolist()]
    counts = Counter(occs)
    occ_modal, modal_count = counts.most_common(1)[0]

    # restrict to frames with modal occupancy
    mask_modal = mask_S.clone()
    for i in idx.tolist():
        if occupancy_tuple(shell_id[i], S) != occ_modal:
            mask_modal[i] = False
    idx_modal = torch.where(mask_modal)[0]
    M = int(idx_modal.numel())

    # pair-hist: global and modal-decomposed
    edges = make_edges_from_r(X, q=cfg.r_q, nbins=cfg.nbins)
    H_all = global_pair_hist(X, edges, chunk=cfg.chunk)

    X_modal = X[mask_modal]
    sid_modal = shell_id[mask_modal]
    H_types = pair_hist_by_shell_types(X_modal, sid_modal, edges, chunk=cfg.chunk)
    w = combinatorial_weights(occ_modal)

    # weighted mix (missing types get zeros)
    H_mix = torch.zeros_like(H_all)
    for t, wt in w.items():
        if t in H_types:
            H_mix += wt * H_types[t]
    cos = cosine_similarity(H_all, H_mix)

    # ring order per shell on modal frames
    ring_stats = {}
    Ym = X_modal
    Sm = sid_modal
    for s in range(S):
        n_s = occ_modal[s]
        if n_s < 3:
            ring_stats[s] = dict(n=n_s, phi=(float("nan"), float("nan")), lind=(float("nan"), float("nan")))
            continue
        # extract angles for that shell
        # Build (M,n_s) tensor of angles by gathering indices per frame
        rows = torch.arange(M).view(-1, 1)
        idx_s = [torch.where(Sm[i] == s)[0] for i in range(M)]
        # stack to (M,n_s)
        idx_s = torch.stack(idx_s, dim=0)
        pts = Ym[rows, idx_s]  # (M,n_s,2)
        thetas = torch.atan2(pts[..., 1], pts[..., 0]).view(M, n_s)

        phi_vals = complex_order_phin(thetas, n_s)
        lind_vals = ring_lindemann_theta(thetas)

        ring_stats[s] = dict(
            n=n_s,
            phi=bootstrap_mean_std(phi_vals, B=cfg.bootstrap_B, seed=cfg.seed_mean),
            lind=bootstrap_mean_std(lind_vals, B=cfg.bootstrap_B, seed=cfg.seed_mean),
        )

    # nice top-k occupancies for printing
    top = counts.most_common(cfg.topk_occ)

    return dict(
        S=S,
        frames_S=int(mask_S.sum().item()),
        occ_modal=occ_modal,
        modal_count=modal_count,
        top_occ=top,
        weights=w,
        cos=cos,
        ring_stats=ring_stats,
    )


# =========================
# Public API
# =========================

def analyze_case(
    N: int,
    omega: Union[float, str],
    cfg: Optional[ShellAnalysisConfig] = None,
    pt_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Full analysis for one (N, omega):
      - loads latest pt (unless pt_path provided)
      - converts to trap units if needed
      - detects S=1 "(0,N)" candidates
      - splits shells for S>=2 up to cfg.max_shells
      - modal analysis for each S in [2..max_shells] (if present)
    """
    cfg = cfg or ShellAnalysisConfig()
    if pt_path is None:
        pt_path = resolve_latest_pt(cfg.base, N=N, omega=omega)

    raw = load_bundle_any(pt_path)
    X, omega_in, meta, used_key = bundle_to_trap_samples(raw)
    K, N_loaded, _ = X.shape
    if N_loaded != N:
        # don’t fail hard; sometimes bundles omit n_particles correctly
        N = N_loaded

    # S=1 detection
    mask_1shell = single_shell_mask(X, tau=cfg.tau, cv_max=cfg.single_shell_cv_max)
    summ_1shell = summarize_single_shell(X, mask_1shell, cfg)

    # S>=2 splitting
    shell_id, Svec = split_shells_by_gaps(
        X, tau=cfg.tau, max_shells=cfg.max_shells,
        min_shell_size=cfg.min_shell_size,
        allow_singleton_center=cfg.allow_singleton_center,
    )

    # Shell count summary (S=1..max_shells)
    counts_S = Counter([int(s.item()) for s in Svec])
    splittable = 1.0 - (counts_S.get(1, 0) / K)

    # modal analysis per S>=2
    modal_by_S = {}
    for S in range(2, cfg.max_shells + 1):
        mask_S = (Svec == S)
        if int(mask_S.sum().item()) == 0:
            continue
        modal_by_S[S] = summarize_modal_shell_structure(X, shell_id, S, mask_S, cfg)

    if verbose:
        print("=" * 88)
        print(f"CASE: N={N}  omega={float(omega):g}  (bundle omega={omega_in})")
        print(f"Loaded: {pt_path}")
        print(f"Key: {used_key} | units: {meta.get('units', None)} | K={K:,} frames")
        print("=" * 88)

        # S=1 candidate report
        f1, f1se = summ_1shell["frac"], summ_1shell["frac_se"]
        phm, phse = summ_1shell["phi"]
        ldm, ldse = summ_1shell["lind"]
        print(f"\nS=1 '(0,{N})' candidate frames (no strong gap + tight radii): "
              f"{f1:.3f} ± {f1se:.3f}   ({summ_1shell['M']}/{K})")
        if summ_1shell["M"] > 0:
            # compare to random baseline ~ sqrt(pi)/(2*sqrt(N))
            baseline = (math.sqrt(math.pi) / (2.0 * math.sqrt(N)))
            print(f"  |Φ_{N}|={phm:.3f}±{phse:.3f}   (random baseline ≈ {baseline:.3f})")
            print(f"  Lind={ldm:.3f}±{ldse:.3f}")

        # shell split summary
        print(f"\nShell-splittable frames (S=2..{cfg.max_shells}): {splittable:.3f}  "
              f"({K - counts_S.get(1,0)}/{K})")
        for S in range(2, cfg.max_shells + 1):
            c = counts_S.get(S, 0)
            if c:
                print(f"  S={S}: {c} frames  ({c / K:.3f})")

        # modal outputs per S
        for S, out in modal_by_S.items():
            frames_S = out["frames_S"]
            occ_modal = out["occ_modal"]
            modal_count = out["modal_count"]
            print(f"\n--- MODAL ANALYSIS for S={S} shells ---")
            print(f"S={S} frames: {frames_S} | modal occ {occ_modal} occurs {modal_count} times "
                  f"({(modal_count/frames_S)*100:.2f}% of S={S} frames)")
            top = out["top_occ"]
            pretty = ", ".join([f"{k}:{v} ({(100*v/frames_S):.2f}%)" for k, v in top])
            print(f"Top occupancies: {pretty}")

            print("\nCombinatorial weights per pair-type:")
            for k in sorted(out["weights"].keys()):
                print(f"  w{k} = {out['weights'][k]:.4f}")

            print(f"\nCosine similarity: global pair-hist vs weighted S={S} mix = {out['cos']:.4f}   (≥0.98 is excellent)")

            print("\nBond order + angular Lindemann per shell (shell 0=inner ...):")
            for s in range(S):
                st = out["ring_stats"][s]
                n_s = st["n"]
                if n_s < 3:
                    print(f"  shell{s} (n={n_s}): skipped (n<3)")
                else:
                    pm, pse = st["phi"]
                    lm, lse = st["lind"]
                    print(f"  shell{s} (n={n_s}): |Φ_{n_s}|={pm:.3f}±{pse:.3f}, Lind={lm:.3f}±{lse:.3f}")

    return dict(
        pt=str(pt_path),
        N=N,
        omega=float(omega),
        omega_bundle=omega_in,
        used_key=used_key,
        K=K,
        counts_S=dict(counts_S),
        single_shell=summ_1shell,
        modal_by_S=modal_by_S,
    )

def run_many(
    Ns: Iterable[int],
    omegas: Iterable[Union[float, str]],
    cfg: Optional[ShellAnalysisConfig] = None,
    pairwise: bool = False,
    verbose: bool = True,
) -> Dict[Tuple[int, float], dict]:
    """
    Run multiple analyses in one go.
      - pairwise=False: Cartesian product Ns x omegas
      - pairwise=True : zip(Ns, omegas)
    """
    cfg = cfg or ShellAnalysisConfig()
    Ns = list(Ns)
    omegas = list(omegas)

    out: Dict[Tuple[int, float], dict] = {}
    if pairwise:
        for N, w in zip(Ns, omegas):
            res = analyze_case(N=N, omega=w, cfg=cfg, verbose=verbose)
            out[(N, float(w))] = res
    else:
        for N in Ns:
            for w in omegas:
                res = analyze_case(N=N, omega=w, cfg=cfg, verbose=verbose)
                out[(N, float(w))] = res
    return out
