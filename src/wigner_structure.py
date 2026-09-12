"""Shell detection, bond order and the held-out polygon shell model for Wigner molecules.

Restored from the analysis notebook that produced
``results/structural_tables/shell_summary.csv`` and
``results/structural_tables_shell_model_fit/shell_model_fit.csv`` (``src/Analysis2.ipynb``,
cells "scan_shells_to_latex" and "shell-model fit"; the notebook was deleted on
2026-01-10 and recovered from the cluster's trash on 2026-09-12). The logic is unchanged;
the only structural change is that the random generator of the shell model is passed in
explicitly instead of being a notebook global.

Shells are defined by *radial rank*: per frame the particles are sorted by radius and a
fixed set of global rank cuts assigns them to shells. The bond order of a shell of n
particles is |n^-1 sum_k exp(i n theta_k)|, which is rotation invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

EPS = 1e-12
CENTRE_RATIO_MAX = 0.55  # a single central particle must sit well inside the next ring


# ---------------------------------------------------------------------------
# Shell detection (radial-gap salience) and per-shell metrics
# ---------------------------------------------------------------------------
@dataclass
class ShellCfg:
    max_shells: int = 6
    n_min: int = 3
    allow_center_1: bool = True
    max_frames: int = 50000
    topk: int = 3
    z_thr: float = 1.10  # lower -> more shells; higher -> fewer
    min_spacing: int = 1  # boundary index spacing (1 = allow adjacent)


def stride_for(K: int, max_frames: int) -> int:
    return max(1, int(math.ceil(K / max_frames)))


def _mad_1d(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + EPS)


def occ_from_cuts(N: int, cuts: tuple[int, ...]) -> tuple[int, ...]:
    edges = (0,) + tuple(cuts) + (N,)
    return tuple(edges[i + 1] - edges[i] for i in range(len(edges) - 1))


def detect_cuts(r_sorted: np.ndarray, cfg: ShellCfg) -> tuple[tuple[int, ...], list[dict]]:
    """Choose global rank cuts from per-frame sorted radii (K, N)."""
    K2, N = r_sorted.shape
    gaps = r_sorted[:, 1:] - r_sorted[:, :-1]
    g_med = np.median(gaps, axis=0)
    z = (g_med - float(np.median(g_med))) / _mad_1d(g_med)

    topk = int(min(cfg.topk, N - 1))
    ord_gap = np.argsort(-gaps, axis=1)
    ranks = np.empty_like(ord_gap)
    ranks[np.arange(K2)[:, None], ord_gap] = np.arange(N - 1)[None, :]
    freq_topk = np.mean(ranks < topk, axis=0)

    left, right = r_sorted[:, :-1], r_sorted[:, 1:]

    def mad_axis(a: np.ndarray) -> np.ndarray:
        med = np.median(a, axis=0, keepdims=True)
        return np.median(np.abs(a - med), axis=0) + EPS

    d_sep = (np.median(right, axis=0) - np.median(left, axis=0)) / np.sqrt(
        mad_axis(left) ** 2 + mad_axis(right) ** 2
    )

    def center_ok(cuts: tuple[int, ...]) -> bool:
        if not cfg.allow_center_1 or not cuts or cuts[0] != 1:
            return False
        k = int(min(3, N - 1))
        r_in = np.mean(r_sorted[:, 1 : 1 + k], axis=1)
        return float(np.median(r_sorted[:, 0]) / (np.median(r_in) + EPS)) < CENTRE_RATIO_MAX

    def valid_occ(cuts: tuple[int, ...]) -> bool:
        for si, n in enumerate(occ_from_cuts(N, cuts)):
            if si == 0 and n == 1:
                if not center_ok(cuts):
                    return False
                continue
            if n < cfg.n_min:
                return False
        return True

    chosen: list[int] = []
    for j in np.argsort(-z):
        if z[j] < cfg.z_thr:
            break
        b = int(j + 1)
        if any(abs(b - c) < cfg.min_spacing for c in chosen):
            continue
        proposal = tuple(sorted(chosen + [b]))
        if len(proposal) > cfg.max_shells - 1:
            continue
        if valid_occ(proposal):
            chosen.append(b)
    cuts = tuple(sorted(chosen))
    diag = [
        dict(b=b, z=float(z[b - 1]), freq_topk=float(freq_topk[b - 1]), d_sep=float(d_sep[b - 1]))
        for b in cuts
    ]
    return cuts, diag


def sorted_polar(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame radius-sorted radii and angles, (K, N) each."""
    r = np.hypot(X[:, :, 0], X[:, :, 1])
    theta = np.arctan2(X[:, :, 1], X[:, :, 0])
    order = np.argsort(r, axis=1)
    rows = np.arange(X.shape[0])[:, None]
    return r[rows, order], theta[rows, order]


def bond_order_frames(theta_sorted: np.ndarray, cuts: tuple[int, ...]) -> list[np.ndarray]:
    """Per-frame |Phi_n| for each rank shell (list over shells of (K,) arrays)."""
    N = theta_sorted.shape[1]
    edges = (0,) + tuple(cuts) + (N,)
    out = []
    for si in range(len(edges) - 1):
        a, b = edges[si], edges[si + 1]
        n = b - a
        out.append(np.abs(np.mean(np.exp(1j * n * theta_sorted[:, a:b]), axis=1)))
    return out


def exact_random_angle_null(n: int) -> float:
    """E|Phi_n| for n i.i.d. uniform angles (Kluyver's integral)."""
    from scipy.integrate import quad
    from scipy.special import j0

    def f(t: float) -> float:
        return (1.0 - j0(t) ** n) / t**2

    head, _ = quad(f, 0.0, 50.0, limit=2000)
    tail, _ = quad(f, 50.0, 5000.0, limit=20000)
    return (head + tail + 1.0 / 5000.0) / n


# ---------------------------------------------------------------------------
# Held-out polygon shell model for the pair-distance distribution
# ---------------------------------------------------------------------------
QUANTILE_MAX_ELEMS = 5_000_000  # torch.quantile input cap; larger inputs are strided


def drop_nonfinite_frames(X: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(X).reshape(X.shape[0], -1).all(dim=1)
    return X[mask]


def edges_from_all_frames(X: torch.Tensor, q: float = 0.997, nbins: int = 240) -> torch.Tensor:
    """Bin edges on [0, 2 * q-quantile of single-particle radius], as in the original fit."""
    r = torch.linalg.norm(X, dim=-1).reshape(-1)
    r = r[torch.isfinite(r)]
    if r.numel() > QUANTILE_MAX_ELEMS:
        r = r[:: int(math.ceil(r.numel() / QUANTILE_MAX_ELEMS))]
    rmax = float(2.0 * torch.quantile(r.to(torch.float64), q))
    eps = max(rmax, 1.0) * 1e-6 + EPS
    return torch.linspace(0.0, rmax + eps, nbins + 1, dtype=X.dtype)


@torch.no_grad()
def pair_hist_counts(X: torch.Tensor, edges: torch.Tensor, chunk_frames: int = 256) -> torch.Tensor:
    """Unnormalised global pair-distance histogram (counts) over frames X (K, N, 2)."""
    K, N, _ = X.shape
    iu = torch.triu_indices(N, N, 1)
    nbins = len(edges) - 1
    H = torch.zeros(nbins, dtype=torch.float64)
    for s in range(0, K, chunk_frames):
        d = torch.cdist(X[s : s + chunk_frames], X[s : s + chunk_frames])[:, iu[0], iu[1]].reshape(
            -1
        )
        d = d[torch.isfinite(d) & (d < edges[-1])]
        idx = (torch.bucketize(d, edges) - 1).clamp(0, nbins - 1)
        H += torch.bincount(idx, minlength=nbins).to(torch.float64)
    return H


@torch.no_grad()
def fit_shell_params_polygon(
    X_train: torch.Tensor, cuts: tuple[int, ...], occ: tuple[int, ...]
) -> dict[str, list[float]]:
    """Per-shell radial mean/sd and regular-n-gon angular jitter, fitted on training frames."""
    K, N, _ = X_train.shape
    edges = (0,) + tuple(cuts) + (N,)
    r = torch.linalg.norm(X_train, dim=-1)
    th = torch.atan2(X_train[..., 1], X_train[..., 0])
    order = torch.argsort(r, dim=1)
    rows = torch.arange(K)[:, None]
    r_sorted, th_sorted = r[rows, order], th[rows, order]
    mu, sig, sig_th = [], [], []
    for s in range(len(occ)):
        a, b = edges[s], edges[s + 1]
        n = b - a
        rs = r_sorted[:, a:b].reshape(-1)
        mu.append(float(rs.mean()))
        sig.append(float(rs.std(unbiased=True)) if rs.numel() > 1 else 0.0)
        if n <= 1:
            sig_th.append(0.0)
            continue
        ths = torch.sort(th_sorted[:, a:b], dim=1).values
        ideal = (2.0 * math.pi / n) * torch.arange(n, dtype=ths.dtype)[None, :]
        phi = torch.angle(torch.mean(torch.exp(1j * (ths - ideal)), dim=1))[:, None]
        resid = (ths - (ideal + phi) + math.pi) % (2.0 * math.pi) - math.pi
        rmse = torch.sqrt(torch.mean(resid * resid, dim=1))
        sig_th.append(float(torch.sqrt(torch.mean(rmse * rmse))))
    return dict(mu=mu, sigma=sig, sigma_theta=sig_th)


@torch.no_grad()
def simulate_shell_model(
    M: int,
    occ: tuple[int, ...],
    pars: dict[str, list[float]],
    angular_mode: str,
    gen: torch.Generator,
    jitter_scale: float = 1.0,
) -> torch.Tensor:
    """M synthetic frames: Gaussian radii per shell; polygon (+rotation, jitter) or uniform."""
    N = sum(occ)
    X = torch.empty((M, N, 2), dtype=torch.float64)
    col = 0
    for s, n in enumerate(occ):
        r = (
            pars["mu"][s]
            + pars["sigma"][s] * torch.randn((M, n), generator=gen, dtype=torch.float64)
        ).clamp(min=0.0)
        if angular_mode == "uniform" or n <= 2:
            th = 2.0 * math.pi * torch.rand((M, n), generator=gen, dtype=torch.float64)
        else:
            base = (2.0 * math.pi / n) * torch.arange(n, dtype=torch.float64)[None, :]
            rot = 2.0 * math.pi * torch.rand((M, 1), generator=gen, dtype=torch.float64)
            jit = (
                jitter_scale
                * pars["sigma_theta"][s]
                * torch.randn((M, n), generator=gen, dtype=torch.float64)
            )
            th = base + rot + jit
        X[:, col : col + n, 0] = r * torch.cos(th)
        X[:, col : col + n, 1] = r * torch.sin(th)
        col += n
    return X


def cosine_sim(u: Any, v: Any) -> float:
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float("nan") if den == 0.0 else float(np.dot(u, v)) / den
