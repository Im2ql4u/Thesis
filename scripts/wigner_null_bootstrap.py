"""Pipeline null for the bond order and uncertainty/sensitivity for the polygon Delta-cos.

Answers two supervisor-review requests on the Wigner analysis (both computed on the same
saved |Psi|^2 frame bundles the thesis tables were built from):

  B2  The random-angle null for |Phi_n|, generated through the *same* pipeline: frames
      keep their radii (so the radial-rank shells and cuts are unchanged) and every angle
      is redrawn uniformly. Reported with block-bootstrap errors next to the measured
      |Phi_n| and the exact i.i.d. value.
  B3  Block-bootstrap intervals for Delta-cos = cos_poly - cos_uni of the held-out
      polygon shell model, resampling contiguous blocks of both the training and the
      held-out frames and refitting each replicate, plus the sensitivity of Delta-cos to
      histogram binning, the train/test split, the angular-jitter scale and the
      shell-detection threshold.

Modes:
  --reproduce        sequential pass that re-derives the published Delta-cos values with
                     the original shared generator (seed 7, cells in (N, omega) order).
  --cell I           bootstrap, null and sensitivity for row I of shell_summary.csv.
Run from the repo root on the machine that holds results/tables/.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wigner_structure import (  # noqa: E402
    ShellCfg,
    bond_order_frames,
    cosine_sim,
    detect_cuts,
    drop_nonfinite_frames,
    edges_from_all_frames,
    exact_random_angle_null,
    fit_shell_params_polygon,
    occ_from_cuts,
    pair_hist_counts,
    simulate_shell_model,
    sorted_polar,
    stride_for,
)

SHELL_CSV = ROOT / "results/structural_tables/shell_summary.csv"
FIT_CSV = ROOT / "results/structural_tables_shell_model_fit/shell_model_fit.csv"
OUT_DIR = ROOT / "results/analysis/2026-09-12_wigner_null_bootstrap"

# Original fit settings (src/Analysis2.ipynb, shell-model cell)
TRAIN_FRAC = 0.50
NBINS = 240
Q_EDGES = 0.997
MODEL_FRAMES = 12000
ORIGINAL_SEED = 7

# Uncertainty settings (new)
N_BLOCKS_PHI = 50  # contiguous blocks of strided frames for the |Phi_n| bootstrap
N_BLOCKS_FIT = 40  # contiguous blocks per half for the Delta-cos bootstrap
N_BOOT_PHI = 2000
N_BOOT_FIT = 200
SENS_MODEL_FRAMES = 48000  # larger model sample so variants are not dominated by MC noise
NULL_SEED = 20260912
PIPELINE_Z_THR = 1.45  # z_thr the published shell_summary was produced with
SENS_Z_THR = (1.10, 1.80)  # shell-detection thresholds bracketing it


def parse_tuple(s: str) -> tuple[int, ...]:
    t = ast.literal_eval(s.strip())
    return (int(t),) if isinstance(t, int) else tuple(int(x) for x in t)


def load_rows() -> list[dict]:
    rows = []
    with SHELL_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                dict(
                    N=int(r["N"]),
                    omega=float(r["omega"]),
                    occ=parse_tuple(r["occ"]),
                    cuts=parse_tuple(r["cuts"]),
                    phis=[float(x) for x in r["phis"].split(",")],
                    pt=ROOT / r["pt"].replace("../", "", 1),
                )
            )
    rows.sort(key=lambda r: (r["N"], r["omega"]))
    return rows


def load_frames(pt: Path) -> np.ndarray:
    b = torch.load(str(pt), map_location="cpu")
    return b["samples_X_bohr"][..., :2].to(torch.float64).numpy()


def block_boot(x: np.ndarray, n_blocks: int, n_boot: int, rng: np.random.Generator) -> float:
    """Standard error of mean(x) from a contiguous-block bootstrap."""
    blocks = np.array_split(x, n_blocks)
    means = np.array([b.mean() for b in blocks])
    sizes = np.array([len(b) for b in blocks], dtype=float)
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    boot = (means[idx] * sizes[idx]).sum(1) / sizes[idx].sum(1)
    return float(boot.std(ddof=1))


# ---------------------------------------------------------------------------
def bond_order_and_null(X: np.ndarray, cuts: tuple[int, ...], rng: np.random.Generator) -> dict:
    Xs = X[:: stride_for(X.shape[0], ShellCfg().max_frames)]
    r_s, th_s = sorted_polar(Xs)
    cuts_re, _ = detect_cuts(r_s, ShellCfg(z_thr=PIPELINE_Z_THR))

    # Null through the same pipeline: same radii, i.i.d. uniform angles.
    u = rng.uniform(0.0, 2.0 * math.pi, size=Xs.shape[:2])
    rad = np.hypot(Xs[:, :, 0], Xs[:, :, 1])
    Xn = np.stack([rad * np.cos(u), rad * np.sin(u)], axis=-1)
    r_n, th_n = sorted_polar(Xn)
    cuts_null, _ = detect_cuts(r_n, ShellCfg(z_thr=PIPELINE_Z_THR))

    occ = occ_from_cuts(X.shape[1], cuts)
    phi = bond_order_frames(th_s, cuts)
    phi0 = bond_order_frames(th_n, cuts)
    shells = []
    for n, f, f0 in zip(occ, phi, phi0, strict=False):
        if n <= 1:
            continue
        m, se = float(f.mean()), block_boot(f, N_BLOCKS_PHI, N_BOOT_PHI, rng)
        m0, se0 = float(f0.mean()), block_boot(f0, N_BLOCKS_PHI, N_BOOT_PHI, rng)
        exact = exact_random_angle_null(n)
        ratio = m / m0
        ratio_se = ratio * math.sqrt((se / m) ** 2 + (se0 / m0) ** 2)
        shells.append(
            dict(
                n=n,
                phi=m,
                phi_se=se,
                null_pipeline=m0,
                null_pipeline_se=se0,
                null_exact=exact,
                ratio_pipeline=ratio,
                ratio_pipeline_se=ratio_se,
                ratio_exact=m / exact,
                ratio_exact_se=se / exact,
            )
        )
    return dict(
        frames_used=int(Xs.shape[0]),
        cuts_redetected=list(cuts_re),
        cuts_null=list(cuts_null),
        shells=shells,
    )


# ---------------------------------------------------------------------------
def fit_and_score(
    X_train: torch.Tensor,
    H_test: torch.Tensor,
    cuts: tuple[int, ...],
    occ: tuple[int, ...],
    edges: torch.Tensor,
    gen: torch.Generator,
    model_frames: int,
    jitter_scale: float = 1.0,
) -> tuple[float, float]:
    pars = fit_shell_params_polygon(X_train, cuts, occ)
    Hp = pair_hist_counts(
        simulate_shell_model(model_frames, occ, pars, "polygon", gen, jitter_scale), edges
    )
    Hu = pair_hist_counts(
        simulate_shell_model(model_frames, occ, pars, "uniform", gen, jitter_scale), edges
    )
    return cosine_sim(H_test, Hp), cosine_sim(H_test, Hu)


def split_halves(Xt: torch.Tensor, how: str) -> tuple[torch.Tensor, torch.Tensor]:
    K = Xt.shape[0]
    k = max(1, int(TRAIN_FRAC * K))
    if how == "first":
        return Xt[:k], Xt[k:]
    if how == "swapped":
        return Xt[K - k :], Xt[: K - k]
    if how == "interleaved":  # alternate contiguous blocks between train and test
        blocks = torch.tensor_split(Xt, 2 * N_BLOCKS_FIT)
        return torch.cat(blocks[0::2]), torch.cat(blocks[1::2])
    raise ValueError(how)


def delta_cos_uncertainty(
    X: np.ndarray, cuts: tuple[int, ...], occ: tuple[int, ...], rng: np.random.Generator
) -> dict:
    Xt = drop_nonfinite_frames(torch.from_numpy(X))
    edges = edges_from_all_frames(Xt, Q_EDGES, NBINS)
    X_train, X_test = split_halves(Xt, "first")

    # Block bootstrap: resample contiguous blocks of both halves, refit, fresh model seed.
    test_blocks = [pair_hist_counts(b, edges) for b in torch.tensor_split(X_test, N_BLOCKS_FIT)]
    train_blocks = torch.tensor_split(X_train, N_BLOCKS_FIT)
    boot_list: list[float] = []
    for rep in range(N_BOOT_FIT):
        it = rng.integers(0, N_BLOCKS_FIT, N_BLOCKS_FIT)
        itr = rng.integers(0, N_BLOCKS_FIT, N_BLOCKS_FIT)
        H_test = sum(test_blocks[j] for j in it)
        Xtr = torch.cat([train_blocks[j] for j in itr])
        cp, cu = fit_and_score(
            Xtr, H_test, cuts, occ, edges, torch.Generator().manual_seed(1000 + rep), MODEL_FRAMES
        )
        boot_list.append(cp - cu)
    boot = np.array(boot_list)

    # Sensitivity: each variant scored with a large model sample and a common seed.
    def variant(
        nbins: int = NBINS,
        split: str = "first",
        jitter: float = 1.0,
        cuts_v: tuple[int, ...] = cuts,
    ) -> dict:
        e = edges_from_all_frames(Xt, Q_EDGES, nbins)
        Xa, Xb = split_halves(Xt, split)
        occ_v = occ_from_cuts(Xt.shape[1], cuts_v)
        cp, cu = fit_and_score(
            Xa,
            pair_hist_counts(Xb, e),
            cuts_v,
            occ_v,
            e,
            torch.Generator().manual_seed(ORIGINAL_SEED),
            SENS_MODEL_FRAMES,
            jitter,
        )
        return dict(
            nbins=nbins,
            split=split,
            jitter=jitter,
            occ=list(occ_v),
            dcos=cp - cu,
            cos_poly=cp,
            cos_uni=cu,
        )

    Xs = X[:: stride_for(X.shape[0], ShellCfg().max_frames)]
    r_s, _ = sorted_polar(Xs)
    sens = [variant()]
    sens += [variant(nbins=120), variant(nbins=480)]
    sens += [variant(split="swapped"), variant(split="interleaved")]
    sens += [variant(jitter=0.75), variant(jitter=1.25)]
    for z in SENS_Z_THR:
        cz, _ = detect_cuts(r_s, ShellCfg(z_thr=z))
        v = variant(cuts_v=cz)
        v["z_thr"] = z
        sens.append(v)

    return dict(
        boot_mean=float(boot.mean()),
        boot_sd=float(boot.std(ddof=1)),
        boot_q025=float(np.quantile(boot, 0.025)),
        boot_q975=float(np.quantile(boot, 0.975)),
        boot=boot.tolist(),
        sensitivity=sens,
    )


# ---------------------------------------------------------------------------
def reproduce(rows: list[dict]) -> None:
    """Re-derive the published Delta-cos with the original shared generator."""
    published = {}
    with FIT_CSV.open() as f:
        for r in csv.DictReader(f):
            published[(int(r["N"]), float(r["omega"]))] = float(r["dcos"])
    gen = torch.Generator().manual_seed(ORIGINAL_SEED)
    out = []
    for r in rows:
        Xt = drop_nonfinite_frames(torch.from_numpy(load_frames(r["pt"])))
        edges = edges_from_all_frames(Xt, Q_EDGES, NBINS)
        Xa, Xb = split_halves(Xt, "first")
        cp, cu = fit_and_score(
            Xa, pair_hist_counts(Xb, edges), r["cuts"], r["occ"], edges, gen, MODEL_FRAMES
        )
        pub = published[(r["N"], r["omega"])]
        out.append(dict(N=r["N"], omega=r["omega"], dcos=cp - cu, published=pub))
        print(f"N={r['N']:2d} w={r['omega']:<6} dcos={cp - cu:.6f} published={pub:.6f}", flush=True)
    (OUT_DIR / "reproduce.json").write_text(json.dumps(out, indent=1))


def run_cell(rows: list[dict], i: int) -> None:
    r = rows[i]
    rng = np.random.default_rng(NULL_SEED + i)
    X = load_frames(r["pt"])
    res = dict(
        N=r["N"],
        omega=r["omega"],
        occ=list(r["occ"]),
        cuts=list(r["cuts"]),
        phis_published=r["phis"],
        frames=int(X.shape[0]),
        pt=str(r["pt"]),
    )
    res["bond_order"] = bond_order_and_null(X, r["cuts"], rng)
    res["delta_cos"] = delta_cos_uncertainty(X, r["cuts"], r["occ"], rng)
    (OUT_DIR / f"cell_N{r['N']}_w{r['omega']}.json").write_text(json.dumps(res, indent=1))
    dc = res["delta_cos"]
    print(
        f"N={r['N']} w={r['omega']}: dcos boot {dc['boot_mean']:.4f} "
        f"[{dc['boot_q025']:.4f}, {dc['boot_q975']:.4f}]",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reproduce", action="store_true")
    g.add_argument("--cell", type=int)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if a.reproduce:
        reproduce(rows)
    else:
        run_cell(rows, a.cell)


if __name__ == "__main__":
    main()
