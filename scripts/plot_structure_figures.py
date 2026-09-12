"""One-body density surfaces and radial/pair-distance densities for Chapter 6.

Restored from the plotting notebook that produced the published figures
(``src/Plottng.ipynb``; the density logic, the thesis colormap and the camera are
unchanged) with three deliberate changes, all from the supervisor review:

  * the vertical axis of the one-body surface is labelled n(x, y) over axes x, y (the
    published files read n(r_1, r_2), which collides with the particle-label convention);
  * the per-omega alpha fading of the radial-density overlays is removed (it was a manual
    override that printed the omega >= 0.1 curves at 50-70 % opacity);
  * figures are rendered at a size at which the thesis style's own font sizes and line
    widths print legibly at the width LaTeX places them; the style itself is untouched.

Run on the machine holding results/tables/ (the |Psi|^2 frame bundles):
    python3.11 scripts/plot_structure_figures.py [--surfaces] [--radial]
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "src" / "Thesis_style.mplstyle"
TABLES = ROOT / "results" / "tables"
SHELL_CSV = ROOT / "results" / "structural_tables" / "shell_summary.csv"
OUT = ROOT / "results" / "figures" / "results"

THESIS_VIRIDIAL = LinearSegmentedColormap.from_list(
    "thesis_viridial", ["#2e003e", "#2f7a7a", "#e8d9bf"], N=256  # violet, teal, ivory
)
THESIS_FACE = (0.9803921568627451, 0.9764705882352941, 0.9647058823529412)

# Published one-body surfaces: (N, omega) -> run directory the figure was drawn from.
SURFACE_RUNS = {
    (2, 1.0): "2/omega_1.00000/20251029_231309",
    (6, 1.0): "6/omega_1.00000/20251030_093439",
    (12, 1.0): "12/omega_1.00000/20251104_194000",
    (2, 0.001): "2/omega_0.00100/20251106_160757",
    (6, 0.001): "6/omega_0.00100/20251106_214930",
    (12, 0.001): "12/omega_0.00100/20251109_110913",
    (20, 1.0): "20/omega_1.00000/20251228_180301",
}
# Printed at one third of the text width (~2 in): 34 pt style labels land near 7 pt.
SURFACE_FIGSIZE = (10.0, 10.0)
# Printed at 0.9 text width (~5.7 in): 34 pt labels near 10 pt, 3.4 pt lines near 1 pt.
RADIAL_FIGSIZE = (19.0, 14.0)
RASTER_DPI = 120  # surface meshes are rasterised (~600 dpi at print size); text stays vector
# The bundles were later capped at 300k frames by a data-cleaning pass, so the 2D histogram
# is noisier than when the figures were first drawn; a slightly wider smoothing (1.5 bins
# instead of 1.0) restores comparable smoothness without changing the structure shown.
SURFACE_SMOOTH_SIGMA = 1.5
OMEGAS = (0.001, 0.01, 0.1, 0.5, 1.0)
LINESTYLES = ("-", "--", "-.", ":")


# --------------------------------------------------------------------------- helpers
def safe_quantile(x: torch.Tensor, q: float, max_elems: int = 5_000_000) -> float:
    x = x[torch.isfinite(x)].flatten()
    if x.numel() > max_elems:
        x = x[:: int(math.ceil(x.numel() / max_elems))]
    return float(torch.quantile(x.to(torch.float64).cpu(), float(q)))


def load_bohr(pt: Path) -> tuple[torch.Tensor, float]:
    b = torch.load(str(pt), map_location="cpu")
    X = b["samples_X_bohr"][..., :2].to(torch.float64)
    return X[torch.isfinite(X).reshape(X.shape[0], -1).all(dim=1)], float(b["omega"])


def pt_in(run_dir: Path) -> Path:
    pts = sorted(run_dir.glob("gr_*.pt"))
    if len(pts) != 1:
        raise RuntimeError(f"expected one bundle in {run_dir}, found {pts}")
    return pts[0]


def shell_summary_bundles() -> dict[tuple[int, float], Path]:
    out = {}
    with SHELL_CSV.open() as f:
        for r in csv.DictReader(f):
            out[(int(r["N"]), float(r["omega"]))] = ROOT / r["pt"].replace("../", "", 1)
    return out


def latest_bundle(n: int, omega: float) -> Path:
    return sorted((TABLES / str(n) / f"omega_{omega:0.5f}").rglob("gr_*.pt"))[-1]


# --------------------------------------------------------------------------- surfaces
def one_body_density_2d(Xb: torch.Tensor, omega: float, nbins: int, q_extent: float):
    xy = (Xb * math.sqrt(omega)).reshape(-1, 2)  # trap units
    xmax = safe_quantile(torch.abs(xy[:, 0]), q_extent)
    ymax = safe_quantile(torch.abs(xy[:, 1]), q_extent)
    H, xe, ye = np.histogram2d(
        xy[:, 0].numpy(), xy[:, 1].numpy(), bins=nbins, range=[[-xmax, xmax], [-ymax, ymax]]
    )
    return xe, ye, (H / H.sum()).T


def plot_surface(
    pt: Path,
    out_pdf: Path,
    nbins: int = 220,
    q_extent: float = 0.999,
    smooth_sigma: float = SURFACE_SMOOTH_SIGMA,
    pad_factor: float = 1.25,
    box_alpha: float = 0.12,
) -> None:
    Xb, omega = load_bohr(pt)
    xe, ye, H = one_body_density_2d(Xb, omega, nbins, q_extent)
    Xc, Yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    H = gaussian_filter(H, sigma=smooth_sigma)
    xlim, ylim = float(Xc.max()) * pad_factor, float(Yc.max()) * pad_factor
    Xe, Ye = np.concatenate(([-xlim], Xc, [xlim])), np.concatenate(([-ylim], Yc, [ylim]))
    H_ext = np.pad(H, pad_width=1, mode="constant", constant_values=0.0)
    X, Y = np.meshgrid(Xe, Ye)

    fig = plt.figure(figsize=SURFACE_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        H_ext,
        rstride=2,
        cstride=2,
        cmap=THESIS_VIRIDIAL,
        alpha=0.96,
        linewidth=0,
        rasterized=True,
    )
    ax.plot_wireframe(
        X, Y, H_ext, rstride=8, cstride=8, color="black", alpha=0.25, linewidth=0.4, rasterized=True
    )
    ax.plot_surface(
        X,
        Y,
        np.zeros_like(H_ext),
        color=THESIS_FACE,
        alpha=1.0,
        zorder=0,
        linewidth=0,
        rasterized=True,
    )
    Hx, Hy = H.sum(axis=0), H.sum(axis=1)
    Hx, Hy = Hx / Hx.max(), Hy / Hy.max()
    zmax = float(H.max())
    ax.plot(
        np.full_like(Ye, -xlim),
        Ye,
        np.concatenate(([0.0], Hy, [0.0])) * zmax,
        color="k",
        lw=1.5,
        alpha=0.9,
    )
    ax.plot(
        Xe,
        np.full_like(Xe, ylim),
        np.concatenate(([0.0], Hx, [0.0])) * zmax,
        color="k",
        lw=1.5,
        alpha=0.9,
    )
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_zlim(0, zmax * 1.6)
    ax.set_xlabel(r"$x$", labelpad=14)
    ax.set_ylabel(r"$y$", labelpad=14)
    ax.zaxis.set_rotate_label(False)
    # The z scale is a per-bin probability with no physical unit; tick marks are kept, their
    # labels dropped, so the axis label no longer collides with them.
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_zticklabels([])
    ax.set_zlabel(r"$n(x,y)$", labelpad=6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.04, top=0.98)
    ax.view_init(elev=42, azim=-55)
    ax.set_box_aspect([1, 1, 1.4])
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_facecolor((*THESIS_FACE, box_alpha))
    ax.yaxis.pane.set_facecolor((*THESIS_FACE, box_alpha))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05, dpi=RASTER_DPI)
    plt.close(fig)
    print(f"[surface] {pt} -> {out_pdf.name} (N={Xb.shape[1]}, omega={omega})", flush=True)


# --------------------------------------------------------------------------- radial
@torch.no_grad()
def rho1_from_samples(
    Xb: torch.Tensor, nbins: int = 420, qR: float = 0.999, smooth_win: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    r = Xb.norm(dim=-1).reshape(-1)
    edges = torch.linspace(0.0, safe_quantile(r, qR), nbins + 1, dtype=r.dtype)
    idx = torch.bucketize(r.clamp_max(edges[-1] - 1e-12), edges) - 1
    H = torch.bincount(idx.clamp(0, nbins - 1), minlength=nbins).to(torch.float64)
    H[-1] = 0.0  # drop edge-touchers
    H = H / H.sum()
    dr = torch.diff(edges)
    rho = H / dr
    rho = F.avg_pool1d(rho.view(1, 1, -1), smooth_win, 1, smooth_win // 2).view(-1)
    rho = rho / torch.sum(rho * dr)
    return (0.5 * (edges[:-1] + edges[1:]))[:-1], rho[:-1]


@torch.no_grad()
def pair_pdf_from_samples(
    Xb: torch.Tensor,
    nbins: int = 420,
    q_rmax: float = 0.997,
    smooth_win: int = 7,
    max_pairs: int = 8_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    K, N, _ = Xb.shape
    r12_max = 2.0 * safe_quantile(Xb.norm(dim=-1).reshape(-1), q_rmax)
    edges = torch.linspace(0.0, r12_max, nbins + 1, dtype=Xb.dtype)
    iu = torch.triu_indices(N, N, 1)
    H = torch.zeros(nbins, dtype=Xb.dtype)
    chunk = max(8, int(64 * 1024 * 1024 // max(N * N * 8, 1)))
    pairs = 0
    for s in range(0, K, chunk):
        d = torch.cdist(Xb[s : s + chunk], Xb[s : s + chunk])[:, iu[0], iu[1]].reshape(-1)
        d = d[torch.isfinite(d)]
        idx = torch.bucketize(d.clamp_max(edges[-1] - 1e-12), edges) - 1
        H += torch.bincount(idx.clamp(0, nbins - 1), minlength=nbins).to(H.dtype)
        pairs += int(d.numel())
        if pairs >= max_pairs:
            break
    H = H / H.sum()
    dr = torch.diff(edges)
    pdf = H / dr
    pdf = F.avg_pool1d(pdf.view(1, 1, -1), smooth_win, 1, smooth_win // 2).view(-1)
    pdf = pdf / torch.sum(pdf * dr)
    return (0.5 * (edges[:-1] + edges[1:]))[:-1], pdf[:-1]


def plot_radial(n: int, bundles: dict[tuple[int, float], Path]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=RADIAL_FIGSIZE)
    styles = cycle(LINESTYLES)
    for om in OMEGAS:
        pt = bundles.get((n, om)) or latest_bundle(n, om)
        Xb, omega = load_bohr(pt)
        s = math.sqrt(omega)  # trap units: r~ = sqrt(omega) r, density rescaled accordingly
        ls = next(styles)
        r1, rho1 = rho1_from_samples(Xb)
        axes[0].plot((r1 * s).numpy(), (rho1 / s).numpy(), ls=ls, label=rf"$\omega={om:g}$")
        r12, p12 = pair_pdf_from_samples(Xb)
        axes[1].plot((r12 * s).numpy(), (p12 / s).numpy(), ls=ls, label=rf"$\omega={om:g}$")
        print(f"[radial] N={n} omega={om}: {pt}", flush=True)
    axes[0].set_ylabel(r"$\rho_1(\tilde r)$")
    axes[0].set_title(rf"One-body radial density for $N={n}$ (trap scale)")
    axes[0].legend()
    axes[1].set_xlabel(r"distance $\tilde r$ (trap units)")
    axes[1].set_ylabel(r"$P(r_{12})$")
    axes[1].set_title(rf"Two-body pair-distance density for $N={n}$ (trap scale)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / f"N{n}_all_densities.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--surfaces", action="store_true")
    ap.add_argument("--radial", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    globals()["OUT"] = a.out
    a.out.mkdir(parents=True, exist_ok=True)
    plt.style.use(str(STYLE))
    bundles = shell_summary_bundles()
    if a.surfaces:
        for (n, om), run in SURFACE_RUNS.items():
            stamp = run.split("/")[-1]
            plot_surface(
                pt_in(TABLES / run), a.out / f"one_body_density_{n}_omega_{om:.5f}_{stamp}.pdf"
            )
        plot_surface(bundles[(20, 0.001)], a.out / "one_body_density_20_omega_0.00100.pdf")
    if a.radial:
        for n in (2, 6, 12):
            plot_radial(n, bundles)


if __name__ == "__main__":
    main()
