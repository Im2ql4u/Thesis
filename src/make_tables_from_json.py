#!/usr/bin/env python3
"""
make_tables_from_json.py

Scan analysis JSON files (from run_compact_analysis) inside a directory and emit a
single LaTeX file containing multiple structured, grouped tables.

Key goals:
- Group rows by N with visible separators (\\midrule + \\addlinespace).
- Keep tables compact but not blind: include the most informative diagnostics.
- Add a few “tail/robustness” signals when present (energy percentiles / max).
- Add “effect size” columns: ΔE/E and SE ratio (BF/noBF).
- Avoid pandas .to_latex() entirely (compat with older pandas).
- Avoid bad labels like \\label{tab:backflow\\_ctnn}; underscores are fine in labels,
  but *do not escape them*.

Expected JSON layout (robust to missing keys):
report["system"]["N"], report["system"]["omega"]
report["mcmc"]["acc_noBF"]["mix"], report["mcmc"]["acc_BF"]["mix"]
report["energy"]["noBF"]["mean"], ["se"], ["std"], maybe ["percentiles"], ["max"]
report["energy"]["withBF"]["mean"], ["se"], ["std"], maybe ["percentiles"], ["max"]
report["pinn"][...], report["backflow"][...]

Run:
  python make_tables_from_json.py --dir ../results/tables/new --out summary_tables.tex

Then in LaTeX:
  \\usepackage{booktabs}
  \\usepackage{adjustbox}
  ...
  \\input{path/to/summary_tables.tex}
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# -------------------------
# Config: grid + ordering
# -------------------------
N_LIST = [2, 6, 12]
OMEGA_LIST = [0.001, 0.01, 0.1, 0.5, 1.0]
PLACEHOLDER_MISSING = [(2, 0.01)]  # user explicitly wants this inserted


# -------------------------
# Robust JSON helpers
# -------------------------
def _get(d: Any, path: list[Any], default: Any = None) -> Any:
    """Nested get with mixed dict keys / list indices. Returns default if missing."""
    cur = d
    for p in path:
        try:
            if isinstance(p, int):
                cur = cur[p]
            else:
                cur = cur.get(p)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _isfinite(x: Any) -> bool:
    try:
        return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def _as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


# -------------------------
# Formatting for LaTeX
# -------------------------
def fmt_omega(w: Any) -> str:
    v = _as_float(w)
    if v is None:
        return ""
    # match your tables: 0.001, 0.010, 0.100, 0.500, 1.000
    if v < 1.0:
        return f"{v:.3f}"
    return f"{v:.3f}"


def fmt_int(x: Any) -> str:
    try:
        return str(int(x))
    except Exception:
        return ""


def fmt_float(x: Any, *, decimals: int = 3, sci_small: float = 1e-3, sci_big: float = 1e4) -> str:
    v = _as_float(x)
    if v is None:
        return ""
    av = abs(v)
    if (av > 0 and av < sci_small) or av >= sci_big:
        # 3 sig figs in scientific, like 2.964e-05
        return f"{v:.3e}"
    return f"{v:.{decimals}f}"


def fmt_ratio(x: Any, *, decimals: int = 3) -> str:
    v = _as_float(x)
    if v is None:
        return ""
    return f"{v:.{decimals}f}"


def latex_escape_text(s: str) -> str:
    """Escape for text cells (not math). We keep this minimal."""
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def label_safe(s: str) -> str:
    """
    Labels should contain NO backslashes.
    Underscores are fine in labels (LaTeX allows them there).
    """
    s = str(s)
    return s.replace("\\", "").strip()


# -------------------------
# Derived metrics
# -------------------------
def combined_se(se_a: float | None, se_b: float | None) -> float | None:
    if se_a is None or se_b is None:
        return None
    return math.sqrt(se_a * se_a + se_b * se_b)


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    if den == 0:
        return None
    return num / den


# -------------------------
# Row dataclasses
# -------------------------
@dataclass
class RowEnergy:
    N: int
    omega: float
    acc_mix_noBF: float | None
    acc_mix_BF: float | None
    E_noBF: float | None
    SE_noBF: float | None
    E_BF: float | None
    SE_BF: float | None
    dE: float | None
    dE_over_SE: float | None
    dE_over_E: float | None
    SE_ratio: float | None


@dataclass
class RowPINN:
    N: int
    omega: float
    eff_rank: float | None
    expvar1: float | None
    expvar2: float | None
    dphi: float | None
    dpsi: float | None
    dex: float | None
    R2_rmean: float | None
    R2_rvar: float | None
    nf_grad_share_1pct: float | None


@dataclass
class RowBackflow:
    N: int
    omega: float
    bf_type: str
    eff_rank_dx: float | None
    eff_rank_frac: float | None  # eff_rank_dx / (2N)
    nf_dx2_share_1pct: float | None
    radial_frac: float | None
    tangential_frac: float | None
    dv_to_e: float | None
    de_to_v: float | None


@dataclass
class RowEnergyTails:
    N: int
    omega: float
    # noBF
    p50_no: float | None
    p90_no: float | None
    p99_no: float | None
    p999_no: float | None
    max_no: float | None
    # BF
    p50_bf: float | None
    p90_bf: float | None
    p99_bf: float | None
    p999_bf: float | None
    max_bf: float | None
    # variance ratio proxy
    std_ratio_bf_over_no: float | None


# -------------------------
# Parsing one JSON report
# -------------------------
def parse_report(obj: dict[str, Any]) -> tuple[int | None, float | None, dict[str, Any]]:
    report = obj.get("report", obj)  # support either wrapped or raw report dict

    N = _as_float(_get(report, ["system", "N"]))
    omega = _as_float(_get(report, ["system", "omega"]))
    N_i = int(N) if N is not None else None
    return N_i, omega, report


def parse_energy_row(N: int, omega: float, report: dict[str, Any]) -> RowEnergy:
    acc_no = _as_float(_get(report, ["mcmc", "acc_noBF", "mix"]))
    acc_bf = _as_float(_get(report, ["mcmc", "acc_BF", "mix"]))

    E_no = _as_float(_get(report, ["energy", "noBF", "mean"]))
    SE_no = _as_float(_get(report, ["energy", "noBF", "se"]))

    E_bf = _as_float(_get(report, ["energy", "withBF", "mean"]))
    SE_bf = _as_float(_get(report, ["energy", "withBF", "se"]))

    dE = (E_bf - E_no) if (E_bf is not None and E_no is not None) else None
    SEc = combined_se(SE_no, SE_bf)
    dE_over_SE = safe_div(dE, SEc)
    dE_over_E = safe_div(dE, E_no)
    se_ratio = safe_div(SE_bf, SE_no)

    return RowEnergy(
        N=N,
        omega=omega,
        acc_mix_noBF=acc_no,
        acc_mix_BF=acc_bf,
        E_noBF=E_no,
        SE_noBF=SE_no,
        E_BF=E_bf,
        SE_BF=SE_bf,
        dE=dE,
        dE_over_SE=dE_over_SE,
        dE_over_E=dE_over_E,
        SE_ratio=se_ratio,
    )


def parse_pinn_row(N: int, omega: float, report: dict[str, Any]) -> RowPINN:
    eff = _as_float(_get(report, ["pinn", "rho_in_pca_eff_rank"]))
    exp1 = _as_float(_get(report, ["pinn", "rho_in_expvar_top8", 0]))
    exp2 = _as_float(_get(report, ["pinn", "rho_in_expvar_top8", 1]))

    dphi = _as_float(_get(report, ["pinn", "branch_ablation", "phi"]))
    dpsi = _as_float(_get(report, ["pinn", "branch_ablation", "psi"]))
    dex = _as_float(_get(report, ["pinn", "branch_ablation", "extras"]))

    R2_mean = _as_float(_get(report, ["pinn", "probes", "R2", 0]))
    R2_var = _as_float(_get(report, ["pinn", "probes", "R2", 1]))

    # NF grad share (1%): report["pinn"]["near_field_grad_share"] is list of rows, pick q=0.01
    nf_rows = _get(report, ["pinn", "near_field_grad_share"], default=[])
    nf1 = None
    if isinstance(nf_rows, list):
        for r in nf_rows:
            q = _as_float(_get(r, ["q"]))
            if q is not None and abs(q - 0.01) < 1e-9:
                nf1 = _as_float(_get(r, ["share"]))
                break

    return RowPINN(
        N=N,
        omega=omega,
        eff_rank=eff,
        expvar1=exp1,
        expvar2=exp2,
        dphi=dphi,
        dpsi=dpsi,
        dex=dex,
        R2_rmean=R2_mean,
        R2_rvar=R2_var,
        nf_grad_share_1pct=nf1,
    )


def parse_backflow_row(N: int, omega: float, report: dict[str, Any]) -> RowBackflow:
    bf_kind = _get(report, ["backflow", "bf_kind"], default="")
    if bf_kind is None:
        bf_kind = ""
    bf_kind = str(bf_kind)

    eff_dx = _as_float(_get(report, ["backflow", "pca_dx", "eff_rank"]))
    eff_frac = safe_div(eff_dx, 2.0 * float(N)) if eff_dx is not None else None

    # NF ||dx||^2 share (1%): report["backflow"]["near_field_dx2"]["rows"] list
    nf_rows = _get(report, ["backflow", "near_field_dx2", "rows"], default=[])
    nf1 = None
    if isinstance(nf_rows, list):
        for r in nf_rows:
            q = _as_float(_get(r, ["q"]))
            if q is not None and abs(q - 0.01) < 1e-9:
                nf1 = _as_float(_get(r, ["share"]))
                break

    radial = _as_float(
        _get(report, ["backflow", "probes_dx", "pc1_decomposition", "radial_frac_of_rel"])
    )
    tang = _as_float(
        _get(report, ["backflow", "probes_dx", "pc1_decomposition", "tangential_frac_of_rel"])
    )

    dv = _as_float(
        _get(report, ["backflow", "ctnn_transport_ablations", "mean_abs_change_no_v_to_e"])
    )
    de = _as_float(
        _get(report, ["backflow", "ctnn_transport_ablations", "mean_abs_change_no_e_to_v"])
    )

    return RowBackflow(
        N=N,
        omega=omega,
        bf_type=bf_kind,
        eff_rank_dx=eff_dx,
        eff_rank_frac=eff_frac,
        nf_dx2_share_1pct=nf1,
        radial_frac=radial,
        tangential_frac=tang,
        dv_to_e=dv,
        de_to_v=de,
    )


def parse_energy_tails_row(N: int, omega: float, report: dict[str, Any]) -> RowEnergyTails:
    """
    Optional: if your analysis JSON includes percentiles/max under energy blocks.
    Expected flexible formats:
      energy.noBF.percentiles.p50 / p90 / p99 / p999, energy.noBF.max
      or percentiles as dict with keys "0.5", "0.9", ...
    If absent, fields remain blank in table.
    """

    def get_p(block: str, key: str) -> float | None:
        # try percentiles dict
        p = _get(report, ["energy", block, "percentiles", key])
        if p is None:
            # try numeric percentile keys like "0.5"
            alt = {"p50": "0.5", "p90": "0.9", "p99": "0.99", "p999": "0.999"}.get(key)
            if alt is not None:
                p = _get(report, ["energy", block, "percentiles", alt])
        return _as_float(p)

    p50_no = get_p("noBF", "p50")
    p90_no = get_p("noBF", "p90")
    p99_no = get_p("noBF", "p99")
    p999_no = get_p("noBF", "p999")
    max_no = _as_float(_get(report, ["energy", "noBF", "max"]))

    p50_bf = get_p("withBF", "p50")
    p90_bf = get_p("withBF", "p90")
    p99_bf = get_p("withBF", "p99")
    p999_bf = get_p("withBF", "p999")
    max_bf = _as_float(_get(report, ["energy", "withBF", "max"]))

    std_no = _as_float(_get(report, ["energy", "noBF", "std"]))
    std_bf = _as_float(_get(report, ["energy", "withBF", "std"]))
    std_ratio = safe_div(std_bf, std_no)

    return RowEnergyTails(
        N=N,
        omega=omega,
        p50_no=p50_no,
        p90_no=p90_no,
        p99_no=p99_no,
        p999_no=p999_no,
        max_no=max_no,
        p50_bf=p50_bf,
        p90_bf=p90_bf,
        p99_bf=p99_bf,
        p999_bf=p999_bf,
        max_bf=max_bf,
        std_ratio_bf_over_no=std_ratio,
    )


# -------------------------
# Table rendering (manual)
# -------------------------
def render_table(
    *,
    rows: list[list[str]],
    col_spec: str,
    header: list[str],
    caption: str,
    label: str,
    table_env: str = "table",
    placement: str = "t",
    tabcolsep: int = 4,
    arraystretch: float = 1.15,
    small: bool = True,
    adjustbox: bool = True,
) -> str:
    """
    rows: already include midrules/addlinespace commands as raw LaTeX lines
          OR plain row lists; here we assume it's already lines (strings).
    """
    label = label_safe(label)
    caption = caption.strip()

    pre = []
    pre.append(f"\\begin{{{table_env}}}[{placement}]")
    pre.append("\\centering")
    if small:
        pre.append("\\small")
    pre.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}")
    pre.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
    if adjustbox:
        pre.append("\\begin{adjustbox}{max width=\\textwidth}")
    pre.append(f"\\begin{{tabular}}{{{col_spec}}}")
    pre.append("\\toprule")
    pre.append(" & ".join(header) + " \\\\")
    pre.append("\\midrule")

    post = []
    post.append("\\bottomrule")
    post.append("\\end{tabular}")
    if adjustbox:
        post.append("\\end{adjustbox}")
    pre.extend(rows)
    pre.extend(post)
    pre.append(f"\\caption{{{caption}}}")
    pre.append(f"\\label{{{label}}}")
    pre.append(f"\\end{{{table_env}}}")
    return "\n".join(pre) + "\n\n"


def grouped_rows_to_latex_lines(
    *,
    N_to_rows: dict[int, list[list[str]]],
    add_space: bool = True,
) -> list[str]:
    """
    Convert rows grouped by N into LaTeX lines with \\midrule separators.
    Each row is list of cell strings.
    """
    lines: list[str] = []
    first_group = True
    for N in N_LIST:
        group = N_to_rows.get(N, [])
        if not group:
            continue
        if not first_group:
            lines.append("\\midrule")
            if add_space:
                lines.append("\\addlinespace[2pt]")
        first_group = False
        for r in group:
            lines.append(" & ".join(r) + " \\\\")
    return lines


# -------------------------
# Build tables from parsed grid
# -------------------------
def build_tables(
    grid: dict[tuple[int, float], dict[str, Any]],
    *,
    use_table_star: bool,
) -> str:
    table_env = "table*" if use_table_star else "table"

    # --- Parse rows ---
    energy_rows: list[RowEnergy] = []
    pinn_rows: list[RowPINN] = []
    bf_rows: list[RowBackflow] = []
    tail_rows: list[RowEnergyTails] = []

    for N in N_LIST:
        for w in OMEGA_LIST:
            rep = grid.get((N, w))
            if rep is None:
                # placeholder row (blank)
                energy_rows.append(parse_energy_row(N, w, {}))
                pinn_rows.append(parse_pinn_row(N, w, {}))
                bf_rows.append(parse_backflow_row(N, w, {}))
                tail_rows.append(parse_energy_tails_row(N, w, {}))
            else:
                energy_rows.append(parse_energy_row(N, w, rep))
                pinn_rows.append(parse_pinn_row(N, w, rep))
                bf_rows.append(parse_backflow_row(N, w, rep))
                tail_rows.append(parse_energy_tails_row(N, w, rep))

    # --- Table 1: Energy + MCMC (with effect sizes) ---
    # Keep your original structure, plus two “robustness” columns:
    #   ΔE/E_noBF and SE_BF/SE_noBF
    t1_header = [
        "$\\bm{N}$",
        "$\\bm{\\omega}$",
        "\\textbf{acc.~mix (noBF)}",
        "\\textbf{acc.~mix (BF)}",
        "\\textbf{$E$ (noBF)}",
        "\\textbf{SE}",
        "\\textbf{$E$ (BF)}",
        "\\textbf{SE}",
        "\\textbf{$\\Delta E$}",
        "\\textbf{$\\Delta E/\\mathrm{SE}$}",
        "\\textbf{$\\Delta E/E$}",
        "\\textbf{SE ratio}",
    ]
    t1_spec = "ll" + "r" * (len(t1_header) - 2)

    t1_groups: dict[int, list[list[str]]] = {N: [] for N in N_LIST}
    for r in energy_rows:
        row = [
            fmt_int(r.N),
            fmt_omega(r.omega),
            fmt_float(r.acc_mix_noBF, decimals=3),
            fmt_float(r.acc_mix_BF, decimals=3),
            fmt_float(r.E_noBF, decimals=3, sci_small=1e-6),
            fmt_float(r.SE_noBF, decimals=3, sci_small=1e-3),
            fmt_float(r.E_BF, decimals=3, sci_small=1e-6),
            fmt_float(r.SE_BF, decimals=3, sci_small=1e-3),
            fmt_float(r.dE, decimals=3, sci_small=1e-3),
            fmt_float(r.dE_over_SE, decimals=3, sci_small=1e-3),
            fmt_float(r.dE_over_E, decimals=3, sci_small=1e-3),
            fmt_float(r.SE_ratio, decimals=3, sci_small=1e-3),
        ]
        t1_groups[r.N].append(row)

    t1_lines = grouped_rows_to_latex_lines(N_to_rows=t1_groups)
    t1 = render_table(
        rows=t1_lines,
        col_spec=t1_spec,
        header=t1_header,
        caption="Energy and MCMC diagnostics (grouped by $N$). Includes effect-size columns $\\Delta E/E$ and the SE ratio (BF/noBF).",
        label="tab:energy_mcmc",
        table_env=table_env,
    )

    # --- Table 2: PINN representation (compact but informative) ---
    t2_header = [
        "$\\bm{N}$",
        "$\\bm{\\omega}$",
        "\\textbf{eff-rank$(\\rho_{\\mathrm{in}})$}",
        "\\textbf{$\\mathrm{expvar}_1$}",
        "\\textbf{$\\mathrm{expvar}_2$}",
        "\\textbf{$\\lvert \\Delta \\rho \\rvert_{\\phi}$}",
        "\\textbf{$\\lvert \\Delta \\rho \\rvert_{\\psi}$}",
        "\\textbf{$\\lvert \\Delta \\rho \\rvert_{\\mathrm{ex}}$}",
        "\\textbf{$R^{2}\\!\\left(r_{\\mathrm{mean}}\\right)$}",
        "\\textbf{$R^{2}\\!\\left(r_{\\mathrm{var}}\\right)$}",
        "\\textbf{NF grad share (1\\%)}",
    ]
    t2_spec = "ll" + "r" * (len(t2_header) - 2)

    t2_groups: dict[int, list[list[str]]] = {N: [] for N in N_LIST}
    for r in pinn_rows:
        row = [
            fmt_int(r.N),
            fmt_omega(r.omega),
            fmt_float(r.eff_rank, decimals=3),
            fmt_float(r.expvar1, decimals=3),
            fmt_float(r.expvar2, decimals=3),
            fmt_float(r.dphi, decimals=3, sci_small=1e-3),
            fmt_float(r.dpsi, decimals=3, sci_small=1e-3),
            fmt_float(r.dex, decimals=3, sci_small=1e-3),
            fmt_float(r.R2_rmean, decimals=3),
            fmt_float(r.R2_rvar, decimals=3),
            fmt_float(r.nf_grad_share_1pct, decimals=3, sci_small=1e-3),
        ]
        t2_groups[r.N].append(row)

    t2_lines = grouped_rows_to_latex_lines(N_to_rows=t2_groups)
    t2 = render_table(
        rows=t2_lines,
        col_spec=t2_spec,
        header=t2_header,
        caption="PINN / Jastrow representation diagnostics. Effective rank and leading explained variances summarize geometry; branch ablations and probes summarize what the representation encodes.",
        label="tab:pinn_repr",
        table_env=table_env,
    )

    # --- Table 3: Backflow / CTNN diagnostics ---
    # Add eff-rank fraction (= eff-rank/(2N)) so the “near full dimension at N=12” point is explicit.
    t3_header = [
        "$\\bm{N}$",
        "$\\bm{\\omega}$",
        "\\textbf{BF type}",
        "\\textbf{eff-rank$(\\Delta x)$}",
        "\\textbf{eff-rank frac}",
        "\\textbf{NF $\\lVert \\Delta x \\rVert^{2}$ share (1\\%)}",
        "\\textbf{radial frac}",
        "\\textbf{tangential frac}",
        "\\textbf{$\\lvert \\Delta x \\rvert$ change (no $v\\!\\to\\!e$)}",
        "\\textbf{$\\lvert \\Delta x \\rvert$ change (no $e\\!\\to\\!v$)}",
    ]
    t3_spec = "lll" + "r" * (len(t3_header) - 3)

    t3_groups: dict[int, list[list[str]]] = {N: [] for N in N_LIST}
    for r in bf_rows:
        bf_type = latex_escape_text(r.bf_type) if r.bf_type else ""
        row = [
            fmt_int(r.N),
            fmt_omega(r.omega),
            bf_type,
            fmt_float(r.eff_rank_dx, decimals=3),
            fmt_float(r.eff_rank_frac, decimals=3),
            fmt_float(r.nf_dx2_share_1pct, decimals=3, sci_small=1e-3),
            fmt_float(r.radial_frac, decimals=3),
            fmt_float(r.tangential_frac, decimals=3),
            fmt_float(r.dv_to_e, decimals=3, sci_small=1e-3),
            fmt_float(r.de_to_v, decimals=3, sci_small=1e-3),
        ]
        t3_groups[r.N].append(row)

    t3_lines = grouped_rows_to_latex_lines(N_to_rows=t3_groups)
    t3 = render_table(
        rows=t3_lines,
        col_spec=t3_spec,
        header=t3_header,
        caption="Backflow / CTNN diagnostics (if present). The eff-rank fraction is eff-rank$(\\Delta x)/(2N)$, i.e. fraction of the maximum coordinate-shift dimension used.",
        label="tab:backflow_ctnn",
        table_env=table_env,
    )

    # --- Table 4: Optional energy tail / stability diagnostics ---
    # Only useful if your JSON actually stores percentiles/max; otherwise mostly blank.
    # Still valuable because once you add percentiles to the analysis JSON, this table becomes automatic.
    t4_header = [
        "$\\bm{N}$",
        "$\\bm{\\omega}$",
        "\\textbf{$E_L$ p50 (noBF)}",
        "\\textbf{p90}",
        "\\textbf{p99}",
        "\\textbf{p999}",
        "\\textbf{max}",
        "\\textbf{$E_L$ p50 (BF)}",
        "\\textbf{p90}",
        "\\textbf{p99}",
        "\\textbf{p999}",
        "\\textbf{max}",
        "\\textbf{std ratio (BF/noBF)}",
    ]
    t4_spec = "ll" + "r" * (len(t4_header) - 2)

    t4_groups: dict[int, list[list[str]]] = {N: [] for N in N_LIST}
    for r in tail_rows:
        row = [
            fmt_int(r.N),
            fmt_omega(r.omega),
            fmt_float(r.p50_no, decimals=3, sci_small=1e-3),
            fmt_float(r.p90_no, decimals=3, sci_small=1e-3),
            fmt_float(r.p99_no, decimals=3, sci_small=1e-3),
            fmt_float(r.p999_no, decimals=3, sci_small=1e-3),
            fmt_float(r.max_no, decimals=3, sci_small=1e-3),
            fmt_float(r.p50_bf, decimals=3, sci_small=1e-3),
            fmt_float(r.p90_bf, decimals=3, sci_small=1e-3),
            fmt_float(r.p99_bf, decimals=3, sci_small=1e-3),
            fmt_float(r.p999_bf, decimals=3, sci_small=1e-3),
            fmt_float(r.max_bf, decimals=3, sci_small=1e-3),
            fmt_float(r.std_ratio_bf_over_no, decimals=3, sci_small=1e-3),
        ]
        t4_groups[r.N].append(row)

    t4_lines = grouped_rows_to_latex_lines(N_to_rows=t4_groups)
    t4 = render_table(
        rows=t4_lines,
        col_spec=t4_spec,
        header=t4_header,
        caption="Local-energy tail diagnostics (optional; filled only if percentiles/max are present in the JSON). Useful for spotting rare high-$E_L$ spikes and variance blow-ups under backflow.",
        label="tab:energy_tails",
        table_env=table_env,
    )

    # --- Full LaTeX output ---
    out = []
    out.append("% Auto-generated by make_tables_from_json.py")
    out.append("% Required packages (in your main .tex preamble):")
    out.append("%   \\usepackage{booktabs}")
    out.append("%   \\usepackage{adjustbox}")
    out.append("%")
    out.append(t1)
    out.append(t2)
    out.append(t3)
    out.append(t4)
    return "\n".join(out)


# -------------------------
# Load all JSONs from dir
# -------------------------
def load_reports(dir_path: Path) -> dict[tuple[int, float], dict[str, Any]]:
    files = sorted(dir_path.glob("*.json"))
    grid: dict[tuple[int, float], dict[str, Any]] = {}

    for fp in files:
        try:
            obj = json.loads(fp.read_text())
        except Exception:
            # skip non-json or broken files
            continue

        N, omega, report = parse_report(obj)
        if N is None or omega is None:
            continue

        # normalize omega to one of the canonical list values (tolerant)
        w_norm = None
        for w in OMEGA_LIST:
            if abs(float(omega) - float(w)) < 1e-9:
                w_norm = w
                break
        if w_norm is None:
            # if user generates other omegas, keep them but rounding may break grouping.
            # Here we round to 3 decimals which matches your grid.
            w_norm = round(float(omega), 3)

        grid[(int(N), float(w_norm))] = report

    # insert placeholder keys explicitly requested (N=2, omega=0.01)
    for N, w in PLACEHOLDER_MISSING:
        grid.setdefault((N, w), None)

    # also ensure full grid exists so table always has the full layout
    for N in N_LIST:
        for w in OMEGA_LIST:
            grid.setdefault((N, w), None)

    return grid


# -------------------------
# CLI
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", type=str, required=True, help="Directory containing analysis JSON files"
    )
    ap.add_argument("--out", type=str, default="summary_tables.tex", help="Output LaTeX file path")
    ap.add_argument(
        "--table-star",
        action="store_true",
        help="Use table* instead of table (useful in two-column layouts)",
    )
    args = ap.parse_args()

    dir_path = Path(args.dir).expanduser().resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    grid = load_reports(dir_path)
    tex = build_tables(grid, use_table_star=bool(args.table_star))

    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(tex)
    print(f"[ok] wrote LaTeX tables to: {out_path}")


if __name__ == "__main__":
    main()
