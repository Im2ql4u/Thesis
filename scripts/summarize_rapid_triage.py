#!/usr/bin/env python3
"""Summarize rapid grand-plan training logs without importing torch."""

import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


NUM = r"[-+0-9.eE]+|[-+]?nan"
LINE_RE = re.compile(
    r"\[\s*(?P<ep>\d+)\]\s+E=(?P<E>" + NUM + r").*?"
    r"ESS=(?P<ess>" + NUM + r").*?"
    r"(?P<dt>" + NUM + r")s\s+err=(?P<err>" + NUM + r")%"
    r"(?:\s+rawESS=(?P<raw>" + NUM + r"))?"
    r"(?:\s+khat=(?P<khat>" + NUM + r"))?"
    r"(?:\s+cand=(?P<cand>\d+))?"
    r"(?:.*?vmc=(?P<vmcE>" + NUM + r")\((?P<vmcErr>" + NUM + r")%\))?"
)


def fnum(value: Optional[str]) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def summarize_log(path: Path, run_name: str = "") -> Dict[str, object]:
    rows: List[Dict[str, float]] = []
    ended = "no"
    rc = ""
    for line in path.read_text(errors="replace").splitlines():
        if "] END" in line:
            ended = "yes"
            rc_match = re.search(r"rc=([-0-9]+)", line)
            rc = rc_match.group(1) if rc_match else ""
        match = LINE_RE.search(line)
        if not match:
            continue
        row = {
            "ep": fnum(match.group("ep")),
            "E": fnum(match.group("E")),
            "err": fnum(match.group("err")),
            "ess": fnum(match.group("ess")),
            "raw": fnum(match.group("raw")),
            "khat": fnum(match.group("khat")),
            "dt": fnum(match.group("dt")),
            "cand": fnum(match.group("cand")),
            "vmcE": fnum(match.group("vmcE")),
            "vmcErr": fnum(match.group("vmcErr")),
        }
        rows.append(row)

    if not rows:
        return {
            "log": path.name,
            "run": run_name,
            "ended": ended,
            "rc": rc,
            "ep": "-",
            "bestE": "-",
            "bestErr": "-",
            "bestVmcE": "-",
            "bestVmcErr": "-",
            "lastRaw": "-",
            "lastKhat": "-",
            "lastCand": "-",
            "lastDt": "-",
        }

    best = min(rows, key=lambda r: r["E"] if math.isfinite(r["E"]) else math.inf)
    vmc_rows = [r for r in rows if math.isfinite(r["vmcE"])]
    best_vmc = min(vmc_rows, key=lambda r: r["vmcE"]) if vmc_rows else None
    last = rows[-1]

    def fmt(value: float, digits: int = 3) -> str:
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"

    return {
        "log": path.name,
        "run": run_name,
        "ended": ended,
        "rc": rc,
        "ep": str(int(last["ep"])),
        "bestE": fmt(best["E"], 4),
        "bestErr": fmt(best["err"], 2),
        "bestVmcE": fmt(best_vmc["vmcE"], 4) if best_vmc else "-",
        "bestVmcErr": fmt(best_vmc["vmcErr"], 2) if best_vmc else "-",
        "lastRaw": fmt(last["raw"], 0),
        "lastKhat": fmt(last["khat"], 2),
        "lastCand": str(int(last["cand"])) if math.isfinite(last["cand"]) else "-",
        "lastDt": fmt(last["dt"], 1),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: summarize_rapid_triage.py <output-dir> [<output-dir> ...]", file=sys.stderr)
        return 2

    rows: List[Dict[str, object]] = []
    multi = len(sys.argv) > 2
    for arg in sys.argv[1:]:
        out_dir = Path(arg)
        log_dir = out_dir / "logs"
        logs = sorted(log_dir.glob("*.log"))
        if not logs:
            print(f"No logs found in {log_dir}", file=sys.stderr)
            return 1
        rows.extend(summarize_log(path, out_dir.name if multi else "") for path in logs)
    headers = [
        "run",
        "log",
        "ended",
        "rc",
        "ep",
        "bestE",
        "bestErr%",
        "bestVmcE",
        "bestVmcErr%",
        "rawESS",
        "khat",
        "cand",
        "dt_s",
    ]
    keys = [
        "run",
        "log",
        "ended",
        "rc",
        "ep",
        "bestE",
        "bestErr",
        "bestVmcE",
        "bestVmcErr",
        "lastRaw",
        "lastKhat",
        "lastCand",
        "lastDt",
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, key in enumerate(keys):
            widths[i] = max(widths[i], len(str(row[key])))

    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(row[key]).ljust(widths[i]) for i, key in enumerate(keys)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
