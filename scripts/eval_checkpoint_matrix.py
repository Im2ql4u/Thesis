from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_WEAK_FORM = ROOT / "src" / "run_weak_form.py"
RESULTS_ARCH_COLLOC = ROOT / "results" / "arch_colloc"


def _finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _resolve_checkpoints(inputs: list[str]) -> list[Path]:
    checkpoints: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            checkpoints.extend(sorted(p.glob("*.pt")))
        elif p.is_file() and p.suffix == ".pt":
            checkpoints.append(p)
        else:
            raise FileNotFoundError(f"Checkpoint path not found or not .pt: {raw}")
    if not checkpoints:
        raise RuntimeError("No checkpoints resolved from input paths")
    return checkpoints


def _run_eval_for_checkpoint(
    ckpt_path: Path,
    n_eval: int,
    allow_missing_dmc_ref: bool,
) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode = str(ckpt.get("mode", "bf"))
    n_elec = int(ckpt.get("n_elec", 6))
    omega = float(ckpt.get("omega", 1.0))
    seed = int(ckpt.get("seed", 42))
    e_dmc = ckpt.get("e_dmc", float("nan"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_tag = f"eval_{ckpt_path.stem}_{ts}"
    save_path = RESULTS_ARCH_COLLOC / f"{eval_tag}.pt"

    cmd = [
        sys.executable,
        str(RUN_WEAK_FORM),
        "--mode",
        mode,
        "--resume",
        str(ckpt_path),
        "--n-elec",
        str(n_elec),
        "--omega",
        str(omega),
        "--seed",
        str(seed),
        "--epochs",
        "0",
        "--n-eval",
        str(n_eval),
        "--tag",
        eval_tag,
        "--no-pretrained",
    ]

    if _finite_float(e_dmc):
        cmd.extend(["--e-dmc", str(float(e_dmc))])
    elif allow_missing_dmc_ref:
        cmd.append("--allow-missing-dmc-ref")

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    out: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "mode": mode,
        "n_elec": n_elec,
        "omega": omega,
        "seed": seed,
        "eval_tag": eval_tag,
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }

    if proc.returncode != 0:
        out["status"] = "failed"
        return out

    if not save_path.exists():
        out["status"] = "failed"
        out["error"] = f"Expected eval checkpoint missing: {save_path}"
        return out

    eval_ckpt = torch.load(save_path, map_location="cpu")
    out.update(
        status="ok",
        eval_checkpoint=str(save_path),
        E=float(eval_ckpt.get("E", float("nan"))),
        se=float(eval_ckpt.get("se", float("nan"))),
        err=float(eval_ckpt.get("err", float("nan"))),
        n_hist=int(len(eval_ckpt.get("hist", []))),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more weak-form checkpoints using heavy VMC by "
            "running run_weak_form.py with --epochs 0 and --resume"
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint file or directory (repeatable)",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=30000,
        help="Final VMC sample count per checkpoint",
    )
    parser.add_argument(
        "--allow-missing-dmc-ref",
        action="store_true",
        help="Allow evaluation when checkpoint has non-finite e_dmc",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional explicit JSON output path",
    )
    args = parser.parse_args()

    checkpoints = _resolve_checkpoints(args.checkpoint)

    run_dir = ROOT / "results" / f"{datetime.now().strftime('%Y-%m-%d')}_checkpoint_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_json) if args.output_json else run_dir / "summary.json"

    results: list[dict[str, Any]] = []
    for ckpt_path in checkpoints:
        result = _run_eval_for_checkpoint(
            ckpt_path=ckpt_path,
            n_eval=args.n_eval,
            allow_missing_dmc_ref=args.allow_missing_dmc_ref,
        )
        results.append(result)
        status = result.get("status", "unknown")
        msg = f"[{status}] {ckpt_path.name}"
        if status == "ok":
            msg += f" E={result['E']:.6f} err={result['err']:+.3f}%"
        print(msg)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_eval": int(args.n_eval),
        "allow_missing_dmc_ref": bool(args.allow_missing_dmc_ref),
        "count": len(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote summary: {output_path}")


if __name__ == "__main__":
    main()
