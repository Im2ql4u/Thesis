"""Independent, blocked MCMC re-evaluation of a collocation checkpoint.

Runs the collocation trainer (`src/run_weak_form.py`) in eval-only mode on a saved
checkpoint, with its final evaluator swapped for one that uses the *same* persistent
Metropolis sampler and exact local energy (`functions.Energy`), but

  * runs several independently seeded chains, each from a fresh Gaussian start with
    its own burn-in, and
  * records the walker-averaged local energy of every sweep, so that the standard error
    can be taken from a Flyvbjerg-Petersen blocking plateau instead of the naive
    sigma/sqrt(n), which ignores the correlation between successive sweeps.

The trainer's model construction and checkpoint loading are used unchanged, so a
checkpoint that does not match the architecture fails to load rather than being
silently evaluated on the wrong network.

Usage (from the repo root, on a GPU node):
    python3.11 scripts/eval_collocation_blocked.py --ckpt results/arch_colloc/X.pt \
        --omega 0.1 --out results/analysis/<dated>/X.json [--chains 4 --sweeps 200]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_weak_form as rwf  # noqa: E402
from functions.Energy import (  # noqa: E402
    _local_energy_multi,
    _metropolis_psi2_persistent,
    _normalize_device_dtype,
)

BURN_IN_STEPS = 2000  # Metropolis steps discarded per chain before recording
THIN_STEPS = 10  # Metropolis steps between recorded sweeps
WALKERS = 512  # walkers per chain (the trainer's final-eval batch size)
TARGET_ACCEPT = 0.45  # the trainer's default adaptive target
ADAPT_LR = 0.05  # the trainer's default step-size adaptation rate
MIN_BLOCKS = 16  # stop blocking when fewer blocks than this remain

RESULT: dict = {}


def blocking_stderr(series: np.ndarray) -> tuple[float, list[float]]:
    """Flyvbjerg-Petersen blocking: return (plateau stderr, stderr at each level)."""
    x = np.asarray(series, dtype=float)
    levels = []
    while len(x) >= MIN_BLOCKS:
        n = len(x)
        levels.append(float(np.std(x, ddof=1) / math.sqrt(n)))
        x = 0.5 * (x[: n // 2 * 2 : 2] + x[1 : n // 2 * 2 : 2])
    # Plateau estimate: the largest stderr over levels, a standard conservative choice
    # when the series is too short for a clean plateau.
    return (max(levels) if levels else float("nan")), levels


def make_blocked_evaluator(n_chains: int, n_sweeps: int, base_seed: int):
    """Build a drop-in replacement for functions.Energy.evaluate_energy_vmc."""

    def evaluate(
        f_net,
        C_occ,
        *,
        psi_fn,
        compute_coulomb_interaction,
        backflow_net=None,
        orbital_bf_net=None,
        spin=None,
        params=None,
        sampler_step_sigma: float = 0.08,
        lap_mode: str = "exact",
        **_ignored,
    ):
        device, dtype = _normalize_device_dtype(params)
        omega = float(params["omega"])
        n_part = int(params["n_particles"])
        dim = int(params["d"])
        ell = 1.0 / math.sqrt(max(omega, 1e-12))  # same omega-invariant scaling as the trainer

        f_net.eval()
        if backflow_net is not None:
            backflow_net.eval()

        def psi_log_fn(x):
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            logpsi, _ = psi_fn(
                f_net,
                x,
                C_occ,
                backflow_net=backflow_net,
                orbital_bf_net=orbital_bf_net,
                spin=spin,
                params=params,
            )
            return logpsi

        chains = []
        for c in range(n_chains):
            seed = base_seed + 1000 * c
            torch.manual_seed(seed)
            x = torch.randn(WALKERS, n_part, dim, device=device, dtype=dtype) * ell
            _, x, _, _ = _metropolis_psi2_persistent(
                psi_log_fn,
                x,
                burn_in=BURN_IN_STEPS,
                thin=1,
                n_keep=1,
                step_sigma=sampler_step_sigma * ell,
                target_accept=TARGET_ACCEPT,
                adapt_lr=ADAPT_LR,
            )
            sweep_E, sweep_T, sweep_Vi, sweep_Vh = [], [], [], []
            acc = prop = 0
            for _ in range(n_sweeps):
                samples, x, a_, p_ = _metropolis_psi2_persistent(
                    psi_log_fn,
                    x.detach(),
                    burn_in=0,
                    thin=THIN_STEPS,
                    n_keep=1,
                    step_sigma=sampler_step_sigma * ell,
                    target_accept=TARGET_ACCEPT,
                    adapt_lr=ADAPT_LR,
                )
                acc += a_
                prop += p_
                xs = samples.reshape(-1, n_part, dim)
                with torch.set_grad_enabled(True):
                    E_L, T, Vi, Vh, _ = _local_energy_multi(
                        psi_log_fn, xs, compute_coulomb_interaction, omega, lap_mode=lap_mode
                    )
                sweep_E.append(float(E_L.detach().mean()))
                sweep_T.append(float(T.detach().mean()))
                sweep_Vi.append(float(Vi.detach().mean()))
                sweep_Vh.append(float(Vh.detach().mean()))
            se_block, levels = blocking_stderr(np.array(sweep_E))
            chains.append(
                dict(
                    seed=seed,
                    E=float(np.mean(sweep_E)),
                    se_block=se_block,
                    se_naive_sweeps=levels[0] if levels else float("nan"),
                    blocking_levels=levels,
                    accept=acc / max(prop, 1),
                    T=float(np.mean(sweep_T)),
                    V_int=float(np.mean(sweep_Vi)),
                    V_trap=float(np.mean(sweep_Vh)),
                    sweep_E=sweep_E,
                )
            )
            print(
                f"    chain {c}: E={chains[-1]['E']:.6f} +- {se_block:.6f} (blocked)  "
                f"acc={chains[-1]['accept']:.2f}",
                flush=True,
            )

        means = np.array([ch["E"] for ch in chains])
        ses = np.array([ch["se_block"] for ch in chains])
        E = float(means.mean())
        se_within = float(math.sqrt(np.sum(ses**2)) / len(ses))
        se_between = (
            float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")
        )
        se = max(se_within, se_between) if math.isfinite(se_between) else se_within
        RESULT.update(
            E=E,
            se=se,
            se_within=se_within,
            se_between=se_between,
            n_chains=n_chains,
            n_sweeps=n_sweeps,
            walkers=WALKERS,
            burn_in_steps=BURN_IN_STEPS,
            thin_steps=THIN_STEPS,
            chains=chains,
        )
        return dict(E_mean=E, E_stderr=se)

    return evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--omega", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260912)
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu")
    # The cusp_len_mode metadata was introduced together with the oscillator-length cusp
    # (commit c407da8, 2026-05-09); a checkpoint without it was trained with the legacy
    # exp(-r) cusp and must be evaluated with it, or a different wavefunction is scored.
    cusp_mode = str(ckpt.get("cusp_len_mode", "legacy"))
    rwf.evaluate_energy_vmc = make_blocked_evaluator(a.chains, a.sweeps, a.seed)
    tag = f"blockeval_{Path(a.ckpt).stem}"
    sys.argv = [
        "run_weak_form.py",
        "--mode",
        str(ckpt.get("mode", "bf")),
        "--resume",
        a.ckpt,
        "--n-elec",
        str(int(ckpt.get("n_elec", 6))),
        "--omega",
        a.omega,
        "--seed",
        str(a.seed),
        "--epochs",
        "0",
        "--n-eval",
        "1",
        "--tag",
        tag,
        "--no-pretrained",
        "--allow-missing-dmc-ref",
        "--cusp-len-mode",
        cusp_mode,
    ]
    rwf.main()

    RESULT.update(
        checkpoint=a.ckpt,
        omega=float(a.omega),
        cusp_len_mode=cusp_mode,
        logged_final_E=ckpt.get("E"),
        logged_final_se=ckpt.get("se"),
        train_seed=ckpt.get("seed"),
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULT, indent=1))
    print(
        f"  wrote {out}: E={RESULT['E']:.6f} +- {RESULT['se']:.6f} "
        f"(logged {RESULT['logged_final_E']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
