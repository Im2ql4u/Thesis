from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from functions.Neural_Networks import lookup_dmc_energy


@pytest.mark.parametrize("n,omega_table", sorted(config.DMC_ENERGIES.items()))
def test_lookup_exact_table_entries(n: int, omega_table: dict[float, float]) -> None:
    for omega, expected in omega_table.items():
        got = lookup_dmc_energy(n, omega)
        assert math.isfinite(got)
        assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("n", [2, 6, 12, 20])
def test_lookup_snaps_to_supported_omega_grid(n: int) -> None:
    # For each available omega at this N, query with a tiny perturbation and
    # confirm we still recover the same snapped value.
    for omega, expected in config.DMC_ENERGIES[n].items():
        got = lookup_dmc_energy(n, omega + 1e-6)
        assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("n,omega", [(12, 0.005), (20, 0.05), (2, 0.03)])
def test_lookup_raises_when_snap_distance_is_too_large(n: int, omega: float) -> None:
    with pytest.raises(ValueError):
        lookup_dmc_energy(n, omega)


@pytest.mark.parametrize("n", [4, 8, 10, 14, 16, 18])
def test_lookup_raises_for_unsupported_particle_counts(n: int) -> None:
    with pytest.raises(KeyError):
        lookup_dmc_energy(n, 1.0)
