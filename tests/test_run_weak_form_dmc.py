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
from run_weak_form import default_dmc


@pytest.mark.parametrize("n,omega_table", sorted(config.DMC_ENERGIES.items()))
def test_default_dmc_reads_config_table(n: int, omega_table: dict[float, float]) -> None:
    for omega, expected in omega_table.items():
        got = default_dmc(n, omega)
        assert math.isfinite(got)
        assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_default_dmc_raises_when_no_config_entry() -> None:
    with pytest.raises(KeyError):
        default_dmc(18, 1.0)


def test_default_dmc_returns_nan_when_missing_is_allowed() -> None:
    got = default_dmc(18, 1.0, allow_missing=True)
    assert math.isnan(got)
