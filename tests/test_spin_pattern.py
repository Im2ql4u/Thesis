import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from functions.Slater_Determinant import slater_determinant_closed_shell
from run_weak_form import parse_spin_pattern


def test_parse_spin_pattern_layouts_and_explicit_forms() -> None:
    assert parse_spin_pattern("", 6, layout="block") == (0, 0, 0, 1, 1, 1)
    assert parse_spin_pattern("", 6, layout="alternating") == (0, 1, 0, 1, 0, 1)
    assert parse_spin_pattern("u,d,u,d,u,d", 6) == (0, 1, 0, 1, 0, 1)


def test_parse_spin_pattern_rejects_unbalanced_patterns() -> None:
    with pytest.raises(ValueError, match="requires closed shell"):
        parse_spin_pattern("001111", 6)


def test_slater_accepts_alternating_spin_order() -> None:
    params = {
        "basis": "cart",
        "omega": 1.0,
        "nx": 3,
        "ny": 3,
        "n_particles": 4,
        "d": 2,
        "device": torch.device("cpu"),
        "torch_dtype": torch.float64,
    }
    c_occ = torch.zeros(9, 2, dtype=torch.float64)
    c_occ[0, 0] = 1.0
    c_occ[1, 1] = 1.0
    x = torch.tensor(
        [
            [[-0.7, 0.1], [0.2, -0.6], [0.9, 0.2], [-0.1, 0.8]],
            [[-0.5, 0.3], [0.4, -0.4], [0.8, 0.4], [0.0, 0.9]],
        ],
        dtype=torch.float64,
    )
    spin = torch.tensor(parse_spin_pattern("", 4, layout="alternating"), dtype=torch.long)

    sign, logabs = slater_determinant_closed_shell(x, c_occ, params=params, spin=spin)

    assert sign.shape == (2,)
    assert logabs.shape == (2,)
    assert torch.isfinite(logabs).all()
