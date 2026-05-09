import sys
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jastrow_architectures import CTNNJastrowVCycle, CTNNShellAwareJastrow


def test_cusp_uses_oscillator_length_scale() -> None:
    model = CTNNJastrowVCycle(
        n_particles=2,
        d=2,
        omega=0.01,
        node_hidden=8,
        edge_hidden=8,
        bottleneck_hidden=4,
    ).to(torch.float64)
    x = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
    spin = torch.tensor([0, 1], dtype=torch.long)

    got = model._compute_cusps(x, spin)
    assert got.item() == pytest.approx(math.exp(-0.1), rel=0.0, abs=1e-12)


def test_shellaware_jastrow_forward_and_gradients_are_finite() -> None:
    torch.manual_seed(5)
    model = CTNNShellAwareJastrow(
        n_particles=6,
        d=2,
        omega=0.01,
        node_hidden=16,
        edge_hidden=16,
        n_shells=3,
        shell_radius_aho=3.0,
        shell_width_aho=0.7,
        n_mp_steps=1,
        msg_layers=1,
        node_layers=1,
        readout_hidden=16,
    ).to(torch.float64)
    x = torch.randn(4, 6, 2, dtype=torch.float64, requires_grad=True)
    spin = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    y = model(x, spin)
    loss = y.pow(2).mean()
    loss.backward()

    assert y.shape == (4, 1)
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
