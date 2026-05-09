import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from functions.Neural_Networks import AdaptiveShellFlowProposal
from functions.Neural_Networks import importance_resample as nn_importance_resample
from functions.Neural_Networks import rayleigh_hybrid_loss as nn_rayleigh_hybrid_loss
from run_weak_form import (
    collapse_recovery_fit_probs,
    default_shell_radii_init,
    parse_shell_templates,
    summarize_collapse_recovery_fit,
)
from jastrow_architectures import CTNNJastrowVCycle


def test_parse_shell_templates_known_layouts() -> None:
    got6 = parse_shell_templates("", 6)
    assert got6 == ((6, 0), (1, 5))

    got12 = parse_shell_templates("12-0,1-11,3-9", 12)
    assert got12 == ((12, 0), (1, 11), (3, 9))

    got20 = parse_shell_templates("", 20)
    assert got20 == ((20, 0, 0), (1, 19, 0), (1, 7, 12))


def test_default_shell_radii_init_is_omega_aware() -> None:
    low = default_shell_radii_init(6, 2, omega=0.001)
    mid = default_shell_radii_init(6, 2, omega=0.01)
    high = default_shell_radii_init(6, 2, omega=1.0)
    assert low[0] == pytest.approx(0.0, abs=1e-12)
    assert low[1] < mid[1] < high[1]


def test_shellflow_inner_probability_matches_template_mix() -> None:
    p = AdaptiveShellFlowProposal(
        n_elec=6,
        dim=2,
        omega=0.1,
        shell_templates=((6, 0), (1, 5)),
        mix_logits_init=(0.0, 0.0),
        shell_radii_init=(0.0, 1.25),
        shell_sigmas=(0.3, 1.5),
        refit_every=10,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=0,
        device="cpu",
        dtype=torch.float64,
    )
    probs = p.shell_probabilities().tolist()
    radii = p.shell_radii_aho().tolist()

    # Equal template weights => expected shell occupancy fractions:
    # inner: ((6/6) + (1/6)) / 2 = 7/12, outer = 5/12.
    assert probs[0] == pytest.approx(7.0 / 12.0, rel=0.0, abs=1e-12)
    assert probs[1] == pytest.approx(5.0 / 12.0, rel=0.0, abs=1e-12)
    assert radii[0] >= 0.0
    assert radii[1] > radii[0]


def test_shellflow_sample_and_log_prob_are_finite() -> None:
    torch.manual_seed(7)
    p = AdaptiveShellFlowProposal(
        n_elec=6,
        dim=2,
        omega=0.1,
        shell_templates=((6, 0), (1, 5)),
        mix_logits_init=(1.0, -1.0),
        shell_radii_init=(0.0, 1.5),
        shell_sigmas=(0.35, 1.4),
        refit_every=10,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=2,
        flow_hidden=32,
        device="cpu",
        dtype=torch.float64,
    )
    x, lq = p.sample(64)
    assert x.shape == (64, 6, 2)
    assert lq.shape == (64,)
    assert torch.isfinite(x).all()
    assert torch.isfinite(lq).all()

    lq_eval = p.log_prob(x)
    assert lq_eval.shape == (64,)
    assert torch.isfinite(lq_eval).all()

    # Known-answer style check: proposal density should be in a plausible range.
    mean_lq = float(lq_eval.mean().item())
    assert math.isfinite(mean_lq)
    assert mean_lq < 0.0


def test_shellflow_supports_multi_shell_templates() -> None:
    p = AdaptiveShellFlowProposal(
        n_elec=20,
        dim=2,
        omega=0.05,
        shell_templates=((20, 0, 0), (3, 9, 8)),
        mix_logits_init=(0.0, 0.0),
        shell_radii_init=(0.0, 1.0, 2.0),
        shell_sigmas=(0.25, 0.5, 0.9),
        refit_every=10,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=0,
        device="cpu",
        dtype=torch.float64,
    )
    probs = p.shell_probabilities().tolist()
    radii = p.shell_radii_aho().tolist()
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert radii[0] < radii[1] < radii[2]


def test_importance_resample_can_return_proposal_fit_data() -> None:
    torch.manual_seed(3)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return -0.5 * x.reshape(x.shape[0], -1).pow(2).sum(dim=-1)

    x_keep, ess, stats, proposal_data = nn_importance_resample(
        psi_log_fn,
        n_keep=16,
        n_elec=2,
        dim=2,
        omega=0.5,
        device="cpu",
        dtype=torch.float64,
        n_cand_mult=4,
        return_stats=True,
        return_proposal_data=True,
    )

    assert x_keep.shape == (16, 2, 2)
    assert math.isfinite(float(ess))
    assert math.isfinite(float(stats["ess_eff"]))
    assert proposal_data["x_candidates"].shape == (64, 2, 2)
    assert proposal_data["raw_target_probs"].shape == (64,)
    assert proposal_data["resample_probs"].shape == (64,)
    assert proposal_data["sample_indices"].shape == (16,)
    assert proposal_data["sample_reweight"].shape == (16,)
    assert float(proposal_data["resample_probs"].sum().item()) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert float(proposal_data["raw_target_probs"].sum().item()) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    idx = proposal_data["sample_indices"]
    expected = proposal_data["raw_target_probs"][idx] / proposal_data["resample_probs"][idx]
    assert torch.allclose(proposal_data["sample_reweight"], expected, atol=1e-12, rtol=0.0)


def test_shellflow_weighted_refit_accepts_candidate_cloud() -> None:
    torch.manual_seed(11)
    p = AdaptiveShellFlowProposal(
        n_elec=6,
        dim=2,
        omega=0.1,
        shell_templates=((6, 0), (1, 5)),
        mix_logits_init=(0.0, 0.0),
        shell_radii_init=(0.0, 1.25),
        shell_sigmas=(0.3, 1.2),
        refit_every=5,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=0,
        device="cpu",
        dtype=torch.float64,
    )
    x_candidates, _ = p.sample(64)
    weights = torch.ones(64, dtype=torch.float64) / 64.0

    changed = p.maybe_refit_weighted(5, x_candidates, weights)

    assert changed is True
    assert torch.isfinite(p.shell_probabilities()).all()
    assert torch.isfinite(p.shell_radii_aho()).all()
    assert torch.isfinite(p.shell_sigmas_aho()).all()


def test_shellflow_curriculum_starts_centered_before_unlock() -> None:
    p = AdaptiveShellFlowProposal(
        n_elec=20,
        dim=2,
        omega=0.01,
        shell_templates=((20, 0, 0), (1, 19, 0), (1, 7, 12)),
        mix_logits_init=(0.0, 0.0, 0.0),
        shell_radii_init=(0.0, 0.75, 1.50),
        shell_sigmas=(0.45, 1.00, 1.80),
        refit_every=10,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=0,
        curriculum_mode="epoch",
        curriculum_unlock_epoch=20,
        curriculum_inactive_logit_offset=-12.0,
        device="cpu",
        dtype=torch.float64,
    )

    weights = p.template_weights().tolist()
    state = p.curriculum_state()

    assert state["unlocked"] is False
    assert weights[0] > 0.999
    assert weights[1] < 1e-4
    assert weights[2] < 1e-4


def test_shellflow_curriculum_unlocks_on_epoch_refit() -> None:
    torch.manual_seed(13)
    p = AdaptiveShellFlowProposal(
        n_elec=6,
        dim=2,
        omega=0.1,
        shell_templates=((6, 0), (1, 5)),
        mix_logits_init=(0.0, 0.0),
        shell_radii_init=(0.0, 1.25),
        shell_sigmas=(0.3, 1.2),
        refit_every=5,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=0,
        curriculum_mode="epoch",
        curriculum_unlock_epoch=5,
        device="cpu",
        dtype=torch.float64,
    )
    x_candidates, _ = p.sample(64)
    weights = torch.ones(64, dtype=torch.float64) / 64.0

    changed = p.maybe_refit_weighted(5, x_candidates, weights)
    state = p.curriculum_state()

    assert changed is True
    assert state["unlocked"] is True
    assert state["reason"] == "epoch>=5"


def test_shellflow_curriculum_unlocks_on_radius_quantile() -> None:
    p = AdaptiveShellFlowProposal(
        n_elec=20,
        dim=2,
        omega=0.01,
        shell_templates=((20, 0, 0), (1, 19, 0), (1, 7, 12)),
        mix_logits_init=(0.0, 0.0, 0.0),
        shell_radii_init=(0.0, 0.75, 1.50),
        shell_sigmas=(0.45, 1.00, 1.80),
        refit_every=5,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=0,
        curriculum_mode="radius",
        curriculum_radius_quantile=0.8,
        curriculum_radius_threshold_aho=1.0,
        device="cpu",
        dtype=torch.float64,
    )
    x_candidates = torch.full((64, 20, 2), 10.0, dtype=torch.float64)
    weights = torch.ones(64, dtype=torch.float64) / 64.0

    changed = p.maybe_refit_weighted(5, x_candidates, weights)
    state = p.curriculum_state()

    assert changed is True
    assert state["unlocked"] is True
    assert state["metric"] > 1.0


def test_shellflow_curriculum_unlocks_on_ess_patience() -> None:
    torch.manual_seed(17)
    p = AdaptiveShellFlowProposal(
        n_elec=6,
        dim=2,
        omega=0.1,
        shell_templates=((6, 0), (1, 5)),
        mix_logits_init=(0.0, 0.0),
        shell_radii_init=(0.0, 1.25),
        shell_sigmas=(0.3, 1.2),
        refit_every=5,
        refit_min_samples=32,
        refit_steps=2,
        refit_lr=1e-3,
        flow_layers=0,
        curriculum_mode="ess",
        curriculum_unlock_ess=18.0,
        curriculum_unlock_patience=2,
        device="cpu",
        dtype=torch.float64,
    )
    x_candidates, _ = p.sample(64)
    weights = torch.ones(64, dtype=torch.float64) / 64.0

    first = p.maybe_refit_weighted(5, x_candidates, weights, diagnostic_ess=20.0)
    mid_state = p.curriculum_state()
    second = p.maybe_refit_weighted(10, x_candidates, weights, diagnostic_ess=21.0)
    final_state = p.curriculum_state()

    assert first is True
    assert second is True
    assert mid_state["unlocked"] is False
    assert mid_state["streak"] == 1
    assert final_state["unlocked"] is True
    assert final_state["reason"] == "ESS>=18.00"


def test_shellflow_curriculum_centered_mass_floor_applies_after_unlock() -> None:
    p = AdaptiveShellFlowProposal(
        n_elec=6,
        dim=2,
        omega=0.1,
        shell_templates=((6, 0), (1, 5)),
        mix_logits_init=(0.0, 0.0),
        shell_radii_init=(0.0, 1.25),
        shell_sigmas=(0.3, 1.2),
        refit_every=5,
        refit_min_samples=32,
        refit_steps=1,
        refit_lr=1e-3,
        flow_layers=0,
        curriculum_mode="epoch",
        curriculum_unlock_epoch=5,
        curriculum_centered_mass_floor=0.25,
        device="cpu",
        dtype=torch.float64,
    )

    p._curriculum_unlocked = True
    weights = p.template_weights().tolist()

    assert weights[0] == pytest.approx(0.625, rel=0.0, abs=1e-12)
    assert weights[1] == pytest.approx(0.375, rel=0.0, abs=1e-12)


def test_shellflow_curriculum_floor_is_reported_in_state() -> None:
    p = AdaptiveShellFlowProposal(
        n_elec=20,
        dim=2,
        omega=0.01,
        shell_templates=((20, 0, 0), (1, 19, 0), (1, 7, 12)),
        mix_logits_init=(0.0, 0.0, 0.0),
        shell_radii_init=(0.0, 0.75, 1.50),
        shell_sigmas=(0.45, 1.00, 1.80),
        refit_every=5,
        refit_min_samples=32,
        refit_steps=1,
        refit_lr=1e-3,
        flow_layers=0,
        curriculum_mode="radius",
        curriculum_radius_quantile=0.8,
        curriculum_radius_threshold_aho=1.0,
        curriculum_centered_mass_floor=0.15,
        device="cpu",
        dtype=torch.float64,
    )

    assert p.curriculum_state()["centered_mass_floor"] == pytest.approx(0.15, rel=0.0, abs=1e-12)


def test_weighted_hybrid_loss_matches_weighted_energy_mean() -> None:
    torch.manual_seed(5)

    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        return -0.25 * x.reshape(x.shape[0], -1).pow(2).sum(dim=-1)

    x = torch.randn(6, 2, 2, dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5, 3.0, 4.0], dtype=torch.float64)
    params = {"omega": 0.2, "n_particles": 2, "d": 2}

    _, weighted_mean, el_det, _ = nn_rayleigh_hybrid_loss(
        psi_log_fn,
        x,
        omega=0.2,
        params=params,
        direct_weight=0.0,
        clip_el=0.0,
        reward_qtrim=0.0,
        sample_weights=weights,
    )

    expected = float((weights * el_det).sum().item() / weights.sum().item())
    assert weighted_mean == pytest.approx(expected, rel=0.0, abs=1e-10)


def test_collapse_recovery_fit_probs_focuses_elite_low_radius_candidates() -> None:
    raw_target_probs = torch.tensor([0.01, 0.30, 0.25, 0.24, 0.20], dtype=torch.float64)
    resample_probs = torch.ones(5, dtype=torch.float64) / 5.0
    x_candidates = torch.tensor(
        [
            [[[0.1, 0.0]]],
            [[[0.4, 0.0]]],
            [[[0.8, 0.0]]],
            [[[1.2, 0.0]]],
            [[[2.5, 0.0]]],
        ],
        dtype=torch.float64,
    )

    fit_probs = collapse_recovery_fit_probs(
        {
            "raw_target_probs": raw_target_probs,
            "resample_probs": resample_probs,
            "x_candidates": x_candidates,
        },
        omega=1.0,
        refit_min_samples=2,
        elite_fraction=0.4,
        elite_cap=4,
        radius_quantile=0.5,
    )

    assert float(fit_probs.sum().item()) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert fit_probs[4].item() == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert fit_probs[1].item() > 0.0
    assert fit_probs[2].item() == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_collapse_recovery_summary_reports_retained_mass_and_radius_cap() -> None:
    raw_target_probs = torch.tensor([0.01, 0.30, 0.25, 0.24, 0.20], dtype=torch.float64)
    resample_probs = torch.ones(5, dtype=torch.float64) / 5.0
    x_candidates = torch.tensor(
        [
            [[[0.1, 0.0]]],
            [[[0.4, 0.0]]],
            [[[0.8, 0.0]]],
            [[[1.2, 0.0]]],
            [[[2.5, 0.0]]],
        ],
        dtype=torch.float64,
    )
    proposal_fit_data = {
        "raw_target_probs": raw_target_probs,
        "resample_probs": resample_probs,
        "x_candidates": x_candidates,
    }

    fit_probs = collapse_recovery_fit_probs(
        proposal_fit_data,
        omega=1.0,
        refit_min_samples=2,
        elite_fraction=0.4,
        elite_cap=4,
        radius_quantile=0.5,
    )
    summary = summarize_collapse_recovery_fit(
        proposal_fit_data,
        fit_probs,
        omega=1.0,
        refit_min_samples=2,
        elite_fraction=0.4,
        elite_cap=4,
        radius_quantile=0.5,
    )

    assert summary["candidate_count"] == pytest.approx(5.0, rel=0.0, abs=1e-12)
    assert summary["elite_count"] == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert summary["kept_count"] == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert summary["retained_mass"] == pytest.approx(0.30, rel=0.0, abs=1e-12)
    assert summary["top1_radius"] == pytest.approx(0.4, rel=0.0, abs=1e-12)
    assert summary["kept_radius_q50"] == pytest.approx(0.4, rel=0.0, abs=1e-12)
    assert summary["radius_cap"] == pytest.approx(0.4, rel=0.0, abs=1e-12)


def test_vcycle_tail_guard_penalizes_large_radius_only() -> None:
    model = CTNNJastrowVCycle(
        n_particles=6,
        d=2,
        omega=1.0,
        node_hidden=8,
        edge_hidden=8,
        bottleneck_hidden=4,
        n_down=1,
        n_up=1,
        msg_layers=1,
        node_layers=1,
        readout_hidden=8,
        readout_layers=1,
        act="silu",
        tail_guard_radius_aho=1.5,
        tail_guard_strength=10.0,
        tail_guard_power=4.0,
    ).to(dtype=torch.float64)

    near = torch.full((2, 6, 2), 0.5, dtype=torch.float64)
    far = torch.full((2, 6, 2), 3.0, dtype=torch.float64)

    near_guard = model._radial_tail_guard(near)
    far_guard = model._radial_tail_guard(far)

    assert torch.allclose(near_guard, torch.zeros_like(near_guard), atol=1e-12, rtol=0.0)
    assert torch.all(far_guard > 0.0)


def test_vcycle_input_radius_cap_clamps_particle_norms() -> None:
    model = CTNNJastrowVCycle(
        n_particles=6,
        d=2,
        omega=1.0,
        node_hidden=8,
        edge_hidden=8,
        bottleneck_hidden=4,
        n_down=1,
        n_up=1,
        msg_layers=1,
        node_layers=1,
        readout_hidden=8,
        readout_layers=1,
        act="silu",
        input_radius_cap_aho=1.5,
    ).to(dtype=torch.float64)

    x_sc = torch.tensor([[[3.0, 4.0], [0.3, 0.4]]], dtype=torch.float64)
    capped = model._cap_model_radius(x_sc)

    norms = capped.norm(dim=-1)
    assert norms[0, 0].item() == pytest.approx(1.5, rel=0.0, abs=1e-12)
    assert norms[0, 1].item() == pytest.approx(0.5, rel=0.0, abs=1e-12)
