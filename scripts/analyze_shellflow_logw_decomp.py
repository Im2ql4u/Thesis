from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_weak_form as rw
from functions.Neural_Networks import AdaptiveShellFlowProposal, psi_fn, slater_determinant_closed_shell
from jastrow_architectures import CTNNJastrowVCycle


def build_jastrow_model(omega: float) -> CTNNJastrowVCycle:
    return CTNNJastrowVCycle(
        n_particles=20,
        d=rw.DIM,
        omega=omega,
        node_hidden=24,
        edge_hidden=24,
        bottleneck_hidden=12,
        n_down=1,
        n_up=1,
        msg_layers=1,
        node_layers=1,
        readout_hidden=64,
        readout_layers=2,
        act="silu",
        input_radius_cap_aho=ARGS.jas_input_radius_cap_aho,
        tail_guard_radius_aho=ARGS.jas_tail_guard_radius_aho,
        tail_guard_strength=ARGS.jas_tail_guard_strength,
        tail_guard_power=ARGS.jas_tail_guard_power,
    ).to(rw.DEVICE).to(rw.DTYPE)


def load_jastrow_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=rw.DEVICE)
    state = checkpoint.get("jas_state", checkpoint.get("state"))
    if state is None:
        raise RuntimeError(f"No Jastrow state found in {checkpoint_path}")
    model.load_state_dict(state)


def build_proposal(args: argparse.Namespace) -> AdaptiveShellFlowProposal:
    shell_templates = rw.parse_shell_templates(args.shell_templates, 20)
    n_shells = len(shell_templates[0])
    shell_radii_init = rw.default_shell_radii_init(20, n_shells, args.omega)
    shell_sigmas = tuple(float(v) for v in args.shell_sigmas.split(",") if v.strip())
    shell_mix_logits = tuple(float(v) for v in args.shell_mix_logits.split(",") if v.strip())
    return AdaptiveShellFlowProposal(
        n_elec=20,
        dim=rw.DIM,
        omega=args.omega,
        shell_templates=shell_templates,
        mix_logits_init=shell_mix_logits,
        shell_radii_init=shell_radii_init,
        shell_sigmas=shell_sigmas,
        refit_every=15,
        refit_min_samples=args.refit_min_samples,
        refit_steps=120,
        refit_lr=1e-3,
        flow_layers=2,
        flow_hidden=128,
        ring_warmup_steps=120,
        radius_anchor_weight=0.35,
        device=rw.DEVICE,
        dtype=rw.DTYPE,
    )


def chunked_logpsi(
    model: torch.nn.Module,
    x: torch.Tensor,
    c_occ: torch.Tensor,
    params: dict,
    chunk_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            stop = min(start + chunk_size, x.shape[0])
            lp, _ = psi_fn(model, x[start:stop], c_occ, backflow_net=None, spin=None, params=params)
            outputs.append(lp)
    return torch.cat(outputs, dim=0)


def chunked_wavefunction_parts(
    model: torch.nn.Module,
    x: torch.Tensor,
    c_occ: torch.Tensor,
    params: dict,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    up = x.shape[1] // 2
    spin = torch.cat(
        [
            torch.zeros(up, dtype=torch.long, device=x.device),
            torch.ones(x.shape[1] - up, dtype=torch.long, device=x.device),
        ]
    )
    spin_bn = spin.unsqueeze(0)

    jastrow_parts: list[torch.Tensor] = []
    slater_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            stop = min(start + chunk_size, x.shape[0])
            xb = x[start:stop]
            sb = spin_bn.expand(xb.shape[0], -1)
            jastrow_parts.append(model(xb, spin=sb).squeeze(-1))
            _, logabs = slater_determinant_closed_shell(
                x_config=xb,
                C_occ=c_occ,
                params=params,
                spin=sb,
                normalize=True,
            )
            slater_parts.append(logabs.view(-1))
    return torch.cat(jastrow_parts, dim=0), torch.cat(slater_parts, dim=0)


def print_quantiles(name: str, values: torch.Tensor) -> None:
    q = torch.quantile(
        values,
        torch.tensor([0.0, 0.1, 0.5, 0.9, 0.99, 1.0], device=values.device, dtype=values.dtype),
    )
    print(name, " ".join(f"{float(v):.6f}" for v in q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose ShellFlow importance weights into logq and 2log|Psi| components.")
    parser.add_argument("--omega", type=float, required=True)
    parser.add_argument("--oversample", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--shell-templates", type=str, required=True)
    parser.add_argument("--shell-mix-logits", type=str, required=True)
    parser.add_argument("--shell-sigmas", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-keep", type=int, default=2048)
    parser.add_argument("--refit-min-samples", type=int, default=500)
    parser.add_argument("--show-components", action="store_true")
    parser.add_argument("--jas-input-radius-cap-aho", type=float, default=0.0)
    parser.add_argument("--jas-tail-guard-radius-aho", type=float, default=0.0)
    parser.add_argument("--jas-tail-guard-strength", type=float, default=0.0)
    parser.add_argument("--jas-tail-guard-power", type=float, default=4.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "arch_colloc" / "camp_jastrow_transfer_stabilized_n20_o0p1_s11.pt",
    )
    args = parser.parse_args()
    global ARGS
    ARGS = args

    rw.N_ELEC = 20
    rw.OMEGA = float(args.omega)
    rw.E_DMC = rw.default_dmc(20, args.omega, allow_missing=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    c_occ, params, _, _ = rw.setup(
        n_elec=20,
        omega=args.omega,
        e_dmc=rw.E_DMC,
        seed=args.seed,
        allow_missing_dmc=True,
    )
    model = build_jastrow_model(args.omega)
    load_jastrow_checkpoint(model, args.checkpoint)
    model.eval()

    proposal = build_proposal(args)
    n_candidates = args.n_keep * args.oversample
    x, logq = proposal.sample(n_candidates)
    lp = chunked_logpsi(model, x, c_occ, params, args.chunk_size)
    logw = 2.0 * lp - logq
    rho_aho = x.norm(dim=-1).reshape(x.shape[0], -1).mean(dim=1) * math.sqrt(args.omega)
    order = torch.argsort(logw, descending=True)

    if args.show_components:
        jastrow, slater = chunked_wavefunction_parts(model, x, c_occ, params, args.chunk_size)
        print_quantiles("jastrow", jastrow)
        print_quantiles("slater", slater)
        print_quantiles("logpsi", slater + jastrow)

    print_quantiles("rho", rho_aho)
    print_quantiles("logq", logq)
    print_quantiles("2logpsi", 2.0 * lp)
    print_quantiles("logw", logw)

    best = int(order[0].item())
    median = int(order[len(order) // 2].item())
    worst = int(order[-1].item())
    for label, idx in (("best", best), ("median", median), ("worst", worst)):
        print(
            f"{label} idx={idx} rho_aho={float(rho_aho[idx]):.6f} "
            f"logq={float(logq[idx]):.6f} 2logpsi={float((2.0 * lp)[idx]):.6f} logw={float(logw[idx]):.6f}"
        )

    probs = torch.softmax(logw - logw.max(), dim=0)
    for k in (1, 2, 5, 10, 50, 100):
        mass = torch.topk(probs, k=k).values.sum()
        print(f"top{k}_mass={float(mass):.6f}")

    print("shell_prob_init", " ".join(f"{v:.6f}" for v in proposal.shell_probabilities().detach().cpu().tolist()))
    shell_radii_aho = proposal.shell_radii_aho().to(device=rho_aho.device, dtype=rho_aho.dtype)
    assignments = torch.argmin((rho_aho.unsqueeze(1) - shell_radii_aho.unsqueeze(0)).abs(), dim=1)
    for rank in range(10):
        idx = int(order[rank].item())
        print(
            f"top rank={rank + 1} rho_aho={float(rho_aho[idx]):.6f} nearest_shell={int(assignments[idx].item())} "
            f"logq={float(logq[idx]):.6f} 2logpsi={float((2.0 * lp)[idx]):.6f} logw={float(logw[idx]):.6f}"
        )


if __name__ == "__main__":
    main()