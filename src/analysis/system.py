"""Generalizable NQS system builder for the analysis program.

Builds a closed-shell Slater x Jastrow (x optional backflow) ansatz for *any*
even N, dimension d, and trap frequency omega, reusing the repository's Slater
determinant, psi_fn, and architecture modules. Nothing here is specialised to a
particular N or omega.

A `System` bundles the networks, the orbital occupation matrix, a params dict,
and ready-to-use closures:
    sys.log_psi(x)      -> log|Psi(x)|            (full ansatz, B,)
    sys.log_slater(x)   -> log|det| of bare HO core (no backflow, B,)
    sys.sample(n, ...)  -> MCMC configurations drawn from |Psi|^2
    sys.modules()       -> trainable modules (for kernel / SR)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

import config
from functions.Neural_Networks import _make_closed_shell_spin, psi_fn
from functions.Slater_Determinant import slater_determinant_closed_shell
from functions.Stochastic_Reconfiguration import _persistent_rw
from jastrow_architectures import CTNNJastrow, CTNNJastrowVCycle, DeepSetJastrow
from PINN import BackflowNet, CTNNBackflowNet, PINN


def _closed_shell_occupation(n_occ: int, nx: int, ny: int, dtype, device) -> torch.Tensor:
    """Occupation matrix selecting the n_occ lowest 2D-HO Cartesian orbitals."""
    energies = sorted((ix + iy, ix, iy) for ix in range(nx) for iy in range(ny))
    C = np.zeros((nx * ny, n_occ))
    for k in range(n_occ):
        _, ix, iy = energies[k]
        C[ix * ny + iy, k] = 1.0
    return torch.tensor(C, dtype=dtype, device=device)


# Jastrow architectures available to the program (extend as needed).
def _build_jastrow(arch: str, *, N: int, d: int, omega: float, **kw) -> nn.Module:
    arch = arch.lower()
    if arch == "ctnn_vcycle":
        return CTNNJastrowVCycle(n_particles=N, d=d, omega=omega, **kw)
    if arch == "ctnn":
        return CTNNJastrow(n_particles=N, d=d, omega=omega, **kw)
    if arch == "deepset":
        return DeepSetJastrow(n_particles=N, d=d, omega=omega, **kw)
    if arch == "pinn":
        return PINN(n_particles=N, d=d, omega=omega, **kw)
    raise ValueError(f"Unknown jastrow arch '{arch}'")


@dataclass
class System:
    N: int
    omega: float
    d: int = 2
    arch: str = "ctnn_vcycle"
    arch_kwargs: dict = field(default_factory=dict)
    use_backflow: bool = False
    backflow_kwargs: dict = field(default_factory=dict)
    backflow_arch: str = "conv"  # "conv" = BackflowNet; "ctnn" = message-passing CTNNBackflowNet
    device: str | None = None
    dtype: torch.dtype = torch.float64
    seed: int | None = None

    f_net: nn.Module = field(init=False)
    backflow_net: nn.Module | None = field(init=False, default=None)
    C_occ: torch.Tensor = field(init=False)
    spin: torch.Tensor = field(init=False)
    params: dict = field(init=False)

    def __post_init__(self) -> None:
        if self.N % 2 != 0:
            raise ValueError("Closed-shell builder requires even N.")
        dev = self.device or config._default_device()
        self.device = dev
        n_occ = self.N // 2
        nx = ny = n_occ + 1  # generous: always hosts the n_occ lowest orbitals

        dtype_str = {torch.float64: "float64", torch.float32: "float32"}[self.dtype]
        L = max(8.0, 3.0 / math.sqrt(self.omega))
        config.update(
            omega=self.omega,
            n_particles=self.N,
            d=self.d,
            L=L,
            nx=nx,
            ny=ny,
            basis="cart",
            device=dev,
            dtype=dtype_str,
            seed=self.seed,
        )
        self.C_occ = _closed_shell_occupation(n_occ, nx, ny, self.dtype, dev)
        self.spin = _make_closed_shell_spin(1, self.N, dev)[0]

        self.f_net = (
            _build_jastrow(self.arch, N=self.N, d=self.d, omega=self.omega, **self.arch_kwargs)
            .to(dev)
            .to(self.dtype)
        )
        if self.use_backflow:
            # BackflowNet / CTNNBackflowNet take d (+ kwargs); they infer N at forward time.
            if self.backflow_arch == "ctnn":
                self.backflow_net = (
                    CTNNBackflowNet(self.d, omega=self.omega, **self.backflow_kwargs)
                    .to(dev).to(self.dtype)
                )
            else:
                self.backflow_net = (
                    BackflowNet(self.d, **self.backflow_kwargs).to(dev).to(self.dtype)
                )

        p = config.get().as_dict()
        p.update(device=dev, torch_dtype=self.dtype, E=config.get().E)
        self.params = p

    # ---- closures ----
    def log_psi(self, x: torch.Tensor) -> torch.Tensor:
        logpsi, _ = psi_fn(
            self.f_net,
            x,
            self.C_occ,
            backflow_net=self.backflow_net,
            spin=self.spin,
            params=self.params,
        )
        return logpsi

    def log_slater(self, x: torch.Tensor) -> torch.Tensor:
        """Bare HO Slater log|det| at x (no backflow) - the non-interacting core."""
        _, logabs = slater_determinant_closed_shell(
            x_config=x, C_occ=self.C_occ, params=self.params, spin=self.spin, normalize=True
        )
        return logabs.view(-1)

    def modules(self) -> list[nn.Module]:
        mods: list[nn.Module] = [self.f_net]
        if self.backflow_net is not None:
            mods.append(self.backflow_net)
        return mods

    def n_params(self) -> int:
        return sum(p.numel() for m in self.modules() for p in m.parameters())

    def train(self) -> None:
        for m in self.modules():
            m.train()

    def eval(self) -> None:
        for m in self.modules():
            m.eval()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        *,
        steps: int = 200,
        burn_in: int = 400,
        sigma: float | None = None,
        init_scale: float = 1.0,
    ) -> torch.Tensor:
        """Draw n configs from |Psi|^2 via random-walk Metropolis (reuses _persistent_rw)."""
        ell = 1.0 / math.sqrt(self.omega)  # oscillator length
        sig = (sigma if sigma is not None else 0.4) * ell
        x = torch.randn(n, self.N, self.d, device=self.device, dtype=self.dtype) * (
            init_scale * ell
        )
        logp = self.log_psi
        x, _, _, _ = _persistent_rw(
            logp, x, steps=burn_in, sigma=sig, adapt=True, target=0.5, adapt_lr=0.05
        )
        x, _, _, _ = _persistent_rw(
            logp, x, steps=steps, sigma=sig, adapt=False, target=0.5, adapt_lr=0.0
        )
        return x.detach()


def make_log_psi_fn(system: System) -> Callable[[torch.Tensor], torch.Tensor]:
    return system.log_psi


def _bf_kwargs(big: bool) -> dict:
    if big:
        return dict(msg_hidden=64, msg_layers=2, hidden=64, layers=3, act="silu",
                    out_bound="tanh", bf_scale_init=0.05, zero_init_last=True)
    return dict(msg_hidden=32, msg_layers=2, hidden=32, layers=2, act="silu",
                out_bound="tanh", bf_scale_init=0.05, zero_init_last=True)


def load_system(checkpoint_path: str, *, device: str | None = None, seed: int | None = 0) -> "System":
    """Rebuild a System from a checkpoint.pt saved by run_phase_analysis (arch + weights)."""
    ck = torch.load(checkpoint_path, map_location="cpu")
    arch = ck["arch"]
    if arch.startswith("ctnn_vcycle"):
        builder = "ctnn_vcycle"
    elif arch.startswith("deepset"):
        builder = "deepset"
    else:
        builder = arch
    use_bf = ck.get("backflow") is not None
    bf_arch = ck.get("backflow_arch", "conv")
    bf_kwargs = dict(ck["backflow_kwargs"]) if ck.get("backflow_kwargs") else _bf_kwargs(use_bf)
    sysm = System(N=int(ck["N"]), omega=float(ck["omega"]), d=2, arch=builder,
                  arch_kwargs=dict(ck["arch_kwargs"]), use_backflow=use_bf,
                  backflow_kwargs=bf_kwargs, backflow_arch=bf_arch, device=device, seed=seed)
    sysm.f_net.load_state_dict(ck["f_net"])
    if use_bf and ck.get("backflow") is not None:
        sysm.backflow_net.load_state_dict(ck["backflow"])
    sysm.eval()
    return sysm
