"""Is the SR step's cost intrinsic, or the per-sample gradient loop that builds O?

`build_O` (via Stochastic_Reconfiguration._score_rows) forms the score matrix O (B x P) with one
backward pass per sample. This times it against a vectorised construction, torch.func.vmap of
torch.func.grad over the samples, checks the two agree, and times the rest of the SR step
(centring, the B x B Gram matrix and its solve) separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call, grad, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.diagnostics import build_O  # noqa: E402
from analysis.system import System, _bf_kwargs  # noqa: E402


class _Ansatz(nn.Module):
    """Registers the System's networks so functional_call can swap their parameters."""

    def __init__(self, system: System) -> None:
        super().__init__()
        self.system = system
        self.nets = nn.ModuleList([m for m in system.modules() if m is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.log_psi(x)


def build_O_vmap(system: System, x: torch.Tensor, chunk: int) -> torch.Tensor:
    model = _Ansatz(system)
    names = [n for n, _ in model.named_parameters()]
    params = {n: p.detach() for n, p in model.named_parameters()}

    def log_psi_one(p: dict, xi: torch.Tensor) -> torch.Tensor:
        return functional_call(model, p, (xi.unsqueeze(0),)).squeeze(0)

    per_sample = vmap(grad(log_psi_one), in_dims=(None, 0))
    rows = []
    for xb in torch.split(x, chunk):
        g = per_sample(params, xb)
        rows.append(torch.cat([g[n].reshape(xb.shape[0], -1) for n in names], dim=1))
    O = torch.cat(rows)
    return O - O.mean(dim=0, keepdim=True)


def _timed(fn, device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--N", type=int, nargs="+", default=[6])
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=1024, help="B, the SR sample count")
    ap.add_argument("--vmap-chunk", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rec = dict(args=vars(a), torch=torch.__version__, results={})
    if a.device.startswith("cuda"):
        rec["gpu"] = torch.cuda.get_device_name(torch.device(a.device))
    for N in a.N:
        torch.manual_seed(0)
        system = System(N=N, omega=a.omega, d=2, arch="ctnn_vcycle", use_backflow=True,
                        backflow_kwargs=_bf_kwargs(True), backflow_arch="ctnn",
                        device=a.device, seed=0)
        x = torch.randn(a.samples, N, 2, device=a.device, dtype=system.dtype) / a.omega ** 0.5
        build_O_vmap(system, x[:8], a.vmap_chunk)   # warm-up / compile
        O_loop, t_loop = _timed(lambda: build_O(system.log_psi, x, system.modules()), a.device)
        O_vmap, t_vmap = _timed(lambda: build_O_vmap(system, x, a.vmap_chunk), a.device)
        rel = float((O_loop - O_vmap).norm() / O_loop.norm())

        def solve() -> torch.Tensor:
            G = O_vmap @ O_vmap.T / a.samples
            G.diagonal().add_(1e-3)
            u = torch.randn(a.samples, device=a.device, dtype=O_vmap.dtype)
            return O_vmap.T @ torch.linalg.solve(G, u) / a.samples
        _, t_solve = _timed(solve, a.device)
        rec["results"][str(N)] = dict(n_params=O_loop.shape[1], t_loop=t_loop, t_vmap=t_vmap,
                                      t_gram_solve=t_solve, rel_diff=rel)
        print(f"N={N:2d} P={O_loop.shape[1]}  O by loop {t_loop:7.2f} s   O by vmap {t_vmap:7.2f} s"
              f"   Gram+solve {t_solve:6.3f} s   |dO|/|O| = {rel:.1e}", flush=True)

    out = Path(f"results/analysis/{date.today().isoformat()}_score_matrix_"
               f"{a.device.replace(':', '')}{('_' + a.tag) if a.tag else ''}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_matrix.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
