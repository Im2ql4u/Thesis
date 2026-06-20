"""Over-smoothing / non-separability / collectivity probes (see
plans/2026-06-15_oversmoothing-nonseparability.md).

  dirichlet_smoothing : Dirichlet energy of node features across V-cycle stages (over-smoothing) [1,5]
  non_separability    : cross/self sensitivity of node features (full vs message-ablated)        [2]
  spectral_content    : effective k_max / centroid of log|Psi| via a coordinate scan + FFT        [3]
  meanfield_alignment : decode node features to instantaneous-local vs configuration-mean field   [7]
(kappa(S) conditioning [4] = diagnostics.kernel_spectrum; training-speed [6] = separate 2x2 runs.)
"""

from __future__ import annotations

import numpy as np
import torch

from .ablation import ablate_messages, has_messages


# ----------------------------------------------------------------------
# [1,5] over-smoothing: Dirichlet energy of node features across V-cycle stages
# ----------------------------------------------------------------------
@torch.no_grad()
def _capture_nodes(system, x) -> dict:
    net = system.f_net
    cap = {}
    hooks = []
    ne, nd, nu = (getattr(net, n, None) for n in ("node_embed", "node_down", "node_up"))
    if ne is not None:
        hooks.append(ne.register_forward_hook(lambda m, i, o: cap.__setitem__("0_embed", o.detach())))
    if nd is not None:
        hooks.append(nd.register_forward_pre_hook(lambda m, i: cap.__setitem__("1_pre_bottleneck", i[0].detach())))
    if nu is not None:
        hooks.append(nu.register_forward_hook(lambda m, i, o: cap.__setitem__("2_post_bottleneck", o.detach())))
    _ = system.log_psi(x)
    for h in hooks:
        h.remove()
    return cap


def _dirichlet(H: torch.Tensor) -> float:
    """mean_config mean_{i,j} ||h_i - h_j||^2 / mean_i ||h_i||^2  (0 = full consensus)."""
    diff = H.unsqueeze(2) - H.unsqueeze(1)
    d = (diff**2).sum(-1).mean(dim=(1, 2))
    norm = (H**2).sum(-1).mean(1) + 1e-30
    return float((d / norm).mean())


def _node_var(H: torch.Tensor) -> float:
    """across-node feature variance, normalised (individuality; 0 = consensus)."""
    v = H.var(dim=1).mean(-1)            # var across nodes, per config, per channel -> mean
    s = H.pow(2).mean(dim=(1, 2)) + 1e-30
    return float((v / s).mean())


@torch.no_grad()
def dirichlet_smoothing(system, x) -> dict:
    cap = _capture_nodes(system, x)
    stages = sorted(k for k, v in cap.items() if isinstance(v, torch.Tensor) and v.dim() == 3)
    return {
        "dirichlet": {k: _dirichlet(cap[k]) for k in stages},
        "node_var": {k: _node_var(cap[k]) for k in stages},
        "stages": stages,
    }


# ----------------------------------------------------------------------
# [2] non-separability: cross/self sensitivity of node features (full vs ablated)
# ----------------------------------------------------------------------
def _cross_self(system, x, n_probe: int = 4) -> tuple[float, float]:
    net = system.f_net
    nu = getattr(net, "node_up", None)
    if nu is None:
        return float("nan"), float("nan")
    cap = {}
    h = nu.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    xr = x.detach().requires_grad_(True)
    _ = system.log_psi(xr)
    h.remove()
    H = cap["h"]  # (B,N,Hd) in graph
    B, N, Hd = H.shape
    self_acc, cross_acc = [], []
    gen = torch.Generator(device=x.device).manual_seed(0)
    for _p in range(n_probe):
        v = torch.randn(Hd, generator=gen, device=x.device, dtype=x.dtype)
        v = v / v.norm()
        s = H @ v  # (B,N)
        for i in range(N):
            g = torch.autograd.grad(s[:, i].sum(), xr, retain_graph=True)[0]  # (B,N,d) = d s_i / d x_j
            gn = g.norm(dim=-1)  # (B,N)
            self_acc.append(float(gn[:, i].mean()))
            cross_acc.append(float((gn.sum(1) - gn[:, i]).mean() / max(N - 1, 1)))
    return float(np.mean(self_acc)), float(np.mean(cross_acc))


def non_separability(system, x, n_probe: int = 4) -> dict:
    fs, fc = _cross_self(system, x, n_probe)
    with ablate_messages(system.f_net, "zero"):
        as_, ac = _cross_self(system, x, n_probe)
    return {
        "self": fs, "cross": fc, "cross_self_ratio": fc / (fs + 1e-30),
        "self_ablated": as_, "cross_ablated": ac, "cross_self_ratio_ablated": ac / (as_ + 1e-30),
    }


# ----------------------------------------------------------------------
# [3] spectral content of log|Psi| via a 1D coordinate scan + FFT
# ----------------------------------------------------------------------
@torch.no_grad()
def spectral_content(system, x, *, n_base: int = 96, n_scan: int = 96, span_aho: float = 4.0) -> dict:
    """Scan particle-0 x-coordinate through base configs, FFT log|Psi| along the scan, average power.
    Returns spectral centroid and k95 (lower = smoother). k in 1/a_ho units."""
    ell = 1.0 / np.sqrt(system.omega)
    base = x[:n_base].clone()
    span = span_aho * ell
    t = torch.linspace(-span, span, n_scan, device=x.device, dtype=x.dtype)
    curves = torch.empty(base.shape[0], n_scan, device=x.device, dtype=torch.float64)
    for k in range(n_scan):
        cfg = base.clone()
        cfg[:, 0, 0] = t[k]
        curves[:, k] = system.log_psi(cfg).double()
    curves = curves - curves.mean(dim=1, keepdim=True)
    win = torch.hann_window(n_scan, device=x.device, dtype=torch.float64)
    F = torch.fft.rfft(curves * win, dim=1)
    P = (F.abs() ** 2).mean(0).cpu().numpy()  # avg power spectrum
    dt = float((t[1] - t[0]).item())
    kfreq = 2 * np.pi * np.fft.rfftfreq(n_scan, d=dt)  # angular wavenumber (1/a_ho since t in a_ho)
    Psum = P.sum() + 1e-30
    centroid = float((kfreq * P).sum() / Psum)
    cum = np.cumsum(P) / Psum
    k95 = float(kfreq[np.searchsorted(cum, 0.95)]) if (cum >= 0.95).any() else float(kfreq[-1])
    return {"k_centroid": centroid, "k95": k95, "kfreq": kfreq, "power": P}


# ----------------------------------------------------------------------
# [7] mean-field / self-consistency alignment: node features -> local vs collective field
# ----------------------------------------------------------------------
@torch.no_grad()
def meanfield_alignment(system, x) -> dict:
    """R^2 of a linear probe from post-bottleneck node features to (a) the instantaneous per-particle
    Coulomb field, and (b) the configuration-mean (collective) field. Higher (b) relative to (a)
    toward Wigner => the consensus encodes a collective/self-consistent field, not local detail."""
    cap = _capture_nodes(system, x)
    H = cap.get("2_post_bottleneck")
    if H is None:
        return {"available": False}
    B, N, _ = x.shape
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    r = diff.norm(dim=-1)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    r = r.masked_fill(eye, float("inf"))
    field = torch.where(torch.isinf(r), torch.zeros_like(r), 1.0 / r).sum(-1)  # (B,N) local field
    coll = field.mean(dim=1, keepdim=True).expand(B, N)                        # (B,N) collective mean

    Hf = H.reshape(B * N, -1).double().cpu().numpy()
    Hf = np.concatenate([Hf, np.ones((Hf.shape[0], 1))], 1)

    def r2(target):
        y = target.reshape(B * N).double().cpu().numpy()
        beta, *_ = np.linalg.lstsq(Hf, y, rcond=None)
        pred = Hf @ beta
        return 1.0 - ((y - pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-30)

    return {"available": True, "r2_local_field": float(r2(field)),
            "r2_collective_field": float(r2(coll))}
