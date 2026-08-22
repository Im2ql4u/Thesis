"""Mechanistic depth diagnostics: open the trained network and the tangent kernel.

Implements the §2.5 depth layer of the kernel-analysis program (D1, D3, D4, D6):

  ntk_eigenmodes        : real-space shapes of the soft (top) vs stiff (bottom) NTK modes (D1)
  update_fields         : where the natural-gradient update lands -- delta_psi_SR vs delta_psi_plain
                          and the residual, all as functions of pair distance (D1)
  effective_coordinate  : perturb parameters along the dominant NTK direction and read the
                          resulting wavefunction change in real space -- the network's discovered
                          collective coordinate (D3)
  decode_hidden         : per-layer activations vs pair distance + effective rank -- circuit decode (D3)
  kernel_cka            : linear CKA between NTK Grams across training checkpoints -- lazy vs rich (D4)
  decode_message        : linear-probe the message / node features to physical local quantities (D6)

Everything is built from the per-sample score matrix O (diagnostics.build_O) or from forward hooks,
and is binned against a per-sample scalar (pair distance) so it can be read in real space. Generic
in N and omega.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def pair_scalar(x: torch.Tensor, mode: str = "min") -> np.ndarray:
    """A single pair-distance scalar per configuration for real-space binning.

    N=2 -> the unique pair distance. N>2 -> nearest-neighbour ('min') or mean pair distance.
    """
    B, N, _ = x.shape
    ii, jj = torch.triu_indices(N, N, offset=1)
    d = (x[:, ii, :] - x[:, jj, :]).norm(dim=-1)  # (B, n_pairs)
    if mode == "min":
        s = d.min(dim=1).values
    elif mode == "mean":
        s = d.mean(dim=1)
    else:
        raise ValueError(mode)
    return s.detach().cpu().double().numpy()


def _bin(r: np.ndarray, y: np.ndarray, nb: int):
    edges = np.quantile(r, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(r, edges[1:-1]), 0, nb - 1)
    rc, yc = [], []
    for b in range(nb):
        m = idx == b
        if np.any(m):
            rc.append(r[m].mean()); yc.append(y[m].mean())
    return np.array(rc), np.array(yc)


# ----------------------------------------------------------------------
# D1 -- NTK eigenmodes and where the natural-gradient update lands
# ----------------------------------------------------------------------
def ntk_eigenmodes(
    O: torch.Tensor, pair_dists: np.ndarray, *, n_top: int = 3, n_bottom: int = 3, nb: int = 30
) -> dict:
    """Real-space shapes of the dominant (soft) and sub-dominant (stiff) NTK eigenmodes.

    The eigenvectors of K = O O^T are functions on the sample points. Binning them vs pair
    distance reveals what the plain gradient pushes (top, soft) vs what it starves (stiff)."""
    Od = O.double()
    K = Od @ Od.t()
    mu, V = torch.linalg.eigh(K)  # ascending
    mu = torch.clamp(mu, min=0.0)
    mu_np = mu.flip(0).cpu().numpy()
    Vd = V.flip(1)  # columns now descending in mu
    mu_max = mu_np[0] if mu_np.size else 0.0
    supp = int((mu_np > mu_max * 1e-10).sum())

    out = {"mu_desc": mu_np, "numerical_rank": supp}
    top, bot = [], []
    rc_ref = None
    for a in range(min(n_top, supp)):
        rc, yc = _bin(pair_dists, Vd[:, a].cpu().double().numpy(), nb)
        rc_ref = rc; top.append(yc)
    for a in range(max(0, supp - n_bottom), supp):
        rc, yc = _bin(pair_dists, Vd[:, a].cpu().double().numpy(), nb)
        bot.append(yc)
    out["r_centers"] = rc_ref if rc_ref is not None else np.array([])
    out["top_modes"] = np.array(top) if top else np.array([])
    out["bottom_modes"] = np.array(bot) if bot else np.array([])
    out["top_mu"] = mu_np[:n_top]
    out["bottom_mu"] = mu_np[max(0, supp - n_bottom):supp]
    return out


def update_fields(
    O: torch.Tensor, E_L: torch.Tensor, pair_dists: np.ndarray, *, nb: int = 30, rel_tol: float = 1e-10
) -> dict:
    """delta(log|Psi|) per configuration under SR vs plain gradient, binned in real space.

    plain step ~ K r  (NTK applied to the residual r = E_L - <E_L>)
    SR step    ~ P_T r (projection of r onto the tangent space)
    Shows *where* (in pair distance) each optimiser changes the wavefunction.
    """
    Od = O.double()
    r = (E_L.detach().double() - E_L.detach().double().mean()).to(Od.device)
    K = Od @ Od.t()
    mu, V = torch.linalg.eigh(K)
    mu = torch.clamp(mu, min=0.0)
    mu_max = float(mu[-1])
    supp = mu > mu_max * rel_tol
    c = V.t() @ r
    psi_sr = V[:, supp] @ c[supp]
    psi_plain = K @ r

    def norm(v):
        v = v.cpu().double().numpy()
        return v / (np.abs(v).max() + 1e-30)

    rc, sr = _bin(pair_dists, norm(psi_sr), nb)
    _, pl = _bin(pair_dists, norm(psi_plain), nb)
    _, rr = _bin(pair_dists, norm(r), nb)
    return {"r_centers": rc, "delta_sr": sr, "delta_plain": pl, "residual": rr}


# ----------------------------------------------------------------------
# D3 -- the effective variational coordinate
# ----------------------------------------------------------------------
@torch.no_grad()
def effective_coordinate(
    O: torch.Tensor, system, x: torch.Tensor, pair_dists: np.ndarray, *, eps: float = 1e-2, nb: int = 30
) -> dict:
    """Perturb parameters along the dominant NTK direction and read the wavefunction response.

    The top right-singular vector of O is the parameter direction the network most easily moves
    the wavefunction along (the eff-rank~1 'knob'). We step theta -> theta + eps*v0, measure
    delta log|Psi|(x), and bin it vs pair distance. That real-space shape *is* the learned
    collective coordinate."""
    params = [p for m in system.modules() for p in m.parameters()]
    theta0 = parameters_to_vector(params).detach().clone()
    # top right singular vector via SVD of O (B,P)
    _, _, Vh = torch.linalg.svd(O.double(), full_matrices=False)
    v0 = Vh[0].to(theta0.dtype)
    v0 = v0 / (v0.norm() + 1e-30)

    log0 = system.log_psi(x).detach()
    vector_to_parameters(theta0 + eps * v0, params)
    logp = system.log_psi(x).detach()
    vector_to_parameters(theta0, params)  # restore

    d = (logp - log0).cpu().double().numpy()
    rc, dc = _bin(pair_dists, d, nb)
    s = torch.linalg.svdvals(O.double()).cpu().numpy()
    return {"r_centers": rc, "delta_logpsi": dc, "singular_values": s}


# ----------------------------------------------------------------------
# D3 -- circuit decode: per-layer activations vs pair distance
# ----------------------------------------------------------------------
@torch.no_grad()
def decode_hidden(system, x: torch.Tensor, pair_dists: np.ndarray, *, nb: int = 30) -> dict:
    """Hook the Jastrow's named submodules; report per-layer effective rank and (for the lowest
    pair distance binning) the channel profiles vs pair distance.

    Best-effort: silently skips layers whose output shape is not (B,*,H)."""
    net = system.f_net
    candidates = ["node_embed", "edge_embed", "node_down", "edge_down", "f_head"]
    hooks, captured = [], {}

    def mk(name):
        def hook(_m, _i, o):
            captured[name] = o.detach() if isinstance(o, torch.Tensor) else None
        return hook

    for name in candidates:
        mod = getattr(net, name, None)
        if mod is not None:
            hooks.append(mod.register_forward_hook(mk(name)))
    _ = system.log_psi(x)
    for h in hooks:
        h.remove()

    out = {}
    for name, act in captured.items():
        if act is None or act.dim() < 2:
            continue
        H = act.shape[-1]
        flat = act.reshape(-1, H).double()  # (B*., H) collapse all but channel
        flat = flat - flat.mean(0, keepdim=True)
        s = torch.linalg.svdvals(flat)
        lam = (s**2).cpu().numpy()
        eff_rank = float((lam.sum() ** 2) / (lam**2).sum()) if lam.sum() > 0 else 0.0
        layer = {"eff_rank": eff_rank, "n_channels": int(H)}
        # per-config channel profiles: reduce middle dims by mean so we have (B, H)
        if act.dim() >= 3:
            per_cfg = act.reshape(act.shape[0], -1, H).mean(1).cpu().double().numpy()
        else:
            per_cfg = act.cpu().double().numpy()
        if per_cfg.shape[0] == pair_dists.shape[0]:
            profs, rc_ref = [], None
            for ch in range(min(H, 8)):
                rc, yc = _bin(pair_dists, per_cfg[:, ch], nb)
                rc_ref = rc; profs.append(yc)
            layer["r_centers"] = rc_ref if rc_ref is not None else np.array([])
            layer["profiles"] = np.array(profs)
        out[name] = layer
    return out


# ----------------------------------------------------------------------
# D4 -- lazy vs rich: NTK drift across training checkpoints
# ----------------------------------------------------------------------
def kernel_cka(O_list: list[torch.Tensor]) -> np.ndarray:
    """Linear CKA between NTK Grams K_t = O_t O_t^T evaluated on the SAME samples.

    Returns a (T,T) matrix; row T-1 (vs final) is the drift curve. CKA~1 across training => lazy
    (fixed feature space); CKA falling => rich (feature learning / NTK rotation)."""
    T = len(O_list)
    Ks = []
    for O in O_list:
        Od = O.double()
        K = Od @ Od.t()
        B = K.shape[0]
        H = torch.eye(B, dtype=K.dtype, device=K.device) - 1.0 / B
        Ks.append(H @ K @ H)
    M = np.eye(T)
    for i in range(T):
        for j in range(i + 1, T):
            num = float((Ks[i] * Ks[j]).sum())
            den = float(Ks[i].norm() * Ks[j].norm()) + 1e-30
            M[i, j] = M[j, i] = num / den
    return M


# ----------------------------------------------------------------------
# D6 -- decode the message: do node/message features encode local physics?
# ----------------------------------------------------------------------
@torch.no_grad()
def physical_local_targets(x: torch.Tensor, spin: torch.Tensor, omega: float, *, radius_aho: float = 1.5):
    """Per-particle physical descriptors to probe for: local density (neighbour count within
    radius), local Coulomb sum_j 1/r_ij, nearest-neighbour distance, same-spin neighbour count."""
    B, N, _ = x.shape
    ell = 1.0 / np.sqrt(omega)
    R = radius_aho * ell
    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
    r = diff.norm(dim=-1)  # (B,N,N)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    r = r.masked_fill(eye, float("inf"))
    dens = (r < R).sum(-1).double()  # (B,N)
    coul = torch.where(torch.isinf(r), torch.zeros_like(r), 1.0 / r).sum(-1).double()
    nn_dist = r.min(-1).values.double()
    sp = spin.to(x.device).long()
    same = (sp.view(1, N, 1) == sp.view(1, 1, N)) & (~eye)
    same_cnt = (same & (r < R)).sum(-1).double()
    return {
        "local_density": dens, "local_coulomb": coul,
        "nn_distance": nn_dist, "same_spin_count": same_cnt,
    }


@torch.no_grad()
def decode_message(system, x: torch.Tensor, *, radius_aho: float = 1.5) -> dict:
    """Linear-probe the post-message node features h_v to physical local descriptors (R^2).

    High R^2 => message passing built that physical quantity. Captures node-level features via a
    hook on the readout-feeding aggregation; falls back to node_embed if the net exposes nothing
    deeper. Meaningful for N>=6 (at N=2 there is a single neighbour)."""
    net = system.f_net
    captured = {}

    def hook(_m, _i, o):
        captured["h"] = o.detach() if isinstance(o, torch.Tensor) else None

    # Prefer the LAST post-message node update (the aggregated message content), then the
    # skip-fusion, then the bare embedding. DeepSet/FFNN have none -> message decode N/A.
    target_mod = None
    for name in ["node_updates_up", "node_updates_down"]:
        m = getattr(net, name, None)
        if m is not None and len(m) > 0:
            target_mod = m[-1]; break
    if target_mod is None:
        for name in ["node_skip_fuse", "node_embed"]:
            m = getattr(net, name, None)
            if m is not None:
                target_mod = m; break
    if target_mod is None:
        return {"available": False}
    h = target_mod.register_forward_hook(hook)
    _ = system.log_psi(x)
    h.remove()
    feat = captured.get("h")
    if feat is None or feat.dim() != 3:
        return {"available": False}
    B, N, H = feat.shape
    X = feat.reshape(B * N, H).double().cpu().numpy()
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # bias

    targets = physical_local_targets(x, system.spin, system.omega, radius_aho=radius_aho)
    r2 = {}
    for name, t in targets.items():
        y = t.reshape(B * N).cpu().numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-30
        r2[name] = 1.0 - ss_res / ss_tot
    return {"available": True, "n_feat": int(H), "r2": r2}
