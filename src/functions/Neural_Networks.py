# stable_training.py — capped-simplex sampler, clean, shape-safe

import math
from typing import Literal

import torch
from torch import nn

from utils import inject_params

from .Physics import compute_coulomb_interaction
from .Slater_Determinant import slater_determinant_closed_shell


# ------------------ small helpers ------------------
def _hyperspherical_shell(B, N, d, radii, occ_probs, jitter_sigma, device, dtype):
    """
    radii: (K,) absolute radii (already scaled into physical units by caller)
    jitter_sigma: dimensionless, used as relative scale for both radial & tangential
    """
    K = int(radii.numel())
    x = torch.empty(B, N, d, device=device, dtype=dtype)
    occ_probs = torch.nan_to_num(occ_probs, nan=0.0, posinf=0.0, neginf=0.0)
    occ_probs = occ_probs / (occ_probs.sum() + 1e-12)
    for b in range(B):
        m = torch.multinomial(occ_probs, N, replacement=True)  # (N,)
        r_sel = radii[m].to(device=device, dtype=dtype)  # (N,)
        dirs = torch.randn(N, d, device=device, dtype=dtype)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)

        # Relative radial jitter + relative tangential jitter ~ O(jitter_sigma * r)
        rad_jitter = 1.0 + jitter_sigma * torch.randn(N, device=device, dtype=dtype)
        tang = jitter_sigma * r_sel.unsqueeze(-1) * torch.randn(N, d, device=device, dtype=dtype)

        pts = r_sel.unsqueeze(-1) * dirs * rad_jitter.unsqueeze(-1) + tang
        x[b] = pts
    return x


def _k_shell_layout(B, N, d, omega, device, dtype, K=None):
    """
    Heuristic initial shell radii/occupancy given N,d,omega.
    Returns radii(K,) in a_ho units (dimensionless), occ(K,) on CPU float32.
    """
    if K is None:
        K = max(2, int(round(math.sqrt(max(3, N)) / (1.0 if d == 2 else 1.5))))

    # Radii are in a_ho; DO NOT include 1/sqrt(omega) here
    c0 = 0.55 if d == 2 else 0.45
    base_aho = c0 * math.sqrt(N)
    radii_aho = torch.linspace(0.6, 1.15, K) * base_aho

    # Slight outward bias for weak traps; uniform otherwise
    if omega < 0.2:
        idx = torch.arange(K, dtype=torch.float32)
        occ = torch.softmax(0.9 * idx, dim=0)
    else:
        occ = torch.ones(K, dtype=torch.float32) / K

    return radii_aho.float(), occ.float()


@torch.no_grad()
def _per_particle_potentials(x, omega: float, eps_coul_soft: float = 1e-6):
    """
    x: (B,N,d). Returns V_i per particle: (B,N)
    Splits Coulomb pair energy half-half to each partner.
    """
    B, N, d = x.shape
    V_harm_i = 0.5 * (omega**2) * (x**2).sum(dim=-1)  # (B,N)
    diff = x[:, :, None, :] - x[:, None, :, :]  # (B,N,N,d)
    r2 = (diff**2).sum(dim=-1) + (eps_coul_soft**2)  # (B,N,N)
    rinv = torch.where(
        torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0),
        torch.zeros_like(r2),
        1.0 / torch.sqrt(r2),
    )
    V_coul_i = 0.5 * rinv.sum(dim=-1)  # (B,N) half allocation
    return V_harm_i + V_coul_i  # (B,N)


def _make_closed_shell_spin(B: int, N: int, device) -> torch.Tensor:
    if (N % 2) != 0:
        raise ValueError(f"Closed shell requires even N, got N={N}.")
    half = N // 2
    row = torch.cat(
        [
            torch.zeros(half, dtype=torch.long, device=device),
            torch.ones(N - half, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    return row.unsqueeze(0).expand(B, -1)  # (B,N)


def _batch_quantile_mask(vec: torch.Tensor, lo: float = 0.02, hi: float = 0.98) -> torch.Tensor:
    # vec must be (B,)
    assert vec.dim() == 1, f"expected (B,), got {tuple(vec.shape)}"
    B = vec.numel()
    if B < 8 or not (0.0 < lo < 0.5 < hi < 1.0):
        return torch.ones(B, dtype=torch.bool, device=vec.device)
    finite = torch.isfinite(vec)
    if finite.sum() < 8:
        return torch.ones(B, dtype=torch.bool, device=vec.device)
    v = vec[finite]
    q = v.new_tensor([lo, hi])
    ql, qh = torch.quantile(v, q)
    m = (vec >= ql) & (vec <= qh) & finite
    if m.sum().item() == 0:
        m = torch.ones(B, dtype=torch.bool, device=vec.device)
    return m


def _ensure_B1(name: str, t: torch.Tensor) -> torch.Tensor:
    # Allow (B,), (B,1), and (B,1,1)...(trailing singletons)
    if t.dim() == 1:
        return t.unsqueeze(1)
    if t.dim() == 2 and t.shape[1] == 1:
        return t
    if t.dim() >= 2 and all(s == 1 for s in t.shape[1:]):
        return t.reshape(t.shape[0], 1)
    raise RuntimeError(f"{name} must be (B,1) or (B,), got {tuple(t.shape)}")


def _collapse_to_B1(name: str, t: torch.Tensor) -> torch.Tensor:
    B = t.shape[0]
    if t.dim() == 1:
        return t.view(B, 1)
    if t.dim() == 2 and t.shape[1] == 1:
        return t
    return t.view(B, -1).sum(dim=1, keepdim=True)


def _safe_probs(p: torch.Tensor) -> torch.Tensor:
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    s = p.sum()
    if (not torch.isfinite(s)) or (s <= 0):
        return torch.full_like(p, 1.0 / max(1, p.numel()))
    return p / s


def _pairwise_rmin(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rij = torch.cdist(x, x) + eps
    B, N, _ = x.shape
    mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    rij = rij.masked_fill(mask, float("inf"))
    return rij.min(dim=-1).values


def _allocate_counts_with_caps(
    B: int,
    probs: torch.Tensor,  # (K,) on device
    min_frac: torch.Tensor,  # (K,)
    max_frac: torch.Tensor,  # (K,)
) -> torch.Tensor:
    """
    Return integer counts (K,) that sum to B and respect per-component floors/ceilings:
      floor_k = ceil(min_frac[k]*B), ceil_k = floor(max_frac[k]*B)
    Start from largest-remainder rounding of B*probs, then rebalance within caps.
    """
    K = int(probs.numel())
    p = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    p = p / (p.sum() + 1e-12)

    target = p * float(B)
    base = torch.floor(target).to(torch.long)
    rem = int(B - int(base.sum().item()))
    if rem > 0:
        frac = target - base.to(target.dtype)
        order = torch.argsort(frac, descending=True)
        base[order[:rem]] += 1

    floor_ = torch.ceil(min_frac * float(B)).to(torch.long)
    ceil_ = torch.floor(max_frac * float(B)).to(torch.long)
    ceil_ = torch.maximum(ceil_, floor_)
    counts = base.clamp(min=floor_, max=ceil_)

    total = int(counts.sum().item())
    if total < B:
        need = B - total
        headroom = ceil_ - counts
        score = p * headroom.to(p.dtype)
        while need > 0 and int(headroom.sum().item()) > 0:
            k = int(torch.argmax(score).item())
            if headroom[k] > 0:
                counts[k] += 1
                headroom[k] -= 1
                need -= 1
                score[k] = p[k] * headroom[k]
            else:
                score[k] = -1.0
    elif total > B:
        give = total - B
        slack = counts - floor_
        score = slack.to(p.dtype) / (p + 1e-12)
        while give > 0 and int(slack.sum().item()) > 0:
            k = int(torch.argmax(score).item())
            if slack[k] > 0:
                counts[k] -= 1
                slack[k] -= 1
                give -= 1
                score[k] = slack[k] / (p[k] + 1e-12)
            else:
                score[k] = -1.0

    diff = int(counts.sum().item()) - B
    if diff != 0:
        if diff > 0:
            for _ in range(diff):
                k = int(torch.argmax(counts - floor_).item())
                if counts[k] > floor_[k]:
                    counts[k] -= 1
        else:
            for _ in range(-diff):
                k = int(torch.argmax(ceil_ - counts).item())
                if counts[k] < ceil_[k]:
                    counts[k] += 1

    return counts


def _project_simplex_with_caps(
    p: torch.Tensor,  # (K,) nonnegative (not necessarily normalized)
    min_frac: torch.Tensor,  # (K,) lower bounds (e.g., 0)
    max_frac: torch.Tensor,  # (K,) upper bounds (e.g., 0.30)
    tol: float = 1e-10,
    iters: int = 60,
) -> torch.Tensor:
    """
    Returns w in [min_frac, max_frac], sum(w)=1 that is the projection of p
    under clamp(p - tau, min, max), solved by bisection on tau.
    """
    p = p.clone()
    device = p.device
    min_frac = min_frac.to(device=device, dtype=p.dtype)
    max_frac = max_frac.to(device=device, dtype=p.dtype)

    if (max_frac.sum() + 1e-12) < 1.0:
        raise ValueError(f"Infeasible caps: sum(max_frac)={float(max_frac.sum()):.3f} < 1.")
    if (min_frac.sum() - 1e-12) > 1.0:
        raise ValueError(f"Infeasible floors: sum(min_frac)={float(min_frac.sum()):.3f} > 1.")

    lo_tau = float((p - max_frac).min().item())
    hi_tau = float((p - min_frac).max().item())

    for _ in range(iters):
        tau = 0.5 * (lo_tau + hi_tau)
        w = torch.clamp(p - tau, min=min_frac, max=max_frac)
        s = float(w.sum().item())
        if abs(s - 1.0) <= tol:
            return w
        if s > 1.0:
            lo_tau = tau
        else:
            hi_tau = tau

    w = torch.clamp(p - tau, min=min_frac, max=max_frac)
    diff = float(w.sum().item()) - 1.0
    if abs(diff) > 1e-8:
        if diff > 0:
            free = (w > (min_frac + 1e-12)).to(p.dtype)
            cap = (w - min_frac) * free
            cap_sum = float(cap.sum().item())
            if cap_sum > 0:
                w = w - (diff * cap / cap_sum)
        else:
            free = (w < (max_frac - 1e-12)).to(p.dtype)
            cap = (max_frac - w) * free
            cap_sum = float(cap.sum().item())
            if cap_sum > 0:
                w = w + ((-diff) * cap / cap_sum)
        w = torch.clamp(w, min=min_frac, max=max_frac)
    return w


# ------------------ psi_fn ------------------
@inject_params
def psi_fn(
    f_net: nn.Module,
    x_batch: torch.Tensor,
    C_occ: torch.Tensor,
    *,
    backflow_net: nn.Module | None = None,
    spin: torch.Tensor | None = None,
    params=None,
):
    x_batch = x_batch.contiguous()
    B, N, _ = x_batch.shape
    dev = x_batch.device
    C_occ = C_occ.to(device=dev, dtype=x_batch.dtype).contiguous()

    if spin is None:
        spin_bn = _make_closed_shell_spin(B, N, dev)
    else:
        s = spin.to(dev).long()
        spin_bn = s.unsqueeze(0).expand(B, -1) if s.dim() == 1 else s

    x_eff = x_batch + (backflow_net(x_batch, spin=spin_bn) if backflow_net is not None else 0.0)

    sign, logabs = slater_determinant_closed_shell(
        x_config=x_eff, C_occ=C_occ, params=params, spin=spin_bn, normalize=True
    )

    f = f_net(x_batch, spin=spin_bn).squeeze(-1)  # (B,)

    logpsi = logabs.view(-1) + f
    psi = sign.view(-1) * torch.exp(logpsi)
    return logpsi, psi


# ------------------ Laplacian utilities ------------------
def grad_and_laplace_logpsi(psi_log_fn, x, probes: int = 4, fd_eps: float = 1e-4):
    x = x.requires_grad_(True)
    logpsi = psi_log_fn(x)
    g = torch.autograd.grad(logpsi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)

    terms = []
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
        hv = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=True, create_graph=True)[0]

        if not torch.isfinite(hv).all():
            xp = (x + fd_eps * v).requires_grad_(True)
            xm = (x - fd_eps * v).requires_grad_(True)
            gp = torch.autograd.grad(
                psi_log_fn(xp).sum(), xp, retain_graph=True, create_graph=True
            )[0]
            gm = torch.autograd.grad(
                psi_log_fn(xm).sum(), xm, retain_graph=True, create_graph=True
            )[0]
            hv = (gp - gm) / (2.0 * fd_eps)
            hv = torch.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0)

        terms.append((v * hv).sum(dim=(1, 2)))
    lap = torch.stack(terms, dim=0).mean(dim=0).unsqueeze(1)
    return g, lap


def compute_laplacian_fast(psi_only, f_net, x, C_occ, **psi_kwargs):
    x = x.requires_grad_(True)
    B, N, d = x.shape
    Psi = psi_only(f_net, x, C_occ, **psi_kwargs)  # (B,)
    grads = torch.autograd.grad(Psi.sum(), x, create_graph=True, retain_graph=True)[0]
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            g_ij = grads[:, i, j]
            second = torch.autograd.grad(g_ij.sum(), x, create_graph=True, retain_graph=True)[0]
            lap = lap + second[:, i, j]
    return Psi.unsqueeze(1), lap.unsqueeze(1)


def _laplacian_logpsi_fd(psi_log_fn, x_eff, eps: float, probes: int = 2):
    logpsi = psi_log_fn(x_eff)
    grad_logpsi = torch.autograd.grad(logpsi.sum(), x_eff, create_graph=True)[0]
    g2 = (grad_logpsi**2).sum(dim=(1, 2), keepdim=False).unsqueeze(1)

    terms = []
    for _ in range(probes):
        v = torch.empty_like(x_eff).bernoulli_(0.5).mul_(2).add_(-1)
        x_plus = (x_eff + eps * v).requires_grad_(True)
        x_minus = (x_eff - eps * v).requires_grad_(True)

        gp = torch.autograd.grad(
            psi_log_fn(x_plus).sum(), x_plus, create_graph=True, retain_graph=True
        )[0]
        gm = torch.autograd.grad(
            psi_log_fn(x_minus).sum(), x_minus, create_graph=True, retain_graph=True
        )[0]

        terms.append(((gp * v).sum(dim=(1, 2)) - (gm * v).sum(dim=(1, 2))) / (2.0 * eps))
    lap_logpsi = torch.stack(terms, dim=0).mean(dim=0).unsqueeze(1)
    return grad_logpsi, g2, lap_logpsi


def _laplacian_logpsi_exact(psi_log_fn, x: torch.Tensor):
    x = x.requires_grad_(True)
    lp = psi_log_fn(x)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True, retain_graph=True)[0]
    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            gij = g[:, i, j]
            sec = torch.autograd.grad(gij.sum(), x, create_graph=True, retain_graph=True)[0]
            lap += sec[:, i, j]
    g2 = (g**2).sum(dim=(1, 2), keepdim=True)
    return g, g2, lap.view(B, 1)


def psi_only(_f, _x, _C, **kw):
    logpsi, psi = psi_fn(_f, _x, _C, **kw)
    return psi.view(-1)


# ------------------ trainer ------------------
@inject_params
def train_model(
    f_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    C_occ: torch.Tensor,
    *,
    psi_fn,
    lap_mode: Literal["fd-hutch", "hvp-hutch", "exact"] = "exact",
    objective: Literal["residual", "energy", "energy_var"] = "energy_var",
    alpha_start: float = 0.10,
    alpha_end: float = 0.90,
    alpha_decay_frac: float = 0.70,
    adapt_warmup: int = 10,
    eg_eta: float = 0.25,
    eg_temp: float = 1.5,
    eg_momentum: float = 0.8,
    explore_gamma: float = 0.05,
    prob_floor: float = 1e-3,
    rmin_clip: float = 1e-3,
    probes: int = 2,
    fd_eps_scale: float = 1e-3,
    std: float = 2.0,
    N_collocation: int | None = None,
    micro_batch: int = 112,
    grad_clip: float | None = 0.3,
    print_every: int = 50,
    quantile_trim: float = 0.03,
    backflow_net: nn.Module | None = None,
    spin: torch.Tensor | None = None,
    use_huber=True,
    huber_delta=1.0,
    params=None,
):
    device = params["device"]
    dtype = params.get("torch_dtype", None)
    w = float(params["omega"])
    nP = int(params["n_particles"])
    d = int(params["d"])
    n_epochs = int(params["n_epochs"])
    E_DMC = params.get("E", None)
    if objective in ("energy", "energy_var") and (E_DMC is None):
        raise ValueError("objective requires params['E'].")

    if N_collocation is None:
        N_collocation = int(params["N_collocation"])

    f_net.to(device)
    if backflow_net is not None:
        backflow_net.to(device)
    if dtype is not None:
        for p in f_net.parameters():
            p.data = p.data.to(device=device, dtype=dtype)
        if backflow_net is not None:
            for p in backflow_net.parameters():
                p.data = p.data.to(device=device, dtype=dtype)

    if spin is None:
        spin = _make_closed_shell_spin(1, nP, device).squeeze(0)
    else:
        spin = spin.to(device)

    inv_sqrt_om = 1.0 / math.sqrt(max(w, 1e-12))
    QHO_const = 0.5 * (w**2)

    def psi_log_closure(y):
        return psi_fn(f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params)[0]

    def _omega_mult(omega: float) -> float:
        return 1.0 if omega >= 0.9 else (1.2 if omega >= 0.2 else 2.0)

    def _huber(resid, delta: float):
        # resid: (B,); delta > 0
        abs_r = resid.abs()
        quad = 0.5 * (abs_r.clamp(max=delta) ** 2)
        lin = delta * (abs_r - delta).clamp(min=0.0)
        return quad + lin

    # mixture weights: [center, tails, mixed, shell, dimers, clusters]
    user_mix = params.get("sampler_mix_weights", None)
    if user_mix is None:
        mix_w = torch.tensor(
            [0.25, 0.20, 0.25, 0.20, 0.05, 0.05], device=device, dtype=torch.float32
        )
    else:
        mix_w = torch.tensor(
            [
                float(user_mix.get("center", 0.25)),
                float(user_mix.get("tails", 0.20)),
                float(user_mix.get("mixed", 0.25)),
                float(user_mix.get("ring", 0.20)),
                float(user_mix.get("dimers", 0.05)),
                float(user_mix.get("clusters", 0.05)),
            ],
            device=device,
            dtype=torch.float32,
        )
    mix_w = _safe_probs(mix_w)

    def _init_shell_layout(N: int, d: int, omega: float, K: int | None = None):
        if K is None:
            K = max(2, int(round(math.sqrt(max(3, N)) / (1.0 if d == 2 else 1.5))))
        c0 = 0.55 if d == 2 else 0.45
        base_aho = c0 * math.sqrt(N)
        radii_aho = torch.linspace(0.6, 1.15, K, device=device) * base_aho
        if omega < 0.2:
            idx = torch.arange(K, device=device)
            occ = torch.softmax(0.9 * idx, dim=0)
        else:
            occ = torch.full((K,), 1.0 / K, device=device)
        return radii_aho, _safe_probs(occ)

    K_shells = int(params.get("sampler_shells", 0)) or None
    shell_radii_aho, shell_occ = _init_shell_layout(nP, d, w, K_shells)

    jitter_rot = bool(params.get("sampler_rot", True))
    n_perm = int(params.get("sampler_perm", 1))
    hard_enable = bool(params.get("sampler_hard_enable", True))
    hard_q = float(params.get("sampler_hard_q", 0.80))
    hard_cap = int(params.get("sampler_hard_cap", N_collocation))
    hard_inject = float(params.get("sampler_hard_inject", 0.25))
    hard_sigma = float(params.get("sampler_hard_sigma", 0.15))
    pair_pull_p = float(params.get("sampler_pair_pull_prob", 0.5))
    g2_weight = float(params.get("sampler_g2_weight", 0.30))
    cusp_gamma = float(params.get("sampler_cusp_gamma", 0.08))
    eps_coul_soft = float(params.get("eps_coul_soft", 1e-6))

    hard_buf = torch.empty(0, nP, d, device=device, dtype=(dtype or torch.float32))

    def _sample_stratified(
        B,
        N,
        d,
        omega,
        *,
        device,
        dtype,
        mix_w,
        shell_radii_aho,
        shell_occ,
        jitter_rot: bool = True,
        n_perm: int = 0,
        min_frac: torch.Tensor = None,
        max_frac: torch.Tensor = None,
    ):
        inv_s = 1.0 / math.sqrt(max(omega, 1e-12))
        widen = _omega_mult(omega)
        unit_gauss = inv_s * widen
        unit_shell = inv_s

        K = int(mix_w.numel())
        probs = mix_w.detach()
        if min_frac is None:
            min_frac = torch.zeros(K, device=device, dtype=torch.float32)
        if max_frac is None:
            max_frac = torch.full((K,), 1.0, device=device, dtype=torch.float32)

        counts = _allocate_counts_with_caps(B, probs, min_frac=min_frac, max_frac=max_frac)  # (K,)

        comp_id = torch.empty(B, dtype=torch.long, device=device)
        idx = 0
        for cid in range(K):
            n_c = int(counts[cid].item())
            if n_c > 0:
                comp_id[idx : idx + n_c] = cid
                idx += n_c
        if idx < B:
            comp_id[idx:] = 0
        comp_id = comp_id[torch.randperm(B, device=device)]

        x = torch.empty(B, N, d, device=device, dtype=(dtype or torch.float32))

        def _gauss(b, s):
            return torch.randn(b, N, d, device=device, dtype=(dtype or torch.float32)) * (
                s * unit_gauss
            )

        def _log_uniform(lo: float, hi: float, shape=(), *, device, dtype):
            u = torch.empty(shape, device=device, dtype=dtype).uniform_(math.log(lo), math.log(hi))
            return torch.exp(u)

        for cid in range(K):
            sel = comp_id == cid
            b = int(sel.sum().item())
            if b == 0:
                continue
            if cid == 0:  # center
                x[sel] = _gauss(b, 0.20)
            elif cid == 1:  # tails
                x[sel] = _gauss(b, 1.20)
            elif cid == 2:  # mixed
                tmp = _gauss(b, 0.90)
                k = torch.randint(1, N, (b,), device=device)
                for e in range(b):
                    tmp[e, : k[e]] = torch.randn(
                        k[e], d, device=device, dtype=(dtype or torch.float32)
                    ) * (0.25 * unit_gauss)
                    tmp[e] = tmp[e, torch.randperm(N, device=device)]
                x[sel] = tmp
            elif cid == 3:  # shells
                radii = (shell_radii_aho * unit_shell).to(
                    device=device, dtype=(dtype or torch.float32)
                )
                occ = _safe_probs(shell_occ)
                x[sel] = _hyperspherical_shell(
                    b,
                    N,
                    d,
                    radii=radii,
                    occ_probs=occ,
                    jitter_sigma=0.06,
                    device=device,
                    dtype=(dtype or torch.float32),
                )
            elif cid == 4:  # ---- dimers (allow many pairs) ----
                base = _gauss(b, 0.55)
                if N >= 2:
                    # radii scale with ℓ so it's ω-invariant
                    rmin_pair = 0.004 * unit_shell
                    rmax_pair = 0.15 * unit_shell
                    max_pairs = max(1, N // 2)  # let the sampler create multiple close pairs
                    for e in range(b):
                        perm = torch.randperm(N, device=device)
                        npairs = torch.randint(1, max_pairs + 1, (), device=device).item()
                        npairs = min(npairs, N // 2)
                        for t in range(npairs):
                            i, j = perm[2 * t].item(), perm[2 * t + 1].item()
                            u = torch.randn(d, device=device, dtype=(dtype or torch.float32))
                            u = u / (u.norm() + 1e-9)
                            r = _log_uniform(
                                float(rmin_pair),
                                float(rmax_pair),
                                (),
                                device=device,
                                dtype=(dtype or torch.float32),
                            )
                            base[e, j] = base[e, i] + r * u
                x[sel] = base

            elif cid == 5:  # ---- clusters (many-body lumps: triads/quads/...) ----
                # Slightly tighter global spread for the centers
                base = _gauss(b, 0.40)

                # cluster radii scale with ℓ
                rmin_clu = 0.006 * unit_shell
                rmax_clu = 0.10 * unit_shell

                for e in range(b):
                    # Typically 1 cluster; sometimes 2 if N is larger
                    n_clusters = 1 if N < 8 else (1 if torch.rand((), device=device) < 0.6 else 2)

                    # pool of free indices to assign to clusters
                    free = torch.arange(N, device=device)

                    for _ in range(n_clusters):
                        if free.numel() == 0:
                            break
                        # choose a cluster size m in [3, min(6, remaining)]
                        m_lo = 3
                        m_hi = min(6, int(free.numel()))
                        if m_lo > m_hi:
                            break
                        m = int(torch.randint(m_lo, m_hi + 1, (), device=device))

                        # pick m distinct electrons
                        pick = free[torch.randperm(free.numel(), device=device)[:m]]

                        # cluster center: mean of current positions (gives coherent nucleus)
                        c = base[e, pick].mean(dim=0)

                        # place members on a small ball around c with log-uniform radii
                        dirs = torch.randn(m, d, device=device, dtype=(dtype or torch.float32))
                        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
                        radii = _log_uniform(
                            float(rmin_clu),
                            float(rmax_clu),
                            (m, 1),
                            device=device,
                            dtype=(dtype or torch.float32),
                        )
                        base[e, pick] = c + radii * dirs

                        # remove used indices from the pool
                        mask = torch.ones_like(free, dtype=torch.bool)
                        mask[(free.unsqueeze(1) == pick.unsqueeze(0)).any(dim=1)] = False
                        free = free[mask]

                x[sel] = base

        if d == 2 and jitter_rot:
            theta = 2 * math.pi * torch.rand(B, device=device, dtype=(dtype or torch.float32))
            c, s = torch.cos(theta), torch.sin(theta)
            R = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
            x[:, :, :2] = torch.einsum("bnc,bcC->bnC", x[:, :, :2], R)

        if n_perm > 0:
            for _ in range(n_perm):
                perm = torch.stack([torch.randperm(N, device=device) for _ in range(B)], dim=0)
                x = x.gather(dim=1, index=perm.unsqueeze(-1).expand(B, N, d))

        return x, comp_id

    # ---- history ----
    history = []

    # static caps for ALL components
    K_mix = int(mix_w.numel())
    min_frac_cap = torch.zeros(K_mix, device=device, dtype=torch.float32)  # 0.0 floor
    max_frac_cap = torch.full((K_mix,), 0.30, device=device, dtype=torch.float32)  # 0.30 ceiling

    for epoch in range(n_epochs):
        f_net.train()
        backflow_net and backflow_net.train()

        # project current mix_w into capped simplex BEFORE using it
        mix_w = _project_simplex_with_caps(mix_w, min_frac=min_frac_cap, max_frac=max_frac_cap)
        mix_w_used = mix_w.detach().clone()
        shell_occ_used = shell_occ.detach().clone()

        X_base, comp_ids = _sample_stratified(
            B=N_collocation,
            N=nP,
            d=d,
            omega=w,
            device=device,
            dtype=(dtype or torch.float32),
            mix_w=mix_w_used,
            shell_radii_aho=shell_radii_aho,
            shell_occ=shell_occ_used,
            jitter_rot=jitter_rot,
            n_perm=n_perm,
            min_frac=min_frac_cap,
            max_frac=max_frac_cap,
        )

        inj_count = 0
        if hard_enable and (hard_buf.numel() > 0):
            b_inj = min(int(hard_inject * N_collocation), hard_buf.shape[0])
            if b_inj > 0:
                idx_inj = torch.randint(0, hard_buf.shape[0], (b_inj,), device=device)
                out = hard_buf[idx_inj].clone()
                s = 1.0 + 0.10 * torch.randn(b_inj, 1, 1, device=device, dtype=out.dtype)
                out = out * s + (hard_sigma * inv_sqrt_om) * torch.randn_like(out)
                X_base[:b_inj] = out
                comp_ids[:b_inj] = -1
                inj_count = int(b_inj)

        valid = comp_ids >= 0
        bc = torch.bincount(comp_ids[valid], minlength=K_mix).to(torch.int64).tolist()
        comp_names = ["center", "tails", "mixed", "shell", "dimers"] + (
            ["clusters"] if K_mix >= 6 else []
        )
        usage = {name: (bc[i] if i < len(bc) else 0) for i, name in enumerate(comp_names)}
        usage["injected"] = inj_count

        X = X_base
        total_rows, loss_acc = 0, 0.0
        denom = max(1, math.ceil(N_collocation / micro_batch))

        comp_sum = torch.zeros(K_mix, device=device, dtype=torch.float64)
        comp_cnt = torch.zeros(K_mix, device=device, dtype=torch.float64)
        shell_sum = torch.zeros_like(shell_occ, dtype=torch.float64, device=device)
        shell_cnt = torch.zeros_like(shell_occ, dtype=torch.float64, device=device)

        trim = quantile_trim if not (w < 0.2 and quantile_trim > 0.01) else 0.01
        fd_eps_abs = params.get("fd_eps_abs", None)
        if fd_eps_abs is not None:
            fd_eps_abs = float(fd_eps_abs)

        # α schedule
        t = epoch / max(1, n_epochs - 1)
        t_alpha = min(1.0, t / max(1e-8, alpha_decay_frac))
        alpha = alpha_start + 0.5 * (alpha_end - alpha_start) * (1 - math.cos(math.pi * t_alpha))
        alpha = float(max(0.0, min(1.0, alpha)))

        for s in range(0, N_collocation, micro_batch):
            e = min(s + micro_batch, N_collocation)
            x = X[s:e].requires_grad_(True)
            comp_slice = comp_ids[s:e]

            if lap_mode == "exact":
                g, g2, lap_log = _laplacian_logpsi_exact(psi_log_closure, x)
            elif lap_mode == "hvp-hutch":
                g, lap_log = grad_and_laplace_logpsi(psi_log_closure, x, probes=probes, fd_eps=1e-4)
                g2 = (g**2).sum(dim=(1, 2), keepdim=True)
            else:
                eps_used = fd_eps_abs if fd_eps_abs is not None else fd_eps_scale * float(std)
                g, g2, lap_log = _laplacian_logpsi_fd(
                    psi_log_closure, x, eps=float(eps_used), probes=probes
                )

            lap_log = _ensure_B1("lap_log", lap_log)
            g2 = _ensure_B1("g2", g2)

            V_harm_raw = QHO_const * (x**2).sum(dim=(1, 2), keepdim=True)
            V_harm = _collapse_to_B1("V_harm", V_harm_raw)
            V_int = _collapse_to_B1("V_int", compute_coulomb_interaction(x))
            V = _ensure_B1("V", V_harm + V_int)

            T = -0.5 * (lap_log.squeeze(1) + g2.squeeze(1))
            EL = T + V.squeeze(1)

            if trim > 0.0:
                m = _batch_quantile_mask(EL.detach(), lo=trim, hi=1.0 - trim)
                if m.sum().item() == 0:
                    continue
                x, lap_log, g2, V, EL = x[m], lap_log[m], g2[m], V[m], EL[m]
                g = g[m]
                comp_slice = comp_slice[m]
                T = T[m]

            mu = EL.mean().detach()
            if objective == "residual":
                E_eff = mu
            elif objective == "energy":
                E_eff = float(E_DMC)
            else:  # energy_var
                E_eff = alpha * float(E_DMC) + (1.0 - alpha) * mu

            resid = EL - E_eff
            if use_huber:
                loss = _huber(resid, huber_delta).mean()
            else:
                loss = (resid**2).mean()
            (loss / denom).backward()
            loss_acc += float(loss.detach())
            total_rows += EL.numel()

            # per-component adaptation stats
            for cid in range(K_mix):
                sel = comp_slice == cid
                if sel.any():
                    comp_sum[cid] += ((EL.detach()[sel] - E_eff) ** 2).sum().to(comp_sum.dtype)
                    comp_cnt[cid] += sel.sum().to(comp_cnt.dtype)

            # shell difficulty stats
            if (comp_slice == 3).any():
                sel = comp_slice == 3
                xr = x[sel]
                radii = xr.norm(dim=-1)
                shell_radii_phys = shell_radii_aho * inv_sqrt_om
                idx = (radii.unsqueeze(-1) - shell_radii_phys.view(1, 1, -1)).abs().argmin(dim=-1)
                g_sel = g[sel]
                V_i = _per_particle_potentials(xr, w, eps_coul_soft=eps_coul_soft)
                rmin_i = _pairwise_rmin(xr, eps=1e-6).clamp_min(rmin_clip)
                diff_i = V_i + g2_weight * (g_sel.pow(2).sum(dim=-1)) + cusp_gamma * (1.0 / rmin_i)
                for k in range(shell_occ.numel()):
                    mk = idx == k
                    if mk.any():
                        shell_sum[k] += diff_i[mk].sum().to(shell_sum.dtype)
                        shell_cnt[k] += mk.sum().to(shell_cnt.dtype)

        # step
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(f_net.parameters(), grad_clip)
            if backflow_net is not None:
                torch.nn.utils.clip_grad_norm_(backflow_net.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ---- EG updates on mix_w, then PROJECT to capped simplex ----
        if epoch >= adapt_warmup and (comp_cnt.sum() > 0):
            score = torch.where(
                comp_cnt > 0, comp_sum / comp_cnt.clamp_min(1.0), torch.zeros_like(comp_sum)
            ).to(mix_w.dtype)
            score = (score - score.mean()) / (score.std() + 1e-6)
            logits = torch.log(mix_w.clamp_min(1e-12)) + (eg_eta / eg_temp) * score
            p = torch.softmax(logits, dim=0)
            p = eg_momentum * mix_w + (1.0 - eg_momentum) * p
            p = (1.0 - explore_gamma) * p + (explore_gamma / K_mix)
            p = p.clamp_min(prob_floor)
            p = p / p.sum()
            mix_w = _project_simplex_with_caps(p, min_frac=min_frac_cap, max_frac=max_frac_cap)

        # ---- EG update for shells (no caps here, but normalized) ----
        if epoch >= adapt_warmup and (shell_cnt.sum() > 0):
            sh = torch.where(
                shell_cnt > 0, shell_sum / shell_cnt.clamp_min(1.0), torch.zeros_like(shell_sum)
            ).to(shell_occ.dtype)
            sh = (sh - sh.mean()) / (sh.std() + 1e-6)
            logits = torch.log(shell_occ.clamp_min(1e-12)) + (eg_eta / eg_temp) * sh
            q = torch.softmax(logits, dim=0)
            q = eg_momentum * shell_occ + (1.0 - eg_momentum) * q
            q = (1.0 - explore_gamma) * q + (explore_gamma / shell_occ.numel())
            q = q.clamp_min(1e-4)
            shell_occ = q / q.sum()

        # ---- log ----
        with torch.no_grad():
            shell_radii_phys = (shell_radii_aho * inv_sqrt_om).detach()
            epoch_entry = {
                "epoch": int(epoch),
                "objective": str(objective),
                "alpha": float(alpha),
                "mix_w": [float(v) for v in mix_w.tolist()],
                "shell_occ": [float(v) for v in shell_occ.tolist()],
                "shell_radii_phys": [float(v) for v in shell_radii_phys.tolist()],
                "usage": {k: int(v) for k, v in usage.items()},
            }
            history.append(epoch_entry)

        if (epoch % print_every) == 0:
            w_now = ", ".join([f"{v:.2f}" for v in mix_w.tolist()])
            var_now = float((EL).var(unbiased=False).detach().item())
            print(
                f"[ep {epoch:05d} |α={alpha:.3f}|] El={EL.mean().detach():.6f}  var={var_now:.3e}  mix=[{w_now}]"
            )

    return f_net, backflow_net, optimizer, history
