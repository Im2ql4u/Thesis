from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# ======================================================================================
# 0) Utilities
# ======================================================================================


def _gather_params(mods: list[nn.Module | None]) -> tuple[list[torch.Tensor], torch.Tensor]:
    ps: list[torch.Tensor] = []
    for m in mods:
        if m is None:
            continue
        ps += [p for p in m.parameters() if p.requires_grad]
    flat = parameters_to_vector(ps) if ps else torch.tensor([], device="cpu")
    return ps, flat


@torch.no_grad()
def _persistent_rw(
    psi_log_fn, x, steps: int, sigma: float, adapt: bool, target: float, adapt_lr: float
) -> tuple[torch.Tensor, float, int, int]:
    """Simple RW Metropolis on |ψ|^2 with optional adaptation (first burn-in only)."""
    lp = psi_log_fn(x) * 2.0  # log |ψ|^2
    acc_cnt = 0
    prop_cnt = 0
    sig = float(sigma)
    for _ in range(steps):
        prop = x + torch.randn_like(x) * sig
        lp_prop = psi_log_fn(prop) * 2.0
        logu = torch.log(torch.rand_like(lp_prop))
        acc = logu < (lp_prop - lp)  # (B,)
        x = torch.where(acc.view(-1, 1, 1), prop, x)
        lp = torch.where(acc, lp_prop, lp)
        a_hat = float(acc.float().mean().item())
        acc_cnt += int(acc.sum().item())
        prop_cnt += acc.numel()
        if adapt:
            sig *= math.exp(adapt_lr * (a_hat - target))
            sig = float(min(max(sig, 1e-4), 2.0))
    return x, sig, acc_cnt, prop_cnt


# ======================================================================================
# 1) Laplacian backends and Local Energy (multi-mode)
# ======================================================================================


def _lap_log_fd_hutch(psi_log_fn, x, probes=16, eps=1e-3):
    x = x.detach()
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    eps_t = torch.as_tensor(eps, device=x.device, dtype=x.dtype)
    for _ in range(max(1, int(probes))):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        xp = (x + eps_t * v).requires_grad_(True)
        xm = (x - eps_t * v).requires_grad_(True)
        with torch.set_grad_enabled(True):
            gp = torch.autograd.grad(
                psi_log_fn(xp).sum(), xp, create_graph=False, retain_graph=False
            )[0]
            gm = torch.autograd.grad(
                psi_log_fn(xm).sum(), xm, create_graph=False, retain_graph=False
            )[0]
        acc += ((gp - gm) * v).sum(dim=(1, 2)) / (2.0 * eps_t)
    return acc / float(max(1, int(probes)))


def _lap_log_fd_central_hutch(psi_log_fn, x, probes=16, eps=1e-3):
    x0 = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        g0 = torch.autograd.grad(psi_log_fn(x0).sum(), x0, create_graph=False, retain_graph=False)[
            0
        ]
    x = x0.detach()
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    eps_t = torch.as_tensor(eps, device=x.device, dtype=x.dtype)
    for _ in range(max(1, int(probes))):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        xp = (x + eps_t * v).requires_grad_(True)
        xm = (x - eps_t * v).requires_grad_(True)
        with torch.set_grad_enabled(True):
            gp = torch.autograd.grad(
                psi_log_fn(xp).sum(), xp, create_graph=False, retain_graph=False
            )[0]
            gm = torch.autograd.grad(
                psi_log_fn(xm).sum(), xm, create_graph=False, retain_graph=False
            )[0]
        acc += ((gp - 2.0 * g0 + gm) * v).sum(dim=(1, 2)) / (eps_t * eps_t)
    return acc / float(max(1, int(probes)))


def _lap_log_hvp_hutch(psi_log_fn, x, probes=16):
    # exact Hessian-vector product Hutchinson
    x = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        g = torch.autograd.grad(psi_log_fn(x).sum(), x, create_graph=True, retain_graph=True)[0]
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(max(1, int(probes))):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        s = (g * v).sum()
        Hv = torch.autograd.grad(s, x, create_graph=False, retain_graph=True)[0]
        acc += (Hv * v).sum(dim=(1, 2))
    return acc / float(max(1, int(probes)))


def _lap_log_exact(psi_log_fn, x):
    # full Laplacian by summing second partials — slow, use only for tiny batches
    x = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        g = torch.autograd.grad(psi_log_fn(x).sum(), x, create_graph=True, retain_graph=True)[0]
        B, N, d = x.shape
        lap = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(N):
            for k in range(d):
                gi = g[:, i, k].sum()
                Hiik = torch.autograd.grad(gi, x, retain_graph=True, create_graph=False)[0][:, i, k]
                lap += Hiik
    return lap


def _local_energy_multi(
    psi_log_fn,
    x: torch.Tensor,
    compute_coulomb_interaction,
    omega: float,
    *,
    lap_mode: str = "hvp-hutch",
    lap_probes: int = 16,
    fd_eps: float = 1e-3,
):
    """
    E_L = -1/2 * (Δ logψ + ||∇ logψ||^2) + V(x), with selectable Laplacian:
    lap_mode ∈ {"fd-hutch", "fd-central-hutch", "hvp-hutch", "exact"}
    """
    x = x.requires_grad_(True)
    with torch.set_grad_enabled(True):
        l0 = psi_log_fn(x)
        need_graph = lap_mode in ("hvp-hutch", "exact")
        g = torch.autograd.grad(l0.sum(), x, create_graph=need_graph, retain_graph=True)[0]
    g2 = (g * g).sum(dim=(1, 2))

    if lap_mode == "fd-hutch":
        lap = _lap_log_fd_hutch(psi_log_fn, x, probes=lap_probes, eps=fd_eps)
    elif lap_mode == "fd-central-hutch":
        lap = _lap_log_fd_central_hutch(psi_log_fn, x, probes=lap_probes, eps=fd_eps)
    elif lap_mode == "hvp-hutch":
        lap = _lap_log_hvp_hutch(psi_log_fn, x, probes=lap_probes)
    elif lap_mode == "exact":
        lap = _lap_log_exact(psi_log_fn, x)
    else:
        raise ValueError(f"Unknown lap_mode: {lap_mode}")

    V_harm = 0.5 * (omega**2) * (x * x).sum(dim=(1, 2))
    V_int = compute_coulomb_interaction(x)
    V_int = V_int.view(-1) if V_int.ndim > 1 else V_int
    E_L = -0.5 * (lap + g2) + (V_harm + V_int)
    return E_L.detach(), l0.detach()


# ======================================================================================
# 2) Score matrix per-sample (robust to None grads)
# ======================================================================================


def _score_rows(psi_log_fn, x: torch.Tensor, modules: list[nn.Module | None]):
    """Return score matrix O: (B, P_full) and parameter list."""
    params_list, flat = _gather_params(modules)
    P = flat.numel()
    B = x.shape[0]
    if P == 0:
        return torch.zeros(B, 0, device=x.device, dtype=x.dtype), params_list
    O = torch.zeros(B, P, device=flat.device, dtype=flat.dtype)
    for i in range(B):
        xi = x[i : i + 1].requires_grad_(True)
        li = psi_log_fn(xi)  # (1,)
        if not torch.isfinite(li).all():  # skip non-finite
            continue
        grads = torch.autograd.grad(li, params_list, retain_graph=False, allow_unused=True)
        grads = [
            (g if g is not None else torch.zeros_like(p))
            for g, p in zip(grads, params_list, strict=False)
        ]
        O[i].copy_(parameters_to_vector(grads))
    return O, params_list


# ======================================================================================
# 3) Microbatched SR step with multi-Laplacian + ω-invariant controls
# ======================================================================================


def sr_step_energy_mb(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    *,
    psi_fn,
    compute_coulomb_interaction,
    backflow_net: nn.Module | None = None,
    spin=None,
    params: dict,
    # sampling / chunks
    micro_batch: int = 1200,
    total_rows: int = 9000,
    sampler_steps: int = 90,
    sampler_step_sigma: float = 0.10,  # ω-invariant (scaled by ℓ internally)
    # Laplacian controls
    lap_mode: str = "hvp-hutch",  # "fd-hutch" | "fd-central-hutch" | "hvp-hutch" | "exact"
    fd_probes: int = 12,  # probes for FD/HVP Hutchinson
    fd_eps_scale: float = 1e-3,  # ω-invariant ε; only used for FD modes
    # SR / solver / trust region
    center_O: bool = True,
    damping: float = 1e-2,
    cg_tol: float = 1e-6,
    cg_iters: int = 150,
    restart_every: int = 60,
    step_size: float = 0.010,
    max_param_step: float = 0.010,
    max_damping: float = 5e-1,
    # storage
    store_device: str = "cuda",
    store_dtype: torch.dtype = torch.float64,
    # safety
    do_backtrack: bool = True,
):
    device = params["device"]
    net_dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    ell = 1.0 / math.sqrt(max(omega, 1e-12))  # oscillator length

    # spin default
    if spin is None:
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)

    # nets to device/dtype
    f_net.to(device).to(net_dtype).train()
    if backflow_net is not None:
        backflow_net.to(device).to(net_dtype).train()

    # logψ closure
    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi.view(-1)

    # persistent sampler state
    if not hasattr(sr_step_energy_mb, "_state"):
        sr_step_energy_mb._state = {
            "prev_x": None,
            "sigma": float(sampler_step_sigma * ell),
            "did_burn": False,
        }
    S = sr_step_energy_mb._state

    # eval-mode for MCMC
    was_train = f_net.training
    f_net.eval()
    bf_train = backflow_net.training if backflow_net is not None else False
    if backflow_net is not None:
        backflow_net.eval()

    B0 = int(micro_batch)
    init_std = ell
    x = (
        (torch.randn(B0, N, d, device=device, dtype=net_dtype) * init_std)
        if (S["prev_x"] is None or S["prev_x"].shape[0] < B0)
        else S["prev_x"][:B0]
    )

    # one-time burn-in with adaptation, then thin-only
    target_accept = 0.4
    adapt_lr = 0.05
    burn_in0 = max(400, 20 * sampler_steps)
    thin = max(10, min(20, sampler_steps))
    keep_per = 2

    x, sig, a1, p1 = _persistent_rw(
        psi_log_fn,
        x,
        burn_in0 if not S["did_burn"] else thin,
        S["sigma"],
        adapt=(not S["did_burn"]),
        target=target_accept,
        adapt_lr=adapt_lr,
    )
    S["sigma"] = sig
    S["did_burn"] = True
    S["prev_x"] = x.detach()

    if was_train:
        f_net.train()
    if backflow_net is not None and bf_train:
        backflow_net.train()

    # accumulators (64-bit on CPU for stability)
    O_blocks, E_blocks = [], []
    sumE = torch.zeros((), dtype=torch.float64, device=store_device)
    sumE2 = torch.zeros((), dtype=torch.float64, device=store_device)
    sumO = None
    sumO2 = None
    total = 0
    filtered = 0

    # ε for FD modes (ω-invariant → physical)
    eps_phys = fd_eps_scale * ell

    while total < int(total_rows):
        # thin and keep a few
        keeps = []
        with torch.no_grad():
            for _ in range(keep_per):
                x, _, a2, p2 = _persistent_rw(
                    psi_log_fn,
                    x,
                    thin,
                    S["sigma"],
                    adapt=False,
                    target=target_accept,
                    adapt_lr=adapt_lr,
                )
                keeps.append(x.clone())

        for xk in keeps:
            xk = xk[:B0].detach().to(device=device, dtype=net_dtype).requires_grad_(True)

            # Local energy with selected Laplacian
            with torch.set_grad_enabled(True):
                E_L, _ = _local_energy_multi(
                    psi_log_fn,
                    xk,
                    compute_coulomb_interaction,
                    omega,
                    lap_mode=lap_mode,
                    lap_probes=fd_probes,
                    fd_eps=eps_phys,
                )

            good = torch.isfinite(E_L)
            if not good.all():
                filtered += int((~good).sum().item())
                xk = xk[good]
                E_L = E_L[good]
            if xk.numel() == 0:
                continue

            # robust outlier clamp (MAD)
            with torch.no_grad():
                med = E_L.median()
                mad = (E_L - med).abs().median().clamp_min(1e-12)
                mask = (E_L - med).abs() <= 20.0 * mad
            if not mask.all():
                filtered += int((~mask).sum().item())
                xk = xk[mask]
                E_L = E_L[mask]
            if xk.numel() == 0:
                continue

            # score rows
            O_chunk, params_list = _score_rows(
                psi_log_fn, xk, [f_net] if backflow_net is None else [f_net, backflow_net]
            )
            P = O_chunk.shape[1]
            O_cpu = O_chunk.detach().to(store_device, torch.float64)
            E_cpu = E_L.detach().to(store_device, torch.float64)

            if sumO is None:
                sumO = torch.zeros(P, dtype=torch.float64, device=store_device)
                sumO2 = torch.zeros(P, dtype=torch.float64, device=store_device)

            sumE += E_cpu.sum()
            sumE2 += (E_cpu * E_cpu).sum()
            sumO += O_cpu.sum(dim=0)
            sumO2 += (O_cpu * O_cpu).sum(dim=0)

            O_blocks.append(O_cpu.to(store_dtype))  # store as fp32 to save RAM
            E_blocks.append(E_cpu.to(store_dtype))
            total += int(E_cpu.numel())
            if total >= int(total_rows):
                break

    if total == 0:
        return {
            "E_mean": float("nan"),
            "E_std": float("nan"),
            "step_norm": 0.0,
            "cg_iters": 0,
            "g_norm": 0,
            "kept_params": 0,
            "damping": float(damping),
            "B_eff": 0,
            "filtered": filtered,
            "sigma": float(S["sigma"]),
        }

    B_eff = float(total)
    mu_E = float((sumE / B_eff).item())
    var_E = max(0.0, float((sumE2 / B_eff - (sumE / B_eff) ** 2).item()))
    E_std = var_E**0.5

    # variance of O columns, clamp to avoid 1/sqrt(var) explosions
    meanO = sumO / B_eff
    varO = (sumO2 / B_eff) - (meanO * meanO)
    varO = varO.clamp_min(1e-12)
    v10 = float(torch.quantile(varO, 0.10).item())
    v99 = float(torch.quantile(varO, 0.99).item())
    v_floor = max(v10, 1e-6)
    v_ceil = max(v99, v_floor)
    varO = varO.clamp(min=v_floor, max=v_ceil)

    # drop weakly observed parameters (lowest ~15% variance)
    P_full = int(varO.numel())
    thresh = float(torch.quantile(varO, 0.1).item()) if P_full > 10 else 0.0
    keep_mask = (
        (varO > thresh)
        if P_full > 10
        else torch.ones(P_full, dtype=torch.bool, device=store_device)
    )
    keep_idx = keep_mask.nonzero(as_tuple=False).flatten()
    P_kept = int(keep_idx.numel())
    if P_kept == 0:
        return {
            "E_mean": mu_E,
            "E_std": E_std,
            "step_norm": 0.0,
            "cg_iters": 0,
            "kept_params": 0,
            "damping": float(damping),
            "B_eff": int(B_eff),
            "filtered": filtered,
            "sigma": float(S["sigma"]),
        }

    meanO_k = meanO[keep_idx]
    D_inv_sqrt = varO[keep_idx].rsqrt()

    # whitened blocks
    Ow_blocks = []
    for Oc in O_blocks:
        Ok = Oc[:, keep_idx].to(torch.float64)
        if center_O:
            Ok = Ok - meanO_k
        Ow_blocks.append((Ok * D_inv_sqrt).to(store_dtype))

    # gradient in whitened coords: g_w = 2/B Σ (Ow * (E - mu))
    g_w = torch.zeros(P_kept, dtype=torch.float64, device=store_device)
    for Ow, E in zip(Ow_blocks, E_blocks, strict=False):
        g_w += (Ow.to(torch.float64) * (E.to(torch.float64).view(-1, 1) - mu_E)).sum(dim=0)
    g_w = 2.0 * (g_w / B_eff)

    # matvec in whitened coords
    def A_mv_w(v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(v, dtype=torch.float64)
        for Ow in Ow_blocks:
            Ow64 = Ow.to(torch.float64)
            out += Ow64.T @ (Ow64 @ v)
        return out / B_eff

    # CG solve: (A + λI) x = -g
    lam = float(damping)
    b = -g_w.clone()
    xk = torch.zeros_like(b)
    r = b - (A_mv_w(xk) + lam * xk)
    r0 = float(torch.linalg.norm(r))
    rel_floor = max(float(cg_tol), 2.0 / (B_eff**0.5))
    p = r.clone()
    its = 0
    while its < cg_iters:
        Ap = A_mv_w(p) + lam * p
        alpha = float((r @ r) / (p @ Ap + 1e-20))
        xk = xk + alpha * p
        r_new = r - alpha * Ap
        its += 1
        if float(torch.linalg.norm(r_new)) <= rel_floor * r0:
            r = r_new
            break
        if (its % restart_every) == 0:
            r = r_new
            p = r.clone()
        else:
            beta = float((r_new @ r_new) / (r @ r + 1e-20))
            p = r_new + beta * p
            r = r_new

    # map back to kept/full param space
    delta_kept = D_inv_sqrt * xk
    quad_w = float((xk @ (A_mv_w(xk) + lam * xk)).item())
    quad_w = max(quad_w, 1e-8)
    scale_tr = step_size / (quad_w**0.5)
    norm_dk = float(torch.linalg.norm(delta_kept))
    scale_cap = max_param_step / (norm_dk + 1e-12)
    scale = min(scale_tr, scale_cap)
    step_kept = scale * delta_kept
    step_norm = float(torch.linalg.norm(step_kept))

    # optional backtracking on quadratic model
    if do_backtrack:
        mr = float((g_w @ step_kept) + 0.5 * (step_kept @ (A_mv_w(step_kept) + lam * step_kept)))
        if (not math.isfinite(mr)) or (mr > 0.0):
            lam = min(max_damping, max(2.0 * lam, 1e-2))
            step_kept *= 0.5
            step_norm = float(torch.linalg.norm(step_kept))
            mr = float(
                (g_w @ step_kept) + 0.5 * (step_kept @ (A_mv_w(step_kept) + lam * step_kept))
            )
            if mr > 0.0:
                step_kept *= 0.5
                step_norm = float(torch.linalg.norm(step_kept))

    # expand to full vector and apply
    step_full_cpu = torch.zeros(P_full, dtype=torch.float64, device=store_device)
    step_full_cpu[keep_idx] = step_kept
    step_full = step_full_cpu.to(dtype=net_dtype, device=device)

    params_list = _gather_params([f_net] if backflow_net is None else [f_net, backflow_net])[0]
    theta0 = parameters_to_vector(params_list)
    vector_to_parameters(theta0 + step_full, params_list)

    return {
        "E_mean": mu_E,
        "E_std": E_std,
        "g_norm": float(torch.linalg.norm(g_w).item()),
        "step_norm": step_norm,
        "filtered": int(filtered),
        "cg_iters": int(its),
        "kept_params": int(P_kept),
        "damping": float(lam),
        "B_eff": int(B_eff),
        "sigma": float(S["sigma"]),
    }


# ======================================================================================
# 4) Trainer loop wrapper (microbatch SR; multi-laplacian)
# ======================================================================================


def train_model_sr_energy(
    f_net: nn.Module,
    C_occ: torch.Tensor,
    *,
    psi_fn,
    compute_coulomb_interaction,
    backflow_net: nn.Module | None = None,
    spin=None,
    params: dict,
    n_sr_steps: int = 30000,
    log_every: int = 5,
    # sampling / chunks
    micro_batch: int = 1200,
    total_rows: int = 9000,
    sampler_steps: int = 90,
    sampler_step_sigma: float = 0.10,  # ω-invariant
    # Laplacian controls
    lap_mode: str = "hvp-hutch",  # "fd-hutch" | "fd-central-hutch" | "hvp-hutch" | "exact"
    fd_probes: int = 12,
    fd_eps_scale: float = 1e-3,
    # SR solver / trust region
    step_size: float = 0.010,
    max_param_step: float = 0.010,
    damping: float = 1e-2,
    cg_tol: float = 1e-6,
    cg_iters: int = 150,
    restart_every: int = 60,
    drop_frac_ignored: float = 0.0,  # kept for API compatibility
):
    lap_tag = lap_mode
    for t in range(n_sr_steps):
        info = sr_step_energy_mb(
            f_net,
            C_occ,
            psi_fn=psi_fn,
            compute_coulomb_interaction=compute_coulomb_interaction,
            backflow_net=backflow_net,
            spin=spin,
            params=params,
            micro_batch=micro_batch,
            total_rows=total_rows,
            sampler_steps=sampler_steps,
            sampler_step_sigma=sampler_step_sigma,
            lap_mode=lap_mode,
            fd_probes=fd_probes,
            fd_eps_scale=fd_eps_scale,
            step_size=step_size,
            max_param_step=max_param_step,
            damping=damping,
            cg_tol=cg_tol,
            cg_iters=cg_iters,
            restart_every=restart_every,
            center_O=True,
            do_backtrack=True,
        )

        if (t % log_every) == 0:
            print(
                f"[SR {t:04d}]  E={info['E_mean']:.8f}  σ(E)={info['E_std']:.6f}  "
                f"‖g‖={info['g_norm']:.3e}  ‖Δθ‖={info['step_norm']:.3e}  "
                f"B_eff={info['B_eff']}  damp={info['damping']:.3e}  σ_prop={info['sigma']:.4f}  "
                f"[lap={lap_tag}]"
            )

    return f_net, backflow_net
