from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# ======================================================================================
# 0) Utilities
# ======================================================================================


def _gather_params(mods: list[nn.Module | None]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collect trainable parameters from a list of modules and return (list, flat_vector)."""
    ps: list[torch.Tensor] = []
    for m in mods:
        if m is None:
            continue
        ps += [p for p in m.parameters() if p.requires_grad]
    flat = parameters_to_vector(ps) if ps else torch.tensor([], device="cpu")
    return ps, flat


@torch.no_grad()
def _persistent_rw(
    psi_log_fn,
    x: torch.Tensor,
    steps: int,
    sigma: float,
    *,
    adapt: bool,
    target: float,
    adapt_lr: float,
) -> tuple[torch.Tensor, float, int, int]:
    """
    Random-walk Metropolis on |ψ|^2 with optional Robbins–Monro adaptation (only when adapt=True).
    x: (B, N, d), sigma: proposal std (scalar, same units as x)
    Returns: (new_x, new_sigma, accepted_count, proposed_count)
    """
    lp = psi_log_fn(x) * 2.0  # log |ψ|^2
    acc_cnt = 0
    prop_cnt = 0
    sig = float(sigma)

    for _ in range(int(steps)):
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


@torch.no_grad()
def _persistent_rw_mixture(
    psi_log_fn,
    x: torch.Tensor,
    steps: int,
    sigma_small: float,
    *,
    p_big: float = 0.15,
    sigma_big: float | None = None,  # if None → 3 * sigma_small (updated live)
    use_cauchy_jumps: bool = False,  # symmetric heavy-tail jumps
    cauchy_scale: float = 0.05,  # in same units as x (use ℓ-scaled)
    adapt: bool = False,
    target: float = 0.45,
    adapt_lr: float = 0.05,
) -> tuple[torch.Tensor, float, int, int]:
    """
    RW Metropolis targeting |ψ|^2 with a *mixture* of proposal kernels:
      - small Gaussian step (σ = sigma_small)
      - big Gaussian step (σ = sigma_big, used with prob p_big)
      - optional symmetric Cauchy step (scale = cauchy_scale) on the same grid
    Symmetric proposals ⇒ standard Metropolis ratio π(x')/π(x).
    Only sigma_small is adapted (Robbins–Monro) when adapt=True.
    """
    lp = psi_log_fn(x) * 2.0
    acc_cnt = 0
    prop_cnt = 0
    sig_s = float(sigma_small)
    sig_b = float(3.0 * sig_s) if sigma_big is None else float(sigma_big)

    for _ in range(int(steps)):
        B = x.shape[0]
        # choose which kernel per walker
        use_big = torch.rand(B, device=x.device) < p_big
        use_cauchy = torch.zeros(B, dtype=torch.bool, device=x.device)
        if use_cauchy_jumps:
            # ~half of the big jumps are Cauchy by default
            mask = (torch.rand(B, device=x.device) < 0.5) & use_big
            use_cauchy = mask

        # draw noise
        noise = torch.randn_like(x)
        scale = torch.full((B, 1, 1), sig_s, device=x.device, dtype=x.dtype)
        scale[use_big & ~use_cauchy] = sig_b

        if use_cauchy.any():
            # symmetric heavy-tail increments (no drift)
            cauchy = torch.distributions.Cauchy(
                torch.zeros_like(x[use_cauchy]),
                torch.full_like(x[use_cauchy], cauchy_scale),
            ).rsample()
            prop = x.clone()
            prop[use_cauchy] = prop[use_cauchy] + cauchy
            prop[~use_cauchy] = prop[~use_cauchy] + noise[~use_cauchy] * scale[~use_cauchy]
        else:
            prop = x + noise * scale

        lp_prop = psi_log_fn(prop) * 2.0
        logu = torch.log(torch.rand_like(lp_prop))
        acc = logu < (lp_prop - lp)
        x = torch.where(acc.view(-1, 1, 1), prop, x)
        lp = torch.where(acc, lp_prop, lp)

        a_hat = float(acc.float().mean().item())
        acc_cnt += int(acc.sum().item())
        prop_cnt += acc.numel()

        if adapt:
            sig_s *= math.exp(adapt_lr * (a_hat - target))
            sig_s = float(min(max(sig_s, 1e-4), 2.0))
            # keep big step anchored to small step unless user fixed it
            if sigma_big is None:
                sig_b = 3.0 * sig_s

    return x, sig_s, acc_cnt, prop_cnt


# ======================================================================================
# 1) Laplacian backends and Local Energy (multi-mode)
# ======================================================================================


def _lap_log_fd_hutch(psi_log_fn, x, probes=16, eps=1e-3):
    """Hutchinson on ∆logψ via forward finite differences of ∇logψ."""
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
    """Hutchinson on ∆logψ using central second-difference of ∇logψ (uses one extra ∇ at x)."""
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
    """Hutchinson on ∆logψ via exact HVPs of ∇logψ."""
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
    x = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        l = psi_log_fn(x)  # (B,)
        g = torch.autograd.grad(
            l, x, grad_outputs=torch.ones_like(l), create_graph=True, retain_graph=True
        )[
            0
        ]  # (B,N,d)
        B, N, d = x.shape
        lap = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(N):
            for k in range(d):
                gi = g[:, i, k]  # (B,)
                Hiik = torch.autograd.grad(
                    gi, x, grad_outputs=torch.ones_like(gi), retain_graph=True, create_graph=False
                )[0][
                    :, i, k
                ]  # (B,)
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
    E_L = -1/2 * (∆ logψ + ||∇ logψ||^2) + V(x), with selectable Laplacian:
    lap_mode ∈ {"fd-hutch", "fd-central-hutch", "hvp-hutch", "exact"}.
    Returns (E_L (B,), logψ (B,))
    """
    x = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        logpsi = psi_log_fn(x)  # (B,)
        need_graph = lap_mode in ("hvp-hutch", "exact")
        g = torch.autograd.grad(
            logpsi,
            x,
            grad_outputs=torch.ones_like(logpsi),
            create_graph=need_graph,
            retain_graph=True,
        )[0]
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
    return E_L.detach(), logpsi.detach()


# ======================================================================================
# 2) Score matrix per-sample (robust, fast, no vmap/jvp required)
# ======================================================================================


def _score_rows(
    psi_log_fn,
    x: torch.Tensor,
    modules: list[nn.Module | None],
    *,
    chunk_size: int = 2048,
):
    """
    Build per-sample score matrix O: (B, P) where P = #params(modules).
    Reuses one forward graph per chunk, then cheap backprops (retain_graph=True) per row.
    """
    params_list, flat = _gather_params(modules)
    P = flat.numel()
    B = x.shape[0]

    if P == 0:
        dev = x.device
        dtype = x.dtype
        return torch.zeros(B, 0, device=dev, dtype=dtype), params_list

    dev = flat.device
    dtype = flat.dtype
    O = torch.empty(B, P, device=dev, dtype=dtype)

    for s in range(0, B, chunk_size):
        xb = x[s : s + chunk_size].detach().requires_grad_(True)
        with torch.set_grad_enabled(True):
            lb = psi_log_fn(xb)  # (K,)
            K = lb.shape[0]
            for j in range(K):
                gj = torch.autograd.grad(
                    lb[j], params_list, retain_graph=True, allow_unused=True, create_graph=False
                )
                # robust to None grads
                gj = [
                    (g if g is not None else torch.zeros_like(p))
                    for g, p in zip(gj, params_list, strict=False)
                ]
                O[s + j].copy_(parameters_to_vector(gj))
        del xb, lb  # free graph

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
    sampler_step_sigma: float = 0.10,  # initial σ in units of ℓ
    sampler_sigma_bounds: tuple[float, float] | None = None,  # (lo, hi) in units of ℓ
    # Laplacian controls
    lap_mode: str = "hvp-hutch",  # "fd-hutch" | "fd-central-hutch" | "hvp-hutch" | "exact"
    fd_probes: int = 12,  # probes for FD/HVP Hutchinson (ignored for "exact")
    fd_eps_scale: float = 1e-3,  # ε in units of ℓ for FD modes
    # SR / solver / trust region
    center_O: bool = True,
    damping: float = 1e-2,
    cg_tol: float = 1e-6,
    cg_iters: int = 150,
    restart_every: int = 60,
    step_size: float = 0.010,
    max_param_step: float = 0.010,
    max_damping: float = 5e-1,
    # storage (kept ON DEVICE by default per request)
    store_device: str | torch.device | None = None,
    store_dtype: torch.dtype = torch.float64,
    # safety
    do_backtrack: bool = True,
    # score chunking
    score_chunk_size: int = 2048,
):
    device = params["device"]
    net_dtype = params.get("torch_dtype", torch.float64)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])
    ell = 1.0 / math.sqrt(max(omega, 1e-12))  # oscillator length

    if store_device is None:
        store_device = device  # keep ALL accumulators on the same device

    # spin default
    if spin is None:
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)

    # Nets to device/dtype
    f_net.to(device).to(net_dtype).train()
    if backflow_net is not None:
        backflow_net.to(device).to(net_dtype).train()

    # try compiling only the psi closure (safe fallback)
    psi_forward = psi_fn
    #    if hasattr(torch, "compile"):
    #        try:
    #            psi_forward = torch.compile(psi_fn, dynamic=False, fullgraph=False)  # type: ignore
    #        except Exception:
    #            psi_forward = psi_fn

    # logψ closure
    def psi_log_fn(x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _ = psi_forward(
            f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params
        )
        return logpsi.view(-1)

    # ---------- persistent sampler state (kept on DEVICE) ----------
    key = (N, d, float(omega), str(device), str(net_dtype))
    S = getattr(sr_step_energy_mb, "_state", None)
    if (S is None) or (S.get("key") != key):
        S = {
            "key": key,
            "prev_x": None,  # will hold tensor on `device`
            "sigma": float(sampler_step_sigma * ell),
            "did_burn": False,
        }
        sr_step_energy_mb._state = S

    # Eval mode for sampling (avoid dropout/bn randomness during MCMC)
    was_train = f_net.training
    f_net.eval()
    bf_was_train = False
    if backflow_net is not None:
        bf_was_train = backflow_net.training
        backflow_net.eval()

    # Initialize or reuse chain state
    B0 = int(micro_batch)
    init_std = ell
    if S["prev_x"] is None:
        x = torch.randn(B0, N, d, device=device, dtype=net_dtype) * init_std
    else:
        px = S["prev_x"]
        if px.shape[0] >= B0:
            x = px[:B0].to(device=device, dtype=net_dtype)
        else:
            # grow the batch with fresh noise for the extra rows
            extra = torch.randn(B0 - px.shape[0], N, d, device=device, dtype=net_dtype) * init_std
            x = torch.cat([px.to(device=device, dtype=net_dtype), extra], dim=0)

    # Burn-in & thinning params
    target_accept = 0.4
    adapt_lr = 0.05
    burn_in0 = max(400, 20 * sampler_steps)
    thin = max(10, min(20, sampler_steps))
    keep_per = 2

    acc_t, prop_t = 0, 0

    if not S["did_burn"]:
        x, sig, a1, p1 = _persistent_rw(
            psi_log_fn, x, burn_in0, S["sigma"], adapt=True, target=target_accept, adapt_lr=adapt_lr
        )
        acc_t += a1
        prop_t += p1
        S["sigma"] = sig
        S["did_burn"] = True
    else:
        x, sig, a1, p1 = _persistent_rw(
            psi_log_fn, x, thin, S["sigma"], adapt=False, target=target_accept, adapt_lr=adapt_lr
        )
        acc_t += a1
        prop_t += p1
        S["sigma"] = sig

    # Clamp σ to ω-invariant bounds
    if sampler_sigma_bounds is None:
        lo_hi = (0.06 * ell, 0.14 * ell) if omega >= 0.5 else (0.10 * ell, 0.18 * ell)
    else:
        lo, hi = sampler_sigma_bounds
        lo_hi = (lo * ell, hi * ell)
    S["sigma"] = float(min(max(S["sigma"], lo_hi[0]), lo_hi[1]))

    # Persist the current chain head (on DEVICE)

    # Restore train modes
    if was_train:
        f_net.train()
    if backflow_net is not None and bf_was_train:
        backflow_net.train()

    # Accumulators (ON DEVICE)
    O_blocks: list[torch.Tensor] = []
    E_blocks: list[torch.Tensor] = []
    sumE = torch.zeros((), dtype=torch.float64, device=store_device)
    sumE2 = torch.zeros((), dtype=torch.float64, device=store_device)
    sumO = None
    sumO2 = None
    total = 0
    filtered = 0

    # ε for FD modes (ω-invariant → physical)
    eps_phys = fd_eps_scale * ell

    # Sampling + chunk accumulation
    while total < int(total_rows):
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
                acc_t += a2
                prop_t += p2
                keeps.append(x.clone())

        for xk in keeps:
            xk = xk[:B0].detach().to(device=device, dtype=net_dtype).requires_grad_(True)

            # Local energy
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

            # Filter non-finite
            good = torch.isfinite(E_L)
            if not good.all():
                filtered += int((~good).sum().item())
                xk = xk[good]
                E_L = E_L[good]
            if xk.numel() == 0:
                continue

            # MAD outlier clamp (very permissive)
            with torch.no_grad():
                med = E_L.median()
                mad = (E_L - med).abs().median().clamp_min(1e-12)
                mask = (E_L - med).abs() <= 60.0 * mad
            if not mask.all():
                filtered += int((~mask).sum().item())
                xk = xk[mask]
                E_L = E_L[mask]
            if xk.numel() == 0:
                continue

            # Score rows (per-sample ∂logψ/∂θ)
            modules = [f_net] if backflow_net is None else [f_net, backflow_net]
            O_chunk, params_list = _score_rows(psi_log_fn, xk, modules, chunk_size=score_chunk_size)
            P = O_chunk.shape[1]

            # Accumulate (keep on device)
            O_dev = O_chunk.detach().to(device=store_device, dtype=torch.float64)
            E_dev = E_L.detach().to(device=store_device, dtype=torch.float64)

            if sumO is None:
                sumO = torch.zeros(P, dtype=torch.float64, device=store_device)
                sumO2 = torch.zeros(P, dtype=torch.float64, device=store_device)

            sumE += E_dev.sum()
            sumE2 += (E_dev * E_dev).sum()
            sumO += O_dev.sum(dim=0)
            sumO2 += (O_dev * O_dev).sum(dim=0)

            O_blocks.append(
                O_dev
            )  # could cast to float32 to save VRAM, but keep 64-bit for stability
            E_blocks.append(E_dev)
            total += int(E_dev.numel())
            if total >= int(total_rows):
                break
    S["prev_x"] = x.detach()
    if total == 0:
        return {
            "E_mean": float("nan"),
            "E_std": float("nan"),
            "step_norm": 0.0,
            "cg_iters": 0,
            "g_norm": 0.0,
            "kept_params": 0,
            "damping": float(damping),
            "B_eff": 0,
            "filtered": filtered,
            "sigma": float(S["sigma"]),
            "acc_rate": 0.0,
        }

    B_eff = float(total)
    mu_E = float((sumE / B_eff).item())
    var_E = max(0.0, float((sumE2 / B_eff - (sumE / B_eff) ** 2).item()))
    E_std = var_E**0.5

    # Variance of O columns (for whitening + pruning)
    meanO = sumO / B_eff
    varO = (sumO2 / B_eff) - (meanO * meanO)
    varO = varO.clamp_min(1e-12)
    v10 = float(torch.quantile(varO, 0.01).item())
    v99 = float(torch.quantile(varO, 0.999).item())
    v_floor = max(v10, 1e-8)
    v_ceil = max(v99, v_floor)
    varO = varO.clamp(min=v_floor, max=v_ceil)

    # Drop weakly observed parameters (lowest ~10% variance)
    P_full = int(varO.numel())
    # thresh = float(torch.quantile(varO, 0.10).item()) if P_full > 10 else 0.0
    # keep_mask = (varO > thresh) if P_full > 10 else
    # torch.ones(P_full, dtype=torch.bool, device=store_device)
    # keep_idx = keep_mask.nonzero(as_tuple=False).flatten()
    keep_idx = torch.arange(P_full, device=store_device)  # keep ALL params
    P_kept = int(keep_idx.numel())

    if P_kept == 0:
        return {
            "E_mean": mu_E,
            "E_std": E_std,
            "step_norm": 0.0,
            "cg_iters": 0,
            "g_norm": 0.0,
            "kept_params": 0,
            "damping": float(damping),
            "B_eff": int(B_eff),
            "filtered": filtered,
            "sigma": float(S["sigma"]),
            "acc_rate": float(acc_t) / float(max(prop_t, 1)),
        }

    meanO_k = meanO[keep_idx]
    D_inv_sqrt = varO[keep_idx].rsqrt()

    # Whitened blocks
    Ow_blocks: list[torch.Tensor] = []
    for Oc in O_blocks:
        Ok = Oc[:, keep_idx].to(dtype=torch.float64, device=store_device)
        if center_O:
            Ok = Ok - meanO_k
        Ow_blocks.append(Ok * D_inv_sqrt)

    # Gradient in whitened coords: g_w = 2/B Σ Ow * (E - μ)
    g_w = torch.zeros(P_kept, dtype=torch.float64, device=store_device)
    for Ow, E in zip(Ow_blocks, E_blocks, strict=False):
        g_w += (Ow * (E.view(-1, 1) - mu_E)).sum(dim=0)
    g_w = 2.0 * (g_w / B_eff)

    # Matvec in whitened coords
    def A_mv_w(v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(v, dtype=torch.float64, device=store_device)
        for Ow in Ow_blocks:
            out += Ow.T @ (Ow @ v)
        return out / B_eff

    # Conjugate Gradients: (A + λI) x = -g
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

    # Trust region scaling in whitened coords
    quad_w = float((xk @ (A_mv_w(xk) + lam * xk)).item())
    quad_w = max(quad_w, 1e-12)
    scale_tr = step_size / (quad_w**0.5)
    norm_dk = float(torch.linalg.norm(xk))
    scale_cap = max_param_step / (norm_dk + 1e-12)
    scale = min(scale_tr, scale_cap)
    step_kept = scale * xk
    step_norm = float(torch.linalg.norm(step_kept))

    # Optional backtracking on quadratic model in whitened coords
    if do_backtrack:
        mr = float((g_w @ step_kept) + 0.5 * (step_kept @ (A_mv_w(step_kept) + lam * step_kept)))
        if (not math.isfinite(mr)) or (mr > 0.0):
            lam = min(max_damping, max(1.5 * lam, 1.5e-3))
            step_kept *= 0.5
            step_norm = float(torch.linalg.norm(step_kept))
            mr = float(
                (g_w @ step_kept) + 0.5 * (step_kept @ (A_mv_w(step_kept) + lam * step_kept))
            )
            if mr > 0.0:
                step_kept *= 0.5
                step_norm = float(torch.linalg.norm(step_kept))

    # Expand to full vector and apply to parameters
    step_full_cpu = torch.zeros(P_full, dtype=torch.float64, device=store_device)
    step_full_cpu[keep_idx] = D_inv_sqrt * step_kept  # invert whitening
    step_full = step_full_cpu.to(dtype=net_dtype, device=device)

    params_list = _gather_params([f_net] if backflow_net is None else [f_net, backflow_net])[0]
    theta0 = parameters_to_vector(params_list)
    vector_to_parameters(theta0 + step_full, params_list)

    acc_rate = float(acc_t) / float(max(prop_t, 1))
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
        "acc_rate": acc_rate,
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
    sampler_step_sigma: float = 0.10,  # in units of ℓ
    sampler_sigma_bounds: tuple[float, float] | None = None,  # in units of ℓ
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
    # storage (stay ON DEVICE by default)
    store_device: str | torch.device | None = None,
    store_dtype: torch.dtype = torch.float64,
    # score chunking
    score_chunk_size: int = 2048,
):
    if store_device is None:
        store_device = params["device"]

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
            sampler_sigma_bounds=sampler_sigma_bounds,
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
            store_device=store_device,
            store_dtype=store_dtype,
            score_chunk_size=score_chunk_size,
        )

        if (t % log_every) == 0:
            print(
                f"[SR {t:04d}]  E={info['E_mean']:.8f}  σ(E)={info['E_std']:.6f}  "
                f"‖g‖={info.get('g_norm', 0.0):.3e}  ‖Δθ‖={info['step_norm']:.3e}  "
                f"B_eff={info['B_eff']}  damp={info['damping']:.3e}  σ_prop={info['sigma']:.4f}  "
                f"acc={info.get('acc_rate', 0.0):.3f}  [lap={lap_tag}]"
            )

    return f_net, backflow_net
