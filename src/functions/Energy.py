# =========================
# Thorough energy evaluation (multi-laplacian + stable sampler)
# =========================
import math

import torch
from tqdm.auto import tqdm

from utils import inject_params


# --------------------
# Small helper: robust params
# --------------------
def _normalize_device_dtype(p: dict):
    dev = p["device"]
    if isinstance(dev, str):
        dev = torch.device(dev)
        p["device"] = dev
    dt = p.get("torch_dtype", torch.float32)
    if isinstance(dt, str):
        dt = getattr(torch, dt)
        p["torch_dtype"] = dt
    return dev, dt


# --------------------
# ψ wrapper helper (kept lightweight)
# --------------------
def _make_psi_log_fn(f_net, C_occ, *, backflow_net=None, spin=None, params=None):
    device, dtype = _normalize_device_dtype(params)
    f_net.to(device).to(dtype).eval()
    if backflow_net is not None:
        backflow_net.to(device).to(dtype).eval()

    # default closed-shell spin if not provided
    if spin is None:
        N = int(params["n_particles"])
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)
    else:
        spin = spin.to(device)

    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _ = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi  # (B,)

    return psi_log_fn


# --------------------
# Laplacian backends for Δ logψ
# --------------------
def _lap_log_fd_hutch(psi_log_fn, x, probes=16, eps=1e-3):
    x = x.detach()
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        xp = (x + eps * v).detach().requires_grad_(True)
        xm = (x - eps * v).detach().requires_grad_(True)
        with torch.set_grad_enabled(True):
            lp = psi_log_fn(xp)
            gp = torch.autograd.grad(lp.sum(), xp, create_graph=False, retain_graph=False)[0]
            lm = psi_log_fn(xm)
            gm = torch.autograd.grad(lm.sum(), xm, create_graph=False, retain_graph=False)[0]
        acc += ((gp * v).sum(dim=(1, 2)) - (gm * v).sum(dim=(1, 2))) / (2.0 * eps)
    return acc / max(1, probes)


def _lap_log_fd_central_hutch(psi_log_fn, x, probes=16, eps=1e-3):
    x0 = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        l0 = psi_log_fn(x0)
        g0 = torch.autograd.grad(l0.sum(), x0, create_graph=False, retain_graph=False)[0]
    x = x0.detach()
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        xp = (x + eps * v).detach().requires_grad_(True)
        xm = (x - eps * v).detach().requires_grad_(True)
        with torch.set_grad_enabled(True):
            lp = psi_log_fn(xp)
            gp = torch.autograd.grad(lp.sum(), xp, create_graph=False, retain_graph=False)[0]
            lm = psi_log_fn(xm)
            gm = torch.autograd.grad(lm.sum(), xm, create_graph=False, retain_graph=False)[0]
        acc += ((gp - 2.0 * g0 + gm) * v).sum(dim=(1, 2)) / (eps * eps)
    return acc / max(1, probes)


def _lap_log_hvp_hutch(psi_log_fn, x, probes=16):
    x = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        l = psi_log_fn(x)
        g = torch.autograd.grad(l.sum(), x, create_graph=True, retain_graph=True)[0]
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        s = (g * v).sum()
        Hv = torch.autograd.grad(s, x, create_graph=False, retain_graph=True)[0]
        acc += (Hv * v).sum(dim=(1, 2))
    return acc / max(1, probes)


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


# --------------------
# Unified local energy with selectable Laplacian
# --------------------
def _local_energy_multi(
    psi_log_fn,
    x,
    compute_coulomb_interaction,
    omega,
    *,
    lap_mode="hvp-hutch",
    lap_probes=24,
    fd_eps=3e-3,
):
    """
    Returns:
      E_L (B,), T_loc (B,), V_int (B,), V_harm (B,), logψ (B,)
    with the selected Laplacian backend.

    T_loc = -1/2 * [Δ logψ + ||∇ logψ||^2]
    V_harm = 1/2 * ω^2 * ||x||^2
    """
    x = x.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        logpsi = psi_log_fn(x)  # (B,)
        g = torch.autograd.grad(
            logpsi,
            x,
            grad_outputs=torch.ones_like(logpsi),
            create_graph=(lap_mode in ("hvp-hutch", "exact")),
            retain_graph=True,
        )[0]
    g2 = (g * g).sum(dim=(1, 2))

    # Δ logψ
    if lap_mode == "fd-hutch":
        lap_log = _lap_log_fd_hutch(psi_log_fn, x, probes=lap_probes, eps=fd_eps)
    elif lap_mode == "fd-central-hutch":
        lap_log = _lap_log_fd_central_hutch(psi_log_fn, x, probes=lap_probes, eps=fd_eps)
    elif lap_mode == "hvp-hutch":
        lap_log = _lap_log_hvp_hutch(psi_log_fn, x, probes=lap_probes)
    elif lap_mode == "exact":
        lap_log = _lap_log_exact(psi_log_fn, x)
    else:
        raise ValueError(f"Unknown lap_mode: {lap_mode}")

    # Potentials & local parts
    with torch.no_grad():
        V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
        V_int = compute_coulomb_interaction(x).view(-1)  # (B,)
        T_loc = -0.5 * (lap_log + g2)  # (B,)
        E_L = T_loc + V_harm + V_int  # (B,)

    return E_L.detach(), T_loc.detach(), V_int.detach(), V_harm.detach(), logpsi.detach()


# --------------------
# Original Metropolis (kept as-is for backward-compat)
# --------------------
def _metropolis_psi2(
    psi_log_fn, x0, n_steps: int = 30, step_sigma: float = 0.2, return_accept: bool = False
):
    x = x0.clone().requires_grad_(True)
    accepts = 0.0
    with torch.no_grad():
        lp = psi_log_fn(x) * 2.0
        for _ in range(n_steps):
            prop = (x + torch.randn_like(x) * step_sigma).requires_grad_(True)
            lp_prop = psi_log_fn(prop) * 2.0
            accept_logprob = lp_prop - lp
            accept = (torch.rand_like(accept_logprob).log() < accept_logprob).view(-1, 1, 1).float()
            accepts += float(accept.mean().item())
            x = accept * prop + (1.0 - accept) * x
            lp = accept.view(-1) * lp_prop + (1.0 - accept.view(-1)) * lp
    if return_accept:
        return x, accepts / max(1, n_steps)
    return x


# --------------------
# Persistent Metropolis with burn-in/thinning/adaptation
# --------------------
def _metropolis_psi2_persistent(
    psi_log_fn,
    x0,
    *,
    burn_in: int = 200,
    thin: int = 5,
    n_keep: int = 1,
    step_sigma: float = 0.15,
    target_accept: float | None = 0.45,
    adapt_lr: float = 0.05,
):
    x = x0.clone()
    sigma = torch.as_tensor(step_sigma, device=x.device, dtype=x.dtype)
    accepted = 0
    proposals = 0
    with torch.no_grad():
        lp = psi_log_fn(x) * 2.0
        # burn-in
        for _ in range(burn_in):
            prop = x + torch.randn_like(x) * sigma
            lp_prop = psi_log_fn(prop) * 2.0
            logu = torch.log(torch.rand_like(lp_prop))
            acc_m = (logu < (lp_prop - lp)).view(-1, 1, 1)
            accepted += int(acc_m.sum().item())
            proposals += acc_m.numel()
            x = torch.where(acc_m, prop, x)
            lp = torch.where(acc_m.view(-1), lp_prop, lp)
            if target_accept is not None:
                a_hat = acc_m.float().mean().item()
                sigma = sigma * math.exp(adapt_lr * (a_hat - target_accept))
        # collect
        kept = []
        for _ in range(n_keep):
            for _ in range(max(1, thin)):
                prop = x + torch.randn_like(x) * sigma
                lp_prop = psi_log_fn(prop) * 2.0
                logu = torch.log(torch.rand_like(lp_prop))
                acc_m = (logu < (lp_prop - lp)).view(-1, 1, 1)
                accepted += int(acc_m.sum().item())
                proposals += acc_m.numel()
                x = torch.where(acc_m, prop, x)
                lp = torch.where(acc_m.view(-1), lp_prop, lp)
            kept.append(x.clone())
    samples = torch.stack(kept, dim=0).requires_grad_(True)  # (K,B,N,d)
    return samples, x.detach(), accepted, proposals


# --------------------
# Thorough energy estimator (classic + persistent; multi-laplacian)
# --------------------
@inject_params
def evaluate_energy_vmc(
    f_net,
    C_occ,
    *,
    psi_fn,  # your (f_net, x, C_occ, ...) -> (logψ, ψ)
    compute_coulomb_interaction,  # Physics.py function
    backflow_net=None,
    spin=None,
    params=None,  # dict with device, torch_dtype, omega, n_particles, d
    # ---- sampling / batching ----
    n_samples: int = 50_000,
    batch_size: int = 1024,
    sampler_steps: int = 40,
    sampler_step_sigma: float = 0.15,
    init_std_scale: float = 1.0,
    # ---- Laplacian controls ----
    lap_mode: str = "hvp-hutch",
    lap_probes: int = 24,
    fd_eps: float = 3e-3,
    # ---- UI / stats ----
    progress: bool = True,
    ci_level: float = 0.95,
    # ---- Persistent stable mode ----
    persistent: bool = False,
    sampler_burn_in: int = 200,
    sampler_thin: int = 5,
    samples_per_chain: int = 1,
    sampler_target_accept: float | None = 0.45,
    sampler_adapt_lr: float = 0.05,
    # ---- compatibility knobs ----
    assume_omega_invariant: bool = True,
):
    """
    Evaluates component-wise expectations under |Ψ|^2:
      ⟨T⟩, ⟨V_int⟩, ⟨V_trap⟩, and ⟨E⟩ = ⟨T + V_int + V_trap⟩
    Returns means, stds, stderrs, CIs, and acceptance stats.
    """
    assert (
        params is not None
    ), "params dict (device, torch_dtype, omega, n_particles, d) is required."
    device, dtype = _normalize_device_dtype(params)
    omega = float(params["omega"])
    N = int(params["n_particles"])
    d = int(params["d"])

    # spin default (closed shell)
    if spin is None:
        up = N // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(N - up, dtype=torch.long)]
        ).to(device)
    else:
        spin = spin.to(device)

    f_net.to(device).to(dtype).eval()
    if backflow_net is not None:
        backflow_net.to(device).to(dtype).eval()

    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _psi = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi  # (B,)

    # ω-invariant scaling
    ell = 1.0 / math.sqrt(max(omega, 1e-12))
    sigma_phys = sampler_step_sigma * ell if assume_omega_invariant else sampler_step_sigma
    init_std = init_std_scale * ell if assume_omega_invariant else init_std_scale
    fd_eps_phys = fd_eps * ell if assume_omega_invariant else fd_eps

    # running stats
    total = 0
    sum_E = 0.0
    sum_E2 = 0.0
    sum_T = 0.0
    sum_T2 = 0.0
    sum_Vi = 0.0
    sum_Vi2 = 0.0
    sum_Vh = 0.0
    sum_Vh2 = 0.0

    # acceptance stats
    accepted_global = 0
    proposals_global = 0
    acc_avg_simple = 0.0

    pbar = (
        tqdm(total=n_samples, desc=f"Evaluating energy ({lap_mode})", leave=True)
        if progress
        else None
    )

    # persistent chain state
    prev_x = None
    did_burn = False

    while total < n_samples:
        bsz = min(batch_size, n_samples - total)

        if not persistent:
            # classic batch
            x0 = torch.randn(bsz, N, d, device=device, dtype=dtype) * (
                init_std if assume_omega_invariant else 1.0
            )
            x, acc = _metropolis_psi2(
                psi_log_fn, x0, n_steps=sampler_steps, step_sigma=sigma_phys, return_accept=True
            )
            acc_avg_simple += acc

            with torch.set_grad_enabled(True):
                E_L, T_loc, V_int, V_harm, _ = _local_energy_multi(
                    psi_log_fn,
                    x,
                    compute_coulomb_interaction,
                    omega,
                    lap_mode=lap_mode,
                    lap_probes=lap_probes,
                    fd_eps=fd_eps_phys,
                )

            E = E_L.detach().cpu()
            T = T_loc.detach().cpu()
            Vi = V_int.detach().cpu()
            Vh = V_harm.detach().cpu()

            sum_E += float(E.sum().item())
            sum_E2 += float((E * E).sum().item())
            sum_T += float(T.sum().item())
            sum_T2 += float((T * T).sum().item())
            sum_Vi += float(Vi.sum().item())
            sum_Vi2 += float((Vi * Vi).sum().item())
            sum_Vh += float(Vh.sum().item())
            sum_Vh2 += float((Vh * Vh).sum().item())
            total += bsz

            if pbar is not None:
                mean = sum_E / total
                var = max(sum_E2 / total - mean * mean, 0.0)
                acc_disp = acc_avg_simple / max(1, math.ceil(total / bsz))
                pbar.update(bsz)
                pbar.set_postfix_str(f"E≈{mean:.6f}, σ≈{math.sqrt(var):.4f}, acc≈{acc_disp:.2f}")

            del x, E_L, T_loc, V_int, V_harm, E, T, Vi, Vh

        else:
            # persistent batch
            if prev_x is None or prev_x.shape[0] < bsz:
                x0 = torch.randn(bsz, N, d, device=device, dtype=dtype) * init_std
            else:
                x0 = prev_x[:bsz]

            burn = sampler_burn_in if not did_burn else 0
            samples, x_last, accepted, proposals = _metropolis_psi2_persistent(
                psi_log_fn,
                x0,
                burn_in=burn,
                thin=sampler_thin,
                n_keep=samples_per_chain,
                step_sigma=sigma_phys,
                target_accept=sampler_target_accept,
                adapt_lr=sampler_adapt_lr,
            )
            did_burn = True
            prev_x = x_last.detach()

            K = samples.shape[0]
            x_flat = samples.reshape(K * bsz, N, d)

            with torch.set_grad_enabled(True):
                E_L, T_loc, V_int, V_harm, _ = _local_energy_multi(
                    psi_log_fn,
                    x_flat,
                    compute_coulomb_interaction,
                    omega,
                    lap_mode=lap_mode,
                    lap_probes=lap_probes,
                    fd_eps=fd_eps_phys,
                )

            E = E_L.detach()
            T = T_loc.detach()
            Vi = V_int.detach()
            Vh = V_harm.detach()

            batch_sumE = float(E.sum().item())
            batch_sumE2 = float((E * E).sum().item())
            batch_sumT = float(T.sum().item())
            batch_sumT2 = float((T * T).sum().item())
            batch_sumVi = float(Vi.sum().item())
            batch_sumVi2 = float((Vi * Vi).sum().item())
            batch_sumVh = float(Vh.sum().item())
            batch_sumVh2 = float((Vh * Vh).sum().item())

            kept_batch = K * bsz
            sum_E += batch_sumE
            sum_E2 += batch_sumE2
            sum_T += batch_sumT
            sum_T2 += batch_sumT2
            sum_Vi += batch_sumVi
            sum_Vi2 += batch_sumVi2
            sum_Vh += batch_sumVh
            sum_Vh2 += batch_sumVh2
            total += kept_batch

            accepted_global += accepted
            proposals_global += proposals
            acc_disp = accepted_global / max(1, proposals_global)

            if pbar is not None:
                mean = sum_E / total
                var = max(sum_E2 / total - mean * mean, 0.0)
                pbar.update(kept_batch)
                pbar.set_postfix_str(f"E≈{mean:.6f}, σ≈{math.sqrt(var):.4f}, acc={acc_disp:.2f}")

            del samples, x_flat, E_L, T_loc, V_int, V_harm, E, T, Vi, Vh

        torch.cuda.empty_cache()

    if pbar is not None:
        pbar.close()

    # finalize statistics
    def _finish(sum1, sum2, n):
        mean = sum1 / n
        var = max(sum2 / n - mean * mean, 0.0)
        std = math.sqrt(var)
        stderr = std / math.sqrt(n)
        return mean, std, stderr

    E_mean, E_std, E_stderr = _finish(sum_E, sum_E2, total)
    T_mean, T_std, T_stderr = _finish(sum_T, sum_T2, total)
    Vi_mean, Vi_std, Vi_stderr = _finish(sum_Vi, sum_Vi2, total)
    Vh_mean, Vh_std, Vh_stderr = _finish(sum_Vh, sum_Vh2, total)

    def _z_from_alpha(alpha):
        if abs(alpha - 0.95) < 1e-6:
            return 1.96
        if abs(alpha - 0.90) < 1e-6:
            return 1.645
        if abs(alpha - 0.99) < 1e-6:
            return 2.576
        return 1.96

    z = _z_from_alpha(ci_level)
    E_CI = (E_mean - z * E_stderr, E_mean + z * E_stderr)
    T_CI = (T_mean - z * T_stderr, T_mean + z * T_stderr)
    Vi_CI = (Vi_mean - z * Vi_stderr, Vi_mean + z * Vi_stderr)
    Vh_CI = (Vh_mean - z * Vh_stderr, Vh_mean + z * Vh_stderr)

    if persistent:
        acc_out = accepted_global / max(1, proposals_global)
        n_eff = total
    else:
        acc_out = acc_avg_simple / max(1, math.ceil(total / max(1, batch_size)))
        n_eff = total

    return {
        # totals
        "E_mean": E_mean,
        "E_std": E_std,
        "E_stderr": E_stderr,
        "E_CI": E_CI,
        # decomposition
        "T_mean": T_mean,
        "T_std": T_std,
        "T_stderr": T_stderr,
        "T_CI": T_CI,
        "V_int_mean": Vi_mean,
        "V_int_std": Vi_std,
        "V_int_stderr": Vi_stderr,
        "V_int_CI": Vi_CI,
        "V_trap_mean": Vh_mean,
        "V_trap_std": Vh_std,
        "V_trap_stderr": Vh_stderr,
        "V_trap_CI": Vh_CI,
        # sampling stats
        "n_samples_effective": n_eff,
        "accept_rate_avg": acc_out,
    }
