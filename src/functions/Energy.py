# =========================
# Thorough energy evaluation
# =========================
import math

import torch
from tqdm.auto import tqdm

from utils import inject_params


# --- (re)use the same FD-Hutch local energy from the SR module ---
# If you already imported it, you can remove this duplicate.
def _local_energy_fd2(psi_log_fn, x, compute_coulomb_interaction, omega, probes=2, eps=1e-3):
    """
    E_L = -1/2 * (Δ logψ + ||∇ logψ||^2) + V(x), with Δ logψ via FD-Hutch (first-order only).
    Returns:
      E_L  : (B,)
      logψ : (B,)
    """
    x = x.requires_grad_(True)

    # ∇ logψ
    logpsi = psi_log_fn(x)  # (B,)
    g = torch.autograd.grad(logpsi.sum(), x, create_graph=True)[0]  # (B,N,d)
    g2 = (g**2).sum(dim=(1, 2))  # (B,)

    # Δ logψ via FD-Hutch
    B = x.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2).add_(-1)
        x_p = (x + eps * v).requires_grad_(True)
        lp = psi_log_fn(x_p)
        gp = torch.autograd.grad(lp.sum(), x_p, create_graph=True)[0]
        x_m = (x - eps * v).requires_grad_(True)
        lm = psi_log_fn(x_m)
        gm = torch.autograd.grad(lm.sum(), x_m, create_graph=True)[0]
        acc += ((gp * v).sum(dim=(1, 2)) - (gm * v).sum(dim=(1, 2))) / (2.0 * eps)

    lap_log = acc / probes

    # Potentials at physical coords x
    V_harm = 0.5 * (omega**2) * (x**2).sum(dim=(1, 2))  # (B,)
    V_int = compute_coulomb_interaction(x).view(-1)  # (B,)
    V = V_harm + V_int

    E_L = -0.5 * (lap_log + g2) + V
    return E_L.detach(), logpsi.detach()


# --- Metropolis sampler that also reports acceptance rate ---
def _metropolis_psi2(
    psi_log_fn, x0, n_steps: int = 30, step_sigma: float = 0.2, return_accept: bool = False
):
    """
    Simple independent-proposal Metropolis for |Ψ|^2. Ensures x fed to psi_fn has requires_grad=True
    (your psi_fn asserts this). Evaluations are under no_grad, so no big graphs are built.
    Returns:
      x_final (B,N,d) [requires_grad=True]
      (optional) accept_rate in [0,1]
    """
    x = x0.clone().requires_grad_(True)
    accepts = 0.0

    with torch.no_grad():
        lp = psi_log_fn(x) * 2.0  # (B,)
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


# --- Thorough energy estimator with tqdm ---
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
    n_samples: int = 50_000,  # total |Ψ|^2 samples to use
    batch_size: int = 1024,  # evaluated per iteration
    sampler_steps: int = 40,  # Metropolis steps per batch
    sampler_step_sigma: float = 0.15,  # proposal std; tune for ~30–70% accept
    fd_probes: int = 4,  # FD-Hutch probes for Δ logψ
    fd_eps: float = 1e-3,  # FD step size (absolute; set ~1e-3 of typical length scale)
    progress: bool = True,
    ci_level: float = 0.95,  # confidence interval level
):
    """
    Evaluates the energy E = ⟨E_L⟩_{|Ψ|^2} with lots of samples, showing a tqdm progress bar.

    Returns a dict with:
      E_mean, E_std, E_stderr, E_CI (tuple), n_samples_effective, accept_rate_avg
    """
    assert (
        params is not None
    ), "params dict (device, torch_dtype, omega, n_particles, d) is required."
    device = params["device"]
    dtype = params.get("torch_dtype", torch.float32)
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

    # nets to device/dtype
    f_net.to(device).to(dtype).eval()
    if backflow_net is not None:
        backflow_net.to(device).to(dtype).eval()

    # wrapper that guarantees requires_grad=True for your psi_fn assert
    def psi_log_fn(x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logpsi, _psi = psi_fn(f_net, x, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi  # (B,)

    # running stats
    total = 0
    sum_E = 0.0
    sum_E2 = 0.0
    acc_accum = 0.0

    pbar = tqdm(total=n_samples, desc="Evaluating energy", leave=True) if progress else None

    while total < n_samples:
        bsz = min(batch_size, n_samples - total)
        # draw initial states, sample to |Ψ|^2
        x0 = torch.randn(bsz, N, d, device=device, dtype=dtype)
        x, acc = _metropolis_psi2(
            psi_log_fn, x0, n_steps=sampler_steps, step_sigma=sampler_step_sigma, return_accept=True
        )

        # local energy on this batch
        E_L, _ = _local_energy_fd2(
            psi_log_fn, x, compute_coulomb_interaction, omega, probes=fd_probes, eps=fd_eps
        )  # (bsz,)

        # update stats
        E_L_cpu = E_L.detach().cpu()
        sum_E += float(E_L_cpu.sum().item())
        sum_E2 += float((E_L_cpu**2).sum().item())
        total += bsz
        acc_accum += acc

        if pbar is not None:
            pbar.update(bsz)
            pbar.set_postfix_str(
                f"E≈{sum_E/total:.6f},"
                f" σ≈{math.sqrt(max(sum_E2/total - (sum_E/total)**2, 0.0)):.4f},"
                f" acc={acc_accum * batch_size / (total):.2f}"
            )

        # free per-batch tensors
        del x, E_L, E_L_cpu

    if pbar is not None:
        pbar.close()

    # finalize statistics
    mean = sum_E / total
    var = max(sum_E2 / total - mean * mean, 0.0)
    std = math.sqrt(var)
    stderr = std / math.sqrt(total)

    # normal approx CI
    # z for ci_level (two-sided): invert CDF of normal; quick approx for 95% => 1.96
    def _z_from_alpha(alpha):
        # crude lookup; replace with scipy if you have it
        return 1.96 if abs(alpha - 0.95) < 1e-6 else 1.645 if abs(alpha - 0.90) < 1e-6 else 2.576

    z = _z_from_alpha(ci_level)
    ci = (mean - z * stderr, mean + z * stderr)

    metrics = {
        "E_mean": mean,
        "E_std": std,
        "E_stderr": stderr,
        "E_CI": ci,
        "n_samples_effective": total,  # raw count; if you thin chains, adjust here
        "accept_rate_avg": acc_accum / max(1, math.ceil(n_samples / batch_size)),
    }
    return metrics
