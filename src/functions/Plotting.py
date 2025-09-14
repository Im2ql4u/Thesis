import math
from collections.abc import Mapping
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .Slater_Determinant import slater_determinant_closed_shell


def _coerce_int(d: Mapping[str, Any], keys: tuple[str, ...], default: int) -> int:
    """Return the first key present in d as int,
    accepting ints/floats/numpy scalars; else default."""
    for k in keys:
        v = d.get(k)
        # Py311+ allows union types in isinstance:
        if isinstance(v, int | np.integer | float | np.floating):
            return int(v)
    return default


def make_mala_sample_fn(
    psi_fn,
    f_net,
    C_occ,
    params,
    *,
    backflow_net=None,
    spin: torch.Tensor | None = None,  # (N,) or (B,N)
    step_size: float = 0.02,  # Langevin step (per-dim variance = step_size)
    n_steps: int = 40,
    burn_in: int = 80,
    thinning: int = 2,
    init_std: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    # Burn-in adaptation (optional)
    adapt_step: bool = True,
    target_accept: float = 0.55,  # aim for ~0.35–0.60
    adapt_rate: float = 0.05,  # learning rate for log(step) updates during burn-in
    grad_clip: float | None = None,  # e.g., 10.0 to clip ||grad||
):
    """
    Returns sample_fn(batch_size) -> (B, N, D) ~ |ψ|^2 via MALA using your psi_fn API:
      psi_fn(f_net, x, C_occ, backflow_net=..., spin=..., params=...) -> (logpsi, psi)

    Exposes:
      sample_fn.last_accept: Optional[float]
      sample_fn.step_size: float
    """
    P: Mapping[str, Any] = params if isinstance(params, dict) else vars(params)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    f_net = f_net.to(device=device, dtype=dtype).eval()
    C_occ = C_occ.to(device=device, dtype=dtype)
    if backflow_net is not None:
        backflow_net = backflow_net.to(device=device, dtype=dtype).eval()

    Np = _coerce_int(P, ("n_particles", "N"), 2)
    D = _coerce_int(P, ("d",), 2)

    if spin is None:
        up = Np // 2
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(Np - up, dtype=torch.long)], dim=0
        ).to(device)
    else:
        spin = spin.to(device)

    # keep step as a tensor
    step = torch.tensor(float(step_size), device=device, dtype=dtype)

    def _logp_and_grad(X: torch.Tensor):
        # log p(x) = 2 * logψ(x)
        logpsi, _psi = psi_fn(f_net, X, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        if logpsi.ndim > 1:
            logpsi = logpsi.squeeze(-1)
        logp = 2.0 * logpsi
        (g,) = torch.autograd.grad(logp.sum(), X, create_graph=False, retain_graph=False)
        if grad_clip is not None:
            B = g.shape[0]
            g_flat = g.view(B, -1)
            norms = g_flat.norm(dim=1).clamp_min(1e-12)
            scale = (norms.clamp(max=grad_clip) / norms).view(B, 1, 1)
            g = g * scale
        return logp.detach(), g.detach()

    def _log_q(x_to, x_from, grad_from, step_scalar: torch.Tensor):
        mean = x_from + 0.5 * step_scalar * grad_from
        diff = x_to - mean
        return -(diff.pow(2).sum(dim=(1, 2))) / (2.0 * step_scalar)

    def _accept_move(X, X_prop, logp_x, grad_x, logp_y, grad_y, step_scalar: torch.Tensor):
        log_acc = (logp_y - logp_x) + (
            _log_q(X, X_prop, grad_y, step_scalar) - _log_q(X_prop, X, grad_x, step_scalar)
        )
        u = torch.rand(X.shape[0], device=device).log()
        accept = u < log_acc
        X = X.detach()
        X_prop = X_prop.detach()
        X[accept] = X_prop[accept]
        return X.requires_grad_(True), float(accept.float().mean().item())

    def sample_fn(batch_size: int):
        X = (init_std * torch.randn(batch_size, Np, D, device=device, dtype=dtype)).requires_grad_(
            True
        )

        acc_hist: list[float] = []

        # ---- burn-in (with optional tensor-safe step adaptation) ----
        for _ in range(burn_in):
            logp_x, grad_x = _logp_and_grad(X)
            noise = torch.randn_like(X)
            X_prop = (X + 0.5 * step * grad_x + torch.sqrt(step) * noise).requires_grad_(True)
            logp_y, grad_y = _logp_and_grad(X_prop)
            X, acc_rate = _accept_move(X, X_prop, logp_x, grad_x, logp_y, grad_y, step)
            acc_hist.append(acc_rate)

            if adapt_step:
                # Tensor-safe Robbins–Monro on log(step)
                with torch.no_grad():
                    err_t = torch.as_tensor(acc_rate - target_accept, device=device, dtype=dtype)
                    step.mul_(torch.exp(adapt_rate * err_t)).clamp_(min=1e-6, max=1.0)

        # ---- production with thinning ----
        prod_acc: list[float] = []
        for _ in range(n_steps):
            for __ in range(thinning):
                logp_x, grad_x = _logp_and_grad(X)
                noise = torch.randn_like(X)
                X_prop = (X + 0.5 * step * grad_x + torch.sqrt(step) * noise).requires_grad_(True)
                logp_y, grad_y = _logp_and_grad(X_prop)
                X, acc_rate = _accept_move(X, X_prop, logp_x, grad_x, logp_y, grad_y, step)
                prod_acc.append(acc_rate)

        # Attach attributes on the function object (cast to Any for mypy)
        cast(Any, sample_fn).last_accept = (
            (sum(prod_acc) / max(1, len(prod_acc)))
            if prod_acc
            else (sum(acc_hist) / max(1, len(acc_hist)))
        )
        cast(Any, sample_fn).step_size = float(step.item())

        return X.detach()

    # Initialize attributes so they exist before first call
    cast(Any, sample_fn).last_accept = None
    cast(Any, sample_fn).step_size = float(step.item())

    # Return as Any so downstream attribute access type-checks
    return cast(Any, sample_fn)


# -------------------------
# 2) Radial two-body density n2(r_i, r_j)
# -------------------------


def radial_two_body_density_2d(
    sample_fn,  # () -> (B,N,2) samples ~ |ψ|^2
    Rmax=6.0,
    nbins=128,
    batches=200,
    batch_size=1024,
    weight_by_jacobian=True,
    device=None,
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    H = torch.zeros(nbins, nbins, device=device)
    edges = torch.linspace(0.0, Rmax, nbins + 1, device=device)

    def bin_index(r):
        idx = torch.clamp(((r / Rmax) * nbins).long(), 0, nbins - 1)
        return idx

    for _ in tqdm(range(batches)):
        X = sample_fn(batch_size)  # (B,N,2)
        r = torch.linalg.norm(X, dim=-1)  # (B,N)
        B, N = r.shape

        tri = torch.tril_indices(N, N, offset=-1, device=device)  # (2, P)
        r_i = r[:, tri[0]].reshape(-1)  # (B*P,)
        r_j = r[:, tri[1]].reshape(-1)

        mask = (r_i <= Rmax) & (r_j <= Rmax)
        r_i = r_i[mask]
        r_j = r_j[mask]

        w = torch.ones_like(r_i)
        if weight_by_jacobian:
            # true radial density integrates out angles → divide by (2π r_i)(2π r_j)
            w = w / (
                (2 * math.pi) * torch.clamp(r_i, 1e-6) * (2 * math.pi) * torch.clamp(r_j, 1e-6)
            )

        ix = bin_index(r_i)
        iy = bin_index(r_j)

        # fill both (ri,rj) and (rj,ri) for symmetry
        H.index_put_((iy, ix), w, accumulate=True)
        H.index_put_((ix, iy), w, accumulate=True)

    H = H / (H.max() + 1e-12)
    extent = [-Rmax, Rmax, -Rmax, Rmax]
    plt.imshow(H.t().cpu().numpy(), origin="lower", extent=extent, cmap="magma", aspect="equal")
    plt.colorbar(label=r"$n_2(r_i,r_j)/\max$")
    plt.xlabel(r"$r_i$")
    plt.ylabel(r"$r_j$")
    plt.title("Radial two-body density (2D quantum dot)")
    plt.show()
    return edges.cpu().numpy(), H.cpu().numpy()


# -------------------------
# 3) Example entry-point
# -------------------------


def run_radial_map(
    psi_fn,
    f_net,
    C_occ,
    params,
    *,
    backflow_net=None,  # NEW
    spin: torch.Tensor | None = None,  # NEW
    Rmax=6.0,
    nbins=128,
    step_size=0.02,
    n_steps=40,
    burn_in=80,
    thinning=2,
    init_std=1.0,
    batches=200,
    batch_size=1024,
    dtype=torch.float32,
):
    """
    If backflow_net is provided, the samples (and thus the radial map) reflect |ψ_bf|^2.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sample_fn = make_mala_sample_fn(
        psi_fn,
        f_net,
        C_occ,
        params,
        backflow_net=backflow_net,  # <- wired through
        spin=spin,
        step_size=step_size,
        n_steps=n_steps,
        burn_in=burn_in,
        thinning=thinning,
        init_std=init_std,
        device=device,
        dtype=dtype,
    )

    edges, H = radial_two_body_density_2d(
        sample_fn,
        Rmax=Rmax,
        nbins=nbins,
        batches=batches,
        batch_size=batch_size,
        device=device,
    )
    return edges, H


def mirror_quadrants(H_pos):
    """
    H_pos: (nbins, nbins) with y=r_j rows, x=r_i cols, both in [0,R].
    Returns H_full: (2*nbins, 2*nbins) mirrored to [-R,R]×[-R,R].
    """
    # Left-right mirror (x -> -x), up-down mirror (y -> -y)
    H_xneg = np.fliplr(H_pos)  # (-x, +y)
    H_yneg = np.flipud(H_pos)  # (+x, -y)
    H_both = np.flipud(np.fliplr(H_pos))  # (-x, -y)

    top = np.concatenate([H_xneg, H_pos], axis=1)  # y >= 0
    bottom = np.concatenate([H_both, H_yneg], axis=1)  # y < 0
    H_full = np.concatenate([bottom, top], axis=0)
    return H_full


def construct_grid_configurations(
    n_particles: int,
    dims: int,
    Ngrid: int,
    L: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    fixed_others: torch.Tensor | None = None,
):
    assert dims == 2, "This helper is for 2D coordinates per electron."
    x_vals = torch.linspace(-L, L, Ngrid, device=device, dtype=dtype)
    y_vals = torch.linspace(-L, L, Ngrid, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")  # (Ngrid,Ngrid)

    e0_coords = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=-1)  # (Ntot,2)
    Ntot = e0_coords.shape[0]

    if n_particles > 1:
        if fixed_others is None:
            fixed_others = torch.zeros((n_particles - 1, dims), device=device, dtype=dtype)
        else:
            fixed_others = fixed_others.to(device=device, dtype=dtype)
            assert fixed_others.shape == (n_particles - 1, dims)
        others = fixed_others.unsqueeze(0).expand(Ntot, -1, -1)  # (Ntot, n_particles-1, 2)
        x_configs = torch.cat((e0_coords.unsqueeze(1), others), dim=1)
    else:
        x_configs = e0_coords.unsqueeze(1)  # (Ntot,1,2)

    return x_configs, X.detach().cpu().numpy(), Y.detach().cpu().numpy()


@torch.no_grad()
def plot_f_psi_sd_with_backflow(
    f_net,
    C_occ,
    *,
    backflow_net=None,  # optional; evaluate at x + Δx if provided
    spin: torch.Tensor | None = None,  # (N,) or (B,N); defaults to closed shell if None
    n_particles: int,
    Ngrid: int,
    L: float,
    n_basis_x: int,  # kept for signature parity (basis handled via params)
    n_basis_y: int,  # kept for signature parity (basis handled via params)
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    cmap_main: str = "viridis",
    cmap_ratio: str = "coolwarm",
    clamp_max: float = 30.0,  # stability clamp for exp (applied to 2*log values)
    batch_points: int = 8192,  # chunk size
    show_quiver: bool = True,
    arrow_scale: float = 1.0,  # multiply Δx arrows by this factor
    arrow_flip: bool = False,  # set True if you want arrows reversed
    params=None,  # forwarded to Slater for basis/omega/etc.
    title_prefix: str = "f, |ψ|², and |SD|² (vary electron 0)",
):
    """
    Sweeps electron 0 over a 2D grid and renders:
      - Panel 1: f_net(x_eff)
      - Panel 2: |ψ(x_eff)|^2 where ψ = det(Slater(x_eff)) * exp(f_net(x_eff))
      - Panel 3: |SD(x_eff)|^2   (with SD = det(Slater(x_eff)))
    with x_eff = x + Δx from backflow if backflow_net is provided.
    Quiver shows Δx_0 at each grid point (arrow from x → x+Δx).

    Notes:
      - Uses log-domain for stability: |SD|^2 = exp(2*logabs), |ψ|^2 = exp(2*(logabs+f)).
      - clamp_max applies to the *2×log* exponents to avoid overflow only for plotting.
    """

    # --- grid for electron 0 (others fixed) ---
    # Assumes you have this helper elsewhere in your codebase:
    # construct_grid_configurations(n_particles, dims=2, Ngrid, L, device, dtype)
    x_configs, X, Y = construct_grid_configurations(
        n_particles=n_particles, dims=2, Ngrid=Ngrid, L=L, device=device, dtype=dtype
    )  # (G, N, 2) where G=Ngrid*Ngrid

    # Set eval mode, correct device/dtype
    f_net = f_net.to(device=device, dtype=dtype).eval()
    if backflow_net is not None:
        backflow_net = backflow_net.to(device=device, dtype=dtype).eval()

    C_occ_t = torch.as_tensor(C_occ, device=device, dtype=dtype).contiguous()

    # default closed-shell spin if not provided
    if spin is None:
        up = n_particles // 2
        down = n_particles - up
        spin = torch.cat(
            [torch.zeros(up, dtype=torch.long), torch.ones(down, dtype=torch.long)]
        ).to(device)
    else:
        spin = spin.to(device)

    G = x_configs.shape[0]
    # We'll store f, log|SD|, and Δx_0 (optional)
    f_vals = torch.empty(G, device=device, dtype=dtype)
    logabsSD = torch.empty(G, device=device, dtype=dtype)
    dx0_all = torch.zeros(G, 2, device=device, dtype=dtype) if backflow_net is not None else None

    def _chunks(T, bs):
        for s in range(0, T, bs):
            e = min(T, s + bs)
            yield s, e

    for s, e in _chunks(G, batch_points):
        xb = x_configs[s:e]  # (b, N, 2)

        if backflow_net is not None:
            dx = backflow_net(xb, spin=spin)  # (b,N,2) = Δx
            if arrow_flip:
                dx = -dx
            x_eff = xb + dx
            if dx0_all is not None:
                dx0_all[s:e] = dx[:, 0, :] * arrow_scale
        else:
            x_eff = xb

        # Slater at x_eff (now returns sign, logabs)
        sign, logabs = slater_determinant_closed_shell(
            x_config=x_eff,
            C_occ=C_occ_t,
            params=params,
            normalize=True,
        )  # sign: (b,), logabs: (b,)

        # f_net at x_eff
        f = f_net(x_eff)
        if f.ndim > 1:
            f = f.squeeze(-1)  # (b,)

        f_vals[s:e] = f
        logabsSD[s:e] = logabs

    # Build |ψ|^2 and |SD|^2 from logs for stability
    # Apply clamp to the *2×log* exponents to avoid overflow in visualization.
    two_logpsi = 2.0 * (logabsSD + f_vals)  # (G,)
    two_logSD = 2.0 * logabsSD

    if clamp_max is not None:
        two_logpsi = two_logpsi.clamp(max=clamp_max)
        two_logSD = two_logSD.clamp(max=clamp_max)

    psi2 = torch.exp(two_logpsi)
    sd2 = torch.exp(two_logSD)

    # Reshape to grids
    F = f_vals.reshape(Ngrid, Ngrid).cpu().numpy()
    PSI = psi2.reshape(Ngrid, Ngrid).cpu().numpy()
    SD2 = sd2.reshape(Ngrid, Ngrid).cpu().numpy()

    # Quiver fields (Δx_0), if any
    if dx0_all is not None:
        DX = dx0_all[:, 0].reshape(Ngrid, Ngrid).cpu().numpy()
        DY = dx0_all[:, 1].reshape(Ngrid, Ngrid).cpu().numpy()

    # --- plotting ---
    fig, axs = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)

    im0 = axs[0].pcolormesh(X, Y, F, shading="auto", cmap=cmap_main)
    axs[0].set_title(rf"{title_prefix}\n$\,f_{{\mathrm{{net}}}}(x_\mathrm{{eff}})$")
    axs[0].set_xlabel("x (electron 0)")
    axs[0].set_ylabel("y (electron 0)")
    cb0 = fig.colorbar(im0, ax=axs[0])
    cb0.set_label(r"$f_{\mathrm{net}}$")

    im1 = axs[1].pcolormesh(X, Y, PSI, shading="auto", cmap=cmap_main)
    axs[1].set_title(r"$|\psi(x_\mathrm{eff})|^2 = |SD|^2 \, e^{2f}$")
    axs[1].set_xlabel("x (electron 0)")
    axs[1].set_ylabel("y (electron 0)")
    cb1 = fig.colorbar(im1, ax=axs[1])
    cb1.set_label(r"$|\psi|^2$")

    im2 = axs[2].pcolormesh(X, Y, SD2, shading="auto", cmap=cmap_main)
    axs[2].set_title(r"$|SD(x_\mathrm{eff})|^2$")
    axs[2].set_xlabel("x (electron 0)")
    axs[2].set_ylabel("y (electron 0)")
    cb2 = fig.colorbar(im2, ax=axs[2])
    cb2.set_label(r"$|SD|^2$")

    # Quiver overlay (on panel 3 by default)
    if show_quiver and dx0_all is not None:
        step = max(1, Ngrid // 24)
        Xq = X[::step, ::step]
        Yq = Y[::step, ::step]
        Uq = DX[::step, ::step]
        Vq = DY[::step, ::step]
        axs[2].quiver(Xq, Yq, Uq, Vq, angles="xy", scale_units="xy", scale=1.0, width=0.003)

    plt.show()

    out = {
        "X": X,
        "Y": Y,
        "f": F,
        "psi2": PSI,
        "sd2": SD2,
    }
    if dx0_all is not None:
        out["dx0"] = (DX, DY)
    return out
