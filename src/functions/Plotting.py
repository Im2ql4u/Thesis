from tqdm import tqdm
import numpy as np
import torch
import math
import matplotlib.pyplot as plt


def make_mala_sample_fn(
    psi_fn,
    f_net,
    C_occ,
    params,
    *,
    step_size=0.02,  # Langevin step η (tune: 0.01–0.1 typically)
    n_steps=40,  # steps per returned sample
    burn_in=80,  # extra steps before collecting
    thinning=2,  # keep every 'thinning' steps after burn-in
    init_std=1.0,  # std of Gaussian init around origin
    device=None,
    dtype=torch.float32,
):
    """
    Returns sample_fn(batch_size) -> (B, N, 2) approximately ~ |ψ|^2.
    Runs batch_size independent chains in parallel for n_steps (+burn-in).
    """

    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    f_net_local = f_net.to(device=device, dtype=dtype)
    C_occ_local = C_occ.to(device=device, dtype=dtype)

    eps = torch.finfo(dtype).eps

    def logp_and_grad(X):
        # X: (B,N,2), requires_grad True
        psi = psi_fn(f_net_local, X, C_occ_local, params)  # (B,)
        if torch.is_complex(psi):
            amp2 = (psi.real**2 + psi.imag**2) + eps
        else:
            amp2 = (psi**2) + eps
        logp = torch.log(amp2)  # (B,)
        (g,) = torch.autograd.grad(logp.sum(), X, create_graph=False)
        return logp, g

    def log_q(x_to, x_from, grad_from, step):
        # Gaussian proposal with mean = x_from + (step/2)*grad_from, cov = step * I
        mean = x_from + 0.5 * step * grad_from
        diff = x_to - mean
        # log N(diff; 0, step I) up to additive const: -||diff||^2/(2*step)
        return -(diff.pow(2).sum(dim=[1, 2])) / (2.0 * step)

    def sample_fn(batch_size):
        # init positions ~ N(0, init_std^2 I)
        N = params["n_particles"] if "n_particles" in params else params.get("N", None)
        if N is None:
            # try to infer N from C_occ (rows?)
            N = (
                C_occ_local.shape[0]
                if C_occ_local.ndim >= 2
                else int(params.get("N_particles", 2))
            )
        X = init_std * torch.randn(batch_size, N, 2, device=device, dtype=dtype)

        step = torch.as_tensor(step_size, device=device, dtype=dtype)
        # burn-in
        X.requires_grad_(True)
        for _ in range(burn_in):
            logp_x, grad_x = logp_and_grad(X)
            noise = torch.randn_like(X)
            X_prop = X + 0.5 * step * grad_x + torch.sqrt(step) * noise

            # MALA accept
            X_prop.requires_grad_(True)
            logp_y, grad_y = logp_and_grad(X_prop)
            log_acc = (logp_y - logp_x) + (
                log_q(X, X_prop, grad_y, step) - log_q(X_prop, X, grad_x, step)
            )
            u = torch.rand(batch_size, device=device).log()
            accept = u < log_acc
            X = torch.where(accept.view(-1, 1, 1), X_prop, X).detach()
            X.requires_grad_(True)

        # production with thinning; collect last state only (fast & simple)
        for _ in range(n_steps):
            for __ in range(thinning):
                logp_x, grad_x = logp_and_grad(X)
                noise = torch.randn_like(X)
                X_prop = X + 0.5 * step * grad_x + torch.sqrt(step) * noise
                X_prop.requires_grad_(True)
                logp_y, grad_y = logp_and_grad(X_prop)
                log_acc = (logp_y - logp_x) + (
                    log_q(X, X_prop, grad_y, step) - log_q(X_prop, X, grad_x, step)
                )
                u = torch.rand(batch_size, device=device).log()
                accept = u < log_acc
                X = torch.where(accept.view(-1, 1, 1), X_prop, X).detach()
                X.requires_grad_(True)
        return X.detach()

    return sample_fn


# -------------------------
# 2) Radial two-body density n2(r_i, r_j)
# -------------------------


def radial_two_body_density_2d(
    sample_fn,  # () -> (B,N,2) samples ~ |ψ|^2  (on device)
    Rmax=6.0,
    nbins=128,
    batches=200,
    batch_size=1024,
    weight_by_jacobian=True,
    device=None,
):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
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
                (2 * math.pi)
                * torch.clamp(r_i, 1e-6)
                * (2 * math.pi)
                * torch.clamp(r_j, 1e-6)
            )

        ix = bin_index(r_i)
        iy = bin_index(r_j)

        # fill both (ri,rj) and (rj,ri) for symmetry
        H.index_put_((iy, ix), w, accumulate=True)
        H.index_put_((ix, iy), w, accumulate=True)

    H = H / (H.max() + 1e-12)
    extent = [-Rmax, Rmax, -Rmax, Rmax]
    plt.imshow(
        H.t().cpu().numpy(), origin="lower", extent=extent, cmap="magma", aspect="equal"
    )
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sample_fn = make_mala_sample_fn(
        psi_fn,
        f_net,
        C_occ,
        params,
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
