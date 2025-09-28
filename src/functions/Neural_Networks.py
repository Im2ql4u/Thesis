# stable_training.py — clean, shape-safe, same function names

import math
from typing import Literal

import torch
from torch import nn

from utils import inject_params

from .Physics import compute_coulomb_interaction
from .Slater_Determinant import slater_determinant_closed_shell

# ------------------ small helpers ------------------


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


def _rowmask_finite(*tensors: torch.Tensor) -> torch.Tensor:
    m = torch.ones(tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device)
    for t in tensors:
        m &= torch.isfinite(t).view(t.shape[0], -1).all(dim=1)
    return m


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


# ------------------ psi_fn (same name) ------------------


@inject_params
def psi_fn(
    f_net: nn.Module,
    x_batch: torch.Tensor,
    C_occ: torch.Tensor,
    *,
    backflow_net: nn.Module | None = None,
    spin: torch.Tensor | None = None,  # (N,) or (B,N)
    params=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logpsi : (B,)
      psi    : (B,)
    """
    x_batch = x_batch.contiguous()
    B, N, _ = x_batch.shape
    dev = x_batch.device
    C_occ = C_occ.to(device=dev, dtype=x_batch.dtype).contiguous()

    # spin -> (B,N)
    if spin is None:
        spin_bn = _make_closed_shell_spin(B, N, dev)
    else:
        s = spin.to(dev).long()
        if s.dim() == 1:
            if s.numel() != N:
                raise ValueError(f"spin has {s.numel()} entries but N={N}.")
            spin_bn = s.unsqueeze(0).expand(B, -1)
        elif s.dim() == 2:
            if s.shape != (B, N):
                raise ValueError(f"spin shape {tuple(s.shape)} != (B,N)=({B},{N}).")
            spin_bn = s
        else:
            raise ValueError("spin must be (N,) or (B,N).")

    # backflow
    x_eff = x_batch + (backflow_net(x_batch, spin=spin_bn) if backflow_net is not None else 0.0)
    if not x_eff.requires_grad:
        x_eff = x_eff.requires_grad_(True)

    # Slater
    sign, logabs = slater_determinant_closed_shell(
        x_config=x_eff, C_occ=C_occ, params=params, spin=spin_bn, normalize=True
    )

    # guard
    """
    with torch.no_grad():
        bad = (~torch.isfinite(logabs)) | (sign == 0)
        if bad.any():
            logabs[bad] = torch.as_tensor(-1e6, dtype=logabs.dtype, device=logabs.device)
            sign[bad]   = torch.as_tensor( 1.0, dtype=sign.dtype,    device=sign.device)
    """
    # Jastrow/correlator; center to stabilize
    f = f_net(x_eff, spin=spin_bn).squeeze(-1)  # (B,)

    logpsi = logabs.view(-1) + f
    psi = sign.view(-1) * torch.exp(logpsi)
    return logpsi, psi


# ------------------ Laplacian utilities (same names) ------------------
def grad_and_laplace_logpsi(psi_log_fn, x, probes: int = 4, fd_eps: float = 1e-4):
    """
    psi_log_fn(x)->(B,)
    Returns:
      grad_logpsi: (B,N,d)
      lap_logpsi : (B,1)
    """
    x = x.requires_grad_(True)
    logpsi = psi_log_fn(x)  # (B,)
    g = torch.autograd.grad(logpsi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)

    terms = []  # collect, don't do in-place on a non-grad tensor
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

        terms.append((v * hv).sum(dim=(1, 2)))  # (B,)

    lap = torch.stack(terms, dim=0).mean(dim=0).unsqueeze(1)  # (B,1), requires_grad=True
    return g, lap


def compute_laplacian_fast(psi_only, f_net, x, C_occ, **psi_kwargs):
    """
    Exact Laplacian of ψ via nested autograd.
    Returns:
      Psi: (B,1), Laplacian: (B,1)
    """
    x = x.requires_grad_(True)
    B, N, d = x.shape
    Psi = psi_only(f_net, x, C_occ, **psi_kwargs)  # (B,)
    grads = torch.autograd.grad(Psi.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            g_ij = grads[:, i, j]
            second = torch.autograd.grad(g_ij.sum(), x, create_graph=True, retain_graph=True)[0]
            lap = lap + second[:, i, j]
    return Psi.unsqueeze(1), lap.unsqueeze(1)


def _laplacian_logpsi_fd(psi_log_fn, x_eff, eps: float, probes: int = 2):
    """
    Returns:
      grad_logpsi : (B,N,d)
      g2          : (B,1)
      lap_logpsi  : (B,1)
    """
    logpsi = psi_log_fn(x_eff)  # (B,)
    grad_logpsi = torch.autograd.grad(logpsi.sum(), x_eff, create_graph=True)[0]  # (B,N,d)
    g2 = (grad_logpsi**2).sum(dim=(1, 2), keepdim=False).unsqueeze(1)  # (B,1)

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

        terms.append(((gp * v).sum(dim=(1, 2)) - (gm * v).sum(dim=(1, 2))) / (2.0 * eps))  # (B,)

    lap_logpsi = torch.stack(terms, dim=0).mean(dim=0).unsqueeze(1)  # (B,1), requires_grad=True
    return grad_logpsi, g2, lap_logpsi


# exact Δ logψ helper (new, internal)
def _laplacian_logpsi_exact(psi_log_fn, x: torch.Tensor):
    x = x.requires_grad_(True)
    lp = psi_log_fn(x)  # (B,)
    g = torch.autograd.grad(lp.sum(), x, create_graph=True, retain_graph=True)[0]  # (B,N,d)
    B, N, d = x.shape
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(N):
        for j in range(d):
            gij = g[:, i, j]
            sec = torch.autograd.grad(gij.sum(), x, create_graph=True, retain_graph=True)[0]
            lap += sec[:, i, j]
    g2 = (g**2).sum(dim=(1, 2), keepdim=True)
    return g, g2, lap.view(B, 1)


# ------------------ ψ-only wrapper (same name) ------------------


def psi_only(_f, _x, _C, **kw):
    logpsi, psi = psi_fn(_f, _x, _C, **kw)
    return psi.view(-1)


# ------------------ trainer (same name) ------------------


@inject_params
def train_model(
    f_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    C_occ: torch.Tensor,
    *,
    psi_fn,  # keep name
    lap_mode: Literal["fd-hutch", "hvp-hutch", "exact"] = "fd-hutch",
    objective: Literal["residual", "energy", "energy_var"] = "residual",
    lambda_var: float = 0.05,
    probes: int = 2,
    fd_eps_scale: float = 1e-3,
    std: float = 2.0,
    N_collocation: int | None = None,
    micro_batch: int = 512,
    grad_clip: float | None = 0.3,
    print_every: int = 50,
    quantile_trim: float = 0.03,
    backflow_net: nn.Module | None = None,
    spin: torch.Tensor | None = None,
    params=None,
):
    """
    Stable, simple: train in LOG-DOMAIN using E_L = -1/2(Δ logψ + ||∇ logψ||²) + V.
    Keeps your lap_mode/objective switches.
    """
    device = params["device"]
    dtype = params.get("torch_dtype", None)
    w = float(params["omega"])
    nP = int(params["n_particles"])
    d = int(params["d"])
    n_epochs = int(params["n_epochs"])
    E_DMC = params["E"]
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

    QHO_const = 0.5 * (w**2)

    def psi_log_closure(y):
        logpsi_y, _ = psi_fn(f_net, y, C_occ, backflow_net=backflow_net, spin=spin, params=params)
        return logpsi_y  # (B,)

    for epoch in range(n_epochs):
        f_net.train()
        if backflow_net is not None:
            backflow_net.train()

        # epoch cloud
        x_kwargs = dict(device=device)
        if dtype is not None:
            x_kwargs["dtype"] = dtype
        X = torch.normal(0, std, size=(N_collocation, nP, d), **x_kwargs).clamp(min=-9, max=9)

        total_rows, loss_acc = 0, 0.0
        denom = max(1, math.ceil(N_collocation / micro_batch))

        for s in range(0, N_collocation, micro_batch):
            e = min(s + micro_batch, N_collocation)
            x = X[s:e].requires_grad_(True)  # (B, N, d)

            # ∇logψ & Δlogψ
            if lap_mode == "exact":
                g, g2, lap_log = _laplacian_logpsi_exact(psi_log_closure, x)  # (B,N,d),(B,1),(B,1)
            elif lap_mode == "hvp-hutch":
                g, lap_log = grad_and_laplace_logpsi(psi_log_closure, x, probes=probes, fd_eps=1e-4)
                g2 = (g**2).sum(dim=(1, 2), keepdim=True)  # (B,1)
            elif lap_mode == "fd-hutch":
                eps = fd_eps_scale * float(std)
                g, g2, lap_log = _laplacian_logpsi_fd(psi_log_closure, x, eps=eps, probes=probes)
            else:
                raise ValueError(f"Unknown lap_mode={lap_mode!r}")

            # strict shapes (B,1)
            # ---- E_L (keep graph) ----
            # strict shapes
            lap_log = _ensure_B1("lap_log", lap_log)
            g2 = _ensure_B1("g2", g2)

            # potentials → (B,1)
            V_harm_raw = QHO_const * (x**2).sum(dim=(1, 2), keepdim=True)  # (B,1,1)
            V_harm = _collapse_to_B1("V_harm", V_harm_raw)  # (B,1)
            V_int = _collapse_to_B1("V_int", compute_coulomb_interaction(x))
            V = _ensure_B1("V", V_harm + V_int)  # (B,1)

            EL = (-0.5 * (lap_log + g2) + V).squeeze(1)  # (B,), requires_grad=True

            # per-batch trimming: build mask from detached copy, but index original tensors
            if quantile_trim > 0.0:
                m = _batch_quantile_mask(EL.detach(), lo=quantile_trim, hi=1.0 - quantile_trim)
                if m.sum().item() == 0:
                    continue
                x, lap_log, g2, V, EL = x[m], lap_log[m], g2[m], V[m], EL[m]

            # losses (log-domain)
            if objective == "residual":
                E_hat = EL.mean()
                loss = ((EL - E_hat) ** 2).mean()
            elif objective == "energy":
                loss = ((EL - E_DMC) ** 2).mean()
            elif objective == "energy_var":
                loss = ((EL - E_DMC) ** 2).mean() + lambda_var * EL.var(unbiased=False)
            else:
                raise ValueError(...)

            (loss / denom).backward()
            loss_acc += float(loss.detach())
            total_rows += EL.numel()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(f_net.parameters(), grad_clip)
            if backflow_net is not None:
                torch.nn.utils.clip_grad_norm_(backflow_net.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (epoch % print_every) == 0:
            avg_loss = loss_acc / max(1, math.ceil(N_collocation / micro_batch))
            mean_EL = EL.mean().detach()
            print(
                f"[{objective}|{lap_mode}] ep {epoch:05d}  loss≈{avg_loss:.3e}, rows={total_rows} "
                f"El = {mean_EL}"
            )
    return f_net, optimizer
