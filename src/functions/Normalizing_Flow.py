import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import MLP  # you already used this
from tqdm import tqdm

from utils import inject_params

from .Slater_Determinant import slater_determinant_closed_shell


# ------------------------------------------------------------
# Utilities: device/dtype
# ------------------------------------------------------------
def _dev_dtype_like(t: torch.Tensor):
    return t.device, t.dtype


def _ensure_shape(x, n_particles, d):
    """
    Ensure (B, N, D). Accepts (B, N*D) or (B, N, D).
    """
    if x.ndim == 2 and x.shape[1] == n_particles * d:
        return x.view(x.shape[0], n_particles, d)
    if x.ndim == 3 and x.shape[1:] == (n_particles, d):
        return x
    raise ValueError(f"Unexpected shape {x.shape}, expected (B,N,D) or (B,N*D)")


# ------------------------------------------------------------
# Metropolis sampler (N,D) with parallel chains
# ------------------------------------------------------------
@torch.no_grad()
def parallel_metropolis_sampler(
    C_occ,
    n_particles: int,
    d: int,
    *,
    n_chains: int = 2048,
    n_burn: int = 1000,
    n_steps: int = 200,
    step_size: float = 0.5,
    L: float = 3.0,
    device=None,
    dtype=torch.float64,
):
    """
    Returns a batch (B, N, D) of samples ~ |ψ|^2 using parallel Metropolis-Hastings.
    - ψ is defined by the Slater determinant implied by C_occ via your function
      slater_determinant_from_C_occ_batch((B,N,D), C_occ).
    """
    if device is None:
        device = C_occ.device if hasattr(C_occ, "device") else "cpu"

    # init uniform in a box
    x = torch.empty(n_chains, n_particles, d, device=device, dtype=dtype).uniform_(-L, L)

    def _psi_abs_sq(z):  # (B,N,D) -> (B,)
        # Your function must accept (B,N,D) or (B,N,1) for 1D; we give it (B,N,D)
        val = slater_determinant_closed_shell(z, C_occ).abs().squeeze()
        return (val + 1e-12) ** 2  # stability

    def _mh_step(x_cur):
        prop = x_cur + step_size * torch.randn_like(x_cur)  # random-walk proposal
        psi_old = _psi_abs_sq(x_cur)  # (B,)
        psi_new = _psi_abs_sq(prop)  # (B,)
        ratio = torch.clamp(psi_new / psi_old, max=1e6)

        u = torch.rand_like(ratio)
        accept = (u < ratio).view(-1, 1, 1)  # (B,1,1) broadcast to (B,N,D)
        return torch.where(accept, prop, x_cur)

    # burn-in
    for _ in range(n_burn):
        x = _mh_step(x)
    # sample thinning (optional): we just do n_steps single-chain moves
    for _ in tqdm(range(n_steps)):
        x = _mh_step(x)
    return x  # (n_chains, N, D)


# ------------------------------------------------------------
# Mapper / Velocity field vθ(x,t)
# Input: concat([x_flat, t]), Output: ẋ_flat
# ------------------------------------------------------------
def build_mapper(n_particles: int, d: int, hidden=(256, 256, 256), act=nn.SiLU):
    in_channels = n_particles * d + 1
    hidden_channels = list(hidden) + [n_particles * d]
    # torchvision.ops.MLP uses same activation per layer
    return MLP(in_channels=in_channels, hidden_channels=hidden_channels, activation_layer=act)


# ------------------------------------------------------------
# Conditional Flow Matching training loop
# ------------------------------------------------------------
def train_flow_with_cfm(
    C_occ,
    *,
    n_particles: int,
    d: int,
    mapper: nn.Module,
    base_dist="normal",
    n_epochs=20,
    batch_size=16384,
    target_pool=2621,  # how many target points to pre-generate per refresh
    target_refresh_every=1,  # epochs between refills; can raise for speed
    lr=1e-3,
    grad_clip=1.0,
    use_amp=True,
    device=None,
    dtype=torch.float64,
):
    """
    Trains vθ by CFM: x_t = (1-t) x_0 + t y, with v*(x_t,t) = y - x_0.
    - x_0 ~ base (standard normal), y ~ |ψ|^2 (from Metropolis sampler).
    - mapper takes concat([x_t_flat, t]) and predicts vθ(x_t,t) in flat space.
    """
    if device is None:
        device = C_occ.device if hasattr(C_occ, "device") else "cpu"

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (device != "cpu"))

    mapper.to(device=device, dtype=dtype)
    optimizer = optim.Adam(mapper.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    def _sample_base(B):
        if base_dist == "normal":
            return torch.randn(B, n_particles, d, device=device, dtype=dtype)
        raise ValueError("Only standard normal base supported here.")

    # warm target pool
    with torch.no_grad():
        y_pool = parallel_metropolis_sampler(
            C_occ,
            n_particles,
            d,
            n_chains=target_pool,
            n_burn=800,
            n_steps=200,
            step_size=0.5,
            L=3.0,
            device=device,
            dtype=dtype,
        )  # (target_pool, N, D)

    steps_per_epoch = math.ceil(target_pool / batch_size)

    for epoch in tqdm(range(1, n_epochs + 1)):
        if (epoch == 1) or (epoch % target_refresh_every == 0):
            with torch.no_grad():
                y_pool = parallel_metropolis_sampler(
                    C_occ,
                    n_particles,
                    d,
                    n_chains=target_pool,
                    n_burn=400,
                    n_steps=120,
                    step_size=0.5,
                    L=3.0,
                    device=device,
                    dtype=dtype,
                )

        perm = torch.randperm(y_pool.shape[0], device=device)
        y_pool = y_pool[perm]

        running = 0.0
        for k in tqdm(range(steps_per_epoch), desc=f"CFM Epoch {epoch}/{n_epochs}"):
            start = k * batch_size
            end = min((k + 1) * batch_size, y_pool.shape[0])
            if start >= end:
                break

            B = end - start
            y = y_pool[start:end]  # (B,N,D)
            x0 = _sample_base(B)  # (B,N,D)
            t = torch.rand(B, 1, 1, device=device, dtype=dtype)  # (B,1,1)
            x_t = (1.0 - t) * x0 + t * y  # (B,N,D)
            v_star = y - x0  # (B,N,D)

            # flatten for mapper
            x_t_flat = x_t.reshape(B, n_particles * d).to(dtype=dtype)
            t_flat = t.reshape(B, 1).to(dtype=dtype)
            model_in = torch.cat([x_t_flat, t_flat], dim=-1)  # (B, N*D + 1)

            optimizer.zero_grad(set_to_none=True)

            if use_amp and device != "cpu":
                with torch.cuda.amp.autocast():
                    v_pred = mapper(model_in)  # (B, N*D)
                    loss = criterion(v_pred, v_star.reshape(B, -1))
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(mapper.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                v_pred = mapper(model_in)
                loss = criterion(v_pred, v_star.reshape(B, -1))
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(mapper.parameters(), grad_clip)
                optimizer.step()

            running += loss.item()

        print(f"[Epoch {epoch:3d}] loss = {running / max(1, steps_per_epoch):.6f}")

    return mapper


# ------------------------------------------------------------
# Sampler: integrate ẋ = vθ(x,t) from t=0 -> 1
# (simple Euler; you can switch to Heun/ RK2 for stability)
# ------------------------------------------------------------
@torch.no_grad()
@inject_params
def sample_with_flow(mapper: nn.Module, *, n_steps=10, params=None):
    device = params["torch_device"]
    dtype = params["torch_dtype"]
    n_particles = params["n_particles"]
    d = params["d"]
    size = params["N_collocation"]
    if device is None:
        device = next(mapper.parameters()).device
    # start from base
    x = torch.randn(size, n_particles, d, device=device, dtype=dtype)  # x(0)
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    dt = ts[1] - ts[0]

    for t in ts[:-1]:
        t_flat = t.expand(size, 1)
        x_flat = x.reshape(size, n_particles * d).to(
            dtype=torch.float32 if next(mapper.parameters()).dtype == torch.float32 else dtype
        )
        model_in = torch.cat([x_flat, t_flat], dim=-1)
        v = mapper(model_in)  # (B, N*D)
        v = v.view(size, n_particles, d).to(dtype=dtype)
        x = x + dt * v

    return x  # (size, N, D)


# ------------------------------------------------------------
# Minimal “main” usage (adapt to your script)
# ------------------------------------------------------------
@inject_params
def train_flow_driver(C_occ, *, params=None):
    device = params["torch_device"]
    dtype = params["torch_dtype"]
    n_particles = params["n_particles"]
    d = params["d"]
    mapper = build_mapper(n_particles, d, hidden=(200, 200, 200), act=nn.SiLU)
    mapper = mapper.to(device=device, dtype=dtype)
    mapper = train_flow_with_cfm(
        C_occ,
        n_particles=n_particles,
        d=d,
        mapper=mapper,
        n_epochs=20,
        batch_size=1638,
        target_pool=262144,
        target_refresh_every=1,
        lr=1e-3,
        grad_clip=1.0,
        use_amp=(device != "cpu"),
        device=device,
        dtype=dtype,
    )
    return mapper
