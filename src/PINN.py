import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroJastrow(nn.Module):
    """f(x) ≡ 0. No parameters; always returns zeros of shape (B,1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)


class DetachWrapper(nn.Module):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod
        for p in self.mod.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.mod(*args, **kwargs)


def _softplus_mlp(in_dim: int, hidden: int, out_dim: int):
    mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.Softplus(), nn.Linear(hidden, out_dim))
    # small, well-conditioned init
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
    return mlp


class PINN(nn.Module):
    """
    f(x) = rho( [ mean_i φ(x_i),  mean_{i<j} ψ(||x_i - x_j||) ] ),  output shape: (B,1)
    Use exp(f) as your correlation factor: Ψ(x) = SD(x) * exp(f(x)).
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        dL: int = 5,  # feature width per branch
        hidden_dim: int = 128,
        n_layers: int = 2,  # hidden layers per φ/ψ MLP
        act: str = "gelu",
        init: str = "xavier",  # "xavier"|"he"|"custom"|"lecun"
    ):
        super().__init__()
        self.n_particles = n_particles
        self.d = d
        self.dL = dL
        self.omega = omega

        # --- activations ---
        self.act = self._make_act(act)

        # --- i<j index buffers (move with device) ---
        idx_i, idx_j = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", idx_i, persistent=False)
        self.register_buffer("idx_j", idx_j, persistent=False)

        # --- φ: per-particle MLP, maps ℝ^d → ℝ^{dL} ---
        self.phi = self._build_mlp(
            in_dim=d, hidden_dim=hidden_dim, out_dim=dL, n_layers=n_layers, act=self.act
        )

        # --- ψ: pairwise MLP, maps ℝ → ℝ^{dL} (input is r_ij = ||x_i - x_j||) ---
        self.psi = self._build_mlp(
            in_dim=3, hidden_dim=hidden_dim, out_dim=dL, n_layers=n_layers, act=self.act
        )
        self.psi_norm = nn.LayerNorm(dL, elementwise_affine=True)  # optional, see §3

        # --- ρ: tiny readout
        self.rho = nn.Linear(2 * dL + 3, 1)
        self.rho_norm = nn.LayerNorm(2 * dL + 3, elementwise_affine=True)

        # init
        self._initialize_weights(init)
        # Optional: start ρ as small non-zero so grads flow even if you freeze it
        with torch.no_grad():
            self.rho.weight.fill_(1.0 / (2 * dL + 3))
            self.rho.bias.zero_()

    # ---------- helpers ----------
    def _make_act(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name in ("silu", "swish"):
            return nn.SiLU()
        if name == "mish":
            return getattr(nn, "Mish", nn.SiLU)()
        if name == "leakyrelu":
            return nn.LeakyReLU(0.1)
        raise ValueError(f"Unknown act '{name}'")

    def _build_mlp(self, in_dim, hidden_dim, out_dim, n_layers, act):
        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, out_dim)]
        return nn.Sequential(*layers)

    def _initialize_weights(self, scheme: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "custom":
                    gain = 1.13  # good for GELU-ish
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif scheme == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "he":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif scheme == "lecun":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                else:
                    raise ValueError(f"Unknown init scheme {scheme}")
                nn.init.zeros_(m.bias)

    # ---------- trainability controls ----------
    def set_trainable(self, *, phi: bool = True, psi: bool = True, rho: bool = True):
        for p in self.phi.parameters():
            p.requires_grad = phi
        for p in self.psi.parameters():
            p.requires_grad = psi
        for p in self.rho.parameters():
            p.requires_grad = rho

    def freeze_rho_as_avg(self):
        """
        Freeze ρ as a fixed averaging readout so φ/ψ still backpropagate.
        """
        with torch.no_grad():
            self.rho.weight.fill_(1.0 / (2 * self.dL + 3))
            self.rho.bias.zero_()
        for p in self.rho.parameters():
            p.requires_grad = False

    def unfreeze_rho(self, *, reinit_zero: bool = False):
        if reinit_zero:
            nn.init.zeros_(self.rho.weight)
            nn.init.zeros_(self.rho.bias)
        for p in self.rho.parameters():
            p.requires_grad = True

    def param_groups(self, *, lr_phi=1e-3, lr_psi=1e-3, lr_rho=2e-4, wd=0.0):
        groups = []
        if any(p.requires_grad for p in self.phi.parameters()):
            groups.append({"params": list(self.phi.parameters()), "lr": lr_phi, "weight_decay": wd})
        if any(p.requires_grad for p in self.psi.parameters()):
            groups.append({"params": list(self.psi.parameters()), "lr": lr_psi, "weight_decay": wd})
        if any(p.requires_grad for p in self.rho.parameters()):
            groups.append({"params": list(self.rho.parameters()), "lr": lr_rho, "weight_decay": wd})
        return groups

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, spin: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, d)  ->  out: (B, 1) = log-correlation f(x)
        """
        B, N, d = x.shape
        assert N == self.n_particles and d == self.d

        # scaled coords
        x = x * (self.omega**0.5)

        # ---------------- particle branch (φ) ----------------
        phi_flat = x.reshape(B * N, d)
        phi_out = self.phi(phi_flat).reshape(B, N, self.dL)  # (B,N,dL)
        phi_mean = phi_out.mean(dim=1)  # (B,dL)

        # ---------------- pairwise branch (ψ) ----------------
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]  # (B,P,d)
        r2 = (diff_pairs * diff_pairs).sum(dim=-1, keepdim=True)  # (B,P,1)

        # soft-core distance for bounded ∂ and ∂²
        delta = getattr(self, "delta_soft", 3e-2)  # scaled units
        delta = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
        r_soft = torch.sqrt(r2 + delta * delta)  # (B,P,1)

        inv_r = 1.0 / (1.0 + r_soft)  # bounded, smooth
        cos_r = torch.cos(r_soft)  # low-k radial
        psi_in = torch.cat([r_soft, inv_r, cos_r], dim=-1)  # (B,P,3)

        psi_out = self.psi(psi_in.reshape(-1, 3)).reshape(B, -1, self.dL)
        psi_mean = psi_out.mean(dim=1)  # (B,dL)

        # ---------------- simple global extras ----------------
        r2_mean = (x**2).mean(dim=(1, 2), keepdim=True).reshape(B, 1)  # (B,1)
        mean_r = r_soft.mean(dim=(1, 2), keepdim=True).reshape(B, 1)  # (B,1)
        cos1 = torch.cos(r_soft).mean(dim=(1, 2), keepdim=True).mul(0.5).reshape(B, 1)  # (B,1)

        extras = torch.cat([r2_mean, mean_r, cos1], dim=1)  # (B,3)

        # ---------------- readout ----------------
        rho_in = torch.cat([phi_mean, psi_mean, extras], dim=1)
        out = self.rho(rho_in)  # (B,1)

        # ---------------- exact cusp term ----------------
        # if not provided, assume closed shell
        if spin is None:
            up = N // 2
            spin = torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=x.device),
                    torch.ones(N - up, dtype=torch.long, device=x.device),
                ]
            )
        # spin pair mask (P,)
        same_spin = (spin[self.idx_i] == spin[self.idx_j]).to(x.dtype)

        # γ coefficients (register as buffers; defaults shown)
        gamma_para = torch.as_tensor(
            getattr(self, "gamma_para", 0.25), dtype=x.dtype, device=x.device
        )
        gamma_apara = torch.as_tensor(
            getattr(self, "gamma_apara", 0.50), dtype=x.dtype, device=x.device
        )

        gamma = same_spin * gamma_para + (1.0 - same_spin) * gamma_apara  # (P,)
        gamma = gamma.view(1, -1, 1)  # (1,P,1) for broadcast

        # Add ∑_{i<j} γ_ij * r_soft as a fixed analytic contribution to log Ψ
        cusp = (gamma * r_soft).sum(dim=1)  # (B,1)
        out = out + cusp

        return out


class BackflowNet(nn.Module):
    """
    Minimal, configurable backflow network.
    Returns Δx with shape (B, N, d).

    Configurable:
      - act: "gelu"|"relu"|"tanh"|"silu"|"mish"|"leakyrelu"|"identity"
      - msg_layers / msg_hidden  (message MLP φ)
      - layers / hidden          (node/update MLP ψ)
      - aggregation: "sum"|"mean"|"max"
      - use_spin & same_spin_only
      - out_bound: "tanh"|"identity"
      - zero_init_last (start near identity)
    """

    def __init__(
        self,
        d: int,
        *,
        msg_hidden: int = 128,
        msg_layers: int = 2,
        hidden: int = 128,
        layers: int = 3,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
        same_spin_only: bool = False,
        out_bound: str = "tanh",
        bf_scale_init: float = 0.05,
        zero_init_last: bool = True,
    ):
        super().__init__()
        self.d = d
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound

        # --- tiny activation factory (no helpers) ---
        def make_act(name: str) -> nn.Module:
            name = name.lower()
            if name == "relu":
                return nn.ReLU()
            if name == "gelu":
                return nn.GELU()
            if name == "tanh":
                return nn.Tanh()
            if name in ("silu", "swish"):
                return nn.SiLU()
            if name == "mish":
                return getattr(nn, "Mish", nn.SiLU)()
            if name == "leakyrelu":
                return nn.LeakyReLU(0.1)
            if name in ("identity", "none"):
                return nn.Identity()
            raise ValueError(f"Unknown activation '{name}'")

        self._act = make_act(act)

        # φ: message MLP → maps (3d+2) → msg_hidden
        msg_in = 3 * d + 2
        self.phi = self._mlp(msg_in, msg_hidden, msg_hidden, msg_layers, self._act)

        # ψ: node/update MLP → maps (d + msg_hidden) → d
        upd_in = d + msg_hidden
        self.psi = self._mlp(upd_in, hidden, d, layers, self._act)

        # positive learnable scale via softplus
        self.bf_scale_raw = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1.0)))
        # zero-init last ψ layer to start Δx≈0 (identity backflow)
        if zero_init_last:
            last = self.psi[-1]  # Linear
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _mlp(self, in_dim: int, hid: int, out_dim: int, num_layers: int, act: nn.Module):
        assert num_layers >= 1
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(act)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hid, hid))
                layers.append(act)
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    @property
    def bf_scale(self):
        return F.softplus(self.bf_scale_raw)

    def _aggregate(self, m_ij: torch.Tensor) -> torch.Tensor:
        # m_ij: (B, N, N, H)
        if self.aggregation == "sum":
            return m_ij.sum(dim=2)
        if self.aggregation == "mean":
            return m_ij.mean(dim=2)
        if self.aggregation == "max":
            return m_ij.max(dim=2).values
        raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    def forward(self, x: torch.Tensor, spin: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, d), spin optional: (N,) or (B, N) with {0,1}
        returns Δx: (B, N, d)
        """
        B, N, d = x.shape
        assert d == self.d

        # Pairwise deltas and norms
        r = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        r2 = (r**2).sum(dim=-1, keepdim=True)  # (B,N,N,1)
        r1 = torch.sqrt(r2 + 1e-12)  # (B,N,N,1)

        xi = x.unsqueeze(2).expand(B, N, N, d)
        xj = x.unsqueeze(1).expand(B, N, N, d)
        msg_in = torch.cat([xi, xj, r, r1, r2], dim=-1)  # (B,N,N,3d+2)

        # Spin weights (optional)
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                s_i = spin.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            else:  # (B,N)
                s_i = spin.view(B, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(B, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            same = (s_i == s_j).to(x.dtype)
            weight = same if self.same_spin_only else torch.ones_like(same)
        else:
            weight = torch.ones_like(r[..., :1])

        # Remove self-messages
        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = weight * (1.0 - eye)

        # Message pass + aggregate
        m_ij = self.phi(msg_in) * weight  # (B,N,N,H)
        m_i = self._aggregate(m_ij)  # (B,N,H)

        # Node/update → Δx_i
        upd = torch.cat([x, m_i], dim=-1)  # (B,N,d+H)
        dx = self.psi(upd)  # (B,N,d)

        # Output bound + positive scale
        if self.out_bound == "tanh":
            dx = torch.tanh(dx)
        elif self.out_bound == "identity":
            pass
        else:
            raise ValueError(f"Unknown out_bound '{self.out_bound}'")

        bf_scale = F.softplus(self.bf_scale_raw)  # > 0
        return dx * bf_scale


# Usage:
# --- Training Backflow ---
# opt = torch.optim.AdamW([
#     {"params": [p for n,p in bf.named_parameters() if n != "bf_scale_raw"], "lr": 1e-3},
#     {"params": [bf.bf_scale_raw], "lr": 1e-3, "weight_decay": 1e-5},
# ])

# _, bf = train_model(
#     f_net_zero,
#     opt,
#     C_occ,
#     mapper,
#     backflow_net=bf,
#     std=std,
#     print_e=10,
# )
# bf_frozen = DetachWrapper(bf).to(cfg.torch_device, cfg.torch_dtype)

# --- Training phi only ---
# f_net.freeze_rho_as_avg()
# f_net.set_trainable(phi=True, psi=False, rho=False)
# opt = torch.optim.AdamW(f_net.param_groups(lr_phi=1e-3, lr_psi=0, lr_rho=0, wd=1e-4))

# --- Training psi only ---
# f_net.freeze_rho_as_avg()
# f_net.set_trainable(phi=False, psi=True, rho=False)
# opt = torch.optim.AdamW(f_net.param_groups(lr_phi=0, lr_psi=1e-3, lr_rho=0, wd=1e-4))

# --- Training entire network ---
# f_net.unfreeze_rho(reinit_zero=False)   # keep averaging weights as a warm start
# f_net.set_trainable(phi=True, psi=True, rho=True)
# opt = torch.optim.AdamW(f_net.param_groups(lr_phi=1e-4, lr_psi=1e-3, lr_rho=1e-4, wd=0))

# f_net, _ = train_model(
#     f_net,
#     opt,
#     C_occ,
#     mapper,
#     backflow_net=None,
#     std=std,
#     print_e=10,
# )
