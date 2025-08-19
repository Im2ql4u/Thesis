import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
