import torch
import torch.nn as nn


class BackflowNet(nn.Module):
    """
    Permutation–equivariant backflow:
      x_tilde_i = x_i + scale * tanh( MLP( [x_i, sum_j φ(x_i, x_j, ||x_i-x_j||)] ) )

    Shapes:
      x : (B, N, d)
      spin (optional) : (N,) integers {0,1} for up/down; if provided, messages can be split.
    """

    def __init__(
        self,
        d: int,
        hidden: int = 128,
        msg_hidden: int = 128,
        use_spin: bool = True,
        same_spin_only: bool = False,
        bf_scale_init: float = 0.05,
    ):
        super().__init__()
        self.d = d
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only

        # Message MLP φ: takes [x_i, x_j, r_ij, ||r_ij||, ||r_ij||^2]
        msg_in = 2 * d + d + 2  # x_i, x_j, r_ij, |r_ij|, |r_ij|^2
        self.phi = nn.Sequential(
            nn.Linear(msg_in, msg_hidden),
            nn.SiLU(),
            nn.Linear(msg_hidden, msg_hidden),
            nn.SiLU(),
        )

        # Node update ψ: takes [x_i, m_i] -> Δx_i
        upd_in = d + msg_hidden
        self.psi = nn.Sequential(
            nn.Linear(upd_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d),
        )

        # Scale parameter (starts small so training is stable)
        self.bf_scale = nn.Parameter(torch.tensor(bf_scale_init))

        # Zero-init last layer ⇒ starts as identity transform
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.zeros_(self.psi[-1].bias)

    def forward(
        self, x: torch.Tensor, spin: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B,N,d). Returns Δx of shape (B,N,d).
        """
        B, N, d = x.shape
        assert d == self.d

        # Pairwise differences r_ij and norms
        # r: (B,N,N,d), rij = x_i - x_j
        r = x.unsqueeze(2) - x.unsqueeze(1)
        r2 = (r**2).sum(dim=-1, keepdim=True)  # (B,N,N,1)
        r1 = torch.sqrt(r2 + 1e-12)  # (B,N,N,1)

        # Build message inputs [x_i, x_j, r_ij, |r_ij|, |r_ij|^2]
        xi = x.unsqueeze(2).expand(B, N, N, d)  # (B,N,N,d)
        xj = x.unsqueeze(1).expand(B, N, N, d)  # (B,N,N,d)
        msg_in = torch.cat([xi, xj, r, r1, r2], dim=-1)  # (B,N,N,2d+d+2) = (B,N,N,3d+2)

        # Optional spin mask
        if self.use_spin and spin is not None:
            # spin: (N,) with {0,1}. Broadcast to (B,N,N,1)
            s_i = spin.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
            s_j = spin.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            if self.same_spin_only:
                mask = (s_i == s_j).to(x.dtype)
            else:
                # use both, but you could weight same vs different spin differently if you’d like
                mask = torch.ones_like(s_i)
        else:
            mask = torch.ones_like(r[..., :1])

        # Remove self-messages
        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        mask = mask * (1.0 - eye)

        # Messages φ and aggregate (sum = permutation invariant)
        m_ij = self.phi(msg_in)  # (B,N,N,msg_hidden)
        m_ij = m_ij * mask  # (B,N,N,msg_hidden)
        m_i = m_ij.sum(dim=2)  # (B,N,msg_hidden)

        # Node update
        upd_in = torch.cat([x, m_i], dim=-1)  # (B,N,d+msg_hidden)
        dx = self.psi(upd_in)  # (B,N,d)

        # Softly bound early; scale is learnable
        return torch.tanh(dx) * self.bf_scale.clamp(min=0.0)
