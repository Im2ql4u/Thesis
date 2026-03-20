import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroJastrow(nn.Module):
    """f(x) ≡ 0. No parameters; always returns zeros of shape (B,1)."""

    def forward(self, x: torch.Tensor, spin=None) -> torch.Tensor:
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


class PINN(nn.Module):
    """
    f(x) = rho([ mean_i φ(x_i),  pooled_{i<j} ψ(safe_features(r_ij)) , safe_extras ]) -> (B,1)
    Ψ(x) = SD(x) * exp(f(x)).
    Adds analytic cusp u(r)=γ r exp(-r/ℓ) directly to log Ψ.

    Safe-by-design:
      • Pair features: ds/dr → 0 as r→0 (no extra 1/r in ∇/Δ logΨ).
      • No raw r in 'extras' (uses ⟨r²⟩ and safe ⟨s1⟩).
      • Optional gate χ(r): χ(0)=χ'(0)=0, fades to 1 past ~0.3 a_ho.
      • Optional simple radial attention (degree-normalized).
    """

    def __init__(
        self,
        n_particles: int,
        d: int,
        omega: float,
        *,
        dL: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 2,
        act: str = "gelu",
        init: str = "xavier",
        use_pair_attn: bool = False,
        use_gate: bool = True,
        gate_radius_aho: float = 0.30,
        eps_feat_aho: float = 0.20,
    ):
        super().__init__()
        self.n_particles = int(n_particles)
        self.d = int(d)
        self.dL = int(dL)
        self.omega = float(omega)
        self.use_pair_attn = bool(use_pair_attn)
        self.use_gate = bool(use_gate)
        self.gate_radius_aho = float(gate_radius_aho)
        self.eps_feat_aho = float(eps_feat_aho)

        # indices for i<j pairs
        ii, jj = torch.triu_indices(self.n_particles, self.n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        # activations / MLPs
        self.act = self._make_act(act)
        self.phi = self._build_mlp(self.d, hidden_dim, self.dL, n_layers, self.act)
        self.psi_in_dim = 6
        self.psi = self._build_mlp(self.psi_in_dim, hidden_dim, self.dL, n_layers, self.act)
        # inputs: mean φ (dL) + pooled ψ (dL) + safe extras (2) = 2*dL + 2
        self.rho = self._build_mlp(2 * self.dL + 2, hidden_dim, 1, 2, self.act)

        # optional norms (off by default)
        self.psi_norm = nn.LayerNorm(self.dL, elementwise_affine=True)
        self.rho_norm = nn.LayerNorm(2 * self.dL + 2, elementwise_affine=True)

        self._initialize_weights(init)

        # analytic cusp params (general d): γ_ud=1/(d-1), γ_uu=1/(d+1)
        self.gamma_apara = 1.0 / (self.d - 1)
        self.gamma_para = 1.0 / (self.d + 1)
        self.cusp_len = 1.0 / (self.omega**0.5)

        # attention defaults
        self.attn_rc = 0.4
        self.attn_p = 6.0

    # ----- utils -----
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
                    nn.init.xavier_normal_(m.weight, gain=1.13)
                elif scheme == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "he":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif scheme == "lecun":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                else:
                    raise ValueError(f"Unknown init scheme {scheme}")
                nn.init.zeros_(m.bias)

    # ----- safe radial features & gate -----
    def _safe_pair_features(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        r: (B,P,1)
        Returns (features, s1_mean) where features is (B,P,6) and s1_mean is (B,1) for extras.
        All channels have ds/dr -> 0 as r->0.
        """
        a_ho = 1.0 / (self.omega**0.5)
        eps = self.eps_feat_aho * a_ho
        r2 = r * r
        rt = torch.sqrt(r2 + eps * eps)  # mollified radius

        s1 = torch.log1p((rt / eps) ** 2)  # ds/dr ~ 2r/eps^2
        s2 = r2 / (r2 + eps * eps)  # ds/dr ~ O(r)
        s3 = (rt / eps) ** 2 * torch.exp(-((rt / eps) ** 2))  # bump, ds/dr ~ O(r)

        g = torch.as_tensor([0.25, 1.0, 4.0], device=r.device, dtype=r.dtype).view(1, 1, -1)
        rbf = torch.exp(-g * s1)  # χ(0)=1, χ'(0)=0

        feat = torch.cat([s1, s2, s3, rbf], dim=-1)  # (B,P,6)
        s1_mean = s1.mean(dim=1, keepdim=False)  # (B,1) safe extra
        return feat, s1_mean

    def _short_range_gate(self, r: torch.Tensor) -> torch.Tensor:
        """χ(0)=0, χ'(0)=0, χ→1 after ~gate_radius_aho * a_ho. r: (B,P,1)"""
        if not self.use_gate:
            return torch.ones_like(r)
        a_ho = 1.0 / (self.omega**0.5)
        rg = self.gate_radius_aho * a_ho
        r2 = r * r
        return r2 / (r2 + rg * rg)

    # ----- forward -----
    def forward(self, x: torch.Tensor, spin: torch.Tensor | None = None) -> torch.Tensor:
        """
        x : (B,N,d)  -> used for NN features (phi/psi/extras)
        cusp_coords : (B,N,d) or None; if provided, cusp distances are computed
                      from these *physical* coords (default: x).
        returns: (B,1) = f_NN(x) + u_cusp(cusp_coords)
        """
        B, N, d = x.shape
        assert N == self.n_particles and d == self.d

        # --- NN features & pooling use (possibly backflowed) 'x' ---
        x_scaled = x * (self.omega**0.5)
        diff = x_scaled.unsqueeze(2) - x_scaled.unsqueeze(1)  # (B,N,N,d)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]  # (B,P,d)
        r2 = (diff_pairs * diff_pairs).sum(dim=-1, keepdim=True)  # (B,P,1)
        r = torch.sqrt(r2 + torch.finfo(x.dtype).eps)

        # φ branch
        phi_flat = x_scaled.reshape(B * N, d)
        phi_out = self.phi(phi_flat).reshape(B, N, self.dL)
        phi_mean = phi_out.mean(dim=1)  # (B,dL)

        # ψ branch (safe pair features)
        psi_in, s1_mean = self._safe_pair_features(r)  # (B,P,6), (B,1)
        psi_out = self.psi(psi_in.reshape(-1, self.psi_in_dim)).reshape(B, -1, self.dL)

        # optional short-range gate
        gate = self._short_range_gate(r)  # (B,P,1)
        psi_out = psi_out * gate
        psi_mean = psi_out.mean(dim=1)  # (B,dL)  (or your attn)

        # SAFE extras only
        r2_mean = (x_scaled**2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # (B,1)
        extras = torch.cat([r2_mean, s1_mean], dim=1)  # (B,2)

        # readout
        rho_in = torch.cat([phi_mean, psi_mean, extras], dim=1)  # (B,2*dL+2)
        out = self.rho(rho_in)  # (B,1)

        # --- Analytic cusp uses *physical* coords by default ---
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]  # (B,P,d)
        r2_c = (diff_pairs * diff_pairs).sum(dim=-1, keepdim=True)  # (B,P,1)
        r_c = torch.sqrt(r2_c + torch.finfo(x.dtype).eps)

        # spin handling (same as before)
        if spin is None:
            up = N // 2
            spin = (
                torch.cat(
                    [
                        torch.zeros(up, dtype=torch.long, device=x.device),
                        torch.ones(N - up, dtype=torch.long, device=x.device),
                    ],
                    dim=0,
                )
                .unsqueeze(0)
                .expand(B, -1)
            )  # (B,N)
        else:
            spin = spin.to(x.device).long()
            if spin.dim() == 1:
                spin = spin.unsqueeze(0).expand(B, -1)

        si = spin[:, self.idx_i]
        sj = spin[:, self.idx_j]
        same_spin = (si == sj).to(x.dtype).unsqueeze(-1)  # (B,P,1)

        gamma_para = torch.as_tensor(self.gamma_para, dtype=x.dtype, device=x.device).view(1, 1, 1)
        gamma_apara = torch.as_tensor(self.gamma_apara, dtype=x.dtype, device=x.device).view(
            1, 1, 1
        )
        gamma = same_spin * gamma_para + (1.0 - same_spin) * gamma_apara  # (B,P,1)

        pair_u = gamma * r_c * torch.exp(-r_c)  # (B,P,1)
        cusp_term = pair_u.sum(dim=1)  # (B,1)

        return out + cusp_term


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


class CTNNBackflowNet(nn.Module):
    """
    Copresheaf / graph-style backflow network.
    Drop-in replacement for BackflowNet.

    Interface:
      forward(x, spin=None) -> Δx with shape (B, N, d)

    Differences vs BackflowNet:
      - Maintains explicit node and edge features.
      - Message passing uses learned linear "transport" maps:
          node -> edge  (rho_v_to_e)
          edge -> node  (rho_e_to_v)
      - Still respects aggregation='sum'|'mean'|'max',
        spin masking, zero-mean Δx, and positive scale via bf_scale.
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
        omega: float = 1.0,
        hard_cusp_gate: bool = False,
        cusp_gate_radius_aho: float = 0.30,
        cusp_gate_power: float = 2.0,
    ):
        super().__init__()
        self.d = d
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound
        self.omega = omega
        self.hard_cusp_gate = bool(hard_cusp_gate)
        self.cusp_gate_radius_aho = float(cusp_gate_radius_aho)
        self.cusp_gate_power = float(cusp_gate_power)

        # --- activation factory (same semantics as your original) ---
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

        # ---------- Feature dimensions ----------
        # Node input: positions (+ optional spin scalar)
        node_in_dim = d + (1 if use_spin else 0)
        node_hidden = hidden  # node feature dim = hidden
        edge_hidden = msg_hidden  # edge feature dim = msg_hidden

        # ---------- Embeddings ----------
        # Node embedding: (x_i, spin_i) -> h_i
        self.node_embed = nn.Linear(node_in_dim, node_hidden)

        # Edge initial features: use relative geometry only:
        #   [r_ij (d), |r_ij|, |r_ij|^2]  -> edge_hidden
        edge_in_dim = d + 2
        self.edge_embed = self._mlp(
            edge_in_dim,
            edge_hidden,
            edge_hidden,
            msg_layers,
            self._act,
        )

        # ---------- Copresheaf-style transport maps ----------
        # Node -> Edge: maps node space to edge space
        self.rho_v_to_e = nn.Linear(node_hidden, edge_hidden, bias=False)
        # Edge -> Node: maps edge space to node space
        self.rho_e_to_v = nn.Linear(edge_hidden, node_hidden, bias=False)

        # ---------- Update MLPs ----------
        # Edge update: uses current edge feat + transported node feats
        #   [h_e, rho_v_to_e(h_i), rho_v_to_e(h_j)] with dim 3 * edge_hidden
        self.edge_update = self._mlp(
            3 * edge_hidden,
            edge_hidden,
            edge_hidden,
            msg_layers,
            self._act,
        )

        # Node update: residual update on node features
        #   [h_v, m_v] with dim 2 * node_hidden
        self.node_update = self._mlp(
            2 * node_hidden,
            node_hidden,
            node_hidden,
            layers,
            self._act,
        )

        # Final head to produce Δx from node features
        self.dx_head = nn.Linear(node_hidden, d)

        # positive learnable scale via softplus (same semantics)
        self.bf_scale_raw = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1.0)))

        # zero-init last dx layer to start Δx≈0 (identity backflow)
        if zero_init_last:
            nn.init.zeros_(self.dx_head.weight)
            nn.init.zeros_(self.dx_head.bias)

    # ---------- helpers ----------

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

    def _aggregate(self, msgs: torch.Tensor) -> torch.Tensor:
        """
        msgs: (B, N, N, H) — messages from j -> i along last dim H
        returns m_i: (B, N, H)
        """
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        if self.aggregation == "max":
            return msgs.max(dim=2).values
        raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    # ---------- forward pass ----------

    def forward(self, x: torch.Tensor, spin: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, N, d)
        spin: (N,) or (B, N) with {0,1}, optional

        returns:
          Δx: (B, N, d)
        """
        B, N, d = x.shape
        assert d == self.d
        x = x * (self.omega**0.5)
        # -------- node input features --------
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                spin_feat = spin.view(1, N, 1).to(x.dtype).expand(B, N, 1)
            else:  # (B,N)
                spin_feat = spin.view(B, N, 1).to(x.dtype)
            node_in = torch.cat([x, spin_feat], dim=-1)  # (B,N,d+1)
        else:
            node_in = x  # (B,N,d)

        # Initial node features h_v: (B,N,node_hidden)
        h_v = self.node_embed(node_in)

        # -------- edge geometry --------
        # pairwise relative vectors and norms
        r = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,d)
        r2 = (r**2).sum(dim=-1, keepdim=True)  # (B,N,N,1)
        r1 = torch.sqrt(r2 + 1e-12)  # (B,N,N,1)

        edge_in = torch.cat([r, r1, r2], dim=-1)  # (B,N,N,d+2)
        # Initial edge features h_e: (B,N,N,edge_hidden)
        h_e = self.edge_embed(edge_in)

        # -------- spin-based edge weights (mask) --------
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                s_i = spin.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            else:  # (B,N)
                s_i = spin.view(B, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(B, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            same = (s_i == s_j).to(x.dtype)
            if self.same_spin_only:
                weight = same  # only same-spin edges
            else:
                weight = torch.ones_like(same)  # all edges allowed
        else:
            weight = torch.ones_like(r[..., :1])  # (B,N,N,1)

        # remove self-messages
        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = weight * (1.0 - eye)  # (B,N,N,1)

        # -------- node -> edge transport --------
        # expand node features to edge positions
        h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
        h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])

        v_i_to_e = self.rho_v_to_e(h_v_i)  # (B,N,N,edge_hidden)
        v_j_to_e = self.rho_v_to_e(h_v_j)  # (B,N,N,edge_hidden)

        # edge update (local CTNN step on edges)
        edge_update_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)  # (B,N,N,3*edge_hidden)
        h_e_new = self.edge_update(edge_update_in)  # (B,N,N,edge_hidden)

        # -------- edge -> node transport --------
        msgs_e_to_v = self.rho_e_to_v(h_e_new)  # (B,N,N,node_hidden)
        msgs_e_to_v = msgs_e_to_v * weight  # apply spin + no-self mask

        # aggregate messages per node i over neighbors j
        m_v = self._aggregate(msgs_e_to_v)  # (B,N,node_hidden)

        # -------- node update with residual --------
        node_update_in = torch.cat([h_v, m_v], dim=-1)  # (B,N,2*node_hidden)
        delta_h = self.node_update(node_update_in)  # (B,N,node_hidden)
        h_v = h_v + delta_h  # residual update

        # -------- map to Δx --------
        dx = self.dx_head(h_v)  # (B,N,d)

        # output bound
        if self.out_bound == "tanh":
            dx = torch.tanh(dx)
        elif self.out_bound == "identity":
            pass
        else:
            raise ValueError(f"Unknown out_bound '{self.out_bound}'")

        # Enforce short-range suppression by construction:
        # gate_i -> 0 as nearest-neighbor distance -> 0, gate_i -> 1 in mid/far field.
        if self.hard_cusp_gate:
            dmat = torch.cdist(x, x, p=2.0) * (self.omega**0.5)
            eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
            dmat = dmat.masked_fill(eye, float("inf"))
            rmin = dmat.amin(dim=2).clamp_min(1e-12)
            rc = max(1e-6, self.cusp_gate_radius_aho)
            p = max(1.0, self.cusp_gate_power)
            gate = 1.0 - torch.exp(-torch.pow(rmin / rc, p))
            dx = dx * gate.unsqueeze(-1)

        # enforce zero center-of-mass shift
        dx = dx - dx.mean(dim=1, keepdim=True)

        # positive scale factor (same semantics as original)
        bf_scale = F.softplus(self.bf_scale_raw)  # > 0
        return dx * bf_scale


# ===========================================================================
# Unified CTNN:  shared graph backbone  →  backflow  Δx  +  Jastrow  f
# ===========================================================================


class UnifiedCTNN(nn.Module):
    """
    Single network that produces BOTH:
      - Δx  (B, N, d)   backflow displacement
      - f   (B, 1)      Jastrow log-factor

    from a shared copresheaf / graph-style backbone.

    The key idea: node features h_v and edge features h_e are computed once
    and then fed to TWO lightweight heads:
      • dx_head  →  per-particle backflow shift  (same as CTNNBackflowNet)
      • f_head   →  scalar Jastrow  (replaces the separate PINN)

    The Jastrow head also adds analytic cusps u(r) = γ r exp(-r/ℓ).

    This fuses all gradient paths, enabling:
      • coupled Jastrow ↔ backflow learning from epoch 0
      • a single parameter vector → L-BFGS-compatible closure
      • shared feature computation (no duplicated pair-distance work)

    Drop-in usage:
        unified = UnifiedCTNN(d=2, n_particles=6, omega=0.5)
        # In psi_fn: if isinstance(f_net, UnifiedCTNN) → use unified path
    """

    def __init__(
        self,
        d: int,
        n_particles: int,
        omega: float,
        *,
        # Backbone
        node_hidden: int = 128,
        edge_hidden: int = 128,
        msg_layers: int = 2,
        node_layers: int = 3,
        n_mp_steps: int = 1,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
        same_spin_only: bool = False,
        # Backflow head
        out_bound: str = "tanh",
        bf_scale_init: float = 0.05,
        zero_init_last: bool = True,
        # Jastrow head
        jastrow_hidden: int = 64,
        jastrow_layers: int = 2,
        # Asymptotic envelope: f_nn × exp(-ω r²_eff / envelope_width²)
        # Ensures Jastrow → 0 at large r (bound-state BC).
        # envelope_width in units of ℓ = 1/√ω.  Default 3.0 → envelope
        # starts biting around 3 oscillator lengths.
        envelope_width_aho: float = 3.0,
    ):
        super().__init__()
        self.d = d
        self.n_particles = n_particles
        self.omega = float(omega)
        self.envelope_width_aho = float(envelope_width_aho)
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound
        self.n_mp_steps = n_mp_steps

        # ---- activation ----
        def make_act(name: str) -> nn.Module:
            name = name.lower()
            acts = {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
                "silu": nn.SiLU,
                "mish": lambda: getattr(nn, "Mish", nn.SiLU)(),
                "leakyrelu": lambda: nn.LeakyReLU(0.1),
            }
            if name in ("swish",):
                name = "silu"
            if name not in acts:
                raise ValueError(f"Unknown activation '{name}'")
            return acts[name]()

        self._act_fn = act
        _act = make_act(act)

        # ---- pair index buffers ----
        ii, jj = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer("idx_i", ii, persistent=False)
        self.register_buffer("idx_j", jj, persistent=False)

        # ---- embeddings ----
        node_in_dim = d + (1 if use_spin else 0)
        edge_in_dim = d + 2  # [r_ij, |r_ij|, |r_ij|²]

        self.node_embed = nn.Linear(node_in_dim, node_hidden)
        self.edge_embed = self._mlp(
            edge_in_dim, edge_hidden, edge_hidden, msg_layers, make_act(act)
        )

        # ---- per-step transport + update MLPs ----
        self.rho_v_to_e = nn.ModuleList()
        self.edge_updates = nn.ModuleList()
        self.rho_e_to_v = nn.ModuleList()
        self.node_updates = nn.ModuleList()

        for _ in range(n_mp_steps):
            self.rho_v_to_e.append(nn.Linear(node_hidden, edge_hidden, bias=False))
            self.edge_updates.append(
                self._mlp(3 * edge_hidden, edge_hidden, edge_hidden, msg_layers, make_act(act))
            )
            self.rho_e_to_v.append(nn.Linear(edge_hidden, node_hidden, bias=False))
            self.node_updates.append(
                self._mlp(2 * node_hidden, node_hidden, node_hidden, node_layers, make_act(act))
            )

        # ====== HEAD 1: backflow  Δx  (per-particle) ======
        self.dx_head = nn.Linear(node_hidden, d)
        self.bf_scale_raw = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1.0)))
        if zero_init_last:
            nn.init.zeros_(self.dx_head.weight)
            nn.init.zeros_(self.dx_head.bias)

        # ====== HEAD 2: Jastrow  f  (scalar) ======
        # Reads from:
        #   - mean node features  (node_hidden)
        #   - mean edge features  (edge_hidden)
        #   - safe extras         (2)
        f_in_dim = node_hidden + edge_hidden + 2
        layers_f: list[nn.Module] = []
        dim = f_in_dim
        for _ in range(jastrow_layers):
            layers_f += [nn.Linear(dim, jastrow_hidden), make_act(act)]
            dim = jastrow_hidden
        layers_f.append(nn.Linear(dim, 1))
        # zero-init so f starts ≈ 0  (wavefunction starts as pure SD)
        nn.init.zeros_(layers_f[-1].weight)
        nn.init.zeros_(layers_f[-1].bias)
        self.f_head = nn.Sequential(*layers_f)

        # ---- analytic cusp params ----
        self.gamma_apara = 1.0 / (d - 1)  # opposite-spin
        self.gamma_para = 1.0 / (d + 1)  # same-spin
        self.cusp_len = 1.0 / (omega**0.5)

    # ---- helpers ----
    def _mlp(self, in_dim: int, hid: int, out_dim: int, n_layers: int, act: nn.Module):
        assert n_layers >= 1
        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(act)
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hid, hid))
                layers.append(act)
            layers.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*layers)

    def _aggregate_dense(self, msgs: torch.Tensor) -> torch.Tensor:
        """msgs: (B,N,N,H) → (B,N,H)"""
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        if self.aggregation == "max":
            return msgs.max(dim=2).values
        raise ValueError(self.aggregation)

    # ---- forward: returns (backflow_dx, jastrow_f) ----
    def forward(
        self, x: torch.Tensor, spin: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:    (B, N, d)   raw particle positions
        spin: (N,) or (B,N), optional

        Returns
        -------
        dx : (B, N, d)  backflow displacement  (zero-mean, scaled)
        f  : (B, 1)     Jastrow scalar  (includes analytic cusps)
        """
        B, N, d = x.shape
        assert d == self.d and N == self.n_particles

        x_sc = x * (self.omega**0.5)  # scale to a_ho units for features

        # ---- node input ----
        if self.use_spin and spin is not None:
            sf = (spin.view(1, N, 1) if spin.ndim == 1 else spin.view(B, N, 1)).to(x.dtype)
            if sf.shape[0] == 1:
                sf = sf.expand(B, N, 1)
            node_in = torch.cat([x_sc, sf], dim=-1)
        else:
            node_in = x_sc

        h_v = self.node_embed(node_in)  # (B, N, node_hidden)

        # ---- edge geometry (dense N×N) ----
        r_vec = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)  # (B,N,N,d)
        r2 = (r_vec**2).sum(dim=-1, keepdim=True)  # (B,N,N,1)
        r1 = torch.sqrt(r2 + 1e-12)  # (B,N,N,1)
        edge_in = torch.cat([r_vec, r1, r2], dim=-1)  # (B,N,N,d+2)
        h_e = self.edge_embed(edge_in)  # (B,N,N,edge_hidden)

        # ---- spin mask ----
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                s_i = spin.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            else:
                s_i = spin.view(B, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(B, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            same = (s_i == s_j).to(x.dtype)
            weight = same if self.same_spin_only else torch.ones_like(same)
        else:
            weight = torch.ones_like(r_vec[..., :1])

        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = weight * (1.0 - eye)

        # ---- message passing iterations ----
        for step in range(self.n_mp_steps):
            # node → edge transport
            h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
            h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])
            v_i_to_e = self.rho_v_to_e[step](h_v_i)
            v_j_to_e = self.rho_v_to_e[step](h_v_j)

            # edge update
            edge_upd_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)
            h_e = h_e + self.edge_updates[step](edge_upd_in)  # residual

            # edge → node transport
            msgs = self.rho_e_to_v[step](h_e) * weight
            m_v = self._aggregate_dense(msgs)

            # node update (residual)
            node_upd_in = torch.cat([h_v, m_v], dim=-1)
            h_v = h_v + self.node_updates[step](node_upd_in)

        # ====== HEAD 1: backflow ======
        dx = self.dx_head(h_v)  # (B, N, d)
        if self.out_bound == "tanh":
            dx = torch.tanh(dx)
        dx = dx - dx.mean(dim=1, keepdim=True)  # zero COM shift
        dx = dx * F.softplus(self.bf_scale_raw)

        # ====== HEAD 2: Jastrow ======
        # Pool backbone features → global descriptors
        h_v_mean = h_v.mean(dim=1)  # (B, node_hidden)
        # Mean of upper-triangle edge features only (no self-loops)
        h_e_pairs = h_e[:, self.idx_i, self.idx_j, :]  # (B, P, edge_hidden)
        h_e_mean = h_e_pairs.mean(dim=1)  # (B, edge_hidden)

        # Safe global extras (ω-scaled)
        r2_mean = (x_sc**2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # (B, 1)
        # Mean mollified pair distance
        eps_feat = 0.20 / (self.omega**0.5)
        r_pairs_phys = torch.sqrt(
            (x.unsqueeze(2) - x.unsqueeze(1))[:, self.idx_i, self.idx_j, :]
            .pow(2)
            .sum(-1, keepdim=True)
            + eps_feat**2
        )  # (B, P, 1)
        s1_mean = torch.log1p((r_pairs_phys / eps_feat) ** 2).mean(dim=1)  # (B, 1)

        extras = torch.cat([r2_mean, s1_mean], dim=1)  # (B, 2)
        f_in = torch.cat([h_v_mean, h_e_mean, extras], dim=1)
        f_nn = self.f_head(f_in)  # (B, 1)

        # ---- Gaussian decay envelope on f_nn ----
        # Enforces f_nn → 0 as particles move far from origin.
        # Uses mean |r_i|² (physical coords) so envelope ≈ 1 near center.
        # threshold r²: envelope_width² × ℓ² × N
        ell2 = 1.0 / self.omega  # ℓ²
        r2_phys_total = (x**2).sum(dim=(1, 2))  # (B,) total Σ_i |r_i|²
        # Normalize by N so envelope is per-particle-average
        r2_per_particle = r2_phys_total / N  # (B,)
        env_scale = self.envelope_width_aho**2 * ell2  # width² in physical units
        envelope = torch.exp(-r2_per_particle / env_scale)  # (B,)
        f_nn = f_nn * envelope.unsqueeze(-1)  # (B, 1)

        # ---- analytic cusps (on raw / physical x) ----
        diff_phys = x[:, self.idx_i, :] - x[:, self.idx_j, :]  # (B, P, d)
        r_phys = torch.sqrt((diff_phys**2).sum(-1, keepdim=True) + 1e-30)  # (B,P,1)

        if spin is not None:
            sp = spin.to(x.device).long()
            if sp.ndim == 1:
                sp = sp.unsqueeze(0).expand(B, -1)
            si = sp[:, self.idx_i]
            sj = sp[:, self.idx_j]
            same_sp = (si == sj).to(x.dtype).unsqueeze(-1)
        else:
            up = N // 2
            sp_row = torch.cat(
                [
                    torch.zeros(up, dtype=torch.long, device=x.device),
                    torch.ones(N - up, dtype=torch.long, device=x.device),
                ]
            )
            si = sp_row[self.idx_i].unsqueeze(0).expand(B, -1)
            sj = sp_row[self.idx_j].unsqueeze(0).expand(B, -1)
            same_sp = (si == sj).to(x.dtype).unsqueeze(-1)

        gamma = same_sp * self.gamma_para + (1.0 - same_sp) * self.gamma_apara
        cusp = (gamma * r_phys * torch.exp(-r_phys)).sum(dim=1)  # (B, 1)

        f = f_nn + cusp  # (B, 1)
        return dx, f


# ===========================================================================
# Orbital Backflow  (modifies orbital matrix, not coordinates)
# ===========================================================================


class OrbitalBackflowNet(nn.Module):
    """
    Orbital backflow network — perturbs the orbital *values* rather than
    the electron coordinates.

    Instead of  x_eff = x + Δx  →  Psi(x_eff),
    this computes  δΨ(x; θ) with shape (B, N, n_occ) and the caller adds
    it to the orbital matrix:  Ψ̃ = Φ·C_occ + δΨ.

    Architecture:  same copresheaf / graph-style backbone as CTNNBackflowNet
    (node embed → edge embed → transport maps → update → readout)
    but the final head outputs n_occ values per electron instead of d
    coordinate shifts.
    """

    def __init__(
        self,
        d: int,
        n_occ: int,
        *,
        msg_hidden: int = 64,
        msg_layers: int = 2,
        hidden: int = 64,
        layers: int = 2,
        act: str = "silu",
        aggregation: str = "sum",
        use_spin: bool = True,
        same_spin_only: bool = False,
        out_bound: str = "tanh",
        bf_scale_init: float = 0.05,
        zero_init_last: bool = True,
        omega: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.n_occ = n_occ
        self.use_spin = use_spin
        self.same_spin_only = same_spin_only
        self.aggregation = aggregation
        self.out_bound = out_bound
        self.omega = omega

        # --- activation factory ---
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

        # ---------- Feature dimensions ----------
        node_in_dim = d + (1 if use_spin else 0)
        node_hidden = hidden
        edge_hidden = msg_hidden

        # ---------- Embeddings ----------
        self.node_embed = nn.Linear(node_in_dim, node_hidden)
        edge_in_dim = d + 2  # [r_ij, |r_ij|, |r_ij|^2]
        self.edge_embed = self._mlp(edge_in_dim, edge_hidden, edge_hidden, msg_layers, self._act)

        # ---------- Copresheaf transport maps ----------
        self.rho_v_to_e = nn.Linear(node_hidden, edge_hidden, bias=False)
        self.rho_e_to_v = nn.Linear(edge_hidden, node_hidden, bias=False)

        # ---------- Update MLPs ----------
        self.edge_update = self._mlp(
            3 * edge_hidden, edge_hidden, edge_hidden, msg_layers, self._act
        )
        self.node_update = self._mlp(2 * node_hidden, node_hidden, node_hidden, layers, self._act)

        # ---------- Orbital head: node features → n_occ orbital corrections ----------
        self.orb_head = nn.Linear(node_hidden, n_occ)

        # positive learnable scale via softplus
        self.bf_scale_raw = nn.Parameter(torch.tensor(math.log(math.exp(bf_scale_init) - 1.0)))

        # zero-init the orbital head to start δΨ ≈ 0
        if zero_init_last:
            nn.init.zeros_(self.orb_head.weight)
            nn.init.zeros_(self.orb_head.bias)

        # External scale override (set by training loop, used if not None)
        self._scale_override: float | None = None

    # ---------- helpers ----------

    def _mlp(self, in_dim: int, hid: int, out_dim: int, num_layers: int, act: nn.Module):
        assert num_layers >= 1
        mods: list[nn.Module] = []
        if num_layers == 1:
            mods.append(nn.Linear(in_dim, out_dim))
        else:
            mods.append(nn.Linear(in_dim, hid))
            mods.append(act)
            for _ in range(num_layers - 2):
                mods.append(nn.Linear(hid, hid))
                mods.append(act)
            mods.append(nn.Linear(hid, out_dim))
        return nn.Sequential(*mods)

    @property
    def bf_scale(self):
        return F.softplus(self.bf_scale_raw)

    def _aggregate(self, msgs: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "sum":
            return msgs.sum(dim=2)
        if self.aggregation == "mean":
            return msgs.mean(dim=2)
        if self.aggregation == "max":
            return msgs.max(dim=2).values
        raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    # ---------- forward ----------

    def forward(
        self,
        x: torch.Tensor,
        spin: torch.Tensor | None = None,
        *,
        scale_override: float | None = None,
    ) -> torch.Tensor:
        """
        x:    (B, N, d)
        spin: (N,) or (B, N) with {0,1}
        scale_override: if given, use this instead of the learned bf_scale

        returns δΨ: (B, N, n_occ) — orbital perturbations
        """
        B, N, d = x.shape
        assert d == self.d
        x_sc = x * (self.omega**0.5)

        # -------- node input features --------
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                spin_feat = spin.view(1, N, 1).to(x.dtype).expand(B, N, 1)
            else:
                spin_feat = spin.view(B, N, 1).to(x.dtype)
            node_in = torch.cat([x_sc, spin_feat], dim=-1)
        else:
            node_in = x_sc

        h_v = self.node_embed(node_in)  # (B,N,hidden)

        # -------- edge geometry --------
        r = x_sc.unsqueeze(2) - x_sc.unsqueeze(1)  # (B,N,N,d)
        r2 = (r**2).sum(dim=-1, keepdim=True)  # (B,N,N,1)
        r1 = torch.sqrt(r2 + 1e-12)  # (B,N,N,1)

        edge_in = torch.cat([r, r1, r2], dim=-1)  # (B,N,N,d+2)
        h_e = self.edge_embed(edge_in)  # (B,N,N,edge_hidden)

        # -------- spin edge mask --------
        if self.use_spin and spin is not None:
            if spin.ndim == 1:
                s_i = spin.view(1, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(1, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            else:
                s_i = spin.view(B, N, 1, 1).to(x.dtype).expand(B, N, N, 1)
                s_j = spin.view(B, 1, N, 1).to(x.dtype).expand(B, N, N, 1)
            same = (s_i == s_j).to(x.dtype)
            weight = same if self.same_spin_only else torch.ones_like(same)
        else:
            weight = torch.ones_like(r[..., :1])

        eye = torch.eye(N, device=x.device, dtype=x.dtype).view(1, N, N, 1)
        weight = weight * (1.0 - eye)

        # -------- node → edge transport --------
        h_v_i = h_v.unsqueeze(2).expand(B, N, N, h_v.shape[-1])
        h_v_j = h_v.unsqueeze(1).expand(B, N, N, h_v.shape[-1])
        v_i_to_e = self.rho_v_to_e(h_v_i)
        v_j_to_e = self.rho_v_to_e(h_v_j)

        edge_update_in = torch.cat([h_e, v_i_to_e, v_j_to_e], dim=-1)
        h_e_new = self.edge_update(edge_update_in)

        # -------- edge → node transport --------
        msgs_e_to_v = self.rho_e_to_v(h_e_new)
        msgs_e_to_v = msgs_e_to_v * weight
        m_v = self._aggregate(msgs_e_to_v)

        # -------- node update with residual --------
        node_update_in = torch.cat([h_v, m_v], dim=-1)
        delta_h = self.node_update(node_update_in)
        h_v = h_v + delta_h

        # -------- map to orbital corrections δΨ --------
        dPsi = self.orb_head(h_v)  # (B, N, n_occ)

        if self.out_bound == "tanh":
            dPsi = torch.tanh(dPsi)
        elif self.out_bound != "identity":
            raise ValueError(f"Unknown out_bound '{self.out_bound}'")

        scale = (
            scale_override
            if scale_override is not None
            else (
                self._scale_override
                if self._scale_override is not None
                else F.softplus(self.bf_scale_raw)
            )
        )
        return dPsi * scale
