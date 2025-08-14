import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(
        self,
        n_particles,
        d,
        hidden_dim=100,
        n_layers=2,
        omega=1.0,
        r=0.1,
        act=nn.GELU(),
        dL=5,
        init="xavier",
    ):
        """
        Constructs a network where the logarithm of the correlation factor is given by:
          sum_i phi(x_i) + sum_{i<j} psi(x_i, x_j)

        For the pairwise term, here we use (x_i - x_j)^2 as input (element-wise).

        Args:
            n_particles (int): Number of particles.
            d (int): Dimension per particle.
            hidden_dim (int): Hidden layer size for sub-networks.
            n_layers (int): Number of hidden layers for each sub-network.
            act (nn.Module): Activation function.
        """
        super(PINN, self).__init__()
        self.idx_i, self.idx_j = torch.triu_indices(
            n_particles, n_particles, offset=1
        )  # shape (2, P)
        self.n_particles = n_particles
        self.d = d
        self.dL = dL
        self.r = r
        self.omega = omega
        # Build the per-particle network φ: maps d -> 1.
        self.phi = self.build_mlp(
            in_dim=d, hidden_dim=hidden_dim, out_dim=dL, n_layers=n_layers, act=act
        )
        self.psi = self.build_mlp(
            in_dim=1, hidden_dim=hidden_dim, out_dim=dL, n_layers=n_layers, act=act
        )
        self.rho = self.build_mlp(
            in_dim=dL * 2, hidden_dim=hidden_dim, out_dim=1, n_layers=n_layers, act=act
        )
        # Apply Kaiming (He) initialization to every Linear layer.
        self._initialize_weights(init)

    def build_mlp(self, in_dim, hidden_dim, out_dim, n_layers, act):
        """
        Build a multi-layer perceptron (MLP).
        Args:
            in_dim (int): Input dimension.
            hidden_dim (int): Hidden layer size.
            out_dim (int): Output dimension.
            n_layers (int): Number of hidden layers (excluding the first and final Linear layers).
            act (nn.Module): Activation function.

        Returns:
            nn.Sequential: The constructed MLP.
        """
        layers = []
        # First layer: in_dim -> hidden_dim.
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act)
        # Additional hidden layers.
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        # Final output layer: hidden_dim -> out_dim.
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self, scheme):
        for m in self.modules():
            if scheme == "custom":
                gain = 1.13  # ≈1/√0.1580 for GELU
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight, gain=gain)
                        nn.init.zeros_(m.bias)
            elif scheme == "xavier":
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        nn.init.zeros_(m.bias)
            elif scheme == "he":
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode="fan_in")
                        nn.init.zeros_(m.bias)
            # (you can remove 'lecun' if not used)
            elif scheme == "lecun":
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_in", nonlinearity="linear"
                        )
                        nn.init.zeros_(m.bias)
            else:
                raise ValueError(f"Unknown init scheme {scheme}")

            # Always zero out biases
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Compute the logarithm of the correlation factor.

        Args:
            x (torch.Tensor): Input of shape (batch, n_particles, d).

        Returns:
            torch.Tensor: Output of shape (batch, 1) representing log(Ψ_corr),
                          so that the full wavefunction is given by
                          Ψ(x) = SD(x) * exp(f(x)).
        """
        batch, n, d = x.shape

        # 1) φ-branch: apply φ to each particle, then average across particles.
        #    φ expects input shape (batch * n_particles, d) -> outputs (batch*n, dL)
        phi_flat = x.view(-1, d)  # (batch*n, d)
        phi_out = self.phi(phi_flat)  # (batch*n, dL)
        phi_out = phi_out.view(batch, n, -1)  # (batch, n, dL)
        phi_mean = phi_out.mean(dim=1)  # (batch, dL)

        # 2) Pairwise ψ-branch: compute all pairwise distances in one call.

        diff = x.unsqueeze(2) - x.unsqueeze(1)
        # Select only i<j pairs: (batch, num_pairs, d)
        diff_pairs = diff[:, self.idx_i, self.idx_j, :]  # (batch, P, d)
        r_ij = diff_pairs.norm(dim=-1, keepdim=True)
        psi_in = r_ij.view(-1, 1)  # (batch*P, 1)
        psi_out = self.psi(psi_in)  # (batch*P, k)
        psi_out = psi_out.view(batch, -1, self.dL)  # (batch, P, k)
        psi_sum = psi_out.sum(dim=1)  # (batch, k)

        # --- Combine and finalize ---
        rho_in = torch.cat([phi_mean, psi_sum], dim=1)  # (batch, dL + k)
        out = self.rho(rho_in)  # (batch, 1)
        return out
