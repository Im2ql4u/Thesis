"""Quick test: MCMC variance-minimization with high-quality samples."""
import sys
import torch

sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

from PINN import UnifiedCTNN
from train_residual import setup_system, train_residual, evaluate_vmc


def main():
    device, dtype = "cpu", torch.float64
    C_occ, params = setup_system(2, 1.0, device=device, dtype=dtype)

    net = UnifiedCTNN(
        d=2, n_particles=2, omega=1.0,
        node_hidden=64, edge_hidden=64,
        msg_layers=2, node_layers=2, n_mp_steps=1,
        jastrow_hidden=32, jastrow_layers=2,
        envelope_width_aho=3.0,
    ).to(device).to(dtype)
    print(f"Params: {sum(p.numel() for p in net.parameters()):,}")

    # Large batch, lots of decorrelation, short training
    net, hist = train_residual(
        net, C_occ, params,
        n_epochs=150, lr=3e-4,
        n_walkers=1024, rw_steps=20, burn_in=500,
        micro_batch=64, print_every=10,
        huber_delta=1.0, grad_clip=0.5,
        quantile_trim=0.05,
        warmup_epochs=15,
        rechain_every=0,   # no rechain — trust the large batch
        patience=0,         # no early stopping
    )

    result = evaluate_vmc(net, C_occ, params, n_samples=30_000, label="2e final")

    E = result["E_mean"]
    print(f"\nFinal: E={E:.5f} (target=3.00000, err={abs(E-3.0)/3.0*100:.2f}%)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
