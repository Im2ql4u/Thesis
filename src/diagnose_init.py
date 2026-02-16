"""Diagnose the initial energy: SD alone vs SD+cusp vs SD+cusp+envelope."""
import math, sys, torch
import numpy as np
sys.path.insert(0, "/Users/aleksandersekkelsten/thesis/src")

from PINN import UnifiedCTNN
from functions.Neural_Networks import psi_fn, _laplacian_logpsi_exact
from functions.Physics import compute_coulomb_interaction
from functions.Slater_Determinant import (
    compute_integrals, hartree_fock_closed_shell,
    slater_determinant_closed_shell,
)
import config

device, dtype = "cpu", torch.float64
N, omega = 2, 1.0
d = 2

# --- Set the global config FIRST! ---
L = max(8.0, 3.0 / math.sqrt(omega))
nx_basis = 2; ny_basis = 2
config.update(
    omega=omega, n_particles=N, d=d,
    L=L, n_grid=50, nx=nx_basis, ny=ny_basis,
    basis="cart", device=str(device), dtype="float64",
    hf_verbose=True, hf_damping=0.3,
)
params = config.get().as_dict()
params["device"] = device
params["torch_dtype"] = dtype
params["basis_n_max"] = 5

Hcore, two_dirac, basis_info = compute_integrals(params=params)
C_occ_np, orb_e, E_hf = hartree_fock_closed_shell(Hcore, two_dirac, params=params)
print(f"\nE_HF = {E_hf:.6f}, orbital energies = {orb_e}")
print(f"C_occ shape = {C_occ_np.shape}")
print(f"C_occ =\n{C_occ_np}")

# Check: the Hcore diagonal should be HO energies
print(f"\nHcore diagonal = {np.diag(Hcore)}")
# For cart basis (nx,ny): energies should be omega*(nx+ny+1)
# (0,0)->1, (1,0)->2, (0,1)->2, (1,1)->3
print(f"Expected HO energies: {[omega*(ix+iy+1) for ix in range(nx_basis) for iy in range(ny_basis)]}")

C_occ = torch.tensor(C_occ_np, dtype=dtype, device=device)
params["E"] = 3.0
params["E_HF"] = float(E_hf)

# Spin
spin = torch.cat([torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long)]).to(device)

# --- Sample points ---
ell = 1.0 / math.sqrt(omega)
n_pts = 2000
x_strat = torch.randn(n_pts, N, d, dtype=dtype, device=device) * ell

# --- Helper: compute E_L given a log-psi function ---
def compute_EL(psi_log_fn, x, label):
    x = x.detach().requires_grad_(True)
    g, g2, lap_log = _laplacian_logpsi_exact(psi_log_fn, x)
    T = -0.5 * (lap_log.view(-1) + g2.view(-1))
    V_harm = 0.5 * omega**2 * (x**2).sum(dim=(1,2))
    V_int = compute_coulomb_interaction(x).view(-1)
    E_L = T + V_harm + V_int
    good = torch.isfinite(E_L)
    E_ok = E_L[good]
    print(f"  {label}: E_L = {E_ok.mean():.4f} ± {E_ok.std():.4f}  "
          f"(T={T[good].mean():.4f}, V_h={V_harm[good].mean():.4f}, V_int={V_int[good].mean():.4f})  "
          f"n_good={good.sum()}/{x.shape[0]}")
    return E_ok

# --- Test 1: Bare SD (no NN, no cusp) ---
print("\n=== Test 1: Bare SD ===")
def psi_log_sd(y):
    sign, logabs = slater_determinant_closed_shell(
        x_config=y, C_occ=C_occ, params=params, spin=spin.unsqueeze(0).expand(y.shape[0],-1), normalize=True
    )
    return logabs.view(-1)

EL_sd = compute_EL(psi_log_sd, x_strat[:200], "stratified(200)")

# --- Test 2: UnifiedCTNN at initialization (f_nn≈0, cusp ON, envelope ON) ---
print("\n=== Test 2: UnifiedCTNN (init, cusp+envelope) ===")
net = UnifiedCTNN(
    d=2, n_particles=2, omega=1.0,
    node_hidden=64, edge_hidden=64,
    msg_layers=2, node_layers=2, n_mp_steps=1,
    jastrow_hidden=32, jastrow_layers=2,
    envelope_width_aho=3.0,
).to(device).to(dtype)

def psi_log_unified(y):
    lp, _ = psi_fn(net, y, C_occ, backflow_net=None, spin=spin, params=params)
    return lp

EL_unified = compute_EL(psi_log_unified, x_strat[:200], "stratified(200)")

# --- Test 3: MCMC from |Ψ_unified|² ---
print("\n=== Test 3: MCMC from |Ψ_unified|² ===")
with torch.no_grad():
    x_mc = torch.randn(256, N, d, dtype=dtype, device=device) * ell
    lp2 = psi_log_unified(x_mc) * 2.0
    sigma2 = 0.4 * ell
    for step in range(500):
        prop = x_mc + torch.randn_like(x_mc) * sigma2
        lp_prop = psi_log_unified(prop) * 2.0
        log_u = torch.log(torch.rand(256, dtype=dtype, device=device))
        accept = log_u < (lp_prop - lp2)
        x_mc = torch.where(accept.view(-1,1,1), prop, x_mc)
        lp2 = torch.where(accept, lp_prop, lp2)
    r_mc = x_mc.norm(dim=-1).mean()
    print(f"  MCMC <|r|> from |Ψ_unified|² = {r_mc:.3f}")

EL_unified_mc = compute_EL(psi_log_unified, x_mc[:200], "MCMC(|Ψ|²)")

print("\nDone!")

