#!/usr/bin/env python3
"""Quick test: load one model of each type and print state_dict shapes."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from PINN import PINN, BackflowNet, CTNNBackflowNet

ROOT = "results/models 2"


def show_sd(label, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"]
    print(f"\n{label}: {path}")
    for k, v in sorted(sd.items()):
        print(f"  {k}: {tuple(v.shape)}")
    return sd


# BackflowNet
sd_bf = show_sd("BackflowNet", f"{ROOT}/2p/w_10/backflow.pt")

# PINN
sd_pinn = show_sd("PINN", f"{ROOT}/2p/w_10/f_net.pt")

# CTNNBackflowNet
sd_ctnn = show_sd("CTNNBackflowNet", f"{ROOT}/2p/w_10/backflowCTNN.pt")

# PINN paired with CTNN
sd_pinn_ctnn = show_sd("PINN (CTNN pair)", f"{ROOT}/2p/w_10/f_netCTNN.pt")

# Try constructing and loading
print("\n\n=== CONSTRUCTION TEST ===")

# PINN
d_in = sd_pinn["phi.0.weight"].shape[1]
hidden = sd_pinn["phi.0.weight"].shape[0]
phi_w = [k for k in sd_pinn if k.startswith("phi.") and k.endswith(".weight")]
last_phi = sorted(phi_w, key=lambda x: int(x.split(".")[1]))[-1]
dL = sd_pinn[last_phi].shape[0]
n_layers = len(phi_w) - 1
print(f"PINN: d={d_in}, hidden={hidden}, dL={dL}, n_layers={n_layers}")
pinn = PINN(2, d_in, 1.0, dL=dL, hidden_dim=hidden, n_layers=n_layers, act="gelu")
pinn.load_state_dict(sd_pinn, strict=True)
print("  ✓ PINN loaded OK")

# BackflowNet
msg_hidden = sd_bf["phi.0.weight"].shape[0]
phi_w = [k for k in sd_bf if k.startswith("phi.") and k.endswith(".weight")]
msg_layers = len(phi_w)
hid = sd_bf["psi.0.weight"].shape[0]
psi_w = [k for k in sd_bf if k.startswith("psi.") and k.endswith(".weight")]
layers = len(psi_w)
print(f"BF: msg_hidden={msg_hidden}, msg_layers={msg_layers}, hidden={hid}, layers={layers}")
bf = BackflowNet(
    d_in, msg_hidden=msg_hidden, msg_layers=msg_layers, hidden=hid, layers=layers, act="silu"
)
bf.load_state_dict(sd_bf, strict=True)
print("  ✓ BackflowNet loaded OK")

# CTNNBackflowNet
node_hidden = sd_ctnn["node_embed.weight"].shape[0]
node_in = sd_ctnn["node_embed.weight"].shape[1]
use_spin = node_in == d_in + 1
edge_hidden = sd_ctnn["edge_embed.0.weight"].shape[0]
ee_w = [k for k in sd_ctnn if k.startswith("edge_embed.") and k.endswith(".weight")]
msg_layers_c = len(ee_w)
nu_w = [k for k in sd_ctnn if k.startswith("node_update.") and k.endswith(".weight")]
layers_c = len(nu_w)
eu_w = [k for k in sd_ctnn if k.startswith("edge_update.") and k.endswith(".weight")]
msg_layers_eu = len(eu_w)
print(
    f"CTNN: node_hidden={node_hidden}, edge_hidden={edge_hidden}, msg_layers(embed)={msg_layers_c}, msg_layers(update)={msg_layers_eu}, node_layers={layers_c}, use_spin={use_spin}"
)
ctnn = CTNNBackflowNet(
    d_in,
    msg_hidden=edge_hidden,
    msg_layers=msg_layers_eu,
    hidden=node_hidden,
    layers=layers_c,
    act="silu",
    use_spin=use_spin,
    omega=1.0,
)
ctnn.load_state_dict(sd_ctnn, strict=True)
print("  ✓ CTNNBackflowNet loaded OK")

print("\nAll models load successfully!")
