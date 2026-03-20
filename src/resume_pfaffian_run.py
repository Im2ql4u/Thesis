import argparse
import math
import time
from pathlib import Path

import torch

import run_pfaffian as rp
from PINN import CTNNBackflowNet
from jastrow_architectures import CTNNJastrowVCycle


def main():
    ap = argparse.ArgumentParser(description="Resume Pfaffian(+BF) collocation training from checkpoint")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--n-coll", type=int, default=768)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha-end", type=float, default=0.85)
    ap.add_argument("--vmc-every", type=int, default=50)
    ap.add_argument("--vmc-n", type=int, default=8000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--tag", type=str, default="pfaffian_bf_resume")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location=rp.DEVICE)
    C_occ, params = rp.setup()

    f_net = CTNNJastrowVCycle(
        n_particles=rp.N_ELEC,
        d=rp.DIM,
        omega=rp.OMEGA,
        node_hidden=24,
        edge_hidden=24,
        bottleneck_hidden=12,
        n_down=1,
        n_up=1,
        msg_layers=1,
        node_layers=1,
        readout_hidden=64,
        readout_layers=2,
        act="silu",
    ).to(rp.DEVICE).to(rp.DTYPE)
    f_net.load_state_dict(ckpt["jas_state"])
    for p in f_net.parameters():
        p.requires_grad = False

    pfc = ckpt.get("pf_config", {"n_basis": 9, "n_occ": rp.N_ELEC // 2, "nx": 3, "ny": 3})
    pfaffian_net = rp.PfaffianNet(
        pfc["n_basis"],
        pfc["n_occ"],
        C_occ,
        pfc.get("nx", 3),
        pfc.get("ny", 3),
        use_mlp=False,
    ).to(rp.DEVICE).to(rp.DTYPE)
    pfaffian_net.load_state_dict(ckpt["pf_state"])

    bf_net = None
    if "bf_state" in ckpt and "bf_config" in ckpt:
        bfc = ckpt["bf_config"]
        bf_net = CTNNBackflowNet(
            d=bfc["d"],
            msg_hidden=bfc["msg_hidden"],
            msg_layers=bfc["msg_layers"],
            hidden=bfc["hidden"],
            layers=bfc["layers"],
            act=bfc["act"],
            aggregation=bfc["aggregation"],
            use_spin=bfc["use_spin"],
            same_spin_only=bfc["same_spin_only"],
            out_bound=bfc["out_bound"],
            bf_scale_init=bfc["bf_scale_init"],
            zero_init_last=bfc["zero_init_last"],
            omega=bfc["omega"],
        ).to(rp.DEVICE).to(rp.DTYPE)
        bf_net.load_state_dict(ckpt["bf_state"])
        for p in bf_net.parameters():
            p.requires_grad = False

    print(f"Resuming from: {a.ckpt}")
    print(f"Device: {rp.DEVICE}")
    print(f"Backflow loaded: {bf_net is not None}")

    t0 = time.time()
    pfaffian_net, hist = rp.train_pfaffian(
        pfaffian_net,
        f_net,
        params,
        bf_net=bf_net,
        n_epochs=a.epochs,
        lr=a.lr,
        alpha_end=a.alpha_end,
        n_coll=a.n_coll,
        oversample=8,
        micro_batch=32,
        print_every=10,
        replay_frac=0.25,
        patience=a.patience,
        vmc_every=a.vmc_every,
        vmc_n=a.vmc_n,
        tag=a.tag,
    )

    E = float("nan")
    se = float("nan")
    err = float("nan")
    if len(hist) > 0 and isinstance(hist[-1], dict) and math.isfinite(hist[-1].get("E", float("nan"))):
        E = float(hist[-1]["E"])
        err = (E - rp.E_DMC) / abs(rp.E_DMC) * 100

    save_path = rp.RESULTS_DIR / f"{a.tag}.pt"
    torch.save(
        {
            "tag": a.tag,
            "pf_state": pfaffian_net.state_dict(),
            "jas_state": f_net.state_dict(),
            "bf_state": bf_net.state_dict() if bf_net is not None else None,
            "bf_config": ckpt.get("bf_config", None),
            "pf_class": "PfaffianNet",
            "jas_class": "CTNNJastrowVCycle",
            "pf_config": pfc,
            "E": E,
            "se": se,
            "err": err,
            "hist": hist,
            "wall": time.time() - t0,
        },
        save_path,
    )
    print(f"Saved resumed checkpoint -> {save_path}")


if __name__ == "__main__":
    main()
