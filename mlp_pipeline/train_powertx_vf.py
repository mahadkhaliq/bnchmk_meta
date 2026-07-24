"""Train ConceptOneVF (Concept #1 trunk + vector-fitting head) with beta2 loss.

Config via env: POWERTX_GRID, POWERTX_KERNEL (K), POWERTX_SEED.
CLI: --n_real (real poles), --n_pole, --rel, --loss {beta2,mse}.

Fixed recipe: SiLU trunk 512x4, latent 64, Adam 1e-3/1e-5, Cosine(500),
500 epochs, batch 128, select on best plain val MSE. Reports BOTH plain MSE
and beta2 on the held-out test set. Saves ckpt + per-epoch history + meta.

    POWERTX_GRID=2x2v4 POWERTX_KERNEL=3 python train_powertx_vf.py --n_real 0
"""
import os
import csv
import argparse
import random
from datetime import datetime

import numpy as np
import torch

import config_powertx as C
from data.powertx import load_grid, get_freq_axis
from models.vector_fitting import ConceptOneVF
from losses import beta2_loss, plain_mse

HIDDEN, N_HIDDEN, LATENT = 512, 4, 64
EPOCHS = int(os.environ.get("POWERTX_EPOCHS", "500"))
LR, WD, BATCH = 1e-3, 1e-5, 128
GRAD_CLIP = 1.0


@torch.no_grad()
def eval_metrics(model, x_grid, y, dev, batch=256):
    """Return (plain_mse, beta2) on the full array."""
    model.eval()
    preds = []
    tx = torch.tensor(x_grid, dtype=torch.float32)
    for i in range(0, len(tx), batch):
        preds.append(model(tx[i:i + batch].to(dev)).cpu())
    pred = torch.cat(preds, 0)
    yt = torch.tensor(y, dtype=torch.float32)
    return plain_mse(pred, yt).item(), beta2_loss(pred, yt).item()


@torch.no_grad()
def eval_val_mse(model, loader, dev):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        tot += plain_mse(model(x), y).item() * len(x)
        n += len(x)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_real", type=int, default=0)
    ap.add_argument("--n_pole", type=int, default=8)
    ap.add_argument("--rel", default="offset",
                    choices=["offset", "offset_dist", "embed", "none"])
    ap.add_argument("--loss", default="beta2", choices=["beta2", "mse"])
    args = ap.parse_args()
    loss_fn = beta2_loss if args.loss == "beta2" else plain_mse

    dev = C.device
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)

    TAG = (f"vf_{C.GRID}_K{C.KERNEL}_np{args.n_pole}_nr{args.n_real}"
           f"_{args.rel}_{args.loss}_512x4_500ep_seed{C.SEED}")
    CKPT = f"ckpts/{TAG}.pt"
    HIST = f"logs/history/{TAG}.csv"
    META = f"logs/{TAG}_meta.txt"
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("logs/history", exist_ok=True)

    print(f"ConceptOneVF | {C.GRID} K={C.KERNEL} | n_pole={args.n_pole} "
          f"n_real={args.n_real} rel={args.rel} loss={args.loss} | dev {dev}")

    tr, va, test_grid, test_y = load_grid(batch_size=BATCH)
    freqs = get_freq_axis()
    if freqs is None:
        print("WARNING: no freq_GHz; using placeholder axis.")

    model = ConceptOneVF(
        K=C.KERNEL, C=C.CHANNELS, n_freq=C.OUTPUT_DIM, latent_dim=LATENT,
        hidden=HIDDEN, n_hidden=N_HIDDEN, n_pole=args.n_pole, n_real=args.n_real,
        freqs=freqs, rel_encoding=args.rel).to(dev)
    print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val, best_ep = float("inf"), -1
    history = {"epoch": [], "train_loss": [], "val_mse": []}

    for ep in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # VF poles are stiff
            opt.step()
            tot += loss.item() * len(x); n += len(x)
        train_loss = tot / n
        val_mse = eval_val_mse(model, va, dev)
        sched.step()

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"Epoch {ep:3d} | train({args.loss}) {train_loss:.6f} "
                  f"| val(MSE) {val_mse:.6f} | lr {opt.param_groups[0]['lr']:.2e}")

        if val_mse < best_val - 1e-9:
            best_val, best_ep = val_mse, ep
            torch.save(model.state_dict(), CKPT)

    with open(HIST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_mse"])
        for row in zip(history["epoch"], history["train_loss"], history["val_mse"]):
            w.writerow(row)

    model.load_state_dict(torch.load(CKPT, map_location=dev))
    test_mse, test_beta2 = eval_metrics(model, test_grid, test_y, dev)
    print(f"\nBest val MSE {best_val:.6f} @ep {best_ep} | "
          f"TEST mse {test_mse:.6f} | TEST beta2 {test_beta2:.6f}")

    lines = [
        f"run        : {datetime.now().isoformat(timespec='seconds')}",
        f"model      : ConceptOneVF (Concept#1 trunk + VF head, SiLU no-BN)",
        f"dataset    : {C.GRID}  ({C.NPZ_PATH})",
        f"K={C.KERNEL} n_pole={args.n_pole} n_real={args.n_real} rel={args.rel} "
        f"loss={args.loss}",
        f"trunk      : hidden={HIDDEN} n_hidden={N_HIDDEN} latent={LATENT}",
        f"optim      : Adam lr={LR} wd={WD} Cosine(T_max={EPOCHS}) batch={BATCH} epochs={EPOCHS}",
        f"samples    : train={len(tr.dataset)} val={len(va.dataset)} test={len(test_y)} seed={C.SEED}",
        f"device     : {dev}",
        f"RESULT     : best val MSE={best_val:.6f} @ep {best_ep} | "
        f"test MSE={test_mse:.6f} | test beta2={test_beta2:.6f}",
        f"checkpoint : {CKPT}",
    ]
    with open(META, "w") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__":
    main()
