"""Train SiLU/Sigmoid + beta2 with the neighbourhood mechanism.

Data flow:
    grid (B, N, N, 4)
      -> wrap-around K x K windows
      -> shared MLPSiLU per cell
      -> average per-cell spectra
      -> T_clean power-transmission spectrum
"""
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch

import config_powertx as C
from data.powertx import load_grid
from losses import beta2_loss, plain_mse
from models.scale_invariant_silu import ScaleInvariantSiLU


CFG = {
    "model": "neighbourhood_SiLU_512x4_sigmoid",
    "grid": C.GRID,
    "grid_n": C.GRID_N,
    "kernel": C.KERNEL,
    "channels": C.CHANNELS,
    "d_in": C.KERNEL * C.KERNEL * C.CHANNELS,
    "hidden": 512,
    "n_layers": 4,
    "act": "silu",
    "out_act": "sigmoid",
    "d_out": C.OUTPUT_DIM,
    "loss": "beta2",
    "optimizer": "Adam",
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "scheduler": "CosineAnnealingLR",
    "batch_size": 128,
    "epochs": 500,
    "patience": 500,
    "model_select": "best plain val MSE",
}

TAG = f"silu_neigh_{C.GRID}_K{C.KERNEL}_512x4_500ep"
if "integrated" in os.path.basename(C.NPZ_PATH):
    TAG = f"silu_neigh_{C.GRID}_integrated_K{C.KERNEL}_512x4_500ep"
RUN_SUFFIX = os.environ.get("POWERTX_RUN_SUFFIX", "").strip()
if RUN_SUFFIX:
    TAG = f"{TAG}_{RUN_SUFFIX}"
CKPT = f"ckpts/{TAG}.pt"
HIST = f"logs/history/{TAG}.csv"
META = f"logs/{TAG}_meta.txt"
os.makedirs("ckpts", exist_ok=True)
os.makedirs("logs/history", exist_ok=True)


@torch.no_grad()
def eval_plain(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot += plain_mse(model(x), y).item() * len(x)
        n += len(x)
    return tot / n


@torch.no_grad()
def predict(model, x, device, batch=256):
    model.eval()
    out = []
    tx = torch.tensor(x, dtype=torch.float32)
    for i in range(0, len(tx), batch):
        out.append(model(tx[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def write_meta(n_train, n_val, n_test, best_val, best_ep, test_mse, stopped_ep):
    lines = [
        f"run           : {datetime.now().isoformat(timespec='seconds')}",
        f"variant       : {CFG['model']}  (+ K={CFG['kernel']} neighbours + beta2 loss)",
        f"dataset       : power-transmission {C.GRID}  ({C.NPZ_PATH})",
        f"grid_n={CFG['grid_n']}  channels={CFG['channels']}  kernel K={CFG['kernel']}  "
        f"per-cell d_in={CFG['d_in']}  d_out={CFG['d_out']}",
        f"hidden={CFG['hidden']}  n_layers={CFG['n_layers']}  act={CFG['act']}  "
        f"out_act={CFG['out_act']}  loss={CFG['loss']}",
        f"optim         : {CFG['optimizer']} lr={CFG['lr']} weight_decay={CFG['weight_decay']} "
        f"sched={CFG['scheduler']}(T_max={CFG['epochs']})",
        f"batch={CFG['batch_size']}  epochs={CFG['epochs']}  patience={CFG['patience']}  "
        f"select='{CFG['model_select']}'",
        f"samples       : train={n_train}  val={n_val}  test={n_test}  (test_split={C.TEST_SPLIT})",
        f"split         : deterministic random 68/17/15  seed={C.SEED}",
        f"device        : {C.device}",
        f"RESULT        : best plain val MSE={best_val:.6f} @epoch {best_ep} "
        f"(stopped @ {stopped_ep}) | test MSE={test_mse:.6f}",
        f"checkpoint    : {CKPT}",
    ]
    with open(META, "w") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        print("  " + ln)


def main():
    dev = C.device
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    print(f"SiLU/Sigmoid + beta2 neighbourhood on power-tx {C.GRID} | device {dev} | "
          f"grid {C.GRID_N}x{C.GRID_N} K={C.KERNEL} ch={C.CHANNELS} "
          f"per-cell d_in={CFG['d_in']} d_out={CFG['d_out']} 512x4")

    tr, va, test_grid, test_y = load_grid(batch_size=CFG["batch_size"])

    model = ScaleInvariantSiLU(
        N=C.GRID_N,
        K=C.KERNEL,
        C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM,
        hidden=CFG["hidden"],
        n_layers=CFG["n_layers"],
    ).to(dev)
    print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])

    best_val, best_ep, wait, stopped_ep = float("inf"), -1, 0, CFG["epochs"] - 1
    history = {"epoch": [], "train_beta2": [], "val_mse": []}

    for ep in range(CFG["epochs"]):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = beta2_loss(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        train_loss = tot / n
        val_mse = eval_plain(model, va, dev)
        sched.step()

        history["epoch"].append(ep)
        history["train_beta2"].append(train_loss)
        history["val_mse"].append(val_mse)
        if ep % 10 == 0 or ep == CFG["epochs"] - 1:
            print(f"Epoch {ep:3d} | train(beta2) {train_loss:.6f} | val(MSE) {val_mse:.6f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e}")

        if val_mse < best_val - 1e-9:
            best_val, best_ep, wait = val_mse, ep, 0
            torch.save(model.state_dict(), CKPT)
        else:
            wait += 1
            if wait >= CFG["patience"]:
                stopped_ep = ep
                print(f"Early stop at epoch {ep} (no val improvement for {CFG['patience']} epochs).")
                break

    with open(HIST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_beta2", "val_mse"])
        for row in zip(history["epoch"], history["train_beta2"], history["val_mse"]):
            w.writerow(row)

    model.load_state_dict(torch.load(CKPT, map_location=dev))
    pred = predict(model, test_grid, dev)
    test_mse = float(((pred - test_y) ** 2).mean())

    print(f"\nBest plain val MSE: {best_val:.6f} @ epoch {best_ep} | test MSE: {test_mse:.6f}")
    write_meta(len(tr.dataset), len(va.dataset), len(test_y),
               best_val, best_ep, test_mse, stopped_ep)


if __name__ == "__main__":
    main()
