"""Train full ReLU baseline + neighbourhood models on the integrated 2x2 data.

This is the old 40M-parameter recipe, but with integrated-specific output names
so the earlier 706-sample checkpoints remain untouched.
"""
import csv
import os
from datetime import datetime

import numpy as np
import torch

import config_powertx as C
from data.powertx import load_flat, load_grid
from engine import fit
from models.mlp import MLP
from models.scale_invariant import ScaleInvariantMetasurface


TAG = f"{C.GRID}_integrated_full_relu_{C.device.type}"
META = f"logs/{TAG}_meta.txt"
RESULTS = f"logs/{TAG}_results.csv"
BASE_CKPT = f"ckpts/{TAG}_baseline_2000x10.pt"
NEIGH_CKPT = f"ckpts/{TAG}_neigh_K{C.KERNEL}_2000x10.pt"
BASE_HIST = f"logs/history/{TAG}_baseline_2000x10.csv"
NEIGH_HIST = f"logs/history/{TAG}_neigh_K{C.KERNEL}_2000x10.csv"


def save_history(history, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train", "val"])
        for row in zip(history["epoch"], history["train"], history["val"]):
            w.writerow(row)


@torch.no_grad()
def test_mse(model, ckpt, x, y, batch=128):
    model.load_state_dict(torch.load(ckpt, map_location=C.device))
    model = model.to(C.device).eval()
    preds = []
    tx = torch.tensor(x, dtype=torch.float32)
    for i in range(0, len(tx), batch):
        preds.append(model(tx[i:i + batch].to(C.device)).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return float(((pred - y) ** 2).mean())


def main():
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("logs/history", exist_ok=True)

    if "integrated" not in os.path.basename(C.NPZ_PATH):
        raise RuntimeError(f"This script is for the integrated dataset, got: {C.NPZ_PATH}")

    print(f"Full ReLU integrated run | {C.GRID} | {C.NPZ_PATH} | device {C.device}")
    print(f"layers: input={C.INPUT_DIM}, hidden={C.HIDDEN} x {C.N_HIDDEN}, output={C.OUTPUT_DIM}")
    print(f"samples use TEST_SPLIT={C.TEST_SPLIT}; batch={C.BATCH_SIZE}; epochs={C.EPOCHS}")

    tr_f, va_f, test_x, test_y = load_flat()
    tr_g, va_g, test_grid, test_y_g = load_grid()

    rows = []

    print("\n################ FULL BASELINE ReLU ################")
    base = MLP(C.LAYERS)
    base_params = sum(p.numel() for p in base.parameters())
    base_hist, base_val = fit(
        base, tr_f, va_f, device=C.device, ckpt_path=BASE_CKPT,
        epochs=C.EPOCHS, lr=C.LR, weight_decay=C.WEIGHT_DECAY,
        lr_decay_rate=C.LR_DECAY_RATE, lr_patience=C.LR_PATIENCE,
        eval_step=C.EVAL_STEP, stop_threshold=C.STOP_THRESHOLD)
    save_history(base_hist, BASE_HIST)
    base_test = test_mse(MLP(C.LAYERS), BASE_CKPT, test_x, test_y)
    rows.append(("baseline_relu", base_params, base_val, base_test, BASE_CKPT, BASE_HIST))
    print(f"[baseline] best val {base_val:.6f} | test {base_test:.6f}")

    print("\n################ FULL NEIGHBOURHOOD ReLU K=3 ################")
    neigh = ScaleInvariantMetasurface(
        N=C.GRID_N, K=C.KERNEL, C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM, hidden=C.HIDDEN, n_hidden=C.N_HIDDEN)
    neigh_params = sum(p.numel() for p in neigh.parameters())
    neigh_hist, neigh_val = fit(
        neigh, tr_g, va_g, device=C.device, ckpt_path=NEIGH_CKPT,
        epochs=C.EPOCHS, lr=C.LR, weight_decay=C.WEIGHT_DECAY,
        lr_decay_rate=C.LR_DECAY_RATE, lr_patience=C.LR_PATIENCE,
        eval_step=C.EVAL_STEP, stop_threshold=C.STOP_THRESHOLD)
    save_history(neigh_hist, NEIGH_HIST)
    neigh_test = test_mse(
        ScaleInvariantMetasurface(
            N=C.GRID_N, K=C.KERNEL, C=C.CHANNELS,
            n_freq=C.OUTPUT_DIM, hidden=C.HIDDEN, n_hidden=C.N_HIDDEN),
        NEIGH_CKPT, test_grid, test_y_g)
    rows.append(("neighbourhood_relu_K3", neigh_params, neigh_val, neigh_test, NEIGH_CKPT, NEIGH_HIST))
    print(f"[neighbourhood] best val {neigh_val:.6f} | test {neigh_test:.6f}")

    with open(RESULTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "best_val_mse", "test_mse", "checkpoint", "history"])
        w.writerows(rows)

    lines = [
        f"run           : {datetime.now().isoformat(timespec='seconds')}",
        f"dataset       : power-transmission {C.GRID} ({C.NPZ_PATH})",
        f"samples       : train={len(tr_f.dataset)}  val={len(va_f.dataset)}  test={len(test_y)}",
        f"architecture  : ReLU MLP, hidden={C.HIDDEN}, n_hidden={C.N_HIDDEN}, K={C.KERNEL}",
        f"optim         : epochs={C.EPOCHS} lr={C.LR} weight_decay={C.WEIGHT_DECAY} "
        f"lr_decay={C.LR_DECAY_RATE} lr_patience={C.LR_PATIENCE}",
        f"device        : {C.device}",
        f"baseline      : params={base_params:,} best_val={base_val:.6f} test={base_test:.6f}",
        f"neighbourhood : params={neigh_params:,} best_val={neigh_val:.6f} test={neigh_test:.6f}",
    ]
    with open(META, "w") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    main()
