"""Capacity sweep for the power-transmission dataset (current GRID in config_powertx.py).

Trains BOTH variants (baseline + neighbourhood) at several (HIDDEN, N_HIDDEN)
sizes on the SAME data split, and records best val loss + test MSE for each.
Data is loaded ONCE and reused across configs, so every model sees identical
splits and normalisation.

Edit CONFIGS to choose the sizes. Writes a results CSV and (for >1 config) a
summary plot.  Run:
    python sweep_powertx.py
"""
import os
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_powertx as C
from data.powertx import load_flat, load_grid
from models.mlp import MLP
from models.scale_invariant import ScaleInvariantMetasurface
from engine import fit

# (HIDDEN, N_HIDDEN) sizes to sweep, spanning ~0.3M to ~40M params.
# Kept FIXED across all grid sizes (1x1 / 2x2 / 3x3) so the capacity sweep is
# the same experiment for every dataset and the results are directly comparable.
# Do not trim this per-size — change it only if re-running the whole set.
CONFIGS = [
    (128, 3),
    (256, 4),
    (512, 4),
    (1024, 6),
    (2000, 10),
]

os.makedirs("ckpts", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/history", exist_ok=True)   # per-run epoch/train/val histories


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _test_mse(model, ckpt, test_x, test_y, batch=256):
    """Reload the best checkpoint and compute mean test MSE."""
    model.load_state_dict(torch.load(ckpt, map_location=C.device))
    model = model.to(C.device).eval()
    preds = []
    with torch.no_grad():
        tx = torch.tensor(test_x, dtype=torch.float32)
        for i in range(0, len(tx), batch):
            preds.append(model(tx[i:i + batch].to(C.device)).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return float(((pred - test_y) ** 2).mean())


def _fit(model, tr, va, ckpt):
    history, best_val = fit(
        model, tr, va, device=C.device, ckpt_path=ckpt,
        epochs=C.EPOCHS, lr=C.LR, weight_decay=C.WEIGHT_DECAY,
        lr_decay_rate=C.LR_DECAY_RATE, lr_patience=C.LR_PATIENCE,
        eval_step=C.EVAL_STEP, stop_threshold=C.STOP_THRESHOLD)
    return best_val, history


def save_history(history, path):
    """Persist the per-epoch (epoch, train, val) curve behind each loss plot."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train", "val"])
        for e, t, v in zip(history["epoch"], history["train"], history["val"]):
            w.writerow([e, t, v])


def write_meta(path, n_train, n_val, n_test):
    """Record exactly which experiment this sweep was, so runs are traceable."""
    from datetime import datetime
    lines = [
        f"sweep run     : {datetime.now().isoformat(timespec='seconds')}",
        f"dataset       : power-transmission {C.GRID}  ({C.NPZ_PATH})",
        f"grid_n={C.GRID_N}  channels={C.CHANNELS}  kernel K={C.KERNEL}",
        f"input_dim={C.INPUT_DIM}  output_dim={C.OUTPUT_DIM}",
        f"samples       : train={n_train}  val={n_val}  test={n_test}  (test_split={C.TEST_SPLIT})",
        f"optim         : epochs={C.EPOCHS} lr={C.LR} weight_decay={C.WEIGHT_DECAY} "
        f"batch={C.BATCH_SIZE} lr_decay={C.LR_DECAY_RATE} lr_patience={C.LR_PATIENCE}",
        f"device        : {C.device}",
        f"CONFIGS       : {CONFIGS}   (HIDDEN x N_HIDDEN)",
        f"variants      : baseline (flat MLP) + neighbourhood (ScaleInvariantMetasurface)",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", path)
    for ln in lines:
        print("  " + ln)


def main():
    print(f"Sweep on power-transmission {C.GRID} | device {C.device} | configs {CONFIGS}")

    # load data ONCE — identical split/normalisation for every config & variant
    tr_f, va_f, test_x, test_y = load_flat()
    tr_g, va_g, test_grid, test_y_g = load_grid()

    # record exactly which experiment this is
    write_meta(f"logs/sweep_{C.GRID}_meta.txt",
               n_train=len(tr_f.dataset), n_val=len(va_f.dataset), n_test=len(test_y))

    rows = []
    histories = []   # (tag, baseline_history, neigh_history) for the loss-curve plot
    for hidden, n_hidden in CONFIGS:
        tag = f"{hidden}x{n_hidden}"
        print(f"\n========== config {tag} ==========")

        # ---- baseline (no neighbouring) ----
        layers = [C.INPUT_DIM] + [hidden] * n_hidden + [C.OUTPUT_DIM]
        b_ckpt = f"ckpts/sweep_{C.GRID}_baseline_{tag}.pt"
        print(f"---- baseline {tag} ----")
        b_val, b_hist = _fit(MLP(layers), tr_f, va_f, b_ckpt)
        save_history(b_hist, f"logs/history/sweep_{C.GRID}_baseline_{tag}.csv")
        b_test = _test_mse(MLP(layers), b_ckpt, test_x, test_y)
        b_params = n_params(MLP(layers))

        # ---- neighbourhood (with neighbouring) ----
        n_ckpt = f"ckpts/sweep_{C.GRID}_neigh_{tag}.pt"
        print(f"---- neighbourhood {tag} ----")
        mk = lambda: ScaleInvariantMetasurface(
            N=C.GRID_N, K=C.KERNEL, C=C.CHANNELS,
            n_freq=C.OUTPUT_DIM, hidden=hidden, n_hidden=n_hidden)
        n_val, n_hist = _fit(mk(), tr_g, va_g, n_ckpt)
        save_history(n_hist, f"logs/history/sweep_{C.GRID}_neigh_{tag}.csv")
        n_test = _test_mse(mk(), n_ckpt, test_grid, test_y_g)
        n_par = n_params(mk())
        histories.append((tag, b_hist, n_hist))

        rows.append(dict(config=tag, hidden=hidden, n_hidden=n_hidden,
                         baseline_params=b_params, baseline_val=b_val, baseline_test=b_test,
                         neigh_params=n_par, neigh_val=n_val, neigh_test=n_test))
        print(f"[{tag}] baseline test {b_test:.5f} (val {b_val:.5f}) | "
              f"neigh test {n_test:.5f} (val {n_val:.5f})")

    # ---- summary table ----
    print("\n================ SWEEP SUMMARY ================")
    print(f"{'config':<9}{'base params':>13}{'base val':>11}{'base test':>11}"
          f"{'neigh val':>11}{'neigh test':>12}")
    for r in rows:
        print(f"{r['config']:<9}{r['baseline_params']:>13,}{r['baseline_val']:>11.5f}"
              f"{r['baseline_test']:>11.5f}{r['neigh_val']:>11.5f}{r['neigh_test']:>12.5f}")

    csv_path = f"logs/sweep_{C.GRID}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved", csv_path)

    # ---- summary plot (needs >= 2 configs to be meaningful) ----
    if len(rows) >= 2:
        x = range(len(rows))
        labels = [r["config"] for r in rows]
        plt.figure(figsize=(9, 5))
        plt.plot(x, [r["baseline_test"] for r in rows], "o-", label="baseline test")
        plt.plot(x, [r["neigh_test"] for r in rows], "s-", label="neighbourhood test")
        plt.plot(x, [r["baseline_val"] for r in rows], "o--", alpha=0.5, label="baseline val")
        plt.plot(x, [r["neigh_val"] for r in rows], "s--", alpha=0.5, label="neighbourhood val")
        plt.xticks(list(x), labels)
        plt.xlabel("config (HIDDEN x N_HIDDEN)"); plt.ylabel("MSE")
        plt.title(f"Power-transmission {C.GRID}: capacity sweep")
        plt.legend(); plt.grid(True, ls=":", alpha=0.5); plt.tight_layout()
        out = f"plots/sweep_{C.GRID}.png"
        plt.savefig(out, dpi=120); plt.close()
        print("saved", out)

    # ---- loss curves: validation MSE vs epoch, one line per config ----
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, which, title in [(axes[0], 1, "baseline"), (axes[1], 2, "neighbourhood")]:
            for tag, b_hist, n_hist in histories:
                h = b_hist if which == 1 else n_hist
                ax.plot(h["epoch"], h["val"], label=tag)
            ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title(f"{title} val loss")
            ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(title="HIDDENxN")
        axes[0].set_ylabel("val MSE (log)")
        fig.suptitle(f"Power-transmission {C.GRID}: validation loss per capacity")
        plt.tight_layout()
        out = f"plots/sweep_{C.GRID}_losscurves.png"
        plt.savefig(out, dpi=120); plt.close()
        print("saved", out)


if __name__ == "__main__":
    main()
