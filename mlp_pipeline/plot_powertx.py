"""Plot results for the power-transmission run (current GRID in config_powertx.py).

Produces three PNGs in plots/:
    loss_curves_<GRID>.png   train/val loss vs epoch (parsed from the training log)
    spectra_<GRID>.png       a few test spectra: ground truth vs both models
    mse_hist_<GRID>.png      per-sample test-MSE distribution for both models

Run AFTER training (needs the checkpoints and the log):
    python plot_powertx.py
"""
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # no display: save to file
import matplotlib.pyplot as plt

import config_powertx as C
from data.powertx import load_flat, load_grid
from models.mlp import MLP
from models.scale_invariant import ScaleInvariantMetasurface

PLOT_DIR = "plots"
LOG_PATH = f"logs/powertx_{C.GRID}_full.log"
os.makedirs(PLOT_DIR, exist_ok=True)


def _predict(model, x, device, batch=256):
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        tx = torch.tensor(x, dtype=torch.float32)
        for i in range(0, len(tx), batch):
            out.append(model(tx[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def parse_log(path):
    """Split the combined log by MODEL markers and pull (epoch, train, val) per model."""
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    sections = re.split(r"#+ MODEL \d+: ([^#]+?) #+", text)
    # sections = [pre, name1, body1, name2, body2, ...]
    runs = {}
    pat = re.compile(r"Epoch\s+(\d+) \| train (\S+) \| val (\S+)")
    for i in range(1, len(sections), 2):
        name = sections[i].strip()
        body = sections[i + 1]
        e, tr, va = [], [], []
        for m in pat.finditer(body):
            e.append(int(m.group(1))); tr.append(float(m.group(2))); va.append(float(m.group(3)))
        if e:
            runs[name] = (np.array(e), np.array(tr), np.array(va))
    return runs


def plot_loss_curves():
    runs = parse_log(LOG_PATH)
    if not runs:
        print(f"[skip] no parseable log at {LOG_PATH}")
        return
    plt.figure(figsize=(10, 6))
    for name, (e, tr, va) in runs.items():
        short = "baseline" if "BASELINE" in name.upper() else "neighbourhood"
        line, = plt.plot(e, tr, label=f"{short} train")
        plt.plot(e, va, "--", color=line.get_color(), label=f"{short} val")
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss (log)")
    plt.title(f"Power-transmission {C.GRID}: training/validation loss")
    plt.legend(); plt.grid(True, which="both", ls=":", alpha=0.5); plt.tight_layout()
    out = f"{PLOT_DIR}/loss_curves_{C.GRID}.png"
    plt.savefig(out, dpi=120); plt.close()
    print("saved", out)


def load_predictions():
    """Returns (freq, test_y, preds dict). Both loaders share the same test split."""
    freq = np.load(C.NPZ_PATH, allow_pickle=True)["freq_GHz"]
    preds = {}

    _, _, test_x, test_y = load_flat()
    if os.path.exists(C.BASELINE_CKPT):
        m = MLP(C.LAYERS)
        m.load_state_dict(torch.load(C.BASELINE_CKPT, map_location=C.device))
        preds["baseline"] = _predict(m, test_x, C.device)

    _, _, test_grid, test_y2 = load_grid()
    if os.path.exists(C.NEIGHBORHOOD_CKPT):
        m = ScaleInvariantMetasurface(N=C.GRID_N, K=C.KERNEL, C=C.CHANNELS,
                                      n_freq=C.OUTPUT_DIM, hidden=C.HIDDEN, n_hidden=C.N_HIDDEN)
        m.load_state_dict(torch.load(C.NEIGHBORHOOD_CKPT, map_location=C.device))
        preds["neighbourhood"] = _predict(m, test_grid, C.device)

    return freq, test_y, preds


def plot_spectra(freq, test_y, preds, n=4):
    if not preds:
        print("[skip] no checkpoints to plot spectra"); return
    # spread the chosen samples by baseline error (best, median-ish, worst) if available
    ref = next(iter(preds.values()))
    err = ((ref - test_y) ** 2).mean(axis=1)
    order = np.argsort(err)
    idxs = [order[0], order[len(order)//3], order[2*len(order)//3], order[-1]][:n]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8)); axes = axes.flatten()
    for ax, idx in zip(axes, idxs):
        ax.plot(freq, test_y[idx], "k", lw=2, alpha=0.8, label="ground truth")
        for name, p in preds.items():
            ax.plot(freq, p[idx], lw=1.3, alpha=0.85, label=name)
        ax.set_title(f"test sample {idx}  (MSE={err[idx]:.4f})")
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("T (|S21|^2)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Power-transmission {C.GRID}: predicted vs ground-truth spectra")
    plt.tight_layout()
    out = f"{PLOT_DIR}/spectra_{C.GRID}.png"
    plt.savefig(out, dpi=120); plt.close()
    print("saved", out)


def plot_mse_hist(test_y, preds):
    if not preds:
        print("[skip] no checkpoints to plot MSE"); return
    plt.figure(figsize=(9, 5))
    for name, p in preds.items():
        ps = ((p - test_y) ** 2).mean(axis=1)
        plt.hist(ps, bins=40, alpha=0.55, label=f"{name}  (mean {ps.mean():.4f})")
    plt.xlabel("per-sample test MSE"); plt.ylabel("count")
    plt.yscale("log")
    plt.title(f"Power-transmission {C.GRID}: per-sample test MSE")
    plt.legend(); plt.tight_layout()
    out = f"{PLOT_DIR}/mse_hist_{C.GRID}.png"
    plt.savefig(out, dpi=120); plt.close()
    print("saved", out)


if __name__ == "__main__":
    plot_loss_curves()
    freq, test_y, preds = load_predictions()
    for name, p in preds.items():
        print(f"{name:<14} test MSE: {((p - test_y) ** 2).mean():.6f}")
    plot_spectra(freq, test_y, preds)
    plot_mse_hist(test_y, preds)
    print("done.")
