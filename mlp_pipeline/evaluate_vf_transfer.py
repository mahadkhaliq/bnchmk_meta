"""Evaluate a trained ConceptOneVF checkpoint on the CURRENT dataset's test split.

Used for (A) re-check and (B) cross-dataset transfer (e.g. v4-trained -> v3 test).
Reports plain MSE + beta2 on the held-out test split of whatever POWERTX_GRID/
POWERTX_TARGET are set, and saves predictions for a set of random test samples.

    POWERTX_GRID=2x2v3 POWERTX_TARGET=s21 python evaluate_vf_transfer.py \
        --ckpt ckpts/vf_2x2v4_K3_np8_nr0_..._seed0.pt --n_real 0 --K 3 --tag v4to_v3_K3nr0
"""
import os
import csv
import argparse

import numpy as np
import torch

import config_powertx as C
from data.powertx import load_grid, get_freq_axis
from models.vector_fitting import ConceptOneVF
from models.rel_encoding import MODES as REL_MODES


@torch.no_grad()
def predict(model, x, dev, batch=256):
    model.eval()
    out = []
    tx = torch.tensor(x, dtype=torch.float32)
    for i in range(0, len(tx), batch):
        out.append(model(tx[i:i + batch].to(dev)).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_pole", type=int, default=8)
    ap.add_argument("--n_real", type=int, default=0)
    ap.add_argument("--K", type=int, default=None, help="override C.KERNEL for arch")
    ap.add_argument("--rel", default="offset", choices=REL_MODES)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_random", type=int, default=12)
    ap.add_argument("--summary", default="logs/transfer/summary.csv")
    ap.add_argument("--pred_dir", default="logs/transfer/preds")
    args = ap.parse_args()
    K = args.K if args.K is not None else C.KERNEL
    dev = C.device
    summary_dir = os.path.dirname(args.summary)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    _, _, test_grid, test_y = load_grid(batch_size=128)
    model = ConceptOneVF(K=K, C=C.CHANNELS, n_freq=C.OUTPUT_DIM, latent_dim=64,
                         hidden=512, n_hidden=4, n_pole=args.n_pole, n_real=args.n_real,
                         freqs=get_freq_axis(), rel_encoding=args.rel).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))

    pred = predict(model, test_grid, dev)
    per = ((pred - test_y) ** 2).mean(1)          # per-sample MSE
    weights = 1.0 + 2.0 * (1.0 - test_y) ** 2
    per_beta2 = (weights * (pred - test_y) ** 2).mean(1)
    mse = float(per.mean())
    b2 = float(per_beta2.mean())
    print(f"[{args.tag}] GRID={C.GRID} target={C.TARGET} n={len(test_y)} "
          f"| test MSE={mse:.6f} | test beta2={b2:.6f}")

    # Save summary row plus reproducible random and best/worst spectra.
    write_header = not os.path.exists(args.summary) or os.path.getsize(args.summary) == 0
    with open(args.summary, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["tag", "grid", "target", "n", "test_mse", "test_beta2", "checkpoint"]
            )
        writer.writerow([args.tag, C.GRID, C.TARGET, len(test_y),
                         f"{mse:.6f}", f"{b2:.6f}", os.path.basename(args.ckpt)])
    rng = np.random.default_rng(0)
    random_idx = rng.choice(
        len(test_y), size=min(args.n_random, len(test_y)), replace=False
    )
    best_idx = int(np.argmin(per))
    worst_idx = int(np.argmax(per))
    idx = np.unique(np.concatenate([random_idx, [best_idx, worst_idx]]))
    np.savez(os.path.join(args.pred_dir, f"{args.tag}.npz"),
             freq=get_freq_axis(), idx=idx, random_idx=random_idx,
             best_idx=best_idx, worst_idx=worst_idx,
             truth=test_y[idx], pred=pred[idx],
             per_sample_mse=per, per_sample_beta2=per_beta2)


if __name__ == "__main__":
    main()
