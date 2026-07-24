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
from losses import beta2_loss, plain_mse


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
    ap.add_argument("--rel", default="offset")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_random", type=int, default=12)
    args = ap.parse_args()
    K = args.K if args.K is not None else C.KERNEL
    dev = C.device
    os.makedirs("logs/transfer", exist_ok=True)
    os.makedirs("logs/transfer/preds", exist_ok=True)

    _, _, test_grid, test_y = load_grid(batch_size=128)
    model = ConceptOneVF(K=K, C=C.CHANNELS, n_freq=C.OUTPUT_DIM, latent_dim=64,
                         hidden=512, n_hidden=4, n_pole=args.n_pole, n_real=args.n_real,
                         freqs=get_freq_axis(), rel_encoding=args.rel).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))

    pred = predict(model, test_grid, dev)
    per = ((pred - test_y) ** 2).mean(1)          # per-sample MSE
    mse = float(per.mean())
    b2 = float(beta2_loss(torch.tensor(pred), torch.tensor(test_y)))
    print(f"[{args.tag}] GRID={C.GRID} target={C.TARGET} n={len(test_y)} "
          f"| test MSE={mse:.6f} | test beta2={b2:.6f}")

    # save summary row + random-sample predictions (for plotting)
    with open("logs/transfer/summary.csv", "a", newline="") as f:
        csv.writer(f).writerow([args.tag, C.GRID, C.TARGET, len(test_y),
                                f"{mse:.6f}", f"{b2:.6f}", os.path.basename(args.ckpt)])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test_y), size=min(args.n_random, len(test_y)), replace=False)
    np.savez(f"logs/transfer/preds/{args.tag}.npz",
             freq=get_freq_axis(), idx=idx, truth=test_y[idx], pred=pred[idx],
             per_sample_mse=per)


if __name__ == "__main__":
    main()
