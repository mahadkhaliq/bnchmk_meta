"""Staged training of the vector-fitting variant (ConceptOneVF).

Same protocol as the other two models:
  Stage 1 (GRID="1x1"):  python train_powertx_vf.py --stage pretrain [--rel MODE]
  Stage 2 (GRID="2x2"):  python train_powertx_vf.py --stage joint    [--rel MODE]

--rel selects the relative-position encoding for the interaction network:
  offset (default) | offset_dist | embed | none
Checkpoints are rel-aware so ablation runs do not overwrite each other.
"""
import argparse

import torch

import config_powertx as C
from data.powertx import load_grid, get_freq_axis
from engine import fit
from models.vector_fitting import ConceptOneVF

N_POLE = 8   # conjugate pole pairs per cell (VF notebook default)
N_REAL = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "joint"], required=True)
    ap.add_argument("--rel", choices=["offset", "offset_dist", "embed", "none"],
                    default="offset", help="relative-position encoding ablation")
    args = ap.parse_args()
    pre_ckpt = f"best_powertx_vf_pretrain_{args.rel}.pt"
    joint_ckpt = f"best_powertx_vf_joint_{C.GRID}_{args.rel}.pt"

    train_loader, val_loader, test_grid, test_y = load_grid()

    freqs = get_freq_axis()
    if freqs is None:
        print("WARNING: no freq_GHz in this npz; using the placeholder axis.")
    model = ConceptOneVF(
        K=C.KERNEL,
        C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM,
        latent_dim=C.CONCEPT_LATENT_DIM,
        hidden=C.HIDDEN,
        n_hidden=C.N_HIDDEN,
        n_pole=N_POLE,
        n_real=N_REAL,
        freqs=freqs,
        rel_encoding=args.rel,
    )

    if args.stage == "pretrain":
        assert C.GRID == "1x1", "Stage 1 must run on the 1x1 (unitary) dataset."
        ckpt = pre_ckpt
    else:
        state = torch.load(pre_ckpt, map_location="cpu")
        model.load_state_dict(state)
        ckpt = joint_ckpt
        print(f"Warm-started from {pre_ckpt}; joint training on {C.GRID}.")

    fit(
        model,
        train_loader,
        val_loader,
        device=C.device,
        ckpt_path=ckpt,
        epochs=C.EPOCHS,
        lr=C.LR,
        weight_decay=C.WEIGHT_DECAY,
        lr_decay_rate=C.LR_DECAY_RATE,
        lr_patience=C.LR_PATIENCE,
        eval_step=C.EVAL_STEP,
        stop_threshold=C.STOP_THRESHOLD,
    )


if __name__ == "__main__":
    main()
