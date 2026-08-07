"""Staged training of the full Lorentz model (slides' training protocol).

Stage 1 (slides: "Pre-Training f_theta1 and f_theta_r"):
    set GRID = "1x1" in config_powertx.py, then
        python train_powertx_lorentz.py --stage pretrain
    Trains unitary net + physics scalars on unitary (periodic) cells; the
    interaction net is untouched (N=1 path skips it).

Stage 2 (slides: "Training f_theta2", joint fine-tune of everything):
    set GRID = "2x2" (then "3x3"), then
        python train_powertx_lorentz.py --stage joint
    Warm-starts from the pretrain checkpoint and trains all parameters.
"""
import argparse

import torch

import config_powertx as C
from data.powertx import load_grid, get_freq_axis
from engine import fit
from models.lorentz import LorentzMetasurface

PRETRAIN_CKPT = "best_powertx_lorentz_pretrain_{rel}.pt"
JOINT_CKPT = f"best_powertx_lorentz_joint_{C.GRID}_{{rel}}.pt"
N_OSC = 2  # oscillators per cell per (electric, magnetic) branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "joint"], required=True)
    ap.add_argument("--rel", choices=["offset", "offset_dist", "embed", "none"],
                    default="offset", help="relative-position encoding ablation")
    args = ap.parse_args()
    pre_ckpt = PRETRAIN_CKPT.format(rel=args.rel)
    joint_ckpt = JOINT_CKPT.format(rel=args.rel)

    train_loader, val_loader, test_grid, test_y = load_grid()

    freqs = get_freq_axis()   # v3 npz: real freq_GHz axis; v1: None -> fallback
    if freqs is None:
        print("WARNING: no freq_GHz in this npz; using the placeholder axis.")
    model = LorentzMetasurface(
        rel_encoding=args.rel,
        K=C.KERNEL,
        C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM,
        n_osc=N_OSC,
        hidden=C.HIDDEN,
        n_hidden=C.N_HIDDEN,
        freqs=freqs,
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
