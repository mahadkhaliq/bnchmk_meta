"""Staged training of Concept #1 (relaxed Lorentz), per the slides' protocol.

Stage 1 ("Pre-Training f_theta1 and f_theta_r"):
    f_theta_r(f_theta1(x)) — decoder(unitary(x)) — is trained to predict 1x1
    spectra (equivalently a unitary 2x2 with four identical cells; the v3
    loader collapses that to 1x1 automatically). The interaction net is
    untouched because the N == 1 forward path skips it.
        set GRID = "1x1" in config_powertx.py, then
        python train_powertx_concept_one_staged.py --stage pretrain

Stage 2 ("Training f_theta2", joint fine-tune):
    Warm-starts every network from Stage 1 and trains all parameters on
    heterogeneous supercells.
        set GRID = "2x2", then
        python train_powertx_concept_one_staged.py --stage joint
"""
import argparse

import torch

import config_powertx as C
from data.powertx import load_grid
from engine import fit
from models.concept_one import ConceptOneMetasurface

PRETRAIN_CKPT = "best_powertx_concept_one_pretrain_{rel}.pt"
JOINT_CKPT = f"best_powertx_concept_one_joint_{C.GRID}_{{rel}}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "joint"], required=True)
    ap.add_argument("--rel", choices=["offset", "offset_dist", "embed", "none"],
                    default="offset", help="relative-position encoding ablation")
    args = ap.parse_args()
    pre_ckpt = PRETRAIN_CKPT.format(rel=args.rel)
    joint_ckpt = JOINT_CKPT.format(rel=args.rel)

    train_loader, val_loader, test_grid, test_y = load_grid()

    model = ConceptOneMetasurface(
        rel_encoding=args.rel,
        K=C.KERNEL,
        C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM,
        latent_dim=C.CONCEPT_LATENT_DIM,
        hidden=C.HIDDEN,
        n_hidden=C.N_HIDDEN,
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
