"""Train Concept #1 from the heterogeneous-metasurface slides.

Data flow:
    grid (N, N, 4)
      -> U(cell) base latent
      -> sum_j V(cell, neighbour_j, relative offset) interaction latent
      -> D(latent) neural spectrum decoder
      -> average per-cell spectra

This is the relaxed-Lorentz concept: the hard Lorentzian physics block is
replaced by a trainable decoder, but the additive neighbour structure remains.

Pick the super-cell size by editing GRID in config_powertx.py.
Run from inside the project directory:
    python train_powertx_concept_one.py
"""
import config_powertx as C
from data.powertx import load_grid
from models.concept_one import ConceptOneMetasurface
from engine import fit


def main():
    print(
        f"Dataset: power-transmission {C.GRID}  |  Concept #1 grid {C.GRID_N}x{C.GRID_N}, "
        f"{C.CHANNELS} ch, K={C.KERNEL}, latent={C.CONCEPT_LATENT_DIM} -> output {C.OUTPUT_DIM}"
    )
    train_loader, val_loader, test_grid, test_y = load_grid()

    model = ConceptOneMetasurface(
        K=C.KERNEL,
        C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM,
        latent_dim=C.CONCEPT_LATENT_DIM,
        hidden=C.HIDDEN,
        n_hidden=C.N_HIDDEN,
    )

    fit(
        model,
        train_loader,
        val_loader,
        device=C.device,
        ckpt_path=C.CONCEPT_ONE_CKPT,
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
