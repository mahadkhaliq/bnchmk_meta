"""Train the MLP WITH neighbouring (scale-invariant) on a power-transmission dataset.

Data flow:  grid (N, N, 4)
              -> wrap-around padding
              -> per-cell K x K window (K*K*4)
              -> SHARED MLP
              -> average over cells
              -> spectrum T_clean (OUTPUT_DIM)

Pick the super-cell size by editing GRID in config_powertx.py.
(Note: at 1x1 a cell has no neighbours, so use the baseline there instead.)
Run from inside the project directory:
    python train_powertx_neighborhood.py
"""
import config_powertx as C
from data.powertx import load_grid
from models.scale_invariant import ScaleInvariantMetasurface
from engine import fit


def main():
    print(f"Dataset: power-transmission {C.GRID}  |  grid {C.GRID_N}x{C.GRID_N}, "
          f"{C.CHANNELS} ch, K={C.KERNEL} -> output {C.OUTPUT_DIM}")
    train_loader, val_loader, test_grid, test_y = load_grid()

    model = ScaleInvariantMetasurface(
        N=C.GRID_N,
        K=C.KERNEL,
        C=C.CHANNELS,
        n_freq=C.OUTPUT_DIM,
        hidden=C.HIDDEN,
        n_hidden=C.N_HIDDEN,
    )

    fit(
        model, train_loader, val_loader,
        device=C.device,
        ckpt_path=C.NEIGHBORHOOD_CKPT,
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
