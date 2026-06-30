"""Train the MLP WITH the neighbouring mechanism (scale-invariant model).

Data flow:  grid (N, N, C)
              -> wrap-around padding
              -> per-cell K x K window (K*K*C)
              -> SHARED MLP
              -> average over cells
              -> spectrum (OUTPUT_DIM)

Run from inside the project directory:
    python train_mlp_neighborhood.py
"""
import config
from data.loaders import load_grid
from models.scale_invariant import ScaleInvariantMetasurface
from engine import fit


def main():
    train_loader, val_loader, test_grid, test_y = load_grid(config.BATCH_SIZE)

    model = ScaleInvariantMetasurface(
        N=config.GRID_N,
        K=config.KERNEL,
        C=config.CHANNELS,
        n_freq=config.OUTPUT_DIM,
        hidden=config.HIDDEN,
        n_hidden=config.N_HIDDEN,
    )

    fit(
        model, train_loader, val_loader,
        device=config.device,
        ckpt_path=config.NEIGHBORHOOD_CKPT,
        epochs=config.EPOCHS,
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        lr_decay_rate=config.LR_DECAY_RATE,
        lr_patience=config.LR_PATIENCE,
        eval_step=config.EVAL_STEP,
        stop_threshold=config.STOP_THRESHOLD,
    )


if __name__ == "__main__":
    main()
