"""Train the BASELINE MLP (no neighbouring) on a power-transmission .npz dataset.

Data flow:  flat params (INPUT_DIM)  ->  MLP  ->  spectrum T_clean (OUTPUT_DIM)

Pick the super-cell size by editing GRID in config_powertx.py.
Run from inside the project directory:
    python train_powertx_baseline.py
"""
import config_powertx as C
from data.powertx import load_flat
from models.mlp import MLP
from engine import fit


def main():
    print(f"Dataset: power-transmission {C.GRID}  |  input {C.INPUT_DIM} -> output {C.OUTPUT_DIM}")
    train_loader, val_loader, test_x, test_y = load_flat()

    model = MLP(C.LAYERS)

    fit(
        model, train_loader, val_loader,
        device=C.device,
        ckpt_path=C.BASELINE_CKPT,
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
