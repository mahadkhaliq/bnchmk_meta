"""Train the BASELINE MLP — WITHOUT the neighbouring mechanism.

Data flow:  flat geometry (INPUT_DIM)  ->  MLP  ->  spectrum (OUTPUT_DIM)

Run from inside the project directory:
    python train_mlp_baseline.py
"""
import config
from data.loaders import load_flat
from models.mlp import MLP
from engine import fit


def main():
    train_loader, val_loader, test_x, test_y = load_flat(config.BATCH_SIZE)

    model = MLP(config.LAYERS)

    fit(
        model, train_loader, val_loader,
        device=config.device,
        ckpt_path=config.BASELINE_CKPT,
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
