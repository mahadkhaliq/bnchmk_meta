"""Best model (HIDDEN=128, N_HIDDEN=3 neighbourhood) at K=3 vs K=5.

'One increment in the number of neighbours' = bump the window from 3x3 to 5x5.
K is a MODEL-side knob only, so both variants train on the SAME grid split
(load_grid is independent of K) -> the comparison is apples-to-apples.

NOTE on 2x2: with wrap-around padding the 3x3 window already covers all 4
distinct cells, so K=5 adds no NEW neighbours, only periodic repeats (input
36 -> 100 dims). This script measures whether that extra redundant context
helps or hurts.

    python kernel_check.py
"""
import numpy as np
import torch

import config_powertx as C
from data.powertx import load_grid
from models.scale_invariant import ScaleInvariantMetasurface
from engine import fit

HIDDEN, N_HIDDEN = 128, 3          # the best config from the sweep
KERNELS = [3, 5]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_mse(model, ckpt, test_grid, test_y, batch=256):
    model.load_state_dict(torch.load(ckpt, map_location=C.device))
    model = model.to(C.device).eval()
    preds = []
    with torch.no_grad():
        tx = torch.tensor(test_grid, dtype=torch.float32)
        for i in range(0, len(tx), batch):
            preds.append(model(tx[i:i + batch].to(C.device)).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return float(((pred - test_y) ** 2).mean())


def main():
    print(f"Kernel check on power-tx {C.GRID} | device {C.device} | "
          f"HIDDEN={HIDDEN} N_HIDDEN={N_HIDDEN} | kernels {KERNELS}")

    # ONE grid split, shared across kernels (K is model-side only)
    tr_g, va_g, test_grid, test_y = load_grid()

    rows = []
    for K in KERNELS:
        print(f"\n========== K={K}  ({K*K} cells in window, input {K*K*C.CHANNELS} dims) ==========")
        ckpt = f"ckpts/kcheck_{C.GRID}_K{K}.pt"
        mk = lambda: ScaleInvariantMetasurface(
            N=C.GRID_N, K=K, C=C.CHANNELS,
            n_freq=C.OUTPUT_DIM, hidden=HIDDEN, n_hidden=N_HIDDEN)
        _, best_val = fit(
            mk(), tr_g, va_g, device=C.device, ckpt_path=ckpt,
            epochs=C.EPOCHS, lr=C.LR, weight_decay=C.WEIGHT_DECAY,
            lr_decay_rate=C.LR_DECAY_RATE, lr_patience=C.LR_PATIENCE,
            eval_step=C.EVAL_STEP, stop_threshold=C.STOP_THRESHOLD)
        t = test_mse(mk(), ckpt, test_grid, test_y)
        p = n_params(mk())
        rows.append((K, p, best_val, t))
        print(f"[K={K}] params {p:,} | best val {best_val:.5f} | test {t:.5f}")

    print("\n================ KERNEL CHECK SUMMARY ================")
    print(f"{'K':>3}{'window':>9}{'in_dim':>9}{'params':>12}{'val':>11}{'test':>11}")
    for K, p, v, t in rows:
        print(f"{K:>3}{K*K:>9}{K*K*C.CHANNELS:>9}{p:>12,}{v:>11.5f}{t:>11.5f}")


if __name__ == "__main__":
    main()
