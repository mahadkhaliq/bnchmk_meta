"""
Config for the power-transmission .npz datasets (separate from ADM's config.py).

Power-transmission datasets live in /Users/mkfqm/malof_lab/power_tx_data:
    dataset_1x1.npz   3006 samples,  4 params,  freq 2001
    dataset_2x2.npz    706 samples, 16 params,  freq 2001
    dataset_3x3.npz    484 samples, 36 params,  freq 2003
    version_2/dataset_2x2_integrated.npz  21625 samples, 16 params, freq 2001

Each cell has 4 params (d, g, l, w). Target = T_clean (|S21|^2 power transmission,
already in [0, 1]). Unlike ADM these are single .npz files with no separate test
set, so we carve the test split out ourselves (TEST_SPLIT).
"""
import os

import torch

POWERTX_DIR = "/Users/mkfqm/malof_lab/power_tx_data"

# ===========================================================================
# >>> PICK THE SUPER-CELL SIZE HERE <<<
# ===========================================================================
GRID = "2x2"   # one of: "1x1", "2x2", "3x3"

_SPECS = {
    "1x1": dict(npz="dataset_1x1.npz", grid_n=1, input_dim=4,  output_dim=2001, batch_size=128),
    "2x2": dict(npz="version_2/dataset_2x2_integrated.npz", grid_n=2, input_dim=16, output_dim=2001, batch_size=128),
    "3x3": dict(npz="dataset_3x3.npz", grid_n=3, input_dim=36, output_dim=2003, batch_size=64),
}
_s = _SPECS[GRID]

NPZ_PATH   = os.environ.get("POWERTX_NPZ_PATH", f"{POWERTX_DIR}/{_s['npz']}")
GRID_N     = _s["grid_n"]
INPUT_DIM  = _s["input_dim"]
OUTPUT_DIM = _s["output_dim"]
BATCH_SIZE = _s["batch_size"]

CHANNELS   = 4      # per-cell params: d, g, l, w
KERNEL     = 3      # K x K neighbourhood window (used by the neighbourhood model)
TEST_SPLIT = 0.15   # fraction of the single .npz held out as the test set

# ---------------------------------------------------------------------------
# MODEL WIDTH
# Defaults match the ADM "MLP as in the paper" (2000 x 10, ~40M params).
# WARNING: these power-transmission files are small (hundreds of samples), so a
# 40M-param net WILL overfit. For real results shrink these, e.g.:
#     HIDDEN = 256 ; N_HIDDEN = 4
# ---------------------------------------------------------------------------
HIDDEN   = 2000
N_HIDDEN = 10
LAYERS   = [INPUT_DIM] + [HIDDEN] * N_HIDDEN + [OUTPUT_DIM]

# ---------------------------------------------------------------------------
# OPTIMISATION
# ---------------------------------------------------------------------------
EPOCHS         = 500
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
LR_DECAY_RATE  = 0.2
LR_PATIENCE    = 10
EVAL_STEP      = 10
STOP_THRESHOLD = 1e-7

# ---------------------------------------------------------------------------
# CHECKPOINTS  (namespaced by grid size)
# ---------------------------------------------------------------------------
BASELINE_CKPT     = f"best_powertx_baseline_{GRID}.pt"
NEIGHBORHOOD_CKPT = f"best_powertx_neighborhood_{GRID}.pt"
CONCEPT_ONE_CKPT  = f"best_powertx_concept_one_{GRID}.pt"

# ---------------------------------------------------------------------------
# CONCEPT #1 MODEL
# Physics-inspired additive interactions with a trainable neural decoder:
#   U(cell) + sum_j V(cell, neighbour_j, relative_position_j) -> D(latent)
# This latent vector plays the role of relaxed, learned Lorentz parameters.
# ---------------------------------------------------------------------------
CONCEPT_LATENT_DIM = 64

# Prefer CUDA, then Apple-Silicon MPS, then CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
