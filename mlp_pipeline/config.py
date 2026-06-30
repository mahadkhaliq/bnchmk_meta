"""
Central configuration for both MLP variants on the ADM dataset.

(The power-transmission .npz datasets have their own separate config in
config_powertx.py and their own train_powertx_*.py scripts.)

To switch ADM to a different CSV dataset you normally only need to edit:
    1. "DATA PATHS"     -> point at your 4 CSV files
    2. "DATASET SHAPE"  -> set INPUT_DIM / OUTPUT_DIM (and, for the neighbourhood
                           model, GRID_N / KERNEL / CHANNELS + the grid mapping
                           in data/grid.py)
"""
import torch

# ---------------------------------------------------------------------------
# DATA PATHS
# 4 CSV files, no header. *_X = geometry/features, *_Y = target spectrum.
# Dataset: ADM (local copy on this machine).
# ---------------------------------------------------------------------------
ADM_DIR = "/Users/mkfqm/reproducing_work/data/ADM"

TRAIN_X_PATH = f"{ADM_DIR}/data_g.csv"
TRAIN_Y_PATH = f"{ADM_DIR}/data_s.csv"
TEST_X_PATH  = f"{ADM_DIR}/testset/test_g.csv"
TEST_Y_PATH  = f"{ADM_DIR}/testset/test_s.csv"

# ---------------------------------------------------------------------------
# DATASET SHAPE
# ---------------------------------------------------------------------------
INPUT_DIM  = 14     # raw geometry features per sample (ADM = 14)
OUTPUT_DIM = 2001   # length of the target spectrum (ADM = 2001)

# ---------------------------------------------------------------------------
# BASELINE MLP  (NO neighbouring): flat INPUT_DIM -> OUTPUT_DIM
# LAYERS is the full per-layer width list. The ADM baseline is 11 Linear layers
# (10 hidden of width 2000), reproducing Deng et al. 2021.
# ---------------------------------------------------------------------------
HIDDEN   = 2000
N_HIDDEN = 10
LAYERS   = [INPUT_DIM] + [HIDDEN] * N_HIDDEN + [OUTPUT_DIM]

# ---------------------------------------------------------------------------
# NEIGHBOURHOOD / SCALE-INVARIANT MLP  (WITH neighbouring)
# The geometry is reshaped into a GRID_N x GRID_N grid of cells, each holding
# CHANNELS features. Each cell is described by its KERNEL x KERNEL neighbourhood
# (with wrap-around padding), flattened to KERNEL*KERNEL*CHANNELS and fed through
# a SHARED MLP; the per-cell outputs are averaged.
# ---------------------------------------------------------------------------
GRID_N   = 2    # grid is GRID_N x GRID_N cells  (ADM = 2x2 = 4 resonators)
KERNEL   = 3    # K x K neighbourhood window (odd number; 3 or 5 in the notebook)
CHANNELS = 5    # features stored per cell   (ADM = rx, ry, theta, h, p)

# ---------------------------------------------------------------------------
# OPTIMISATION  (shared by both variants)
# ---------------------------------------------------------------------------
BATCH_SIZE     = 1024
EPOCHS         = 500
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
LR_DECAY_RATE  = 0.2     # ReduceLROnPlateau factor
LR_PATIENCE    = 10
EVAL_STEP      = 10      # validate + maybe-checkpoint every N epochs
STOP_THRESHOLD = 1e-7    # early stop once best val loss drops below this

# ---------------------------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------------------------
BASELINE_CKPT     = "best_mlp_baseline.pt"
NEIGHBORHOOD_CKPT = "best_mlp_neighborhood.pt"

# Prefer CUDA, then Apple-Silicon MPS, then CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
