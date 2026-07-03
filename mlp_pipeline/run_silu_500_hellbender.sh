#!/bin/bash
set -euo pipefail

cd "$HOME/rep_benchmark/mlp_pipeline"
mkdir -p slurm_logs logs logs/history ckpts

set +u
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
set -u
conda activate synthgrad

export PYTHONUNBUFFERED=1
OLD_NPZ="$HOME/rep_benchmark/power_tx_data/dataset_2x2.npz"
INTEGRATED_NPZ="$HOME/rep_benchmark/power_tx_data/version_2/dataset_2x2_integrated.npz"

echo "===== SILU 500 RUNS START $(date) ====="
echo "host: $(hostname)"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "===== FLAT SILU OLD 706 $(date) ====="
POWERTX_NPZ_PATH="$OLD_NPZ" python train_powertx_silu.py

echo "===== FLAT SILU INTEGRATED 21625 $(date) ====="
POWERTX_NPZ_PATH="$INTEGRATED_NPZ" python train_powertx_silu.py

echo "===== NEIGH SILU K=3 INTEGRATED 21625 $(date) ====="
POWERTX_NPZ_PATH="$INTEGRATED_NPZ" python train_powertx_silu_neighborhood.py

echo "===== SILU 500 RUNS DONE $(date) ====="
