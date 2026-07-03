#!/bin/bash
#SBATCH --job-name=powertx_full_relu
#SBATCH --output=/home/mkfqm/rep_benchmark/mlp_pipeline/slurm_logs/powertx_full_relu_%j.out
#SBATCH --error=/home/mkfqm/rep_benchmark/mlp_pipeline/slurm_logs/powertx_full_relu_%j.err
#SBATCH --account=engineering
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

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

export POWERTX_NPZ_PATH="$HOME/rep_benchmark/power_tx_data/version_2/dataset_2x2_integrated.npz"
export PYTHONUNBUFFERED=1

echo "===== START $(date) ====="
echo "host: $(hostname)"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

python train_powertx_full_integrated_relu.py

echo "===== DONE $(date) ====="
