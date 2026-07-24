#!/bin/bash
# 8-run ConceptOneVF ablation on v4 synthetic: {2x2,3x3} x K{1,3} x n_real{0,4}.
# beta2 loss, 512x4 SiLU trunk, 8 conjugate poles. Reports MSE + beta2 on test.
# Full per-run stdout saved to logs/vf_ablation/<TAG>.log
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs/vf_ablation ckpts logs/history
export PYTHONUNBUFFERED=1
export POWERTX_EPOCHS="${POWERTX_EPOCHS:-500}"

for GRID in 2x2v4 3x3v4; do
  for K in 1 3; do
    for NR in 0 4; do
      TAG="vf_${GRID}_K${K}_nr${NR}"
      echo "===== $TAG  $(date) ====="
      POWERTX_GRID="$GRID" POWERTX_KERNEL="$K" \
        python train_powertx_vf.py --n_pole 8 --n_real "$NR" --rel offset --loss beta2 \
        2>&1 | tee "logs/vf_ablation/${TAG}.log"
    done
  done
done
echo "===== ALL 8 RUNS DONE $(date) ====="
