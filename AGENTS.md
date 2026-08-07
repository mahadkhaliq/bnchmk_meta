# Malof Lab Agent Guide

## Project Scope

This repository contains machine-learning experiments for metasurface power
transmission, including direct MLP surrogates, neighborhood models, vector
fitting, synthetic data generation, and a differentiable Lorentz finite-slab
decoder.

The user is learning the mathematics and implementation together. Explain
equations, parameter meanings, and tensor shapes step by step when discussing
model changes.

## Important Paths

- `mlp_pipeline/`: primary MLP data, model, training, evaluation, and plotting
  code.
- `lorentz/`: geometry-to-Lorentz MLP and differentiable finite-slab physics.
- `power_tx_data/version_3/`: measured/simulated v3 datasets used by the main
  1x1, 2x2, and 3x3 experiments.
- `version_4_codes_all/`: synthetic v4 generation code. Its forward model is
  distinct from the finite-slab Lorentz model.
- `reproducing_benchmark/`: benchmark reproduction material.
- `references/`: source documents used to design the physics integration,
  including the distilled metasurface-modeling slides.

## Dataset Semantics

The v3 1x1 dataset is `power_tx_data/version_3/preprocessed_1x1.npz`:

```text
atoms:    (2000, 1, 4), feature order [d, l, w, g]
T:        (2000, 2001), power transmittance
S11/S21:  (2000, 2001), complex reflection/transmission coefficients
freq_GHz: (2001,), 12 to 26 GHz
```

Use the terminology carefully:

- `S21` is the complex transmission coefficient, including amplitude and phase.
- `T = |S21|^2` is power transmittance.
- A curve of `T` versus frequency is a transmittance spectrum. A transmission
  spectrum is a broader term and must state whether it means amplitude, power,
  or complex `S21`.

## Lorentz Experiment

The active 1x1 Lorentz pipeline is documented in `lorentz/README.md`. Its main
flow is:

```text
geometry (B,1,4)
  -> shared F1 MLP
  -> constrained Lorentz parameters
  -> epsilon(omega), mu(omega), n(omega), Z(omega)
  -> finite-slab complex S11 and S21
  -> T = |S21|^2
```

The verified baseline uses a 512x3 SiLU MLP, beta-2 weighted MSE, one electric
oscillator, one magnetic oscillator, and 500 epochs. The matched `n_m=0`
ablation keeps `mu_inf` learnable but sets magnetic susceptibility to zero.

Run from the repository root:

```bash
python -m lorentz.test_1x1
python -m lorentz.train_1x1 --epochs 500
python -m lorentz.report_1x1
python -m lorentz.compare_magnetic_ablation_paired
```

## Working Rules

- Preserve existing datasets, checkpoints, plots, logs, and unrelated dirty
  worktree changes unless explicitly asked to remove them.
- Keep ablations matched in seed, split, architecture, optimizer, loss, and
  epoch budget; change only the variable being studied.
- Fit normalization statistics on the training split only.
- Report plain test MSE and beta-2 loss separately.
- Do not claim that a transmission-only fit uniquely identifies electric versus
  magnetic physics; complex reflection/phase data are needed for stronger
  identification.
- Use the existing local patterns before introducing new abstractions.
