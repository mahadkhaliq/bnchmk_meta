# Unary Lorentz validation

This directory implements the two validation experiments that should precede
the interaction model:

1. Directly fit the finite-slab Lorentz decoder to v3 1x1 spectra.
2. Train the unary F1 network on self-consistent synthetic spectra with known
   intermediate Lorentz parameters.

The experiments use the same one-electric/one-magnetic decoder, parameter
mappings, 0.2 mm slab, and 12 to 26 GHz frequency grid as the active unary
model. They do not contain a neighbor interaction network.

## Shared model and shapes

For one cell, F1 and the decoder operate as follows:

```text
geometry                       (B, 1, 4)
F1 raw output                  (B, 1, 8)
constrained physical P_i       (B, 1, 8)
complex S11 and S21            (B, 2001)
T = abs(S21)^2                 (B, 2001)
```

The eight parameters are ordered as:

```text
[wp_e, w0_e, gamma_e, wp_m, w0_m, gamma_m, epsilon_inf, mu_inf]
```

`LorentzPhysics.physical_parameters()` exposes the per-cell physical vector
without averaging backgrounds across cells. The forward pass then computes:

```text
epsilon(omega) = epsilon_inf + chi_e(omega)
mu(omega)      = mu_inf      + chi_m(omega)
n(omega)       = sqrt(epsilon * mu)
Z(omega)       = sqrt(mu / epsilon)
(S11, S21)     = finite_slab(n, Z, omega, thickness)
```

## Experiment A: direct-fit oracle

The oracle removes F1. A separate raw parameter vector is optimized for every
selected CST spectrum:

```text
raw candidates                 (B, R, 1, 8)
coarse predictions             (B, R, 501)
retain best R=8 candidates     (B, 8, 1, 8)
full-grid predictions          (B, 8, 2001)
select best candidate          (B, 1, 8)
```

The completed pilot uses 100 seed-0 test spectra, 64 starts per spectrum, 250
Adam steps on every fourth frequency, and 500 full-grid refinement steps. One
candidate is initialized from the unary Reference checkpoint and the other 63
are random. The objective is plain power-transmittance MSE.

| Predictor | Mean MSE on the same 100 spectra |
|---|---:|
| Training-target mean | 6.137e-2 |
| Unary Reference | 8.921e-5 |
| Direct-fit oracle | 5.039e-5 |

The oracle is `43.5%` lower than the unary Reference and the median oracle
sample MSE is `3.182e-5`. This means the fixed decoder can fit these spectra
better than F1 currently predicts. It does not prove that `5.039e-5` is the
global optimum because this is a 64-start pilot rather than the proposed
10,000-start exhaustive study.

The oracle optimized only `T`. Its complex-S MSE (`1.0225`) is not lower than
the Reference complex-S MSE (`1.0216`). Matching power therefore does not
recover phase or a unique physical parameter vector.

Run the numerical fit on a CUDA host without Matplotlib:

```bash
python -m lorentz.validation.fit_oracle_1x1 \
  --output-root lorentz/experiments/unary_validation_1x1_20260806/oracle_v3_1x1 \
  --samples 100 --restarts 64 --keep 8 \
  --coarse-stride 4 --coarse-steps 250 --refine-steps 500 \
  --sample-batch-size 4 --device cuda --no-plots
```

Generate the figures after the numerical artifacts are local:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lorentz /opt/anaconda3/bin/python \
  -m lorentz.validation.report_oracle_1x1 \
  --experiment-root lorentz/experiments/unary_validation_1x1_20260806/oracle_v3_1x1
```

## Experiment B: synthetic unary recovery

### Step 1: sample geometry

The generator samples `[d,l,w,g]` uniformly inside the observed v3 1x1 feature
ranges. The generated file has fixed, explicit splits:

```text
train: 2000
val:   2000
test:  2000
```

Model-input normalization is still fit using only the 2,000 training
geometries.

### Step 2: create known parameters

`synthetic.teacher_raw_parameters()` is a fixed nonlinear analytic function:

```text
normalized geometry (6000,1,4)
    -> teacher raw P (6000,1,8)
    -> constrained physical P (6000,1,8)
```

The function contains linear terms, geometry interactions, squares, and a sine
term. It is never trained, so both raw and physical `P_i` are exact labels.

### Step 3: generate spectra

The standard finite-slab decoder generates:

```text
S11: (6000,2001) complex64
S21: (6000,2001) complex64
T:   (6000,2001) float32
```

The generated `T` values range from `0.1648` to `0.99998`. This dataset uses
the active finite-slab model rather than the distinct v4 thin-sheet generator,
so it isolates encoder training and inverse identifiability from decoder model
mismatch.

Generate it with:

```bash
python -m lorentz.validation.generate_synthetic_1x1 \
  --output lorentz/experiments/unary_validation_1x1_20260806/data/synthetic_unary_1x1.npz \
  --n-train 2000 --n-val 2000 --n-test 2000
```

### Step 4: spectrum-only training

The primary experiment trains the existing 512x3 SiLU F1 for 500 epochs using
only beta-2 loss on `T`. The known parameters and complex S values are not in
the loss. They are held-out diagnostics.

| Seed | Best epoch | Test T MSE | Physical P standardized MSE |
|---:|---:|---:|---:|
| 0 | 412 | 6.753e-4 | 96.35 |
| 1 | 437 | 6.742e-4 | 166.82 |
| 2 | 497 | 9.219e-6 | 20.08 |
| Mean | - | 4.529e-4 | 94.42 |

The training-target mean MSE is `2.735e-3`, so every seed learns useful
spectral structure. However, only seed 2 reaches very low spectrum error, and
none recovers the physical teacher parameters accurately. This confirms both
an optimization-basin problem and a non-identifiable spectrum-only inverse.

Launch matched seeds with:

```bash
python -m lorentz.validation.run_synthetic_recovery \
  --dataset lorentz/experiments/unary_validation_1x1_20260806/data/synthetic_unary_1x1.npz \
  --output-root lorentz/experiments/unary_validation_1x1_20260806/synthetic_recovery \
  --epochs 500 --seeds 0 1 2 --gpus 0,1,2
```

### Step 5: parameter-supervised capacity control

The control adds standardized raw-parameter MSE with weight 1. It deliberately
uses information unavailable for CST spectra and is not the proposed final
training objective. It asks only whether the same F1 architecture can represent
the teacher mapping.

| Quantity | Spectrum only | Parameter-supervised control |
|---|---:|---:|
| Mean test T MSE | 4.529e-4 | 2.645e-6 |
| Mean complex-S MSE | 6.382e-3 | 1.951e-6 |
| Physical P standardized MSE | 94.42 | 2.550e-4 |

All three supervised seeds recover the parameters accurately. Therefore, poor
parameter recovery in the primary experiment is not caused by insufficient F1
capacity.

Generate reports with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lorentz /opt/anaconda3/bin/python \
  -m lorentz.validation.report_synthetic_recovery \
  --experiment-root lorentz/experiments/unary_validation_1x1_20260806/synthetic_recovery

MPLCONFIGDIR=/tmp/matplotlib-lorentz /opt/anaconda3/bin/python \
  -m lorentz.validation.compare_synthetic_supervision
```

## Artifact layout

```text
lorentz/experiments/unary_validation_1x1_20260806/
  data/
    synthetic_unary_1x1.npz
    synthetic_unary_1x1.metadata.json
  oracle_v3_1x1/
    fits.npz
    sample_metrics.csv
    optimization_history.csv
    summary.json
    01_error_ladder.png
    02_spectra.png
    03_optimization.png
  synthetic_recovery/
    seed_0/, seed_1/, seed_2/
    report/
  synthetic_recovery_supervised/
    seed_0/, seed_1/, seed_2/
    report/
  synthetic_comparison/
```

Run the focused compatibility test with:

```bash
/opt/anaconda3/bin/python -m lorentz.validation.test_validation
```

## Current conclusion

The unary architecture and finite-slab decoder are differentiable and capable
of representing the controlled teacher. The current decoder also has useful
capacity on v3 CST spectra. The main unresolved issue is that power-only
training admits multiple parameter solutions and can become trapped in
different optimization basins. Complex S11/S21 training is the next clean
unary experiment before introducing the pairwise interaction network.
