# Finite-slab Lorentz 1x1 experiment

This directory contains the shared two-stage model:

1. `F1` is a 512-wide SiLU MLP that maps each cell's normalized `[d,l,w,g]`
   geometry to electric and magnetic Lorentz parameters.
2. `LorentzPhysics` converts those parameters to complex `S11` and `S21`
   using the finite-thickness slab equations.

The supervised 1x1 target is power transmittance:

```text
T_pred = abs(S21_pred)^2
```

Training uses the beta-2 dip-weighted loss:

```text
weight = 1 + 2 * (1 - T_target)^2
loss = mean(weight * (T_pred - T_target)^2)
```

Plain validation MSE is retained for checkpoint selection and comparable
reporting.

The implementation frequency-normalizes `wp`, `w0`, and `gamma` and maps raw
network outputs into physical ranges. This preserves the Lorentz ratios while
avoiding an MLP that must directly emit angular frequencies near `1e11 rad/s`.
Oscillators are summed and cells are averaged.

The positive `wp` and `gamma` mappings have configurable numerical scales:

```text
wp    = wp_scale    * softplus(raw_wp)    + 1e-5
gamma = gamma_scale * softplus(raw_gamma) + gamma_floor
```

They default to `0.5` and `0.1`. These values condition the optimization; they
are not upper bounds. Temporary alternatives can be selected with
`--wp-scale` and `--gamma-scale`.

The default slab thickness is `0.2 mm`, matching the fixed FR4 thickness
recorded for v3. It is an explicit command-line option because effective slab
thickness should eventually be confirmed or fitted.

From the repository root:

```bash
/opt/anaconda3/bin/python -m lorentz.test_1x1
/opt/anaconda3/bin/python -m lorentz.train_1x1 --epochs 500
/opt/anaconda3/bin/python -m lorentz.report_1x1
/opt/anaconda3/bin/python -m lorentz.compare_magnetic_ablation
/opt/anaconda3/bin/python -m lorentz.compare_magnetic_ablation_paired
/opt/anaconda3/bin/python -m lorentz.compare_scale_ablation
```

For a short pipeline check:

```bash
/opt/anaconda3/bin/python -m lorentz.train_1x1 \
  --epochs 5 --max-samples 256 \
  --checkpoint /tmp/lorentz_1x1_smoke.pt
```

## Unary validation experiments

The direct-fit decoder oracle and self-consistent synthetic parameter-recovery
experiments are organized under [`validation/`](validation/). Their complete
step-by-step protocol, tensor shapes, commands, results, and interpretation are
in [`validation/README.md`](validation/README.md).

The completed validation artifacts are under
`experiments/unary_validation_1x1_20260806/`. The main findings are:

```text
100-spectrum v3 unary Reference MSE:       8.921e-5
100-spectrum direct-fit oracle MSE:        5.039e-5
synthetic spectrum-only mean T MSE:        4.529e-4
synthetic supervised-control mean T MSE:   2.645e-6
```

The synthetic spectrum-only runs fit T better than the training-target mean,
but do not recover the known physical parameters consistently. A supervised
parameter control recovers them accurately, showing that F1 capacity is not
the limiting factor. The next unary study should use complex S11/S21 targets
before the pairwise interaction model is added.

This is the finite-slab model from the complex-S-fitting slide. It is distinct
from the thin-sheet forward model used to generate synthetic v4.

## Verified 1x1 run

The full v3 1x1 dataset was tested on CPU with seed 0, a deterministic
1360/340/300 train/validation/test split, a 512x3 SiLU `F1`, beta-2 training,
one electric oscillator, one magnetic oscillator, and 500 epochs:

```text
selected epoch:      489
best validation MSE: 0.00010635
test MSE:            0.00009772
test beta2:          0.00014772
test MAE:            0.006779
median sample MSE:   0.00007125
predicted T range:   2.39e-20 to 0.989066
max test R + T:      0.999855
```

The checkpoint is `artifacts/best_1x1_silu_beta2_512_500ep.pt`; its 500-row
history and final metrics are stored beside it. Test plots, a summary, and
per-sample metrics are in `artifacts/report_1x1_500ep/`.

## Zero-magnetic-oscillator ablation

The same 500-epoch protocol was repeated with one electric oscillator and no
magnetic oscillator:

```bash
/opt/anaconda3/bin/python -m lorentz.train_1x1 \
  --epochs 500 --n-e 1 --n-m 0 \
  --checkpoint lorentz/artifacts/best_1x1_silu_beta2_512_500ep_ne1_nm0.pt
```

This changes the per-cell F1 output from 8 to 5:

```text
(B,1,5) = [wp_e, w0_e, gamma_e, eps_inf, mu_inf]
```

The magnetic susceptibility is exactly zero, while the constant `mu_inf`
remains learnable. It converged very close to one on the test set
(`1.000000` to `1.000136`). Therefore, this is a zero-magnetic-oscillator
ablation, not the stronger fixed constraint `mu_r=1`.

```text
selected epoch:      462
best validation MSE: 0.00024663
test MSE:            0.00025869
test beta2:          0.00038757
test MAE:            0.012400
median sample MSE:   0.00020610
max test R + T:      0.999979
```

The electric-only model has 530,437 trainable parameters versus 531,976 for
the full model. Removing the magnetic oscillator increased test MSE by
`164.7%` (`2.647x`). The full model had lower MSE on `279/300` paired test
samples and at `1993/2001` frequency points. This demonstrates that the second
dispersive pole improves the fit. It does not uniquely establish a magnetic
physical origin: transmission through the finite slab is symmetric under
interchanging effective epsilon and mu, and this training target contains no
complex S11 phase information.

The zero-magnetic diagnostics and compact run-level comparison are in
`artifacts/report_1x1_500ep_ne1_nm0/`. The detailed paired figure, per-sample
CSV, and JSON summary are in `artifacts/compare_magnetic_ablation_500ep/`.
Regenerate both comparisons with:

```bash
/opt/anaconda3/bin/python -m lorentz.compare_magnetic_ablation
/opt/anaconda3/bin/python -m lorentz.compare_magnetic_ablation_paired
```

## Temporary scale ablation

Four controlled 100-epoch, seed-0 runs compare `(wp_scale, gamma_scale)` values
of `(0.5,0.1)`, `(1,0.1)`, `(0.5,1)`, and `(1,1)`. Setting a scale to one
removes that numerical multiplier but retains positivity, the gamma floor,
the bounded resonance frequency, constrained backgrounds, and passive
square-root branches.

The checkpoints, histories, metrics, comparison plot, and JSON summary are in
`artifacts/scale_ablation_100ep/`. Regenerate the report with:

```bash
/opt/anaconda3/bin/python -m lorentz.compare_scale_ablation
```

## Raw-parameter experiment

Use `--raw-physics-parameters` to feed F1 outputs directly into the Lorentz
equations:

```bash
/opt/anaconda3/bin/python -m lorentz.train_1x1 \
  --epochs 100 --raw-physics-parameters \
  --checkpoint lorentz/artifacts/raw_parameters_100ep/raw_1e1m.pt
```

This removes the `softplus` mappings, oscillator floors, bounded `w0`, and
constrained `epsilon_inf`/`mu_inf`. Frequency normalization and passive
square-root branch selection remain. The standard-learning-rate run became
non-finite after epoch 50. A `3e-5` learning-rate run completed but learned
non-passive values and had substantially higher error. Results are in
`artifacts/raw_parameters_100ep/`; regenerate their comparison with:

```bash
/opt/anaconda3/bin/python -m lorentz.compare_raw_parameters
```

## Output-mapping constraint matrix

The reproducible 12-profile add-one/leave-one-out matrix is documented in
[`CONSTRAINT_ABLATION.md`](CONSTRAINT_ABLATION.md). The GPU campaign launcher
is `run_constraint_ablation.py`; `analyze_constraint_ablation.py` aggregates
paired seeds and creates accuracy, optimization, passivity, parameter-range,
effect-size, and raw-versus-constrained plots.

The completed three-seed results and low-learning-rate raw control are also
summarized in that document. Regenerate the cross-rate control plots with:

```bash
/opt/anaconda3/bin/python -m lorentz.compare_raw_learning_rate \
  --standard-root lorentz/experiments/constraint_ablation_1x1_20260805 \
  --low-lr-root lorentz/experiments/raw_lr_control_1x1_20260805
```

Arbitrary mapping subsets can also be selected directly. For example, this
leaves only `wp` raw while retaining the other four mappings:

```bash
/opt/anaconda3/bin/python -m lorentz.train_1x1 \
  --constraints w0,gamma,epsilon_inf,mu_inf
```

## Mapping-constant ablation

The focused 500-epoch campaign keeps all five mappings enabled and removes one
constant at a time: the `0.5` wp scale, `0.1` gamma scale, `1e-4` gamma floor,
`+1` epsilon background offset, or `+1` mu background offset. It uses the same
v3 1x1 data, seeds 0/1/2, 512x3 SiLU F1 model, beta-2 loss, and `3e-4` learning
rate as the constraint campaign.

```bash
/home/mkfqm/software/miniconda3/envs/metasurface-gen/bin/python \
  -m lorentz.run_mapping_constant_ablation \
  --output-root lorentz/experiments/mapping_constants_1x1_20260805 \
  --epochs 500 --seeds 0 1 2 --gpus 0,1,2,3

MPLCONFIGDIR=/tmp/matplotlib-lorentz /opt/anaconda3/bin/python \
  -m lorentz.analyze_mapping_constant_ablation \
  --experiment-root lorentz/experiments/mapping_constants_1x1_20260805
```

The complete matrix, completion rates, numerical results, and interpretation
are in [`CONSTRAINT_ABLATION.md`](CONSTRAINT_ABLATION.md).

The gamma supplement compares the Reference conversion
`0.1*softplus(x) + 1e-4` with removing the scale, floor, or both:

```bash
/home/mkfqm/software/miniconda3/envs/metasurface-gen/bin/python \
  -m lorentz.run_mapping_constant_ablation \
  --output-root lorentz/experiments/mapping_gamma_constants_1x1_20260806 \
  --epochs 500 --seeds 0 1 2 --gpus 0,1,2,3 \
  --profiles baseline no_gamma_scale no_gamma_floor gamma_softplus_only
```

Plain `gamma = softplus(x)` completed seeds 0 and 1 with mean test MSE
`2.618e-4`, but seed 2 reproducibly became non-finite at epoch 20. This is
`152.2%` higher MSE than the same two Reference seeds. Removing the floor while
retaining the `0.1` scale failed all three seeds. See the gamma supplement in
[`CONSTRAINT_ABLATION.md`](CONSTRAINT_ABLATION.md) for the full comparison.

The follow-up wp-floor campaign compares the original mapping with both
`0.5*softplus(x)` and plain `softplus(x)`:

```bash
/home/mkfqm/software/miniconda3/envs/metasurface-gen/bin/python \
  -m lorentz.run_mapping_constant_ablation \
  --output-root lorentz/experiments/mapping_wp_floor_1x1_20260805 \
  --epochs 500 --seeds 0 1 2 --gpus 0,1,2,3 \
  --profiles baseline no_wp_floor wp_softplus_only
```

Plain `softplus(x)` completed all three seeds with mean test MSE `1.517e-4`.
Removing only the wp floor completed two seeds; seed 2 reproducibly became
non-finite at epoch 21.

Generate equation-level illustrations of how these mappings change the
physical parameters, Lorentz susceptibility, and finite-slab transmittance:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lorentz python \
  -m lorentz.plot_mapping_constant_effects
```

The figures and selected numerical values are written to
`lorentz/artifacts/mapping_effects/`.

The matched background-offset interaction campaign additionally tests removing
the epsilon and mu `+1` offsets together:

```bash
/home/mkfqm/software/miniconda3/envs/metasurface-gen/bin/python \
  -m lorentz.run_mapping_constant_ablation \
  --output-root lorentz/experiments/mapping_background_offsets_1x1_20260805 \
  --epochs 500 --seeds 0 1 2 --gpus 0,1,2,3 \
  --profiles baseline no_epsilon_offset no_mu_offset no_background_offsets
```

The simultaneous removal completed all three seeds with mean test MSE
`1.270e-4`. Its learned backgrounds exhibit epsilon/mu branch exchange rather
than uniquely identified material parameters.

Plot paired training beta-2 and validation-MSE histories for all three seeds:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lorentz python \
  -m lorentz.plot_background_offset_seed_curves
```

## Zero-shot 1x1 to 2x2

The earlier 100-epoch 1x1 checkpoint was evaluated without retraining on the
deterministic 975-sample v3 2x2 test split:

```text
per-sample shapes: (2,2,4) -> (4,4) -> F1 -> (4,8) -> T (2001)
test MSE:          0.11140456
test beta2:        0.22695905
test MAE:          0.21881695
max test R + T:    0.996829
```

Permuting the four cells changes predictions by at most `2.98e-7`, confirming
that the current isolated-cell averaging model cannot represent spatial
arrangement or neighbour coupling. The evaluator is
`evaluate_zero_shot_2x2.py`; machine-readable results are stored in
`artifacts/zero_shot_1x1_to_2x2.json`.
