# Lorentz output-mapping ablation

This campaign measures what each differentiable mapping between the F1 output
and the Lorentz equations contributes. A mapping is either enabled (`1`) or
bypassed so the corresponding F1 value enters the decoder raw (`0`).

| Profile | wp | w0 | gamma | epsilon_inf | mu_inf |
|---|---:|---:|---:|---:|---:|
| `all` | 1 | 1 | 1 | 1 | 1 |
| `raw` | 0 | 0 | 0 | 0 | 0 |
| `wp_only` | 1 | 0 | 0 | 0 | 0 |
| `w0_only` | 0 | 1 | 0 | 0 | 0 |
| `gamma_only` | 0 | 0 | 1 | 0 | 0 |
| `epsilon_inf_only` | 0 | 0 | 0 | 1 | 0 |
| `mu_inf_only` | 0 | 0 | 0 | 0 | 1 |
| `drop_wp` | 0 | 1 | 1 | 1 | 1 |
| `drop_w0` | 1 | 0 | 1 | 1 | 1 |
| `drop_gamma` | 1 | 1 | 0 | 1 | 1 |
| `drop_epsilon_inf` | 1 | 1 | 1 | 0 | 1 |
| `drop_mu_inf` | 1 | 1 | 1 | 1 | 0 |

The add-one profiles are compared with `raw`. The drop-one profiles are
compared with `all`. Every profile is retrained independently with paired
seeds because changing a mapping changes both the forward values and the
backpropagated Jacobian.

## Fixed protocol

The full campaign uses the v3 1x1 dataset, seeds 0/1/2, 500 epochs, a 512x3
SiLU F1 model, one electric and one magnetic oscillator, beta-2 training loss,
plain validation MSE for checkpoint selection, Adam at `3e-4`, `wp_scale=0.5`,
`gamma_scale=0.1`, `gamma_floor=1e-4`, and a `0.2 mm` slab. This is 12 profiles
times 3 seeds, or 36 trainings.

The launcher writes:

```text
<campaign>/
  manifest.json
  experiment_matrix.csv
  campaign.log
  campaign_status.json
  runs/<profile>/seed_<seed>/
    config.json
    status.json
    train.log
    history.csv
    metrics.json
    model.pt
  plots/
    run_summary.csv
    summary.json
    01_performance_and_stability.png
    02_training_and_gradients.png
    03_physical_diagnostics.png
    04_mapping_effect_sizes.png
    05_raw_vs_constrained.png
```

Failures are scientific outcomes. If a raw run becomes non-finite, its
partial history, failure epoch/message, and best finite checkpoint diagnostics
are retained. Reporting includes MSE/beta-2/MAE, completion rate, pre-clipping
gradient norms, maximum T, maximum R+T, negative damping, backgrounds below
one, resonance values outside the mapped bounds, and learned parameter ranges.

## Launch on mlflb_m1

From `/home/mkfqm/projects/malof_lab`:

```bash
nohup /home/mkfqm/software/miniconda3/envs/metasurface-gen/bin/python -u \
  -m lorentz.run_constraint_ablation \
  --output-root lorentz/experiments/constraint_ablation_1x1_20260805 \
  --epochs 500 --seeds 0 1 2 --gpus 0,1,2,3 \
  --python /home/mkfqm/software/miniconda3/envs/metasurface-gen/bin/python \
  > lorentz/experiments/constraint_ablation_1x1_20260805/campaign.log 2>&1 &
```

The launcher assigns one training process per GPU and queues the remaining
runs. Re-running the same command skips both completed and failed terminal
runs unless `--rerun` is supplied.

## Plot locally

After copying the campaign directory into the local project:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lorentz /opt/anaconda3/bin/python \
  -m lorentz.analyze_constraint_ablation \
  --experiment-root lorentz/experiments/constraint_ablation_1x1_20260805
```

Positive values in `04_mapping_effect_sizes.png` mean that the mapping lowers
test MSE. The add-one effect is `log10(MSE_raw / MSE_only)`, while the
leave-one-out effect is `log10(MSE_drop / MSE_all)`.

## Completed results: 2026-08-05

The full 36-run campaign completed on `mlflb_m1`. The table reports the mean
test MSE and maximum R+T of each run's best finite checkpoint. A failed run is
not treated as a completed model even when a pre-failure checkpoint exists.

| Profile | Completed | Mean test MSE | Mean max R+T |
|---|---:|---:|---:|
| `all` | 3/3 | 1.450e-4 | 0.99984 |
| `raw` | 0/3 | 3.834e-3 | 1.07681 |
| `wp_only` | 0/3 | 1.803e-2 | 1.04820 |
| `w0_only` | 0/3 | 1.441e-2 | 4.07673 |
| `gamma_only` | 3/3 | 1.046e-3 | 0.99984 |
| `epsilon_inf_only` | 3/3 | 2.613e-4 | 1.10327 |
| `mu_inf_only` | 0/3 | 1.957e-2 | 2.67816 |
| `drop_wp` | 3/3 | 1.452e-4 | 0.99990 |
| `drop_w0` | 3/3 | 1.572e-4 | 0.99974 |
| `drop_gamma` | 1/3 | 1.485e-2 | 1.45271 |
| `drop_epsilon_inf` | 3/3 | 1.047e-4 | 0.99972 |
| `drop_mu_inf` | 2/3 | 1.822e-3 | 0.99930 |

The fully raw runs became non-finite at epochs 65, 63, and 103. Their best
checkpoints had negative damping for an average 65.8% of electric/magnetic
test parameters, and the mean largest pre-clipping gradient norm was
`8.74e5`. In contrast, all mapped runs completed, had no negative damping,
and remained below R+T=1.

The strongest first-order result is the damping mapping. `gamma_only`
completed all seeds and remained passive, while removing gamma completed only
one seed and produced gain. The wp mapping had little effect when the other
four mappings were active, which is consistent with wp entering susceptibility
as `wp**2`. Dropping the epsilon background mapping improved MSE in this
three-seed sample, so the assumption `epsilon_inf > 1` deserves a separate
physical-validity study rather than being accepted only because it stabilizes
the decoder.

Numerical completion alone is insufficient: `epsilon_inf_only` completed all
seeds but retained negative damping and mean max R+T above one. Accuracy,
stability, and passivity must therefore be reported together.

## Raw learning-rate control

A separate six-run endpoint control used the same three paired seeds at
`lr=3e-5` for both `all` and `raw`:

| Condition | Completed | Test MSE | Mean max R+T | Mean gamma < 0 |
|---|---:|---:|---:|---:|
| Mapped, `lr=3e-4` | 3/3 | 1.450e-4 | 0.99984 | 0.0% |
| Raw, `lr=3e-4` | 0/3 | 3.834e-3 best finite | 1.07681 | 65.8% |
| Mapped, `lr=3e-5` | 3/3 | 2.508e-4 | 0.99940 | 0.0% |
| Raw, `lr=3e-5` | 2/3 | 1.757e-4 completed only | 1.21677 completed only | 50.0% |

Lowering the learning rate rescued two raw seeds and those two fitted T well,
but one seed still failed at epoch 88. Both completed raw models learned
negative damping and violated passivity. This separates two effects: a smaller
step can mitigate raw optimization instability, but it does not restore the
physical meaning enforced by the mappings.

The combined control outputs are in
`plots/raw_learning_rate_control/01_raw_learning_rate_control.png`,
`02_raw_learning_rate_seed_comparison.png`, and the adjacent CSV/JSON files.

## Mapping-constant ablation: 2026-08-05

This focused campaign leaves every nonlinear mapping enabled and changes only
one constant relative to the Reference configuration. The internal launcher
key for the Reference is `baseline`, retained for checkpoint compatibility:

| Profile | wp scale | gamma scale | gamma floor | epsilon offset | mu offset |
|---|---:|---:|---:|---:|---:|
| Reference (`baseline`) | 0.5 | 0.1 | 1e-4 | 1 | 1 |
| `no_wp_scale` | 1 | 0.1 | 1e-4 | 1 | 1 |
| `no_gamma_scale` | 0.5 | 1 | 1e-4 | 1 | 1 |
| `no_gamma_floor` | 0.5 | 0.1 | 0 | 1 | 1 |
| `no_epsilon_offset` | 0.5 | 0.1 | 1e-4 | 0 | 1 |
| `no_mu_offset` | 0.5 | 0.1 | 1e-4 | 1 | 0 |

The protocol is 500 epochs with paired seeds 0/1/2. The table reports mean
test MSE from each run's best finite checkpoint. For failed runs, this value is
diagnostic only and is not evidence of successful training.

| Profile | Completed | Mean test MSE | Change from Reference | Main observation |
|---|---:|---:|---:|---|
| Reference (`baseline`) | 3/3 | 1.450e-4 | reference | Stable and passive |
| `no_wp_scale` | 3/3 | 1.434e-4 | -1.1% | Essentially neutral |
| `no_gamma_scale` | 3/3 | 2.648e-4 | +82.6% | Stable, but less accurate |
| `no_gamma_floor` | 0/3 | 3.876e-3 | failed | Non-finite loss at epochs 23, 223, and 21 |
| `no_epsilon_offset` | 3/3 | 1.430e-4 | -1.4% | Accurate, but epsilon background collapses below one |
| `no_mu_offset` | 1/3 | 3.245e-3 | failed | Two seeds become non-finite at epoch 20 |

Removing the wp multiplier is harmless in this experiment because the network
can compensate for a fixed positive rescaling. Removing the gamma multiplier
also preserves positivity and passivity, but gives a consistently worse fit.
The gamma floor is essential: `softplus(raw_gamma)` is strictly positive in
exact arithmetic, but can approach zero closely enough to form sharp Lorentz
poles and non-finite optimization values.

Removing the epsilon offset lets all three models use `epsilon_inf < 1`; their
mean epsilon background is about `0.0026`, paired with a mean mu background of
about `20.9`. The similar transmission MSE therefore does not demonstrate a
more physical model. It exposes non-identifiability in a transmission-only
target: epsilon and mu can trade roles while preserving much of the slab
response. Removing the mu offset is less forgiving in this parameterization:
two seeds drive mu near zero and fail, while one seed moves to the alternate
epsilon/mu branch and completes.

Outputs are under
`lorentz/experiments/mapping_constants_1x1_20260805/`. Each run retains its
config, status, log, history, metrics, and checkpoint. Aggregate CSV/JSON and
the five comparison figures are in the `plots/` subdirectory.

## Gamma scale-and-floor supplement: 2026-08-06

This twelve-run campaign tests the gamma multiplier and floor separately and
together. Only the gamma conversion changes; the wp, w0, epsilon background,
mu background, split, architecture, optimizer, loss, and epoch budget remain
matched to the Reference.

| Profile | Gamma conversion | Completed | Mean test MSE | Change from paired Reference |
|---|---|---:|---:|---:|
| Reference (`baseline`) | `0.1*softplus(x) + 1e-4` | 3/3 | 1.450e-4 | reference |
| `no_gamma_scale` | `softplus(x) + 1e-4` | 3/3 | 2.648e-4 | 82.6% higher |
| `no_gamma_floor` | `0.1*softplus(x)` | 0/3 | no completed-run MSE | failed |
| `gamma_softplus_only` | `softplus(x)` | 2/3 | 2.618e-4 | 152.2% higher* |

`*` The plain-softplus percentage uses only its completed seeds 0 and 1 and
the same two Reference seeds. Their mean Reference MSE is `1.038e-4`. Seed 2
failed at epoch 20, after 19 completed epochs, so its best-finite checkpoint
MSE (`3.576e-3`) is a failure diagnostic and is excluded from the completed-
run mean. Repeating seed 2 independently on a different GPU failed again at
epoch 20.

Removing only the `0.1` scale keeps all seeds stable, but increases the learned
mean gamma from about `0.098` to `2.153` and raises MSE. Removing only the
`1e-4` floor fails every seed at epochs 23, 223, and 21. Removing both constants
rescues seeds 0 and 1 relative to the no-floor condition, but does not make the
mapping reliably stable. Its smallest learned gamma reaches about `1e-5` on a
completed run, and seed 2 still follows the reproducible non-finite trajectory.

Therefore `softplus` enforces positive gamma, but without an additive floor it
does not enforce a useful numerical distance from zero. Among these four gamma
mappings, the Reference is both fully stable and the most accurate. All
completed runs remain passive, with maximum `R + T` below one.

The campaign outputs are in
`lorentz/experiments/mapping_gamma_constants_1x1_20260806/`. The independent
seed-2 reproduction is in
`lorentz/experiments/mapping_gamma_softplus_repeat_seed2_20260806/`.

## Wp-floor supplement: 2026-08-05

A separate nine-run campaign isolates the wp floor and tests the requested
plain-softplus mapping. All non-wp mappings and training settings remain at
their baseline values.

| Profile | Wp conversion | Completed | Test MSE |
|---|---|---:|---:|
| `baseline` | `0.5*softplus(x) + 1e-5` | 3/3 | 1.450e-4 |
| `no_wp_floor` | `0.5*softplus(x)` | 2/3 | 1.010e-4 completed runs only |
| `wp_softplus_only` | `softplus(x)` | 3/3 | 1.517e-4 |

For the two seeds that completed without the floor, paired mean MSE was
`1.010e-4` versus `1.038e-4` for the same two baseline seeds, a `2.7%`
reduction. Seed 2 became non-finite at epoch 21. Repeating that exact seed on a
different GPU produced the same failure at epoch 21, so the instability is
reproducible. It is not caused by wp approaching zero in the retained
checkpoint: the physical wp values were finite and positive. Instead, the tiny
forward change sends this nonlinear optimization trajectory into a different,
unstable basin.

Plain `softplus(x)` removes both the `0.5` multiplier and the `1e-5` floor. It
completed every seed, stayed passive, and increased mean test MSE by only
`4.6%` relative to baseline. The learned mean wp was almost unchanged
(`1.245` versus baseline `1.244`), showing that F1 compensated for the mapping
scale. Therefore plain softplus is a viable simpler mapping, but the original
baseline remains the more accurate of these two fully stable three-seed
configurations.

The main outputs are in
`lorentz/experiments/mapping_wp_floor_1x1_20260805/`; the independent seed-2
reproduction is in
`lorentz/experiments/mapping_wp_floor_repeat_seed2_20260805/`.

## Background-offset interaction: 2026-08-05

This twelve-run campaign compares the two background offsets individually and
together while keeping every oscillator mapping at baseline:

| Profile | epsilon mapping | mu mapping | Completed | Mean test MSE |
|---|---|---|---:|---:|
| `baseline` | `1 + softplus(x)` | `1 + softplus(x)` | 3/3 | 1.450e-4 |
| `no_epsilon_offset` | `softplus(x)` | `1 + softplus(x)` | 3/3 | 1.430e-4 |
| `no_mu_offset` | `1 + softplus(x)` | `softplus(x)` | 1/3 | 1.028e-4 completed run only |
| `no_background_offsets` | `softplus(x)` | `softplus(x)` | 3/3 | 1.270e-4 |

Removing both offsets reduced mean test MSE by `12.4%` and mean beta-2 loss by
`13.6%` relative to baseline, while all three seeds completed and remained
passive. The seed-level MSE changes were not uniform: seeds 0 and 2 improved,
while seed 1 worsened. This is therefore a useful three-seed result rather than
evidence of a universally better mapping.

The learned backgrounds are not physically identified. Seeds 0 and 1 learned
epsilon backgrounds near `0.002` and mu backgrounds near `20.5`; seed 2 learned
the opposite branch, with epsilon near `21.4` and mu near `0.002`. The stable
transmittance fit despite this exchange reinforces that power transmittance
alone cannot uniquely distinguish effective epsilon from mu. Complex S11/S21
amplitude and phase would be needed for stronger identification.

Outputs are in
`lorentz/experiments/mapping_background_offsets_1x1_20260805/`, including all
checkpoints, logs, histories, aggregate CSV/JSON, and five comparison figures.
