# ConceptOneVF relative-position ablation

This document fixes the experiment contract for comparing relative-position
encodings in the ConceptOneVF interaction trunk. The executable study is
[`run_vf_relative_ablation.sh`](run_vf_relative_ablation.sh).

Last updated: 2026-07-24.

## Research question

For a fixed `K=3` neighbourhood, does explicitly telling the interaction
network where each neighbour lies improve power-transmission prediction? The
architecture, optimizer, split seed, target construction, and training budget
remain fixed while only the relative encoding and `n_real` change.

The informal names used in discussion map to CLI names as follows:

- `offset_distance` -> `offset_dist`
- `anglething` -> `polar`

## Relative encodings

For centre cell \(i\), neighbour \(k\), row offset \(d_r\), column offset
\(d_c\), and \(p=\lfloor K/2\rfloor\), the interaction input is

\[
[x_i,\;x_k,\;\operatorname{rel}(d_r,d_c)].
\]

| CLI mode | Relative features | Extra width | V input width for 4 channels |
|---|---|---:|---:|
| `none` | no relative features | 0 | 8 |
| `offset` | \([d_r/p,\ d_c/p]\) | 2 | 10 |
| `offset_dist` | offset plus \(\sqrt{d_r^2+d_c^2}/(p\sqrt{2})\) | 3 | 11 |
| `embed` | learned 8-vector for each discrete offset slot | 8 | 16 |
| `polar` | \([r/(p\sqrt{2}),\cos\theta,\sin\theta]\) | 3 | 11 |

The polar angle is

\[
\theta=\operatorname{atan2}(-d_r,d_c),
\]

so right is \(0\), up is \(+\pi/2\), left is \(\pi\), and down is
\(-\pi/2\). Using sine and cosine avoids the discontinuity of a raw angle at
\(-\pi/\pi\).

With fixed `K=3`, increasing the grid from 2x2 to 3x3 or larger increases the
number of centre cells but does not change the eight relative offset angles.
On a periodic 2x2 grid, multiple directional slots can wrap to the same
physical cell; their relative encodings still distinguish the slots.

## Fixed training recipe

| Setting | Value |
|---|---|
| model | ConceptOneVF |
| neighbourhood | `K=3`, eight neighbour slots |
| cell features | 4: `d`, `l`, `w`, `g` for v3/v4 |
| hidden trunk | 4 hidden layers, width 512, SiLU |
| latent width | 64 |
| complex pole pairs | `n_pole=8` |
| real poles | `n_real` in `{0, 4}` |
| train loss | beta2 weighted MSE |
| validation/model selection | plain MSE |
| reported test metrics | plain MSE and beta2 |
| optimizer | Adam, learning rate \(3\times10^{-4}\), weight decay \(10^{-5}\) |
| schedule | cosine annealing over 500 epochs |
| gradient clipping | global norm 1.0 |
| batch size | 128 |
| split | deterministic random 68/17/15, seed 0 |
| epochs | 500 |

The losses are

\[
\operatorname{MSE}
=\frac{1}{NF}\sum_{i=1}^{N}\sum_{f=1}^{F}
(\hat T_{if}-T_{if})^2,
\]

\[
w_{if}=1+2(1-T_{if})^2,
\qquad
\mathcal L_{\beta2}
=\frac{1}{NF}\sum_{i=1}^{N}\sum_{f=1}^{F}
w_{if}(\hat T_{if}-T_{if})^2.
\]

## Data and splits

Both versions use 2,001 frequency points from 12 to 26 GHz.

| Domain | Grid | Total | Train | Validation | Test | Target used here |
|---|---:|---:|---:|---:|---:|---|
| v3 CST | 2x2 | 6,497 | 4,417 | 1,105 | 975 | `clip(abs(S21)^2, 0, 1)` |
| v3 CST | 3x3 | 397 | 269 | 68 | 60 | `clip(abs(S21)^2, 0, 1)` |
| v4 synthetic | 2x2 | 20,000 | 13,600 | 3,400 | 3,000 | stored `T` |
| v4 synthetic | 3x3 | 20,000 | 13,600 | 3,400 | 3,000 | stored `T` |

The split is by samples, not by mini-batches. Mini-batches are formed from the
already-created training partition. The v3 split is not batch-disjoint, so its
numbers are interpolation benchmarks.

## Training matrix

The Cartesian product below defines 40 native checkpoints:

| Factor | Values | Count |
|---|---|---:|
| training domain | v3 CST, v4 synthetic | 2 |
| grid | 2x2, 3x3 | 2 |
| real poles | 0, 4 | 2 |
| relative mode | none, offset, offset_dist, embed, polar | 5 |
| **Total** | \(2\times2\times2\times5\) | **40** |

Each checkpoint is evaluated on its native domain and the opposite domain at
the same grid size. This produces 80 evaluation rows:

| Evaluation direction | Rows |
|---|---:|
| v3 -> v3 | 20 |
| v3 -> v4 | 20 |
| v4 -> v4 | 20 |
| v4 -> v3 | 20 |
| **Total** | **80** |

## Parameter counts

Only V's input projection and the optional embedding table change between
relative modes.

| `n_real` | none | offset | offset_dist | embed | polar |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,650,978 | 1,652,002 | 1,652,514 | 1,655,146 | 1,652,514 |
| 4 | 1,651,498 | 1,652,522 | 1,653,034 | 1,655,666 | 1,653,034 |

## Remote execution and artifacts

Remote project:

```text
/home/mkfqm/rep_benchmark/mlp_pipeline
```

Remote environment:

```text
/home/mkfqm/software/miniconda3/envs/metasurface-gen
```

Remote dataset root:

```text
/mnt/DATA/mkfqm/rep_benchmark/datasets
```

Primary artifacts:

| Artifact | Path |
|---|---|
| master progress log | `logs/vf_rel_ablation_master.log` |
| per-run training logs | `logs/vf_rel_ablation/train/` |
| per-evaluation logs | `logs/vf_rel_ablation/eval/` |
| 80-row metric table | `logs/vf_rel_ablation/evaluations.csv` |
| seeded random plus best/worst predictions and per-sample losses | `logs/vf_rel_ablation_relabl_stable_v3/preds/` |
| epoch histories | `logs/history/vf_*_beta2_512x4_500ep_seed0.csv` |
| run metadata | `logs/vf_*_beta2_512x4_500ep_seed0_meta.txt` |
| checkpoints | `ckpts/vf_*_beta2_512x4_500ep_seed0.pt` |

The runner is resumable: a training run is skipped only when its metadata file
contains a `RESULT` line. Evaluation is deliberately sequential so appends to
the shared CSV cannot race.

The corrected controlled run uses study ID `relabl_stable_v3`. Its checkpoints,
histories, and metadata end in `_relabl_stable_v3`; its evaluation directory is
`logs/vf_rel_ablation_relabl_stable_v3/`. This deliberately avoids reusing
older checkpoints produced before the remote code-parity audit.

The VF head uses a straight-through physical clamp: forward predictions are
exactly bounded to `[0,1]`, while gradients follow the underlying
pole-residue power and are controlled by global norm clipping. A conventional
hard clamp created an absorbing all-ones state in some relative modes.

Before the final launch, matched 120-epoch stability runs compared learning
rates \(10^{-4}\) and \(3\times10^{-4}\). The latter remained finite and
produced lower held-out MSE on both tested difficult v4 cases, so it is fixed
for every combination in this ablation.

## Verified results

Study `relabl_stable_v3` completed all 40 trainings and all 80 native/transfer
evaluations on 2026-07-24. Every history and reported metric is finite. Entries
below are `test MSE / test beta2`; bold marks the best value within one dataset,
grid, and `n_real` column.

### Configuration matrix

The study contains 40 independently trained checkpoints and 80 evaluations:

| Ablation axis | Values | Count |
|---|---|---:|
| Training dataset | v3 CST, v4 synthetic | 2 |
| Grid | 2x2, 3x3 | 2 |
| Relative encoding | none, offset, offset_dist, embed, polar | 5 |
| Real poles | `n_real=0`, `n_real=4` | 2 |
| Evaluation domain | native, opposite-domain transfer | 2 |

Thus, \(2\times2\times5\times2=40\) trained models and
\(40\times2=80\) test evaluations. The complete row-level table is
`plots/vf_research_suite_relabl_stable_v3/all_80_experiment_configurations.csv`.

#### Experiment switches

| Purpose | Exact switch | Values used | Status |
|---|---|---|---|
| Dataset and grid | `POWERTX_GRID` | `2x2v3`, `3x3v3`, `2x2v4`, `3x3v4` | Ablated |
| Relative-position representation | `--rel` | `none`, `offset`, `offset_dist`, `embed`, `polar` | Ablated |
| Real VF poles | `--n_real` | `0`, `4` | Ablated |
| Test domain | evaluation `POWERTX_GRID` | native dataset, opposite dataset | Ablated at evaluation |
| Target construction | `POWERTX_TARGET` | `s21` for v3, `t` for v4 | Dataset-linked |
| Complex pole pairs | `--n_pole` | `8` | Fixed |
| Neighborhood size | `POWERTX_KERNEL`, evaluation `--K` | `3` | Fixed |
| Training objective | `--loss` | `beta2` | Fixed |
| Epochs | `POWERTX_EPOCHS` | `500` | Fixed |
| Learning rate | `POWERTX_LR` | `3e-4` | Fixed |
| Gradient clipping | `POWERTX_GRAD_CLIP` | `1.0` | Fixed |
| Random seed | `POWERTX_SEED` | `0` | Fixed |
| Study namespace | `--run_id`, `VF_STUDY_ID` | `relabl_stable_v3` | Bookkeeping |

`POWERTX_GRID` on the training command selects the data used to fit the
checkpoint. On the evaluation command it selects the held-out data presented
to that unchanged checkpoint; switching only the evaluation grid creates the
cross-domain test.

| Fixed setting | Value |
|---|---:|
| Neighborhood | K=3 |
| Complex pole pairs | `n_pole=8` |
| Trunk | 4 hidden layers, width 512 |
| Latent dimension | 64 |
| Training loss | beta2-weighted MSE |
| Validation/model-selection metric | plain MSE |
| Reported test metrics | plain MSE and beta2 |
| Epochs | 500 |
| Batch size | 128 |
| Adam learning rate | 0.0003 |
| Weight decay | 0.00001 |
| Gradient clipping | 1.0 |
| Seed | 0 |

| Dataset | Grid | Relative mode | `n_real=0` | `n_real=4` |
|---|---:|---|---:|---:|
| v4 synthetic | 2x2 | none | 0.004320 / 0.006389 | 0.004252 / 0.006257 |
| v4 synthetic | 2x2 | offset | 0.001251 / 0.001941 | 0.000972 / 0.001525 |
| v4 synthetic | 2x2 | offset_dist | 0.001793 / 0.002747 | 0.001001 / 0.001576 |
| v4 synthetic | 2x2 | embed | **0.001030 / 0.001571** | 0.000966 / 0.001511 |
| v4 synthetic | 2x2 | polar | 0.001759 / 0.002716 | **0.000902 / 0.001419** |
| v4 synthetic | 3x3 | none | 0.005436 / 0.008665 | 0.004643 / 0.007273 |
| v4 synthetic | 3x3 | offset | 0.002705 / 0.004437 | 0.002756 / 0.004514 |
| v4 synthetic | 3x3 | offset_dist | 0.002729 / 0.004430 | 0.002783 / 0.004515 |
| v4 synthetic | 3x3 | embed | 0.002650 / 0.004206 | **0.002515 / 0.004053** |
| v4 synthetic | 3x3 | polar | **0.002574 / 0.004141** | 0.002896 / 0.004690 |
| v3 CST | 2x2 | none | 0.007864 / 0.012601 | 0.008656 / 0.014248 |
| v3 CST | 2x2 | offset | 0.006315 / 0.009997 | 0.006022 / 0.009844 |
| v3 CST | 2x2 | offset_dist | 0.004800 / 0.007392 | 0.004525 / 0.006946 |
| v3 CST | 2x2 | embed | **0.004031 / 0.006138** | **0.003934 / 0.006051** |
| v3 CST | 2x2 | polar | 0.005243 / 0.008091 | 0.004790 / 0.007376 |
| v3 CST | 3x3 | none | 0.031294 / 0.050308 | 0.031389 / 0.051477 |
| v3 CST | 3x3 | offset | 0.031801 / 0.052096 | **0.030988** / 0.050285 |
| v3 CST | 3x3 | offset_dist | **0.031284** / 0.050647 | 0.032458 / 0.049764 |
| v3 CST | 3x3 | embed | 0.031312 / **0.049019** | 0.031999 / 0.051350 |
| v3 CST | 3x3 | polar | 0.033641 / 0.050588 | 0.032283 / **0.049621** |

### Best native configurations

| Dataset | Grid | Best test-MSE configuration | Test MSE | Test beta2 | Reduction from same-`n_real` `none` |
|---|---:|---|---:|---:|---:|
| v4 synthetic | 2x2 | polar, `n_real=4` | 0.000902 | 0.001419 | 78.8% |
| v4 synthetic | 3x3 | embed, `n_real=4` | 0.002515 | 0.004053 | 45.8% |
| v3 CST | 2x2 | embed, `n_real=4` | 0.003934 | 0.006051 | 54.6% |
| v3 CST | 3x3 | offset, `n_real=4` | 0.030988 | 0.050285 | 1.3% |

The v3 3x3 differences are too small to support a strong encoding conclusion:
there are only 397 total samples and 60 test samples. Its MSE and beta2 winners
also differ.

### Best transfer configurations

| Train -> test | Grid | Best test-MSE configuration | Test MSE | Test beta2 |
|---|---:|---|---:|---:|
| v4 -> v3 | 2x2 | none, `n_real=0` | 0.037953 | 0.057272 |
| v4 -> v3 | 3x3 | none, `n_real=0` | 0.096178 | 0.180215 |
| v3 -> v4 | 2x2 | none, `n_real=4` | 0.035368 | 0.064745 |
| v3 -> v4 | 3x3 | embed, `n_real=4` | 0.081931 | 0.120876 |

Relative encoding improves native fits but does not solve the synthetic/CST
domain shift. In three of four transfer settings, the MSE winner is the
no-position model. Full results for every mode are in
`logs/vf_rel_ablation_relabl_stable_v3/evaluations.csv`.

The filtered 40-row transfer table is
`plots/vf_research_suite_relabl_stable_v3/cross_domain_test_metrics.csv`.
It reports both test error definitions for every train-domain/test-domain,
grid, relative-encoding, and `n_real` combination:

\[
\mathrm{MSE}=\operatorname{mean}\left[(\hat T-T)^2\right],
\]

\[
\mathrm{beta2}=\operatorname{mean}\left[
\left(1+2(1-T)^2\right)(\hat T-T)^2
\right].
\]

All models in this controlled study were trained with beta2-weighted MSE.
Checkpoint selection used plain validation MSE. The held-out test set was then
reported with both plain MSE (`test_mse`) and weighted beta2
(`test_beta2`); beta2 is therefore not the only test metric.

These are seed-0 ablations, not uncertainty estimates. Differences such as
0.000902 versus 0.000966 should be treated as near ties until repeated across
seeds.

## Local figures

Run:

```bash
/opt/anaconda3/envs/powertx_mps/bin/python plot_vf_relative_ablation.py
```

Figures are written to
`plots/vf_relative_ablation_relabl_stable_v3/`:

- `native_test_metrics.png`
- `transfer_test_metrics.png`
- `training_curves_nreal0.png`
- `training_curves_nreal4.png`
- `random_samples/`: three identical seeded samples for each native
  dataset/grid/`n_real` case, comparing all five encodings
- `best_worst/`: best and worst per-sample MSE spectra for the best native mode
  in each dataset/grid/`n_real` case

All 80 stored best/worst indices were independently checked against
`argmin(per_sample_mse)` and `argmax(per_sample_mse)`, and their spectral MSEs
were recomputed from the saved truth and prediction arrays.

## Publication-oriented research suite

Run:

```bash
/opt/anaconda3/envs/powertx_mps/bin/python plot_vf_research_suite.py
```

The script writes PNG and vector PDF versions under
`plots/vf_research_suite_relabl_stable_v3/`, together with
`top_configuration_ranking.csv`.

Raw MSE values cannot be ranked directly across v3/v4 and 2x2/3x3 because the
benchmarks have different error scales. The suite therefore normalizes every
configuration to the matched `none, n_real=0` result in each of the four native
tasks and ranks configurations by the geometric mean of those four ratios.
The selected top five are:

| Rank | Relative encoding | `n_real` | Geometric-mean normalized MSE |
|---:|---|---:|---:|
| 1 | embed | 4 | 0.479628 |
| 2 | embed | 0 | 0.494123 |
| 3 | polar | 4 | 0.514176 |
| 4 | offset_dist | 4 | 0.515828 |
| 5 | offset | 4 | 0.542317 |

The v3 baseline figure uses the locally stored, directly comparable SiLU
seed-0 results: 0.008661 (flat) and 0.008124 (K=3) for 2x2; 0.061176
(flat) and 0.049949 (K=3) for 3x3. Against the K=3 SiLU baseline, the best VF
ablation lowers test MSE by 51.6% on v3 2x2 and 38.0% on v3 3x3.
No separately trained SiLU-v4 baseline is stored locally, so v4 comparisons use
the matched VF `none, n_real=0` reference rather than importing an incompatible
number.

The research suite contains:

- cross-benchmark normalized ablation ranking;
- v3 comparison against the stored flat and K=3 SiLU baselines;
- paired plain-MSE and beta2 test figures for the top five configurations on
  all four native benchmarks and all four cross-domain transfers;
- log-binned per-sample test-MSE histograms for v4 synthetic simulation data
  and v3 CST-generated data;
- raw and smoothed training-beta2 and validation-MSE curves for the top five;
- each top configuration's independently selected best and worst test spectrum
  for all four native benchmarks.
