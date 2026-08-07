# Power-transmission dataset — version history & comparison

Tracks how the power-transmission `.npz` datasets changed across versions, so
results stay interpretable. Our loader ([data/powertx.py](data/powertx.py)) is
**schema-aware** and auto-detects the layout from the npz keys; sizes are
registered in [config_powertx.py](config_powertx.py) `_SPECS`.

Last updated: 2026-07-08.

---

## A. What changed across versions

| aspect | **v1** (original) | **v2** (integrated) | **v3** (preprocessed) |
|---|---|---|---|
| location | `power_tx_data/dataset_*.npz` | `power_tx_data/version_2/dataset_2x2_integrated.npz` | `power_tx_data/version_3/preprocessed_*.npz` |
| geometry key | `params` | `params` | **`geom`** (+ `atoms` `(B,N,4)`) |
| target key | `T_clean` | `T_clean` | **`T`** |
| feature order | `d, g, l, w` | `d, g, l, w` | **`d, l, w, g`** |
| flatten grouping | **by-parameter** then cell | **by-parameter** then cell | **by-atom** then feature |
| freq band | 1–30 GHz (2001) | 1–30 GHz (2001) | **12–26 GHz (2001)** |
| 3×3 output dim | 2003 | — (2×2 only) | 2001 (common grid) |
| split metadata | `source_cst`, `source_run` | `dropped_constants` | **`batch`** (leak-free group split), `coords`, `native_band` |
| extra keys | — | — | `atoms`, `feat_names`, `atoms_per_cell` |

**Two order changes in v3** (both handled): the channel order (`d,l,w,g` vs
`d,g,l,w`) *and* the grouping (by-atom vs by-parameter). Harmless for the flat
MLP (order-agnostic); the grid/neighbourhood path reshapes v3's already-by-atom
`geom` directly instead of the legacy by-parameter mapping.

## B. Sample counts (per super-cell size)

| size | v1 | v2 | v3 |
|---|--:|--:|--:|
| 1×1 | 3,006 | — | 2,000 |
| 2×2 | 706 | 21,625 | 6,497 |
| 3×3 | 484 | — | 397 |

## C. SiLU/Sigmoid + beta2 (512×4) results — NOT directly comparable

| dataset | test MSE | split | epochs | test n | notes |
|---|--:|---|--:|--:|---|
| v1 2×2 | 0.033493 | random 68/17/15 | 500 | 106 | flat model |
| v1 2×2 | 0.019482 | random 68/17/15 | 500 | 106 | K=3 neighbourhood |
| v2 2×2 | 0.006866 | random 68/17/15 | 500 | 3,244 | flat model |
| v2 2×2 | 0.006277 | random 68/17/15 | 500 | 3,244 | K=3 neighbourhood |
| v3 1×1 | 0.000323 | random 68/17/15 | 500 | 300 | flat model |
| v3 1×1 | 0.000265 | random 68/17/15 | 500 | 300 | K=3 repeated-cell control |
| v3 2×2 | 0.008661 | random 68/17/15 | 500 | 975 | flat model |
| v3 2×2 | 0.008124 | random 68/17/15 | 500 | 975 | K=3 neighbourhood |
| v3 3×3 | 0.061176 | random 68/17/15 | 500 | 60 | flat model |
| v3 3×3 | 0.049949 | random 68/17/15 | 500 | 60 | K=3 neighbourhood |

> ⚠️ These numbers are **not** apples-to-apples: v3 has a different frequency
> band (12–26 vs 1–30 GHz) and ~9× more 2×2 data than v1. Differences across
> versions therefore reflect the data and target band, not purely the model.

## Known caveats to control for

- **Batch overlap (v3):** the random split puts samples from the same source
  `batch` in train, validation, and test. A batch-disjoint three-way split is
  impossible across every scale because 1×1 has two batches and 3×3 has one.
  Treat these results as interpolation benchmarks, not unseen-batch estimates.
- **Band mismatch:** never compare MSE across v1/v2 (1–30 GHz) and v3 (12–26 GHz).
- **For a fair cross-version comparison**, hold fixed: same model/config (e.g.
  500 ep), same split strategy, and report per-version separately.

## Target consistency note

The stored `T` array is the authoritative training target. For 1×1 it agrees
with `clip(|S21|², 0, 1)` to float precision. For 2×2 and 3×3 it remains very
close on average but is not pointwise identical after preprocessing:

| scale | mean absolute difference | maximum difference |
|---|---:|---:|
| 1×1 | 4.13e-8 | 2.98e-7 |
| 2×2 | 6.34e-5 | 0.520 |
| 3×3 | 3.09e-4 | 0.250 |

The isolated maxima likely come from resampling/clipping, but this cannot be
confirmed without the original preprocessing source. Do not recompute the
target from the stored complex `S21` unless that pipeline is verified.

## How the code selects a version

Set `POWERTX_GRID` in the environment (default: `2x2v3`):
`"1x1" | "2x2" | "3x3"` (legacy) or
`"1x1v3" | "2x2v3" | "3x3v3"` (v3).
The loader detects the schema automatically — no other code changes needed.
