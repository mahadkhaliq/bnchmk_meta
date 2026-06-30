# MLP Pipeline (extracted from the reproducing-benchmark notebook)

This is the MLP part of `reproducing_benchmarking_data_driven___.py` pulled out
of the exploratory Colab notebook and reorganised into a clean, runnable project.
The notebook reproduces Deng et al. 2021's metasurface surrogate (geometry →
absorption spectrum) on the **ADM** dataset.

It gives you **two MLP variants** that share all the data / training code:

| Variant | Script | Input | Idea |
|---|---|---|---|
| **Baseline (no neighbouring)** | `train_mlp_baseline.py` | flat geometry vector `(INPUT_DIM)` | one big MLP maps the whole supercell → spectrum |
| **Neighbourhood (with neighbouring)** | `train_mlp_neighborhood.py` | grid `(N, N, C)` | a *shared* MLP runs on each cell's wrap-around `K×K` neighbourhood, then the per-cell spectra are averaged |
| **Concept #1 / relaxed Lorentz** | `train_powertx_concept_one.py` | grid `(N, N, C)` | physics-inspired `U(cell) + ΣV(cell, neighbour)` latent model with a trainable neural spectrum decoder |

---

## File structure

```
mlp_pipeline/
├── models/                     # SHARED, dataset-agnostic
│   ├── mlp.py                  # MLP  (used by both variants)
│   └── scale_invariant.py      # ScaleInvariantMetasurface (the WITH-neighbouring model)
├── engine.py                   # train_one_epoch / evaluate / fit  (SHARED training loop)
│
├── data/
│   ├── normalize.py            # min-max scaling to [-1, 1] per column  (SHARED)
│   ├── datasets.py             # ArrayDataset (torch Dataset wrapper)   (SHARED)
│   ├── grid.py                 # ADM build_grid() + neighbourhood extraction
│   ├── loaders.py              # ADM CSV loaders: load_flat() / load_grid()
│   └── powertx.py              # power-transmission .npz loaders + grid mapping
│
│   # ---- Dataset 1: ADM (CSV) ----
├── config.py                   # ADM config (paths, dims, hyperparameters)
├── train_mlp_baseline.py       # ADM: WITHOUT neighbouring
├── train_mlp_neighborhood.py   # ADM: WITH neighbouring
├── evaluate.py                 # ADM: load checkpoints, report test MSE
│
│   # ---- Dataset 2: power-transmission (.npz, sizes 1x1/2x2/3x3) ----
├── config_powertx.py           # power-tx config (pick GRID = 1x1/2x2/3x3)
├── train_powertx_baseline.py   # power-tx: WITHOUT neighbouring
├── train_powertx_neighborhood.py # power-tx: WITH neighbouring
├── train_powertx_concept_one.py # power-tx: relaxed-Lorentz Concept #1
├── evaluate_powertx.py         # power-tx: load checkpoints, report test MSE
│
├── smoke_test.py               # synthetic-data wiring check (no data files needed)
└── requirements.txt
```

The two datasets are kept as **separate scripts** that share the model + engine
code. ADM uses `config.py` + `train_mlp_*.py`; the power-transmission `.npz`
files use `config_powertx.py` + `train_powertx_*.py`.

## Quick start

```bash
pip install -r requirements.txt

# 1. sanity-check the wiring (no data needed)
python smoke_test.py

# 2. point config.py at your CSVs, then train either variant
python train_mlp_baseline.py
python train_mlp_neighborhood.py

# 3. evaluate on the test set
python evaluate.py
```

Run the scripts **from inside `mlp_pipeline/`** (imports are rooted at the
project directory).

---

## The three things extracted from the notebook

### 1. How the data is processed

The dataset is 4 header-less CSVs: `*_X` = geometry features, `*_Y` = target
spectrum. The pipeline (`data/loaders.py`) is:

1. **Read** train/test geometry + spectrum CSVs.
2. **Normalise** the geometry to `[-1, 1]` per column (`data/normalize.py`):
   `x_norm = (x - (max+min)/2) / ((max-min)/2)`.
   The min/max are **fit on the training split only** and reused for the test
   split. The spectrum targets are left unscaled.
3. **Split** off 20% of train as validation (`random_state=0`, so it's fixed).
4. **Wrap** in `ArrayDataset` and hand to `DataLoader`
   (train shuffled, val/test not).

The **baseline** keeps the geometry flat `(n, INPUT_DIM)`. The **neighbourhood**
variant additionally reshapes it into a grid (next section).

### 2. The MLP model (`models/mlp.py`)

```
for each hidden layer:   Linear -> BatchNorm1d -> ReLU
output layer:            Linear            (no BN, no activation)
```

Widths come from a list, e.g. the ADM baseline `[14, 2000×10, 2001]`
(11 Linear layers). The **same** `MLP` class is reused as the shared per-cell
network `f_theta` inside the neighbourhood model.

### 3. The neighbouring of cells / grid (`data/grid.py` + `models/scale_invariant.py`)

This is the core of the WITH-neighbouring variant.

**a. Build the grid** — `build_grid()` reshapes the flat geometry into a
`(N, N, C)` grid of cells. For ADM the supercell is `2×2` (4 resonators) with
`C=5` channels per cell `(rx, ry, theta, h, p)`. The column→cell→channel mapping
is **dataset specific** and is the one block you rewrite for a new dataset.

**b. Periodic padding** — a supercell tiles the plane, so the grid is padded with
**wrap-around** indices (`src = (arange(N+2·pad) − pad) % N`): cell `(0,0)`'s left
neighbour is the last column, etc.

**c. Neighbourhood extraction** — for each of the `N·N` cells, take the `K×K`
window centred on it and flatten to `K·K·C`. With `K=3, C=5` that's a 45-vector
per cell. (`data/grid.py::build_neighborhoods` is a NumPy reference; the model
does the identical thing in torch.)

**d. Shared MLP + average** — `ScaleInvariantMetasurface.forward`:

```
grid (B, N, N, C)
  -> wrap-around pad           (B, N+2p, N+2p, C)
  -> K×K window per cell        (B, N*N, K*K*C)
  -> SHARED MLP f_theta         (B, N*N, n_freq)     # same weights on every cell
  -> mean over the N*N cells    (B, n_freq)
```

Sharing one `f_theta` across all cells and averaging makes the model independent
of how many cells there are — hence "scale-invariant". `K` is configurable
(`config.KERNEL`); the notebook trained both `K=3` and `K=5`.

---

## Using your own dataset

Most changes live in **`config.py`**:

1. **Paths** — set `TRAIN_X_PATH`, `TRAIN_Y_PATH`, `TEST_X_PATH`, `TEST_Y_PATH`
   to your 4 CSVs (header-less; rows = samples).
2. **Shapes** — set `INPUT_DIM` (geometry features) and `OUTPUT_DIM` (target
   length). `LAYERS` rebuilds automatically.

If you want the **baseline (no neighbouring)** only, that's all you need — it
treats the geometry as a flat vector and doesn't care about its internal layout.

For the **neighbourhood variant** you must also describe the grid:

3. In `config.py` set `GRID_N` (grid is `GRID_N × GRID_N`), `CHANNELS`
   (features per cell), and `KERNEL` (`3` or `5`).
4. In `data/grid.py::build_grid`, rewrite the assignment loop so each
   `grid[:, r, c, channel]` is filled from the correct raw column of *your*
   geometry. This is the only dataset-specific code in the neighbouring path;
   everything after it (padding, windowing, the model) is generic.

> Note: the ADM `build_grid` assumes `GRID_N*GRID_N` resonators packed as
> `rx,ry` pairs (cols `2+2i`, `3+2i`), `theta` (col `10+i`), plus a shared
> height/period. Your dataset's columns will differ — that loop is where you map
> them.

After editing, re-run `python smoke_test.py` to confirm the shapes line up.

---

## Second dataset: power-transmission (`.npz`)

A second dataset lives in `/Users/mkfqm/malof_lab/power_tx_data/` as three
`.npz` files — CST-simulated metasurfaces at three super-cell sizes:

| File | Samples | Params (P) | Grid | Freq |
|---|---:|---:|:---:|---:|
| `dataset_1x1.npz` | 3006 | 4 | 1×1 | 2001 |
| `dataset_2x2.npz` | 706 | 16 | 2×2 | 2001 |
| `dataset_3x3.npz` | 484 | 36 | 3×3 | 2003 |

- Each cell has **4 params** `(d, g, l, w)`; target is `T_clean` = |S21|² power
  transmission, already cleaned and in `[0, 1]`.
- `params` columns are grouped **by parameter then row-major cell**, so the grid
  mapping (`data/powertx.py::build_grid_powertx`) is
  `grid[:, r, c, ch] = params[:, ch*N*N + r*N + c]` with channel order `d,g,l,w`.
- These files have **no separate test set**, so the loaders carve out a 15%
  held-out test split (`TEST_SPLIT`), then a 20% validation split.

**Run it:**
```bash
# edit GRID = "1x1" | "2x2" | "3x3" in config_powertx.py, then:
python train_powertx_baseline.py        # no neighbouring
python train_powertx_neighborhood.py    # with neighbouring
python train_powertx_concept_one.py     # Concept #1: additive interactions + neural decoder
python evaluate_powertx.py
```

**Two caveats:**
- These datasets are small (hundreds of samples). The default width matches the
  ADM paper (~40M params) and **will overfit** — shrink `HIDDEN` / `N_HIDDEN` in
  `config_powertx.py` (e.g. 256 / 4) for real results.
- At **1×1** a cell has no neighbours, so the neighbourhood model degenerates to
  the baseline with a replicated input — use the baseline there.

**Scale-invariance experiment:** because the per-cell channel layout `(d,g,l,w)`
is identical across 1×1/2×2/3×3, the neighbourhood model trained on one size can
be evaluated on another (the baseline cannot — its input width is tied to the
grid size). That cross-size transfer is the headline experiment this dataset
enables.

**Concept #1 / relaxed Lorentz experiment:** `models/concept_one.py` implements
the slide-deck architecture where the fixed Lorentzian physics block is replaced
by a trainable neural decoder. For each cell, the model computes a base latent
`U(g_i)`, adds pairwise neighbour perturbations `Σ_j V(g_i, g_j, dx, dy)`, decodes
that latent to a per-cell spectrum with `D`, then averages over cells. At `1×1`
the interaction term is skipped, so the same model behaves like the intended
single-cell pretraining path `U(g) -> D(z) -> spectrum`.

## Notes / provenance

- Extracted from `reproducing_benchmark/initial_reproducing_notebook/reproducing_benchmarking_data_driven___.py`.
- The notebook also contains two **Transformer** variants and a lot of
  exploratory/plotting cells; those are intentionally **not** included here — this
  project is the MLP path only, as requested.
- Default hyperparameters reproduce the paper's baseline
  (`lr=1e-4`, `weight_decay=1e-4`, `batch=1024`, `500` epochs,
  `ReduceLROnPlateau`). Reference test-MSE targets on ADM: baseline ≈ `1.20e-3`,
  greedy-searched MLP ≈ `1.23e-3`.
```
