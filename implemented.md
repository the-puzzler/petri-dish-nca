# Implemented Changes

This repo no longer matches the original Sakana AI `petri-dish-nca` mechanics exactly. It has been partially rewritten to implement the proposal described in [new.md](/Users/matteo/Documents/petri-dish-nca/new.md).

## High-Level Change

Original PD-NCA:
- agents competed via attack/defense channel cosine similarity
- the environment acted as a baseline opponent
- the model directly emitted state updates into the grid

Current version:
- each cell has a fixed read-only environment embedding
- each NCA is an explicit encoder/decoder agent
- competition is based on cross-decoding reconstruction error
- the grid stores alive channels plus per-agent `a`, `d`, and `h` state

## What Was Changed

### 1. Environment Tensor

Implemented in [src/world.py](/Users/matteo/Documents/petri-dish-nca/src/world.py).

Changed from:
- no data-grounded per-cell environment

Changed to:
- fixed environment tensor `env_pool`
- default sine substrate with `x -> sin(x)` embedded using trig features
- generic table-backed environment path

Supported environment modes:
- `env_kind: "sine"`
- `env_kind: "table"`

Supported data-to-grid assignments:
- `x_axis`
- `tile`
- `random`

For `table` mode, config can provide either:
- `data_env_vectors`
- or `data_x_values` and `data_y_values`

### 2. Agent Architecture

Implemented in [src/model.py](/Users/matteo/Documents/petri-dish-nca/src/model.py).

Changed from:
- one grouped-convolution model producing direct attack/defense-style updates

Changed to:
- one `AgentAutoencoder` per NCA
- encoder path reads:
  - local environment embedding
  - that agent's own hidden channels
- decoder reconstructs the local environment from the latent code
- hidden update head writes back only to that agent's own hidden channels

Important state-layout change:
- the grid now stores `alive_dim + n_ncas * (latent_dim + env_dim + cell_hidden_dim)`
- each agent has a persistent local `a`, `d`, and `h`

### 3. Competition Formula

Implemented in [src/model.py](/Users/matteo/Documents/petri-dish-nca/src/model.py).

Changed from:
- attack/defense cosine similarity

Changed to:
- self-reconstruction against the environment
- cross-decoding between agents

Operationally:
- each agent produces a latent code and self-reconstruction
- every decoder is evaluated on every agent's latent
- agent `i` gains strength when its decoder reconstructs the environment from agent `j`'s latent better than `j` can reconstruct from `i`'s latent

The environment still occupies the index-0 alive channel and acts as the default territorial baseline.

### 4. Training / Logging

Implemented in [src/train.py](/Users/matteo/Documents/petri-dish-nca/src/train.py).

Added metrics:
- `growth`
- `reconstruction_mse`
- `cross_mse_matrix`
- `territory_by_region`
- `mse_by_region`
- `region_winners`

Console logs now show:
- per-agent MSE
- diagonal self-cross reconstruction values
- x-region dominant agent ids

### 5. Config Changes

Implemented in [src/config.py](/Users/matteo/Documents/petri-dish-nca/src/config.py).

Removed dependence on:
- `cell_state_dim` split into attack/defense
- original grouped-update assumptions

Added config fields for:
- environment type and assignment
- latent dimension and decoder hidden size
- region bin analysis
- table-backed sample input

## What Is Implemented From `new.md`

Implemented:
- fixed read-only environment tensor
- sine experiment substrate
- explicit encoder/decoder competition
- latent normalization
- per-agent `a`, `d`, and `h` state
- neighborhood-aware hidden aggregation via spatial conv
- cross-decoding competition
- generic sample-to-grid assignment path
- region specialization diagnostics

## What Is Not Fully Implemented Yet

Not yet implemented:
- image dataset path such as MNIST
- frozen external feature encoders for richer datasets
- entropy / open-endedness metrics from the note
- optimized batched cross-decoder evaluation
- richer visualization of cross-decoding matrices and regional boundaries

## Practical Notes

- The current recommended smoke test is:

```bash
uv run python src/train.py --config configs/tiny-config.json --epochs 2 --device cpu
```

- For more readable short-run diagnostics, set `log_every` to `1` in `configs/tiny-config.json`.

- The implementation is intended as a first working pass of the `new.md` design, not a strict reproduction of the original PD-NCA codepath.
