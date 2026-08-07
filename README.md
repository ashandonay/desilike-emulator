# Num-tracers neural emulators (BAO & ShapeFit)

This package builds PyTorch regressors that emulate **desilike** outputs: Fisher covariance elements for BAO / ShapeFit observables (and, for ShapeFit only, mean ShapeFit parameters). Training data are `.npz` grids of inputs `x` and targets `y`; scripts live next to per-analysis subpackages `bao/` and `shapefit/`.

## Requirements

- **`desilike`** (and its dependencies, including JAX where Fisher workers use it).
- **`SCRATCH`** environment variable for default data and MLflow paths (`get_default_save_path`, training logs). If it is unset, pass explicit `--save-path` to prep scripts and `--save-path` / absolute paths for `eval.py` as needed.

Default layout (when `SCRATCH` is set):

- Training data: `$SCRATCH/bedcosmo/num_tracers/emulator/{analysis}/training_data/{cosmo_model}/{quantity}/v{N}/`
- MLflow runs: `file:$SCRATCH/bedcosmo/num_tracers/emulator/{analysis}/mlruns`

Run scripts from this directory (`emulator/`) so imports such as `util` → `bao` / `shapefit` resolve correctly:

```bash
cd src/bedcosmo/num_tracers/emulator
```

---

## Analysis directories (`bao/`, `shapefit/`)

| Directory   | Role |
|------------|------|
| **`bao/`** | BAO forecast pipeline (Fourier + config-space frames): per-tracer `(N_tracers, cosmology)` → distance-error targets (`sigma_DH_over_rd`, `sigma_DM_over_rd`, `rho_DH_DM`; iso tracers `sigma_DV_over_rd`). See **`bao/README.md`** — that file, not this one, documents the current CLIs. |
| **`shapefit/`** | Full-shape (ShapeFit) forecast pipeline: per-tracer errors (`--quantity covar`: 4 `sigma_*` + 6 `rho_*` of `qiso, qap, f_sigmar, m`) and means (`--quantity mean`: `qiso, qap, f_sigmar, m`). See **`shapefit/README.md`**. |

Shared per-analysis files:

| File | Purpose |
|------|---------|
| **`model_config.yaml`** | Named blocks of NN and optimizer hyperparameters. The key you pass as `--nn-model` (or the default derived from `--cosmo-model`, falling back to `default`) selects one block. |
| **`model.py`** | PyTorch module registered in `util.ARCHITECTURE_REGISTRY` (currently **`resnet`**: SiLU residual MLP — `ResNetRegressor` for BAO, `base_regressor` for ShapeFit). |
| **`generate_covar_data.py`** | Per-tracer covariance/error training-data CLI (parallel Fisher workers, versioned `v{N}` `.npz` output). |
| **`generate_training_data.sh`** | All-tracer driver; resolves one shared `v{N}` up front. |
| **`regress_sigmas.py`** | Bit-exact dump/compare regression harness (run before/after any dependency change). |

ShapeFit-only:

| File | Purpose |
|------|---------|
| **`generate_mean_data.py`** | Samples cosmological priors, runs the ShapeFit extractor per tracer at its derived z_eff, saves mean targets `qiso`, `qap`, `f_sigmar`, `m`. Used for `--quantity mean` training and evaluation. |

---

## `model_config.yaml`

Top-level keys are arbitrary labels (e.g. `base`, `base_scaled`, `base_omegak_w_wa`). Training selects a block via:

1. **`--nn-model <key>`** if set, else  
2. **`--cosmo-model`** as the key, else  
3. If that key is missing, **`--nn-model` was not set**, and the file contains a **`default`** key, that block is used.

For **ShapeFit**, `shapefit/model_config.yaml` defines a single **`default`** block, so any `--cosmo-model` resolves to it unless you add per-model keys.

Typical fields (all consumed by `train.py` where applicable):

- **`architecture`**: must exist in `util.ARCHITECTURE_REGISTRY` for that analysis (e.g. `resnet`).
- **`hidden_dim`**, **`n_hidden`**, **`expand`**, **`dropout`**: passed into the regressor.
- **`lr`**, **`final_lr_frac`**, **`weight_decay`**, **`batch_size`**
- **`warmup_fraction`**, **`scheduler_type`** (`constant`, `cosine`, `linear`, `exponential`, `lambda`), **`lr_restarts`**, **`lr_gamma`** (for the lambda-style schedule)
- **`grad_clip`**: global norm clipping (default in code if omitted: `1.0`)

BAO’s YAML may use keys that do not match `cosmo-model` names (e.g. `base_scaled`); use **`--nn-model`** explicitly in that case.

---

## Training-data generation

The per-analysis CLIs are documented in their own READMEs (`bao/README.md`,
`shapefit/README.md`). Both write per-tracer **`{tracer}_train.npz`** /
**`{tracer}_test.npz`** (arrays **`x`**, **`y`**, **`param_names`**,
**`target_names`**) under
`{analysis}/training_data/{data_release}/{cosmo_model}/{quantity}/v{N}/`, and both
anchor the `N_tracers` box via `util.ntracers_range` (tracers.yaml low/high
factors × the data release's `passed` counts — never hardcode N).

```bash
# BAO (from bao/):     one shared v{N} for all 6 tracers
bao/generate_training_data.sh --space config --cosmo-model base --n-samples 10000

# ShapeFit (from shapefit/): errors and means
shapefit/generate_training_data.sh --quantity covar --cosmo-model base --n-samples 5000
shapefit/generate_training_data.sh --quantity mean  --cosmo-model base --n-samples 10000
```

---

## Top-level components

| File | Role |
|------|------|
| **`util.py`** | `build_model`, `get_default_save_path`, `get_pipeline` (loads the per-analysis ground-truth generators for `eval.py`), `save_dataset`, LHS sampling, tracer bins for **`--tracer-bin`**. |
| **`train.py`** | Loads YAML + `.npz` data, standardizes inputs/targets, trains with MLflow logging, saves checkpoints under the run’s artifacts. |
| **`eval.py`** | Loads a checkpoint, draws LHS parameters, compares NN to `get_pipeline` ground truth, writes diagnostic plots. |
| **`scale_data.py`** | Post-processes a directory of `.npz` files: multiplies `y` by user-defined factors of input variables; writes a sibling directory and **`scale_info.json`** (used at train/eval time to track scaling). |
| **`test_cov_scaling.py`** | Tests for the scale expression language. |
| **`notebooks/`** | Exploration (training, errors, shapefit extractor). |

---

## `scale_data.py`

Positional usage (not `argparse`):

```bash
python scale_data.py <data_dir> <expr1> [expr2 ...]
```

Each **expression** is an infix formula using `+ - * / ^ **`, parentheses, `exp(...)`, `log(...)`, and variables among: **`N_tracers`**, **`Om`**, **`Ok`**, **`w0`**, **`wa`**, **`hrdrag`**. Variables must appear in the dataset’s `param_names`.

Output directory: `<data_dir>_<suffix>_scaled` where `suffix` encodes the expressions. A **`scale_info.json`** records `scale_expressions` and `source_dir`. **`train.py`** copies `scale_expressions` into the checkpoint so **`eval.py`** can invert scaling when comparing to the true Fisher / extractor.

---

## `train.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--analysis` | `shapefit` | `shapefit` or `bao`. |
| `--quantity` | `mean` | `mean` or `covar` (must match data and available pipelines; BAO only supports `covar`). |
| `--cosmo-model` | `base` | `base`, `base_w`, `base_w_wa`, `base_omegak`, `base_omegak_w_wa` — selects subdirectory under `{analysis}/training_data/...` and default YAML key. |
| `--nn-model` | *(see above)* | YAML key in `{analysis}/model_config.yaml`. |
| `--data-dir` | `latest` | Absolute path to a data folder, or a folder name under the cosmo/quantity root (e.g. `v3`, or a scaled dir), or `latest` → highest `v{N}`. |
| `--epochs` | `10000` | |
| `--patience` | `0` | Early stopping on test loss (`0` = off). |
| `--seed` | `0` | |
| `--mlflow-exp` | `default` | MLflow experiment name. |
| `--run-name` | `None` | Optional run name. |
| `--log-normalize` | off | Symlog targets before z-score. |
| `--eval-atol`, `--eval-rtol` | `2e-3` | Passed to post-training eval. |
| `--tracer-bin` | `None` | One of `BGS`, `LRG1`, `LRG2`, `LRG3_ELG1`, `ELG2`, `QSO`, `Lya_QSO`: loads `{name}_train.npz` / `{name}_test.npz` if present and scopes eval priors / redshift. |

Training uses the first available CUDA device; evaluation inside `eval.py` uses **`cuda:1`** if CUDA is available, else CPU.

---

## `eval.py`

Exactly one of **`--run-id`**, **`--run-dir`**, or **`--model-path`** must identify a checkpoint (`model_best.pt` preferred, then `model.pt`).

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-id` | `None` | MLflow run ID; resolves artifacts under `file:$SCRATCH/bedcosmo/num_tracers/emulator/{analysis}/mlruns`. |
| `--run-dir` | `None` | Directory containing `checkpoints/model_best.pt` or `model.pt`. |
| `--model-path` | `None` | Direct path to `.pt`. |
| `--n-samples` | `200` | LHS comparison samples. |
| `--seed` | `42` | |
| `--hist-xlims` | `None` | JSON dict mapping target name to `[lo, hi]` for histograms. |
| `--rtol`, `--atol` | `2e-3` | `numpy.allclose` tolerances for outlier reporting. |
| `--log-scale` | off | Log/symlog axes on triangle plots. |
| `--analysis` | `shapefit` | Must match the trained model. |
| `--quantity` | `covar` | `covar` or `mean`. |
| `--save-path` | auto | Plot output directory. |
| `--tracer-bin` | `None` | Same choices as training; aligns priors and `zrange` / `z_eff` with DESI bin definitions. |

---

## Typical workflows

**1. BAO error emulator**

```bash
bao/generate_training_data.sh --space config --cosmo-model base --n-samples 10000
python train.py --analysis bao --quantity config --cosmo-model base --data-release dr1 --data-dir latest --tracer-bin LRG2
python eval.py --run-id <mlflow_run_id> --analysis bao --tracer-bin LRG2
```

**2. ShapeFit error emulator**

```bash
shapefit/generate_training_data.sh --quantity covar --cosmo-model base --n-samples 5000
python train.py --analysis shapefit --quantity covar --cosmo-model base --data-release dr1 --data-dir latest --tracer-bin LRG2
python eval.py --run-dir <path_to_run_artifacts> --analysis shapefit --quantity covar --tracer-bin LRG2
```

**3. ShapeFit mean parameters**

```bash
shapefit/generate_training_data.sh --quantity mean --cosmo-model base --n-samples 10000
python train.py --analysis shapefit --quantity mean --cosmo-model base --data-release dr1 --data-dir latest --tracer-bin LRG2
python eval.py --model-path /path/to/model.pt --analysis shapefit --quantity mean --tracer-bin LRG2
```

---

## `.npz` layout

- **`x`**: `(n_samples, n_params)` float32  
- **`y`**: `(n_samples, n_targets)` float32  
- **`param_names`**, **`target_names`**: string arrays  

After **`scale_data.py`**, the scaled directory also contains **`scale_info.json`**.
