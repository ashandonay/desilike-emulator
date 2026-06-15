# autoresearch — desilike emulator

Have the LLM search for the best NN architecture for a given emulator **config**
(BAO/shapefit σ-emulators), on its own git branch, then **promote the winning
architecture into the production `bao/` pipeline** and merge to `main`.

The sandbox in `autoresearch/` is yours to edit freely during a search. The
durable output is the change to `bao/model.py` / `bao/model_config.yaml` (and
`util.py` if a new architecture class is added), which gets merged to `main`.

## Runtime

Conda `emulator` env (torch + CUDA); no `uv`. Use only packages in that env.

```bash
cd ~/desilike-emulator/autoresearch
PY="$HOME/miniconda3/envs/emulator/bin/python"
export EMULATOR_DATA_DIR=$HOME/scratch/bedcosmo/num_tracers/emulator/training_data/bao/dr1/base/config/v2
export EMULATOR_TRACER=LRG2
$PY prepare.py    # verify the data loads + print stats
```

## The sandbox (edit freely during a search)

- **`model.py`** — the architecture under search (`ResNetRegressor`). This is the
  single place to change model *structure*. Kept identical in class name/signature
  to `bao/model.py` so promotion is a clean copy.
- **`screen.py`** — fast pre-filter: trains each `dim,blocks,expand` config on a
  short compressed schedule, ranks cheaply. **Caveat: it mis-ranks** when the
  metric has a late LR-restart deep minimum (see `experiments.md`) — use it only
  to prune obviously-bad capacity, never for the final call.
- **`validate.py`** — AUTHORITATIVE: faithful full production schedule (mirrors
  `train_nn.py`: 10000 epochs, `scheduler_type=lambda`, 10 restarts, gamma 0.85),
  seed-swept. All decisions rest on this.
- **`prepare.py`** — READ-ONLY: data loading, z-score standardization, and
  `evaluate_test_mse` (the fixed metric). Do not modify.
- **`dev/`** — gitignored scratch for `screen.py`/`validate.py` `.tsv`/`.log` outputs.

Width/depth/expand sweeps need no code edits — pass `dim,blocks,expand` tokens.
For a *structural* change (new block type, normalization, etc.), edit `model.py`.

## Workflow

1. **Branch**: `git checkout -b autoresearch/<release>-<cosmo_model>-<space>-<tracer>`
   (e.g. `autoresearch/dr1-base-config-lrg2`).
2. **Point at the data**: set `EMULATOR_DATA_DIR` / `EMULATOR_TRACER` for the
   target config's `{TRACER}_{train,test}.npz`.
3. **Search** (log reasoning + results to `experiments.md` as you go):
   - prune with `screen.py` (fast), then confirm with `validate.py` (faithful,
     ≥3 seeds). Edit `model.py` for structural ideas.
   - Apply the simplicity criterion: prefer the smallest model at the noise floor.
4. **Promote the winner** (see next section) — indexed by config.
5. **Train for real** with the production `train_nn.py` (saves a real checkpoint).
6. **Evaluate** with `eval_nn.py` on that checkpoint.
7. **Merge the branch to `main`**.

## Promotion (the point of the whole exercise)

The architecture lives in **two** places, indexed by the config
`(release, cosmo_model, space)`:

- **Hyperparameters** (`hidden_dim`/`n_hidden`/`expand`/`dropout` + LR schedule)
  → **always** add/update a `bao/model_config.yaml` entry, keyed
  `<release>_<cosmo_model>_<space>` (e.g. `dr1_base_config`). Include
  `architecture: <name>` (default `resnet`).
- **The architecture class** → **only if structurally new**: copy the winning
  `model.py` definition into `bao/model.py` as a new class, register it in
  `util.py:ARCHITECTURE_REGISTRY["bao"]` under a name, and point the YAML entry's
  `architecture:` at it. For a plain width/depth/expand winner, `bao/model.py`
  does NOT change — only the YAML entry.

Promotion is a **deliberate, documented edit you make** — never have `validate.py`
auto-overwrite `bao/` source.

## Train + eval the promoted model (production path)

```bash
cd ~/desilike-emulator
PY="$HOME/miniconda3/envs/emulator/bin/python"
export SCRATCH=$HOME/scratch
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH

# Train (uses bao/model.py + the model_config.yaml key); saves model_best.pt
# under the MLflow run artifacts, then auto-runs eval.
$PY train_nn.py --analysis bao --dataset dr1 --cosmo-model base --quantity config \
    --data-dir v2 --tracer-bin LRG2 --nn-model dr1_base_config --epochs 10000

# Eval standalone on a saved checkpoint (live config-space ground truth):
$PY eval_nn.py --model-path <run>/artifacts/checkpoints/model_best.pt \
    --analysis bao --quantity config --tracer-bin LRG2 --save-path <out>
```

`train_nn.py` standardizes identically to `prepare.py` and saves the scaler +
arch metadata into the checkpoint; `eval_nn.py` rebuilds the model from that
metadata via `build_model` and scores it against the live σ generator
(`bao/config_space.XiSigmaGenerator`).

## The goal & search guidance

- **Minimize the standardized test MSE** (`evaluate_test_mse`), then apply the
  **simplicity criterion**: smallest architecture at the noise floor wins.
- **Trust only `validate.py`** for decisions. Short screens rank capacity-to-floor
  but miss the late deep minimum from LR warm restarts (best epoch ~9000). Seed-
  sweep before recording a winner.

## Logging — `experiments.md`

Append a short entry per experiment (what you tried, why, and the result). Keep
the running tables of `validate.py` seed sweeps and the final promotion decision.

## What you CANNOT do

- Modify `prepare.py` or its `evaluate_test_mse` metric (read-only).
- Install new packages — use only the conda `emulator` env.
- Auto-mutate `bao/` from a script — promotion is a reviewed, hand-made edit.
