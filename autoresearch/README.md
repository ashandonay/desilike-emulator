# autoresearch (emulator)

Architecture-search sandbox (after Karpathy's `autoresearch`) for the desilike
NN σ-emulators. An agent searches over architectures here on its own branch, then
**promotes the winner into the production `bao/` pipeline** (`bao/model.py` +
`bao/model_config.yaml`) and merges to `main`. `prepare.py` stays read-only.

See `program.md` for the full workflow (search → promote → train → eval → merge).

## Files

- `model.py` — the architecture under search (`ResNetRegressor`). Single place to
  edit model *structure*; kept identical to `bao/model.py` so promotion is a copy.
- `screen.py` — fast pre-filter (compressed schedule, cheap ranking). **Mis-ranks
  near the noise floor — prune only, never decide with it.**
- `validate.py` — AUTHORITATIVE: faithful full production schedule, seed-swept.
- `prepare.py` — fixed data loading, z-score standardization, `evaluate_test_mse`
  metric. **Read-only.**
- `program.md` — the search/promotion protocol.
- `experiments.md` — running log of what was tried and why.
- `dev/` — gitignored scratch for `screen.py`/`validate.py` `.tsv`/`.log` outputs.

Training is **not** done here — the real model is trained by the production
`train_nn.py` after promotion. (The old standalone `train.py` was removed; it
duplicated `train_nn.py`.)

## Running a search

Uses the repo's conda `emulator` env — no `uv`.

```bash
cd ~/desilike-emulator/autoresearch
PY="$HOME/miniconda3/envs/emulator/bin/python"

export EMULATOR_DATA_DIR="$HOME/scratch/bedcosmo/num_tracers/emulator/training_data/bao/dr1/base/config/v2"
export EMULATOR_TRACER=LRG2

$PY prepare.py                                   # verify data loads
CUDA_VISIBLE_DEVICES=0 $PY screen.py   --out dev/screen.tsv   16,4,4 24,6,4 48,6,4   # fast prune
CUDA_VISIBLE_DEVICES=0 $PY validate.py --out dev/validate.tsv 24,6,4,42 24,6,4,123    # faithful, seed-swept
```

## Data

Consumes the same `{TRACER}_train.npz` / `{TRACER}_test.npz` arrays
(`x`, `y`, `param_names`, `target_names`) produced for the production
`train_nn.py`, so any existing emulator training set works unchanged.
