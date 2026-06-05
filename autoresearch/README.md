# autoresearch (emulator)

Autonomous-experiment harness (after Karpathy's `autoresearch`) for the desilike
NN emulators. An agent iterates on `train.py` to minimize the test MSE on a
fixed evaluation set, while `prepare.py` stays read-only.

See `program.md` for the full agent protocol.

## Files

- `prepare.py` — fixed data loading, z-score standardization, and the
  `evaluate_test_mse` ground-truth metric. **Read-only.**
- `train.py` — ResNet-MLP regressor + cosine-warm-restart training loop.
  **This is the only file the agent edits.**
- `program.md` — the experiment protocol / lightweight skill.
- `experiments.md` — running log of what was tried and why.
- `results.tsv` — one row per eval checkpoint (gitignored, append-only).

## Running

Uses the repo's conda `emulator` env (torch 2.5.1, CUDA) — no `uv`.

```bash
cd ~/desilike-emulator/autoresearch
PY="$HOME/miniconda3/envs/emulator/bin/python"

# select the dataset (same {TRACER}_{train,test}.npz format as train_nn.py)
export EMULATOR_DATA_DIR="$HOME/scratch/bedcosmo/num_tracers/emulator/training_data/bao/base_omegak_w_wa/covar/v3"
export EMULATOR_TRACER=LRG2

$PY prepare.py        # verify data loads + print stats
$PY train.py          # run one experiment -> results.tsv
```

Pin a GPU with `CUDA_VISIBLE_DEVICES=0` (or `1`).

## Data

Consumes the same `{TRACER}_train.npz` / `{TRACER}_test.npz` arrays
(`x`, `y`, `param_names`, `target_names`) produced for the production
`train_nn.py`, so any existing emulator training set works unchanged.
