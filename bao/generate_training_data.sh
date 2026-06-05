#!/usr/bin/env bash
# Generate emulator training data (σ-triplet) for all tracer bins, one per
# invocation of generate_emulator_data.py, all written into the SAME version
# folder so they form one coherent dataset. Output tree:
#   training_data/bao/{dataset}/{cosmo_model}/{space}/v{N}/{tracer}_{train,test}.npz
#
# Usage (from anywhere):
#   bao/generate_training_data.sh [options]
#
# Options (all have defaults):
#   --space        config | fourier        (default: config)
#   --dataset      dr1 | dr2                (default: dr1) anchors N_tracers box
#   --cosmo-model  base | base_w | base_w_wa | base_omegak | base_omegak_w_wa
#                                           (default: base_w_wa)
#   --n-samples    LHS draws per tracer     (default: 5000)
#   --workers      spawn-Pool processes     (default: 32; each pinned to 1 BLAS
#                                            thread, so scale toward ncores)
#   --version      shared v{N} folder       (default: auto = next free v{N}, no
#                                            overwrite; pin a number to target a
#                                            specific/existing version)
#   --tracers      space-separated subset   (default: all 6 DR1 bins)
#   --save-path    override save root        (default: $SCRATCH default path)
#   --             everything after is passed through to generate_emulator_data.py
#
# Examples:
#   bao/generate_training_data.sh --cosmo-model base --n-samples 64 --workers 8 --version 99
#   bao/generate_training_data.sh --space fourier --tracers "LRG2 QSO"
#   bao/generate_training_data.sh --dataset dr2 --cosmo-model base --n-samples 10000
set -euo pipefail

# --- env (emulator conda env; cwd must be the bao/ dir) ---------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYBIN="${PYBIN:-$HOME/miniconda3/envs/emulator/bin/python}"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/emulator/lib:${LD_LIBRARY_PATH:-}"

# --- defaults ---------------------------------------------------------------
SPACE="config"
DATASET="dr1"
COSMO_MODEL="base_w_wa"
N_SAMPLES=5000
WORKERS=32          # workers are pinned to 1 BLAS thread each; scale toward ncores
VERSION=""          # empty => auto-resolve the next free v{N} once for the batch
TRACERS="BGS LRG1 LRG2 LRG3_ELG1 ELG2 QSO"
SAVE_PATH=""
EXTRA=()

# --- arg parse --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --space)       SPACE="$2"; shift 2 ;;
    --dataset)     DATASET="$2"; shift 2 ;;
    --cosmo-model) COSMO_MODEL="$2"; shift 2 ;;
    --n-samples)   N_SAMPLES="$2"; shift 2 ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --version)     VERSION="$2"; shift 2 ;;
    --tracers)     TRACERS="$2"; shift 2 ;;
    --save-path)   SAVE_PATH="$2"; shift 2 ;;
    --)            shift; EXTRA=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

save_args=()
[[ -n "$SAVE_PATH" ]] && save_args=(--save-path "$SAVE_PATH")

# Resolve the version ONCE up front so all tracers land in the same folder.
# Default (empty) = next free v{N} via util._next_version (no overwrite). All
# tracers share one v{N} dir (distinguished by the {tracer}_ file prefix), so we
# must NOT let each per-tracer save_dataset() call auto-increment on its own.
if [[ -z "$VERSION" ]]; then
  VERSION="$("$PYBIN" - "$SPACE" "$COSMO_MODEL" "$DATASET" "$SAVE_PATH" <<'PY'
import os, sys
sys.path.insert(0, os.path.dirname(os.getcwd()))   # repo root (parent of bao/)
from util import get_default_save_path, _next_version
space, cosmo_model, dataset, save_path = sys.argv[1:5]
root = save_path or get_default_save_path(
    analysis="bao", quantity=space, cosmo_model=cosmo_model, dataset=dataset)
print(_next_version(root))
PY
)"
  echo "Auto-resolved version: v$VERSION (next free, no overwrite)"
fi

# Fail loudly rather than scatter the batch across per-tracer auto-increments.
if ! [[ "$VERSION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not resolve a numeric version (got '$VERSION'). " \
       "Set \$SCRATCH, or pass --save-path / --version explicitly." >&2
  exit 1
fi

echo "=== generate_emulator_data: space=$SPACE dataset=$DATASET model=$COSMO_MODEL n=$N_SAMPLES workers=$WORKERS version=v$VERSION ==="
echo "Tracers: $TRACERS"
echo

for T in $TRACERS; do
  echo "----------------------------------------------------------------------"
  echo "  $T   ($(date '+%H:%M:%S'))"
  echo "----------------------------------------------------------------------"
  "$PYBIN" generate_emulator_data.py \
    --space "$SPACE" --dataset "$DATASET" --tracer-bin "$T" --cosmo-model "$COSMO_MODEL" \
    --n-samples "$N_SAMPLES" --workers "$WORKERS" --version "$VERSION" \
    "${save_args[@]}" "${EXTRA[@]}"
  echo
done

echo "=== done: all tracers written to .../bao/$DATASET/$COSMO_MODEL/$SPACE/v$VERSION/ ==="
