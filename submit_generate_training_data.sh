#!/usr/bin/env bash
# SLURM submission script to generate emulator training data on NERSC
# Perlmutter, submitting ONE JOB PER TRACER so the bins run in parallel.
#
# Per-tracer SLURM counterpart of {bao,shapefit}/generate_training_data.sh.
# Both analyses are supported via --analysis; the generator and its
# quantity flag are selected from it:
#
#   --analysis bao       -> bao/generate_covar_data.py       --space {config,fourier}
#   --analysis shapefit  -> shapefit/generate_covar_data.py  (--quantity covar)
#                        -> shapefit/generate_mean_data.py   (--quantity mean)
#
# Output tree (one version folder shared by every tracer in a batch):
#   {analysis}/training_data/{data_release}/{cosmo_model}/{quantity}/v{N}/{tracer}_{train,test}.npz
#
# The shared v{N} is resolved ONCE up front (on the login node) and passed
# explicitly to every job, so the parallel jobs don't each auto-increment into
# separate version folders.
#
# ⚠ shapefit on NERSC: the pipeline is regression-frozen against desilike
# 4cfd6bec / cosmoprimo 1b100803. If the Perlmutter env is still on the older
# pins, labels generated here will differ SILENTLY from the local ones. Run
# shapefit/regress_sigmas.py against golden_4cfd6bec.npz there before trusting
# any shapefit dataset produced on Perlmutter.
#
# Usage:
#   bash submit_generate_training_data.sh
#   bash submit_generate_training_data.sh --space fourier --cosmo-model base
#   bash submit_generate_training_data.sh --analysis shapefit --quantity covar
#   bash submit_generate_training_data.sh --analysis shapefit --quantity mean --cosmo-model base
#   bash submit_generate_training_data.sh --tracers "LRG2 QSO" --queue debug --time 00:30:00
#
# Options (SLURM flags consumed here):
#   --workers N        parallel processes per job (default: 128; cpus-per-task)
#   --time HH:MM:SS    walltime per job          (default: 02:00:00)
#   --queue Q          SLURM queue               (default: regular)
#   --nodes N          nodes per job             (default: 1)
#
# Options consumed here and forwarded to the generator:
#   --analysis         bao | shapefit            (default: bao)
#   --space            config | fourier          (bao only; default: config)
#   --quantity         covar | mean              (shapefit only; default: covar)
#   --data-release          dr1 | dr2                 (default: dr1)
#   --cosmo-model      MODEL                     (default: per analysis, below)
#   --n-samples N      LHS draws per tracer      (default: 5000)
#   --version N        shared v{N} folder        (default: auto = next free v{N})
#   --tracers "A B C"  space-separated subset    (default: all 6 DR1 bins)
#   --save-path PATH   override save root        (default: $SCRATCH default path)
#   any other flag is passed straight through to the generator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Interpreter for the login-node version resolution below (the jobs themselves
# activate the conda env). util.py uses `X | None` annotations, so a system
# python3 older than 3.10 fails to import it — prefer the emulator env.
if [[ -z "${PYBIN:-}" ]]; then
    for _cand in "$HOME/.conda/envs/emulator/bin/python" \
                 "$HOME/miniconda3/envs/emulator/bin/python" \
                 python3; do
        if command -v "$_cand" &>/dev/null; then PYBIN="$_cand"; break; fi
    done
fi

# ── Defaults ──
# 128 = physical cores on a Perlmutter CPU node, 1 worker/core. The real
# requirement for high worker counts is pinning XLA to 1 thread/worker (see batch
# script) — without it the jax workers oversubscribe threads and deadlock. With
# that pin, 128 works fine; expect a few-minute warmup (each worker imports the
# ~/.conda env + reads the DESI bundle off $HOME, a slow metadata storm) BEFORE
# the first progress line, then it bursts. Don't mistake the quiet warmup for a
# hang. Lower this only if $HOME contention makes warmup unacceptably long.
WORKERS=128
TIME="02:00:00"
QUEUE="regular"
NODES=1
ANALYSIS="bao"
SPACE=""            # bao only;      resolved to 'config' below if unset
QUANTITY=""         # shapefit only; resolved to 'covar'  below if unset
DATA_RELEASE="dr1"
COSMO_MODEL=""      # resolved per analysis below
N_SAMPLES=5000
VERSION=""
TRACERS="BGS LRG1 LRG2 LRG3_ELG1 ELG2 QSO"
SAVE_PATH=""

# ── Parse flags ──
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)      WORKERS="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --queue)        QUEUE="$2"; shift 2 ;;
        --nodes)        NODES="$2"; shift 2 ;;
        --analysis)     ANALYSIS="$2"; shift 2 ;;
        --space)        SPACE="$2"; shift 2 ;;
        --quantity)     QUANTITY="$2"; shift 2 ;;
        --data-release)      DATA_RELEASE="$2"; shift 2 ;;
        --cosmo-model)  COSMO_MODEL="$2"; shift 2 ;;
        --n-samples)    N_SAMPLES="$2"; shift 2 ;;
        --version)      VERSION="$2"; shift 2 ;;
        --tracers)      TRACERS="$2"; shift 2 ;;
        --save-path)    SAVE_PATH="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Resolve analysis -> generator, quantity, and the generator's own flag ──
# bao names its quantity with --space {config,fourier}; shapefit splits covar
# and mean across two scripts and takes no quantity flag at all (the script IS
# the quantity). QUANTITY is the path segment in both cases, so the version
# resolution and output tree below stay analysis-agnostic.
case "$ANALYSIS" in
    bao)
        SPACE="${SPACE:-config}"
        QUANTITY="$SPACE"
        GEN_SCRIPT="$SCRIPT_DIR/bao/generate_covar_data.py"
        GEN_QUANTITY_FLAG="--space $SPACE"
        COSMO_MODEL="${COSMO_MODEL:-base_omegak_w_wa}"
        ;;
    shapefit)
        QUANTITY="${QUANTITY:-covar}"
        COSMO_MODEL="${COSMO_MODEL:-base}"
        GEN_QUANTITY_FLAG=""
        case "$QUANTITY" in
            covar) GEN_SCRIPT="$SCRIPT_DIR/shapefit/generate_covar_data.py" ;;
            mean)  GEN_SCRIPT="$SCRIPT_DIR/shapefit/generate_mean_data.py" ;;
            *) echo "ERROR: --quantity must be covar or mean for shapefit (got '$QUANTITY')" >&2; exit 2 ;;
        esac
        if [[ -n "$SPACE" ]]; then
            echo "ERROR: --space is bao-only; shapefit selects covar/mean via --quantity." >&2
            exit 2
        fi
        ;;
    *)
        echo "ERROR: --analysis must be bao or shapefit (got '$ANALYSIS')" >&2
        exit 2
        ;;
esac

# Logs live under the analysis subtree, matching training_data/ and models/.
# $SCRATCH-rooted (as in submit_train.sh) rather than a hardcoded /pscratch
# path, so the dispatch is exercisable off-NERSC; the fallback is Perlmutter's.
EMU_SCRATCH="${SCRATCH:-/pscratch/sd/a/ashandon}"
LOG_DIR="$EMU_SCRATCH/bedcosmo/num_tracers/emulator/$ANALYSIS/logs"

if [[ ! -f "$GEN_SCRIPT" ]]; then
    echo "Error: $GEN_SCRIPT not found"
    exit 1
fi

read -r -a TRACER_NAMES <<< "$TRACERS"
if [[ ${#TRACER_NAMES[@]} -eq 0 ]]; then
    echo "Error: no tracers given"
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Resolve the shared version ONCE so every per-tracer job lands in the same
#    v{N} folder. Default (empty) = next free v{N} via util._next_version. ──
if [[ -z "$VERSION" ]]; then
    VERSION="$("$PYBIN" - "$SCRIPT_DIR" "$ANALYSIS" "$QUANTITY" "$COSMO_MODEL" "$DATA_RELEASE" "$SAVE_PATH" <<'PY'
import os, sys
script_dir, analysis, quantity, cosmo_model, data_release, save_path = sys.argv[1:7]
sys.path.insert(0, os.path.abspath(script_dir))
from util import get_default_save_path, _next_version
root = save_path or get_default_save_path(
    analysis=analysis, quantity=quantity, cosmo_model=cosmo_model,
    data_release=data_release)
print(_next_version(root))
PY
)"
    echo "Auto-resolved version: v$VERSION (next free, no overwrite)"
fi

if ! [[ "$VERSION" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not resolve a numeric version (got '$VERSION'). " \
         "Set \$SCRATCH, or pass --save-path / --version explicitly. " \
       "If the snippet above raised, that is the real cause -- read its " \
       "traceback rather than this line." >&2
    exit 1
fi

SAVE_PATH_FLAG=""
[[ -n "$SAVE_PATH" ]] && SAVE_PATH_FLAG="--save-path $SAVE_PATH"

echo "=== $(basename "$GEN_SCRIPT"): analysis=$ANALYSIS quantity=$QUANTITY data_release=$DATA_RELEASE model=$COSMO_MODEL n=$N_SAMPLES workers=$WORKERS version=v$VERSION ==="
echo "Submitting one job per tracer: ${TRACER_NAMES[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "SLURM: time=$TIME, queue=$QUEUE, nodes=$NODES"
echo

for NAME in "${TRACER_NAMES[@]}"; do
    JOB_NAME="gen_data_${ANALYSIS}_${QUANTITY}_${COSMO_MODEL}_${NAME}"

    # Write a temporary batch script so we get full bash (not /bin/sh from --wrap)
    BATCH_SCRIPT=$(mktemp /tmp/gen_data_XXXXXX.sh)
    cat > "$BATCH_SCRIPT" << 'INNEREOF'
#!/bin/bash
set -euo pipefail

# Combined log file (stdout + stderr)
JOB_LOG="__LOG_DIR__/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"
mkdir -p "__LOG_DIR__"
touch "$JOB_LOG"
exec > >(tee -a "$JOB_LOG") 2>&1

echo "=========================================="
echo "SLURM Job Started"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Node:         $(hostname)"
echo "Tracer:       __NAME__"
echo "Analysis:     __ANALYSIS__"
echo "Quantity:     __QUANTITY__"
echo "Data release: __DATA_RELEASE__"
echo "Cosmo Model:  __COSMO_MODEL__"
echo "Version:      v__VERSION__"
echo "Workers:      __WORKERS__"
echo "N samples:    __N_SAMPLES__"
echo "Start Time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Extra Args:   __EXTRA_ARGS__"
echo "=========================================="
echo ""

module load conda
conda activate emulator

# mpi4py (ABI build) has no bundled MPI; provide Cray MPICH's libmpi.so.12.
# Disable GPU MPI support on these CPU nodes (no GTL lib linked).
module load cray-mpich-abi
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${CRAY_MPICH_DIR}/lib-abi-mpich:${LD_LIBRARY_PATH:-}"

# Pin JAX/XLA's CPU threadpool to 1 thread per worker. generate_dataset already
# pins BLAS (OMP/MKL/...) to 1 thread per spawn worker, but NOT XLA — left
# unpinned, each of the N jax workers spins up an XLA eigen pool sized to all
# cores, so N workers x ~ncores threads massively oversubscribe and the run
# deadlocks at high --workers. This keeps each worker single-threaded so workers
# scale cleanly up to the core count. Inherited by the spawn children via env.
# Needed by bao's jax config-space path; harmless (and left in place) for
# shapefit, whose Fisher path does not go through jax.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_PLATFORMS=cpu
# Workers all open the same DESI .h5 bundle read-only; HDF5's default file lock
# serializes that on $HOME. Safe to disable for read-only access.
export HDF5_USE_FILE_LOCKING=FALSE

cd __SCRIPT_DIR__

python __GEN_SCRIPT__ \
    __GEN_QUANTITY_FLAG__ \
    --data-release __DATA_RELEASE__ \
    --tracer-bin "__NAME__" \
    --cosmo-model __COSMO_MODEL__ \
    --n-samples __N_SAMPLES__ \
    --workers __WORKERS__ \
    --version __VERSION__ \
    __SAVE_PATH_FLAG__ \
    __EXTRA_ARGS_RAW__

echo ""
echo "Tracer __NAME__ complete (saved to .../__ANALYSIS__/training_data/__DATA_RELEASE__/__COSMO_MODEL__/__QUANTITY__/v__VERSION__/)."
echo "End: $(date)"
INNEREOF

    # Substitute placeholders
    sed -i "s|__LOG_DIR__|$LOG_DIR|g" "$BATCH_SCRIPT"
    sed -i "s|__SCRIPT_DIR__|$SCRIPT_DIR|g" "$BATCH_SCRIPT"
    sed -i "s|__GEN_SCRIPT__|$GEN_SCRIPT|g" "$BATCH_SCRIPT"
    sed -i "s|__NAME__|$NAME|g" "$BATCH_SCRIPT"
    sed -i "s|__ANALYSIS__|$ANALYSIS|g" "$BATCH_SCRIPT"
    sed -i "s|__QUANTITY__|$QUANTITY|g" "$BATCH_SCRIPT"
    sed -i "s|__GEN_QUANTITY_FLAG__|$GEN_QUANTITY_FLAG|g" "$BATCH_SCRIPT"
    sed -i "s|__DATA_RELEASE__|$DATA_RELEASE|g" "$BATCH_SCRIPT"
    sed -i "s|__COSMO_MODEL__|$COSMO_MODEL|g" "$BATCH_SCRIPT"
    sed -i "s|__N_SAMPLES__|$N_SAMPLES|g" "$BATCH_SCRIPT"
    sed -i "s|__WORKERS__|$WORKERS|g" "$BATCH_SCRIPT"
    sed -i "s|__VERSION__|$VERSION|g" "$BATCH_SCRIPT"
    sed -i "s|__SAVE_PATH_FLAG__|$SAVE_PATH_FLAG|g" "$BATCH_SCRIPT"
    sed -i "s|__EXTRA_ARGS__|${EXTRA_ARGS[*]:-<none>}|g" "$BATCH_SCRIPT"
    sed -i "s|__EXTRA_ARGS_RAW__|${EXTRA_ARGS[*]:-}|g" "$BATCH_SCRIPT"

    echo "Submitting $JOB_NAME"
    sbatch \
        --job-name="$JOB_NAME" \
        -A desi \
        -C cpu \
        -q "$QUEUE" \
        -N "$NODES" \
        --ntasks=1 \
        --cpus-per-task="$WORKERS" \
        --mem=0 \
        -t "$TIME" \
        -o /dev/null \
        -e /dev/null \
        "$BATCH_SCRIPT"
done

echo
echo "All jobs submitted. Outputs will land in .../$ANALYSIS/training_data/$DATA_RELEASE/$COSMO_MODEL/$QUANTITY/v$VERSION/"
