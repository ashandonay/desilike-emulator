#!/usr/bin/env bash
# Submit / launch NN emulator training — one job (or local process) per tracer.
#
# Auto-detects SLURM (sbatch). On NERSC Perlmutter submits GPU jobs; elsewhere
# (or with --local) runs train.py directly. Scratch paths follow $SCRATCH
# (same convention as ~/bedcosmo/submit.sh).
#
# Usage:
#   bash submit_train.sh --analysis bao --quantity config --epochs 5000
#   bash submit_train.sh --analysis bao --dataset dr1 --quantity config \
#       --cosmo-model base --nn-model dr1_base_config --data-dir v2
#   bash submit_train.sh --tracers "BGS LRG2" --queue debug --time 00:30:00 \
#       --analysis bao --dataset dr1 --quantity config --nn-model dr1_base_config \
#       --data-dir v2 --epochs 500
#   bash submit_train.sh --local --tracers BGS --analysis bao --quantity config \
#       --epochs 100
#   bash submit_train.sh --analysis shapefit --quantity mean --lr 1e-3 --scheduler-type cosine
#
# Options (consumed here; everything else is forwarded to train.py):
#   --time HH:MM:SS    walltime per SLURM job     (default: 01:00:00; shared max 04:00:00)
#   --queue Q          SLURM queue                (default: regular)
#                      shared: --gpus-per-task=1 + 32 CPUs; regular/debug: --gpus-per-node=1 + 32 CPUs
#   --tracers "A B C"  space-separated subset     (default: all 6 DR1 bins)
#   --tracer-bin NAME  train a single tracer      (alias for --tracers NAME)
#   --local            force local execution even if sbatch is available
#   --slurm            force SLURM submission even if sbatch is not detected
#   (default: auto-detect based on sbatch availability)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure SCRATCH is set (logs, MLflow, models — same as ~/bedcosmo/submit.sh).
if [[ -z "${SCRATCH:-}" ]]; then
    export SCRATCH="$HOME/scratch"
    echo "SCRATCH was unset; set to: $SCRATCH"
fi
LOG_DIR="${SCRATCH}/bedcosmo/num_tracers/emulator/logs"

# ── Defaults ──
TIME="01:00:00"
QUEUE="regular"
TRACERS="BGS LRG1 LRG2 LRG3_ELG1 ELG2 QSO"
FORCE_LOCAL=false
FORCE_SLURM=false

# ── Parse SLURM / mode flags vs train.py flags ──
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time)
            TIME="$2"; shift 2 ;;
        --queue)
            QUEUE="$2"; shift 2 ;;
        --tracers)
            if [[ $# -ge 2 && ! "$2" =~ ^-- ]]; then
                TRACERS="$2"; shift 2
            else
                shift
            fi
            ;;
        --tracer-bin)
            TRACERS="$2"; shift 2 ;;
        --local)
            FORCE_LOCAL=true; shift ;;
        --slurm)
            FORCE_SLURM=true; shift ;;
        *)
            TRAIN_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
    echo "Usage: bash submit_train.sh [--time HH:MM:SS] [--queue Q] [--tracers \"A B ...\"] [--local|--slurm] [train.py args ...]"
    echo "  One GPU job/process per tracer (default: all 6 DR1 bins)."
    echo "  Use --tracers or --tracer-bin to restrict to a subset."
    echo "  Mode: auto (sbatch if present), or --local / --slurm."
    exit 1
fi

if [[ "$FORCE_LOCAL" == true && "$FORCE_SLURM" == true ]]; then
    echo "Error: --local and --slurm are mutually exclusive" >&2
    exit 1
fi

if [[ "$FORCE_LOCAL" == true ]]; then
    EXECUTION_MODE="local"
elif [[ "$FORCE_SLURM" == true ]]; then
    EXECUTION_MODE="slurm"
elif command -v sbatch &>/dev/null; then
    EXECUTION_MODE="slurm"
else
    EXECUTION_MODE="local"
fi

mkdir -p "$LOG_DIR"

_time_to_secs() {
    local t="$1" h m s
    IFS=: read -r h m s <<< "$t"
    echo $((10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0}))
}

_activate_emulator_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "emulator" ]]; then
        echo "Using existing conda environment: emulator"
        return 0
    fi
    local conda_base=""
    if [[ -n "${CONDA_PREFIX:-}" && "$CONDA_PREFIX" == *"/envs/"* ]]; then
        conda_base="${CONDA_PREFIX%/envs/*}"
    else
        conda_base="$(conda info --base 2>/dev/null || true)"
    fi
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$conda_base/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook 2>/dev/null)" || {
            echo "Error: conda not found. Activate the emulator env first: conda activate emulator" >&2
            exit 1
        }
    fi
    conda activate emulator
    echo "Activated conda environment: emulator"
    if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}

submit_one_job() {
    local job_name="$1"
    local tracer_bin="$2"
    shift 2
    local -a args=("$@")

    local args_str="${args[*]}"
    if [[ -n "$tracer_bin" ]]; then
        args_str="${args_str} --tracer-bin ${tracer_bin}"
    fi

    echo "Submitting $job_name: time=$TIME, queue=$QUEUE"
    echo "train.py args: $args_str"

    local -a sbatch_extra_args=()
    local -a sbatch_gpu_args
    if [[ "$QUEUE" == "shared" ]]; then
        # Shared GPU: partial-node allocation. Do NOT use -N 1 or --mem=0 (full node);
        # those pass --test-only but fail real submission on gpu_shared_ss11.
        sbatch_gpu_args=(--gpus-per-task=1 --cpus-per-task=32)
        echo "SLURM resources: 1 GPU, 32 CPUs (--gpus-per-task, shared queue)"
    else
        sbatch_extra_args=(-N 1 --mem=0)
        sbatch_gpu_args=(--gpus-per-node=1 --cpus-per-task=32)
        echo "SLURM resources: 1 GPU, 32 CPUs (--gpus-per-node)"
    fi

    local batch_script
    batch_script=$(mktemp /tmp/emulator_train_XXXXXX.sh)
    cat > "$batch_script" << 'INNEREOF'
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
echo "Tracer:       __TRACER__"
echo "SCRATCH:      ${SCRATCH:-<unset>}"
echo "Start Time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Args:         __TRAIN_ARGS__"
echo "=========================================="
echo ""

module load conda
conda activate emulator

# mpi4py (ABI build) has no bundled MPI; provide Cray MPICH's libmpi.so.12.
# Needed for post-training eval (desilike import). Disable GPU MPI (no GTL lib).
module load cray-mpich-abi
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${CRAY_MPICH_DIR}/lib-abi-mpich:${LD_LIBRARY_PATH:-}"

cd __SCRIPT_DIR__

python __SCRIPT_DIR__/train.py __TRAIN_ARGS__

echo ""
echo "Training complete."
echo "End: $(date)"
INNEREOF

    sed -i "s|__LOG_DIR__|$LOG_DIR|g" "$batch_script"
    sed -i "s|__SCRIPT_DIR__|$SCRIPT_DIR|g" "$batch_script"
    sed -i "s|__TRACER__|${tracer_bin:-<none>}|g" "$batch_script"
    sed -i "s|__TRAIN_ARGS__|$args_str|g" "$batch_script"

    sbatch \
        --job-name="$job_name" \
        -A desi \
        -C gpu \
        -q "$QUEUE" \
        "${sbatch_extra_args[@]}" \
        --ntasks=1 \
        "${sbatch_gpu_args[@]}" \
        -t "$TIME" \
        -o /dev/null \
        -e /dev/null \
        "$batch_script"
}

run_one_local() {
    local job_name="$1"
    local tracer_bin="$2"
    shift 2
    local -a args=("$@")

    if [[ -n "$tracer_bin" ]]; then
        args+=(--tracer-bin "$tracer_bin")
    fi

    local timestamp
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    local log_file="${LOG_DIR}/${timestamp}_${job_name}.log"

    echo "Launching local $job_name"
    echo "train.py args: ${args[*]}"
    echo "Logging to: $log_file"

    cd "$SCRIPT_DIR"
    # Prefer env python if already in emulator; else rely on PATH after activate.
    local py="${CONDA_PREFIX:-}/bin/python"
    if [[ ! -x "$py" ]]; then
        py="python"
    fi
    "$py" "$SCRIPT_DIR/train.py" "${args[@]}" > >(tee -a "$log_file") 2>&1
}

read -r -a TRACER_NAMES <<< "$TRACERS"
if [[ ${#TRACER_NAMES[@]} -eq 0 ]]; then
    echo "Error: no tracers given"
    exit 1
fi

# Optional job-name tags from common train.py flags.
ANALYSIS="train"
QUANTITY=""
COSMO_MODEL=""
NN_MODEL=""
for ((i = 0; i < ${#TRAIN_ARGS[@]}; i++)); do
    case "${TRAIN_ARGS[i]}" in
        --analysis)     ANALYSIS="${TRAIN_ARGS[i + 1]:-train}" ;;
        --quantity)     QUANTITY="${TRAIN_ARGS[i + 1]:-}" ;;
        --cosmo-model)  COSMO_MODEL="${TRAIN_ARGS[i + 1]:-}" ;;
        --nn-model)     NN_MODEL="${TRAIN_ARGS[i + 1]:-}" ;;
    esac
done

echo "=== emulator train: analysis=$ANALYSIS quantity=${QUANTITY:-<default>} cosmo=${COSMO_MODEL:-<default>} nn_model=${NN_MODEL:-<default>} ==="
echo "Execution mode: $EXECUTION_MODE"
echo "SCRATCH: $SCRATCH"
echo "Logs: $LOG_DIR"
echo "One job/process per tracer: ${TRACER_NAMES[*]}"
echo "Shared train.py args: ${TRAIN_ARGS[*]}"
if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    echo "SLURM: time=$TIME, queue=$QUEUE"
fi
echo

if [[ "$EXECUTION_MODE" == "slurm" ]]; then
    if ! command -v sbatch &>/dev/null; then
        echo "Error: --slurm requested but sbatch not found on PATH" >&2
        exit 1
    fi
    if [[ "$QUEUE" == "shared" ]]; then
        max_shared_secs=$((4 * 3600))
        req_secs=$(_time_to_secs "$TIME")
        if (( req_secs > max_shared_secs )); then
            echo "Error: shared GPU queue max walltime is 04:00:00 (requested $TIME)." >&2
            echo "Use --queue regular for longer jobs, or reduce --time." >&2
            exit 1
        fi
    fi
    for NAME in "${TRACER_NAMES[@]}"; do
        JOB_NAME="train_${ANALYSIS}"
        [[ -n "$QUANTITY" ]] && JOB_NAME+="_${QUANTITY}"
        [[ -n "$COSMO_MODEL" ]] && JOB_NAME+="_${COSMO_MODEL}"
        [[ -n "$NN_MODEL" ]] && JOB_NAME+="_${NN_MODEL}"
        JOB_NAME+="_${NAME}"
        submit_one_job "$JOB_NAME" "$NAME" "${TRAIN_ARGS[@]}"
    done
    echo
    echo "All jobs submitted (${#TRACER_NAMES[@]} tracers)."
else
    _activate_emulator_env
    for NAME in "${TRACER_NAMES[@]}"; do
        JOB_NAME="train_${ANALYSIS}"
        [[ -n "$QUANTITY" ]] && JOB_NAME+="_${QUANTITY}"
        [[ -n "$COSMO_MODEL" ]] && JOB_NAME+="_${COSMO_MODEL}"
        [[ -n "$NN_MODEL" ]] && JOB_NAME+="_${NN_MODEL}"
        JOB_NAME+="_${NAME}"
        run_one_local "$JOB_NAME" "$NAME" "${TRAIN_ARGS[@]}"
    done
    echo
    echo "All local runs finished (${#TRACER_NAMES[@]} tracers)."
fi
