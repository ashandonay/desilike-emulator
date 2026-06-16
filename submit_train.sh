#!/usr/bin/env bash
# Submit SLURM job(s) to train the NN emulator on NERSC Perlmutter (GPU).
# With --tracers, submits ONE JOB PER TRACER so bins train in parallel.
#
# Usage:
#   bash submit_train.sh --analysis bao --quantity config --epochs 5000
#   bash submit_train.sh --tracers --analysis bao --dataset dr1 --quantity config \
#       --cosmo-model base --nn-model dr1_base_config --data-dir v2
#   bash submit_train.sh --tracers "BGS LRG2" --queue debug --time 00:30:00 \
#       --analysis bao --dataset dr1 --quantity config --nn-model dr1_base_config \
#       --data-dir v2 --epochs 500
#   bash submit_train.sh --analysis shapefit --quantity mean --lr 1e-3 --scheduler-type cosine
#
# Options (SLURM flags consumed here):
#   --time HH:MM:SS    walltime per job          (default: 01:00:00)
#   --queue Q          SLURM queue               (default: regular)
#   --tracers "A B C"  space-separated subset     (default with --tracers alone: all 6 DR1 bins)
#                      Omit --tracers for a single job (forwards --tracer-bin if given).
# Everything else is forwarded to train.py (per-tracer jobs get --tracer-bin <name> appended).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/pscratch/sd/a/ashandon/bedcosmo/num_tracers/emulator/logs"

# ── Defaults ──
TIME="01:00:00"
QUEUE="regular"
TRACERS_DEFAULT="BGS LRG1 LRG2 LRG3_ELG1 ELG2 QSO"
TRACERS=""
MULTI_TRACER=false

# ── Parse SLURM flags vs train.py flags ──
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time)
            TIME="$2"; shift 2 ;;
        --queue)
            QUEUE="$2"; shift 2 ;;
        --tracers)
            MULTI_TRACER=true
            if [[ $# -ge 2 && ! "$2" =~ ^-- ]]; then
                TRACERS="$2"; shift 2
            else
                TRACERS="$TRACERS_DEFAULT"; shift
            fi
            ;;
        --tracer-bin)
            if $MULTI_TRACER; then
                echo "Warning: ignoring --tracer-bin $2 (--tracers submits one job per tracer)" >&2
            else
                TRAIN_ARGS+=("$1" "$2")
            fi
            shift 2 ;;
        *)
            TRAIN_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#TRAIN_ARGS[@]} -eq 0 && ! $MULTI_TRACER ]]; then
    echo "Usage: bash submit_train.sh [--time HH:MM:SS] [--queue Q] [--tracers [\"A B ...\"]] [train.py args ...]"
    echo "  With --tracers: one GPU job per tracer (default list: all 6 DR1 bins)."
    echo "  Without --tracers: single job; all other flags forwarded to train.py."
    exit 1
fi

mkdir -p "$LOG_DIR"

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
echo "Start Time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Args:         __TRAIN_ARGS__"
echo "=========================================="
echo ""

module load conda
conda activate emulator
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
        -N 1 \
        --ntasks=1 \
        --cpus-per-task=32 \
        --gpus-per-node=1 \
        --mem=0 \
        -t "$TIME" \
        -o /dev/null \
        -e /dev/null \
        "$batch_script"
}

if ! $MULTI_TRACER; then
    submit_one_job "emulator_train" "" "${TRAIN_ARGS[@]}"
    exit 0
fi

read -r -a TRACER_NAMES <<< "$TRACERS"
if [[ ${#TRACER_NAMES[@]} -eq 0 ]]; then
    echo "Error: no tracers given"
    exit 1
fi

# Drop any --tracer-bin from forwarded args (each job sets its own).
FILTERED_ARGS=()
skip_next=false
for arg in "${TRAIN_ARGS[@]}"; do
    if $skip_next; then
        skip_next=false
        echo "Warning: ignoring --tracer-bin $arg (--tracers submits one job per tracer)" >&2
        continue
    fi
    if [[ "$arg" == "--tracer-bin" ]]; then
        skip_next=true
        continue
    fi
    FILTERED_ARGS+=("$arg")
done
TRAIN_ARGS=("${FILTERED_ARGS[@]}")

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
echo "Submitting one job per tracer: ${TRACER_NAMES[*]}"
echo "Shared train.py args: ${TRAIN_ARGS[*]}"
echo "SLURM: time=$TIME, queue=$QUEUE"
echo

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
