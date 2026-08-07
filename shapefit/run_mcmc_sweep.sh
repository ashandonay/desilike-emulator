#!/usr/bin/env bash
# Overnight MCMC sweep: every tracer bin, several seeds, all in parallel.
#
# mcmc.py cannot use a multiprocessing Pool -- make_log_prob returns a closure
# over the desilike likelihood, which is not picklable -- so parallelism has to
# be ACROSS PROCESSES. One process per (tracer, seed), each pinned to a single
# BLAS thread so 24 jobs on a 64-core box do not oversubscribe each other.
#
# Each job writes its own JSON; comparison_plots.py `_load_mcmc` globs and
# unions the seeds per tracer, so the sweep needs no merge step.
#
#   ./run_mcmc_sweep.sh                          # 6 tracers x 4 seeds
#   NITER=1000 SEEDS="1 2" ./run_mcmc_sweep.sh   # shorter
#   TRACERS="LRG2 QSO" ./run_mcmc_sweep.sh       # subset
#   COSMO=fiducial ./run_mcmc_sweep.sh           # reproduce S57/S63
#
# NITER is a CEILING, not a target (S72). Each chain stops when emcee's
# criterion is met -- niter > 50*tau AND the tau estimate has settled to 1% --
# so a job that exits early converged, and one that reaches NITER did not.
# Which is which is recorded per seed in the JSON (`converged`, `converged_at`).
# S63's fixed 2500 landed at 9-12 tau because tau's ESTIMATE grows with chain
# length: a Kaiser smoke run watched it go 6 -> 13 -> 19 -> 24 while niter/tau
# sat at 8-10 throughout. Picking a bigger fixed number repeats that mistake.
#
# Seed sweeps are not optional here: `feedback_mcmc_chain_noise_sparse_tracers`
# and the bao CHANGELOG both record that per-seed scatter is what bites, and
# that it is worst for the sparse tracers (QSO, BGS) -- exactly the ones a
# single-seed run would mislead us about.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/emulator/bin/python}"
TRACERS="${TRACERS:-BGS LRG1 LRG2 LRG3 ELG2 QSO}"
SEEDS="${SEEDS:-42 43 44 45}"
NITER="${NITER:-20000}"      # ceiling; see the convergence note above
BURN="${BURN:-0.4}"
THEORY="${THEORY:-rept}"
COSMO="${COSMO:-dr1_map}"    # S70/S71: DESI's published ShapeFit-alone MAP

LOGDIR="$("$PY" -c "import sys; sys.path.insert(0, '$HERE/..'); from util import logs_dir; print(logs_dir('shapefit'))")"
RUNDIR="$LOGDIR/mcmc_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUNDIR"

# One thread per process. Without this every job spawns its own BLAS pool and
# 24 jobs fight over 64 cores, which is slower than running them sequentially.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
       JAX_PLATFORMS=cpu MPLBACKEND=Agg

njobs=0
for t in $TRACERS; do
  for s in $SEEDS; do
    njobs=$((njobs + 1))
    ( cd "$HERE" && "$PY" mcmc.py \
        --tracers "$t" --seeds "$s" --theory "$THEORY" \
        --max-iterations "$NITER" --burnin-frac "$BURN" \
        --cosmology "$COSMO" \
        --json "$LOGDIR/shapefit_mcmc_${t}_seed${s}.json" \
        --save-chain "$LOGDIR/shapefit_mcmc_${t}.npz" \
    ) > "$RUNDIR/${t}_seed${s}.log" 2>&1 &
  done
done

cat > "$RUNDIR/RUN.md" <<EOF
# ShapeFit MCMC sweep

started    $(date)
tracers    $TRACERS
seeds      $SEEDS
iterations <=$NITER  (CEILING; stops early on convergence, burn-in $BURN)
theory     $THEORY
cosmology  $COSMO
jobs       $njobs, one process per (tracer, seed), 1 BLAS thread each
json       $LOGDIR/shapefit_mcmc_<tracer>_seed<seed>.json
chains     $LOGDIR/shapefit_mcmc_<tracer>_seed<seed>.npz

Progress:  tail -f $RUNDIR/LRG2_seed42.log
Summary:   grep -h 'tau_max' $RUNDIR/*.log
Converged: grep -c CONVERGED $RUNDIR/*.log
Plot:      python comparison_plots.py forecast

Partial chains land at $LOGDIR/shapefit_mcmc_<tracer>_seed<seed>_partial.npz
every $((2 * (NITER / 40 > 50 ? NITER / 40 : 50))) iterations and are deleted once the seed finishes, so an
interrupted sweep is recoverable rather than lost.
EOF

echo "launched $njobs jobs -> $RUNDIR"
cat "$RUNDIR/RUN.md"
wait
echo "sweep finished $(date)"
