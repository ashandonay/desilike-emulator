#!/usr/bin/env bash
# Seed sweep for MCMC chain-noise estimate on Fisher-vs-DESI scatter plot.
#
# For each (tracer, cov, seed) combination, runs the corresponding MCMC and
# stores the σ JSON under bao/seed_sweep/. Existing production runs (seed=42)
# are NOT re-run — they are already in mcmc_results/ and mcmc_results_bundle_native/
# and will be picked up by the aggregator alongside the new seeds {1,2,3}.

set -e
set -o pipefail

PY=~/miniconda3/envs/emulator/bin/python
export LD_LIBRARY_PATH=/home/ashandonay/miniconda3/envs/emulator/lib:${LD_LIBRARY_PATH:-}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/seed_sweep"
BUNDLE_OUT="$HERE/mcmc_results_bundle_native"
mkdir -p "$OUT"

cd "$HERE"

TRACERS=(BGS LRG1 LRG2 LRG3_ELG1 ELG2 QSO)
SEEDS=(1 2 3)

# Sparse tracers use qiso; multipole tracers use qparqper.
apmode_for() {
    case "$1" in
        BGS|QSO) echo qiso ;;
        *)       echo qparqper ;;
    esac
}

bundle_suffix_for() {
    case "$1" in
        BGS|QSO) echo "_qiso_bbdesi_adame24" ;;
        *)       echo "_bbdesi_adame24" ;;
    esac
}

t0=$(date +%s)
n_total=$(( ${#TRACERS[@]} * ${#SEEDS[@]} * 2 ))
n_done=0

for seed in "${SEEDS[@]}"; do
    for tracer in "${TRACERS[@]}"; do
        apmode=$(apmode_for "$tracer")
        bsuf=$(bundle_suffix_for "$tracer")

        # ---------- production analytic-cov MCMC ----------
        prod_out="$OUT/${tracer}_prod_seed${seed}.json"
        if [[ ! -s "$prod_out" ]]; then
            echo "[$(date +%H:%M:%S)] [$((n_done+1))/$n_total] PROD  $tracer  seed=$seed"
            $PY mcmc_bao.py --tracer-bin "$tracer" --dataset dr1 \
                --nwalkers 128 --seed "$seed" --save-json "$prod_out" \
                > "$OUT/${tracer}_prod_seed${seed}.log" 2>&1
        else
            echo "[$(date +%H:%M:%S)] [$((n_done+1))/$n_total] PROD  $tracer  seed=$seed (cached)"
        fi
        n_done=$((n_done+1))

        # ---------- bundle-cov native-theory MCMC ----------
        bundle_out="$OUT/${tracer}_bundle_seed${seed}.json"
        if [[ ! -s "$bundle_out" ]]; then
            echo "[$(date +%H:%M:%S)] [$((n_done+1))/$n_total] BUND  $tracer  seed=$seed  apmode=$apmode"
            $PY mcmc_bundle_native_theory.py --tracers "$tracer" \
                --apmode "$apmode" --bb-basis desi_adame24 --seed "$seed" \
                > "$OUT/${tracer}_bundle_seed${seed}.log" 2>&1
            # mcmc_bundle_native_theory writes to a fixed path; move to seed-tagged name.
            mv "$BUNDLE_OUT/${tracer}_dr1_native_theory${bsuf}.json" "$bundle_out"
        else
            echo "[$(date +%H:%M:%S)] [$((n_done+1))/$n_total] BUND  $tracer  seed=$seed (cached)"
        fi
        n_done=$((n_done+1))

        elapsed=$(( $(date +%s) - t0 ))
        rate=$(awk -v e=$elapsed -v n=$n_done 'BEGIN{ if(n>0) printf "%.1f", e/n; else print "0" }')
        eta=$(awk -v r=$rate -v rem=$((n_total-n_done)) 'BEGIN{ printf "%.0f", r*rem/60 }')
        echo "         elapsed=${elapsed}s  rate=${rate}s/run  ETA=${eta}min"
    done
done

# Restore the production seed=42 bundle JSONs so plot_fisher_vs_desi.py
# still works (they were never touched; this is just a paranoia check).
echo "Done. Bundle production JSONs in $BUNDLE_OUT (seed=42, untouched):"
ls -la "$BUNDLE_OUT" | grep -v "^total\|^d" | head
