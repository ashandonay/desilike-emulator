"""Seed sweep on sparse-tracer (BGS DV, QSO DV) bundle MCMC.

Per the project memory: sparse-tracer 2500-iter chains have unreliable σ_DV;
need a seed sweep + long chain. This script runs N_SEEDS chains per tracer at
both the standard 2500 iter and a longer 5000 iter, then reports seed-to-seed
scatter on σ_DV.

If seed scatter on σ_DV is comparable to the BGS overshoot factor (3.25× DESI),
the all-six headline was a chain artifact. If σ_DV is stable across seeds and
still overshoots, the BGS bundle cov really is that much wider than what DESI
analysed.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import mcmc_bundle as mb

SEEDS = [42, 137, 271, 4096, 8192]
N_ITER_STANDARD = 2500
N_ITER_LONG = 5000

OUT_DIR = Path(__file__).resolve().parent / "mcmc_results_bundle"
OUT_DIR.mkdir(exist_ok=True)


def run_sweep(tracer, n_iter):
    sigmas_DV = []
    sigmas_DH = []
    sigmas_DM = []
    for seed in SEEDS:
        result = mb.run(
            tracer=tracer,
            nwalkers=64,
            max_iterations=n_iter,
            burnin_frac=0.3,
            seed=seed,
            alpha_low=0.8,
            alpha_high=1.2,
            save_chain=None,
        )
        # Save per-seed result with seed in filename
        per_seed_path = OUT_DIR / f"{tracer}_dr1_bundle_seed{seed}_iter{n_iter}.json"
        per_seed_path.write_text(json.dumps(result, indent=2))
        sigmas_DV.append(result["sigma_DV_over_rd_mcmc"])
        sigmas_DH.append(result["sigma_DH_over_rd_mcmc"])
        sigmas_DM.append(result["sigma_DM_over_rd_mcmc"])
    arr_DV = np.asarray(sigmas_DV)
    arr_DH = np.asarray(sigmas_DH)
    arr_DM = np.asarray(sigmas_DM)
    desi_DV = result["sigma_DV_over_rd_data"]
    summary = {
        "tracer": tracer,
        "n_iter": n_iter,
        "seeds": SEEDS,
        "sigma_DV_per_seed": sigmas_DV,
        "sigma_DH_per_seed": sigmas_DH,
        "sigma_DM_per_seed": sigmas_DM,
        "sigma_DV_mean": float(arr_DV.mean()),
        "sigma_DV_std":  float(arr_DV.std(ddof=1)) if len(arr_DV) > 1 else 0.0,
        "sigma_DV_min":  float(arr_DV.min()),
        "sigma_DV_max":  float(arr_DV.max()),
        "sigma_DH_mean": float(arr_DH.mean()),
        "sigma_DH_std":  float(arr_DH.std(ddof=1)) if len(arr_DH) > 1 else 0.0,
        "sigma_DM_mean": float(arr_DM.mean()),
        "sigma_DM_std":  float(arr_DM.std(ddof=1)) if len(arr_DM) > 1 else 0.0,
        "desi_DV": desi_DV,
        "DV_ratio_mean_to_desi": float(arr_DV.mean()) / desi_DV if desi_DV > 0 else float("nan"),
        "DV_scatter_frac": float(arr_DV.std(ddof=1) / arr_DV.mean()) if arr_DV.mean() > 0 else float("nan"),
    }
    summary_path = OUT_DIR / f"{tracer}_dr1_bundle_seedsweep_iter{n_iter}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  *** {tracer} seed-sweep summary (iter={n_iter}) ***", flush=True)
    print(f"      σ(DV/rd) per seed: {[f'{s:.4f}' for s in sigmas_DV]}", flush=True)
    print(f"      σ(DV/rd) mean ± std: {arr_DV.mean():.4f} ± {arr_DV.std(ddof=1):.4f}  "
          f"(scatter {summary['DV_scatter_frac']*100:.1f}%)", flush=True)
    print(f"      DESI σ(DV/rd) = {desi_DV:.4f},  mean-ratio = {summary['DV_ratio_mean_to_desi']:.3f}", flush=True)
    return summary


def main():
    for tracer in ["BGS", "QSO"]:
        print(f"\n{'#' * 70}")
        print(f"#  {tracer} seed sweep ({len(SEEDS)} seeds × {N_ITER_STANDARD} iter)")
        print(f"{'#' * 70}", flush=True)
        run_sweep(tracer, N_ITER_STANDARD)
        print(f"\n{'#' * 70}")
        print(f"#  {tracer} long-chain ({len(SEEDS)} seeds × {N_ITER_LONG} iter)")
        print(f"{'#' * 70}", flush=True)
        run_sweep(tracer, N_ITER_LONG)


if __name__ == "__main__":
    main()
