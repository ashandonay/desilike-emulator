"""Aggregate σ-of-σ across MCMC seed sweep.

Reads:
  - seed_sweep/{tracer}_{cov}_seed{N}.json for N in {1,2,3}
  - mcmc_results_bundle_native_seed42_backup/*.json   (seed=42 bundle)
  - mcmc_results/{tracer}_dr1.json                    (seed=42 prod)

For each (tracer, quantity, cov), computes mean and std of σ across seeds.
Prints a table and writes seed_sweep/sigma_of_sigma.json which the plot
script reads to draw error bars on the open markers.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SWEEP = _HERE / "seed_sweep"
_PROD42 = _HERE / "mcmc_results"
_BUNDLE42 = _HERE / "mcmc_results_bundle_native_seed42_backup"

_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_QUANTS = ["DH", "DM", "DV"]
_SPARSE = ("BGS", "QSO")


def _load_prod(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {
        "DH": d.get("sigma_DH_over_rd_mcmc"),
        "DM": d.get("sigma_DM_over_rd_mcmc"),
        "DV": d.get("sigma_DV_over_rd_mcmc"),
    }


def _load_bundle(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {"DH": d.get("sig_DH"), "DM": d.get("sig_DM"), "DV": d.get("sig_DV")}


def _gather_seeds(tracer, cov):
    """Return {seed: {DH, DM, DV}} dict for the given (tracer, cov)."""
    out = {}
    # seed=42 (production)
    if cov == "prod":
        v = _load_prod(_PROD42 / f"{tracer}_dr1.json")
    else:
        suffix = "_qiso_bbdesi_adame24" if tracer in _SPARSE else "_bbdesi_adame24"
        v = _load_bundle(_BUNDLE42 / f"{tracer}_dr1_native_theory{suffix}.json")
    if v is not None:
        out[42] = v
    # seeds 1,2,3 (sweep)
    for seed in (1, 2, 3):
        p = _SWEEP / f"{tracer}_{cov}_seed{seed}.json"
        if cov == "prod":
            v = _load_prod(p)
        else:
            v = _load_bundle(p)
        if v is not None:
            out[seed] = v
    return out


def main():
    rows = []
    summary = {}
    for tracer in _TRACERS:
        is_sparse = tracer in _SPARSE
        for cov in ("prod", "bundle"):
            seeds = _gather_seeds(tracer, cov)
            for q in _QUANTS:
                # mask irrelevant quantities (DV for multipole, DH/DM for sparse)
                if is_sparse and q != "DV":  continue
                if not is_sparse and q == "DV":  continue
                vals = [v[q] for v in seeds.values() if v.get(q) is not None]
                if len(vals) < 2:
                    continue
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1))
                rows.append((tracer, cov, q, len(vals), mu, sd, sd / mu if mu > 0 else float("nan")))
                summary.setdefault(tracer, {}).setdefault(cov, {})[q] = {
                    "n_seeds": len(vals), "mean": mu, "std": sd,
                    "values": {str(k): v[q] for k, v in seeds.items()},
                }

    # Print table
    print(f"  {'tracer':<11} {'cov':<7} {'q':<3} {'N':>3} {'mean(σ)':>9} {'σ-of-σ':>9} {'%':>6}")
    print("  " + "-" * 56)
    for tracer, cov, q, n, mu, sd, frac in rows:
        print(f"  {tracer:<11} {cov:<7} {q:<3} {n:>3} {mu:>9.4f} {sd:>9.4f} {frac*100:>5.1f}%")

    out_path = _SWEEP / "sigma_of_sigma.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
