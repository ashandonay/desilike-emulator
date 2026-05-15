"""MCMC sampling of the BAO likelihood built by prep_covar.build_bao_likelihood.

Motivation: Fisher (quadratic-log-likelihood + implicitly-flat priors) under-predicts
DESI BAO sigma(DH/rd), sigma(DM/rd) on low-S/N tracers (QSO, ELG2). MCMC with a truncated
uniform prior on (qpar, qper) tests whether the gap is due to posterior non-Gaussianity /
prior truncation rather than something in the Fisher pipeline itself.
"""
from __future__ import annotations

import os
# Match prep_covar env setup *before* any heavy imports.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import prep_covar as pc
from util import TRACER_CONFIGS


_DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}


def _get_nominal_ntracers(dataset: str, tracer_key: str) -> float:
    desi_dir = Path.home() / "data" / "desi" / f"bao_{dataset}"
    data_path = desi_dir / "desi_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data vector file: {data_path}")
    df = pd.read_csv(data_path)[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    n_by_tracer = {row["tracer"]: float(row["passed"]) for _, row in df.iterrows()}
    name_map = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
    return n_by_tracer[name_map.get(tracer_key, tracer_key)]


def _get_data_std(dataset: str, tracer_key: str) -> dict:
    desi_dir = Path.home() / "data" / "desi" / f"bao_{dataset}"
    df = pd.read_csv(desi_dir / "desi_data.csv")
    name_map = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
    tname = name_map.get(tracer_key, tracer_key)
    sub = df[df["tracer"] == tname]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


def run(
    tracer_bin: str,
    dataset: str,
    omega_m: float,
    hrdrag: float,
    alpha_low: float,
    alpha_high: float,
    nwalkers: int,
    max_iterations: int,
    burnin_frac: float,
    seed: int,
    save_chain: str | None,
) -> dict:
    cfg = TRACER_CONFIGS[tracer_bin]
    z_eff = float(cfg["z_eff"])
    zrange = tuple(cfg["zrange"])
    area = _DATASET_AREAS[dataset.lower()]
    n_tracers = _get_nominal_ntracers(dataset, tracer_bin)
    data_std = _get_data_std(dataset, tracer_bin)

    theta_cosmo, hrdrag_eff = pc._to_bao_cosmo_params(
        {**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag}
    )

    info = pc.build_bao_likelihood(
        N_tracers=n_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag_eff,
        tracer_bin=tracer_bin,
        zrange=zrange,
        z_eff=z_eff,
        area=area,
    )
    likelihood = info["likelihood"]
    template = info["template"]
    rd = info["rd"]

    DH_fid_over_rd = float(template.DH_fid) / rd
    DM_fid_over_rd = float(template.DM_fid) / rd

    # Restrict qpar, qper to a uniform box — this is the prior-truncation we want MCMC to see.
    likelihood.all_params["qpar"].update(
        prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
        ref={"dist": "norm", "loc": 1.0, "scale": 0.02},
    )
    likelihood.all_params["qper"].update(
        prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
        ref={"dist": "norm", "loc": 1.0, "scale": 0.02},
    )

    from desilike.samplers import EmceeSampler

    sampler_kwargs = dict(nwalkers=nwalkers, seed=seed)
    if save_chain:
        sampler_kwargs["save_fn"] = save_chain
    sampler = EmceeSampler(likelihood, **sampler_kwargs)
    chains = sampler.run(max_iterations=max_iterations, check=False)

    chain = chains[0]
    n_total = chain.shape[0]
    n_burn = int(burnin_frac * n_total)
    q_par = np.asarray(chain["qpar"][n_burn:]).ravel()
    q_per = np.asarray(chain["qper"][n_burn:]).ravel()

    dh = q_par * DH_fid_over_rd
    dm = q_per * DM_fid_over_rd
    dv = (z_eff * dm * dm * dh) ** (1.0 / 3.0)

    result = {
        "tracer_bin": tracer_bin,
        "dataset": dataset,
        "z_eff": z_eff,
        "n_samples_post_burn": int(q_par.size),
        "mean_qpar": float(q_par.mean()),
        "mean_qper": float(q_per.mean()),
        "sigma_qpar": float(q_par.std(ddof=1)),
        "sigma_qper": float(q_per.std(ddof=1)),
        "sigma_DH_over_rd_mcmc": float(dh.std(ddof=1)),
        "sigma_DM_over_rd_mcmc": float(dm.std(ddof=1)),
        "sigma_DV_over_rd_mcmc": float(dv.std(ddof=1)),
        "sigma_DH_over_rd_data": data_std.get("DH_over_rs", np.nan),
        "sigma_DM_over_rd_data": data_std.get("DM_over_rs", np.nan),
        "sigma_DV_over_rd_data": data_std.get("DV_over_rs", np.nan),
        "alpha_prior": [alpha_low, alpha_high],
        "DH_fid_over_rd": DH_fid_over_rd,
        "DM_fid_over_rd": DM_fid_over_rd,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracer-bin", required=True, choices=list(TRACER_CONFIGS.keys()))
    ap.add_argument("--dataset", default="dr2", choices=list(_DATASET_AREAS.keys()))
    ap.add_argument("--omega-m", type=float, default=0.3152)
    ap.add_argument("--hrdrag", type=float, default=99.08)
    ap.add_argument("--alpha-low", type=float, default=0.8)
    ap.add_argument("--alpha-high", type=float, default=1.2)
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-chain", default=None)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    result = run(
        tracer_bin=args.tracer_bin,
        dataset=args.dataset,
        omega_m=args.omega_m,
        hrdrag=args.hrdrag,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        nwalkers=args.nwalkers,
        max_iterations=args.max_iterations,
        burnin_frac=args.burnin_frac,
        seed=args.seed,
        save_chain=args.save_chain,
    )

    print(f"\n=== MCMC result: {args.tracer_bin} ({args.dataset}) ===")
    print(f"  alpha prior                [{args.alpha_low}, {args.alpha_high}]")
    print(f"  post-burn samples          {result['n_samples_post_burn']}")
    print(f"  mean qpar, qper            {result['mean_qpar']:.4f}, {result['mean_qper']:.4f}")
    print(f"  sigma qpar, qper           {result['sigma_qpar']:.4f}, {result['sigma_qper']:.4f}")
    print(f"  sigma(DH/rd) MCMC          {result['sigma_DH_over_rd_mcmc']:.4f}")
    print(f"  sigma(DH/rd) DESI data     {result['sigma_DH_over_rd_data']:.4f}")
    print(f"  sigma(DM/rd) MCMC          {result['sigma_DM_over_rd_mcmc']:.4f}")
    print(f"  sigma(DM/rd) DESI data     {result['sigma_DM_over_rd_data']:.4f}")
    print(f"  sigma(DV/rd) MCMC          {result['sigma_DV_over_rd_mcmc']:.4f}")
    print(f"  sigma(DV/rd) DESI data     {result['sigma_DV_over_rd_data']:.4f}")

    if args.save_json:
        import json
        with open(args.save_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved result to            {args.save_json}")


if __name__ == "__main__":
    main()
