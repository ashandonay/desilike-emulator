"""MCMC on the BUNDLE's cov_ξ substituted into LRG2's pipeline likelihood.

Counterpart to §20c (`mcmc_bao.py --dataset dr1` for the *pipeline* likelihood)
and §31g (Tier-3 Fisher on the bundle). Asks: when we use DESI's cov + W via
the F2 substitution, does the MCMC posterior widen Fisher's σ by more than
the ~10% §20c measured for the pipeline-cov case?

Mechanism: hijack the pipeline's `likelihood.precision` with
  precision_F2 = M^T (cov_ξ_DR1)^{-1} M
(same construction as cov_substitution_diag.py). The override is respected
by desilike's Fisher (per §31a, F1 ≈ F0 to 4e-4) — and equally by EmceeSampler
which just evaluates the likelihood.

Expected:
  Tier-3 Fisher (§31g):   σ(DH/rd) = 0.4803  (ratio 0.807 to DESI)
  This MCMC:              σ(DH/rd) = ?
  DESI published:         σ(DH/rd) = 0.5954
  §20c pipeline MCMC:     σ(DH/rd) = 0.296  (M/F = 1.10)

If σ_MCMC ≈ Tier-3 × 1.10 = 0.528 → bundle cov doesn't change the non-Gaussianity factor.
If σ_MCMC > Tier-3 × 1.10 → wider cov amplifies posterior non-Gaussianity.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

import prep_covar as pc
from util import TRACER_CONFIGS
from scipy.special import spherical_jn

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_BUNDLE = "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5"
_TRACER = "LRG2"
_DR1_AREA = 7500.0


def _load_bundle():
    p = _LIK_DIR / _BUNDLE
    with h5py.File(p, "r") as f:
        cov_xi = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        theory_ells, theory_s = [], []
        for k in sorted(f["window/theory"].keys(), key=lambda k: (not k.isdigit(), k)):
            if k.isdigit():
                theory_ells.append(int(f[f"window/theory/{k}/meta/ell"][()]))
                theory_s.append(np.asarray(f[f"window/theory/{k}/s"][...], dtype=np.float64))
    return cov_xi, W, theory_ells, theory_s


def _hankel_matrix(ell, k, dk, s):
    sign = (1j ** ell).real
    H = np.empty((len(s), len(k)), dtype=np.float64)
    for i, si in enumerate(s):
        H[i, :] = sign * (k ** 2) * spherical_jn(ell, k * si) * dk / (2.0 * np.pi ** 2)
    return H


def _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk):
    n_k = len(k_centers)
    n_pipe = n_k * len(pipe_ells)
    n_per = [len(s) for s in theory_s_per_ell]
    H_block = np.zeros((sum(n_per), n_pipe), dtype=np.float64)
    row = 0
    for it, te in enumerate(theory_ells):
        if te in pipe_ells:
            ip = pipe_ells.index(te)
            H_ell = _hankel_matrix(te, k_centers, dk, theory_s_per_ell[it])
            H_block[row:row + n_per[it], ip * n_k:(ip + 1) * n_k] = H_ell
        row += n_per[it]
    return W @ H_block


def _precision_from_cov_xi(cov_xi, M, regularize=1e-10):
    C = 0.5 * (cov_xi + cov_xi.T)
    ridge = regularize * np.trace(C) / C.shape[0]
    C_inv = np.linalg.inv(C + ridge * np.eye(C.shape[0]))
    C_inv = 0.5 * (C_inv + C_inv.T)
    P = M.T @ C_inv @ M
    return 0.5 * (P + P.T)


def _get_ntracers():
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    return {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}[_TRACER]


def _get_desi_std():
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    sub = df[df["tracer"] == _TRACER]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


def run(nwalkers, max_iterations, burnin_frac, seed, alpha_low, alpha_high, save_chain):
    cfg = TRACER_CONFIGS[_TRACER]
    theta, hrdrag_eff = pc._to_bao_cosmo_params(
        {**pc.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
    info = pc.build_bao_likelihood(
        N_tracers=_get_ntracers(), theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=_TRACER, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_DR1_AREA,
    )
    likelihood = info["likelihood"]
    template = info["template"]
    rd = info["rd"]
    obs = info["observable"]

    # Build M = W·H from the bundle, then F2 precision
    cov_xi, W, th_ells, th_s = _load_bundle()
    k = np.asarray(obs.k[0], dtype=np.float64)
    dk = float(np.mean(np.diff(k)))
    pipe_ells = [int(l) for l in obs.ells]
    M = _build_M(W, th_ells, th_s, pipe_ells, k, dk)
    prec_F2 = _precision_from_cov_xi(cov_xi, M)
    print(f"M shape: {M.shape},  cov_ξ_DR1: {cov_xi.shape},  precision: {prec_F2.shape}")

    # Override the precision in-place (mirrors cov_substitution_diag.py)
    likelihood.precision = np.asarray(prec_F2, dtype=np.float64)
    if hasattr(likelihood, "_precision_input"):
        likelihood._precision_input = likelihood.precision.copy()
    if hasattr(likelihood, "precision_hartlap2007"):
        likelihood.precision_hartlap2007 = likelihood.precision.copy()

    # Restrict qpar, qper to the same uniform box mcmc_bao.py uses
    likelihood.all_params["qpar"].update(
        prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
        ref={"dist": "norm", "loc": 1.0, "scale": 0.02},
    )
    likelihood.all_params["qper"].update(
        prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
        ref={"dist": "norm", "loc": 1.0, "scale": 0.02},
    )

    # Use template.{DH,DM}_over_rd_fid directly — dimensionally correct.
    # The naive `template.DH_fid / rd` form is dimensionally mixed
    # (DH_fid is Mpc/h, info["rd"] is proper Mpc) and underreports σ by 1/h.
    DH_fid_over_rd = float(template.DH_over_rd_fid)
    DM_fid_over_rd = float(template.DM_over_rd_fid)
    print(f"DH/rd_fid = {DH_fid_over_rd:.4f}, DM/rd_fid = {DM_fid_over_rd:.4f}")

    from desilike.samplers import EmceeSampler
    sampler_kwargs = dict(nwalkers=nwalkers, seed=seed)
    if save_chain:
        sampler_kwargs["save_fn"] = save_chain
    sampler = EmceeSampler(likelihood, **sampler_kwargs)
    print(f"\nRunning emcee: {nwalkers} walkers × {max_iterations} iter, seed={seed}...", flush=True)
    chains = sampler.run(max_iterations=max_iterations, check=False)
    chain = chains[0]
    n_total = chain.shape[0]
    n_burn = int(burnin_frac * n_total)
    q_par = np.asarray(chain["qpar"][n_burn:]).ravel()
    q_per = np.asarray(chain["qper"][n_burn:]).ravel()

    DH = q_par * DH_fid_over_rd
    DM = q_per * DM_fid_over_rd
    print(f"\nPost-burn samples: {q_par.size}")
    print(f"  σ(qpar) = {q_par.std(ddof=1):.5f}    σ(qper) = {q_per.std(ddof=1):.5f}")
    print(f"  σ(DH/rd) = {DH.std(ddof=1):.4f}")
    print(f"  σ(DM/rd) = {DM.std(ddof=1):.4f}")

    desi = _get_desi_std()
    print()
    print("=" * 80)
    print(f"LRG2 MCMC on bundle-cov-substituted likelihood (this run)")
    print("=" * 80)
    print(f"  Tier-3 Fisher (§31g):     σ(DH/rd) = 0.4803    ratio to DESI = 0.807")
    print(f"  This MCMC:                σ(DH/rd) = {DH.std(ddof=1):.4f}    ratio to DESI = {DH.std(ddof=1)/desi['DH_over_rs']:.3f}")
    print(f"  M/Tier-3-F ratio:         {DH.std(ddof=1)/0.4803:.3f}")
    print(f"  §20c pipeline-cov MCMC:   σ(DH/rd) = 0.296     M/F = 1.10")
    print(f"  DESI published:           σ(DH/rd) = {desi['DH_over_rs']:.4f}")
    print()
    print(f"  Tier-3 Fisher (§31g):     σ(DM/rd) = 0.2572    ratio to DESI = 0.805")
    print(f"  This MCMC:                σ(DM/rd) = {DM.std(ddof=1):.4f}    ratio to DESI = {DM.std(ddof=1)/desi['DM_over_rs']:.3f}")
    print(f"  M/Tier-3-F ratio:         {DM.std(ddof=1)/0.2572:.3f}")
    print(f"  §20c pipeline-cov MCMC:   σ(DM/rd) = 0.132     M/F = 0.99")
    print(f"  DESI published:           σ(DM/rd) = {desi['DM_over_rs']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--max-iterations", type=int, default=2500)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-low", type=float, default=0.8)
    ap.add_argument("--alpha-high", type=float, default=1.2)
    ap.add_argument("--save-chain", type=str, default=None)
    args = ap.parse_args()
    run(args.nwalkers, args.max_iterations, args.burnin_frac,
        args.seed, args.alpha_low, args.alpha_high, args.save_chain)


if __name__ == "__main__":
    main()
