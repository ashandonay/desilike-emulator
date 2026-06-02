"""Fourier-space BAO MCMC — sample the desilike likelihood from
core.build_bao_likelihood with emcee.

Two covariance modes (``--cov``):

  analytic : the pipeline's own analytic Gaussian covariance (no substitution).
             Forecast-style run; supports dr1/dr2 areas.
  bundle   : substitute the DESI DR1 RascalC bundle cov_ξ as the likelihood
             precision via  precision = Mᵀ (cov_ξ)⁻¹ M,  M = W·H_Hankel.
             DR1-only (bundle files are DR1).

Both reduce the posterior to σ(DH/rd), σ(DM/rd), σ(DV/rd). Sparse tracers
(core._ISO_TRACERS = BGS, QSO) use the isotropic qiso fit (monopole only);
others use qparqper. Reference σ is DESI bao-recon stat-only (the project
standard — NOT desi_data.csv, whose published σ folds in the systematic budget).

The config-space analog is mcmc_config.py.

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python mcmc_fourier.py \
        --tracers LRG2 QSO --cov analytic --dataset dr1
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.special import spherical_jn

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core as pc
from reference_sigmas import _recon_sigmas
from util import TRACER_CONFIGS

_FID = {"Om": 0.3152, "hrdrag": 99.08}            # canonical fiducial (reconciled)
_DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}

_BUNDLES = {
    "BGS":       "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}

_OUT_DIR = Path(__file__).resolve().parent / "mcmc_results_fourier"


def _get_ntracers(dataset: str, tracer: str) -> float:
    df = pd.read_csv(Path.home() / "data" / "desi" / f"bao_{dataset}" / "desi_data.csv")
    df = df[["tracer", "passed"]].drop_duplicates("tracer")
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP.get(tracer, tracer)]


# --- bundle cov substitution (precision = Mᵀ cov_ξ⁻¹ M, M = W·H_Hankel) ------
def _load_bundle(tracer):
    p = _LIK_DIR / _BUNDLES[tracer]
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


def run(tracer, cov, dataset, omega_m, hrdrag, alpha_low, alpha_high,
        nwalkers, max_iterations, burnin_frac, seed, apmode, save_chain):
    if apmode == "auto":
        apmode = "qiso" if tracer in pc._ISO_TRACERS else "qparqper"
    area = _DATASET_AREAS["dr1"] if cov == "bundle" else _DATASET_AREAS[dataset]
    print(f"\n{'=' * 70}\n  {tracer}  (cov={cov}, apmode={apmode}, area={area:.0f})\n{'=' * 70}", flush=True)

    theta, hrdrag_eff = pc._to_bao_cosmo_params(
        {**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
    cfg = TRACER_CONFIGS[tracer]
    info = pc.build_bao_likelihood(
        N_tracers=_get_ntracers("dr1" if cov == "bundle" else dataset, tracer),
        theta_cosmo=theta, hrdrag=hrdrag_eff, tracer_bin=tracer,
        zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]), area=area, apmode=apmode)
    likelihood = info["likelihood"]
    template = info["template"]

    if cov == "bundle":
        obs = info["observable"]
        cov_xi, W, th_ells, th_s = _load_bundle(tracer)
        k = np.asarray(obs.k[0], dtype=np.float64)
        dk = float(np.mean(np.diff(k)))
        pipe_ells = [int(l) for l in obs.ells]
        M = _build_M(W, th_ells, th_s, pipe_ells, k, dk)
        prec = _precision_from_cov_xi(cov_xi, M)
        print(f"  M {M.shape}, cov_ξ {cov_xi.shape}, precision {prec.shape}; "
              f"bundle ells {th_ells}, pipe ells {pipe_ells}", flush=True)
        likelihood.precision = np.asarray(prec, dtype=np.float64)
        if hasattr(likelihood, "_precision_input"):
            likelihood._precision_input = likelihood.precision.copy()
        if hasattr(likelihood, "precision_hartlap2007"):
            likelihood.precision_hartlap2007 = likelihood.precision.copy()

    # Truncated-uniform AP prior(s).
    ap_params = ["qiso"] if apmode == "qiso" else ["qpar", "qper"]
    for p in ap_params:
        likelihood.all_params[p].update(
            prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
            ref={"dist": "norm", "loc": 1.0, "scale": 0.02})

    DH = float(template.DH_over_rd_fid)
    DM = float(template.DM_over_rd_fid)
    DV = float(template.DV_over_rd_fid)

    from desilike.samplers import EmceeSampler
    sampler_kwargs = dict(nwalkers=nwalkers, seed=seed)
    if save_chain:
        sampler_kwargs["save_fn"] = save_chain
    sampler = EmceeSampler(likelihood, **sampler_kwargs)
    print(f"  emcee: {nwalkers} walkers × {max_iterations} iter, seed={seed}", flush=True)
    chain = sampler.run(max_iterations=max_iterations, check=False)[0]
    n_burn = int(burnin_frac * chain.shape[0])

    if apmode == "qiso":
        q = np.asarray(chain["qiso"][n_burn:]).ravel()
        sig_q = float(q.std(ddof=1))
        sig_DH, sig_DM, sig_DV = sig_q * DH, sig_q * DM, sig_q * DV
        n_post = int(q.size)
    else:
        qp = np.asarray(chain["qpar"][n_burn:]).ravel()
        qq = np.asarray(chain["qper"][n_burn:]).ravel()
        cov_qq = np.cov(qp, qq, ddof=1)
        grad = np.array([1.0 / 3.0, 2.0 / 3.0])
        sig_DH = float(qp.std(ddof=1)) * DH
        sig_DM = float(qq.std(ddof=1)) * DM
        sig_DV = float(np.sqrt(max(float(grad @ cov_qq @ grad), 0.0))) * DV
        n_post = int(qp.size)

    ref = _recon_sigmas(tracer, {"DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV})
    rDH, rDM, rDV = ref.get("DH_over_rs", np.nan), ref.get("DM_over_rs", np.nan), ref.get("DV_over_rs", np.nan)

    result = {
        "tracer": tracer, "dataset": dataset, "cov": cov, "apmode": apmode,
        "n_samples_post_burn": n_post,
        "sigma_DH_over_rd_mcmc": sig_DH, "sigma_DM_over_rd_mcmc": sig_DM,
        "sigma_DV_over_rd_mcmc": sig_DV,
        "sigma_DH_over_rd_recon": rDH, "sigma_DM_over_rd_recon": rDM,
        "sigma_DV_over_rd_recon": rDV,
        "ratio_DH": sig_DH / rDH if np.isfinite(rDH) and rDH > 0 else np.nan,
        "ratio_DM": sig_DM / rDM if np.isfinite(rDM) and rDM > 0 else np.nan,
        "ratio_DV": sig_DV / rDV if np.isfinite(rDV) and rDV > 0 else np.nan,
        "DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV,
        "alpha_prior": [alpha_low, alpha_high], "seed": seed,
        "nwalkers": nwalkers, "max_iterations": max_iterations,
    }
    _OUT_DIR.mkdir(exist_ok=True)
    suffix = "" if apmode == "qparqper" else f"_{apmode}"
    out_path = _OUT_DIR / f"{tracer}_{dataset}_{cov}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"  σ(DH/rd) = {sig_DH:.4f}   recon = {rDH:.4f}   M/D = {result['ratio_DH']:.3f}", flush=True)
    print(f"  σ(DM/rd) = {sig_DM:.4f}   recon = {rDM:.4f}   M/D = {result['ratio_DM']:.3f}", flush=True)
    print(f"  σ(DV/rd) = {sig_DV:.4f}   recon = {rDV:.4f}   M/D = {result['ratio_DV']:.3f}", flush=True)
    print(f"  saved {out_path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", nargs="+",
                    default=["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"],
                    choices=list(TRACER_CONFIGS.keys()))
    ap.add_argument("--cov", choices=["analytic", "bundle"], default="analytic",
                    help="analytic = pipeline Gaussian cov; bundle = DR1 RascalC cov_ξ substitution.")
    ap.add_argument("--dataset", choices=list(_DATASET_AREAS.keys()), default="dr1",
                    help="Survey area for --cov analytic (bundle forces dr1).")
    ap.add_argument("--apmode", choices=["auto", "qparqper", "qiso"], default="auto",
                    help="auto = qiso for BGS/QSO, qparqper otherwise.")
    ap.add_argument("--omega-m", type=float, default=_FID["Om"])
    ap.add_argument("--hrdrag", type=float, default=_FID["hrdrag"])
    ap.add_argument("--alpha-low", type=float, default=0.8)
    ap.add_argument("--alpha-high", type=float, default=1.2)
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-chain", action="store_true",
                    help="Save chain h5 to mcmc_results_fourier/{tracer}_chain.h5.")
    args = ap.parse_args()

    _OUT_DIR.mkdir(exist_ok=True)
    for t in args.tracers:
        chain_fn = str(_OUT_DIR / f"{t}_{args.dataset}_{args.cov}_chain.h5") if args.save_chain else None
        run(t, args.cov, args.dataset, args.omega_m, args.hrdrag,
            args.alpha_low, args.alpha_high, args.nwalkers, args.max_iterations,
            args.burnin_frac, args.seed, args.apmode, chain_fn)


if __name__ == "__main__":
    main()
