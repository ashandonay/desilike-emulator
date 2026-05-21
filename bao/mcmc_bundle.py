"""MCMC on the BUNDLE's cov_ξ substituted into any tracer's pipeline likelihood.

Generalization of mcmc_bundle_lrg2.py to all six DR1 tracers (now that all
six bundle files are local). Mechanism: hijack `likelihood.precision` with

  precision_F2 = M^T (cov_ξ_DR1)^{-1} M

where M = W · H_Hankel, then run emcee.

For DV-only tracers (BGS, QSO), the bundle has only ξ_0(s) — H_block zeros
out the pipeline's ℓ=2 contribution automatically through theory_ells.

Saves results to mcmc_results_bundle/{tracer}_dr1_bundle.json.
"""

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

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

import prep_covar as pc
from util import TRACER_CONFIGS
from scipy.special import spherical_jn

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_DR1_AREA = 7500.0

_BUNDLES = {
    "BGS":       "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1"}

_OUT_DIR = Path(__file__).resolve().parent / "mcmc_results_bundle"
_OUT_DIR.mkdir(exist_ok=True)


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


def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP.get(tracer, tracer)]


def _get_desi_std(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    sub = df[df["tracer"] == _NAME_MAP.get(tracer, tracer)]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


def run(tracer, nwalkers, max_iterations, burnin_frac, seed, alpha_low, alpha_high, save_chain,
        apmode="qparqper"):
    print(f"\n{'=' * 70}\n  {tracer}  (apmode={apmode})\n{'=' * 70}", flush=True)
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag_eff = pc._to_bao_cosmo_params(
        {**pc.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
    info = pc.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_DR1_AREA, apmode=apmode,
    )
    likelihood = info["likelihood"]
    template = info["template"]
    obs = info["observable"]

    # Build M = W·H from the bundle, then F2 precision
    cov_xi, W, th_ells, th_s = _load_bundle(tracer)
    k = np.asarray(obs.k[0], dtype=np.float64)
    dk = float(np.mean(np.diff(k)))
    pipe_ells = [int(l) for l in obs.ells]
    M = _build_M(W, th_ells, th_s, pipe_ells, k, dk)
    prec_F2 = _precision_from_cov_xi(cov_xi, M)
    print(f"  M shape {M.shape},  cov_ξ_DR1 {cov_xi.shape},  precision {prec_F2.shape}", flush=True)
    print(f"  bundle theory ells: {th_ells}, pipeline ells: {pipe_ells}", flush=True)

    likelihood.precision = np.asarray(prec_F2, dtype=np.float64)
    if hasattr(likelihood, "_precision_input"):
        likelihood._precision_input = likelihood.precision.copy()
    if hasattr(likelihood, "precision_hartlap2007"):
        likelihood.precision_hartlap2007 = likelihood.precision.copy()

    # In qparqper apmode the sampler has two AP params; in qiso it has one (qiso).
    ap_pname_pri = "qiso" if apmode == "qiso" else "qpar"
    ap_pname_sec = None if apmode == "qiso" else "qper"
    likelihood.all_params[ap_pname_pri].update(
        prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
        ref={"dist": "norm", "loc": 1.0, "scale": 0.02},
    )
    if ap_pname_sec is not None:
        likelihood.all_params[ap_pname_sec].update(
            prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
            ref={"dist": "norm", "loc": 1.0, "scale": 0.02},
        )

    DH_over_rd_fid = float(template.DH_over_rd_fid)
    DM_over_rd_fid = float(template.DM_over_rd_fid)
    z_eff_local = float(cfg["z_eff"])
    DV_over_rd_fid = (z_eff_local * DM_over_rd_fid ** 2 * DH_over_rd_fid) ** (1.0 / 3.0)  # BAO definition includes z

    from desilike.samplers import EmceeSampler
    sampler_kwargs = dict(nwalkers=nwalkers, seed=seed)
    if save_chain:
        sampler_kwargs["save_fn"] = save_chain
    sampler = EmceeSampler(likelihood, **sampler_kwargs)
    print(f"  emcee: {nwalkers} walkers × {max_iterations} iter, seed={seed}", flush=True)
    chains = sampler.run(max_iterations=max_iterations, check=False)

    chain = chains[0]
    n_total = chain.shape[0]
    n_burn = int(burnin_frac * n_total)
    if apmode == "qiso":
        # Single isotropic AP parameter: qpar = qper = qiso. σ(DH/rd), σ(DM/rd)
        # and σ(DV/rd) all scale linearly with σ(qiso) since each distance
        # is just qiso * fiducial.
        q_iso = np.asarray(chain["qiso"][n_burn:]).ravel()
        q_par = q_iso.copy()
        q_per = q_iso.copy()
        sig_qpar = float(q_iso.std(ddof=1))
        sig_qper = sig_qpar
        sig_DH = sig_qpar * DH_over_rd_fid
        sig_DM = sig_qper * DM_over_rd_fid
        sig_DV = sig_qpar * DV_over_rd_fid
    else:
        q_par = np.asarray(chain["qpar"][n_burn:]).ravel()
        q_per = np.asarray(chain["qper"][n_burn:]).ravel()
        sig_qpar = float(q_par.std(ddof=1))
        sig_qper = float(q_per.std(ddof=1))
        cov_qq = np.cov(q_par, q_per, ddof=1)
        grad = np.array([1.0 / 3.0, 2.0 / 3.0])
        var_lnDV = float(grad @ cov_qq @ grad)
        sig_DH = sig_qpar * DH_over_rd_fid
        sig_DM = sig_qper * DM_over_rd_fid
        sig_DV = float(np.sqrt(max(var_lnDV, 0.0))) * DV_over_rd_fid

    desi = _get_desi_std(tracer)
    desi_DH = desi.get("DH_over_rs", float("nan"))
    desi_DM = desi.get("DM_over_rs", float("nan"))
    desi_DV = desi.get("DV_over_rs", float("nan"))

    result = {
        "tracer": tracer,
        "dataset": "dr1",
        "source": "bundle_F2_substitution",
        "apmode": apmode,
        "n_samples_post_burn": int(q_par.size),
        "sigma_qpar": sig_qpar,
        "sigma_qper": sig_qper,
        "sigma_DH_over_rd_mcmc": sig_DH,
        "sigma_DM_over_rd_mcmc": sig_DM,
        "sigma_DV_over_rd_mcmc": sig_DV,
        "sigma_DH_over_rd_data": desi_DH,
        "sigma_DM_over_rd_data": desi_DM,
        "sigma_DV_over_rd_data": desi_DV,
        "DH_over_rd_fid": DH_over_rd_fid,
        "DM_over_rd_fid": DM_over_rd_fid,
        "DV_over_rd_fid": DV_over_rd_fid,
        "ratio_DH": sig_DH / desi_DH if (np.isfinite(desi_DH) and desi_DH > 0) else float("nan"),
        "ratio_DM": sig_DM / desi_DM if (np.isfinite(desi_DM) and desi_DM > 0) else float("nan"),
        "ratio_DV": sig_DV / desi_DV if (np.isfinite(desi_DV) and desi_DV > 0) else float("nan"),
        "alpha_prior": [alpha_low, alpha_high],
        "seed": seed,
        "max_iterations": max_iterations,
        "nwalkers": nwalkers,
    }
    suffix = "" if apmode == "qparqper" else f"_{apmode}"
    out_path = _OUT_DIR / f"{tracer}_dr1_bundle{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"  σ(DH/rd) = {sig_DH:.4f}    DESI = {desi_DH:.4f}    M/D = {result['ratio_DH']:.3f}", flush=True)
    print(f"  σ(DM/rd) = {sig_DM:.4f}    DESI = {desi_DM:.4f}    M/D = {result['ratio_DM']:.3f}", flush=True)
    print(f"  σ(DV/rd) = {sig_DV:.4f}    DESI = {desi_DV:.4f}    M/D = {result['ratio_DV']:.3f}", flush=True)
    print(f"  saved {out_path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracers", nargs="+",
                    default=["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"])
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--max-iterations", type=int, default=2500)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-low", type=float, default=0.8)
    ap.add_argument("--alpha-high", type=float, default=1.2)
    ap.add_argument("--save-chain", action="store_true",
                    help="If set, save chain h5 to mcmc_results_bundle/{tracer}_chain.h5")
    ap.add_argument("--apmode", choices=["qparqper", "qiso", "qisoqap"],
                    default="qparqper",
                    help="AP parameterization: qparqper (default, anisotropic) or "
                         "qiso (isotropic, DESI's standard for sparse tracers BGS+QSO)")
    args = ap.parse_args()
    for t in args.tracers:
        chain_fn = str(_OUT_DIR / f"{t}_chain.h5") if args.save_chain else None
        run(t, args.nwalkers, args.max_iterations, args.burnin_frac,
            args.seed, args.alpha_low, args.alpha_high, chain_fn,
            apmode=args.apmode)


if __name__ == "__main__":
    main()
