"""Native-ξ MCMC on bundle data + bundle cov + bundle W (no F2 projection).

§31g.nonus open follow-up. The existing mcmc_bundle.py runs MCMC by F2-
projecting the bundle cov from ξ-space to P-space via
precision_P = M^T C_ξ^{-1} M, then overriding the P-space likelihood's
precision. For DV-only sparse-tracer fits (BGS, QSO), this projection
is rank-deficient (rank 26 of 112), which inflates σ_DV.

This script evaluates the pipeline theory in P-space, FFTLog-projects
to ξ-space at the bundle's theory s grid, applies the bundle's W to
get the windowed ξ on the data s grid, and compares to bundle data
with bundle cov — a fully ξ-space-native MCMC. Equivalent to the
Tier-3 Fisher (test_tier3_all_tracers.py) but with emcee instead of
analytic Schur over the qpar/qper plane.

BB nuisances {a^ℓ_n, n∈[-3,...,1]} are analytically Schur-marginalized
inside the log-likelihood (they enter linearly), keeping the MCMC
dimensionality to 6-7 nonlinear params.

No modification to prep_covar.py / mcmc_bao.py / util.py / hod.yaml.
Wholly additive: this file + a results dir.

Usage:
  python mcmc_bundle_xi.py --tracers BGS QSO --apmode qiso --max-iterations 4000
  python mcmc_bundle_xi.py --tracers LRG2 --apmode qparqper --max-iterations 4000
"""
from __future__ import annotations
import argparse, json, sys, warnings, os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import emcee

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402
from cosmoprimo.fftlog import PowerToCorrelation  # noqa: E402
from desilike.observables.galaxy_clustering import (  # noqa: E402
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.likelihoods import ObservablesGaussianLikelihood  # noqa: E402

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_DR1_AREA = 7500.0
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1"}
_OUT_DIR = Path(__file__).resolve().parent / "mcmc_results_bundle_xi"
_OUT_DIR.mkdir(exist_ok=True)

_BUNDLES = {
    "BGS":       "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}


def _load_bundle(tracer):
    fpath = _LIK_DIR / _BUNDLES[tracer]
    with h5py.File(fpath, "r") as f:
        cov = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        obs_ells, obs_s, obs_data = [], [], []
        for k in sorted(f["observable"].keys()):
            if k.isdigit():
                obs_ells.append(int(f[f"observable/{k}/meta/ell"][()]))
                obs_s.append(np.asarray(f[f"observable/{k}/s"][...], dtype=np.float64))
                obs_data.append(np.asarray(f[f"observable/{k}/value"][...], dtype=np.float64))
        th_ells, th_s = [], []
        for k in sorted(f["window/theory"].keys()):
            if k.isdigit():
                th_ells.append(int(f[f"window/theory/{k}/meta/ell"][()]))
                th_s.append(np.asarray(f[f"window/theory/{k}/s"][...], dtype=np.float64))
    return {"cov": cov, "W": W,
            "obs_ells": obs_ells, "obs_s": obs_s, "obs_data": obs_data,
            "th_ells": th_ells, "th_s": th_s}


def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    return {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}[_NAME_MAP.get(tracer, tracer)]


def _get_desi_std(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    sub = df[df["tracer"] == _NAME_MAP.get(tracer, tracer)]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


def _build_wide_lik(tracer, apmode="qparqper"):
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
    info = prep_covar.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_DR1_AREA, apmode=apmode,
    )
    info["likelihood"](**info["params"])
    theory = None
    for c in info["observable"].runtime_info.pipeline.calculators:
        if type(c).__name__.startswith("DampedBAOWigglesTracerPowerSpectrumMultipoles"):
            theory = c; break
    k_min, k_max, nb = 0.0014, 0.5, 600
    dk = (k_max - k_min) / nb
    new_obs = TracerPowerSpectrumMultipolesObservable(
        data=info["params"],
        klim={0: [k_min, k_max, dk], 2: [k_min, k_max, dk]},
        theory=theory,
    )
    wide_lik = ObservablesGaussianLikelihood(
        observables=[new_obs], covariance=np.eye(2 * nb),
    )
    wide_lik(**info["params"])
    return info, wide_lik, new_obs, np.asarray(new_obs.k[0], dtype=np.float64)


def _xi_at_bundle_grid(wide_obs, k_lin, bundle, ells=(0, 2)):
    P_lk = np.asarray(wide_obs.theory, dtype=np.float64)
    k_log = np.logspace(np.log10(k_lin[0]), np.log10(k_lin[-1]), 1024)
    P_log = np.empty((len(ells), len(k_log)), dtype=np.float64)
    for ie in range(len(ells)):
        P_log[ie] = np.interp(k_log, k_lin, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=list(ells))
    s_out, xi_out = fft(P_log, extrap="log")
    s_out = np.asarray(s_out); xi_out = np.asarray(xi_out)
    target_s = bundle["th_s"][0]
    xi_th_blocks = []
    for ell in [0, 2, 4]:
        if ell in ells:
            ie = list(ells).index(ell)
            s_e = s_out[ie] if s_out.ndim == 2 else s_out
            order = np.argsort(s_e)
            xi_th_blocks.append(np.interp(target_s, s_e[order], xi_out[ie][order]))
        else:
            xi_th_blocks.append(np.zeros_like(target_s))
    by_ell = {0: xi_th_blocks[0], 2: xi_th_blocks[1], 4: xi_th_blocks[2]}
    return np.concatenate([by_ell[e] for e in bundle["th_ells"]])


def _bb_basis(obs_ells, obs_s, n_total, powers=(-3, -2, -1, 0, 1)):
    cols = []
    cur = 0
    for _, s in zip(obs_ells, obs_s):
        n = len(s)
        for p in powers:
            col = np.zeros(n_total)
            col[cur:cur + n] = s ** p
            cols.append(col)
        cur += n
    return np.array(cols).T


def _chi2_marg(residual, C_inv, BB):
    """Schur-marginalized χ² over BB (Gaussian linear nuisance)."""
    C_inv_r = C_inv @ residual
    C_inv_BB = C_inv @ BB
    A = BB.T @ C_inv_BB
    b = BB.T @ C_inv_r
    return float(residual @ C_inv_r - b @ np.linalg.solve(A, b))


def make_log_prob(info, wide_lik, wide_obs, k_lin, bundle, apmode,
                  bb_powers=(-3, -2, -1, 0, 1),
                  alpha_low=0.8, alpha_high=1.2,
                  prior_centers=None, prior_widths_override=None):
    data = np.concatenate(bundle["obs_data"])
    n_data = data.size
    cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
    C_inv = np.linalg.inv(cov_sym)
    BB = _bb_basis(bundle["obs_ells"], bundle["obs_s"], n_data, powers=bb_powers)

    if apmode == "qiso":
        param_names = ["qiso", "b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]
    else:
        param_names = ["qpar", "qper", "b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]

    fid = {}
    for p in param_names:
        if p in info["params"]:
            fid[p] = float(info["params"][p])
        else:
            for q in info["likelihood"].all_params:
                if q.name == p:
                    fid[p] = float(q.value); break

    # Loose Gaussian priors on nuisances; uniform on q's. Override via CLI:
    # --desi-priors flips Σ/σ_s widths to DESI Adame+24 spec (~1 for Σ, 2 for σ_s).
    prior_widths = {"b1": 1.0, "dbeta": 0.5, "sigmas": 4.0, "sigmapar": 4.0, "sigmaper": 3.0}
    if prior_widths_override:
        prior_widths.update(prior_widths_override)
    prior_means = {p: fid[p] for p in prior_widths if p in fid}
    if prior_centers:
        prior_means.update(prior_centers)

    sigma_hard_max = {"sigmas": 8.0, "sigmapar": 12.0, "sigmaper": 8.0}
    b1_hard = (0.2, 5.0)
    dbeta_hard = (-1.5, 1.5)

    def log_prior(theta):
        d = dict(zip(param_names, theta))
        for q in ("qiso", "qpar", "qper"):
            if q in d and not (alpha_low < d[q] < alpha_high):
                return -np.inf
        for p, hi in sigma_hard_max.items():
            if not (0.1 < d[p] < hi): return -np.inf
        if not (b1_hard[0] < d["b1"] < b1_hard[1]): return -np.inf
        if not (dbeta_hard[0] < d["dbeta"] < dbeta_hard[1]): return -np.inf
        lp = 0.0
        for p, w in prior_widths.items():
            if p in d:
                lp += -0.5 * ((d[p] - prior_means.get(p, fid[p])) / w) ** 2
        return lp

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        params = {**info["params"]}
        for name, val in zip(param_names, theta):
            params[name] = float(val)
        try:
            wide_lik(**params)
            xi_th_vec = _xi_at_bundle_grid(wide_obs, k_lin, bundle)
            pred = bundle["W"] @ xi_th_vec
            residual = data - pred
            chi2 = _chi2_marg(residual, C_inv, BB)
        except Exception:
            return -np.inf
        if not np.isfinite(chi2):
            return -np.inf
        return lp - 0.5 * chi2

    return log_prob, param_names, fid


def run(tracer, apmode, nwalkers, max_iterations, burnin_frac, seed,
        alpha_low, alpha_high, save_chain,
        prior_centers=None, prior_widths_override=None):
    print(f"\n{'=' * 70}\n  {tracer}  (apmode={apmode})\n{'=' * 70}", flush=True)
    info, wide_lik, wide_obs, k_lin = _build_wide_lik(tracer, apmode=apmode)
    bundle = _load_bundle(tracer)

    log_prob, param_names, fid = make_log_prob(
        info, wide_lik, wide_obs, k_lin, bundle, apmode,
        alpha_low=alpha_low, alpha_high=alpha_high,
        prior_centers=prior_centers, prior_widths_override=prior_widths_override,
    )
    ndim = len(param_names)

    # Initial walker spread: tight Gaussian around fid for nuisances, uniform for q's.
    rng = np.random.default_rng(seed)
    p0 = np.empty((nwalkers, ndim), dtype=np.float64)
    for i, p in enumerate(param_names):
        if p in ("qiso", "qpar", "qper"):
            p0[:, i] = 1.0 + 0.02 * rng.standard_normal(nwalkers)
        elif p in ("sigmas", "sigmapar", "sigmaper"):
            p0[:, i] = np.maximum(fid[p] + 0.2 * rng.standard_normal(nwalkers), 0.2)
        else:
            p0[:, i] = fid[p] + 0.05 * rng.standard_normal(nwalkers)

    print(f"  ndim={ndim}, walkers={nwalkers}, iter={max_iterations}, seed={seed}", flush=True)
    print(f"  params: {param_names}", flush=True)
    print(f"  fid:    {[f'{fid[p]:.3f}' for p in param_names]}", flush=True)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, max_iterations, progress=False)

    chain = sampler.get_chain()  # (n_iter, n_walkers, ndim)
    n_burn = int(burnin_frac * max_iterations)
    flat = chain[n_burn:].reshape(-1, ndim)
    print(f"  acceptance: {np.mean(sampler.acceptance_fraction):.3f}", flush=True)
    print(f"  chain means / stds (post-burn):", flush=True)
    for i, p in enumerate(param_names):
        m = float(np.mean(flat[:, i])); s = float(np.std(flat[:, i], ddof=1))
        print(f"    {p:<10} mean={m:7.4f}  std={s:7.4f}  fid={fid[p]:7.4f}", flush=True)

    template = info["template"]
    DH = float(template.DH_over_rd_fid)
    DM = float(template.DM_over_rd_fid)
    DV = (DM ** 2 * DH) ** (1.0 / 3.0)

    if apmode == "qiso":
        i_q = param_names.index("qiso")
        q = flat[:, i_q]
        sig_q = float(q.std(ddof=1))
        sig_DH = sig_q * DH
        sig_DM = sig_q * DM
        sig_DV = sig_q * DV
        cov_qq = None
    else:
        ip = param_names.index("qpar"); iq = param_names.index("qper")
        qpar = flat[:, ip]; qper = flat[:, iq]
        sig_qpar = float(qpar.std(ddof=1)); sig_qper = float(qper.std(ddof=1))
        cov_qq = np.cov(qpar, qper, ddof=1)
        grad = np.array([1.0 / 3.0, 2.0 / 3.0])
        var_lnDV = float(grad @ cov_qq @ grad)
        sig_DH = sig_qpar * DH
        sig_DM = sig_qper * DM
        sig_DV = float(np.sqrt(max(var_lnDV, 0.0))) * DV

    desi = _get_desi_std(tracer)
    desi_DH = desi.get("DH_over_rs", float("nan"))
    desi_DM = desi.get("DM_over_rs", float("nan"))
    desi_DV = desi.get("DV_over_rs", float("nan"))

    def _ratio(s, d):
        return s / d if (np.isfinite(d) and d > 0) else float("nan")

    result = {
        "tracer": tracer, "dataset": "dr1", "source": "bundle_native_xi",
        "apmode": apmode, "n_samples_post_burn": int(flat.shape[0]),
        "acceptance": float(np.mean(sampler.acceptance_fraction)),
        "sigma_DH_over_rd_mcmc": sig_DH,
        "sigma_DM_over_rd_mcmc": sig_DM,
        "sigma_DV_over_rd_mcmc": sig_DV,
        "sigma_DH_over_rd_data": desi_DH,
        "sigma_DM_over_rd_data": desi_DM,
        "sigma_DV_over_rd_data": desi_DV,
        "ratio_DH": _ratio(sig_DH, desi_DH),
        "ratio_DM": _ratio(sig_DM, desi_DM),
        "ratio_DV": _ratio(sig_DV, desi_DV),
        "DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV,
        "alpha_prior": [alpha_low, alpha_high],
        "seed": seed, "max_iterations": max_iterations, "nwalkers": nwalkers,
    }

    print(f"  σ(DH/rd) = {sig_DH:.4f}    DESI = {desi_DH:.4f}    M/D = {result['ratio_DH']:.3f}", flush=True)
    print(f"  σ(DM/rd) = {sig_DM:.4f}    DESI = {desi_DM:.4f}    M/D = {result['ratio_DM']:.3f}", flush=True)
    print(f"  σ(DV/rd) = {sig_DV:.4f}    DESI = {desi_DV:.4f}    M/D = {result['ratio_DV']:.3f}", flush=True)

    suffix = "" if apmode == "qparqper" else f"_{apmode}"
    out_path = _OUT_DIR / f"{tracer}_dr1_bundle_xi{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    if save_chain:
        np.save(out_path.with_suffix(".chain.npy"), chain[n_burn:])
    print(f"  saved {out_path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracers", nargs="+",
                    default=["BGS", "LRG2", "QSO"])
    ap.add_argument("--apmode", choices=["qparqper", "qiso"], default="qparqper")
    ap.add_argument("--nwalkers", type=int, default=48)
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-low", type=float, default=0.8)
    ap.add_argument("--alpha-high", type=float, default=1.2)
    ap.add_argument("--save-chain", action="store_true")
    ap.add_argument("--desi-priors", action="store_true",
                    help="Use DESI Adame+24 prior widths on Σ/σ_s and per-tracer "
                         "canonical Σ centers (matches published likelihood definition)")
    args = ap.parse_args()

    # DESI Adame+24 Table 1: post-recon canonical Σ + Gaussian prior widths.
    # BGS uses larger Σ (harder to reconstruct).
    DESI_PRIORS = {
        "BGS":       {"centers": {"sigmapar": 8.0, "sigmaper": 4.5, "sigmas": 2.0},
                      "widths":  {"sigmapar": 1.0, "sigmaper": 1.0, "sigmas": 2.0}},
        "LRG1":      {"centers": {"sigmapar": 6.0, "sigmaper": 3.0, "sigmas": 2.0},
                      "widths":  {"sigmapar": 1.0, "sigmaper": 1.0, "sigmas": 2.0}},
        "LRG2":      {"centers": {"sigmapar": 6.0, "sigmaper": 3.0, "sigmas": 2.0},
                      "widths":  {"sigmapar": 1.0, "sigmaper": 1.0, "sigmas": 2.0}},
        "LRG3_ELG1": {"centers": {"sigmapar": 6.0, "sigmaper": 3.0, "sigmas": 2.0},
                      "widths":  {"sigmapar": 1.0, "sigmaper": 1.0, "sigmas": 2.0}},
        "ELG2":      {"centers": {"sigmapar": 6.0, "sigmaper": 3.0, "sigmas": 2.0},
                      "widths":  {"sigmapar": 1.0, "sigmaper": 1.0, "sigmas": 2.0}},
        "QSO":       {"centers": {"sigmapar": 9.0, "sigmaper": 6.0, "sigmas": 2.0},
                      "widths":  {"sigmapar": 2.0, "sigmaper": 2.0, "sigmas": 2.0}},
    }

    for t in args.tracers:
        pc_ = pw_ = None
        if args.desi_priors and t in DESI_PRIORS:
            pc_ = DESI_PRIORS[t]["centers"]
            pw_ = DESI_PRIORS[t]["widths"]
            print(f"[{t}] using DESI-spec priors: centers={pc_}, widths={pw_}")
        try:
            run(t, args.apmode, args.nwalkers, args.max_iterations, args.burnin_frac,
                args.seed, args.alpha_low, args.alpha_high, args.save_chain,
                prior_centers=pc_, prior_widths_override=pw_)
        except Exception as e:
            print(f"[{t}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
