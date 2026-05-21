"""Native ξ-theory MCMC on bundle data — no FFTLog projection.

Companion to mcmc_bundle_xi.py. Replaces the FFTLog-from-P route with a
direct call to desilike's DampedBAOWigglesTracerCorrelationFunctionMultipoles
at the bundle's theory s grid. Compares σ_qiso to:

  - mcmc_bundle_xi.py (FFTLog-based, native-ξ MCMC):  σ = 0.0257 for BGS
  - DESI bao-recon stat-only file (post-marg α-cov): σ = 0.0205 for BGS
  - Cramér-Rao conditional bound from bundle:        σ = 0.0235 for BGS

If this script (which mirrors DESI's likely theory path more closely)
gives σ ≈ 0.020, then the FFTLog projection in mcmc_bundle_xi.py was
the source of the residual gap and we can swap it out.

If it gives σ ≈ 0.025, then the FFTLog vs native difference is not the
issue — and DESI's tighter σ must come from a different template choice
(no-wiggle splitting, fiducial cosmology) or additional information not
in the bundle.

No modification to pipeline code (prep_covar.py, util.py, hod.yaml, etc.).
"""
from __future__ import annotations
import argparse, json, sys, warnings
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
from desilike.theories.galaxy_clustering import (  # noqa: E402
    DampedBAOWigglesTracerCorrelationFunctionMultipoles,
)

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_DR1_AREA = 7500.0
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1"}
_OUT_DIR = Path(__file__).resolve().parent / "mcmc_results_bundle_native"
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


def _build_native_theory(tracer, apmode):
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
    info = prep_covar.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_DR1_AREA, apmode=apmode,
    )
    info["likelihood"](**info["params"])
    template = info["template"]
    bundle = _load_bundle(tracer)
    smoothing = float(cfg["smoothing_scale"])

    # Use bundle's theory s grid; output ells=0,2,4 to match bundle W layout.
    native = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=bundle["th_s"][0], ells=tuple(bundle["th_ells"]),
        template=template, mode="recsym", smoothing_radius=smoothing,
        model="fog-damping",
    )
    return info, native, bundle


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
    C_inv_r = C_inv @ residual
    C_inv_BB = C_inv @ BB
    A = BB.T @ C_inv_BB
    b = BB.T @ C_inv_r
    return float(residual @ C_inv_r - b @ np.linalg.solve(A, b))


def _predict(native, bundle, params):
    # Set theory params (only those native exposes)
    native_param_names = {p.name for p in native.all_params}
    p = {k: v for k, v in params.items() if k in native_param_names}
    # Pin all BB params on the native theory to zero (we Schur-marg them externally)
    for q in native.all_params:
        if (q.name.startswith("al") or q.name.startswith("bl")):
            p[q.name] = 0.0
    native(**p)
    xi = np.asarray(native.corr, dtype=np.float64)  # (n_ells, n_th_s)
    xi_flat = np.concatenate([xi[i] for i in range(xi.shape[0])])
    return bundle["W"] @ xi_flat


def make_log_prob(info, native, bundle, apmode,
                  bb_powers=(-3, -2, -1, 0, 1),
                  alpha_low=0.8, alpha_high=1.2):
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

    prior_widths = {"b1": 1.0, "dbeta": 0.5, "sigmas": 4.0, "sigmapar": 4.0, "sigmaper": 3.0}
    sigma_hard_max = {"sigmas": 8.0, "sigmapar": 12.0, "sigmaper": 8.0}

    def log_prior(theta):
        d = dict(zip(param_names, theta))
        for q in ("qiso", "qpar", "qper"):
            if q in d and not (alpha_low < d[q] < alpha_high):
                return -np.inf
        for p, hi in sigma_hard_max.items():
            if not (0.1 < d[p] < hi): return -np.inf
        if not (0.2 < d["b1"] < 5.0): return -np.inf
        if not (-1.5 < d["dbeta"] < 1.5): return -np.inf
        lp = 0.0
        for p, w in prior_widths.items():
            if p in d:
                lp += -0.5 * ((d[p] - fid[p]) / w) ** 2
        return lp

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        params = {**info["params"]}
        for name, val in zip(param_names, theta):
            params[name] = float(val)
        try:
            pred = _predict(native, bundle, params)
            residual = data - pred
            chi2 = _chi2_marg(residual, C_inv, BB)
        except Exception:
            return -np.inf
        if not np.isfinite(chi2):
            return -np.inf
        return lp - 0.5 * chi2

    return log_prob, param_names, fid


def run(tracer, apmode, nwalkers, max_iterations, burnin_frac, seed,
        alpha_low, alpha_high):
    print(f"\n{'=' * 70}\n  {tracer}  (apmode={apmode}, NATIVE ξ-theory)\n{'=' * 70}", flush=True)
    info, native, bundle = _build_native_theory(tracer, apmode)
    log_prob, param_names, fid = make_log_prob(
        info, native, bundle, apmode, alpha_low=alpha_low, alpha_high=alpha_high)
    ndim = len(param_names)
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
    chain = sampler.get_chain()
    n_burn = int(burnin_frac * max_iterations)
    flat = chain[n_burn:].reshape(-1, ndim)
    print(f"  acceptance: {np.mean(sampler.acceptance_fraction):.3f}", flush=True)
    print(f"  chain means / stds:", flush=True)
    for i, p in enumerate(param_names):
        m = float(np.mean(flat[:, i])); s = float(np.std(flat[:, i], ddof=1))
        print(f"    {p:<10} mean={m:7.4f}  std={s:7.4f}  fid={fid[p]:7.4f}", flush=True)

    template = info["template"]
    z = float(TRACER_CONFIGS[tracer]["z_eff"])
    DH = float(template.DH_over_rd_fid)
    DM = float(template.DM_over_rd_fid)
    DV = (z * DM ** 2 * DH) ** (1.0 / 3.0)
    if apmode == "qiso":
        i_q = param_names.index("qiso")
        sig_q = float(flat[:, i_q].std(ddof=1))
        sig_DH = sig_q * DH; sig_DM = sig_q * DM; sig_DV = sig_q * DV
    else:
        ip = param_names.index("qpar"); iq = param_names.index("qper")
        qpar = flat[:, ip]; qper = flat[:, iq]
        sig_qpar = float(qpar.std(ddof=1)); sig_qper = float(qper.std(ddof=1))
        cov_qq = np.cov(qpar, qper, ddof=1)
        grad = np.array([1.0 / 3.0, 2.0 / 3.0])
        var_lnDV = float(grad @ cov_qq @ grad)
        sig_DH = sig_qpar * DH; sig_DM = sig_qper * DM
        sig_DV = float(np.sqrt(max(var_lnDV, 0.0))) * DV

    desi = _get_desi_std(tracer)
    desi_DV = desi.get("DV_over_rs", float("nan"))
    desi_DH = desi.get("DH_over_rs", float("nan"))
    desi_DM = desi.get("DM_over_rs", float("nan"))
    def _r(s, d): return s / d if (np.isfinite(d) and d > 0) else float("nan")

    print(f"  σ(DH/rd) = {sig_DH:.4f}    DESI = {desi_DH:.4f}    M/D = {_r(sig_DH, desi_DH):.3f}", flush=True)
    print(f"  σ(DM/rd) = {sig_DM:.4f}    DESI = {desi_DM:.4f}    M/D = {_r(sig_DM, desi_DM):.3f}", flush=True)
    print(f"  σ(DV/rd) = {sig_DV:.4f}    DESI = {desi_DV:.4f}    M/D = {_r(sig_DV, desi_DV):.3f}", flush=True)

    result = {
        "tracer": tracer, "apmode": apmode, "source": "bundle_native_xi_theory",
        "sig_DH": sig_DH, "sig_DM": sig_DM, "sig_DV": sig_DV,
        "desi_DH": desi_DH, "desi_DM": desi_DM, "desi_DV": desi_DV,
        "ratio_DH": _r(sig_DH, desi_DH), "ratio_DM": _r(sig_DM, desi_DM), "ratio_DV": _r(sig_DV, desi_DV),
    }
    suffix = "" if apmode == "qparqper" else f"_{apmode}"
    out_path = _OUT_DIR / f"{tracer}_dr1_native_theory{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracers", nargs="+", default=["BGS"])
    ap.add_argument("--apmode", choices=["qparqper", "qiso"], default="qiso")
    ap.add_argument("--nwalkers", type=int, default=48)
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-low", type=float, default=0.8)
    ap.add_argument("--alpha-high", type=float, default=1.2)
    args = ap.parse_args()
    for t in args.tracers:
        try:
            run(t, args.apmode, args.nwalkers, args.max_iterations,
                args.burnin_frac, args.seed, args.alpha_low, args.alpha_high)
        except Exception as e:
            print(f"[{t}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
