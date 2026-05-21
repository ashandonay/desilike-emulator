"""Sweep recon convention (smoothing radius + mode) and check ℓ=2 BB pull.

Tests hypothesis (3) from §31d: the BB ℓ=2 broadband residual on LRG2
(0.84σ no-BB → 0.33σ with BB) might come from a recon convention mismatch
between our pipeline and DESI's. We fix everything else at fiducial and
sweep:

  smoothing_radius ∈ {5, 10, 15 (DESI baseline), 20, 25} Mpc/h    (mode=recsym)
  mode             ∈ {recsym, reciso}                              (R=15)

For each configuration, compute:
  - per-ℓ RMS residual/σ (with and without BB)
  - BB ℓ=2 polynomial coefficients
  - max |BB·ℓ=2 contribution|_∞

If varying recon convention substantially changes the BB ℓ=2 pull, the
residual was a recon mismatch. If BB stays the same regardless, it's
something else (RSD broadband or template-form issue).
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402
from cosmoprimo.fftlog import PowerToCorrelation  # noqa: E402
from desilike.observables.galaxy_clustering import (  # noqa: E402
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.likelihoods import ObservablesGaussianLikelihood  # noqa: E402
from desilike.theories.galaxy_clustering import (  # noqa: E402
    DampedBAOWigglesTracerPowerSpectrumMultipoles,
)

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_BUNDLE = "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5"
_TRACER = "LRG2"
_DR1_AREA = 7500.0


def _load_bundle():
    with h5py.File(_LIK_DIR / _BUNDLE, "r") as f:
        cov = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        obs_s = [np.asarray(f[f"observable/{e}/s"][...], dtype=np.float64) for e in ["0", "2"]]
        obs_data = [np.asarray(f[f"observable/{e}/value"][...], dtype=np.float64) for e in ["0", "2"]]
        th_s = np.asarray(f["window/theory/0/s"][...], dtype=np.float64)
    return {"cov": cov, "W": W, "obs_s": obs_s, "obs_data": obs_data, "th_s": th_s}


def _build_info(smoothing_radius, mode):
    """Run the pipeline at fiducial with the requested recon convention.

    We mutate the TRACER_CONFIGS entry's smoothing_scale before calling
    build_bao_likelihood (which reads cfg["smoothing_scale"]), then build a
    fresh theory object with the requested mode and smoothing_radius and
    re-attach it to a fresh wide-k observable.
    """
    cfg = dict(TRACER_CONFIGS[_TRACER])
    cfg["smoothing_scale"] = float(smoothing_radius)
    TRACER_CONFIGS[_TRACER] = cfg  # build_bao_likelihood reads this

    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    n_tr = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}[_TRACER]
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})

    info = prep_covar.build_bao_likelihood(
        N_tracers=n_tr, theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=_TRACER, zrange=cfg["zrange"], z_eff=None, area=_DR1_AREA,
    )
    info["likelihood"](**info["params"])

    # The theory inside info was built with mode="recsym" hardcoded in
    # prep_covar.py. To sweep mode we construct a fresh theory using the
    # template that build_bao_likelihood already produced, then build a
    # wide-k observable wrapping that fresh theory.
    template = info["template"]
    fresh_theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
        template=template,
        mode=mode,
        smoothing_radius=float(smoothing_radius),
        model="fog-damping",
    )
    return info, fresh_theory


def _wide_xi(info, fresh_theory, ells=(0, 2)):
    k_min, k_max, nb = 0.0014, 0.5, 600
    dk = (k_max - k_min) / nb
    new_obs = TracerPowerSpectrumMultipolesObservable(
        data=info["params"],
        klim={int(e): [k_min, k_max, dk] for e in ells},
        theory=fresh_theory,
    )
    ObservablesGaussianLikelihood(
        observables=[new_obs], covariance=np.eye(nb * len(ells)),
    )(**info["params"])
    P_lk = np.asarray(new_obs.theory, dtype=np.float64)
    k_lin = np.asarray(new_obs.k[0], dtype=np.float64)

    # FFTLog
    k_log = np.logspace(np.log10(k_lin[0]), np.log10(k_lin[-1]), 1024)
    P_log = np.empty((len(ells), len(k_log)), dtype=np.float64)
    for ie in range(len(ells)):
        P_log[ie] = np.interp(k_log, k_lin, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=list(ells))
    s_out, xi_out = fft(P_log, extrap="log")
    return np.asarray(s_out), np.asarray(xi_out)


def _build_obs_pred(bundle, s_out, xi_out, ells=(0, 2)):
    target_s = bundle["th_s"]
    xi_by_ell = {0: np.zeros_like(target_s), 2: np.zeros_like(target_s), 4: np.zeros_like(target_s)}
    for ie, ell in enumerate(ells):
        s_e = s_out[ie] if s_out.ndim == 2 else s_out
        order = np.argsort(s_e)
        xi_by_ell[ell] = np.interp(target_s, s_e[order], xi_out[ie][order])
    th_vec = np.concatenate([xi_by_ell[0], xi_by_ell[2], xi_by_ell[4]])
    return bundle["W"] @ th_vec


def _bb_basis(obs_s, n_total):
    cols, cur = [], 0
    for s in obs_s:
        n = len(s)
        for p in [-2, -1, 0, 1, 2]:
            col = np.zeros(n_total)
            col[cur:cur + n] = s ** p
            cols.append(col)
        cur += n
    return np.array(cols).T


def _run_one(bundle, smoothing_radius, mode):
    info, fresh_theory = _build_info(smoothing_radius, mode)
    s_out, xi_out = _wide_xi(info, fresh_theory)
    pred = _build_obs_pred(bundle, s_out, xi_out)

    data = np.concatenate(bundle["obs_data"])
    cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
    C_inv = np.linalg.inv(cov_sym)
    resid = data - pred

    BB = _bb_basis(bundle["obs_s"], data.size)
    c_BB = np.linalg.solve(BB.T @ C_inv @ BB, BB.T @ C_inv @ resid)
    resid_BB = resid - BB @ c_BB

    sl0, sl2 = slice(0, 26), slice(26, 52)
    inv00 = np.linalg.inv(cov_sym[sl0, sl0])
    inv22 = np.linalg.inv(cov_sym[sl2, sl2])
    sigma = np.sqrt(np.diag(bundle["cov"]))

    chi2_0_no = float(resid[sl0] @ inv00 @ resid[sl0])
    chi2_2_no = float(resid[sl2] @ inv22 @ resid[sl2])
    chi2_0_bb = float(resid_BB[sl0] @ inv00 @ resid_BB[sl0])
    chi2_2_bb = float(resid_BB[sl2] @ inv22 @ resid_BB[sl2])
    rms0_no = float(np.sqrt(np.mean((resid[sl0] / sigma[sl0]) ** 2)))
    rms2_no = float(np.sqrt(np.mean((resid[sl2] / sigma[sl2]) ** 2)))
    rms0_bb = float(np.sqrt(np.mean((resid_BB[sl0] / sigma[sl0]) ** 2)))
    rms2_bb = float(np.sqrt(np.mean((resid_BB[sl2] / sigma[sl2]) ** 2)))
    bb_l2_max = float(np.max(np.abs((BB @ c_BB)[sl2])))
    return {
        "chi2_0_no": chi2_0_no, "chi2_2_no": chi2_2_no,
        "chi2_0_bb": chi2_0_bb, "chi2_2_bb": chi2_2_bb,
        "rms0_no": rms0_no, "rms2_no": rms2_no,
        "rms0_bb": rms0_bb, "rms2_bb": rms2_bb,
        "bb_l2_coeffs": c_BB[5:],
        "bb_l2_max": bb_l2_max,
    }


def main():
    bundle = _load_bundle()

    configs = []
    # Smoothing sweep at mode=recsym
    for R in [5.0, 10.0, 15.0, 20.0, 25.0]:
        configs.append(("recsym", R))
    # Mode toggle at R=15
    configs.append(("reciso", 15.0))

    rows = []
    for mode, R in configs:
        print(f"\nRunning  mode={mode}  R={R}...", flush=True)
        try:
            r = _run_one(bundle, R, mode)
            r["mode"] = mode; r["R"] = R
            rows.append(r)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

    print()
    print("=" * 112)
    print(f"{'mode':<8} {'R':>5}   {'ℓ=0 RMS':>9} {'ℓ=0 RMS(BB)':>12} {'ℓ=2 RMS':>9} {'ℓ=2 RMS(BB)':>12}   "
          f"{'χ²_0/26':>8} {'χ²_2/26':>8}   {'‖BB_2‖_∞':>10}")
    print("-" * 112)
    for r in rows:
        print(f"{r['mode']:<8} {r['R']:>5.1f}   {r['rms0_no']:>9.3f} {r['rms0_bb']:>12.3f} "
              f"{r['rms2_no']:>9.3f} {r['rms2_bb']:>12.3f}   "
              f"{r['chi2_0_no']/26:>8.3f} {r['chi2_2_no']/26:>8.3f}   {r['bb_l2_max']:>10.3e}")
    print()
    print("BB ℓ=2 polynomial coefficients ({s^-2, s^-1, s^0, s^1, s^2}):")
    for r in rows:
        c = r["bb_l2_coeffs"]
        print(f"  mode={r['mode']:<6} R={r['R']:>5.1f}: "
              f"[{c[0]:+10.3e}, {c[1]:+10.3e}, {c[2]:+10.3e}, {c[3]:+10.3e}, {c[4]:+10.3e}]")


if __name__ == "__main__":
    main()
