"""ξ_4 window-leakage test for the LRG2 BB ℓ=2 broadband residual.

§31d showed our pipeline at fiducial fits DESI's ξ_0(s) data well (RMS 0.75σ) but
leaves a coherent 0.84σ broadband residual on ξ_2 that DESI's standard BB
polynomial absorbs almost completely (0.33σ after BB). The top hypothesis was
that we zero out the pipeline's ξ_4, but the bundle's window W (52×600) mixes
theory ℓ ∈ {0,2,4} → observed ℓ ∈ {0,2}, so the missing ξ_4 leaks into
observed ξ_2 as a smooth broadband.

This script runs two cases on the same fiducial pipeline:

  case A:  ξ_4 = 0          (current §31d baseline)
  case B:  ξ_4 from pipeline (set_k_mu(ells=(0,2,4)) on the wide observable)

For each case we compute:
  - RMS residual/σ per ℓ
  - marginal χ²/dof per ℓ
  - BB-fit ℓ=2 polynomial coefficients (these should shrink if ξ_4 was the
    leakage source — BB ℓ=2 is no longer absorbing a missing-multipole signal)

If case B → small ℓ=2 residual *without* BB: leakage confirmed.
If case B looks like case A: it's something else (RSD broadband, recon convention).
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402
from cosmoprimo.fftlog import PowerToCorrelation  # noqa: E402
from desilike.observables.galaxy_clustering import (  # noqa: E402
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.likelihoods import ObservablesGaussianLikelihood  # noqa: E402

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_BUNDLE = "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5"
_TRACER = "LRG2"
_DR1_AREA = 7500.0


def _load_bundle():
    with h5py.File(_LIK_DIR / _BUNDLE, "r") as f:
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


def _build_pipeline_xi(ells_pipe):
    """Run the pipeline at fiducial, build a wide-k observable on the given ells,
    FFTLog to ξ_ℓ(s) on the bundle's theory s-grid for ℓ ∈ {0,2,4}.
    Missing ells are returned as zeros.
    """
    bundle = _load_bundle()
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    n_tr = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}[_TRACER]
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
    info = prep_covar.build_bao_likelihood(
        N_tracers=n_tr, theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=_TRACER, zrange=TRACER_CONFIGS[_TRACER]["zrange"],
        z_eff=None, area=_DR1_AREA,
    )
    info["likelihood"](**info["params"])

    # Find the outer BAO theory calculator
    theory = None
    for c in info["observable"].runtime_info.pipeline.calculators:
        if type(c).__name__.startswith("DampedBAOWigglesTracerPowerSpectrumMultipoles"):
            theory = c
            break

    # Build wide-k observable on the requested ells
    k_min, k_max, nb = 0.0014, 0.5, 600
    dk = (k_max - k_min) / nb
    new_obs = TracerPowerSpectrumMultipolesObservable(
        data=info["params"],
        klim={int(ell): [k_min, k_max, dk] for ell in ells_pipe},
        theory=theory,
    )
    n_tot = nb * len(ells_pipe)
    ObservablesGaussianLikelihood(
        observables=[new_obs], covariance=np.eye(n_tot),
    )(**info["params"])
    P_lk = np.asarray(new_obs.theory, dtype=np.float64)
    k_lin = np.asarray(new_obs.k[0], dtype=np.float64)
    pipe_ells_in_obs = [int(l) for l in new_obs.ells]

    # FFTLog to ξ on a log-s grid, then interpolate to bundle's theory s-grid
    k_log = np.logspace(np.log10(k_lin[0]), np.log10(k_lin[-1]), 1024)
    P_log = np.empty((len(pipe_ells_in_obs), len(k_log)), dtype=np.float64)
    for ie, _ in enumerate(pipe_ells_in_obs):
        P_log[ie] = np.interp(k_log, k_lin, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=list(pipe_ells_in_obs))
    s_out, xi_out = fft(P_log, extrap="log")
    s_out = np.asarray(s_out)
    xi_out = np.asarray(xi_out)

    # Interpolate to bundle's theory s-grid for each of {0,2,4}; missing → zeros
    target_s = bundle["th_s"][0]
    xi_by_ell = {}
    for ell in (0, 2, 4):
        if ell in pipe_ells_in_obs:
            ie = pipe_ells_in_obs.index(ell)
            s_e = s_out[ie] if s_out.ndim == 2 else s_out
            order = np.argsort(s_e)
            xi_by_ell[ell] = np.interp(target_s, s_e[order], xi_out[ie][order])
        else:
            xi_by_ell[ell] = np.zeros_like(target_s)
    return bundle, xi_by_ell, target_s


def _build_th_vec(bundle, xi_by_ell):
    blocks = [xi_by_ell[ell] for ell in bundle["th_ells"]]
    return np.concatenate(blocks)


def _bb_basis(obs_ells, obs_s, n_total):
    cols = []
    cur = 0
    for _, s in zip(obs_ells, obs_s):
        n = len(s)
        for p in [-2, -1, 0, 1, 2]:
            col = np.zeros(n_total)
            col[cur:cur + n] = s ** p
            cols.append(col)
        cur += n
    return np.array(cols).T


def _gls(BB, C_inv, resid):
    return np.linalg.solve(BB.T @ C_inv @ BB, BB.T @ C_inv @ resid)


def _analyze(case_name, bundle, xi_by_ell):
    W = bundle["W"]
    cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
    C_inv = np.linalg.inv(cov_sym)
    data = np.concatenate(bundle["obs_data"])
    th_vec = _build_th_vec(bundle, xi_by_ell)
    pred = W @ th_vec
    resid_noBB = data - pred

    BB = _bb_basis(bundle["obs_ells"], bundle["obs_s"], data.size)
    c_BB = _gls(BB, C_inv, resid_noBB)
    resid_BB = resid_noBB - BB @ c_BB

    sl0 = slice(0, 26)
    sl2 = slice(26, 52)
    inv00 = np.linalg.inv(cov_sym[sl0, sl0])
    inv22 = np.linalg.inv(cov_sym[sl2, sl2])
    sigma = np.sqrt(np.diag(bundle["cov"]))

    def chi2(r, P): return float(r @ P @ r)
    def rms_per_sig(r): return float(np.sqrt(np.mean(r ** 2)))

    print(f"\n=== {case_name} ===")
    chi2_tot_noBB = chi2(resid_noBB, C_inv)
    chi2_tot_BB   = chi2(resid_BB,   C_inv)
    print(f"  combined χ²(no BB)   = {chi2_tot_noBB:7.2f} / 52 = {chi2_tot_noBB/52:.3f}")
    print(f"  combined χ²(BB fit)  = {chi2_tot_BB:7.2f} / 42 = {chi2_tot_BB/42:.3f}")
    print(f"  per-ℓ marginal:")
    print(f"    ℓ=0  no BB: χ²={chi2(resid_noBB[sl0],inv00):6.2f}/26={chi2(resid_noBB[sl0],inv00)/26:.2f}   "
          f"with BB: χ²={chi2(resid_BB[sl0],inv00):6.2f}/21={chi2(resid_BB[sl0],inv00)/21:.2f}")
    print(f"    ℓ=2  no BB: χ²={chi2(resid_noBB[sl2],inv22):6.2f}/26={chi2(resid_noBB[sl2],inv22)/26:.2f}   "
          f"with BB: χ²={chi2(resid_BB[sl2],inv22):6.2f}/21={chi2(resid_BB[sl2],inv22)/21:.2f}")
    print(f"  RMS residual/σ:")
    print(f"    ℓ=0  no BB: {rms_per_sig(resid_noBB[sl0]/sigma[sl0]):.3f}   with BB: {rms_per_sig(resid_BB[sl0]/sigma[sl0]):.3f}")
    print(f"    ℓ=2  no BB: {rms_per_sig(resid_noBB[sl2]/sigma[sl2]):.3f}   with BB: {rms_per_sig(resid_BB[sl2]/sigma[sl2]):.3f}")
    print(f"  BB ℓ=0 coeffs: {np.array2string(c_BB[:5], precision=3, suppress_small=False)}")
    print(f"  BB ℓ=2 coeffs: {np.array2string(c_BB[5:], precision=3, suppress_small=False)}")
    print(f"  ||BB ℓ=2 contribution||_∞ (max |BB(s)|): {np.max(np.abs(BB[sl2] @ c_BB)):.4e}")
    return {
        "resid_noBB": resid_noBB, "resid_BB": resid_BB,
        "c_BB": c_BB, "sigma": sigma, "BB": BB,
    }


def _plot_xi4_effect(bundle, xi_A, xi_B, out_path):
    target_s = bundle["th_s"][0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: ξ_4 itself
    ax = axes[0]
    ax.plot(target_s, xi_B[4] * target_s ** 2, "-", label="pipeline ξ_4 (case B)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel("s² · ξ_4(s)")
    ax.set_title("Pipeline ξ_4(s) at fiducial cosmology")
    ax.set_xlim(0, 200)
    ax.grid(alpha=0.3)
    ax.legend()
    # Right: leakage into observed ℓ=2
    th_A = _build_th_vec(bundle, xi_A)
    th_B = _build_th_vec(bundle, xi_B)
    pred_A = bundle["W"] @ th_A
    pred_B = bundle["W"] @ th_B
    delta = (pred_B - pred_A)[26:52]   # observed ℓ=2 chunk
    obs_s_2 = bundle["obs_s"][1]
    sigma2 = np.sqrt(np.diag(bundle["cov"])[26:52])
    ax = axes[1]
    ax.plot(obs_s_2, delta / sigma2, "o-", color="C3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("s [Mpc/h]  (observed ℓ=2)")
    ax.set_ylabel("(W·ξ_4) / σ_ℓ=2")
    ax.set_title("ξ_4 leakage into observed ℓ=2 (units of σ)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved {out_path}")


def main():
    out_dir = Path(__file__).parent
    # case A: ℓ=4 = 0 (current baseline)
    bundle, xi_A, _ = _build_pipeline_xi(ells_pipe=(0, 2))
    _analyze("case A:  ξ_4 = 0", bundle, xi_A)

    # case B: ℓ=4 from pipeline
    bundle, xi_B, _ = _build_pipeline_xi(ells_pipe=(0, 2, 4))
    _analyze("case B:  ξ_4 from pipeline", bundle, xi_B)

    _plot_xi4_effect(bundle, xi_A, xi_B, out_dir / "test_l4_leakage_LRG2.png")


if __name__ == "__main__":
    main()
