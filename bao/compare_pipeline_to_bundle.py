"""Tier-1 data-level cross-check of the pipeline against the DESI bundle.

For LRG2 and QSO (the two tracers where we have local likelihood bundles):

  1. Build the standard pipeline at fiducial cosmology (BAO k-window [0.02,0.30]).
  2. Build a *second* observable using the same theory chain but on a much wider
     log-spaced k-grid (k ∈ [1e-4, 5] Mpc⁻¹), so the Hankel of P_ℓ(k) → ξ_ℓ(s)
     is well-posed across the full BAO range.
  3. FFTLog (cosmoprimo's PowerToCorrelation) → ξ_ℓ(s) on a log-s grid, then
     interpolate onto the bundle's 200-bin theory s-grid.
  4. ℓ=4 is set to zero (pipeline only models ℓ ∈ {0,2}).
  5. Apply bundle W (52 × 600) → 52-bin observable prediction.
  6. Compute χ² (theory only) and χ² (with GLS BB polynomial fit per ℓ).

FFTLog handles low/high-k extrapolation analytically (log-log power-law tails),
so we avoid the k_min cutoff ringing that destroyed the linear-Riemann-sum
attempt.
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

sys.path.insert(0, "..")

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
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}

_BUNDLES = {
    "LRG2": "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "QSO":  "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}


def _load_bundle(tracer):
    p = _LIK_DIR / _BUNDLES[tracer]
    with h5py.File(p, "r") as f:
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
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP.get(tracer, tracer)]


def _build_wide_observable(info, k_log, ells=(0, 2)):
    """Re-use the pipeline's theory chain on a wide log-k grid.

    Returns (P_lk, k_log_centers) where P_lk has shape (len(ells), len(k_log_centers)).
    Includes bias, broadband (BB at fiducial is zero), AP — same chain as the BAO fit.
    """
    # Find the outer tracer-multipoles theory calculator in the existing chain.
    theory = None
    for c in info["observable"].runtime_info.pipeline.calculators:
        if type(c).__name__.startswith("DampedBAOWigglesTracerPowerSpectrumMultipoles"):
            theory = c
            break
    if theory is None:
        raise RuntimeError("could not locate DampedBAOWigglesTracerPowerSpectrumMultipoles in chain")

    # Build a klim spec that lands on the requested log-k centers. desilike's
    # TracerPowerSpectrumMultipolesObservable expects linear binning; the wider
    # k-range is what matters for the Hankel, so we use very fine linear bins
    # covering [k_log.min(), k_log.max()] instead.
    k_min, k_max = float(k_log[0]), float(k_log[-1])
    n_bins = 600  # fine linear grid
    dk = (k_max - k_min) / n_bins
    edges = np.linspace(k_min, k_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    new_obs = TracerPowerSpectrumMultipolesObservable(
        data=info["params"],
        klim={int(ell): [k_min, k_max, dk] for ell in ells},
        theory=theory,
    )
    # Wrap in a minimal Gaussian likelihood (cov is irrelevant for theory evaluation).
    n_tot = n_bins * len(ells)
    new_lik = ObservablesGaussianLikelihood(
        observables=[new_obs], covariance=np.eye(n_tot),
    )
    new_lik(**info["params"])
    P_lk = np.asarray(new_obs.theory, dtype=np.float64)  # (n_ells, n_bins)
    k_centers = np.asarray(new_obs.k[0], dtype=np.float64)
    return P_lk, k_centers


def _fftlog_xi(P_lk, k_linear, ells, target_s):
    """FFTLog P_ℓ(k) → ξ_ℓ(s). cosmoprimo wants log-spaced k; interpolate first."""
    k_log = np.logspace(np.log10(k_linear[0]), np.log10(k_linear[-1]), 1024)
    P_log = np.empty((len(ells), len(k_log)), dtype=np.float64)
    for ie, _ell in enumerate(ells):
        P_log[ie] = np.interp(k_log, k_linear, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=list(ells))
    s_out, xi_out = fft(P_log, extrap="log")
    # cosmoprimo returns (n_ells, n_s) for xi_out, s_out is (n_ells, n_s)
    s_out = np.asarray(s_out)
    xi_out = np.asarray(xi_out)
    xi_target = np.empty((len(ells), len(target_s)), dtype=np.float64)
    for ie in range(len(ells)):
        s_e = s_out[ie] if s_out.ndim == 2 else s_out
        xi_e = xi_out[ie]
        # sort by s for safe interp
        order = np.argsort(s_e)
        xi_target[ie] = np.interp(target_s, s_e[order], xi_e[order])
    return xi_target


def _bb_basis(obs_ells, obs_s, n_total):
    cols = []
    cur = 0
    for ie, (_, s) in enumerate(zip(obs_ells, obs_s)):
        n = len(s)
        for p in [-2, -1, 0, 1, 2]:
            col = np.zeros(n_total)
            col[cur:cur + n] = s ** p
            cols.append(col)
        cur += n
    return np.array(cols).T


def _chi2(resid, C_inv):
    return float(resid @ C_inv @ resid)


def _gls_fit(BB, C_inv, resid):
    M = BB.T @ C_inv @ BB
    rhs = BB.T @ C_inv @ resid
    return np.linalg.solve(M, rhs)


def main():
    out_dir = Path(__file__).parent
    for tracer in ["LRG2", "QSO"]:
        print(f"\n{'=' * 60}\n{tracer}\n{'=' * 60}")
        bundle = _load_bundle(tracer)

        cfg = TRACER_CONFIGS[tracer]
        z_min, z_max = cfg["zrange"]
        theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
            {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
        info = prep_covar.build_bao_likelihood(
            N_tracers=_get_ntracers(tracer),
            theta_cosmo=theta, hrdrag=hrdrag_eff,
            tracer_bin=tracer, zrange=(z_min, z_max),
            z_eff=None, area=_DR1_AREA,
        )
        info["likelihood"](**info["params"])

        # Wider-k evaluation. The BAO template's internal P_lin range is finite;
        # safe choice is [1e-3, 0.5] Mpc⁻¹ — well below k where the model breaks.
        k_log = np.logspace(-3, np.log10(0.5), 512)
        P_lk, k_linear = _build_wide_observable(info, k_log, ells=(0, 2))
        print(f"  wide-k P_ℓ(k): shape={P_lk.shape}  k ∈ [{k_linear.min():.4f}, {k_linear.max():.3f}]")
        print(f"  P_0(k≈0.003): {P_lk[0, np.argmin(np.abs(k_linear-0.003))]:.3e}  "
              f"P_0(k≈0.1): {P_lk[0, np.argmin(np.abs(k_linear-0.1))]:.3e}  "
              f"P_0(k≈0.3): {P_lk[0, np.argmin(np.abs(k_linear-0.3))]:.3e}")

        # FFTLog to ξ_ℓ(s) on bundle's theory s-grid
        target_s = bundle["th_s"][0]  # all ells share the same s-grid (verified)
        xi_pipe = _fftlog_xi(P_lk, k_linear, ells=[0, 2], target_s=target_s)

        # Build the full 600-vector matching bundle theory layout (ells 0,2,4)
        xi_th_vec_blocks = []
        for ell in bundle["th_ells"]:
            if ell == 0:
                xi_th_vec_blocks.append(xi_pipe[0])
            elif ell == 2:
                xi_th_vec_blocks.append(xi_pipe[1])
            else:
                xi_th_vec_blocks.append(np.zeros_like(target_s))
        xi_th_vec = np.concatenate(xi_th_vec_blocks)

        pred = bundle["W"] @ xi_th_vec
        data = np.concatenate(bundle["obs_data"])
        cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
        C_inv = np.linalg.inv(cov_sym)
        resid_noBB = data - pred
        chi2_noBB = _chi2(resid_noBB, C_inv)

        BB = _bb_basis(bundle["obs_ells"], bundle["obs_s"], data.size)
        c_BB = _gls_fit(BB, C_inv, resid_noBB)
        resid_BB = resid_noBB - BB @ c_BB
        chi2_BB = _chi2(resid_BB, C_inv)
        dof = data.size
        dof_BB = data.size - BB.shape[1]
        print(f"\n  n_data = {dof}, n_BB = {BB.shape[1]}")
        print(f"  χ²(theory only, no BB)   = {chi2_noBB:10.2f}   /  {dof}  =  {chi2_noBB/dof:.2f}")
        print(f"  χ²(theory + GLS BB fit)  = {chi2_BB:10.2f}   /  {dof_BB}  =  {chi2_BB/dof_BB:.2f}")

        # Plot
        fig, axes = plt.subplots(2, max(len(bundle["obs_ells"]), 2), figsize=(12, 8), squeeze=False)
        sigma_full = np.sqrt(np.diag(bundle["cov"]))
        cur = 0
        for ie, (ell, s) in enumerate(zip(bundle["obs_ells"], bundle["obs_s"])):
            n = len(s)
            sigma = sigma_full[cur:cur + n]
            s_sq = s ** 2
            ax = axes[0, ie]
            ax.errorbar(s, data[cur:cur + n] * s_sq, yerr=sigma * s_sq, fmt="o", ms=4,
                        color="k", label="DESI data")
            ax.plot(s, pred[cur:cur + n] * s_sq, "-", color="C0",
                    label="pipeline theory (no BB)")
            ax.plot(s, (pred[cur:cur + n] + (BB @ c_BB)[cur:cur + n]) * s_sq, "--",
                    color="C3", label="pipeline + BB fit")
            ax.set_xlabel("s [Mpc/h]")
            ax.set_ylabel(f"s²·ξ_{ell}(s)")
            ax.set_title(f"{tracer} ℓ={ell}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = axes[1, ie]
            ax.errorbar(s, resid_noBB[cur:cur + n] / sigma, yerr=1, fmt="o", ms=4,
                        color="C0", label="(d − th) / σ")
            ax.plot(s, resid_BB[cur:cur + n] / sigma, "x", color="C3",
                    label="(d − th − BB) / σ")
            ax.axhline(0, color="k", lw=0.5)
            ax.axhline(+2, color="grey", lw=0.5, ls=":")
            ax.axhline(-2, color="grey", lw=0.5, ls=":")
            ax.set_xlabel("s [Mpc/h]")
            ax.set_ylabel("residual / σ")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            cur += n
        fig.suptitle(f"{tracer}: pipeline (FFTLog) vs DESI bundle    "
                     f"χ²(no BB)={chi2_noBB:.1f}/{dof}    "
                     f"χ²(BB-fit)={chi2_BB:.1f}/{dof_BB}", fontsize=11)
        plt.tight_layout()
        out = out_dir / f"compare_pipeline_bundle_{tracer}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
