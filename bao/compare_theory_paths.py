"""Compare two routes to ξ_BGS at fiducial: FFTLog-from-P vs native ξ-theory.

If σ(DV) for BGS in native-ξ MCMC (§31g.decimus) is fundamentally limited
by the theory we're plugging in (rather than by data + cov + W), the
smoking gun is a difference in BAO-peak shape between:

  (a) our pipeline's P-space theory FFTLog-projected to ξ via cosmoprimo
      (mcmc_bundle_xi.py path), and
  (b) desilike's native DampedBAOWigglesTracerCorrelationFunctionMultipoles
      (which DESI presumably uses).

Both should be the same physics. If they differ at the BAO peak, that's
the missing Fisher information.

Diagnostic only — does not touch any pipeline module. Reads BGS bundle
theory s grid for the comparison.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import h5py
import numpy as np

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
from desilike.theories.galaxy_clustering import (  # noqa: E402
    DampedBAOWigglesTracerCorrelationFunctionMultipoles,
    BAOPowerSpectrumTemplate,
)

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_DR1_AREA = 7500.0


def _bundle_th_s(fname):
    with h5py.File(_LIK_DIR / fname, "r") as f:
        return np.asarray(f["window/theory/0/s"][...], dtype=np.float64)


def _xi_via_fftlog(pipe_obs, k_lin, target_s, ell):
    P_lk = np.asarray(pipe_obs.theory, dtype=np.float64)  # (n_ell, n_k)
    ie = 0 if ell == 0 else (1 if ell == 2 else None)
    if ie is None or ie >= P_lk.shape[0]: return None
    k_log = np.logspace(np.log10(k_lin[0]), np.log10(k_lin[-1]), 1024)
    P_log = np.interp(k_log, k_lin, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=[ell])
    s_out, xi_out = fft(P_log[None, :], extrap="log")
    s_out = np.asarray(s_out).ravel()
    xi_out = np.asarray(xi_out).ravel()
    order = np.argsort(s_out)
    return np.interp(target_s, s_out[order], xi_out[order])


def main():
    tracer = "BGS"
    cfg = TRACER_CONFIGS[tracer]
    z_eff = float(cfg["z_eff"])
    smoothing = float(cfg["smoothing_scale"])
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})

    info = prep_covar.build_bao_likelihood(
        N_tracers=300_000, theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=z_eff,
        area=_DR1_AREA, apmode="qiso",
    )
    info["likelihood"](**info["params"])
    pipe_theory = None
    for c in info["observable"].runtime_info.pipeline.calculators:
        if type(c).__name__.startswith("DampedBAOWigglesTracerPowerSpectrumMultipoles"):
            pipe_theory = c; break

    k_min, k_max, nb = 0.0014, 0.5, 600
    dk = (k_max - k_min) / nb
    pipe_obs = TracerPowerSpectrumMultipolesObservable(
        data=info["params"],
        klim={0: [k_min, k_max, dk], 2: [k_min, k_max, dk]},
        theory=pipe_theory,
    )
    pipe_lik = ObservablesGaussianLikelihood(observables=[pipe_obs], covariance=np.eye(2 * nb))
    pipe_lik(**info["params"])
    k_lin = np.asarray(pipe_obs.k[0], dtype=np.float64)

    # Native ξ-space theory — same template, same mode, same smoothing.
    template = info["template"]
    th_s = _bundle_th_s("likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5")
    native = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=th_s, ells=(0,),
        template=template, mode="recsym", smoothing_radius=smoothing,
        model="fog-damping",
    )
    native(**{k: v for k, v in info["params"].items()
              if k in native.all_params and not k.startswith("al")})
    xi_native = np.asarray(native.corr, dtype=np.float64).ravel()

    xi_fftlog = _xi_via_fftlog(pipe_obs, k_lin, th_s, ell=0)

    print(f"BGS native ξ₀ shape: {xi_native.shape}, fftlog ξ₀ shape: {xi_fftlog.shape}")
    print(f"  s range: [{th_s[0]:.2f}, {th_s[-1]:.2f}] Mpc/h  (200 points)")
    print()
    print(f"  s [Mpc/h]   ξ_native       ξ_fftlog       Δ/ξ_native")
    print(f"  {'-'*60}")
    for s_target in [60, 80, 100, 110, 120, 130, 140, 150]:
        i = int(np.argmin(np.abs(th_s - s_target)))
        n, f = xi_native[i], xi_fftlog[i]
        rel = (f - n) / n if n != 0 else float("nan")
        print(f"  {th_s[i]:7.2f}    {n:11.4e}   {f:11.4e}   {rel:+.4f}")

    # Now check ∂ξ₀/∂qiso for both — that's what drives Fisher info on q.
    print()
    print("  ∂ξ₀/∂qiso comparison (FD step 0.005):")
    qiso_fid = float(info["params"].get("qiso", 1.0))
    plus_params = {**info["params"], "qiso": qiso_fid + 0.005}
    minus_params = {**info["params"], "qiso": qiso_fid - 0.005}

    pipe_lik(**plus_params)
    xi_fftlog_plus = _xi_via_fftlog(pipe_obs, k_lin, th_s, ell=0)
    pipe_lik(**minus_params)
    xi_fftlog_minus = _xi_via_fftlog(pipe_obs, k_lin, th_s, ell=0)
    dxi_dq_fftlog = (xi_fftlog_plus - xi_fftlog_minus) / 0.01

    native_plus = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=th_s, ells=(0,), template=template, mode="recsym",
        smoothing_radius=smoothing, model="fog-damping")
    native_plus(**{k: v for k, v in plus_params.items()
                   if k in native_plus.all_params and not k.startswith("al")})
    xi_native_plus = np.asarray(native_plus.corr, dtype=np.float64).ravel()
    native_minus = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=th_s, ells=(0,), template=template, mode="recsym",
        smoothing_radius=smoothing, model="fog-damping")
    native_minus(**{k: v for k, v in minus_params.items()
                    if k in native_minus.all_params and not k.startswith("al")})
    xi_native_minus = np.asarray(native_minus.corr, dtype=np.float64).ravel()
    dxi_dq_native = (xi_native_plus - xi_native_minus) / 0.01

    print(f"  s [Mpc/h]   ∂ξ/∂q native   ∂ξ/∂q fftlog   ratio")
    print(f"  {'-'*60}")
    for s_target in [60, 80, 100, 110, 120, 130, 140, 150]:
        i = int(np.argmin(np.abs(th_s - s_target)))
        n, f = dxi_dq_native[i], dxi_dq_fftlog[i]
        r = f / n if n != 0 else float("nan")
        print(f"  {th_s[i]:7.2f}    {n:+11.4e}   {f:+11.4e}   {r:.3f}")

    # L2 norm of derivative — proxy for Fisher information.
    print()
    print(f"  ||∂ξ/∂qiso||_2 native = {np.linalg.norm(dxi_dq_native):.4e}")
    print(f"  ||∂ξ/∂qiso||_2 fftlog = {np.linalg.norm(dxi_dq_fftlog):.4e}")
    print(f"  ratio (Fisher info)   = {(np.linalg.norm(dxi_dq_fftlog) / np.linalg.norm(dxi_dq_native))**2:.3f}")

    # Per-s-bin contribution to ||∂ξ/∂q|| to find where they diverge
    print()
    print("  ||∂ξ/∂q|| contributions by s region (sqrt(sum d² over range)):")
    for s_lo, s_hi in [(0, 30), (30, 50), (50, 100), (100, 150), (150, 200)]:
        m = (th_s >= s_lo) & (th_s < s_hi)
        n_l2 = np.linalg.norm(dxi_dq_native[m])
        f_l2 = np.linalg.norm(dxi_dq_fftlog[m])
        print(f"    s ∈ [{s_lo:3d}, {s_hi:3d}]:  native={n_l2:.4e}  fftlog={f_l2:.4e}  ratio²={(f_l2/n_l2)**2 if n_l2 else float('nan'):.3f}")

    # Most relevant: Fisher info AFTER applying the bundle W and cov.
    # F = (W ∂ξ/∂q)^T C^{-1} (W ∂ξ/∂q).
    with h5py.File(_LIK_DIR / "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5", "r") as f:
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        C = np.asarray(f["covariance/value"][...], dtype=np.float64)
    # W is (26, 600) = (n_obs, n_theory_ell × n_theory_s). For ξ_0 only theory,
    # we need ∂ξ_0 in the first 200 entries (assuming the layout is [ℓ=0, ℓ=2, ℓ=4]
    # × 200 s-points each). For BGS, only ℓ=0 contributes to the data side.
    n_th_s = len(th_s)
    pad_native = np.concatenate([dxi_dq_native, np.zeros(W.shape[1] - n_th_s)])
    pad_fftlog = np.concatenate([dxi_dq_fftlog, np.zeros(W.shape[1] - n_th_s)])
    Wd_native = W @ pad_native
    Wd_fftlog = W @ pad_fftlog
    C_inv = np.linalg.inv(0.5 * (C + C.T))
    F_native = float(Wd_native @ C_inv @ Wd_native)
    F_fftlog = float(Wd_fftlog @ C_inv @ Wd_fftlog)
    print()
    print(f"  FISHER INFO ON qiso AFTER W AND C (the actual constraint):")
    print(f"    F_native = {F_native:.4e}  →  σ_q = {1/np.sqrt(F_native):.4f}")
    print(f"    F_fftlog = {F_fftlog:.4e}  →  σ_q = {1/np.sqrt(F_fftlog):.4f}")
    print(f"    Fisher ratio (fftlog/native) = {F_fftlog/F_native:.3f}")
    print(f"    σ_q ratio (fftlog/native)    = {np.sqrt(F_native/F_fftlog):.3f}")


if __name__ == "__main__":
    main()
