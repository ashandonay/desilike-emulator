"""Config-space (ξ) BAO pipeline — the direct analog of prep_covar.py's Fourier
machinery, consolidated into one module so the two frames can be read side by side.

Frame map (config here  ↔  Fourier in prep_covar):

    load_bundle / bb_basis            ↔  observable + BB setup inside build_bao_likelihood
    build_native_theory_*             ↔  the P-space theory in build_bao_likelihood
    gaussian_xi_multipole_cov         ↔  the FKP Gaussian cov (cov_engine="desilike")
      + gaussxi_cov_on_bundle_grid       (Grieb 2016 double-Bessel; config-native)
    bundle_fisher_sigmas              ↔  run_fisher / get_bao_fisher_covariance
    make_log_prob / run_mcmc          ↔  (no Fourier MCMC analog; reference is mcmc_bao.py)

Space-agnostic DESI / Fourier *reference* σ (bao-recon, desi_data.csv, production
Fisher) live in reference_sigmas.py — they are what BOTH frames are compared against.

What this module shares with prep_covar: ONLY the cosmology mapping
(`_to_bao_cosmo_params`), the fiducial template, and the fiducial nuisance values
returned by `build_bao_likelihood`. It never uses prep_covar's Fourier covariance,
theory, FKP/V_eff machinery, or SSC — the covariance comes from the DESI bundle or
the Grieb config-cov here.

A single fiducial cosmology is used across all three paths (Fisher, Grieb-cov, MCMC):
    _FID = {Om: 0.3152, h·r_d: 99.08}
The original scattered scripts had drifted to two slightly different fids (the MCMC
builder sat at h·r_d 99.53, ~0.45% off); they are now reconciled onto _FID.
"""
from __future__ import annotations
import os, sys, argparse, json, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from typing import Dict, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import emcee
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import prep_covar
from util import TRACER_CONFIGS
from desilike.theories.galaxy_clustering import (
    DampedBAOWigglesTracerCorrelationFunctionMultipoles,
    DampedBAOWigglesTracerPowerSpectrumMultipoles,
)
from fkp_analytic_cov import NZSlices, _fkp_integrals, P_FKP_DEFAULT, load_nz_slices

# ---------------------------------------------------------------------------
# Fixed analysis settings
# ---------------------------------------------------------------------------
_FID = {"Om": 0.3152, "hrdrag": 99.08}          # single fiducial: Fisher, Grieb-cov, MCMC
_AREA = 7500.0
_DESI_PRIOR_SCALE = {"sigmapar": 2.0, "sigmaper": 2.0, "sigmas": 2.0}

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1"}

# Wide k-grid for the config-cov double-Bessel transform [h/Mpc].
# kmax MUST be high enough to localize the shot-noise (white) floor onto the
# diagonal: in config space ∫dk k² jₗ(ks_i)jₗ(ks_j) → (π/2)δ(s_i-s_j)/s_i² only
# as kmax→∞; the delta has width ~π/kmax in s. At kmax=0.6 (π/kmax≈5 Mpc/h ≳ the
# ~4 Mpc/h bin width) the shot term smears off-diagonal → the cov is over-smooth
# and NUMERICALLY near-singular (cond ~1e10), which makes the Fisher inverse
# amplify a spurious near-null mode and underestimate σ by ~2× (and breaks MCMC).
# kmax≥1.0 restores cond ~1e3 (≈ the DESI bundle) and σ to the expected band;
# kmax=2.0 is converged with margin. Nk: σ and cond are identical from Nk=400
# (dk=5e-3) up to 10000 — the integrand only needs dk≲π/(2·s_max); Nk=2000
# (dk=1e-3) is safe even on the fine theory s-grid and ~5× cheaper than Nk=1e4,
# which matters for the cosmology-varying emulator cov build.
K_WIDE = np.linspace(1e-4, 2.0, 2000)

_OUT_DIR = Path(__file__).resolve().parent / "mcmc_results_bundle_native"
_OUT_DIR.mkdir(exist_ok=True)

# DESI correlation-recon-poles bundles: (filename, z_eff).
_BUNDLES = {
    "BGS":       ("likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5", 0.295),
    "LRG1":      ("likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5", 0.51),
    "LRG2":      ("likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5", 0.706),
    "LRG3_ELG1": ("likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5", 0.93),
    "ELG2":      ("likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5", 1.317),
    "QSO":       ("likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5", 1.491),
}


# ===========================================================================
# §1  Bundle I/O + broadband basis  (↔ observable/BB setup in build_bao_likelihood)
# ===========================================================================
def load_bundle(tracer):
    """Load a DESI correlation-recon-poles bundle: cov, window W, observable
    (ells/s/data) and theory (ells/s) grids."""
    fname, z = _BUNDLES[tracer]
    with h5py.File(_LIK_DIR / fname, "r") as f:
        cov = np.asarray(f["covariance/value"][...])
        W = np.asarray(f["window/value"][...])
        obs_ells, obs_s, obs_data = [], [], []
        for k in sorted(f["observable"].keys()):
            if k.isdigit():
                obs_ells.append(int(f[f"observable/{k}/meta/ell"][()]))
                obs_s.append(np.asarray(f[f"observable/{k}/s"][...]))
                obs_data.append(np.asarray(f[f"observable/{k}/value"][...]))
        th_ells, th_s = [], []
        for k in sorted(f["window/theory"].keys()):
            if k.isdigit():
                th_ells.append(int(f[f"window/theory/{k}/meta/ell"][()]))
                th_s.append(np.asarray(f[f"window/theory/{k}/s"][...]))
    return {"cov": cov, "W": W, "z": z,
            "obs_ells": obs_ells, "obs_s": obs_s, "obs_data": obs_data,
            "th_ells": th_ells, "th_s": th_s}


def bb_basis(obs_ells, obs_s, n_total, powers=(0, 2)):
    """DESI Adame+24 broadband design matrix: columns s^p per (ell, power).
    Default powers=(0,2) = the {s⁰, s²} per-ℓ basis."""
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


def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    return {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}[_NAME_MAP.get(tracer, tracer)]


def _get_desi_std(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    sub = df[df["tracer"] == _NAME_MAP.get(tracer, tracer)]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


# ===========================================================================
# §2  Config-space Gaussian covariance (Grieb 2016, double-Bessel)
#     ↔ the FKP P-space Gaussian cov in prep_covar (cov_engine="desilike")
# ===========================================================================
def _cross_ell_sign(la: int, lb: int) -> float:
    """Real value of i^{la-lb} for even multipoles."""
    d = abs(la - lb) % 4
    if d == 0:
        return 1.0
    if d == 2:
        return -1.0
    raise ValueError(f"i^(l-l') not real for la={la}, lb={lb} (odd multipoles)")


def per_mode_sigma2(
    k: np.ndarray,
    P_ells_in: np.ndarray,
    ells_in: Sequence[int],
    ells_obs: Sequence[int],
    slices: NZSlices,
    P_FKP: float = P_FKP_DEFAULT,
    n_mu: int = 64,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Per-mode multipole variance sigma^2_{l,l'}(k) (Grieb Eq. 19).

    Identical angular/FKP structure to fkp_analytic_cov.fkp_analytic_cov, but
    WITHOUT the k-shell bin_factor (that is supplied by the Bessel transform
    in gaussian_xi_multipole_cov). Returns a dict of (la, lb) -> (Nk,) arrays.
    """
    k = np.asarray(k, dtype=np.float64)
    P_ells_in = np.asarray(P_ells_in, dtype=np.float64)
    Nk = k.size

    I22, I44, I34, I24 = _fkp_integrals(slices, P_FKP=P_FKP)

    mu, mu_w = leggauss(n_mu)  # nodes in [-1,1], weights sum to 2
    L = {}
    for ell in set(list(ells_in) + list(ells_obs) + [0]):
        coeff = np.zeros(ell + 1)
        coeff[ell] = 1.0
        L[ell] = np.polynomial.legendre.legval(mu, coeff)

    # P(k, mu) on the quadrature grid, shape (n_mu, Nk).
    P_kmu = np.zeros((n_mu, Nk), dtype=np.float64)
    for i_ell, ell in enumerate(ells_in):
        P_kmu += L[ell][:, None] * P_ells_in[i_ell][None, :]

    sigma2: Dict[Tuple[int, int], np.ndarray] = {}
    for la in ells_obs:
        for lb in ells_obs:
            LaLb = L[la] * L[lb]
            mean_LLP2 = 0.5 * np.sum((LaLb * mu_w)[:, None] * P_kmu ** 2, axis=0)
            mean_LLP = 0.5 * np.sum((LaLb * mu_w)[:, None] * P_kmu, axis=0)
            mean_LL = 0.5 * float(np.sum(LaLb * mu_w))
            sigma2[(la, lb)] = (
                (2 * la + 1) * (2 * lb + 1) * 2.0 / I22 ** 2 * (
                    mean_LLP2 * I44 + 2.0 * mean_LLP * I34 + mean_LL * I24
                )
            )
    return sigma2


def _binavg_spherical_jn(ell, k, s_centers, ds, n_sub=16):
    """Volume(s^2)-weighted bin-average of j_l(k s) over each s-bin.

        jbar_l(k; bin_i) = INT_{s_lo}^{s_hi} j_l(ks) s^2 ds / INT s^2 ds

    This matches how the binned xi estimator averages pairs over the bin,
    regularizing the otherwise cutoff-divergent shot-noise transform. Returns
    shape (Ns, Nk). If ds is None, falls back to point evaluation j_l(k s_i).
    """
    from scipy.special import spherical_jn
    k = np.asarray(k, dtype=np.float64)
    s_centers = np.asarray(s_centers, dtype=np.float64)
    if ds is None:
        return spherical_jn(ell, k[None, :] * s_centers[:, None])
    ds = np.broadcast_to(np.asarray(ds, dtype=np.float64), s_centers.shape)
    out = np.empty((s_centers.size, k.size), dtype=np.float64)
    for i, (sc, d) in enumerate(zip(s_centers, ds)):
        s_lo, s_hi = sc - 0.5 * d, sc + 0.5 * d
        ss = np.linspace(s_lo, s_hi, n_sub)
        w = ss ** 2
        # (n_sub, Nk) -> volume-weighted average over the bin
        j = spherical_jn(ell, k[None, :] * ss[:, None])
        out[i] = np.trapz(w[:, None] * j, ss, axis=0) / np.trapz(w, ss)
    return out


def gaussian_xi_multipole_cov(
    s: np.ndarray,
    ells_obs: Sequence[int],
    k: np.ndarray,
    P_ells_in: np.ndarray,
    ells_in: Sequence[int],
    slices: NZSlices,
    P_FKP: float = P_FKP_DEFAULT,
    n_mu: int = 64,
    ds: float | np.ndarray | None = None,
    jbar_cache: Dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Full Gaussian xi-multipole covariance on the s-grid.

    Parameters
    ----------
    s          : (Ns,) separations [Mpc/h] of the xi data vector (bin centers).
    ells_obs   : multipoles of the xi data vector, e.g. (0, 2) or (0,).
    k          : (Nk,) k-grid for the Bessel integral [h/Mpc]. Should span
                 well below the BAO scale to well above kmax of the signal;
                 uniform or quasi-uniform. dk is taken from diff(k).
    P_ells_in  : (N_in, Nk) model P_l(k), shot-noise-REMOVED (1/n added via
                 the I_NM structure). Multipoles listed in ells_in.
    ells_in    : multipoles present in P_ells_in, e.g. (0, 2, 4).
    slices     : NZSlices (cosmology/design-dependent n(z), V).
    ds         : s-bin width [Mpc/h] (scalar or per-bin). When given, the
                 Bessel functions are bin-volume-averaged over each s-bin —
                 the correct treatment of the binned xi estimator (regularizes
                 the shot-noise transform). If None, j_l is evaluated at bin
                 centers (point approximation; under-predicts shot noise).

    Returns
    -------
    (Nell*Ns, Nell*Ns) covariance, block-ordered by ells_obs (matching the
    desilike / bundle xi data-vector stacking: all s for l=0, then l=2, ...).
    """
    s = np.asarray(s, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    Ns = s.size
    Nell = len(ells_obs)

    dk = np.empty_like(k)
    if k.size > 1:
        dk[1:-1] = 0.5 * (k[2:] - k[:-2])
        dk[0] = k[1] - k[0]
        dk[-1] = k[-1] - k[-2]
    else:
        dk[:] = 0.005

    sigma2 = per_mode_sigma2(k, P_ells_in, ells_in, ells_obs, slices,
                             P_FKP=P_FKP, n_mu=n_mu)

    # Bin-averaged (or point) spherical Bessel j_l(k s): (Ns, Nk) per ell.
    # These depend ONLY on (ell, k, s, ds) — all fixed across cosmologies — so a
    # generator sweeping many cosmologies passes a precomputed jbar_cache (see
    # build_jbar_cache) and this expensive step is skipped entirely.
    if jbar_cache is None:
        j_cache = {ell: _binavg_spherical_jn(ell, k, s, ds) for ell in set(ells_obs)}
    else:
        j_cache = jbar_cache

    C = np.zeros((Nell * Ns, Nell * Ns), dtype=np.float64)
    inv2pi2 = 1.0 / (2.0 * np.pi ** 2)
    w_k = dk * k ** 2  # integration measure, shape (Nk,)

    for ia, la in enumerate(ells_obs):
        ja = j_cache[la]  # (Ns, Nk)
        for ib, lb in enumerate(ells_obs):
            jb = j_cache[lb]  # (Ns, Nk)
            sign = _cross_ell_sign(la, lb)
            integrand_w = (w_k * sigma2[(la, lb)])[None, :]  # (1, Nk)
            # C_block[i,j] = sign/(2pi^2) sum_a w_a k_a^2 sigma2_a ja[i,a] jb[j,a]
            block = sign * inv2pi2 * (ja * integrand_w) @ jb.T  # (Ns, Ns)
            C[ia * Ns:(ia + 1) * Ns, ib * Ns:(ib + 1) * Ns] = block

    # Enforce symmetry (kills tiny asymmetries from the la<->lb sign/measure).
    C = 0.5 * (C + C.T)
    return C


def build_jbar_cache(ells, k, s, ds):
    """Precompute the bin-averaged spherical-Bessel matrices jₗ(k·s), (Ns,Nk)
    per ℓ, for a FIXED (s, k, ds). These are cosmology-independent — the only
    cosmology dependence of the Grieb cov is the per-mode σ²(k) — so for a
    multi-cosmology sweep this is built ONCE and passed to
    gaussian_xi_multipole_cov(..., jbar_cache=cache), collapsing the per-cosm
    cov to per_mode_sigma2 + a few BLAS matmuls."""
    return {ell: _binavg_spherical_jn(ell, k, s, ds) for ell in set(ells)}


def _wide_Pell(tracer, info):
    """Evaluate P_ℓ(k) (shot-noise free) on K_WIDE, ells (0,2,4), at fid.

    Reuses the fid template from `info` so the cosmology matches exactly.
    """
    cfg = TRACER_CONFIGS[tracer]
    template = info["template"]
    # broadband="power" not "pcs": the cov uses only the FIDUCIAL P_ℓ(k) (BB
    # nuisance = 0), where the two bases give byte-identical P. "power" avoids a
    # ~600 ms ONE-TIME-PER-PROCESS JAX compile of the "pcs" B-spline basis — a
    # real saving when the 10k sweep is sharded across many short-lived worker
    # processes (each compiles once). NB it does NOT remove the dominant
    # per-cosmology cost: desilike bakes the cosmology-dependent template P(k) as
    # jit constants, so each new cosmology forces a ~800 ms retrace regardless.
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
        template=template, mode="recsym", ells=(0, 2, 4),
        smoothing_radius=float(cfg["smoothing_scale"]),
        model="fog-damping", broadband="power",
    )
    # Drive the theory's k-grid directly to K_WIDE.
    theory.init.update(k=K_WIDE)
    theory(**{k: v for k, v in info["params"].items()
              if k in {p.name for p in theory.all_params}})
    P = np.asarray(theory.power, dtype=np.float64)  # (n_ells, Nk_wide)
    assert P.shape[0] == 3, f"expected 3 multipoles (0,2,4), got {P.shape}"
    return P  # ells_in order (0,2,4)


def gaussxi_cov_on_bundle_grid(tracer, info, bundle):
    """Build the analytic Gaussian ξ-cov on the bundle observable s-grid.

    Returns (C_obsgrid, C_windowed) where C_windowed = W @ C_theory @ W^T is
    the survey-window-convolved cov on the observable grid.
    """
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag = prep_covar._to_bao_cosmo_params({**prep_covar.PARAM_DEFAULTS, **_FID})
    cosmo = prep_covar.get_cosmo(("DESI", dict(theta)))
    slices = load_nz_slices(
        tracer_bin=tracer, cosmo=cosmo, area_deg2=_AREA,
        N_design=float(_get_ntracers(tracer)),
    )

    P_wide = _wide_Pell(tracer, info)  # (3, Nk) for ells (0,2,4)

    # Bundle observable s-grid & ells (data-vector stacking order).
    obs_ells = list(bundle["obs_ells"])
    obs_s = bundle["obs_s"]
    # All obs_s entries should be the same grid; verify and use the first.
    s_grid = np.asarray(obs_s[0], dtype=np.float64)
    for arr in obs_s[1:]:
        if not np.allclose(arr, s_grid):
            raise ValueError("bundle obs_s differs per ell; need per-ell s-grid handling")
    ds_obs = float(np.mean(np.diff(s_grid)))

    C_obsgrid = gaussian_xi_multipole_cov(
        s=s_grid, ells_obs=tuple(obs_ells),
        k=K_WIDE, P_ells_in=P_wide, ells_in=(0, 2, 4),
        slices=slices, ds=ds_obs,
    )

    # Windowed variant: build on the THEORY s-grid (th_s, th_ells) then convolve
    # with the survey window W (computed from the RANDOM catalog — geometry,
    # not data): C_obs = W @ C_theory @ W^T.
    th_ells = list(bundle["th_ells"])
    th_s = bundle["th_s"]
    s_th = np.asarray(th_s[0], dtype=np.float64)
    for arr in th_s[1:]:
        if not np.allclose(arr, s_th):
            raise ValueError("bundle th_s differs per ell")
    ds_th = float(np.mean(np.diff(s_th)))
    C_theory = gaussian_xi_multipole_cov(
        s=s_th, ells_obs=tuple(th_ells),
        k=K_WIDE, P_ells_in=P_wide, ells_in=(0, 2, 4),
        slices=slices, ds=ds_th,
    )
    W = bundle["W"]
    if W.shape[1] != C_theory.shape[0]:
        raise ValueError(f"W cols {W.shape[1]} != C_theory {C_theory.shape[0]}")
    C_windowed = W @ C_theory @ W.T
    C_windowed = 0.5 * (C_windowed + C_windowed.T)
    return C_obsgrid, C_windowed


def _validate_shotnoise_limit(verbose: bool = True) -> bool:
    """With P_l(k)=0 (pure shot noise) and uniform n(z), sigma^2_{00}(k) is a
    k-independent constant C0 = 2 * I_24 / I_22^2, and

        C^xi_00(s_i, s_j) = C0/(2pi^2) * INT_0^kmax dk k^2 j_0(k s_i) j_0(k s_j).

    We check the implementation reproduces this analytic integrand sum exactly
    (it should, to ~1e-12, since it IS the same sum), validating the (2pi^2),
    dk, k^2 and i^{l-l'} normalization that silently broke the prior attempt.
    """
    # Uniform single-slice n(z): one shell, n=3e-4, V=1e9 (Mpc/h)^3.
    slices = NZSlices(z_mid=np.array([0.5]),
                      nbar=np.array([3.0e-4]),
                      V=np.array([1.0e9]))
    k = np.linspace(1e-4, 0.5, 4000)
    P0 = np.zeros_like(k)  # pure shot noise
    s = np.array([30.0, 50.0, 80.0, 120.0])

    C = gaussian_xi_multipole_cov(
        s, ells_obs=(0,), k=k, P_ells_in=P0[None, :], ells_in=(0,),
        slices=slices,
    )

    I22, I44, I34, I24 = _fkp_integrals(slices)
    C0 = 2.0 * I24 / I22 ** 2
    from scipy.special import spherical_jn
    dk = np.empty_like(k)
    dk[1:-1] = 0.5 * (k[2:] - k[:-2]); dk[0] = k[1]-k[0]; dk[-1] = k[-1]-k[-2]
    ref = np.zeros((s.size, s.size))
    for i in range(s.size):
        for j in range(s.size):
            ji = spherical_jn(0, k * s[i]); jj = spherical_jn(0, k * s[j])
            ref[i, j] = C0 / (2*np.pi**2) * np.sum(dk * k**2 * ji * jj)

    rel = np.max(np.abs(C - ref) / np.maximum(np.abs(ref), 1e-300))
    ok = rel < 1e-10
    if verbose:
        print(f"[shotnoise limit] C0={C0:.4e}  max_rel_diff={rel:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"  diag C^xi_00(s,s): {np.diag(C)}")
        print(f"  positive-definite: {np.all(np.linalg.eigvalsh(C) > 0)}")
    return ok


# ===========================================================================
# §3  Native ξ-theory + Fisher  (↔ prep_covar.run_fisher / get_bao_fisher_covariance)
# ===========================================================================
def bundle_fisher_sigmas(tracer, cov_override=None, info=None, bundle=None):
    """Return dict with σ(DH/rd), σ(DM/rd), σ(DV/rd) from native-ξ Fisher.

    cov_override: optional ndarray to substitute for bundle['cov']. Pass the
    Grieb analytic cov (the forecast default for the emulator generator) or a
    banded bundle cov (for the diagnostics that probe which off-diagonal
    structure drives σ_q). Must match bundle data-vector shape.
    info, bundle: optional prebuilt lean info / loaded bundle. The multi-cosmology
    generator builds these ONCE per cosmology (info) / ONCE total (bundle) and
    passes them in, so this function doesn't redundantly rebuild the template.
    """
    cfg = TRACER_CONFIGS[tracer]
    apmode = "qiso" if tracer in ("BGS", "QSO") else "qparqper"
    if info is None:
        # lean=True: skip the desilike observable/Fourier-cov/likelihood build.
        # This path uses only the template + fiducial nuisance values; the ξ-cov
        # and native ξ-theory are built below. ~6.7× faster per cosmology.
        theta, hrdrag = prep_covar._to_bao_cosmo_params(
            {**prep_covar.PARAM_DEFAULTS, **_FID})
        info = prep_covar.build_bao_likelihood(
            N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
            tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
            area=_AREA, apmode=apmode, lean=True,
        )
    if "likelihood" in info:  # full build only; template self-initializes
        info["likelihood"](**info["params"])
    tmpl = info["template"]
    if bundle is None:
        bundle = load_bundle(tracer)
    smoothing = float(cfg["smoothing_scale"])

    # broadband="power": this Fisher zeroes the theory's internal broadband
    # (bb_zero below) and uses a SEPARATE analytic BB basis (bb_basis), so the
    # theory's BB shape is irrelevant and ξ at BB=0 is byte-identical (verified).
    # "power" avoids the ~700 ms one-time-per-process "pcs" JAX compile (helps
    # sharded parallel runs); the per-cosmology template retrace remains.
    native = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=bundle["th_s"][0], ells=tuple(bundle["th_ells"]),
        template=tmpl, mode="recsym", smoothing_radius=smoothing,
        model="fog-damping", broadband="power",
    )

    base = {}
    for k in ("b1", "sigmapar", "sigmaper", "sigmas"):
        if k in info["params"]:
            base[k] = float(info["params"][k])
    bb_zero = {p.name: 0.0 for p in native.all_params
               if (p.name.startswith("al") or p.name.startswith("bl"))}
    base.update(bb_zero)
    native_pnames = {p.name for p in native.all_params}

    def predict(overrides=None):
        params = dict(base)
        if overrides:
            params.update(overrides)
        p_filt = {k: v for k, v in params.items() if k in native_pnames}
        native(**p_filt)
        xi = np.asarray(native.corr)
        xi_flat = np.concatenate([xi[i] for i in range(xi.shape[0])])
        return bundle["W"] @ xi_flat

    cov_src = bundle["cov"] if cov_override is None else cov_override
    cov = 0.5 * (cov_src + cov_src.T)
    C_inv = np.linalg.inv(cov)

    # Parameter set: q's first, then nuisances. For qiso (sparse tracers,
    # ξ₀-only) drop dbeta — matches desilike's auto-fix when ells=(0,) and
    # DESI Adame+24's sparse-tracer analysis. RSD info isn't in ξ₀ data, so
    # keeping dbeta free injects a spurious correlation that biases the
    # Schur-marginalized qiso uncertainty.
    nl_params = (
        ("qiso", "b1", "sigmas", "sigmapar", "sigmaper")
        if apmode == "qiso"
        else ("qpar", "qper", "b1", "dbeta", "sigmas", "sigmapar", "sigmaper")
    )
    steps = {"qpar": 0.01, "qper": 0.01, "qiso": 0.01, "b1": 0.05,
             "dbeta": 0.05, "sigmas": 0.05, "sigmapar": 0.1, "sigmaper": 0.1}

    fid_vals = {}
    for p in nl_params:
        if p in base:
            fid_vals[p] = base[p]
        else:
            for q in native.all_params:
                if q.name == p:
                    fid_vals[p] = float(q.value); break
            else:
                fid_vals[p] = 1.0

    data = np.concatenate(bundle["obs_data"])
    n_data = data.size

    J = np.zeros((n_data, len(nl_params)), dtype=np.float64)
    for i, p in enumerate(nl_params):
        dp = steps[p]
        plus = predict({**fid_vals, p: fid_vals[p] + dp})
        minus = predict({**fid_vals, p: fid_vals[p] - dp})
        J[:, i] = (plus - minus) / (2.0 * dp)

    BB = bb_basis(bundle["obs_ells"], bundle["obs_s"], n_data, powers=(0, 2))
    J_full = np.concatenate([J, BB], axis=1)
    F = J_full.T @ C_inv @ J_full

    # Apply DESI Σ priors on Σ_par, Σ_⊥, σ_s (N(loc=fid, σ=2)).
    for i, p in enumerate(nl_params):
        if p in _DESI_PRIOR_SCALE:
            F[i, i] += 1.0 / _DESI_PRIOR_SCALE[p] ** 2

    # Schur-marginalize down to q params.
    idx_keep = [0] if apmode == "qiso" else [0, 1]
    idx_marg = [i for i in range(F.shape[0]) if i not in idx_keep]
    F_kk = F[np.ix_(idx_keep, idx_keep)]
    F_km = F[np.ix_(idx_keep, idx_marg)]
    F_mm = F[np.ix_(idx_marg, idx_marg)]
    F_marg = F_kk - F_km @ np.linalg.solve(F_mm, F_km.T)
    cov_q = np.linalg.inv(F_marg)

    # Transform σ_q → σ(D/rd) using OUR template fid values.
    DH = float(tmpl.DH_over_rd_fid)
    DM = float(tmpl.DM_over_rd_fid)
    DV = float(tmpl.DV_over_rd_fid)

    if apmode == "qiso":
        sig_q = float(np.sqrt(cov_q[0, 0]))
        # In qiso mode, σ(DH/rd) and σ(DM/rd) are formally σ_qiso × (D/rd)_fid each
        # (with the AP degeneracy meaning these track 1:1 along iso direction).
        sig_DH = sig_q * DH; sig_DM = sig_q * DM; sig_DV = sig_q * DV
    else:
        # σ(DH/rd) = σ(qpar) · DH/rd; σ(DM/rd) = σ(qper) · DM/rd.
        # σ²(DV/rd) via AP transform from cov(qpar, qper).
        sig_qpar = float(np.sqrt(cov_q[0, 0]))
        sig_qper = float(np.sqrt(cov_q[1, 1]))
        sig_DH = sig_qpar * DH
        sig_DM = sig_qper * DM
        # σ²(ln DV) = (1/3)² var(qpar) + (2/3)² var(qper) + 2·(1/3)·(2/3)·cov(qpar, qper)
        var_lnDV = (cov_q[0, 0] / 9.0) + (4.0 * cov_q[1, 1] / 9.0) + (4.0 * cov_q[0, 1] / 9.0)
        sig_DV = float(np.sqrt(max(var_lnDV, 0.0))) * DV

    return {"DH_over_rs": sig_DH, "DM_over_rs": sig_DM, "DV_over_rs": sig_DV,
            "DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV,
            "apmode": apmode, "cov_q": cov_q}


# ===========================================================================
# §3.5  Multi-cosmology σ-triplet GENERATOR  (emulator sample driver)
# ===========================================================================
# Production driver for emulator training samples: θ → σ(DH/rd, DM/rd, DV/rd),
# with the analytic Grieb Gaussian cov as the forecast default (no DESI-data
# dependence). The DESI bundle cov is NOT usable here — it is a single fixed
# realization at one cosmology — so the generator is purely first-principles.
#
# Frame-fixed survey geometry — window W, FKP volume/n̄ slices, the s-grids, and
# the spherical-Bessel basis — is precomputed ONCE at the fiducial frame and
# reused for every cosmology. Cosmology enters the cov ONLY through the template
# P_ℓ(k) (+ AP via the q parameters in the Fisher); the volume/n̄ and window are
# the SAME comoving-frame object and are held at fiducial together (see the
# frame-consistency discussion — varying one without the other mixes frames).
#
# Per-cosmology cost (profiled, LRG2): lean template build (~0.45 s) + Grieb cov
# (per_mode_sigma2 + cached-Bessel matmuls, ~0.1 s) + native-ξ Fisher Jacobian.
class XiSigmaGenerator:
    """Per-tracer σ-triplet generator. Construct once (precomputes the fixed
    survey geometry + Bessel basis), then call ``sigma_triplet(**cosmo)`` for
    each sampled cosmology. Thread the same instance across a cosmology sweep."""

    def __init__(self, tracer):
        self.tracer = tracer
        self.cfg = TRACER_CONFIGS[tracer]
        self.apmode = "qiso" if tracer in ("BGS", "QSO") else "qparqper"
        self.bundle = load_bundle(tracer)

        # FKP volume/n̄ at the FIDUCIAL frame (frame-fixed, like the window —
        # NOT varied per cosmology; see the §3.5 header note).
        theta_fid, _ = prep_covar._to_bao_cosmo_params(
            {**prep_covar.PARAM_DEFAULTS, **_FID})
        cosmo_fid = prep_covar.get_cosmo(("DESI", dict(theta_fid)))
        self.slices = load_nz_slices(
            tracer_bin=tracer, cosmo=cosmo_fid, area_deg2=_AREA,
            N_design=float(_get_ntracers(tracer)))

        # Theory s-grid + window: the windowed (theory-grid) cov is what feeds
        # the Fisher, so only that grid's Bessel basis is needed.
        th_s = self.bundle["th_s"]
        s_th = np.asarray(th_s[0], dtype=np.float64)
        for arr in th_s[1:]:
            if not np.allclose(arr, s_th):
                raise ValueError("bundle th_s differs per ell")
        self.s_th = s_th
        self.th_ells = list(self.bundle["th_ells"])
        self.ds_th = float(np.mean(np.diff(s_th)))
        self.W = self.bundle["W"]
        # Cosmology-independent Bessel basis on (s_th, K_WIDE, ds_th): built ONCE.
        self.jbar_cache = build_jbar_cache(self.th_ells, K_WIDE, s_th, self.ds_th)

    def windowed_cov(self, info):
        """Grieb windowed Gaussian cov C = W @ C_theory @ Wᵀ via the cached
        Bessel basis. Only the windowed cov is built (the obs-grid cov isn't
        used by the Fisher)."""
        P_wide = _wide_Pell(self.tracer, info)
        C_theory = gaussian_xi_multipole_cov(
            s=self.s_th, ells_obs=tuple(self.th_ells), k=K_WIDE,
            P_ells_in=P_wide, ells_in=(0, 2, 4), slices=self.slices,
            ds=self.ds_th, jbar_cache=self.jbar_cache)
        C = self.W @ C_theory @ self.W.T
        return 0.5 * (C + C.T)

    def sigma_triplet(self, **cosmo_overrides):
        """θ → {DH_over_rs, DM_over_rs, DV_over_rs, ...}. Pass cosmology as
        keyword overrides on PARAM_DEFAULTS, e.g. ``sigma_triplet(Om=0.31,
        hrdrag=99.5)``; no args → fiducial."""
        sample = {**prep_covar.PARAM_DEFAULTS, **_FID, **cosmo_overrides}
        theta_cosmo, hrdrag = prep_covar._to_bao_cosmo_params(sample)
        info = prep_covar.build_bao_likelihood(
            N_tracers=_get_ntracers(self.tracer), theta_cosmo=theta_cosmo,
            hrdrag=hrdrag, tracer_bin=self.tracer, zrange=self.cfg["zrange"],
            z_eff=float(self.cfg["z_eff"]), area=_AREA, apmode=self.apmode,
            lean=True)
        cov = self.windowed_cov(info)
        return bundle_fisher_sigmas(self.tracer, cov_override=cov,
                                    info=info, bundle=self.bundle)


# ===========================================================================
# §4  Native ξ-theory + MCMC  (the config-space posterior estimator DESI uses)
# ===========================================================================
def build_native_theory_mcmc(tracer, apmode):
    """Build (info, native ξ-theory, bundle) for the MCMC path.

    Uses the same fiducial (_FID) as the Fisher / Grieb-cov path.
    """
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, **_FID})
    info = prep_covar.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_AREA, apmode=apmode,
    )
    info["likelihood"](**info["params"])
    template = info["template"]
    bundle = load_bundle(tracer)
    smoothing = float(cfg["smoothing_scale"])

    # Use bundle's theory s grid; output ells=0,2,4 to match bundle W layout.
    native = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=bundle["th_s"][0], ells=tuple(bundle["th_ells"]),
        template=template, mode="recsym", smoothing_radius=smoothing,
        model="fog-damping",
    )
    return info, native, bundle


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


# DESI Adame+24 canonical prior centers (used by BAOFit reductions)
_DESI_CANONICAL_CENTERS = {
    "BGS":       {"sigmapar": 8.0, "sigmaper": 3.0, "sigmas": 2.0},
    "QSO":       {"sigmapar": 9.0, "sigmaper": 3.0, "sigmas": 2.0},
    "LRG1":      {"sigmapar": 9.0, "sigmaper": 4.5, "sigmas": 2.0},
    "LRG2":      {"sigmapar": 9.0, "sigmaper": 4.5, "sigmas": 2.0},
    "LRG3_ELG1": {"sigmapar": 8.5, "sigmaper": 4.5, "sigmas": 2.0},
    "ELG2":      {"sigmapar": 8.5, "sigmaper": 4.5, "sigmas": 2.0},
}


def make_log_prob(info, native, bundle, apmode, tracer,
                  bb_powers=(-3, -2, -1, 0, 1),
                  alpha_low=0.8, alpha_high=1.2,
                  desi_canonical_priors=False,
                  freeze_nuisance=False,
                  bb_basis="default"):
    if bb_basis == "desi_adame24":
        bb_powers = (0, 2)
    data = np.concatenate(bundle["obs_data"])
    n_data = data.size
    cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
    C_inv = np.linalg.inv(cov_sym)
    BB = globals()["bb_basis"](bundle["obs_ells"], bundle["obs_s"], n_data, powers=bb_powers)

    if freeze_nuisance:
        if apmode == "qiso":
            param_names = ["qiso", "b1"]
        else:
            param_names = ["qpar", "qper", "b1"]
    elif apmode == "qiso":
        # ξ₀-only data cannot constrain RSD. Match desilike's auto-fix
        # for ells=(0,) (bao.py:85): drop dbeta from the parameter space
        # entirely. Otherwise its weak window-induced coupling to ξ₀ inflates
        # qiso's Schur-marginalized uncertainty in the Fisher comparison
        # without changing the true posterior on qiso. DESI Adame+24 takes
        # the same approach for sparse/qiso fits.
        param_names = ["qiso", "b1", "sigmas", "sigmapar", "sigmaper"]
    else:
        param_names = ["qpar", "qper", "b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]

    # Build fid for ALL relevant params (sampled or pinned), so freeze_nuisance can pin to fid.
    full_names = ["qiso", "qpar", "qper", "b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]
    fid = {}
    for p in full_names:
        if p in info["params"]:
            fid[p] = float(info["params"][p])
        else:
            for q in info["likelihood"].all_params:
                if q.name == p:
                    fid[p] = float(q.value); break

    if desi_canonical_priors:
        overrides = _DESI_CANONICAL_CENTERS.get(tracer, {})
        for p, val in overrides.items():
            if p in fid:
                fid[p] = float(val)

    # Prior config aligned with Fisher (`bundle_fisher_sigmas`) + DESI Adame+24:
    #   - Σ_par, Σ_⊥, σ_s: Gaussian σ=2 about fid (DESI Adame+24 §4.2.1 canonical)
    #   - b1, dbeta: NO informative prior; hard bounds set wide enough to be
    #     fully inactive at >10σ from any posterior in this analysis.
    #   - q: flat in [alpha_low, alpha_high]
    # Goal: MCMC posterior σ should differ from Fisher σ only via non-Gaussianity
    # of the likelihood, NOT via the prior shape. So bounds are sized to never
    # truncate the posterior tails. Earlier dbeta bound [-5, 5] was active for
    # sparse tracers (qiso, ξ₀-only) where the data weakly constrains RSD;
    # widened to [-50, 50] makes it inactive while keeping the sampler stable.
    prior_widths = {"sigmas": 2.0, "sigmapar": 2.0, "sigmaper": 2.0}
    sigma_hard_max = {"sigmas": 50.0, "sigmapar": 50.0, "sigmaper": 50.0}

    def log_prior(theta):
        d = dict(zip(param_names, theta))
        for q in ("qiso", "qpar", "qper"):
            if q in d and not (alpha_low < d[q] < alpha_high):
                return -np.inf
        for p, hi in sigma_hard_max.items():
            # Σ positivity is a physical requirement (negative damping = nonsense);
            # 1e-3 floor is for numerical safety, never active in practice.
            if p in d and not (1e-3 < d[p] < hi): return -np.inf
        if "b1" in d and not (0.05 < d["b1"] < 20.0): return -np.inf
        if "dbeta" in d and not (-50.0 < d["dbeta"] < 50.0): return -np.inf
        lp = 0.0
        for p, w in prior_widths.items():
            if p in d:
                lp += -0.5 * ((d[p] - fid[p]) / w) ** 2
        return lp

    # All nuisances we want to pin to fid when frozen (or pass-through when sampled).
    all_nuisances = ["dbeta", "sigmas", "sigmapar", "sigmaper"]

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        params = {**info["params"]}
        for nu in all_nuisances:
            if nu not in params and nu in fid:
                params[nu] = fid[nu]
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


def run_mcmc(tracer, apmode, nwalkers, max_iterations, burnin_frac, seed,
             alpha_low, alpha_high, desi_canonical_priors=False,
             freeze_nuisance=False, bb_basis="default"):
    tags = []
    if desi_canonical_priors: tags.append("DESI canonical Σ priors")
    if freeze_nuisance: tags.append("Σ/σ_s/dbeta FROZEN to fid")
    if bb_basis != "default": tags.append(f"BB={bb_basis}")
    tag = f" [{', '.join(tags)}]" if tags else ""
    print(f"\n{'=' * 70}\n  {tracer}  (apmode={apmode}, NATIVE ξ-theory){tag}\n{'=' * 70}", flush=True)
    info, native, bundle = build_native_theory_mcmc(tracer, apmode)
    log_prob, param_names, fid = make_log_prob(
        info, native, bundle, apmode, tracer,
        alpha_low=alpha_low, alpha_high=alpha_high,
        desi_canonical_priors=desi_canonical_priors,
        freeze_nuisance=freeze_nuisance,
        bb_basis=bb_basis)
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
    if desi_canonical_priors:
        suffix += "_desipriors"
    if freeze_nuisance:
        suffix += "_frozen"
    if bb_basis != "default":
        suffix += f"_bb{bb_basis}"
    out_path = _OUT_DIR / f"{tracer}_dr1_native_theory{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


def _mcmc_cli():
    ap = argparse.ArgumentParser(description="Native ξ-theory MCMC on bundle data.")
    ap.add_argument("--tracers", nargs="+", default=["BGS"])
    ap.add_argument("--apmode", choices=["qparqper", "qiso"], default="qiso")
    ap.add_argument("--nwalkers", type=int, default=48)
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-low", type=float, default=0.8)
    ap.add_argument("--alpha-high", type=float, default=1.2)
    ap.add_argument("--desi-canonical-priors", action="store_true",
                    help="Override Σ_par,Σ_⊥,σ_s prior centers to Adame+24 canonical values")
    ap.add_argument("--freeze-nuisance", action="store_true",
                    help="Pin Σ_par,Σ_⊥,σ_s,dbeta to fid (only q,b1 are sampled)")
    ap.add_argument("--bb-basis", choices=["default", "desi_adame24"], default="default",
                    help="BB basis: 'default' (powers=-3,-2,-1,0,1) or 'desi_adame24' (powers=0,2)")
    args = ap.parse_args()
    for t in args.tracers:
        try:
            run_mcmc(t, args.apmode, args.nwalkers, args.max_iterations,
                     args.burnin_frac, args.seed, args.alpha_low, args.alpha_high,
                     desi_canonical_priors=args.desi_canonical_priors,
                     freeze_nuisance=args.freeze_nuisance,
                     bb_basis=args.bb_basis)
        except Exception as e:
            print(f"[{t}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Default __main__ runs the cheap shot-noise normalization self-check.
    # The heavy MCMC CLI is reachable via mcmc_bundle_native_theory.py (shim)
    # or by calling _mcmc_cli().
    ok = _validate_shotnoise_limit()
    sys.exit(0 if ok else 1)
