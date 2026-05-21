import os

# Set display/backend env vars before any library imports can trigger X11 or GPU probes.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import traceback
import warnings
from typing import Dict, List, Optional, Tuple
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import erf
from pathlib import Path
import pandas as pd

# In spawned worker processes, additionally suppress all C-level stderr.
if os.environ.get("_PREP_COVAR_WORKER") == "1":
    os.environ.pop("DISPLAY", None)
    warnings.filterwarnings("ignore")
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# velocileptors uses np.trapezoid (numpy >= 2.0); shim for numpy 1.x.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# Suppress desilike import-time warnings (e.g. missing interpax/jax) before importing
warnings.filterwarnings("ignore")

from desilike import Fisher
from desilike.likelihoods.galaxy_clustering import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    CutskyFootprint,
    ObservablesCovarianceMatrix,
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.theories.galaxy_clustering import (
    BAOPowerSpectrumTemplate,
    DampedBAOWigglesTracerPowerSpectrumMultipoles,
)
from desilike.theories.primordial_cosmology import get_cosmo

from util import (
    HOD_CONFIGS,
    TRACER_CONFIGS,
    TRACER_TYPE_CHOICES,
    get_default_save_path,
    get_tracer_config,
    latin_hypercube_samples,
    parse_priors,
    save_dataset,
)

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# Priors matching experiments/num_tracers/prior_args.yaml, plus N_tracers.
DEFAULT_PRIORS = {
    "N_tracers": {"dist": "uniform", "low": 1e5, "high": 1e7},
    "Om": {"dist": "uniform", "low": 0.01, "high": 0.99},
    "Ok": {"dist": "uniform", "low": -0.3, "high": 0.3},
    "w0": {"dist": "uniform", "low": -3.0, "high": 1.0},
    "wa": {"dist": "uniform", "low": -3.0, "high": 2.0},
    "hrdrag": {"dist": "uniform", "low": 10.0, "high": 1000.0},
}

# Constraints from prior_args.yaml.
CONSTRAINTS = {
    "valid_densities": {"params": ["Om", "Ok"], "lower": 0.0, "upper": 1.0},
    "high_z_matter_dom": {"params": ["w0", "wa"], "upper": 0.0},
}

COSMO_MODELS = {
    "base":              ["Om", "hrdrag"],
    "base_w":            ["Om", "w0", "hrdrag"],
    "base_w_wa":         ["Om", "w0", "wa", "hrdrag"],
    "base_omegak":       ["Om", "Ok", "hrdrag"],
    "base_omegak_w_wa":  ["Om", "Ok", "w0", "wa", "hrdrag"],
}

# Fiducial values for fixed parameters
PARAM_DEFAULTS = {"Ok": 0.0, "w0": -1.0, "wa": 0.0}

# DESI fiducial values for parameters that set the power spectrum shape
# but are not varied in the BAO prior.
_OMEGA_B_FID = 0.02237
_MNU_FID = 0.06  # sum of neutrino masses in eV (1 massive neutrino)
_OMEGA_NU_FID = _MNU_FID / 93.14  # neutrino density parameter
_N_S_FID = 0.9649
_LN10A_S_FID = 3.044
_H_FID = 0.6736

_PHYS_NAMES = ["DH_over_rd", "DM_over_rd"]
_TRIU_I, _TRIU_J = np.triu_indices(2)
TARGET_NAMES = [f"cov_{_PHYS_NAMES[i]}_{_PHYS_NAMES[j]}" for i, j in zip(_TRIU_I, _TRIU_J)]

# Integration settings for displacement / reconstruction model
_K_DISP_MIN = 1.0e-4
_K_DISP_MAX = 10.0
_NK_DISP = 512
_NMU_DISP = 64

# Random-to-data catalog ratio assumed in post-reconstruction shot-noise floor.
# Enters the covariance as (1 + 1/alpha) / nbar instead of 1/nbar. A convention,
# not a fit: DESI pipelines use alpha in [20, 50]; 50 is conservative.
_N_RAND_OVER_N_DATA = 50.0

# Path to the cosmology-independent n(z) slices produced by parse_desi_nz.py.
_NZ_SLICES_DIR = Path.home() / "data" / "desi" / "nz_slices"


def _load_nz_slice_fractions(tracer_bin: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the cosmology-independent n(z) shape for a tracer.

    Returns (z_centers, z_edges, slice_fraction, nbar_file) where:
      - slice_fraction sums to ~1 across the bin
      - nbar_file is the published local 3D number density (Mpc/h)^-3 per
        redshift slice, computed at the file's effective area. This is the
        actual local density of the tracer at that redshift, suitable for
        FKP weighting and shot-noise calculation.
    """
    path = _NZ_SLICES_DIR / f"{tracer_bin}_nz_slices.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing n(z) slice file for {tracer_bin}: {path}")
    df = pd.read_csv(path)
    df = df[df["slice_fraction"] > 0.0].reset_index(drop=True)
    z_lo = df["zlow"].to_numpy(dtype=np.float64)
    z_hi = df["zhigh"].to_numpy(dtype=np.float64)
    z_mid = df["zmid"].to_numpy(dtype=np.float64)
    frac = df["slice_fraction"].to_numpy(dtype=np.float64)
    frac = frac / frac.sum()
    # Use nbar_file column if available (published local n̄ from the n(z) file)
    nbar_file = df["nbar_file"].to_numpy(dtype=np.float64) if "nbar_file" in df.columns \
                else np.zeros_like(frac)
    return z_mid, np.column_stack([z_lo, z_hi]), frac, nbar_file


def _compute_z_eff_from_nz(
    tracer_bin: str,
    cosmo,
    fo,
    area_deg2: float,
    b1: float,
    k_pivot: float = 0.14,
) -> float:
    """Fisher-info-weighted effective redshift from the n(z) slices.

    z_eff = Σᵢ zᵢ Vᵢ_eff / Σᵢ Vᵢ_eff,    Vᵢ_eff = Vᵢ × (nᵢ Pᵢ / (1 + nᵢ Pᵢ))²

    with V_i = comoving shell volume of slice i, n_i = nbar_file at slice i,
    P_i = b1² P_lin(k_p, z_i). The (nP/(1+nP))² factor is the FKP weight squared
    that defines V_eff_i (the BAO-mode constraining volume contributed by the
    slice). z_eff is then the V_eff-weighted mean of slice midpoints.

    In the dense-tracer limit (nP ≫ 1), this collapses to V-weighting →
    matches the volume-weighted slice mean. In the sparse-tracer limit
    (nP ≪ 1) it becomes V × n² P²-weighting → up-weights the slices where
    the tracer is densest. Matches DESI Adame+24 Sec 2.5 convention up to
    a slowly-varying overall prefactor that does not change the weighted mean.

    k_pivot ≈ √2/Σ_silk ≈ 0.14 h/Mpc is the BAO Fisher kernel peak.
    """
    z_mid, z_edges, _frac, nbar_file = _load_nz_slice_fractions(tracer_bin)
    if z_mid.size == 0:
        raise ValueError(f"No valid n(z) slices for tracer {tracer_bin}")

    z_lo = z_edges[:, 0]
    z_hi = z_edges[:, 1]
    chi_lo = np.array([float(cosmo.comoving_radial_distance(z)) for z in z_lo])
    chi_hi = np.array([float(cosmo.comoving_radial_distance(z)) for z in z_hi])
    sky_frac = float(area_deg2) / 41252.96
    V_bin = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * sky_frac

    P_g = np.empty_like(z_mid, dtype=np.float64)
    for i, z in enumerate(z_mid):
        pk = _linear_pk_1d(fo, z=float(z))
        P_g[i] = float(b1) ** 2 * float(pk(np.array([k_pivot]))[0])

    nP = np.asarray(nbar_file, dtype=np.float64) * P_g
    fkp_sq = (nP / (1.0 + nP)) ** 2
    V_eff_per_slice = V_bin * fkp_sq

    if V_eff_per_slice.sum() <= 0.0:
        # Degenerate FKP weights (e.g. nbar_file all zero); fall back to V-weighted mean.
        return float(np.sum(V_bin * z_mid) / np.sum(V_bin))
    return float(np.sum(z_mid * V_eff_per_slice) / np.sum(V_eff_per_slice))


# BAO Fisher information kernel for the V_eff band integral.
#
# Per-tracer V_eff for BAO σ(α) forecasting is the survey volume weighted by
# the band-averaged FKP signal-to-noise:
#
#     V_eff = ∫ d³r [n(r)P(k)/(1+n(r)P(k))]²
#
# where the relevant k-band weight is the BAO Fisher kernel
#
#     dF/dk ∝ k² × |dlnP_wig/dlnα|² × (n̄P/(1+n̄P))²
#           ∝ k² × (k r_d cos(k r_d))² × exp(-k²Σ²) × (n̄P/(1+n̄P))²
#           ≈ k⁴ × exp(-k²Σ²) × (n̄P/(1+n̄P))²
#
# after averaging the rapid BAO oscillation. Σ = Σ_silk_eff ≈ 10 Mpc/h gives
# the right BAO peak location; the exact value matters at the ~few-percent
# level because the weight is broad. Evaluating at a single k_BAO=0.14
# (the previous behavior) ignores the high V_eff at low k where P_lin grows
# — this is why sparse low-bias tracers (ELG2, QSO) were so over-pessimistic.
_BAO_K_GRID = np.geomspace(0.01, 0.5, 200)  # h/Mpc, log-spaced
_BAO_SIGMA_KERNEL = 10.0  # Mpc/h, BAO Fisher info envelope (Silk-damped k r_d²)
_BAO_W_K2 = _BAO_K_GRID ** 4 * np.exp(-(_BAO_K_GRID * _BAO_SIGMA_KERNEL) ** 2)
_BAO_W_K2_NORM = float(np.trapezoid(_BAO_W_K2, _BAO_K_GRID))


def _fkp_band_weight_sq(nbar_3d: float, P_gk: np.ndarray) -> float:
    """BAO-Fisher-weighted ⟨(n̄P(k)/(1+n̄P(k)))²⟩ for one redshift slice.

    P_gk is the galaxy power at k = _BAO_K_GRID. The returned value is in
    [0,1] and is the effective fraction of the shell volume that contributes
    to the BAO-mode covariance at this slice.
    """
    nP = nbar_3d * np.asarray(P_gk, dtype=np.float64)
    fkp_sq = (nP / (1.0 + nP)) ** 2
    return float(np.trapezoid(_BAO_W_K2 * fkp_sq, _BAO_K_GRID) / _BAO_W_K2_NORM)


def _compute_v_eff_fkp(
    cosmo,
    area_deg2: float,
    tracer_bin: str,
    fkp_weight_sq_per_bin: np.ndarray,
) -> Tuple[float, float]:
    """Compute FKP-weighted V_eff at the given cosmology.

    Returns (V_eff, V_shell) in (Mpc/h)^3.

    fkp_weight_sq_per_bin is the band-averaged ⟨(n̄P/(1+n̄P))²⟩ per
    redshift slice, computed by the caller via _fkp_band_weight_sq using
    cosmology-driven P_g(k,z) on the _BAO_K_GRID.
    """
    z_mid, z_edges, frac, _ = _load_nz_slice_fractions(tracer_bin)
    z_lo = z_edges[:, 0]
    z_hi = z_edges[:, 1]
    sky_frac = float(area_deg2) / 41252.96
    # cosmoprimo's comoving_radial_distance returns chi in Mpc/h already.
    chi_lo = np.asarray(cosmo.comoving_radial_distance(z_lo))
    chi_hi = np.asarray(cosmo.comoving_radial_distance(z_hi))
    V_bin = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * sky_frac
    V_shell = float(V_bin.sum())
    V_eff = float(np.sum(V_bin * np.asarray(fkp_weight_sq_per_bin)))
    return V_eff, V_shell


def _sample_constrained_pair(
    low1: float, high1: float,
    low2: float, high2: float,
    n: int,
    rng: np.random.Generator,
    sum_lower: float | None = None,
    sum_upper: float | None = None,
) -> np.ndarray:
    """Sample n points uniformly from a 2D box with a linear sum constraint."""
    out = np.empty((n, 2), dtype=np.float64)
    filled = 0
    while filled < n:
        batch = rng.uniform(
            [low1, low2], [high1, high2], size=(max(n - filled, 256) * 3, 2)
        )
        valid = np.ones(len(batch), dtype=bool)
        if sum_lower is not None:
            valid &= (batch[:, 0] + batch[:, 1]) > sum_lower
        if sum_upper is not None:
            valid &= (batch[:, 0] + batch[:, 1]) < sum_upper
        good = batch[valid]
        take = min(len(good), n - filled)
        out[filled : filled + take] = good[:take]
        filled += take
    return out


def _constrained_samples(
    priors: Dict[str, Dict[str, float]],
    constraints: Dict[str, Dict],
    n_samples: int,
    seed: int,
    sigma_clip: float = 4.0,
) -> List[Dict[str, float]]:
    """Draw samples with LHS for unconstrained params and rejection for constrained pairs."""
    rng = np.random.default_rng(seed)

    constrained_keys: set = set()
    for spec in constraints.values():
        constrained_keys.update(spec["params"])

    unconstrained_priors = {k: v for k, v in priors.items() if k not in constrained_keys}
    if unconstrained_priors:
        lhs_draws = latin_hypercube_samples(
            unconstrained_priors, n_samples=n_samples, seed=seed, sigma_clip=sigma_clip,
        )
    else:
        lhs_draws = [{}] * n_samples

    pair_samples: Dict[str, np.ndarray] = {}
    for _, cspec in constraints.items():
        p1, p2 = cspec["params"]
        pair = _sample_constrained_pair(
            low1=priors[p1]["low"], high1=priors[p1]["high"],
            low2=priors[p2]["low"], high2=priors[p2]["high"],
            n=n_samples,
            rng=rng,
            sum_lower=cspec.get("lower"),
            sum_upper=cspec.get("upper"),
        )
        pair_samples[p1] = pair[:, 0]
        pair_samples[p2] = pair[:, 1]

    rows: List[Dict[str, float]] = []
    for i in range(n_samples):
        row = dict(lhs_draws[i])
        for k, arr in pair_samples.items():
            row[k] = float(arr[i])
        rows.append(row)
    return rows


def _load_nz_slices(nz_slices_path: str | Path) -> pd.DataFrame:
    """Load a parsed *_nz_slices.csv file produced by parse_desi_nz.py."""
    path = Path(nz_slices_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find n(z) slices file: {path}")

    df = pd.read_csv(path)

    required = {"zmid", "zlow", "zhigh", "slice_fraction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df = df[df["slice_fraction"] > 0].reset_index(drop=True)

    frac_sum = float(df["slice_fraction"].sum())
    if not np.isfinite(frac_sum) or frac_sum <= 0:
        raise ValueError(f"Bad slice_fraction sum in {path}: {frac_sum}")

    # Make robust against small CSV/rounding errors.
    df["slice_fraction"] = df["slice_fraction"] / frac_sum

    return df


def _regularize_fisher_matrix(F_matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and add tiny diagonal jitter if needed."""
    F_matrix = np.asarray(F_matrix, dtype=np.float64)
    F_matrix = 0.5 * (F_matrix + F_matrix.T)

    eigvals = np.linalg.eigvalsh(F_matrix)
    min_eig = float(eigvals.min())
    if min_eig <= 0:
        diag_scale = float(np.mean(np.diag(F_matrix)))
        jitter = max(abs(min_eig), 1e-12 * abs(diag_scale), 1e-30)
        F_matrix = F_matrix + np.eye(F_matrix.shape[0]) * jitter

    return F_matrix


def _cov_q_from_fisher_matrix(
    F_matrix: np.ndarray,
    all_names: list[str],
    bao_internal: list[str] | None = None,
) -> np.ndarray:
    """
    Return the marginalized covariance of qpar/qper from a full Fisher matrix.

    This marginalizes over all nuisance parameters in the slice.
    """
    if bao_internal is None:
        bao_internal = ["qpar", "qper"]

    F_matrix = _regularize_fisher_matrix(F_matrix)
    bao_idx = [all_names.index(p) for p in bao_internal]

    E = np.zeros((F_matrix.shape[0], len(bao_idx)))
    for j, idx in enumerate(bao_idx):
        E[idx, j] = 1.0

    try:
        c, low = cho_factor(F_matrix, check_finite=False)
        cov_subset = cho_solve((c, low), E, check_finite=False)
        cov_q = cov_subset[np.ix_(bao_idx, range(len(bao_idx)))]
    except np.linalg.LinAlgError:
        cov_full = np.linalg.pinv(F_matrix, rcond=1e-12)
        cov_q = cov_full[np.ix_(bao_idx, bao_idx)]

    cov_q = 0.5 * (cov_q + cov_q.T)
    return cov_q


def _q_fisher_from_bao_likelihood_info(info: Dict) -> np.ndarray:
    """
    Build Fisher for one likelihood and return marginalized 2x2 Fisher
    information on qpar/qper.
    """
    likelihood = info["likelihood"]
    params = info["params"]

    fisher = Fisher(likelihood)
    fisher_result = fisher(**params)

    F_full = -np.array(fisher_result._hessian)
    all_names = [str(p) for p in fisher_result.names()]

    cov_q = _cov_q_from_fisher_matrix(F_full, all_names, bao_internal=["qpar", "qper"])

    # Convert marginalized covariance back to marginalized information.
    # This is equivalent to the Schur-complement Fisher on qpar/qper.
    try:
        F_q = np.linalg.inv(cov_q)
    except np.linalg.LinAlgError:
        F_q = np.linalg.pinv(cov_q, rcond=1e-12)

    F_q = 0.5 * (F_q + F_q.T)
    return F_q


def _to_bao_cosmo_params(sample: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Convert BAO prior parameters to desilike cosmology parameters.

    Fixes h to the DESI fiducial value, matching the DESI BAO parameterization
    where h is fixed and hrdrag is a free parameter entering through rd = hrdrag / h.
    """
    Om = float(sample["Om"])
    omega_cdm = Om * _H_FID**2 - _OMEGA_NU_FID - _OMEGA_B_FID
    if omega_cdm <= 0:
        raise ValueError(
            f"Om={Om:.4f} too small: omega_cdm={omega_cdm:.4f} < 0 "
            f"(need Om > {(_OMEGA_B_FID + _OMEGA_NU_FID) / _H_FID**2:.4f})"
        )

    theta_cosmo = {
        "Omega_m": Om,
        "Omega_k": float(sample["Ok"]),
        "w0_fld": float(sample["w0"]),
        "wa_fld": float(sample["wa"]),
        "h": _H_FID,
        "omega_b": _OMEGA_B_FID,
        "n_s": _N_S_FID,
        "logA": _LN10A_S_FID,
    }
    return theta_cosmo, float(sample["hrdrag"])


def _linear_pk_1d(fo, z: float):
    """
    Return a callable P_lin(k, z) in (Mpc/h)^3 for delta_cb.
    """
    pk = fo.pk_interpolator(of="delta_cb").to_1d(z=z)
    return lambda k: np.asarray(pk(k), dtype=np.float64)


def _sigma_v_sq_1loop_rsd(pk_lin_1d) -> Tuple[float, float, float]:
    """
    Return the three 1-loop Lagrangian displacement-variance pieces needed to
    build the full 1-loop BAO damping in redshift space.

    Returns (sv2, sv2_dot, sv2_ddot) where
        sv2       = 0.5 * sigmaloop        (density-density, coeff of 1 in Sigma_par^2)
        sv2_dot   = 0.5 * sigmaloopdot     (density-velocity cross, coeff of 2f)
        sv2_ddot  = 0.5 * sigmaloopddot    (velocity-velocity, coeff of f^2)

    With these, the 1-loop contributions are
        Sigma_perp^2 += sv2
        Sigma_par^2  += sv2 + 2f * sv2_dot + f^2 * sv2_ddot.

    The naive (1+f)^2 boost on the density piece assumes sv2_dot = sv2_ddot = sv2,
    which is only exact at linear order. At 1-loop the velocity divergence P_theta-theta
    picks up stronger mode-coupling corrections than P_delta-delta, so sv2_dot and
    sv2_ddot differ from sv2 (typically sv2_dot ~ 2*sv2, sv2_ddot ~ 3*sv2).
    """
    from velocileptors.LPT.velocity_moments_fftw import VelocityMoments

    k = np.geomspace(1.0e-4, 20.0, 2000)
    pk = np.clip(pk_lin_1d(k), 0.0, None)
    vm = VelocityMoments(k, pk, one_loop=True, shear=False, third_order=False)
    return (
        0.5 * float(vm.sigmaloop),
        0.5 * float(vm.sigmaloopdot),
        0.5 * float(vm.sigmaloopddot),
    )


def _sigma_nl_pre_from_pk(
    pk_lin_1d,
    f: float,
    sv2_1loop: float = 0.0,
    sv2_1loop_dot: float = 0.0,
    sv2_1loop_ddot: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute pre-reconstruction BAO damping scales from the linear + 1-loop displacement
    variance, using the full 1-loop redshift-space decomposition.

    Linear:
        sv2_lin = (1/6 pi^2) int dk P_lin(k)
        Sigma_perp^2_lin = sv2_lin
        Sigma_par^2_lin  = (1+f)^2 sv2_lin     (exact, since sv2_dot = sv2_ddot = sv2 at lin)

    1-loop:
        Sigma_perp^2 += sv2_1loop
        Sigma_par^2  += sv2_1loop + 2f sv2_1loop_dot + f^2 sv2_1loop_ddot
    """
    k = np.geomspace(_K_DISP_MIN, _K_DISP_MAX, _NK_DISP)
    pk = np.clip(pk_lin_1d(k), 0.0, None)
    sv2_lin = float(np.trapz(pk, k) / (6.0 * np.pi**2))
    f = float(f)

    sigma_perp_sq = sv2_lin + float(sv2_1loop)
    sigma_par_sq = (
        (1.0 + f) ** 2 * sv2_lin
        + float(sv2_1loop)
        + 2.0 * f * float(sv2_1loop_dot)
        + f**2 * float(sv2_1loop_ddot)
    )
    sigma_perp_pre = np.sqrt(max(sigma_perp_sq, 0.0))
    sigma_par_pre = np.sqrt(max(sigma_par_sq, 0.0))
    return sigma_perp_pre, sigma_par_pre


def _lya_flux_bias(
    z: float,
    bF0: float = -0.117,
    z_ref: float = 2.334,
    gamma: float = 2.9,
) -> float:
    """Compute Ly-alpha flux bias b_F(z)."""
    return float(bF0) * ((1.0 + float(z)) / (1.0 + float(z_ref))) ** float(gamma)


def _sigma_nl_post_from_recon(
    pk_lin_1d,
    b1: float,
    f: float,
    nbar_comoving: float,
    smoothing_scale: float,
    sv2_1loop: float = 0.0,
    sv2_1loop_dot: float = 0.0,
    sv2_1loop_ddot: float = 0.0,
    n_iter: int = 1,
) -> Tuple[float, float]:
    """
    Anisotropic post-reconstruction residual BAO damping via 2D (k, mu) integration,
    plus the 1-loop redshift-space displacement variance that survives reconstruction.

    The Wiener filter uses the redshift-space galaxy power spectrum
    P_g(k, mu) = (b1 + f*mu^2)^2 P_lin(k), so reconstruction efficiency
    varies with mu (line-of-sight vs transverse).

    W(k, mu) = P_g(k, mu) / [P_g(k, mu) + 1/nbar]
    T_res(k, mu) = [1 - S(k) W(k, mu)]^n_iter

    The exponent n_iter models *iterative* reconstruction (White 2015,
    Schmittfull+2017). Each iteration of IFT recon re-filters the residual
    field, so after N iterations the residual transfer is approximately
    (1 - SW)^N assuming the Wiener filter is held fixed across iterations.
    DESI BAO uses N=3, which is the default here.

    Linear-theory residual:
        sigma_perp^2 = 1/(2 pi^2) int dk P(k) int_0^1 dmu (1-mu^2)/2 T_res^2
        sigma_par^2  = (1+f)^2 * 1/(2 pi^2) int dk P(k) int_0^1 dmu mu^2 T_res^2

    1-loop (beyond-Zel'dovich) contributions survive recon (recon is a linear-theory
    operation) and add the full redshift-space velocity variance:
        sigma_perp^2 += sv2_1loop
        sigma_par^2  += sv2_1loop + 2f sv2_1loop_dot + f^2 sv2_1loop_ddot

    The parallel 1-loop boost differs from the naive (1+f)^2 on sv2_1loop because
    P_theta-theta at 1-loop differs from P_delta-delta at 1-loop.

    FoG is NOT added in quadrature here. FoG is a Lorentzian streaming effect,
    not Gaussian BAO damping; it enters via the BAO observable's `sigmas`
    parameter [fog = 1 / (1 + (sigmas k mu)^2 / 2)^2] separately (see VEFF_NOTES.md).
    """
    if nbar_comoving <= 0:
        raise ValueError(f"nbar_comoving must be positive, got {nbar_comoving}")

    b1 = float(b1)
    f = float(f)
    nbar_inv = 1.0 / float(nbar_comoving)
    R = float(smoothing_scale)

    k = np.geomspace(_K_DISP_MIN, _K_DISP_MAX, _NK_DISP)
    mu = np.linspace(0.0, 1.0, _NMU_DISP)
    kk, mm = np.meshgrid(k, mu, indexing="ij")  # (NK, NMU)

    pk = np.clip(pk_lin_1d(kk), 0.0, None)
    S = np.exp(-0.5 * (kk * R) ** 2)
    Pg = (b1 + f * mm**2) ** 2 * pk
    W = Pg / (Pg + nbar_inv)
    T_res = (1.0 - S * W) ** int(n_iter)

    integrand_perp = pk * T_res**2 * (1.0 - mm**2) / 2.0
    sigma_perp_sq = (
        np.trapz(np.trapz(integrand_perp, mu, axis=1), k) / (2.0 * np.pi**2)
    )

    integrand_par = pk * T_res**2 * mm**2
    sigma_par_sq = (
        (1.0 + f) ** 2
        * np.trapz(np.trapz(integrand_par, mu, axis=1), k)
        / (2.0 * np.pi**2)
    )

    # Shot-noise-induced residual displacement variance per direction.
    # The reconstructed displacement is psi_rec ~ (-i k/k^2) S(k) delta_obs/b1,
    # and the shot-noise part of delta_obs has white spectrum 1/nbar. Squared
    # and angle-averaged this contributes
    #   <psi_x^2>_shot = (1/(6 pi^2 b1^2 nbar)) * int dk S^2(k)
    # per transverse direction. It is isotropic (no RSD coherence), so the
    # parallel direction gets the same shot contribution without the (1+f)^2
    # boost that signal carries.
    sigma_shot_sq = (
        np.trapz(S[:, 0] ** 2, k) / (6.0 * np.pi**2 * b1**2 * nbar_comoving)
    )
    sigma_perp_sq += sigma_shot_sq
    sigma_par_sq += sigma_shot_sq

    sv2_1loop = float(sv2_1loop)
    sv2_1loop_dot = float(sv2_1loop_dot)
    sv2_1loop_ddot = float(sv2_1loop_ddot)
    sigma_perp_sq += sv2_1loop
    sigma_par_sq += sv2_1loop + 2.0 * f * sv2_1loop_dot + f**2 * sv2_1loop_ddot

    sigma_perp_post = np.sqrt(max(float(sigma_perp_sq), 0.0))
    sigma_par_post = np.sqrt(max(float(sigma_par_sq), 0.0))
    return sigma_perp_post, sigma_par_post


def _tinker10_b1(nu: np.ndarray, delta_vir: float = 200.0) -> np.ndarray:
    """Peak-background-split bias from Tinker et al. 2010 (arXiv:1001.3162), Eq. (6)."""
    delta_c = 1.686
    y = np.log10(delta_vir)
    A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4))
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4))
    c = 2.4
    return 1.0 - A * nu**a / (nu**a + delta_c**a) + B * nu**b + C * nu**c


def _tinker08_f_sigma(sigma: np.ndarray, z: float, delta: float = 200.0) -> np.ndarray:
    """Tinker+2008 halo multiplicity function f(sigma) (arXiv:0803.2706, Eq. 3).

    Default delta=200 mean-density overdensity. Includes the mild z-evolution
    of A, a, b parameters (their Eqs. 5-8). c is z-independent.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    A0 = 0.186
    a0 = 1.47
    b0 = 2.57
    c = 1.19
    one_p_z = 1.0 + float(z)
    A = A0 * one_p_z ** (-0.14)
    a = a0 * one_p_z ** (-0.06)
    alpha = 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))
    b = b0 * one_p_z ** (-alpha)
    return A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c / sigma**2)


# Mass grid for HMF / abundance matching. 10^10 -- 10^16 M_sun/h is wide enough
# to bracket all DESI tracers (BGS galaxies ~10^12 to LRG/QSO hosts ~10^14).
_LOG_M_GRID_AM = np.linspace(10.0, 16.0, 256)


def _abundance_matched_halo_props(
    nbar_comoving: float,
    z: float,
    cosmo,
    fo,
    delta_vir: float = 200.0,
    log_M_grid: np.ndarray = _LOG_M_GRID_AM,
) -> Tuple[float, float]:
    """Abundance-match a target comoving number density to a halo mass threshold.

    Solves cumulative_n(>M_min) = nbar_comoving using the Tinker+2008 HMF, then
    returns (b1, log_M_min) where b1 = Tinker+2010 peak-background-split bias at
    M_min. Pure cosmology + n(z) input -- no DESI-derived bias values used.

    For low n̄ (rare tracers like QSO), M_min is large and b1 is large.
    For high n̄ (BGS), M_min is small and b1 ~ 1.
    """
    log_M = log_M_grid
    M = 10.0 ** log_M
    rho_m_com = float(cosmo.rho_m(0.0)) * 1.0e10  # M_sun/h per (Mpc/h)^3, comoving
    R = (3.0 * M / (4.0 * np.pi * rho_m_com)) ** (1.0 / 3.0)
    sigma = np.asarray(fo.sigma_rz(R, z, of="delta_cb"))
    sigma = np.maximum(sigma, 1.0e-30)

    f_sigma = _tinker08_f_sigma(sigma, z=z, delta=delta_vir)
    log_sigma_inv = -np.log(sigma)
    dlogM = log_M[1] - log_M[0]
    dlnsig_inv_dlnM = np.gradient(log_sigma_inv, log_M) / np.log(10.0)
    dn_dlnM = f_sigma * (rho_m_com / M) * dlnsig_inv_dlnM
    dn_dlnM = np.clip(dn_dlnM, 0.0, None)

    # Cumulative n(>M): integrate from each M to infinity.
    dlnM = dlogM * np.log(10.0)
    cumulative_n = np.cumsum(dn_dlnM[::-1] * dlnM)[::-1]

    # Invert: find log_M_min where cumulative_n(log_M_min) = nbar_comoving
    if nbar_comoving >= cumulative_n[0]:
        log_M_min = log_M[0]
    elif nbar_comoving <= cumulative_n[-1]:
        log_M_min = log_M[-1]
    else:
        # cumulative_n is monotonically decreasing in log_M; reverse for np.interp
        log_M_min = float(np.interp(
            np.log(max(nbar_comoving, 1.0e-30)),
            np.log(np.maximum(cumulative_n[::-1], 1.0e-300)),
            log_M[::-1],
        ))

    M_min = 10.0 ** log_M_min
    R_min = (3.0 * M_min / (4.0 * np.pi * rho_m_com)) ** (1.0 / 3.0)
    sigma_min = float(fo.sigma_rz(np.array([R_min]), z, of="delta_cb")[0])
    delta_c = 1.686
    nu_min = delta_c / max(sigma_min, 1.0e-30)
    b1 = float(_tinker10_b1(np.array([nu_min]), delta_vir=delta_vir)[0])
    return b1, log_M_min


def _sigma_fog_from_M_eff(
    M_eff: float,
    z: float,
    cosmo,
    delta_vir: float = 200.0,
) -> float:
    """FoG line-of-sight velocity-dispersion scale from a virial halo mass.

    sigma_v^2 = G M / (2 R_vir) (virial theorem, 1D dispersion). Convert to
    line-of-sight comoving damping scale in Mpc/h via sigma_FoG = sigma_v (1+z) / (100 E(z)).
    """
    rho_m_com = float(cosmo.rho_m(0.0)) * 1.0e10  # M_sun/h per (Mpc/h)^3, comoving
    R_vir_com = (3.0 * M_eff / (4.0 * np.pi * delta_vir * rho_m_com)) ** (1.0 / 3.0)
    R_vir_phys = R_vir_com / (1.0 + z)
    G_Mpc_Msun = 4.30091e-9
    sigma_v_sq_km_s = G_Mpc_Msun * M_eff / (2.0 * R_vir_phys)
    E_z = float(cosmo.efunc(z))
    return float(np.sqrt(sigma_v_sq_km_s) * (1.0 + z) / (100.0 * E_z))


# ---------------------------------------------------------------------------
# Tinker central/satellite HOD (replaces mass-threshold abundance matching).
#
# Shape parameters are tracer-type properties of galaxy formation -- they are
# cosmology-independent by construction and set once from the HOD literature.
# Cosmology enters via the HMF + Tinker+2010 b1 + the M_cut root-find that
# absorbs cosmology changes to keep the predicted n̄ matched to the data n̄.
#
# Refs:
#   Zheng+07  (2007ApJ...667..760Z)       -- 5-par HOD form (LRG-ish).
#   Reid+14   (2014MNRAS.444..476R)       -- LRG HOD priors.
#   Yuan+22   (2022MNRAS.515..871Y)       -- DESI-Y1 LRG HOD.
#   Avila+20  (2020MNRAS.499.5486A)       -- ELG HMQ-type incompleteness.
#   Rocher+23 (2023JCAP...10..016R)       -- DESI ELG HOD; small f_cen.
#   Richardson+12 (2012ApJ...755...30R)   -- QSO HOD (broad central erf).
#   Smith+20  (2020MNRAS.499..269S)       -- BGS HOD.
# ---------------------------------------------------------------------------

_HOD_SHAPE_PARAMS: Dict[str, Dict[str, float]] = HOD_CONFIGS  # see hod.yaml


def _hmf_arrays(z: float, cosmo, fo, delta_vir: float,
                log_M_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray, np.ndarray]:
    """Per-grid-point HMF, b1_T10, and virial sigma_v^2(M,z).

    Returns (M, dn_dlogM [(Mpc/h)^-3 per dex], b1_T10, sigma_v_sq_kms2, sigma).
    """
    log_M = log_M_grid
    M = 10.0 ** log_M
    rho_m_com = float(cosmo.rho_m(0.0)) * 1.0e10
    R = (3.0 * M / (4.0 * np.pi * rho_m_com)) ** (1.0 / 3.0)
    sigma = np.asarray(fo.sigma_rz(R, z, of="delta_cb"))
    sigma = np.maximum(sigma, 1.0e-30)

    f_sigma = _tinker08_f_sigma(sigma, z=z, delta=delta_vir)
    log_sigma_inv = -np.log(sigma)
    dlnsig_inv_dlnM = np.gradient(log_sigma_inv, log_M) / np.log(10.0)
    dn_dlnM = f_sigma * (rho_m_com / M) * dlnsig_inv_dlnM
    dn_dlnM = np.clip(dn_dlnM, 0.0, None)
    dn_dlogM = dn_dlnM * np.log(10.0)

    delta_c = 1.686
    nu = delta_c / sigma
    b1_T10 = _tinker10_b1(nu, delta_vir=delta_vir)

    R_vir_com = (3.0 * M / (4.0 * np.pi * delta_vir * rho_m_com)) ** (1.0 / 3.0)
    R_vir_phys = R_vir_com / (1.0 + z)
    G_Mpc_Msun = 4.30091e-9
    sigma_v_sq = G_Mpc_Msun * M / (2.0 * R_vir_phys)

    return M, dn_dlogM, b1_T10, sigma_v_sq, sigma


def _hod_halo_props(
    nbar_comoving: float,
    z: float,
    cosmo,
    fo,
    tracer_type: str,
    delta_vir: float = 200.0,
    log_M_grid: np.ndarray = _LOG_M_GRID_AM,
) -> Tuple[float, float, float, float]:
    """HOD-weighted halo properties for a tracer of given type.

    Returns (b1, log_M_cut, sigma_v_sq_eff_km2_s2, f_sat).

    b1 -- HOD-weighted Tinker+2010 bias = <N_tot b1>/<N_tot>.
    log_M_cut -- cosmology-dependent central cutoff mass [M_sun/h, log10].
    sigma_v_sq_eff -- f_sat * <sigma_v^2>_sat. Centrals don't contribute to FoG
        (they sit at the potential minimum); only satellites pair-up with the
        host's virial dispersion. Including the f_sat prefactor gives the
        galaxy-pair-weighted LOS dispersion that the Lorentzian FoG sees.
    f_sat -- satellite fraction, for diagnostics.
    """
    if tracer_type not in _HOD_SHAPE_PARAMS:
        raise ValueError(f"Unknown tracer_type {tracer_type!r}. "
                         f"Choices: {sorted(_HOD_SHAPE_PARAMS)}")
    params = _HOD_SHAPE_PARAMS[tracer_type]
    sigma_logM = float(params["sigma_logM"])
    f_cen = float(params["f_cen"])
    log_Mquench_over_Mcut = float(params["log_Mquench_over_Mcut"])
    eta_quench = float(params["eta_quench"])
    log_M_sat_over_Mcut = float(params["log_M_sat_over_Mcut"])
    log_M1_over_Mcut = float(params["log_M1_over_Mcut"])
    alpha_sat = float(params["alpha_sat"])
    # "erf"      = Zheng+07 rising step central (default).
    # "gaussian" = Avila+20 / Rocher+23 narrow-band central (ELGs live in a
    #              narrow halo-mass range, not above a threshold). For
    #              gaussian, log_Mcut is reinterpreted as log_Mcen (the
    #              peak occupancy mass).
    central_form = str(params.get("central_form", "erf")).lower()
    if central_form not in ("erf", "gaussian"):
        raise ValueError(f"central_form must be 'erf' or 'gaussian', got {central_form!r}")

    log_M = log_M_grid
    M, dn_dlogM, b1_T10, sigma_v_sq, _ = _hmf_arrays(
        z=z, cosmo=cosmo, fo=fo, delta_vir=delta_vir, log_M_grid=log_M)
    dlogM = log_M[1] - log_M[0]

    inv_sigma_logM = 1.0 / max(sigma_logM, 1.0e-6)

    def _occupation(log_Mcut: float) -> Tuple[np.ndarray, np.ndarray]:
        u = (log_M - log_Mcut) * inv_sigma_logM
        if central_form == "gaussian":
            # Narrow-band central: peak at log_Mcen=log_Mcut, falls off both
            # ways. No high-M quenching factor is applied -- gaussian already
            # suppresses high-M occupancy, and applying quench would double-
            # count. Quenching params are silently ignored for this form.
            N_cen_base = np.exp(-0.5 * u * u)
            quench = 1.0
        else:
            N_cen_base = 0.5 * (1.0 + erf(u))
            # High-mass quenching factor (Conroy/Wechsler-style; equivalent to
            # HMQ for star-formers). At log M >> log_Mquench, the factor -> 0.
            log_Mq = log_Mcut + log_Mquench_over_Mcut
            ratio_q = 10.0 ** ((log_M - log_Mq) * eta_quench)
            quench = np.exp(-ratio_q)
        N_cen = f_cen * N_cen_base * quench
        # Sat branch is independent of the quenching factor: cluster infallers
        # are present whether or not the central is quenched.
        log_Mcut_sat = log_Mcut + log_M_sat_over_Mcut
        u_sat = (log_M - log_Mcut_sat) * inv_sigma_logM
        sat_gate = 0.5 * (1.0 + erf(u_sat))
        M1 = 10.0 ** (log_Mcut + log_M1_over_Mcut)
        x = np.clip((M - 10.0 ** log_Mcut_sat) / M1, 0.0, None)
        N_sat = sat_gate * (x ** alpha_sat)
        return N_cen, N_sat

    def _nbar_pred(log_Mcut: float) -> float:
        N_cen, N_sat = _occupation(log_Mcut)
        return float(np.sum(dn_dlogM * (N_cen + N_sat)) * dlogM)

    # Bisection in log_Mcut: n̄_pred is monotonically decreasing in M_cut.
    lo, hi = float(log_M[0]), float(log_M[-1])
    n_lo = _nbar_pred(lo)
    n_hi = _nbar_pred(hi)
    if nbar_comoving >= n_lo:
        log_Mcut = lo
    elif nbar_comoving <= n_hi:
        log_Mcut = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _nbar_pred(mid) > nbar_comoving:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1.0e-5:
                break
        log_Mcut = 0.5 * (lo + hi)

    N_cen, N_sat = _occupation(log_Mcut)
    N_tot = N_cen + N_sat
    n_tot = float(np.sum(dn_dlogM * N_tot) * dlogM)
    if n_tot <= 0.0:
        n_tot = max(nbar_comoving, 1.0e-30)

    b1 = float(np.sum(dn_dlogM * N_tot * b1_T10) * dlogM / n_tot)
    n_sat = float(np.sum(dn_dlogM * N_sat) * dlogM)
    f_sat = n_sat / n_tot
    # f_sat * <sigma_v^2>_sat collapses to <N_sat sigma_v^2>/<N_tot>.
    sigma_v_sq_eff = float(np.sum(dn_dlogM * N_sat * sigma_v_sq) * dlogM / n_tot)

    return b1, log_Mcut, sigma_v_sq_eff, f_sat


def _sigma_fog_from_sigma_v_sq(
    sigma_v_sq_km2_s2: float,
    z: float,
    cosmo,
) -> float:
    """Convert LOS velocity-dispersion^2 [(km/s)^2] to comoving Mpc/h damping scale."""
    E_z = float(cosmo.efunc(z))
    return float(np.sqrt(max(sigma_v_sq_km2_s2, 0.0)) * (1.0 + z) / (100.0 * E_z))


# =====================================================================
# Non-Gaussian covariance additions (Tier-1: bias-only, no HOD)
# =====================================================================

def _sigma_b_sq(
    cosmo,
    fo,
    z: float,
    V_survey: float,
    ) -> float:
    """Variance of the super-survey background mode, top-hat sphere approximation.

    sigma_b^2 ≈ sigma^2(R_eff, z) where R_eff = (3 V_survey / 4pi)^(1/3).
    """
    if V_survey <= 0.0:
        return 0.0
    R_eff = (3.0 * V_survey / (4.0 * np.pi)) ** (1.0 / 3.0)
    sigma_R = float(fo.sigma_rz(np.array([R_eff]), z, of="delta_cb")[0])
    return sigma_R ** 2


def _ssc_cov(
    k_centers: np.ndarray,
    ells: Tuple[int, ...],
    pk_multipoles: np.ndarray,
    sigma_b_sq: float,
    ) -> np.ndarray:
    """Super-sample covariance contribution, rank-1 outer product of response.

    C^SSC(k_i, k_j) = sigma_b^2 * R_i * R_j,
    where R_ell(k) = (68/21 - (1/3) d ln P_ell / d ln k) * P_ell(k) is the
    tree-level response (beat-coupling + dilation).

    pk_multipoles : array of shape (n_ells, n_k) with the fiducial galaxy
        multipole power spectrum on k_centers.
    """
    pk = np.asarray(pk_multipoles, dtype=np.float64)
    n_ells, n_k = pk.shape
    assert n_ells == len(ells) and n_k == len(k_centers)

    # Bail out if the theory returned non-finite or non-positive multipoles
    # (can happen at extreme cosmologies where the P_lin interpolator misbehaves).
    if not np.all(np.isfinite(pk)):
        return np.zeros((n_ells * n_k, n_ells * n_k), dtype=np.float64)

    log_k = np.log(k_centers)
    R_ell = np.empty_like(pk)
    for i in range(n_ells):
        P_i = pk[i]
        # Clamp to a positive floor so log/gradient don't blow up.
        P_clip = np.where(P_i > 0, P_i, 1.0e-30)
        dlnP_dlnk = np.gradient(np.log(P_clip), log_k)
        dlnP_dlnk = np.nan_to_num(dlnP_dlnk, nan=0.0, posinf=0.0, neginf=0.0)
        R_ell[i] = (68.0 / 21.0 - dlnP_dlnk / 3.0) * P_clip

    R_flat = R_ell.reshape(-1)  # shape (n_ells * n_k,)
    C = sigma_b_sq * np.outer(R_flat, R_flat)
    if not np.all(np.isfinite(C)):
        return np.zeros_like(C)
    return C


def _legendre(ell: int, mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu)
    if ell == 0:
        return np.ones_like(mu)
    if ell == 2:
        return 0.5 * (3.0 * mu**2 - 1.0)
    if ell == 4:
        return (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
    raise ValueError(f"Unsupported multipole ell={ell}")


def _mu_integral_blocks(
    integrand_kmu: np.ndarray,
    mu: np.ndarray,
    ells: Tuple[int, ...],
) -> np.ndarray:
    """Integrate a k,mu integrand against L_ell(mu) L_ell'(mu) over mu in [0, 1].

    Exploits mu -> -mu parity: for integrands symmetric in mu and multipoles
    of same parity, int_{-1}^1 dmu/2 ... = int_0^1 dmu ...
    """
    leg = np.stack([_legendre(ell, mu) for ell in ells])  # (n_ells, n_mu)
    n_ells = len(ells)
    n_k = integrand_kmu.shape[0]
    out = np.empty((n_ells, n_ells, n_k), dtype=np.float64)
    for i in range(n_ells):
        for j in range(n_ells):
            weight = leg[i] * leg[j]
            out[i, j] = np.trapz(integrand_kmu * weight[None, :], mu, axis=1)
    return out


def _gaussian_mode_count(k_centers: np.ndarray, dk: float, V: float) -> np.ndarray:
    """Continuous Fourier mode count per spherical k-shell of width dk in volume V."""
    return (np.asarray(k_centers, dtype=np.float64) ** 2) * float(dk) * float(V) / (2.0 * np.pi**2)


def _diag_block_matrix(
    blocks: np.ndarray,
    factor_ll: np.ndarray,
    N_modes: np.ndarray,
    n_k: int,
    n_ells: int,
) -> np.ndarray:
    """Assemble a block matrix diagonal in k, dense in ell.

    blocks    : (n_ells, n_ells, n_k) mu-integrated integrand per (ell, ell', k)
    factor_ll : (n_ells, n_ells) prefactor 2 (2l+1)(2l'+1)
    N_modes   : (n_k,) per-bin mode count
    """
    C = np.zeros((n_ells * n_k, n_ells * n_k), dtype=np.float64)
    inv_N = 1.0 / np.maximum(N_modes, 1.0)
    for i in range(n_ells):
        for j in range(n_ells):
            diag = factor_ll[i, j] * blocks[i, j] * inv_N
            C[i * n_k:(i + 1) * n_k, j * n_k:(j + 1) * n_k] = np.diag(diag)
    return C


def _shifted_random_noise_cov(
    k_centers: np.ndarray,
    dk: float,
    ells: Tuple[int, ...],
    pk_lin_vals: np.ndarray,
    b1: float,
    f: float,
    nbar: float,
    V: float,
    alpha_rand: float,
) -> np.ndarray:
    """Post-reconstruction shot-noise boost from the shifted-random catalog.

    The reconstructed field is delta_data - delta_rand (both shifted), giving
    shot noise (1 + 1/alpha)/nbar where alpha = N_rand/N_data. Implemented as
    the analytic difference [P + (1+1/a)N]^2 - [P + N]^2 so the baseline
    Gaussian covariance from ObservablesCovarianceMatrix is not modified in
    place.
    """
    if alpha_rand <= 0 or not np.isfinite(alpha_rand):
        return np.zeros((len(k_centers) * len(ells),) * 2, dtype=np.float64)

    mu = np.linspace(0.0, 1.0, _NMU_DISP)
    mm = mu[None, :]
    Pg = (float(b1) + float(f) * mm**2) ** 2 * np.clip(pk_lin_vals, 0.0, None)[:, None]
    N_shot = 1.0 / float(nbar)
    a = float(alpha_rand)
    delta = 2.0 * Pg * N_shot / a + N_shot**2 * (2.0 / a + 1.0 / a**2)

    blocks = _mu_integral_blocks(delta, mu, ells)
    factor_ll = 2.0 * np.array(
        [[(2 * li + 1) * (2 * lj + 1) for lj in ells] for li in ells],
        dtype=np.float64,
    )
    N_modes = _gaussian_mode_count(k_centers, dk, V)
    return _diag_block_matrix(blocks, factor_ll, N_modes, len(k_centers), len(ells))


def _get_lya_qso_bao_params(
    cfg: Dict[str, float],
    z_eff: float,
    sigma8_delta: float,
    f: float,
    nbar_comoving: float,
    base_volume: float,
    area: float,
    ) -> Dict[str, float]:
    """Return Ly-alpha QSO BAO nuisance parameters for Fisher forecasts."""
    b1 = _lya_flux_bias(
        z=z_eff,
        bF0=float(cfg.get("bF0", -0.117)),
        gamma=float(cfg.get("gamma_bF", 2.9)),
    )

    Sigma_perp_fid = float(cfg.get("Sigma_perp_fid", 5.0))
    Sigma_par_fid  = float(cfg.get("Sigma_par_fid", 10.0))

    sigma8_fid = float(cfg.get("sigma8_fid", 0.83))
    f_fid      = float(cfg.get("f_fid", 0.97))

    growth_ratio = sigma8_delta / sigma8_fid
    f_ratio = f / f_fid

    sigma_perp_lya = Sigma_perp_fid * growth_ratio
    sigma_par_lya = Sigma_par_fid * growth_ratio * f_ratio**0.5

    # Effective white-noise power:
    # P_N_eff = sigma2_pix * delta_r_pix / n_sightline_3D, then nbar_eff = 1 / P_N_eff.
    P_N_eff = cfg.get("P_N_eff", None)
    if P_N_eff is None:
        sigma2_pix = float(cfg.get("sigma2_pix", 1.0))
        delta_r_pix = float(cfg.get("delta_r_pix", 1.0))
        P_N_eff = sigma2_pix * delta_r_pix / max(nbar_comoving, 1.0e-30)
    P_N_eff = float(P_N_eff)
    if P_N_eff <= 0:
        raise ValueError(f"P_N_eff must be > 0 for LyA tracer, got {P_N_eff}")
    nbar_eff = 1.0 / P_N_eff

    # Convert effective 3D nbar back to an angular density for CutskyFootprint input.
    N_eff = nbar_eff * float(base_volume)
    nbar_for_footprint = N_eff / area

    return {
        "is_lya": True,
        "b1": b1,
        "sigmaper_post": sigma_perp_lya,
        "sigmapar_post": sigma_par_lya,
        "sigmas_post": 0.0,  # Lya: no FoG/streaming
        "nbar_eff": nbar_eff,
        "P_N_eff": P_N_eff,
        "nbar_for_footprint": nbar_for_footprint,
    }


def _get_standard_tracer_bao_params(
    cfg: Dict[str, float],
    fo,
    f: float,
    pk_lin_1d,
    nbar_comoving: float,
    nbar_ang: float,
    cosmo,
    z: float,
    sv2_1loop: float = 0.0,
    sv2_1loop_dot: float = 0.0,
    sv2_1loop_ddot: float = 0.0,
    n_iter: int = 1,
    include_fog: bool = True,
    ) -> Dict[str, float]:
    """Return standard galaxy-tracer BAO nuisance parameters for Fisher forecasts.

    b1 is computed via a Zheng+07-style central/satellite HOD with tracer-type
    shape parameters (see _HOD_SHAPE_PARAMS). M_cut is solved to reproduce the
    cosmology-dependent n̄, and b1 = <N_tot b1_T10> / <N_tot>. No DESI-derived
    bias values are used. The HOD satellite branch sets the FoG via a
    satellite-weighted virial velocity dispersion.
    """
    tracer_type = str(cfg.get("tracer_type", "")).strip()
    if not tracer_type:
        raise ValueError(
            "tracer_type is required in tracers.yaml for HOD-driven forecasts."
        )
    b1, log_M_cut, sigma_v_sq_eff, _f_sat = _hod_halo_props(
        nbar_comoving=float(nbar_comoving),
        z=float(z),
        cosmo=cosmo,
        fo=fo,
        tracer_type=tracer_type,
    )

    # Interloper / catastrophic-redshift contamination: a fraction f_int of
    # catalog objects are misidentified (line confusion for ELG/QSO,
    # essentially zero for LRG/BGS). They contribute to the catalog number
    # density but carry no BAO signal at the assigned z, so the observed
    # effective bias is b1_obs = (1 - f_int) * b1_true. The Kaiser correction
    # at order f_int * f² is sub-percent and ignored. Source: Adame+24
    # Table 2 / Massara+24 / Yuan+24. f_int comes from spectroscopic
    # completeness analysis, not from σ-fitting to DESI -- policy-clean.
    f_int = float(cfg.get("f_interloper", 0.0))
    if not 0.0 <= f_int < 1.0:
        raise ValueError(f"f_interloper must be in [0, 1), got {f_int}")
    b1 = b1 * (1.0 - f_int)

    # Redshift-measurement error (instrumental). Applies to every galaxy
    # equally (no f_sat prefactor — both centrals and satellites carry
    # the same redshift uncertainty). Adds in quadrature to the
    # satellite-virial sigma_v^2. Dominant for QSO (~600 km/s from broad
    # emission lines); negligible for BGS/LRG/ELG (narrow lines).
    z_err_kms = float(cfg.get("z_error_kms", 0.0))
    sigma_v_sq_total = sigma_v_sq_eff + z_err_kms * z_err_kms

    if include_fog:
        sigma_fog = _sigma_fog_from_sigma_v_sq(sigma_v_sq_total, z=float(z), cosmo=cosmo)
    else:
        sigma_fog = 0.0

    # The reconstruction Wiener filter and shot-displacement term use the
    # b1 value DESI committed to at the analysis stage (cfg["bias_recon"]),
    # not the HOD-derived intrinsic bias. These can differ — the HOD b1
    # reflects the underlying tracer-halo relation, while bias_recon is
    # an analysis-pipeline choice (typically a round number matched to
    # the EZmocks). For Σ_post we want the value that drove DESI's actual
    # recon, since Σ_post characterizes the residual damping from that
    # specific Wiener filter.
    b1_recon = float(cfg.get("bias_recon", b1))
    sigma_perp_post, sigma_par_post = _sigma_nl_post_from_recon(
        pk_lin_1d=pk_lin_1d,
        b1=b1_recon,
        f=f,
        nbar_comoving=nbar_comoving,
        smoothing_scale=float(cfg["smoothing_scale"]),
        sv2_1loop=sv2_1loop,
        sv2_1loop_dot=sv2_1loop_dot,
        sv2_1loop_ddot=sv2_1loop_ddot,
        n_iter=n_iter,
    )
    nbar_eff = nbar_comoving
    P_N_eff = 1.0 / max(nbar_eff, 1.0e-30)
    nbar_for_footprint = nbar_ang

    return {
        "is_lya": False,
        "b1": b1,
        "sigmaper_post": sigma_perp_post,
        "sigmapar_post": sigma_par_post,
        "sigmas_post": sigma_fog,  # Lorentzian streaming -> BAO model `sigmas`
        "nbar_eff": nbar_eff,
        "P_N_eff": P_N_eff,
        "nbar_for_footprint": nbar_for_footprint,
    }


def build_bao_likelihood(
    N_tracers: float,
    theta_cosmo: Dict[str, float],
    hrdrag: float,
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    area: float = 14000.0,
    resolution: int = 3,
    tracer_config: Dict[str, float] | None = None,
    override_sigmas: Tuple[float, float] | None = None,
    n_iter: int = 1,
    include_fog: bool = True,
    float_sigma_bao: bool = True,
    kmax_cov: float | None = None,
    apmode: str = "qparqper",
) -> Dict:
    """Build the BAO Gaussian likelihood and return it with the fiducial-value bookkeeping
    needed by downstream consumers (Fisher, MCMC, profile likelihood).

    The FKP-effective covariance volume is always applied for standard
    galaxy tracers: the geometric V_shell footprint is replaced with an
    FKP-weighted V_eff_cov computed from the parse_desi_nz n(z) shape.
    Both ``area`` and ``N_tracers`` are scaled by V_eff_cov/V_shell, so
    the cov and Fisher properly reflect the FKP-effective survey rather
    than the full geometric volume. n(z) shape is cosmology-independent;
    V_eff varies with cosmology because the comoving bin volumes do.
    Lya_QSO keeps V_shell since it has no n(z) FKP weighting.

    Returns a dict with:
        likelihood : ObservablesGaussianLikelihood
        template   : BAOPowerSpectrumTemplate (for DH_fid, DM_fid)
        rd         : float, sound horizon in Mpc/h
        params     : dict of fiducial {b1, sigmaper, sigmapar}
        observable : TracerPowerSpectrumMultipolesObservable
        is_lya     : bool
    """

    # Load tracer-specific configuration from tracers.yaml
    # (bias, smoothing scale, z-bin, fiducial number-density range, etc.)
    cfg = get_tracer_config(tracer_bin)

    # Allow caller-side overrides of tracer config entries.
    if tracer_config is not None:
        cfg.update(tracer_config)

    # Default to tracer-bin redshift configuration if not explicitly supplied.
    if zrange is None:
        zrange = tuple(cfg["zrange"])

    # Construct DESI fiducial cosmology object and Fourier-space helper.
    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    if z_eff is None:
        # Cosmology-clean: derive z_eff from the (data-only) n(z) slices and
        # the BAO-Fisher-info weight V × (nP/(1+nP))² per slice. Falls back to
        # the yaml cfg value only if the slices file is missing/empty.
        try:
            z_eff = _compute_z_eff_from_nz(
                tracer_bin=tracer_bin,
                cosmo=cosmo,
                fo=fo,
                area_deg2=float(area),
                b1=float(cfg.get("bias_recon", 2.0)),
            )
        except (FileNotFoundError, ValueError):
            z_eff = float(cfg["z_eff"])

    z = z_eff

    # Apply FKP V_eff rescaling using the signal-relevant galaxy power
    # P_g(k_BAO, z) = b1(z)^2 * Kaiser(z) * P_lin(k_BAO, z), evaluated per
    # redshift slice. Per-slice nbar_3d (cosmology-dependent via V_bin) drives
    # per-slice abundance matching for b1(z_i); growth suppression enters via
    # P_lin(k_BAO, z). No DESI-derived bias values used.
    nbar_3d_eff: Optional[float] = None
    v_shell_for_footprint: Optional[float] = None
    if tracer_bin != "Lya_QSO":
        try:
            z_mid_slice, z_edges_slice, frac_slice, _ = _load_nz_slice_fractions(tracer_bin)
        except FileNotFoundError as exc:
            print(f"[veff] {tracer_bin}: nz slices missing -- using V_shell. ({exc})")
        else:
            z_lo = z_edges_slice[:, 0]
            z_hi = z_edges_slice[:, 1]
            sky_frac = float(area) / 41252.96
            # cosmoprimo's comoving_radial_distance returns chi in Mpc/h already.
            chi_lo = np.asarray(cosmo.comoving_radial_distance(z_lo))
            chi_hi = np.asarray(cosmo.comoving_radial_distance(z_hi))
            V_bin_slice = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * sky_frac
            nbar_slice = (float(N_tracers) * frac_slice) / np.maximum(V_bin_slice, 1.0)
            tracer_type_slice = str(cfg.get("tracer_type", "")).strip()
            if not tracer_type_slice:
                raise ValueError(
                    f"tracer_type missing from tracers.yaml entry for {tracer_bin!r}."
                )
            fkp_wsq_list = []
            slice_b1_list = []
            P_g_per_slice: List[np.ndarray] = []
            for zi, ni in zip(z_mid_slice, nbar_slice):
                if ni <= 0:
                    fkp_wsq_list.append(0.0)
                    slice_b1_list.append(0.0)
                    P_g_per_slice.append(np.zeros_like(_BAO_K_GRID))
                    continue
                b1_i, _logMcut_i, _sv2_i, _fsat_i = _hod_halo_props(
                    nbar_comoving=float(ni), z=float(zi), cosmo=cosmo, fo=fo,
                    tracer_type=tracer_type_slice,
                )
                s8_d = max(float(fo.sigma8_z(zi, of="delta_cb")), 1.0e-30)
                f_z = float(fo.sigma8_z(zi, of="theta_cb")) / s8_d
                beta = f_z / max(b1_i, 1.0e-30)
                kaiser = 1.0 + 2.0 / 3.0 * beta + beta * beta / 5.0
                P_gk = (b1_i ** 2) * np.asarray(
                    _linear_pk_1d(fo, z=zi)(_BAO_K_GRID)) * kaiser
                fkp_wsq_list.append(_fkp_band_weight_sq(float(ni), P_gk))
                slice_b1_list.append(float(b1_i))
                P_g_per_slice.append(P_gk)
            fkp_wsq_per_bin = np.asarray(fkp_wsq_list, dtype=np.float64)
            P_g_per_slice_arr = np.asarray(P_g_per_slice, dtype=np.float64)
            if os.environ.get("_PREP_COVAR_SLICE_DIAG") == "1":
                print(f"[slice] tracer={tracer_bin}")
                print(f"[slice]   {'z':>6} {'frac':>7} {'V_bin/V_shell':>14} "
                      f"{'nbar':>10} {'b1':>6} {'FKP²_band':>10}")
                v_shell_for_diag = float(np.sum(V_bin_slice))
                for zi, fri, Vi, ni, b1i, fk in zip(
                    z_mid_slice, frac_slice, V_bin_slice, nbar_slice,
                    slice_b1_list, fkp_wsq_list,
                ):
                    print(f"[slice]   {zi:>6.3f} {fri:>7.4f} "
                          f"{Vi/v_shell_for_diag:>14.4f} {ni:>10.3e} "
                          f"{b1i:>6.3f} {fk:>10.4f}")

            v_eff, v_shell = _compute_v_eff_fkp(
                cosmo=cosmo, area_deg2=area, tracer_bin=tracer_bin,
                fkp_weight_sq_per_bin=fkp_wsq_per_bin,
            )
            v_shell_for_footprint = float(v_shell) if v_shell > 0 else None
            # Effective 3D density n_eff such that the BAO-Fisher-band
            # average of FKP^2 at fixed n_eff and P_g(k, z_eff) matches the
            # n(z)-weighted band average that produced V_eff:
            #
            #   <FKP^2(n_eff, P_g(k, z_eff))>_band = V_eff / V_shell
            #
            # This is more accurate than a single-k_peak mapping because the
            # FKP^2(k) curve has finite width in k -- at sparse tracers with
            # small nP, the single-point mapping leaks effective volume.
            # Solved by Brent's method on the monotone band-integrated FKP^2.
            fkp_w2_avg = float(v_eff) / float(v_shell) if v_shell > 0 else 0.0
            if fkp_w2_avg > 0:
                idx_eff = int(np.argmin(np.abs(np.asarray(z_mid_slice) - z_eff)))
                n_at_zeff = float(nbar_slice[idx_eff])
                if n_at_zeff > 0:
                    b1_eff_z, _, _, _ = _hod_halo_props(
                        nbar_comoving=n_at_zeff,
                        z=float(z_mid_slice[idx_eff]),
                        cosmo=cosmo, fo=fo,
                        tracer_type=tracer_type_slice,
                    )
                else:
                    b1_eff_z = 1.0
                s8d_eff = max(float(fo.sigma8_z(z_eff, of="delta_cb")), 1.0e-30)
                f_eff = float(fo.sigma8_z(z_eff, of="theta_cb")) / s8d_eff
                beta_eff = f_eff / max(b1_eff_z, 1.0e-30)
                kaiser_eff = 1.0 + 2.0 / 3.0 * beta_eff + beta_eff * beta_eff / 5.0
                P_g_zeff = (b1_eff_z ** 2) * kaiser_eff * np.asarray(
                    _linear_pk_1d(fo, z=z_eff)(_BAO_K_GRID))

                def _band_minus_target(n_test: float) -> float:
                    return _fkp_band_weight_sq(float(n_test), P_g_zeff) - fkp_w2_avg

                # Bracket: f(0) = -fkp_w2_avg < 0; f(large) -> 1 - fkp_w2_avg > 0.
                n_lo = 1.0e-12
                n_hi = 1.0
                for _ in range(60):
                    if _band_minus_target(n_hi) > 0:
                        break
                    n_hi *= 10.0
                else:
                    n_hi = 1.0e12
                try:
                    nbar_3d_eff = float(brentq(
                        _band_minus_target, n_lo, n_hi, xtol=1.0e-14, rtol=1.0e-8,
                    ))
                except ValueError:
                    nbar_3d_eff = None
            else:
                nbar_3d_eff = None
            if os.environ.get("_PREP_COVAR_DIAG") == "1":
                n_avg_dbg = float(N_tracers) / max(v_shell, 1.0)
                print(
                    f"[veff] tracer={tracer_bin} V_shell={v_shell/1e9:.2f} "
                    f"V_eff_sig={v_eff/1e9:.2f} Gpc^3/h^3 "
                    f"fkp_w2_eff={fkp_w2_avg:.4f} "
                    f"n_avg={n_avg_dbg:.3g} "
                    f"n_eff_sig={float(nbar_3d_eff) if nbar_3d_eff else float('nan'):.3g}",
                    flush=True,
                )

    # Compute linear growth quantities.
    #
    # sigma8_delta  -> RMS density fluctuation amplitude
    # sigma8_theta  -> RMS velocity-divergence amplitude
    #
    # Their ratio gives the linear growth rate:
    #
    #   f(z) = dlnD / dln a
    #
    sigma8_delta = float(fo.sigma8_z(z, of="delta_cb"))
    sigma8_theta = float(fo.sigma8_z(z, of="theta_cb"))

    if sigma8_delta <= 0:
        raise ValueError(f"sigma8_delta <= 0 at z={z:.3f}: {sigma8_delta}")

    f = sigma8_theta / sigma8_delta

    # ------------------------------------------------------------------
    # SURVEY GEOMETRY
    # ------------------------------------------------------------------
    #
    # `area` is assumed to be the EFFECTIVE post-mask survey area (i.e.
    # imaging vetos, bright-star masks, extinction cuts, bad-imaging
    # regions already removed). No further f_mask rescaling is applied.
    #
    # N_tracers is the EFFECTIVE OBSERVED tracer count:
    #
    #   N_observed = f_fiber * f_zsuccess * N_target
    #
    # i.e. fiber-assignment and redshift-success efficiencies are already
    # folded in externally. The pipeline treats N_tracers as the final
    # usable catalog size.
    nbar_ang = N_tracers / area

    # ------------------------------------------------------------------
    # SURVEY FOOTPRINT / EFFECTIVE VOLUME
    # ------------------------------------------------------------------

    # Initial footprint used ONLY to determine the physical survey volume.
    #
    # This is the geometry object:
    #   - survey area
    #   - redshift range
    #   - fiducial cosmology
    #
    # It defines the actual observable Fourier volume.
    base_footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar_ang,
        cosmo=cosmo,
    )

    # Convert angular tracer density into comoving number density:
    #
    #   nbar = N / V
    #
    # Units:
    #   (h/Mpc)^3
    #
    nbar_comoving = float(N_tracers) / float(base_footprint.volume)

    # Diagnostic: dump survey/nbar for a single-sample call when debugging
    # tracer-specific discrepancies. Controlled by _PREP_COVAR_DIAG env var
    # to avoid spamming output during emulator training (parallel workers).
    if os.environ.get("_PREP_COVAR_DIAG") == "1":
        # Rough P(k=0.1) estimate for nbar*P readout at BAO scales
        try:
            pk_at_01 = float(_linear_pk_1d(fo, z=z)(np.array([0.1]))[0])
        except Exception:
            pk_at_01 = float("nan")
        nP_at_01 = nbar_comoving * pk_at_01
        print(
            f"[diag] tracer={tracer_bin} z_eff={z:.3f} "
            f"area={area:.1f} zrange=({zrange[0]:.3f},{zrange[1]:.3f}) "
            f"N_tracers={float(N_tracers):.3g} V_survey={float(base_footprint.volume):.3g} "
            f"nbar_3d={nbar_comoving:.3g} P(k=0.1)={pk_at_01:.3g} nP(k=0.1)={nP_at_01:.3g} "
            f"smoothing_scale={float(cfg.get('smoothing_scale', float('nan'))):.1f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # BAO DAMPING MODEL
    # ------------------------------------------------------------------

    # Linear matter power spectrum interpolator.
    pk_lin_1d = _linear_pk_1d(fo, z=z)

    # One-loop displacement variance terms entering nonlinear BAO damping.
    #
    # These encode:
    #   - density-density displacements
    #   - density-velocity coupling
    #   - velocity-velocity coupling
    #
    # and determine anisotropic nonlinear smearing.
    sv2_1loop, sv2_1loop_dot, sv2_1loop_ddot = _sigma_v_sq_1loop_rsd(pk_lin_1d)

    # Pre-reconstruction BAO damping scales:
    #
    #   Sigma_perp
    #   Sigma_parallel
    #
    # including linear + 1-loop corrections.
    sigma_perp_pre, sigma_par_pre = _sigma_nl_pre_from_pk(
        pk_lin_1d,
        f=f,
        sv2_1loop=sv2_1loop,
        sv2_1loop_dot=sv2_1loop_dot,
        sv2_1loop_ddot=sv2_1loop_ddot,
    )

    # ------------------------------------------------------------------
    # TRACER-SPECIFIC EFFECTIVE NOISE / RECONSTRUCTION MODEL
    # ------------------------------------------------------------------

    # Lyα forest tracers use an effective white-noise model rather than
    # standard galaxy shot noise.
    if tracer_bin == "Lya_QSO":

        tracer_settings = _get_lya_qso_bao_params(
            cfg=cfg,
            z_eff=z,
            sigma8_delta=sigma8_delta,
            f=f,
            nbar_comoving=nbar_comoving,
            base_volume=float(base_footprint.volume),
            area=area,
        )

    # Standard galaxy tracers: LRG, ELG, BGS, QSO
    else:
        tracer_settings = _get_standard_tracer_bao_params(
            cfg=cfg,
            fo=fo,
            f=f,
            pk_lin_1d=pk_lin_1d,
            nbar_comoving=nbar_comoving,
            nbar_ang=nbar_ang,
            cosmo=cosmo,
            z=z,
            sv2_1loop=sv2_1loop,
            sv2_1loop_dot=sv2_1loop_dot,
            sv2_1loop_ddot=sv2_1loop_ddot,
            n_iter=n_iter,
            include_fog=include_fog,
        )

    # Extract nuisance / reconstruction parameters.
    is_lya = bool(tracer_settings["is_lya"])

    b1 = float(tracer_settings["b1"])

    sigma_perp_post = float(tracer_settings["sigmaper_post"])
    sigma_par_post = float(tracer_settings["sigmapar_post"])
    sigma_s_post = float(tracer_settings.get("sigmas_post", 0.0))

    # Optional manual override for debugging / validation studies.
    if override_sigmas is not None:
        sigma_perp_post = float(override_sigmas[0])
        sigma_par_post = float(override_sigmas[1])

    nbar_eff = float(tracer_settings["nbar_eff"])
    P_N_eff = float(tracer_settings["P_N_eff"])
    nbar_for_footprint = float(tracer_settings["nbar_for_footprint"])

    # Inject the V_eff-matched effective 3D density into the footprint.
    # The Brent root-find above solved for n_eff such that
    #
    #     <FKP^2(n_eff, P_g(k, z_eff))>_band  =  V_eff / V_shell
    #
    # so the footprint's internal FKP^2 weighting at the band-integrated
    # BAO scale reproduces V_eff exactly. CutskyFootprint stores a scalar
    # nbar as an angular density (deg^-2) and recovers nbar_3d = size/V,
    # so converting via (n_eff × V_shell / area) lands nbar_3d at n_eff
    # while leaving footprint.volume = V_shell (physical geometry).
    # If the V_eff block did not run (Lya, missing n(z)), the upstream
    # angular density is left in place.
    if (nbar_3d_eff is not None and v_shell_for_footprint is not None
            and float(area) > 0):
        nbar_for_footprint = (
            float(nbar_3d_eff) * float(v_shell_for_footprint) / float(area)
        )

    # Parameters supplied to the BAO observable model.
    # `sigmas` is the Lorentzian FoG/streaming scale (Beutler+17 form),
    # NOT added in quadrature to sigmapar -- see _sigma_nl_post_from_recon.
    params = {
        "b1": b1,
        "sigmaper": sigma_perp_post,
        "sigmapar": sigma_par_post,
        "sigmas": sigma_s_post,
    }

    # ------------------------------------------------------------------
    # FINAL FISHER FOOTPRINT OBJECT
    # ------------------------------------------------------------------

    # This footprint object is used by desilike to build:
    #   - covariance normalization
    #   - effective mode counts
    #   - shot noise
    #
    # attrs are metadata only (not used directly by desilike).
    footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar_for_footprint,
        cosmo=cosmo,
        attrs={
            "tracer_bin": tracer_bin,
            "is_lya": is_lya,
            "bias_recon": cfg.get("bias_recon"),
            "smoothing_scale": cfg.get("smoothing_scale"),
            "nbar_comoving": nbar_comoving,
            "nbar_eff": nbar_eff,
            "P_N_eff": P_N_eff,
            "f_cosmo": f,
            "sigmaper_pre": sigma_perp_pre,
            "sigmapar_pre": sigma_par_pre,
            "sigmaper_post": sigma_perp_post,
            "sigmapar_post": sigma_par_post
        },
    )

    # ------------------------------------------------------------------
    # APPROXIMATE SURVEY WINDOW FUNCTION EFFECT
    # ------------------------------------------------------------------

    # Finite survey geometry suppresses access to modes larger than the
    # survey coherence scale.
    #
    # Approximate the survey side length as:
    #
    #   L ~ V^(1/3)
    #
    # which gives a characteristic fundamental Fourier mode:
    #
    #   k_fund ~ 2pi / L
    #
    # Modes below this scale are strongly mixed/suppressed by the survey
    # window function and are not independently measurable.
    #
    # This is a simplified approximation to the full DESI window-function
    # convolution treatment.  [oai_citation:0‡OUP Academic](https://academic.oup.com/mnras/article-pdf/479/4/5168/25209108/sty1814.pdf?utm_source=chatgpt.com)
    L_survey = base_footprint.volume ** (1 / 3)

    # Fundamental observable Fourier mode.
    kmin_window = 2.0 * np.pi / L_survey

    # Prevent unrealistically small k_min values for huge survey volumes.
    #
    # DESI BAO analyses typically do not fit below k ~ 0.02 h/Mpc anyway.
    kmin_eff = max(0.02, kmin_window)

    # ------------------------------------------------------------------
    # BAO MODEL
    # ------------------------------------------------------------------

    template = BAOPowerSpectrumTemplate(
        z=z,
        fiducial=("DESI", dict(theta_cosmo)),
        apmode=apmode,
    )

    # BAO template matches DESI Y1/Y3 convention:
    #   - mode='recsym' : reconstructed with randoms also shifted (DESI standard).
    #     Lyα QSO is pre-recon so mode='' (no reconstruction template).
    #   - smoothing_radius per tracer from tracers.yaml (15 h^-1 Mpc for
    #     galaxy/ELG, 30 h^-1 Mpc for QSO).
    #   - model='fog-damping' : Beutler+17 model with FoG damping applied to
    #     the wiggle part, used in DESI Y1 BAO (Moon+23, Ross+24).
    if is_lya:
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
            template=template,
            model="fog-damping",
        )
    else:
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
            template=template,
            mode="recsym",
            smoothing_radius=float(cfg["smoothing_scale"]),
            model="fog-damping",
        )

    # Observable power-spectrum multipoles used in the Fisher forecast.
    observable = TracerPowerSpectrumMultipolesObservable(
        data=params,
        klim={
            0: [kmin_eff, kmax_cov if kmax_cov is not None else 0.30, 0.005],
            2: [kmin_eff, kmax_cov if kmax_cov is not None else 0.30, 0.005],
        },
        theory=theory,
    )

    # Gaussian covariance matrix model.
    covariance = ObservablesCovarianceMatrix(
        observable,
        footprints=footprint,
        resolution=resolution,
    )
    gauss_cov_obj = covariance(**params)
    C_gauss_arr = np.array(np.asarray(gauss_cov_obj), dtype=np.float64, copy=True)
    C_full = C_gauss_arr.copy()
    cov_components: Dict[str, np.ndarray] = {"C_gauss": C_gauss_arr}

    # ------------------------------------------------------------------
    # NON-GAUSSIAN COVARIANCE ADDITIONS (Tier-1: bias-only, no HOD)
    #
    #   C_total = C_gauss + C_SSC + C_recon_shot
    #
    # SSC is a rank-1 outer product (sigma_b^2 * R R^T). The recon shot-noise
    # boost replaces 1/nbar with (1 + 1/alpha)/nbar. Lyα skips this block
    # (not halo-based).
    # ------------------------------------------------------------------
    if not is_lya:

        ell_tuple = tuple(int(ell) for ell in observable.ells)
        k_centers = np.asarray(observable.k[0], dtype=np.float64)
        V_survey = float(footprint.volume)

        # Recover the k-bin width from the observable grid; falls back to
        # the klim spec (0.005 h/Mpc) if only one bin is present.
        if k_centers.size > 1:
            dk_bin = float(np.mean(np.diff(k_centers)))
        else:
            dk_bin = 0.005

        try:
            theory(**params)

            pk_multipoles = np.asarray(theory.power, dtype=np.float64)
            pk_lin_vals = pk_lin_1d(k_centers)

            sigma_b_sq = _sigma_b_sq(
                cosmo,
                fo,
                z,
                V_survey,
            )

            C_ssc = _ssc_cov(
                k_centers=k_centers,
                ells=ell_tuple,
                pk_multipoles=pk_multipoles,
                sigma_b_sq=sigma_b_sq,
            )

            # Shifted-random shot-noise boost: (1 + 1/alpha)/nbar instead of 1/nbar.
            C_recon_shot = _shifted_random_noise_cov(
                k_centers=k_centers,
                dk=dk_bin,
                ells=ell_tuple,
                pk_lin_vals=pk_lin_vals,
                b1=b1,
                f=f,
                nbar=nbar_comoving,
                V=V_survey,
                alpha_rand=_N_RAND_OVER_N_DATA,
            )

            if C_ssc.shape == C_full.shape:
                C_full += C_ssc
                cov_components["C_SSC"] = np.array(C_ssc, copy=True)
            if C_recon_shot.shape == C_full.shape:
                C_full += C_recon_shot
                cov_components["C_recon_shot"] = np.array(C_recon_shot, copy=True)

            # Finite-value check is sufficient: the Gaussian cov has
            # rank-deficient multipole blocks (det(C_00,C_02; C_02,C_22) = 0),
            # and SSC adds a positive rank-1 piece — any failure mode we can
            # detect reduces to NaN/inf propagation from extreme cosmologies.
            if not np.all(np.isfinite(C_full)):
                raise ValueError("Non-finite entries in augmented covariance")

        except Exception as exc:

            warnings.warn(
                f"Skipping unstable NG covariance sample: {exc}"
            )

            raise
    augmented_cov = gauss_cov_obj.clone(value=C_full)

    # Gaussian likelihood used for Fisher differentiation.
    likelihood = ObservablesGaussianLikelihood(
        observables=observable,
        covariance=augmented_cov,
    )

    # Float Sigma_s (Lorentzian FoG/streaming) with a Gaussian prior centered
    # on the analytically-computed value (Tinker virial dispersion) so the
    # Fisher error on qpar/qper accounts for its marginalization. Width is
    # set generously since the analytic estimate has ~50% uncertainty.
    likelihood.all_params["sigmas"].update(
        prior={"dist": "norm", "loc": float(sigma_s_post), "scale": 2.0}
    )

    # Float the BAO damping scales with Gaussian priors centered on the
    # analytically-computed values. Width 2.0 Mpc/h on BOTH Σ_⊥ and Σ_∥
    # matches DESI Adame+24 Sec 4.2.1 (σ=2 Mpc/h around the EZmock-fit
    # central values). Floating Σ opens the Σ–B amplitude degeneracy that
    # is naturally present in the BAO fit and inflates σ(qpar, qper) to
    # match the DESI methodology.
    # Lyα QSO is pre-recon so we don't float the damping scales.
    if float_sigma_bao and not is_lya:
        likelihood.all_params["sigmaper"].update(
            fixed=False,
            prior={
                "dist": "norm",
                "loc": float(sigma_perp_post),
                "scale": 2.0,
            },
        )
        likelihood.all_params["sigmapar"].update(
            fixed=False,
            prior={
                "dist": "norm",
                "loc": float(sigma_par_post),
                "scale": 2.0,
            },
        )

    if is_lya:
        # For Lyα: b1 is negative (~-0.12) which falls outside the yaml prior
        # [0.2, 4.0]. Fixing it avoids prior-bounds issues in the Fisher
        # differentiation and removes the b1-dbeta degeneracy.
        # dbeta is also degenerate with b1 in a BAO-only amplitude rescaling
        # and has no physical meaning without reconstruction.
        likelihood.all_params["b1"].update(fixed=True)
        likelihood.all_params["dbeta"].update(fixed=True)

    rd = hrdrag / _H_FID
    cov_components["C_total"] = np.array(C_full, copy=True)
    return {
        "likelihood": likelihood,
        "template": template,
        "rd": rd,
        "params": params,
        "observable": observable,
        "is_lya": is_lya,
        "cov_components": cov_components,
    }


def get_bao_fisher_covariance(
    N_tracers: float,
    theta_cosmo: Dict[str, float],
    hrdrag: float,
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    area: float = 14000.0,
    resolution: int = 3,
    tracer_config: Dict[str, float] | None = None,
    override_sigmas: Tuple[float, float] | None = None,
    n_iter: int = 1,
    include_fog: bool = True,
    float_sigma_bao: bool = True,
    kmax_cov: float | None = None,
) -> Dict[str, float]:
    """Compute the 2x2 BAO covariance matrix from a single-bin Fisher forecast."""
    info = build_bao_likelihood(
        N_tracers=N_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag,
        tracer_bin=tracer_bin,
        zrange=zrange,
        z_eff=z_eff,
        area=area,
        resolution=resolution,
        tracer_config=tracer_config,
        override_sigmas=override_sigmas,
        n_iter=n_iter,
        include_fog=include_fog,
        float_sigma_bao=float_sigma_bao,
        kmax_cov=kmax_cov,
    )

    F_q = _q_fisher_from_bao_likelihood_info(info)

    try:
        cov_q = np.linalg.inv(F_q)
    except np.linalg.LinAlgError:
        cov_q = np.linalg.pinv(F_q, rcond=1e-12)

    template = info["template"]

    # Use template.{DH,DM}_over_rd_fid directly — dimensionally correct.
    # The naive `template.DH_fid / info["rd"]` form is dimensionally mixed
    # (DH_fid is Mpc/h, info["rd"] is proper Mpc) and underreports σ by 1/h.
    DH_over_rd_fid = float(template.DH_over_rd_fid)
    DM_over_rd_fid = float(template.DM_over_rd_fid)

    J = np.diag([DH_over_rd_fid, DM_over_rd_fid])
    cov_phys = J @ cov_q @ J.T

    upper_tri_vals = cov_phys[_TRIU_I, _TRIU_J]
    return dict(zip(TARGET_NAMES, upper_tri_vals))


def get_bao_fisher_covariance_sliced_nz(
    N_tracers: float,
    theta_cosmo: Dict[str, float],
    hrdrag: float,
    nz_slices_path: str | Path,
    tracer_bin: str = "LRG2",
    z_eff_bin: float | None = None,
    area: float = 14000.0,
    resolution: int = 3,
    tracer_config: Dict[str, float] | None = None,
    override_sigmas: Tuple[float, float] | None = None,
    n_iter: int = 1,
    include_fog: bool = True,
    float_sigma_bao: bool = True,
) -> Dict[str, float]:
    """
    Compute BAO covariance using redshift-sliced n(z).

    The CSV supplies slice_fraction, zlow, zhigh, zmid. For a sampled total
    N_tracers, each slice gets N_i = N_tracers * slice_fraction_i.

    We marginalize each slice over its own nuisance parameters and sum only
    the qpar/qper Fisher information.
    """
    slices = _load_nz_slices(nz_slices_path)

    cfg = get_tracer_config(tracer_bin)
    if z_eff_bin is None:
        z_eff_bin = float(cfg["z_eff"])

    F_q_total = np.zeros((2, 2), dtype=np.float64)

    for _, row in slices.iterrows():
        frac = float(row["slice_fraction"])
        if frac <= 0.0:
            continue

        N_i = float(N_tracers) * frac
        zlo_i = float(row["zlow"])
        zhi_i = float(row["zhigh"])
        zmid_i = float(row["zmid"])

        # Skip vanishingly tiny slices.
        if N_i <= 0.0 or zhi_i <= zlo_i:
            continue

        info_i = build_bao_likelihood(
            N_tracers=N_i,
            theta_cosmo=theta_cosmo,
            hrdrag=hrdrag,
            tracer_bin=tracer_bin,
            zrange=(zlo_i, zhi_i),
            z_eff=zmid_i,
            area=area,
            resolution=resolution,
            tracer_config=tracer_config,
            override_sigmas=override_sigmas,
            n_iter=n_iter,
            include_fog=include_fog,
            float_sigma_bao=float_sigma_bao,
        )

        F_q_total += _q_fisher_from_bao_likelihood_info(info_i)

    F_q_total = _regularize_fisher_matrix(F_q_total)

    try:
        cov_q = np.linalg.inv(F_q_total)
    except np.linalg.LinAlgError:
        cov_q = np.linalg.pinv(F_q_total, rcond=1e-12)

    # Convert qpar/qper to DH/rd and DM/rd at the bin-level effective redshift.
    # The slices are used to compute total information, but the target remains
    # the published/bin-level BAO distance at z_eff_bin.
    template_bin = BAOPowerSpectrumTemplate(
        z=z_eff_bin,
        fiducial=("DESI", dict(theta_cosmo)),
        apmode="qparqper",
    )

    # Use template_bin.{DH,DM}_over_rd_fid directly — dimensionally correct.
    # The naive `template_bin.DH_fid / rd` form is dimensionally mixed
    # (DH_fid is Mpc/h, rd here is proper Mpc) and underreports σ by 1/h.
    DH_over_rd_fid = float(template_bin.DH_over_rd_fid)
    DM_over_rd_fid = float(template_bin.DM_over_rd_fid)

    J = np.diag([DH_over_rd_fid, DM_over_rd_fid])
    cov_phys = J @ cov_q @ J.T

    upper_tri_vals = cov_phys[_TRIU_I, _TRIU_J]
    return dict(zip(TARGET_NAMES, upper_tri_vals))


def run_fisher(
    sample: Dict[str, float],
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    param_defaults: Dict[str, float] | None = None,
    area: float = 14000.0,
    override_sigmas: Tuple[float, float] | None = None,
    n_iter: int = 1,
    include_fog: bool = True,
    float_sigma_bao: bool = True,
    nz_slices_path: str | Path | None = None,
    kmax_cov: float | None = None,
) -> Dict[str, float]:
    """Convert a sample dict (with N_tracers + cosmo params) to Fisher covariance elements."""
    if param_defaults:
        sample = {**param_defaults, **sample}

    N_tracers = sample["N_tracers"]
    theta_cosmo, hrdrag = _to_bao_cosmo_params(sample)

    if nz_slices_path is not None:
        return get_bao_fisher_covariance_sliced_nz(
            N_tracers=N_tracers,
            theta_cosmo=theta_cosmo,
            hrdrag=hrdrag,
            nz_slices_path=nz_slices_path,
            tracer_bin=tracer_bin,
            z_eff_bin=z_eff,
            area=area,
            override_sigmas=override_sigmas,
            n_iter=n_iter,
            include_fog=include_fog,
            float_sigma_bao=float_sigma_bao,
        )

    return get_bao_fisher_covariance(
        N_tracers=N_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag,
        tracer_bin=tracer_bin,
        zrange=zrange,
        z_eff=z_eff,
        area=area,
        override_sigmas=override_sigmas,
        n_iter=n_iter,
        include_fog=include_fog,
        float_sigma_bao=float_sigma_bao,
        kmax_cov=kmax_cov,
    )


def compute_pipeline_sigmas(
    sample: Dict[str, float],
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    param_defaults: Dict[str, float] | None = None,
    area: float = 14000.0,
    n_iter: int = 1,
    include_fog: bool = True,
) -> Tuple[float, float]:
    """Return (sigma_perp_post, sigma_par_post) computed by the pipeline for this sample."""
    if param_defaults:
        sample = {**param_defaults, **sample}

    N_tracers = float(sample["N_tracers"])
    theta_cosmo, _hrdrag = _to_bao_cosmo_params(sample)

    cfg = get_tracer_config(tracer_bin)
    if zrange is None:
        zrange = tuple(cfg["zrange"])

    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    if z_eff is None:
        try:
            z_eff = _compute_z_eff_from_nz(
                tracer_bin=tracer_bin,
                cosmo=cosmo,
                fo=fo,
                area_deg2=float(area),
                b1=float(cfg.get("bias_recon", 2.0)),
            )
        except (FileNotFoundError, ValueError):
            z_eff = float(cfg["z_eff"])
    z = z_eff

    sigma8_delta = float(fo.sigma8_z(z, of="delta_cb"))
    sigma8_theta = float(fo.sigma8_z(z, of="theta_cb"))
    f = sigma8_theta / sigma8_delta

    nbar_ang = N_tracers / area
    base_footprint = CutskyFootprint(area=area, zrange=zrange, nbar=nbar_ang, cosmo=cosmo)
    nbar_comoving = N_tracers / float(base_footprint.volume)

    pk_lin_1d = _linear_pk_1d(fo, z=z)
    sv2_1loop, sv2_1loop_dot, sv2_1loop_ddot = _sigma_v_sq_1loop_rsd(pk_lin_1d)
    sigma_perp_pre, sigma_par_pre = _sigma_nl_pre_from_pk(
        pk_lin_1d, f=f,
        sv2_1loop=sv2_1loop,
        sv2_1loop_dot=sv2_1loop_dot,
        sv2_1loop_ddot=sv2_1loop_ddot,
    )

    if tracer_bin == "Lya_QSO":
        settings = _get_lya_qso_bao_params(
            cfg=cfg,
            z_eff=z,
            sigma8_delta=sigma8_delta,
            f=f,
            nbar_comoving=nbar_comoving,
            base_volume=float(base_footprint.volume),
            area=area,
        )
    else:
        settings = _get_standard_tracer_bao_params(
            cfg=cfg, fo=fo, f=f,
            pk_lin_1d=pk_lin_1d, nbar_comoving=nbar_comoving, nbar_ang=nbar_ang,
            cosmo=cosmo, z=z,
            sv2_1loop=sv2_1loop,
            sv2_1loop_dot=sv2_1loop_dot,
            sv2_1loop_ddot=sv2_1loop_ddot,
            n_iter=n_iter,
            include_fog=include_fog,
        )
    return float(settings["sigmaper_post"]), float(settings["sigmapar_post"])


def _worker_init():
    """Silence noisy warnings/logging in spawned worker processes."""
    import warnings
    import logging

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("DISPLAY", None)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _worker_run_fisher(args_tuple):
    """Top-level function for multiprocessing (must be picklable).

    Returns (sample, target_vals, tb_str) where tb_str is None on success,
    a traceback string on failure, and "non-finite" if the Fisher returned
    non-finite target values. The master aggregates and prints the first
    traceback so the worker's silenced stderr doesn't hide real bugs.
    """
    sample, tracer_bin, zrange, z_eff, param_defaults, area, nz_slices_path = args_tuple
    try:
        targets = run_fisher(
            sample,
            tracer_bin=tracer_bin,
            zrange=zrange,
            z_eff=z_eff,
            param_defaults=param_defaults,
            area=area,
            nz_slices_path=nz_slices_path,
        )
        target_vals = [targets[t] for t in TARGET_NAMES]
        if not all(np.isfinite(v) for v in target_vals):
            return None, None, "non-finite target values"
        return sample, target_vals, None
    except Exception:
        return None, None, traceback.format_exc()


def generate_dataset(
    priors: Dict[str, Dict[str, float]],
    n_samples: int,
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    batch_size: int = 64,
    seed: int = 0,
    sigma_clip: float = 4.0,
    workers: int = 1,
    param_defaults: Dict[str, float] | None = None,
    constraints: Dict[str, Dict] | None = None,
    area: float = 14000.0,
    nz_slices_path: str | Path | None = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if constraints is None:
        constraints = CONSTRAINTS

    param_names = list(priors.keys())
    param_rows: List[List[float]] = []
    target_rows: List[List[float]] = []

    total_attempts = 0
    failed = 0
    lhs_seed = seed
    printed_exception = False

    if workers > 1:
        import multiprocessing as mp
        import time as _time

        ctx = mp.get_context("spawn")
        os.environ["_PREP_COVAR_WORKER"] = "1"
        print(f"Using {workers} worker processes (spawn)")
        pool = ctx.Pool(workers, initializer=_worker_init)
        t_start = _time.perf_counter()

        while len(param_rows) < n_samples:
            remaining = n_samples - len(param_rows)
            draw_count = min(max(remaining * 2, batch_size), remaining * 3)
            draws = _constrained_samples(
                priors,
                constraints=constraints,
                n_samples=draw_count,
                seed=lhs_seed,
                sigma_clip=sigma_clip,
            )
            lhs_seed += 1

            tasks = [(s, tracer_bin, zrange, z_eff, param_defaults, area, nz_slices_path) for s in draws]
            for sample, target_vals, tb_str in pool.imap_unordered(_worker_run_fisher, tasks):
                total_attempts += 1
                if sample is None:
                    failed += 1
                    if tb_str is not None and not printed_exception:
                        printed_exception = True
                        print("\nFirst worker Fisher failure (showing traceback once):")
                        print(tb_str)
                else:
                    param_rows.append([sample[p] for p in param_names])
                    target_rows.append(target_vals)

                accepted = len(param_rows)
                if total_attempts % 10 == 0 or accepted >= n_samples:
                    elapsed = _time.perf_counter() - t_start
                    sps = total_attempts / elapsed if elapsed > 0 else 0
                    eta = (n_samples - accepted) / (accepted / elapsed) if accepted > 0 else 0
                    print(
                        f"\r  {accepted:>6}/{n_samples} "
                        f"({100.0 * accepted / n_samples:.1f}%) "
                        f"| {failed} failed | {sps:.1f} samples/s "
                        f"| ETA {eta / 60:.1f}min",
                        end="", flush=True,
                    )

                if accepted >= n_samples:
                    break

        pool.terminate()
        pool.join()
        elapsed = _time.perf_counter() - t_start
        print(
            f"\nDone: {len(param_rows)} accepted, {failed} failed, "
            f"{total_attempts} total in {elapsed:.1f}s"
        )

    else:
        while len(param_rows) < n_samples:
            draws = _constrained_samples(
                priors,
                constraints=constraints,
                n_samples=batch_size,
                seed=lhs_seed,
                sigma_clip=sigma_clip,
            )
            lhs_seed += 1

            for sample in draws:
                total_attempts += 1
                try:
                    targets = run_fisher(
                        sample,
                        tracer_bin=tracer_bin,
                        zrange=zrange,
                        z_eff=z_eff,
                        param_defaults=param_defaults,
                        area=area,
                        nz_slices_path=nz_slices_path,
                    )
                    target_vals = [targets[t] for t in TARGET_NAMES]
                    if not all(np.isfinite(v) for v in target_vals):
                        failed += 1
                        continue
                    param_rows.append([sample[p] for p in param_names])
                    target_rows.append(target_vals)

                    accepted = len(param_rows)
                    if accepted % 10 == 0 or accepted >= n_samples:
                        rate = accepted / max(total_attempts, 1)
                        print(
                            f"Accepted {accepted:>6}/{n_samples}, "
                            f"failed {failed}, attempts {total_attempts}, "
                            f"acceptance={100.0 * rate:.2f}%"
                        )
                except Exception:
                    failed += 1
                    if not printed_exception:
                        printed_exception = True
                        print("First Fisher failure (showing traceback once):")
                        traceback.print_exc()
                        print(f"Failing sample: {sample}")
                    continue

                if len(param_rows) >= n_samples:
                    break

    X = np.asarray(param_rows, dtype=np.float64)
    y = np.asarray(target_rows, dtype=np.float64)
    return param_names, X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data for a BAO Fisher error emulator."
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--sigma-clip", type=float, default=4.0)

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1 = serial).",
    )
    parser.add_argument(
        "--tracer-bin",
        dest="tracer_bin",
        type=str,
        default="LRG2",
        choices=TRACER_TYPE_CHOICES,
        help="DESI tracer bin key (must match tracers.yaml).",
    )
    parser.add_argument(
        "--z-eff",
        type=float,
        default=None,
        help="Effective redshift. If omitted, defaults to tracer-bin config.",
    )
    parser.add_argument(
        "--zrange",
        type=float,
        nargs=2,
        default=None,
        metavar=("Z_MIN", "Z_MAX"),
        help="Redshift bin edges for the footprint volume. If omitted, defaults to tracer-bin config.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Tracer name prefix for saved files (e.g. 'LRG1' -> LRG1_train.npz). Defaults to --tracer-bin.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Explicit version number for the training_data/v{N} directory. "
             "If omitted, auto-increments to the next available version.",
    )
    parser.add_argument(
        "--ntracers-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("NTRACERS_LOW", "NTRACERS_HIGH"),
        help="Override the N_tracers prior range. Defaults to tracer-bin low/high from tracers.yaml.",
    )
    parser.add_argument(
        "--priors-json",
        type=str,
        default="",
        help=(
            "JSON dictionary of priors, e.g. "
            '\'{"N_tracers":{"dist":"uniform","low":1e5,"high":1e7}}\''
        ),
    )
    parser.add_argument(
        "--cosmo-model",
        type=str,
        default="base",
        choices=list(COSMO_MODELS.keys()),
        help="Cosmology model determining which params are varied (default: base).",
    )
    parser.add_argument(
        "--area",
        type=float,
        default=14000.0,
        help="Effective survey area in deg^2 used by CutskyFootprint.",
    )
    parser.add_argument(
        "--nz-slices-path",
        type=str,
        default=None,
        help=(
            "Optional path to a parsed *_nz_slices.csv file. "
            "If provided, N_tracers is distributed across redshift slices "
            "using slice_fraction and Fisher information is summed over slices."
        ),
    )
    sys.argv = [a for a in sys.argv if a.strip()]
    args = parser.parse_args()

    tracer_bin_cfg = get_tracer_config(args.tracer_bin)
    zrange = tuple(args.zrange) if args.zrange is not None else tuple(tracer_bin_cfg["zrange"])
    z_eff = args.z_eff if args.z_eff is not None else float(tracer_bin_cfg["z_eff"])

    cosmo_model = args.cosmo_model
    model_params = COSMO_MODELS[cosmo_model]

    if args.priors_json:
        priors = parse_priors(args.priors_json)
    else:
        varied_keys = ["N_tracers"] + model_params
        priors = {k: dict(DEFAULT_PRIORS[k]) for k in varied_keys}

    if args.ntracers_range is not None:
        priors["N_tracers"] = {
            "dist": "uniform",
            "low": args.ntracers_range[0],
            "high": args.ntracers_range[1],
        }
    else:
        priors["N_tracers"] = {
            "dist": "uniform",
            "low": float(tracer_bin_cfg["low"]),
            "high": float(tracer_bin_cfg["high"]),
        }

    all_cosmo_keys = {"Om", "Ok", "w0", "wa", "hrdrag"}
    fixed_keys = all_cosmo_keys - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys if k in PARAM_DEFAULTS}

    constraints = {
        name: spec for name, spec in CONSTRAINTS.items()
        if all(p in model_params for p in spec["params"])
    }

    save_path = os.path.abspath(
        args.save_path if args.save_path else
        get_default_save_path(analysis="bao", quantity="covar", cosmo_model=cosmo_model)
    )

    print(f"Tracer bin: {args.tracer_bin}")
    print(f"Tracer bin config: {tracer_bin_cfg}")
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
    print(
        f"N_tracers prior: [{priors['N_tracers']['low']:.3g}, "
        f"{priors['N_tracers']['high']:.3g}]"
    )
    print("Using priors:", priors)
    print(f"Active constraints: {list(constraints.keys())}")
    print(f"Redshift range: {zrange}, z_eff = {z_eff:.3f}")
    print("Writing dataset to:", save_path)
    print(f"Area: {args.area:.3f} deg^2")
    if args.nz_slices_path:
        print(f"Using n(z) slices: {args.nz_slices_path}")

    try:
        param_names, X, y = generate_dataset(
            priors=priors,
            n_samples=args.n_samples,
            tracer_bin=args.tracer_bin,
            zrange=zrange,
            z_eff=z_eff,
            batch_size=args.batch_size,
            seed=args.seed,
            sigma_clip=args.sigma_clip,
            workers=args.workers,
            param_defaults=param_defaults,
            constraints=constraints,
            area=args.area,
            nz_slices_path=args.nz_slices_path,
        )
        print(f"Generated dataset with shape X={X.shape}, y={y.shape}")
        save_dataset(
            save_path=save_path,
            param_names=param_names,
            X=X,
            y=y,
            test_size=args.test_size,
            target_names=TARGET_NAMES,
            name=args.name if args.name is not None else args.tracer_bin,
            version=args.version,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
