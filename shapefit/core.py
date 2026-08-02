"""Shared core for the full-shape (ShapeFit) Fisher forecast pipeline.

Parallel to ``bao/core.py`` but for a pre-reconstruction full-shape analysis:
per DESI tracer bin, map ``(N_tracers, cosmology)`` to the Fisher forecast of
the ShapeFit compressed parameters ``(qiso, qap, f_sigmar, m)``.

Design decisions (see shapefit/README.md and shapefit/CHANGELOG.md):

- Template: ``ShapeFitPowerSpectrumTemplate(apmode="qisoqap",
  with_now="wallish2018")``. ``with_now`` MUST be explicit — the desilike
  default ``'peakaverage'`` mislabels sigma ~2x and crashes chaotically across
  wide emulator priors (see the long comment at bao/core.py:1641). ``dn``
  stays fixed (un-fixing it changes the definition of ``m``).
- Theory: ``REPTVelocileptorsTracerPowerSpectrumMultipoles`` by default,
  pluggable via ``theory_cls``. This is DESI's own baseline -- 2024 V S4.7
  item 2, "We select velocileptors with its EPT option as our baseline choice"
  -- and with ``prior_basis='physical'`` it varies exactly DESI's Table 4 set
  (b1p, b2p, bsp, alpha0p, alpha2p, sn0p, sn2p) at their prior widths, all
  marginalised by the Schur reduction. Kaiser remains available and is a
  different model, not an approximation to this one: it under-reports sigma by
  1.6-2.1x and inverts two correlation signs on every tracer (CHANGELOG S15,
  S16, S22).
- Broadband: none, in the polynomial sense -- DESI full-shape has no broadband
  basis either. The EFT counterterms and stochastic terms above ARE the
  small-scale freedom. Do not graft the BAO 'pcs' basis onto full shape.
- Pre-recon everywhere: no reconstruction template, no Sigma_post, no
  smoothing_scale/bias_recon modelling, no shifted-random shot-noise boost.
  Damping fiducials are the pre-recon (linear + 1-loop) Sigma_perp/Sigma_par
  with the HOD FoG dispersion added in quadrature along the LOS.
- All 6 tracers are fit anisotropically with ells (0, 2) over
  klim [0.02, 0.2, 0.005] (DESI KP4.5 reference config). There is no
  iso/aniso split — that is a BAO-recon convention, not a full-shape one.
- Survey physics (HOD b1 + assembly bias + interlopers, n(z) slices, FKP
  V_eff -> n_eff mapping, SSC) is imported from ``bao/core.py`` unchanged;
  only the k-band Fisher kernel is full-shape specific (dF/dk ~ k^2 over the
  fit band instead of the Silk-damped BAO kernel).
"""
import os

# Set display/backend env vars before any library imports can trigger X11 or GPU probes.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# In spawned worker processes, additionally suppress all C-level stderr.
if os.environ.get("_PREP_COVAR_WORKER") == "1":
    os.environ.pop("DISPLAY", None)
    warnings.filterwarnings("ignore")
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.optimize import brentq

# numpy 2.0 renamed trapz -> trapezoid; this env is pinned at 1.26.4 for the
# frozen desilike/cosmoprimo SHAs. bao/core.py:27 installs the same shim (for
# velocileptors), and importing it below would supply this one as a side effect
# -- but module-level np.trapezoid calls here would then depend on import order.
# Guard locally instead.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# Suppress desilike import-time warnings (e.g. missing interpax/jax) before importing
warnings.filterwarnings("ignore")

from desilike.likelihoods.galaxy_clustering import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    CutskyFootprint,
    ObservablesCovarianceMatrix,
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.theories.galaxy_clustering import (
    KaiserTracerPowerSpectrumMultipoles,
    REPTVelocileptorsTracerPowerSpectrumMultipoles,
    ShapeFitPowerSpectrumTemplate,
)
from desilike.theories.primordial_cosmology import get_cosmo

from util import get_tracer_config

# Shared survey-physics + sampling machinery from the production BAO pipeline.
# Imported as a module (never `import core` — that would self-import when the
# scripts run with cwd=shapefit/). bao/ stays untouched and regression-frozen.
try:
    from desilike_emulator.bao import core as bao_core
except (ImportError, ModuleNotFoundError):
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "bao_core", str(Path(__file__).resolve().parent.parent / "bao" / "core.py")
    )
    bao_core = importlib.util.module_from_spec(_spec)
    sys.modules["bao_core"] = bao_core
    _spec.loader.exec_module(bao_core)

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# Re-exported shared engine pieces (analysis-agnostic).
generate_dataset = bao_core.generate_dataset
_worker_init = bao_core._worker_init
_RHO_CLIP = bao_core._RHO_CLIP


# ===========================================================================
# Priors / cosmology models (full-shape omega basis — matches bedcosmo's
# prior_args_fs.yaml; unlike BAO there is no (Om, hrdrag) compression, the
# full P(k) shape and amplitude are constrained).
# ===========================================================================
DEFAULT_PRIORS = {
    # Placeholder box; always overridden per tracer via util.ntracers_range.
    "N_tracers": {"dist": "uniform", "low": 1e5, "high": 1e7},
    "omega_cdm": {"dist": "uniform", "low": 0.01, "high": 0.99},
    "omega_b": {"dist": "normal", "mu": 0.02218, "sigma": 0.00055},
    "h": {"dist": "uniform", "low": 0.2, "high": 1.0},
    "ln10A_s": {"dist": "uniform", "low": 1.61, "high": 3.91},
    "n_s": {"dist": "normal", "mu": 0.9649, "sigma": 0.042},
    "w0": {"dist": "uniform", "low": -3.0, "high": 1.0},
    "wa": {"dist": "uniform", "low": -3.0, "high": 2.0},
}

CONSTRAINTS = {
    "high_z_matter_dom": {"params": ["w0", "wa"], "upper": 0.0},
}

COSMO_MODELS = {
    "base": ["omega_cdm", "omega_b", "h", "ln10A_s", "n_s"],
    "base_w": ["omega_cdm", "omega_b", "h", "ln10A_s", "n_s", "w0"],
    "base_w_wa": ["omega_cdm", "omega_b", "h", "ln10A_s", "n_s", "w0", "wa"],
}

# Fiducial values for fixed parameters.
PARAM_DEFAULTS = {"w0": -1.0, "wa": 0.0}

# ===========================================================================
# Emulator targets: 4x4 (qiso, qap, f_sigmar, m) marginalized covariance,
# emitted as 4 marginal sigmas + 6 pairwise correlations. The sigma_/rho_
# name prefixes are load-bearing: util.transform_emulator_targets_* and the
# bedcosmo decode guards (sigma floor/ceiling, tanh rho clamp) dispatch on
# them. Same targets for every tracer (no iso/aniso split in full shape).
# NOTE: pairwise rho clamps only guarantee PSD of 2x2 sub-blocks; the
# consumer assembling the full 4x4 must apply an eigenvalue-floor/nearest-PSD
# projection (documented follow-up in bedcosmo).
# ===========================================================================
_PHYS_NAMES = ["qiso", "qap", "f_sigmar", "m"]
_TRIU_I, _TRIU_J = np.triu_indices(4, k=1)
TARGET_NAMES = [f"sigma_{name}" for name in _PHYS_NAMES] + [
    f"rho_{_PHYS_NAMES[i]}_{_PHYS_NAMES[j]}" for i, j in zip(_TRIU_I, _TRIU_J)
]
# The MEAN emulator's `m` is DESI's m: the ShapeFit shape parameter of their
# Eq. (4.9),
#     P'_lin(k) = P^fid_lin(k) exp[ (m/a) tanh(a ln(k/kp)) + n ln(k/kp) ],
# which multiplies the FIDUCIAL template, so m = 0 means no shape change. It is
# a deviation by construction and the paper calls it plain `m` -- "dm" appears
# nowhere in DESI 2024 V.
#
# desilike names things differently: ITS `m` is the absolute log-slope of the
# de-wiggled spectrum at kp (about -0.5775 at the DESI fiducial) and its `dm` is
# `m - m_fid`. So DESI's m == desilike's dm, and the mean worker reads
# `extractor.dm` while this target stays named `m`.
#
# Emitting the deviation rather than the absolute slope also removes the
# CHANGELOG S24 hazard from the interface: nothing downstream has to subtract a
# fiducial that turned out to be theory-dependent (REPT's attached template
# reports -0.6699 where the extractor says -0.5775).
#
# The COVAR side is untouched either way: sigma is offset-invariant, so
# sigma(m) = sigma(dm) exactly.
MEAN_TARGET_NAMES = list(_PHYS_NAMES)

# DESI KP4.5 full-shape reference fit range and multipoles.
_ELLS = (0, 2)
_KLIM = (0.02, 0.2, 0.005)

# Effective survey area per data release [deg^2]. Mirrors the convention
# already established in bao/mcmc.py:66 (_DATASET_AREAS) and used by the
# production config-space driver (bao/config_space.py:56 _AREA = 7500) and
# bao/desi_reference.py:33.
#
# shapefit previously defaulted to 14000, which is the DR2 footprint, while
# being a DR1-only pipeline -- so every forecast ran DR1 galaxy counts over
# roughly twice the sky. That inflates V, depresses nbar = N/V, and pushes the
# HOD to a higher b1 for the same sample. Measured on LRG2 at the fiducial:
# b1 2.418 -> 2.171 and P0/DESI_measured 1.250 -> 1.032, i.e. the theory
# amplitude discrepancy against DESI's own spectra essentially closes.
DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}


def dataset_area(dataset: str = "dr1") -> float:
    """Effective footprint for a data release. Callers should route through
    this (or pass ``dataset=``) rather than hardcoding a number: freezing one
    release's area as a module default is exactly how the DR2 footprint ended
    up in a DR1-only pipeline."""
    try:
        return DATASET_AREAS[dataset]
    except KeyError:
        raise ValueError(
            f"No footprint area for dataset {dataset!r}; "
            f"known: {sorted(DATASET_AREAS)}") from None

# ShapeFit template pivot/slope conventions (desilike defaults; DESI/Brieden+21).
_SHAPEFIT_KP = 0.03
_SHAPEFIT_A = 0.6


def emulator_target_names(tracer_bin: str | None = None, dataset: str = "dr1") -> List[str]:
    """Per-tracer emulator targets — identical for all tracers (API symmetry
    with bao.core.emulator_target_names)."""
    return list(TARGET_NAMES)


# Physical-domain bounds on the DERIVED Omega_m, mirroring the BAO pipeline's
# Om prior [0.01, 0.99] (and its 0 < Om+Ok < 1 convention). The raw omega box
# (omega_cdm x h) reaches Omega_m ~ 17 in the high-omega_cdm/low-h corner,
# where cosmoprimo's wallish2018 de-wiggling filter produces non-finite pknow
# (CubicSpline "`y` must contain only finite values", bao_filter.py:420) —
# every observed generator failure had Omega_m >~ 2.9. Rejecting here is a
# fail-fast domain constraint (microseconds, before any CLASS init); the
# sampling rejection loop refills. The emulator's valid domain is therefore
# Omega_m in [0.01, 0.99] — the bedcosmo prior must carry the same constraint.
_OMEGA_M_MIN = 0.01
_OMEGA_M_MAX = 0.99


def _check_omega_m(omega_cdm: float, omega_b: float, h: float) -> float:
    """Validate the derived Omega_m (incl. the fixed fiducial neutrino density)."""
    if h <= 0.0:
        raise ValueError("h must be > 0 to compute Omega_m")
    Omega_m = (omega_cdm + omega_b + bao_core._OMEGA_NU_FID) / (h * h)
    if not (_OMEGA_M_MIN <= Omega_m <= _OMEGA_M_MAX):
        raise ValueError(
            f"unphysical Omega_m={Omega_m:.4f} outside "
            f"[{_OMEGA_M_MIN}, {_OMEGA_M_MAX}] "
            f"(omega_cdm={omega_cdm:.4f}, omega_b={omega_b:.5f}, h={h:.4f})"
        )
    return Omega_m


def _to_shapefit_cosmo_params(sample: Dict[str, float]) -> Dict[str, float]:
    """Convert an emulator prior sample to desilike/cosmoprimo cosmology params.

    Passes omega_cdm straight through (computing Omega_m = (omega_cdm+omega_b)/h^2
    instead would silently drop the neutrino density). w0/wa always map to
    w0_fld/wa_fld (NOT the historical 'w0_fde' typo in util.to_extractor_params,
    which cosmoprimo never accepted). Raises on Omega_m outside the physical
    domain (see _check_omega_m).
    """
    _check_omega_m(float(sample["omega_cdm"]), float(sample["omega_b"]),
                   float(sample["h"]))
    return {
        "omega_cdm": float(sample["omega_cdm"]),
        "omega_b": float(sample["omega_b"]),
        "h": float(sample["h"]),
        "logA": float(sample["ln10A_s"]),
        "n_s": float(sample["n_s"]),
        "w0_fld": float(sample.get("w0", PARAM_DEFAULTS["w0"])),
        "wa_fld": float(sample.get("wa", PARAM_DEFAULTS["wa"])),
    }


def _to_mean_extractor_params(sample: Dict[str, float]) -> Dict[str, float]:
    """Runtime params for ShapeFitPowerSpectrumExtractor's desilike pipeline.

    The pipeline's Cosmoprimo calculator exposes {h, Omega_m, omega_b, logA,
    n_s, w0_fld, wa_fld, ...} — not omega_cdm — so Omega_m must be assembled
    here. It includes the (fixed, DESI-fiducial m_ncdm=0.06) neutrino density:
    cosmoprimo recovers omega_cdm = Omega_m h^2 - omega_b - omega_ncdm, so
    omitting omega_ncdm (as the legacy util.to_extractor_params did) shifts
    omega_cdm by ~0.0006 relative to the covar path. This mapping keeps the
    mean and covar pipelines on the identical cosmology.
    """
    omega_cdm = float(sample["omega_cdm"])
    omega_b = float(sample["omega_b"])
    h = float(sample["h"])
    Omega_m = _check_omega_m(omega_cdm, omega_b, h)
    return {
        "h": h,
        "Omega_m": Omega_m,
        "omega_b": omega_b,
        "logA": float(sample["ln10A_s"]),
        "n_s": float(sample["n_s"]),
        "w0_fld": float(sample.get("w0", PARAM_DEFAULTS["w0"])),
        "wa_fld": float(sample.get("wa", PARAM_DEFAULTS["wa"])),
    }


# ===========================================================================
# Theory-dependent fiducial nuisances.
#
# Kaiser and REPT expose completely different parameter sets, so the fiducial
# `params` dict cannot be hardcoded. Dispatch is on the theory class rather
# than on an instance because params must exist before the observable is
# built. Unknown theories raise instead of silently under-parameterising --
# a missing nuisance would not error downstream, it would just produce a
# forecast that marginalises over less than it should, i.e. sigmas that are
# too tight in exactly the way that matters for experimental design.
# ===========================================================================
# desilike's REPT `tracer` preset selects (fsat, sigv), which set the width of
# the SN2 prior -- DESI's f_sat sigma_v^2 / nbar normalisation (Table 4). It
# accepts BGS/LRG/ELG/QSO only, so our MIX bin has to choose: LRG3_ELG1 maps to
# LRG because DESI's own 0.8-1.1 full-shape bin is LRG-only, which is the
# sample their prior was tuned against.
_REPT_TRACER_PRESET = {"BGS": "BGS", "LRG": "LRG", "ELG": "ELG",
                       "QSO": "QSO", "MIX": "LRG"}


def default_theory_kwargs(theory_cls, tracer_bin: str) -> Dict:
    """Theory kwargs when the caller passes none.

    Kaiser takes none. REPT needs prior_basis='physical' (so its nuisances are
    in DESI's units -- SN0 in 1/nbar, SN2 in f_sat sigma_v^2/nbar) and the
    per-tracer preset behind those units. Getting this from the tracer config
    rather than a caller argument keeps the generators, the regression harness
    and the validators on one definition.
    """
    if "Velocileptors" not in theory_cls.__name__:
        return {}
    ttype = str(get_tracer_config(tracer_bin).get("tracer_type", "")).strip().upper()
    if ttype not in _REPT_TRACER_PRESET:
        raise ValueError(
            f"No REPT tracer preset for tracer_type {ttype!r} (bin "
            f"{tracer_bin!r}). Add one rather than defaulting: the preset sets "
            f"fsat/sigv, which set the SN2 prior width.")
    return {"prior_basis": "physical", "tracer": _REPT_TRACER_PRESET[ttype]}


def theory_fiducial_params(theory_cls, *, b1, sigma_fog, sigma8_z):
    """Fiducial nuisance values for the chosen theory.

    Kaiser : {b1, sn0, sigmaper, sigmapar} -- sigmapar carries the FoG
             dispersion alone and sigmaper is 0 (CHANGELOG S6).
    REPT   : the 'physical' prior basis, b1p = (1 + b1_L) * sigma8(z) = b1_E *
             sigma8(z). Every other counterterm/stochastic parameter is left at
             its desilike prior centre, which is where DESI centres them too;
             they are floated and marginalised, which is the entire point of
             using REPT for a forecast.
    """
    name = theory_cls.__name__
    if "Kaiser" in name:
        return {"b1": float(b1), "sn0": 0.0,
                "sigmaper": 0.0, "sigmapar": float(sigma_fog)}
    if "Velocileptors" in name:
        return {"b1p": float(b1) * float(sigma8_z)}
    raise ValueError(
        f"No fiducial nuisance mapping for theory {name!r}. Add one rather "
        f"than falling back: an under-parameterised forecast produces sigmas "
        f"that are too tight, and nothing downstream will flag it.")


# ===========================================================================
# Full-shape Fisher band kernel.
#
# The BAO pipeline weights the FKP V_eff band integral with the BAO Fisher
# kernel k^4 exp(-k^2 Sigma_silk^2) (wiggle-derivative envelope). Full shape
# has no such envelope: to leading order every mode in the fit band carries
# shape/growth information, and the Gaussian Fisher per log-k is just the
# mode count dF/dk ~ k^2 times the FKP weight. So the band weight is k^2
# over the fit range [0.02, 0.2].
# ===========================================================================
_FS_K_GRID = np.linspace(_KLIM[0], _KLIM[1], 100)
_FS_W_K2 = _FS_K_GRID**2
_FS_W_K2_NORM = float(np.trapezoid(_FS_W_K2, _FS_K_GRID))


def _fs_band_weight_sq(nbar_3d: float, P_gk: np.ndarray) -> float:
    """FS-Fisher-weighted <(nP/(1+nP))^2> for one redshift slice.

    P_gk is the galaxy power at k = _FS_K_GRID. Returned value in [0, 1] is
    the effective fraction of the shell volume contributing to the full-shape
    covariance at this slice. (Clone of bao_core._fkp_band_weight_sq with the
    FS kernel.)
    """
    nP = float(nbar_3d) * np.asarray(P_gk, dtype=np.float64)
    fkp_sq = (nP / (1.0 + nP)) ** 2
    return float(np.trapezoid(_FS_W_K2 * fkp_sq, _FS_K_GRID) / _FS_W_K2_NORM)


def _fs_compute_z_eff(
    tracer_bin: str,
    cosmo,
    fo,
    area_deg2: float,
    b1: float,
) -> float:
    """Effective redshift from the n(z) slices.

    Dispatches on bao_core.Z_EFF_CONVENTION, so the FS and BAO pipelines
    cannot drift apart on this. Under the default ("desi_fkp") the FS-specific
    band averaging is irrelevant -- DESI's weight uses a fixed FKP pivot, not
    P(k) -- so this delegates and `fo`/`b1` go unused.

    The retained V_eff convention ("fisher_veff"):

    z_eff = sum_i z_i V_i_eff / sum_i V_i_eff with
    V_i_eff = V_i x <(n_i P_i(k)/(1+n_i P_i(k)))^2>_band, the FKP weight
    band-averaged over the FS kernel (clone of bao_core._compute_z_eff_from_nz
    but band-averaged instead of the single BAO pivot k=0.14).
    """
    if bao_core.Z_EFF_CONVENTION == "desi_fkp":
        return bao_core._desi_z_eff_from_nz(tracer_bin, cosmo, area_deg2)

    z_mid, z_edges, _frac, nbar_file = bao_core._load_nz_slice_fractions(tracer_bin)
    if z_mid.size == 0:
        raise ValueError(f"No valid n(z) slices for tracer {tracer_bin}")

    z_lo = z_edges[:, 0]
    z_hi = z_edges[:, 1]
    sky_frac = float(area_deg2) / 41252.96
    # cosmoprimo's comoving_radial_distance returns chi in Mpc/h already.
    chi_lo = np.asarray(cosmo.comoving_radial_distance(z_lo))
    chi_hi = np.asarray(cosmo.comoving_radial_distance(z_hi))
    V_bin = (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3) * sky_frac

    w_band = np.empty_like(z_mid, dtype=np.float64)
    for i, z in enumerate(z_mid):
        pk = bao_core._linear_pk_1d(fo, z=float(z))
        P_gk = float(b1) ** 2 * pk(_FS_K_GRID)
        w_band[i] = _fs_band_weight_sq(float(nbar_file[i]), P_gk)

    V_eff_per_slice = V_bin * w_band
    if V_eff_per_slice.sum() <= 0.0:
        # Degenerate FKP weights (e.g. nbar_file all zero); V-weighted fallback.
        return float(np.sum(V_bin * z_mid) / np.sum(V_bin))
    return float(np.sum(z_mid * V_eff_per_slice) / np.sum(V_eff_per_slice))


# ===========================================================================
# Likelihood builder
# ===========================================================================
def build_shapefit_likelihood(
    N_tracers: float,
    theta_cosmo: Dict[str, float],
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    area: float | None = None,
    dataset: str = "dr1",
    resolution: int = 3,
    tracer_config: Dict[str, float] | None = None,
    klim_spec: Tuple[float, float, float] = _KLIM,
    ells: Tuple[int, ...] = _ELLS,
    theory_cls=REPTVelocileptorsTracerPowerSpectrumMultipoles,
    theory_kwargs: Dict | None = None,
    float_sigma_damp: bool = True,
    cov_override: np.ndarray | None = None,
    wmatrix=None,
    skip_kmin_guard: bool = False,
) -> Dict:
    """Build the pre-recon full-shape Gaussian likelihood for one tracer bin.

    Mirrors bao_core.build_bao_likelihood minus every reconstruction step;
    the survey physics (HOD b1, n(z)-driven z_eff and FKP V_eff -> n_eff,
    SSC) is shared with the BAO pipeline, evaluated with the FS band kernel.

    Returns a dict with:
        likelihood   : ObservablesGaussianLikelihood
        template     : ShapeFitPowerSpectrumTemplate
        params       : fiducial {b1, sn0, sigmaper, sigmapar}
        observable   : TracerPowerSpectrumMultipolesObservable
        z_eff        : float
        f_sigmar_fid : float (template fiducial f·sigma_r; f_sigmar = df * this)
        m_fid        : float (template fiducial slope; m = m_fid + dm)
        cov_components : {"C_gauss", "C_SSC", "C_total"}
    """
    cfg = get_tracer_config(tracer_bin)
    if tracer_config is not None:
        cfg.update(tracer_config)
    if zrange is None:
        zrange = tuple(cfg["zrange"])

    tracer_type = str(cfg.get("tracer_type", "")).strip()
    if not tracer_type:
        raise ValueError(
            f"tracer_type missing from tracers.yaml entry for {tracer_bin!r}."
        )

    # Footprint: explicit override wins, else the dataset's area. Never a
    # frozen default -- see dataset_area().
    area = float(area) if area is not None else dataset_area(dataset)

    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    if z_eff is None:
        # Cosmology-clean: derive z_eff from the (data-only) n(z) slices with
        # the FS band weight. The b1 here is only a weighting fiducial (same
        # convention as the BAO pipeline, which uses the recon-committed bias).
        try:
            z_eff = _fs_compute_z_eff(
                tracer_bin=tracer_bin,
                cosmo=cosmo,
                fo=fo,
                area_deg2=float(area),
                b1=float(cfg.get("bias_recon", 2.0)),
            )
        except (FileNotFoundError, ValueError):
            z_eff = float(cfg["z_eff"])

    z = z_eff

    # ------------------------------------------------------------------
    # FKP effective volume -> effective density (FS band kernel).
    # Per-slice nbar (cosmology-dependent via V_bin) drives per-slice HOD
    # b1(z_i); FKP^2 is band-averaged with the FS kernel; the Brent
    # root-find maps V_eff/V_shell back onto a single n_eff at z_eff.
    # Verbatim pattern from bao/core.py (V_eff block), FS kernel swapped in.
    # ------------------------------------------------------------------
    nbar_3d_eff: Optional[float] = None
    v_shell_for_footprint: Optional[float] = None
    try:
        z_mid_slice, z_edges_slice, frac_slice, _ = bao_core._load_nz_slice_fractions(
            tracer_bin
        )
    except FileNotFoundError as exc:
        print(f"[veff] {tracer_bin}: nz slices missing -- using V_shell. ({exc})")
    else:
        z_lo = z_edges_slice[:, 0]
        z_hi = z_edges_slice[:, 1]
        sky_frac = float(area) / 41252.96
        chi_lo = np.asarray(cosmo.comoving_radial_distance(z_lo))
        chi_hi = np.asarray(cosmo.comoving_radial_distance(z_hi))
        V_bin_slice = (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3) * sky_frac
        nbar_slice = (float(N_tracers) * frac_slice) / np.maximum(V_bin_slice, 1.0)

        fkp_wsq_list = []
        for zi, ni in zip(z_mid_slice, nbar_slice):
            if ni <= 0:
                fkp_wsq_list.append(0.0)
                continue
            b1_i, _logMcut_i, _sv2_i, _fsat_i = bao_core._hod_halo_props(
                nbar_comoving=float(ni), z=float(zi), cosmo=cosmo, fo=fo,
                tracer_type=tracer_type,
            )
            s8_d = max(float(fo.sigma8_z(zi, of="delta_cb")), 1.0e-30)
            f_z = float(fo.sigma8_z(zi, of="theta_cb")) / s8_d
            beta = f_z / max(b1_i, 1.0e-30)
            kaiser = 1.0 + 2.0 / 3.0 * beta + beta * beta / 5.0
            P_gk = (b1_i**2) * np.asarray(
                bao_core._linear_pk_1d(fo, z=float(zi))(_FS_K_GRID)) * kaiser
            fkp_wsq_list.append(_fs_band_weight_sq(float(ni), P_gk))
        fkp_wsq_per_bin = np.asarray(fkp_wsq_list, dtype=np.float64)

        v_eff, v_shell = bao_core._compute_v_eff_fkp(
            cosmo=cosmo, area_deg2=area, tracer_bin=tracer_bin,
            fkp_weight_sq_per_bin=fkp_wsq_per_bin,
        )
        v_shell_for_footprint = float(v_shell) if v_shell > 0 else None

        fkp_w2_avg = float(v_eff) / float(v_shell) if v_shell > 0 else 0.0
        if fkp_w2_avg > 0:
            idx_eff = int(np.argmin(np.abs(np.asarray(z_mid_slice) - z)))
            n_at_zeff = float(nbar_slice[idx_eff])
            if n_at_zeff > 0:
                b1_eff_z, _, _, _ = bao_core._hod_halo_props(
                    nbar_comoving=n_at_zeff,
                    z=float(z_mid_slice[idx_eff]),
                    cosmo=cosmo, fo=fo,
                    tracer_type=tracer_type,
                )
            else:
                b1_eff_z = 1.0
            s8d_eff = max(float(fo.sigma8_z(z, of="delta_cb")), 1.0e-30)
            f_eff = float(fo.sigma8_z(z, of="theta_cb")) / s8d_eff
            beta_eff = f_eff / max(b1_eff_z, 1.0e-30)
            kaiser_eff = 1.0 + 2.0 / 3.0 * beta_eff + beta_eff * beta_eff / 5.0
            P_g_zeff = (b1_eff_z**2) * kaiser_eff * np.asarray(
                bao_core._linear_pk_1d(fo, z=z)(_FS_K_GRID))

            def _band_minus_target(n_test: float) -> float:
                return _fs_band_weight_sq(float(n_test), P_g_zeff) - fkp_w2_avg

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

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------
    sigma8_delta = float(fo.sigma8_z(z, of="delta_cb"))
    sigma8_theta = float(fo.sigma8_z(z, of="theta_cb"))
    if sigma8_delta <= 0:
        raise ValueError(f"sigma8_delta <= 0 at z={z:.3f}: {sigma8_delta}")
    f = sigma8_theta / sigma8_delta

    # ------------------------------------------------------------------
    # Survey geometry (area = effective post-mask area; N_tracers = final
    # usable catalog size — same conventions as the BAO pipeline).
    # ------------------------------------------------------------------
    nbar_ang = N_tracers / area
    base_footprint = CutskyFootprint(
        area=area, zrange=zrange, nbar=nbar_ang, cosmo=cosmo,
    )
    nbar_comoving = float(N_tracers) / float(base_footprint.volume)

    # ------------------------------------------------------------------
    # Pre-recon damping fiducials + HOD nuisances.
    # ------------------------------------------------------------------
    pk_lin_1d = bao_core._linear_pk_1d(fo, z=z)
    sv2_1loop, sv2_1loop_dot, sv2_1loop_ddot = bao_core._sigma_v_sq_1loop_rsd(pk_lin_1d)
    sigma_perp_pre, sigma_par_pre = bao_core._sigma_nl_pre_from_pk(
        pk_lin_1d,
        f=f,
        sv2_1loop=sv2_1loop,
        sv2_1loop_dot=sv2_1loop_dot,
        sv2_1loop_ddot=sv2_1loop_ddot,
    )

    b1, _log_M_cut, sigma_v_sq_eff, _f_sat = bao_core._hod_halo_props(
        nbar_comoving=nbar_comoving,
        z=z,
        cosmo=cosmo,
        fo=fo,
        tracer_type=tracer_type,
    )

    # Interloper contamination: b1_obs = (1 - f_int) * b1_true (see
    # bao_core._get_standard_tracer_bao_params for provenance).
    f_int = float(cfg.get("f_interloper", 0.0))
    if not 0.0 <= f_int < 1.0:
        raise ValueError(f"f_interloper must be in [0, 1), got {f_int}")
    b1 = b1 * (1.0 - f_int)

    # Redshift-measurement error adds to the satellite-virial dispersion in
    # quadrature (no f_sat prefactor — every galaxy carries it).
    z_err_kms = float(cfg.get("z_error_kms", 0.0))
    sigma_v_sq_total = sigma_v_sq_eff + z_err_kms * z_err_kms
    sigma_fog = bao_core._sigma_fog_from_sigma_v_sq(sigma_v_sq_total, z=z, cosmo=cosmo)

    # Kaiser's damping is a Gaussian
    #   exp(-k^2 (sigmapar^2 mu^2 + sigmaper^2 (1 - mu^2)) / 2)
    # applied by desilike to the ENTIRE pktable, not to the wiggle component
    # (full_shape.py:492-497 — contrast bao.py:126,138, where the identical
    # algebra multiplies only the oscillatory part). Two consequences fix the
    # parameters we may pass:
    #
    #   sigmapar = sigma_fog ALONE. The pre-recon BAO scales (Sigma_par,
    #     Sigma_perp) describe smearing of the BAO *feature* by large-scale
    #     displacements; feeding them here suppresses the broadband instead.
    #     The original build added sigma_fog to Sigma_par in quadrature, giving
    #     sigma_par = 8.18 Mpc/h for LRG2 — enough to drive P2 NEGATIVE above
    #     k ~ 0.155 where DESI measures +2000, and P0 34% low at k = 0.18
    #     (CHANGELOG S5).
    #   sigmaper = 0. Finger-of-God suppression is purely line-of-sight. A
    #     transverse Gaussian damps the monopole isotropically for no physical
    #     reason; Sigma_perp = 4.28 Mpc/h was doing exactly that.
    #
    # Sigma_par/Sigma_perp are still computed above: they remain meaningful as
    # diagnostics and are recorded in the footprint attrs. They simply do not
    # belong in a full-shape broadband damping.
    #
    # Not modelled, and deliberately so: BAO smearing itself. It is an
    # IR-resummation effect that belongs inside the theory (REPT provides it),
    # and this template hands Kaiser the full linear pk_dd with no wiggle-only
    # damping available. sigma(qiso) is therefore somewhat optimistic in the
    # Kaiser path, since the BAO feature is sharper than reality.
    params = theory_fiducial_params(
        theory_cls, b1=b1, sigma_fog=sigma_fog, sigma8_z=sigma8_delta)

    # ------------------------------------------------------------------
    # Final footprint with the V_eff-matched effective density injected
    # (CutskyFootprint stores an angular density and recovers nbar_3d = N/V,
    # so n_eff * V_shell / area lands nbar_3d at n_eff while keeping
    # footprint.volume = V_shell).
    # ------------------------------------------------------------------
    nbar_for_footprint = nbar_ang
    if (nbar_3d_eff is not None and v_shell_for_footprint is not None
            and float(area) > 0):
        nbar_for_footprint = (
            float(nbar_3d_eff) * float(v_shell_for_footprint) / float(area)
        )

    footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar_for_footprint,
        cosmo=cosmo,
        attrs={
            "tracer_bin": tracer_bin,
            "nbar_comoving": nbar_comoving,
            "f_cosmo": f,
            "sigmaper_pre": sigma_perp_pre,
            "sigmapar_pre": sigma_par_pre,
            "sigma_fog": sigma_fog,
        },
    )

    # Survey-window kmin guard (same approximation as the BAO pipeline; at
    # DESI volumes this leaves the fit kmin untouched).
    L_survey = base_footprint.volume ** (1 / 3)
    kmin_window = 2.0 * np.pi / L_survey
    kmin_eff = max(float(klim_spec[0]), kmin_window)
    if skip_kmin_guard:
        # Only for building an auxiliary covariance on a prescribed k-grid (the
        # window's theory grid starts at k=0.001, below the guard). Never for a
        # fit range.
        kmin_eff = float(klim_spec[0])

    # ------------------------------------------------------------------
    # ShapeFit template + full-shape theory
    # ------------------------------------------------------------------
    template = ShapeFitPowerSpectrumTemplate(
        z=z,
        fiducial=("DESI", dict(theta_cosmo)),
        apmode="qisoqap",
        with_now="wallish2018",
    )

    theory = theory_cls(template=template,
                        **(theory_kwargs if theory_kwargs is not None
                           else default_theory_kwargs(theory_cls, tracer_bin)))

    klim = {int(ell): [kmin_eff, float(klim_spec[1]), float(klim_spec[2])]
            for ell in ells}
    # wmatrix: DESI's survey window, as an lsstypes.WindowMatrix (a bundle path
    # also works but loads the whole GaussianLikelihood, which desilike then
    # rejects -- pass the matrix). Default None keeps the pipeline windowless,
    # which is what every Fourier path in this repo has always been
    # (bao/core.py:1625 calls its kmin guard "a simplified approximation to the
    # full DESI window-function"). Supplying one makes desilike evaluate the
    # theory on the window's own k-grid and convolve it down to the fit bins.
    #
    # Two things to know before using this.
    #
    # 1. The DR1 full-shape bundles ship the ROTATED window. Its theory axis is
    #    the P0/P2/P4 spectrum block (3 x 349) PLUS two "rotation" and one
    #    "photo" nuisance-template columns, i.e. an ObservableTree, and
    #    desilike's WindowedPowerSpectrumMultipoles wants a plain poles->poles
    #    matrix (it reads wmatrix.theory.ells, which a tree has no attribute
    #    for -- the "lsstypes 1.1.0 vs desilike" AttributeError). Select the
    #    spectrum block on both axes first:
    #        W = (bundle.window.at.theory.get(observables='spectrum')
    #                          .at.observable.get(observables='spectrum'))
    #    That drops the 3 systematic-template columns DESI marginalizes over,
    #    so the resulting Fisher is mildly optimistic on that account.
    #
    # 2. *** THE COVARIANCE DOES NOT FOLLOW. *** desilike's
    #    ObservablesCovarianceMatrix has no window handling whatsoever (grep
    #    wmatrix/window in observables/galaxy_clustering/covariance.py: no
    #    hits), so it returns the unconvolved Gaussian covariance regardless.
    #    Measured on LRG2: passing W changes the covariance diagonal by 0.5%
    #    and the P0 nearest-neighbour correlation from 0.059 to 0.063, while
    #    DESI's own covariance for the same estimator sits at 0.666. So a run
    #    with wmatrix set has SMOOTHED DERIVATIVES AND AN UNSMOOTHED
    #    COVARIANCE, which double-counts the window's information loss and
    #    inflates sigma (LRG2: 1.38x on qiso, 1.30x on qap). Do not read those
    #    sigmas as "the windowed answer". The consistent object is
    #    C_obs = M C_kin M^T, with C_kin the Gaussian+SSC covariance on the
    #    window's own theory grid -- build it by calling this function again
    #    with ells=(0,2,4), klim_spec=the window k-edges, skip_kmin_guard=True,
    #    and pass the result back through cov_override alongside wmatrix.
    #
    # Doing that (CHANGELOG S19) shows the window is very nearly a NO-OP for
    # the compressed parameters: on LRG2 the consistently-windowed sigmas are
    # 0.81/0.79/0.85/0.80 of DESI against 0.82/0.85/0.91/0.87 unwindowed, and
    # every rho moves by <0.03. Derivative smoothing and covariance correlation
    # cancel. It is worth ~36 s/cosmology, so production stays windowless; the
    # residual ~20% gap to DESI is nuisance freedom and non-Gaussian covariance,
    # not geometry.
    #
    # Also unresolved for production: W is DR1 geometry derived from the random
    # catalogue at fiducial-cosmology distances, so windowing while also
    # recomputing V/nbar/z_eff per cosmology mixes frames (the issue
    # bao/config_space.py:626-632 documents). Diagnostic use only.
    obs_kwargs = {}
    if wmatrix is not None:
        obs_kwargs["wmatrix"] = str(wmatrix) if isinstance(wmatrix, (str, Path)) else wmatrix
    observable = TracerPowerSpectrumMultipolesObservable(
        data=params,
        klim=klim,
        theory=theory,
        **obs_kwargs,
    )

    # Gaussian covariance. theories= is passed explicitly: with a template-based
    # full-shape chain the auto-detection walk can pick the wrong calculator.
    covariance = ObservablesCovarianceMatrix(
        observable,
        footprints=footprint,
        theories=[theory],
        resolution=resolution,
    )
    C_gauss_arr = bao_core._cov_to_array(covariance(**params))

    C_full = C_gauss_arr.copy()
    cov_components: Dict[str, np.ndarray] = {"C_gauss": C_gauss_arr}

    # ------------------------------------------------------------------
    # Non-Gaussian addition: SSC only (the shifted-random shot boost is a
    # reconstruction-only term and does not exist pre-recon).
    # ------------------------------------------------------------------
    ell_tuple = tuple(int(ell) for ell in observable.ells)
    k_centers = np.asarray(observable.k[0], dtype=np.float64)
    V_survey = float(footprint.volume)

    observable(**params)
    # Take the multipoles off the OBSERVABLE, not the theory calculator. With a
    # wmatrix the theory lives on the window's own k-grid (349 points per ell
    # for the DR1 bundles) while the observable is the 36-bin convolved vector,
    # and the SSC response has to be built on the same grid as C_gauss. Without
    # a wmatrix flattheory is the theory evaluated on the observable bins, so
    # this is a no-op (checked bit-exact against observable.theory on LRG2).
    pk_multipoles = np.asarray(
        observable.flattheory, dtype=np.float64
    ).reshape(len(ell_tuple), -1)

    sigma_b_sq = bao_core._sigma_b_sq(cosmo, fo, z, V_survey)
    C_ssc = bao_core._ssc_cov(
        k_centers=k_centers,
        ells=ell_tuple,
        pk_multipoles=pk_multipoles,
        sigma_b_sq=sigma_b_sq,
    )
    if C_ssc.shape == C_full.shape:
        C_full += C_ssc
        cov_components["C_SSC"] = np.array(C_ssc, copy=True)

    # Validation hook: substitute an externally-supplied covariance (e.g. DESI's
    # own EZmock covariance) for our analytic one, holding the theory, the
    # derivatives and the marginalization fixed. That isolates how much of any
    # sigma difference is the covariance rather than the model — see
    # compare_to_desi.py. Never used by the generators (default None).
    if cov_override is not None:
        cov_override = np.asarray(cov_override, dtype=np.float64)
        if cov_override.shape != C_full.shape:
            raise ValueError(
                f"cov_override shape {cov_override.shape} != observable "
                f"covariance shape {C_full.shape}"
            )
        cov_components["C_analytic"] = np.array(C_full, copy=True)
        C_full = cov_override.copy()

    if not np.all(np.isfinite(C_full)):
        raise ValueError("Non-finite entries in augmented covariance")

    # Template fiducials — the theory ran above, so the template is initialized.
    f_sigmar_fid = float(template.f_sigmar_fid)
    m_fid = float(template.m_fid)

    # Plain ndarray covariance (lsstypes-migration lesson: the array branch of
    # ObservablesGaussianLikelihood is version-stable; the wrapper round-trip
    # is the identity here and routes into upstream debug branches).
    likelihood = ObservablesGaussianLikelihood(
        observables=observable,
        covariance=C_full,
    )

    # Float the Gaussian damping scales with priors centered on the
    # analytically-computed values (analog of the BAO float_sigma_bao;
    # width 2.0 Mpc/h, the DESI Adame+24 Sec 4.2.1 convention). sn0 keeps
    # its desilike yaml prior (norm, scale 1000) — the only stochastic term.
    # Only sigmapar floats. The FoG scale is genuinely uncertain, so it is
    # marginalized with a prior centered on the HOD-derived value. sigmaper
    # stays FIXED at 0: it is not an uncertain quantity but an absent one
    # (FoG is line-of-sight only), and floating a physically-zero parameter
    # would open a marginalization direction that removes real information.
    if float_sigma_damp and "sigmapar" in params:
        likelihood.all_params["sigmapar"].update(
            fixed=False,
            prior={"dist": "norm", "loc": float(params["sigmapar"]), "scale": 2.0},
        )

    cov_components["C_total"] = np.array(C_full, copy=True)
    return {
        "likelihood": likelihood,
        "template": template,
        "params": params,
        "observable": observable,
        "z_eff": float(z),
        "f_sigmar_fid": f_sigmar_fid,
        "m_fid": m_fid,
        "shapefit_kp": _SHAPEFIT_KP,
        "shapefit_a": _SHAPEFIT_A,
        "cov_components": cov_components,
        # Survey-geometry diagnostics. n_eff is the FKP V_eff-matched effective
        # density that sets the shot-noise level of the covariance; DESI ships
        # the measured equivalent as num_shotnoise/norm in its full-shape
        # bundles, so exposing it makes that comparison a one-liner
        # (compare_to_desi.py). None when the n(z) slices are unavailable and
        # the plain V_shell density is used instead.
        "n_eff": (None if nbar_3d_eff is None else float(nbar_3d_eff)),
        "nbar_comoving": float(nbar_comoving),
        "V_survey": float(footprint.volume),
    }
