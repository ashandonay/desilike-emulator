"""Fourier-space ShapeFit Fisher backend.

Single-bin Fisher forecast in P(k)-space with the Gaussian (FKP + SSC)
covariance built by ``core.build_shapefit_likelihood``. Owns the ShapeFit
Fisher reduction (marginalize all nuisances -> 4x4 covariance of
(qiso, qap, df, dm) -> physical basis (qiso, qap, f_sigmar, m)), the
sigma/rho target decomposition, and the picklable multiprocessing workers.

Entry points
------------
* run_fisher(sample, ...)          -> dict of the 10 sigma_/rho_ targets
* _worker_run_fisher_targets       : picklable spawn-Pool worker
  (the training-data CLI is generate_covar_data.py)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desilike import Fisher

# Robust import of the sibling core module (never a bare `import core`, which
# is ambiguous once both bao/ and shapefit/ have been put on sys.path).
try:
    from desilike_emulator.shapefit import core as sf_core
except (ImportError, ModuleNotFoundError):
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "shapefit_core", str(Path(__file__).resolve().parent / "core.py")
    )
    sf_core = importlib.util.module_from_spec(_spec)
    sys.modules["shapefit_core"] = sf_core
    _spec.loader.exec_module(sf_core)

TARGET_NAMES = sf_core.TARGET_NAMES
_PHYS_NAMES = sf_core._PHYS_NAMES

# Internal desilike parameter names of the ShapeFit block (template basis).
_SF_INTERNAL = ["qiso", "qap", "df", "dm"]


# ===========================================================================
# Fisher reduction (copied from bao/fourier_space.py — small, pure-numpy,
# already generic in the kept-parameter name list)
# ===========================================================================
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


def _cov_keep_from_fisher_matrix(
    F_matrix: np.ndarray,
    all_names: List[str],
    keep_names: List[str],
) -> np.ndarray:
    """Marginalized covariance of ``keep_names`` from a full Fisher matrix
    (Cholesky Schur solve; marginalizes every other parameter in the slice)."""
    F_matrix = _regularize_fisher_matrix(F_matrix)
    keep_idx = [all_names.index(p) for p in keep_names]

    E = np.zeros((F_matrix.shape[0], len(keep_idx)))
    for j, idx in enumerate(keep_idx):
        E[idx, j] = 1.0

    try:
        c, low = cho_factor(F_matrix, check_finite=False)
        cov_subset = cho_solve((c, low), E, check_finite=False)
        cov_keep = cov_subset[np.ix_(keep_idx, range(len(keep_idx)))]
    except np.linalg.LinAlgError:
        cov_full = np.linalg.pinv(F_matrix, rcond=1e-12)
        cov_keep = cov_full[np.ix_(keep_idx, keep_idx)]

    cov_keep = 0.5 * (cov_keep + cov_keep.T)
    return cov_keep


def _sf_fisher_reduction(info: Dict) -> np.ndarray:
    """Fisher for one likelihood -> marginalized 4x4 covariance in the
    PHYSICAL basis (qiso, qap, f_sigmar, m).

    desilike's ShapeFit template varies (qiso, qap, df, dm) internally with
        f_sigmar = df * f_sigmar_fid      (template.f_sigmar_fid)
        m        = m_fid + dm             (unit Jacobian)
    so J = diag(1, 1, f_sigmar_fid, 1).
    """
    likelihood = info["likelihood"]
    params = info["params"]

    fisher = Fisher(likelihood)
    fisher_result = fisher(**params)

    F_full = -np.array(fisher_result._hessian)
    all_names = [str(p) for p in fisher_result.names()]
    missing = [p for p in _SF_INTERNAL if p not in all_names]
    if missing:
        raise ValueError(
            f"ShapeFit params {missing} not in Fisher slice {all_names}"
        )

    cov_sf = _cov_keep_from_fisher_matrix(F_full, all_names, _SF_INTERNAL)

    J = np.diag([1.0, 1.0, float(info["f_sigmar_fid"]), 1.0])
    cov_phys = J @ cov_sf @ J.T
    return cov_phys


def fisher_cov_to_emulator_targets(cov_phys: np.ndarray) -> List[float]:
    """4x4 physical covariance -> [sigma_qiso, sigma_qap, sigma_f_sigmar,
    sigma_m, rho_* x 6] in TARGET_NAMES order (mirrors
    bao_core.fisher_sigmas_to_emulator_targets: sigma/rho marginals with the
    rho clip, PSD-safe per 2x2 block)."""
    cov_phys = np.asarray(cov_phys, dtype=np.float64)
    sigmas = np.sqrt(np.clip(np.diag(cov_phys), 0.0, None))
    out = [float(s) for s in sigmas]
    for i, j in zip(sf_core._TRIU_I, sf_core._TRIU_J):
        denom = max(float(sigmas[i] * sigmas[j]), 1e-30)
        rho = float(np.clip(cov_phys[i, j] / denom, -sf_core._RHO_CLIP,
                            sf_core._RHO_CLIP))
        out.append(rho)
    return out


# ===========================================================================
# Sample -> targets
# ===========================================================================
def run_fisher(
    sample: Dict[str, float],
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    param_defaults: Dict[str, float] | None = None,
    area: float | None = None,
    dataset: str = "dr1",
    resolution: int = 3,
    float_sigma_damp: bool = True,
    theory_cls=None,
    theory_kwargs: Dict | None = None,
) -> Dict[str, float]:
    """Convert a sample dict (N_tracers + cosmo params) to the 10 emulator
    targets, plus bookkeeping extras (z_eff, f_sigmar_fid, m_fid)."""
    if param_defaults:
        sample = {**param_defaults, **sample}

    N_tracers = float(sample["N_tracers"])
    theta_cosmo = sf_core._to_shapefit_cosmo_params(sample)

    kwargs = {}
    if theory_cls is not None:
        kwargs["theory_cls"] = theory_cls
    if theory_kwargs is not None:
        kwargs["theory_kwargs"] = theory_kwargs

    info = sf_core.build_shapefit_likelihood(
        N_tracers=N_tracers,
        theta_cosmo=theta_cosmo,
        tracer_bin=tracer_bin,
        zrange=zrange,
        z_eff=z_eff,
        area=area,
        dataset=dataset,
        resolution=resolution,
        float_sigma_damp=float_sigma_damp,
        **kwargs,
    )

    cov_phys = _sf_fisher_reduction(info)
    target_vals = fisher_cov_to_emulator_targets(cov_phys)

    out = dict(zip(TARGET_NAMES, target_vals))
    out["z_eff"] = float(info["z_eff"])
    out["f_sigmar_fid"] = float(info["f_sigmar_fid"])
    out["m_fid"] = float(info["m_fid"])
    return out


# ===========================================================================
# Multiprocessing workers (top-level + picklable for the spawn Pool).
# Contract (core.generate_dataset): worker(task) -> (sample, target_vals,
# tb_str); sample None signals failure.
# ===========================================================================
def _worker_run_fisher_targets(args_tuple):
    sample, tracer_bin, zrange, z_eff, param_defaults, area = args_tuple
    try:
        targets = run_fisher(
            sample,
            tracer_bin=tracer_bin,
            zrange=zrange,
            z_eff=z_eff,
            param_defaults=param_defaults,
            area=area,
        )
        target_vals = [targets[t] for t in TARGET_NAMES]
        if not all(np.isfinite(v) for v in target_vals):
            return None, None, "non-finite target values"
        return sample, target_vals, None
    except Exception:
        return None, None, traceback.format_exc()


def _worker_run_mean_targets(args_tuple):
    """Mean-pipeline worker: cosmology-only sample -> (qiso, qap, f_sigmar, m)
    via a per-process cached ShapeFitPowerSpectrumExtractor at the tracer's
    z_eff (see generate_mean_data.py)."""
    sample, tracer_bin, z_eff, param_defaults = args_tuple
    try:
        extractor = _get_mean_extractor(tracer_bin, z_eff)
        merged = {**(param_defaults or {}), **sample}
        theta = sf_core._to_mean_extractor_params(merged)
        extractor(**theta)
        extractor.get()
        target_vals = [
            float(extractor.qiso),
            float(extractor.qap),
            float(extractor.f_sigmar),
            # DESI's m (Eq. 4.9) == desilike's dm. See core.MEAN_TARGET_NAMES.
            float(extractor.dm),
        ]
        if not all(np.isfinite(v) for v in target_vals):
            return None, None, "non-finite target values"
        return sample, target_vals, None
    except Exception:
        return None, None, traceback.format_exc()


# Per-process extractor cache: construct-once/call-many (CLASS re-runs per
# call anyway, but template/fiducial assembly is not free).
_MEAN_EXTRACTOR_CACHE: Dict[Tuple[str, float], object] = {}


def _get_mean_extractor(tracer_bin: str, z_eff: float):
    key = (str(tracer_bin), round(float(z_eff), 6))
    if key not in _MEAN_EXTRACTOR_CACHE:
        from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumExtractor

        # with_now MUST be explicit — same de-wiggling trap as the template
        # (desilike defaults to 'peakaverage'; DESI uses Wallisch 2018).
        extractor = ShapeFitPowerSpectrumExtractor(
            z=float(z_eff),
            with_now="wallish2018",
        )
        _MEAN_EXTRACTOR_CACHE[key] = extractor
    return _MEAN_EXTRACTOR_CACHE[key]
