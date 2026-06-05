"""Fourier-space BAO Fisher backend.

Single-bin Fisher forecast in P(k)-space with the analytic Gaussian (FKP +
SSC + recon-shot) covariance built by ``core.build_bao_likelihood``. This
module owns the Fourier-specific parts: the q-parameter Fisher reduction, the
σ-mapping, the multiprocessing workers, and the legacy cov-triplet CLI.

Sparse tracers (``_ISO_TRACERS`` = BGS, QSO) use an isotropic monopole-only qiso
fit; all other tracers use the anisotropic qpar/qper fit (see
get_bao_fisher_covariance). The shared BAO modelling, cosmology mapping, sampling
and generate_dataset engine live in core; the config-space counterpart is
config_space.py.

Entry points
------------
* run_fisher(sample, ...)                  → cov triplet + fiducial D/rd
* _worker_run_fisher / _worker_run_fisher_sigma : picklable Pool workers
* main()                                   → legacy cov-triplet training-data CLI
  (the σ-triplet unified CLI is generate_emulator_data.py)
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
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
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate

import core
from core import (
    COSMO_MODELS,
    CONSTRAINTS,
    DEFAULT_PRIORS,
    PARAM_DEFAULTS,
    _ISO_TRACERS,
    _load_nz_slices,
    _to_bao_cosmo_params,
    build_bao_likelihood,
    generate_dataset,
)
from util import (
    TRACER_TYPE_CHOICES,
    get_default_save_path,
    get_tracer_config,
    ntracers_range,
    parse_priors,
    save_dataset,
)

# Fourier cov-triplet target: upper triangle of the 2x2 (DH/rd, DM/rd) covariance.
_PHYS_NAMES = ["DH_over_rd", "DM_over_rd"]
_TRIU_I, _TRIU_J = np.triu_indices(2)
TARGET_NAMES = [f"cov_{_PHYS_NAMES[i]}_{_PHYS_NAMES[j]}" for i, j in zip(_TRIU_I, _TRIU_J)]

# Unified distance-error target (shared with config_space); defined in core.
SIGMA_TARGET_NAMES = core.SIGMA_TARGET_NAMES


# ===========================================================================
# q-parameter Fisher reduction
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

    # Isotropic fit exposes a single qiso dilation; anisotropic fits qpar/qper.
    bao_internal = ["qiso"] if "qiso" in all_names else ["qpar", "qper"]
    cov_q = _cov_q_from_fisher_matrix(F_full, all_names, bao_internal=bao_internal)

    # Convert marginalized covariance back to marginalized information.
    # This is equivalent to the Schur-complement Fisher on qpar/qper.
    try:
        F_q = np.linalg.inv(cov_q)
    except np.linalg.LinAlgError:
        F_q = np.linalg.pinv(cov_q, rcond=1e-12)

    F_q = 0.5 * (F_q + F_q.T)
    return F_q


# ===========================================================================
# Covariance → distance errors
# ===========================================================================
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
    """Compute the 2x2 BAO covariance matrix from a single-bin Fisher forecast.

    Sparse tracers (_ISO_TRACERS) use an isotropic qiso fit: the 1-parameter
    Fisher is mapped to a rank-1 (DH/rd, DM/rd) covariance so the returned triplet
    keeps its shape, and σ(DV/rd) recovered from it equals the direct qiso error.
    """
    apmode = "qiso" if tracer_bin in _ISO_TRACERS else "qparqper"
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
        apmode=apmode,
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

    if apmode == "qiso":
        # qiso scales DH and DM by the same dilation, so J = [DH_fid; DM_fid] and
        # cov_phys = J·Var(qiso)·Jᵀ is rank-1 (fully correlated). _cov_to_sigma_triplet
        # then gives σ(DV/rd) = √Var(qiso)·(DV/rd)_fid — the direct isotropic error.
        J = np.array([[DH_over_rd_fid], [DM_over_rd_fid]])
    else:
        J = np.diag([DH_over_rd_fid, DM_over_rd_fid])
    cov_phys = J @ cov_q @ J.T

    upper_tri_vals = cov_phys[_TRIU_I, _TRIU_J]
    out = dict(zip(TARGET_NAMES, upper_tri_vals))
    # Fiducial D/rd values: needed to convert the covariance to the unified
    # distance-error triplet (σ_DV/rd uses the AP combination — see
    # _cov_to_sigma_triplet). Extra keys; legacy callers read only TARGET_NAMES.
    out["DH_over_rd_fid"] = DH_over_rd_fid
    out["DM_over_rd_fid"] = DM_over_rd_fid
    out["DV_over_rd_fid"] = float(template.DV_over_rd_fid)
    return out


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
    the BAO (qpar/qper, or qiso for sparse tracers) Fisher information.
    """
    slices = _load_nz_slices(nz_slices_path)

    cfg = get_tracer_config(tracer_bin)
    if z_eff_bin is None:
        z_eff_bin = float(cfg["z_eff"])

    apmode = "qiso" if tracer_bin in _ISO_TRACERS else "qparqper"
    n_q = 1 if apmode == "qiso" else 2
    F_q_total = np.zeros((n_q, n_q), dtype=np.float64)

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
            apmode=apmode,
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
        apmode=apmode,
    )

    # Use template_bin.{DH,DM}_over_rd_fid directly — dimensionally correct.
    # The naive `template_bin.DH_fid / rd` form is dimensionally mixed
    # (DH_fid is Mpc/h, rd here is proper Mpc) and underreports σ by 1/h.
    DH_over_rd_fid = float(template_bin.DH_over_rd_fid)
    DM_over_rd_fid = float(template_bin.DM_over_rd_fid)

    # qiso → rank-1 (DH/rd, DM/rd) cov (see get_bao_fisher_covariance).
    if apmode == "qiso":
        J = np.array([[DH_over_rd_fid], [DM_over_rd_fid]])
    else:
        J = np.diag([DH_over_rd_fid, DM_over_rd_fid])
    cov_phys = J @ cov_q @ J.T

    upper_tri_vals = cov_phys[_TRIU_I, _TRIU_J]
    out = dict(zip(TARGET_NAMES, upper_tri_vals))
    out["DH_over_rd_fid"] = DH_over_rd_fid
    out["DM_over_rd_fid"] = DM_over_rd_fid
    out["DV_over_rd_fid"] = float(template_bin.DV_over_rd_fid)
    return out


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


def _cov_to_sigma_triplet(targets: Dict[str, float]) -> List[float]:
    """Convert the 2x2 (DH/rd, DM/rd) Fisher covariance from run_fisher into the
    unified distance-error triplet [σ(DH/rd), σ(DM/rd), σ(DV/rd)].

    σ(DV/rd) uses the AP combination DV/rd = (z·(DM/rd)²·(DH/rd))^(1/3), i.e.
    d ln DV = (1/3) d ln DH + (2/3) d ln DM, evaluated at the fiducial D/rd:
        Var(DV/rd)/(DV/rd)² = (1/9)Var(DH)/DH² + (4/9)Var(DM)/DM²
                              + (4/9)Cov(DH,DM)/(DH·DM).
    """
    c_hh = float(targets["cov_DH_over_rd_DH_over_rd"])
    c_hm = float(targets["cov_DH_over_rd_DM_over_rd"])
    c_mm = float(targets["cov_DM_over_rd_DM_over_rd"])
    x = float(targets["DH_over_rd_fid"])
    y = float(targets["DM_over_rd_fid"])
    dv = float(targets["DV_over_rd_fid"])
    s_dh = float(np.sqrt(max(c_hh, 0.0)))
    s_dm = float(np.sqrt(max(c_mm, 0.0)))
    var_lnDV = c_hh / (9.0 * x * x) + 4.0 * c_mm / (9.0 * y * y) + 4.0 * c_hm / (9.0 * x * y)
    s_dv = float(dv * np.sqrt(max(var_lnDV, 0.0)))
    return [s_dh, s_dm, s_dv]


# ===========================================================================
# Multiprocessing workers (top-level + picklable for the spawn Pool)
# ===========================================================================
def _worker_run_fisher(args_tuple):
    """Legacy cov-triplet worker. Returns (sample, target_vals, tb_str) where
    tb_str is None on success, a traceback string on failure, and
    "non-finite target values" if the Fisher returned non-finite targets."""
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


def _worker_run_fisher_sigma(args_tuple):
    """Fourier-space worker for the unified emulator generator. Same Fisher as
    _worker_run_fisher, but emits the distance-error triplet (σ_DH, σ_DM, σ_DV)
    so its output matches the config-space generator. Top-level + picklable for
    the spawn Pool. Returns (sample, target_vals, tb_str)."""
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
        target_vals = _cov_to_sigma_triplet(targets)
        if not all(np.isfinite(v) for v in target_vals):
            return None, None, "non-finite target values"
        return sample, target_vals, None
    except Exception:
        return None, None, traceback.format_exc()


# ===========================================================================
# Legacy CLI: Fourier cov-triplet training data
# (the unified σ-triplet CLI is generate_emulator_data.py --space fourier)
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data for a BAO Fisher error emulator "
                    "(Fourier-space 2x2 covariance triplet)."
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
        "--dataset", type=str, default="dr1", choices=["dr1", "dr2"],
        help="DESI release whose 'passed' counts anchor the N_tracers box "
             "(box = tracers.yaml low/high FACTORS x passed[dataset]).",
    )
    parser.add_argument(
        "--ntracers-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("NTRACERS_LOW", "NTRACERS_HIGH"),
        help="Override the N_tracers prior with ABSOLUTE bounds. Default is "
             "tracers.yaml low/high factors x the --dataset passed count.",
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
        nt_low, nt_high = args.ntracers_range
    else:
        nt_low, nt_high = ntracers_range(args.tracer_bin, args.dataset)
    priors["N_tracers"] = {"dist": "uniform", "low": nt_low, "high": nt_high}

    all_cosmo_keys = {"Om", "Ok", "w0", "wa", "hrdrag"}
    fixed_keys = all_cosmo_keys - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys if k in PARAM_DEFAULTS}

    constraints = {
        name: spec for name, spec in CONSTRAINTS.items()
        if all(p in model_params for p in spec["params"])
    }

    save_path = os.path.abspath(
        args.save_path if args.save_path else
        get_default_save_path(analysis="bao", quantity="covar",
                              cosmo_model=cosmo_model, dataset=args.dataset)
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
            worker_fn=_worker_run_fisher,
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
