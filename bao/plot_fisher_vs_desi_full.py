"""Unified DR1 σ comparison: production Fisher vs bundle-cov Fisher vs DESI references.

Produces one table with four σ columns per (tracer, quantity):

  σ_prod   — production Fisher (analytic cov + Σ priors), current hod.yaml,
             NO substitutions. Reuses prep_covar.run_fisher.

  σ_bundle — Fisher with DESI bundle cov + W substituted, NATIVE ξ-theory
             (DampedBAOWigglesTracerCorrelationFunctionMultipoles, no FFTLog),
             DESI Σ priors (N(loc=pipeline_fid, σ=2) per Adame+24 §4.2.1),
             DESI {s⁰, s²} BB Schur-marginalized.

  σ_csv    — DESI DR1 published σ from ~/data/desi/bao_dr1/desi_data.csv
             (cosmology-style; uses DESI's own fid (D/rd)).

  σ_recon  — DESI DR1 σ_q from bao-recon stat-only .h5 files, converted to
             σ(D/rd) using OUR pipeline's fiducial (D/rd) values. This is
             the cleanest per-tracer post-marg α-cov reference.

Output:
  bao/fisher_vs_desi_comparison_dr1.csv   (one row per (tracer, quantity))
  bao/fisher_vs_desi_comparison_dr1.png   (scatter plot per tracer)

No cached data is used.
"""
from __future__ import annotations
import os, sys, warnings
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import prep_covar
from util import TRACER_CONFIGS
from desilike.theories.galaxy_clustering import (
    DampedBAOWigglesTracerCorrelationFunctionMultipoles,
)
from compare_pipeline_theory_to_bundle import load_bundle, bb_basis

# ---------------------------------------------------------------------------
# Fixed analysis settings
# ---------------------------------------------------------------------------
_FID = {"Om": 0.3152, "hrdrag": 99.08}
_AREA = 7500.0
_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_NAME_MAP_TO_DATA = {"LRG3_ELG1": "LRG3+ELG1"}
_DESI_PRIOR_SCALE = {"sigmapar": 2.0, "sigmaper": 2.0, "sigmas": 2.0}

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"


def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP_TO_DATA.get(tracer, tracer)]

_MCMC_BUNDLE_DIR = Path(__file__).resolve().parent / "mcmc_results_bundle_native"
_MCMC_PROD_DIR   = Path(__file__).resolve().parent / "mcmc_results"
_MCMC_SUFFIX = {"BGS": "_qiso", "QSO": "_qiso"}  # sparse tracers use qiso file


def _read_mcmc_bundle(tracer):
    """Native-theory MCMC on bundle data+cov+W (bundle substitution)."""
    suffix = _MCMC_SUFFIX.get(tracer, "")
    p = _MCMC_BUNDLE_DIR / f"{tracer}_dr1_native_theory{suffix}.json"
    if not p.exists():
        return {"DH_over_rs": float("nan"), "DM_over_rs": float("nan"), "DV_over_rs": float("nan")}
    import json
    d = json.loads(p.read_text())
    return {"DH_over_rs": float(d.get("sig_DH", float("nan"))),
            "DM_over_rs": float(d.get("sig_DM", float("nan"))),
            "DV_over_rs": float(d.get("sig_DV", float("nan")))}


def _read_mcmc_prod(tracer):
    """Pure-pipeline MCMC (analytic cov, no substitution) — from mcmc_bao.py."""
    p = _MCMC_PROD_DIR / f"{tracer}_dr1.json"
    if not p.exists():
        return {"DH_over_rs": float("nan"), "DM_over_rs": float("nan"), "DV_over_rs": float("nan")}
    import json
    d = json.loads(p.read_text())
    return {"DH_over_rs": float(d.get("sigma_DH_over_rd_mcmc", float("nan"))),
            "DM_over_rs": float(d.get("sigma_DM_over_rd_mcmc", float("nan"))),
            "DV_over_rs": float(d.get("sigma_DV_over_rd_mcmc", float("nan")))}


_BAO_RECON_FILE = {
    "BGS":       "likelihood_bao-recon_stat-only_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_bao-recon_stat-only_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_bao-recon_stat-only_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_bao-recon_stat-only_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_bao-recon_stat-only_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_bao-recon_stat-only_QSO_GCcomb_z0.8-2.1.h5",
}


# ---------------------------------------------------------------------------
# DESI reference σ readers
# ---------------------------------------------------------------------------
def _read_desi_csv():
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    return df


def _csv_sigma(df, tracer, quantity):
    name = _NAME_MAP_TO_DATA.get(tracer, tracer)
    row = df[(df["tracer"] == name) & (df["quantity"] == quantity)]
    if len(row) == 0:
        return float("nan")
    return float(row["std"].iloc[0])


def _read_bao_recon(tracer):
    """Return σ_q values from the bao-recon stat-only file.

    Returns a dict with keys depending on file structure:
      {'qiso': σ}                 for 1×1 file (BGS, QSO)
      {'qpar': σ, 'qper': σ}      for 2×2 file (LRG, LRG+ELG, ELG)
    Returns None if the file does not exist.
    """
    p = _LIK_DIR / _BAO_RECON_FILE[tracer]
    if not p.exists():
        return None
    with h5py.File(p, "r") as h:
        cov = np.asarray(h["covariance/value"][...])
        sigmas = np.sqrt(np.diag(cov))
        if cov.shape == (1, 1):
            return {"qiso": float(sigmas[0])}
        return {"qpar": float(sigmas[0]), "qper": float(sigmas[1])}


# ---------------------------------------------------------------------------
# Production Fisher (analytic cov, no substitution)
# ---------------------------------------------------------------------------
def _prod_fisher_sigmas(tracer):
    """Return dict with σ(DH/rd), σ(DM/rd), σ(DV/rd) from production Fisher."""
    cfg = TRACER_CONFIGS[tracer]
    sample = {**prep_covar.PARAM_DEFAULTS, **_FID, "N_tracers": _get_ntracers(tracer)}
    res = prep_covar.run_fisher(
        sample, tracer_bin=tracer,
        zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_AREA,
    )
    # res is a dict with cov_DH/DH, cov_DH/DM, cov_DM/DM (cosmology units).
    var_DH = res["cov_DH_over_rd_DH_over_rd"]
    var_DM = res["cov_DM_over_rd_DM_over_rd"]
    cov_DHDM = res["cov_DH_over_rd_DM_over_rd"]
    sig_DH = float(np.sqrt(max(var_DH, 0.0)))
    sig_DM = float(np.sqrt(max(var_DM, 0.0)))

    # Compute σ(DV/rd) from (DH/rd, DM/rd) cov via AP transform.
    # DV/rd = (z · (DM/rd)² · (DH/rd))^(1/3)
    # ∂ln(DV)/∂(DH/rd) = 1/3 · 1/(DH/rd); same for DM with 2/3 factor.
    # σ²(DV/rd) = (DV/rd)² · [ (1/3 · 1/(DH/rd))² var_DH + (2/3 · 1/(DM/rd))² var_DM
    #                          + 2 · (1/3 · 1/(DH/rd)) · (2/3 · 1/(DM/rd)) · cov_DHDM ]
    sample_fid = {**prep_covar.PARAM_DEFAULTS, **_FID, "N_tracers": 300_000}
    theta, hrdrag = prep_covar._to_bao_cosmo_params(sample_fid)
    apmode = "qiso" if tracer in ("BGS", "QSO") else "qparqper"
    info = prep_covar.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_AREA, apmode=apmode,
    )
    tmpl = info["template"]
    DH = float(tmpl.DH_over_rd_fid)
    DM = float(tmpl.DM_over_rd_fid)
    DV = float(tmpl.DV_over_rd_fid)
    a = (1.0 / 3.0) / DH
    b = (2.0 / 3.0) / DM
    var_DV = (DV ** 2) * (a * a * var_DH + b * b * var_DM + 2.0 * a * b * cov_DHDM)
    sig_DV = float(np.sqrt(max(var_DV, 0.0)))
    return {"DH_over_rs": sig_DH, "DM_over_rs": sig_DM, "DV_over_rs": sig_DV,
            "DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV}


# ---------------------------------------------------------------------------
# Bundle-cov Fisher (native ξ-theory + DESI Σ priors + DESI BB Schur-marg)
# ---------------------------------------------------------------------------
def _bundle_fisher_sigmas(tracer):
    """Return dict with σ(DH/rd), σ(DM/rd), σ(DV/rd) from bundle-cov Fisher
    using native ξ-theory."""
    cfg = TRACER_CONFIGS[tracer]
    sample = {**prep_covar.PARAM_DEFAULTS, **_FID}
    theta, hrdrag = prep_covar._to_bao_cosmo_params(sample)
    apmode = "qiso" if tracer in ("BGS", "QSO") else "qparqper"
    info = prep_covar.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_AREA, apmode=apmode,
    )
    info["likelihood"](**info["params"])
    tmpl = info["template"]
    bundle = load_bundle(tracer)
    smoothing = float(cfg["smoothing_scale"])

    native = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=bundle["th_s"][0], ells=tuple(bundle["th_ells"]),
        template=tmpl, mode="recsym", smoothing_radius=smoothing,
        model="fog-damping",
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

    cov = 0.5 * (bundle["cov"] + bundle["cov"].T)
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


# ---------------------------------------------------------------------------
# Chi² profile curvature — captures nonlinearity in q that Fisher linearization
# misses. Computes chi²(q) scan with nuisances + BB Schur-marginalized at each
# q point, fits a quadratic at the minimum.
# ---------------------------------------------------------------------------
def _profile_sigmas(tracer):
    cfg = TRACER_CONFIGS[tracer]
    sample = {**prep_covar.PARAM_DEFAULTS, **_FID}
    theta, hrdrag = prep_covar._to_bao_cosmo_params(sample)
    apmode = "qiso" if tracer in ("BGS", "QSO") else "qparqper"
    info = prep_covar.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_AREA, apmode=apmode,
    )
    info["likelihood"](**info["params"])
    tmpl = info["template"]
    bundle = load_bundle(tracer)
    native = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=bundle["th_s"][0], ells=tuple(bundle["th_ells"]),
        template=tmpl, mode="recsym",
        smoothing_radius=float(cfg["smoothing_scale"]), model="fog-damping",
    )
    base = {k: float(info["params"][k]) for k in ("b1", "sigmapar", "sigmaper", "sigmas") if k in info["params"]}
    bb_zero = {p.name: 0.0 for p in native.all_params if (p.name.startswith("al") or p.name.startswith("bl"))}
    base.update(bb_zero)
    native_pnames = {p.name for p in native.all_params}
    fid_full = {"qpar": 1.0, "qper": 1.0, "qiso": 1.0, "dbeta": 1.0, **base}

    def predict(overrides):
        params = {**fid_full, **overrides}
        p_filt = {k: v for k, v in params.items() if k in native_pnames}
        native(**p_filt)
        xi = np.asarray(native.corr)
        xi_flat = np.concatenate([xi[i] for i in range(xi.shape[0])])
        return bundle["W"] @ xi_flat

    data = np.concatenate(bundle["obs_data"])
    n_data = data.size
    cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
    C_inv = np.linalg.inv(cov_sym)
    BB = bb_basis(bundle["obs_ells"], bundle["obs_s"], n_data, powers=(0, 2))
    A_bb = BB.T @ C_inv @ BB

    # Nuisance set matches Fisher's nl_params (minus q) — Schur-marg at each q point.
    nuis = (
        ["b1", "sigmas", "sigmapar", "sigmaper"]
        if apmode == "qiso"
        else ["b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]
    )
    steps = {"b1": 0.05, "dbeta": 0.05, "sigmas": 0.05, "sigmapar": 0.1, "sigmaper": 0.1}
    prior_idx = {"sigmas": 1 / 4.0, "sigmapar": 1 / 4.0, "sigmaper": 1 / 4.0}

    def chi2_marg_at_q(q_overrides):
        """chi² with (nuisances + BB) Schur-marg at this q point.

        Computes Jacobian of nuisances ONLY at this q (q held fixed), then
        analytically marginalizes the nuisance + BB linear sub-problem.
        """
        # Local Jacobian of nuisances at this q
        J_n = np.zeros((n_data, len(nuis)))
        for i, p in enumerate(nuis):
            J_n[:, i] = (predict({**q_overrides, p: fid_full[p] + steps[p]})
                         - predict({**q_overrides, p: fid_full[p] - steps[p]})) / (2 * steps[p])
        # Residual at fid nuisances
        r = data - predict(q_overrides)
        # Stack nuisance + BB and Schur-marginalize
        M = np.concatenate([J_n, BB], axis=1)
        A_full = M.T @ C_inv @ M
        # Σ priors on the nuisance block diagonal
        for i, p in enumerate(nuis):
            if p in prior_idx:
                A_full[i, i] += prior_idx[p]
        b_full = M.T @ C_inv @ r
        chi2_full = float(r @ C_inv @ r)
        chi2_m = chi2_full - float(b_full @ np.linalg.solve(A_full, b_full))
        return chi2_m

    DH = float(tmpl.DH_over_rd_fid)
    DM = float(tmpl.DM_over_rd_fid)
    DV = float(tmpl.DV_over_rd_fid)

    if apmode == "qiso":
        # Sparse tracer likelihood is often asymmetric (ξ₀-only data with sharp
        # BAO feature) — local parabola underestimates the true posterior width.
        # Use Δχ²=1 half-width: find q's where chi² = chi²_min + 1, take half
        # the interval. This is the canonical "1σ profile-likelihood" σ and
        # accounts for tail asymmetry the way a marginal-posterior MCMC does.
        qs = np.linspace(0.90, 1.10, 41)
        c2 = np.array([chi2_marg_at_q({"qiso": float(q)}) for q in qs])
        c2 -= c2.min()
        imin = int(np.argmin(c2))
        # Interpolate Δχ²=1 crossings on each side
        def _cross(q_arr, c_arr, idx_start, direction):
            if direction == "left":
                rng = range(idx_start, -1, -1)
            else:
                rng = range(idx_start, len(q_arr))
            for i in rng:
                if c_arr[i] >= 1.0:
                    j = i + (1 if direction == "left" else -1)
                    if 0 <= j < len(q_arr):
                        t = (1.0 - c_arr[j]) / (c_arr[i] - c_arr[j])
                        return q_arr[j] + t * (q_arr[i] - q_arr[j])
                    return q_arr[i]
            return float("nan")
        q_lo = _cross(qs, c2, imin, "left")
        q_hi = _cross(qs, c2, imin, "right")
        if not (np.isfinite(q_lo) and np.isfinite(q_hi)):
            # Fall back to parabolic fit
            lo, hi = max(0, imin - 2), min(len(qs), imin + 3)
            p = np.polyfit(qs[lo:hi], c2[lo:hi], 2)
            sig_q = float(1.0 / np.sqrt(2.0 * p[0]))
        else:
            sig_q = 0.5 * (q_hi - q_lo)
        sig_DV = sig_q * DV
        sig_DH = sig_q * DH; sig_DM = sig_q * DM
    else:
        # 2D scan: 7×7 grid covers ~±2σ in each direction
        qpars = np.linspace(0.96, 1.04, 7)
        qpers = np.linspace(0.96, 1.04, 7)
        c2 = np.zeros((len(qpars), len(qpers)))
        for i, qp in enumerate(qpars):
            for j, qq in enumerate(qpers):
                c2[i, j] = chi2_marg_at_q({"qpar": float(qp), "qper": float(qq)})
        # Fit 2D quadratic: chi² = a*x² + b*y² + c*x*y + ... around minimum
        # Use least-squares on the full grid with quadratic basis.
        i_min, j_min = np.unravel_index(np.argmin(c2), c2.shape)
        # Build local 3×3 stencil for Hessian
        i0 = max(1, min(len(qpars) - 2, i_min))
        j0 = max(1, min(len(qpers) - 2, j_min))
        x = qpars[i0 - 1:i0 + 2]; y = qpers[j0 - 1:j0 + 2]
        Z = c2[i0 - 1:i0 + 2, j0 - 1:j0 + 2]
        # Hessian by finite differences at (i0, j0)
        hxx = (Z[2, 1] - 2 * Z[1, 1] + Z[0, 1]) / (x[1] - x[0]) ** 2
        hyy = (Z[1, 2] - 2 * Z[1, 1] + Z[1, 0]) / (y[1] - y[0]) ** 2
        hxy = (Z[2, 2] - Z[2, 0] - Z[0, 2] + Z[0, 0]) / (4.0 * (x[1] - x[0]) * (y[1] - y[0]))
        # chi² Hessian = 2 * F_q (since chi² = -2 log L)
        H = 0.5 * np.array([[hxx, hxy], [hxy, hyy]])
        cov_q = np.linalg.inv(H)
        sig_qpar = float(np.sqrt(cov_q[0, 0]))
        sig_qper = float(np.sqrt(cov_q[1, 1]))
        sig_DH = sig_qpar * DH
        sig_DM = sig_qper * DM
        var_lnDV = (cov_q[0, 0] / 9.0) + (4.0 * cov_q[1, 1] / 9.0) + (4.0 * cov_q[0, 1] / 9.0)
        sig_DV = float(np.sqrt(max(var_lnDV, 0.0))) * DV

    return {"DH_over_rs": sig_DH, "DM_over_rs": sig_DM, "DV_over_rs": sig_DV,
            "DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV,
            "apmode": apmode}


# ---------------------------------------------------------------------------
# bao-recon σ converter
# ---------------------------------------------------------------------------
def _recon_sigmas(tracer, fid):
    """Convert bao-recon σ_q to σ(D/rd) using OUR fid (D/rd) values."""
    rec = _read_bao_recon(tracer)
    if rec is None:
        return {"DH_over_rs": float("nan"), "DM_over_rs": float("nan"), "DV_over_rs": float("nan")}
    DH = fid["DH_over_rd_fid"]; DM = fid["DM_over_rd_fid"]; DV = fid["DV_over_rd_fid"]
    if "qiso" in rec:
        s = rec["qiso"]
        return {"DH_over_rs": s * DH, "DM_over_rs": s * DM, "DV_over_rs": s * DV}
    sp = rec["qpar"]; sq = rec["qper"]
    # DV from (qpar, qper) is degenerate here without the cov off-diag from the recon file.
    # We re-read the off-diag to compute σ(DV/rd) correctly.
    p = _LIK_DIR / _BAO_RECON_FILE[tracer]
    with h5py.File(p, "r") as h:
        cov = np.asarray(h["covariance/value"][...])
    var_lnDV = (cov[0, 0] / 9.0) + (4.0 * cov[1, 1] / 9.0) + (4.0 * cov[0, 1] / 9.0)
    return {"DH_over_rs": sp * DH, "DM_over_rs": sq * DM,
            "DV_over_rs": float(np.sqrt(max(var_lnDV, 0.0))) * DV}


# ---------------------------------------------------------------------------
# Main: assemble the table
# ---------------------------------------------------------------------------
def main():
    df_csv = _read_desi_csv()
    rows = []

    for tracer in _TRACERS:
        print(f"== {tracer} ==", flush=True)
        prod = _prod_fisher_sigmas(tracer)
        print(f"  σ_prod  DH={prod['DH_over_rs']:.4f} DM={prod['DM_over_rs']:.4f} DV={prod['DV_over_rs']:.4f}", flush=True)
        bundle = _bundle_fisher_sigmas(tracer)
        print(f"  σ_bundle DH={bundle['DH_over_rs']:.4f} DM={bundle['DM_over_rs']:.4f} DV={bundle['DV_over_rs']:.4f}  (apmode={bundle['apmode']})", flush=True)
        mcmc_prod = _read_mcmc_prod(tracer)
        print(f"  σ_mcmc_prod   DH={mcmc_prod['DH_over_rs']:.4f} DM={mcmc_prod['DM_over_rs']:.4f} DV={mcmc_prod['DV_over_rs']:.4f}", flush=True)
        mcmc = _read_mcmc_bundle(tracer)
        print(f"  σ_mcmc_bundle DH={mcmc['DH_over_rs']:.4f} DM={mcmc['DM_over_rs']:.4f} DV={mcmc['DV_over_rs']:.4f}", flush=True)
        recon = _recon_sigmas(tracer, prod)
        print(f"  σ_recon DH={recon['DH_over_rs']:.4f} DM={recon['DM_over_rs']:.4f} DV={recon['DV_over_rs']:.4f}", flush=True)

        # Decide which quantities to report per tracer.
        if tracer in ("BGS", "QSO"):
            quantities = ["DV_over_rs"]
        else:
            quantities = ["DH_over_rs", "DM_over_rs"]

        for q in quantities:
            sigma_csv = _csv_sigma(df_csv, tracer, q)
            row = {
                "tracer": tracer, "quantity": q,
                "sigma_prod_Fisher":   prod[q],
                "sigma_prod_MCMC":     mcmc_prod[q],
                "sigma_bundle_Fisher": bundle[q],
                "sigma_bundle_MCMC":   mcmc[q],
                "sigma_csv":    sigma_csv,
                "sigma_recon":  recon[q],
            }
            ref = row["sigma_recon"] if (row["sigma_recon"] and row["sigma_recon"] > 0) else sigma_csv
            # Ratios vs bao-recon (preferred per-tracer reference; falls back to csv if missing).
            for k in ("sigma_prod_Fisher", "sigma_prod_MCMC", "sigma_bundle_Fisher", "sigma_bundle_MCMC"):
                row[f"r_{k.replace('sigma_', '')}"] = (row[k] / ref) if (ref and ref > 0 and np.isfinite(row[k])) else float("nan")
            rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = Path(__file__).resolve().parent / "fisher_vs_desi_comparison_dr1.csv"
    out.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved {out_csv}")
    print()
    print(out.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(13, 7))
    qcolor = {"DH_over_rs": "C0", "DM_over_rs": "C1", "DV_over_rs": "C2"}
    qlabel = {"DH_over_rs": r"$\sigma(D_H/r_d)$",
              "DM_over_rs": r"$\sigma(D_M/r_d)$",
              "DV_over_rs": r"$\sigma(D_V/r_d)$"}
    marker = {"sigma_prod_Fisher": "o", "sigma_prod_MCMC": "^",
              "sigma_bundle_Fisher": "s", "sigma_bundle_MCMC": "v",
              "sigma_csv": "x", "sigma_recon": "D"}
    fill = {"sigma_prod_Fisher": "full", "sigma_prod_MCMC": "full",
            "sigma_bundle_Fisher": "full", "sigma_bundle_MCMC": "full",
            "sigma_csv": "none", "sigma_recon": "none"}
    label_of = {"sigma_prod_Fisher": "Fisher (prod analytic cov)",
                "sigma_prod_MCMC":   "MCMC (prod analytic cov)",
                "sigma_bundle_Fisher": "Fisher (bundle cov + native θ)",
                "sigma_bundle_MCMC":   "MCMC (bundle cov + native θ)",
                "sigma_csv": "DESI desi_data.csv",
                "sigma_recon": "DESI bao-recon file"}

    xpos = {t: i for i, t in enumerate(_TRACERS)}
    seen = set()
    for _, row in out.iterrows():
        x = xpos[row["tracer"]]
        q = row["quantity"]
        for col in ("sigma_prod_Fisher", "sigma_prod_MCMC", "sigma_bundle_Fisher", "sigma_bundle_MCMC", "sigma_csv", "sigma_recon"):
            v = row[col]
            if not np.isfinite(v):
                continue
            key = (col, q)
            lab = f"{label_of[col]} {qlabel[q]}" if key not in seen else None
            seen.add(key)
            mfc = qcolor[q] if fill[col] == "full" else "none"
            ax.scatter([x], [v], s=110, marker=marker[col],
                       facecolors=mfc, edgecolors=qcolor[q],
                       linewidths=1.8, label=lab, zorder=3)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([t.replace("_", "+") for t in _TRACERS])
    ax.set_ylabel(r"$\sigma$")
    ax.set_xlabel("Tracer bin")
    ax.set_title("BAO σ — production Fisher, bundle-cov Fisher (native θ), DESI csv, DESI bao-recon\n"
                 f"Current hod.yaml lit-AB at Om={_FID['Om']}, hrdrag={_FID['hrdrag']}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    out_png = Path(__file__).resolve().parent / "fisher_vs_desi_comparison_dr1.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
