"""DR1 σ source helpers — library module, no main entry point.

Provides per-tracer σ(DH/rd, DM/rd, DV/rd) from each of the sources used
in the comparison plot (plot_fisher_vs_desi.py):

  _prod_fisher_sigmas(tracer)   — production Fisher (analytic cov + Σ priors,
                                  current hod.yaml). Reuses prep_covar.run_fisher.

  _bundle_fisher_sigmas(tracer) — Fisher with DESI bundle cov + W substituted,
                                  NATIVE ξ-theory (no FFTLog), DESI Σ priors
                                  (N(loc=pipeline_fid, σ=2) per Adame+24 §4.2.1),
                                  DESI {s⁰, s²} BB Schur-marginalized.

  _read_mcmc_prod(tracer)       — cached production MCMC posterior σ.
  _read_mcmc_bundle(tracer)     — cached bundle-cov MCMC posterior σ.

  _recon_sigmas(tracer, fid)    — DESI DR1 σ_q from bao-recon stat-only .h5
                                  converted to σ(D/rd) using OUR pipeline's
                                  fiducial (D/rd) (post-marg α-cov reference).

  _csv_sigma(df, tracer, q)     — DESI DR1 published σ from desi_data.csv.
"""
from __future__ import annotations
import os, sys, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import h5py
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
def _bundle_fisher_sigmas(tracer, cov_override=None):
    """Return dict with σ(DH/rd), σ(DM/rd), σ(DV/rd) from bundle-cov Fisher
    using native ξ-theory.

    cov_override: optional ndarray to substitute for bundle['cov']. Used by
    diagnostics that band/zero portions of the bundle cov to probe which
    off-diagonal structure is driving σ_q. Must match bundle data-vector shape.
    """
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


