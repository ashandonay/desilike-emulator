"""DR1 σ *reference* sources — space-agnostic. These are what the config-space
pipeline (config_space.py) and the Fourier pipeline (core.py) are each
compared against; they belong to neither frame.

  _prod_fisher_sigmas(tracer)   — production Fisher (Fourier; analytic cov + Σ
                                  priors). Thin wrapper over core.run_fisher.
  _recon_sigmas(tracer, fid)    — DESI DR1 σ_q from bao-recon stat-only .h5,
                                  converted to σ(D/rd) using OUR fiducial (D/rd).
  _read_bao_recon(tracer)       — raw σ_q (and cov) from the bao-recon file.
  _read_desi_csv / _csv_sigma   — DESI DR1 published σ from desi_data.csv.
  _read_mcmc_prod / _bundle     — cached MCMC posterior σ (production / bundle).
"""
from __future__ import annotations
import os, sys, json, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import core
import fourier_space
from util import TRACER_CONFIGS

# ---------------------------------------------------------------------------
# Fixed analysis settings
# ---------------------------------------------------------------------------
_FID = {"Om": 0.3152, "hrdrag": 99.08}
_AREA = 7500.0
_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_NAME_MAP_TO_DATA = {"LRG3_ELG1": "LRG3+ELG1"}

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"

_MCMC_BUNDLE_DIR = Path(__file__).resolve().parent / "mcmc_results_bundle_native"
_MCMC_PROD_DIR   = Path(__file__).resolve().parent / "mcmc_results"
_MCMC_SUFFIX = {"BGS": "_qiso", "QSO": "_qiso"}  # sparse tracers use qiso file


def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP_TO_DATA.get(tracer, tracer)]


def _read_mcmc_bundle(tracer):
    """Native-theory MCMC on bundle data+cov+W (bundle substitution)."""
    suffix = _MCMC_SUFFIX.get(tracer, "")
    p = _MCMC_BUNDLE_DIR / f"{tracer}_dr1_native_theory{suffix}.json"
    if not p.exists():
        return {"DH_over_rs": float("nan"), "DM_over_rs": float("nan"), "DV_over_rs": float("nan")}
    d = json.loads(p.read_text())
    return {"DH_over_rs": float(d.get("sig_DH", float("nan"))),
            "DM_over_rs": float(d.get("sig_DM", float("nan"))),
            "DV_over_rs": float(d.get("sig_DV", float("nan")))}


def _read_mcmc_prod(tracer):
    """Pure-pipeline MCMC (analytic cov, no substitution) — from mcmc_bao.py."""
    p = _MCMC_PROD_DIR / f"{tracer}_dr1.json"
    if not p.exists():
        return {"DH_over_rs": float("nan"), "DM_over_rs": float("nan"), "DV_over_rs": float("nan")}
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
# Production Fisher (Fourier; analytic cov, no substitution)
# ---------------------------------------------------------------------------
def _prod_fisher_sigmas(tracer):
    """Return dict with σ(DH/rd), σ(DM/rd), σ(DV/rd) from production Fisher."""
    cfg = TRACER_CONFIGS[tracer]
    sample = {**core.PARAM_DEFAULTS, **_FID, "N_tracers": _get_ntracers(tracer)}
    res = fourier_space.run_fisher(
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
    sample_fid = {**core.PARAM_DEFAULTS, **_FID, "N_tracers": 300_000}
    theta, hrdrag = core._to_bao_cosmo_params(sample_fid)
    apmode = "qiso" if tracer in ("BGS", "QSO") else "qparqper"
    info = core.build_bao_likelihood(
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
