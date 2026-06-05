"""DR1 σ *reference* sources — space-agnostic. These are what the config-space
pipeline (config_space.py) and the Fourier pipeline (core.py) are each
compared against; they belong to neither frame.

  _prod_fisher_sigmas(tracer)   — production Fisher (Fourier; analytic cov + Σ
                                  priors). Thin wrapper over core.run_fisher.
  _recon_sigmas(tracer, fid)    — DESI DR1 σ_q from bao-recon stat-only .h5,
                                  converted to σ(D/rd) using OUR fiducial (D/rd).
  _read_bao_recon(tracer)       — raw σ_q (and cov) from the bao-recon file.
  _read_desi_csv / _csv_sigma   — DESI DR1 published σ from desi_data.csv.
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

def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP_TO_DATA.get(tracer, tracer)]


_BAO_RECON_FILE = {
    "BGS":       "likelihood_bao-recon_stat-only_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_bao-recon_stat-only_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_bao-recon_stat-only_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_bao-recon_stat-only_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_bao-recon_stat-only_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_bao-recon_stat-only_QSO_GCcomb_z0.8-2.1.h5",
}
# Matching stat+systematics files (same reduction, same fiducial). The σ ratio
# syst/stat-only is DESI's measured systematic budget; see DESI_SYST_INFLATION.
_BAO_RECON_SYST_FILE = {t: f.replace("stat-only", "syst") for t, f in _BAO_RECON_FILE.items()}

# DESI DR1 systematic-error inflation σ_tot/σ_stat per tracer per distance,
# measured as σ(bao-recon_syst)/σ(bao-recon_stat-only) (the syst file IS σ_stat⊕sys,
# so this ratio already folds in the quadrature). Clamped to ≥1 (systematics only
# inflate; sub-noise negatives → 1.0, e.g. ELG2 D_H). Isotropic tracers (BGS, QSO)
# carry the single qiso factor on all three (DH/DM/DV are degenerate projections).
# Regenerate with `python reference_sigmas.py --regen-syst` (needs the _syst .h5).
DESI_SYST_INFLATION = {
    "BGS":       {"DH_over_rs": 1.0071, "DM_over_rs": 1.0071, "DV_over_rs": 1.0071},
    "LRG1":      {"DH_over_rs": 1.0428, "DM_over_rs": 1.0020, "DV_over_rs": 1.0241},
    "LRG2":      {"DH_over_rs": 1.0063, "DM_over_rs": 1.0122, "DV_over_rs": 1.0243},
    "LRG3_ELG1": {"DH_over_rs": 1.0538, "DM_over_rs": 1.0306, "DV_over_rs": 1.0598},
    "ELG2":      {"DH_over_rs": 1.0000, "DM_over_rs": 1.0653, "DV_over_rs": 1.0623},
    "QSO":       {"DH_over_rs": 1.0056, "DM_over_rs": 1.0056, "DV_over_rs": 1.0056},
}
_SIGMA_KEYS = ("DH_over_rs", "DM_over_rs", "DV_over_rs")  # SIGMA_TARGET_NAMES order


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
def _triplet_from_recon_cov(path, fid):
    """σ(D/rd) triplet from a bao-recon .h5 (stat-only or syst), using OUR fid (D/rd).

    1×1 (qiso): σ scales DH/DM/DV identically. 2×2 (qpar,qper): DH←qpar, DM←qper,
    and DV from the full ln-cov combination (incl. off-diagonal).
    """
    nan = {k: float("nan") for k in _SIGMA_KEYS}
    if not Path(path).exists():
        return nan
    with h5py.File(path, "r") as h:
        cov = np.atleast_2d(np.asarray(h["covariance/value"][...]))
    DH = fid["DH_over_rd_fid"]; DM = fid["DM_over_rd_fid"]; DV = fid["DV_over_rd_fid"]
    if cov.shape == (1, 1):
        s = float(np.sqrt(cov[0, 0]))
        return {"DH_over_rs": s * DH, "DM_over_rs": s * DM, "DV_over_rs": s * DV}
    sp = float(np.sqrt(cov[0, 0])); sq = float(np.sqrt(cov[1, 1]))
    var_lnDV = (cov[0, 0] / 9.0) + (4.0 * cov[1, 1] / 9.0) + (4.0 * cov[0, 1] / 9.0)
    return {"DH_over_rs": sp * DH, "DM_over_rs": sq * DM,
            "DV_over_rs": float(np.sqrt(max(var_lnDV, 0.0))) * DV}


def _recon_sigmas(tracer, fid):
    """Convert bao-recon stat-only σ_q to σ(D/rd) using OUR fid (D/rd) values."""
    return _triplet_from_recon_cov(_LIK_DIR / _BAO_RECON_FILE[tracer], fid)


# ---------------------------------------------------------------------------
# DESI systematic-error layer (post-emulator)
# ---------------------------------------------------------------------------
def apply_desi_syst(sigma_triplet, tracer, ratios=None):
    """Inflate emulator σ_stat → σ_tot with DESI's measured per-tracer systematic
    budget (`DESI_SYST_INFLATION`): σ_tot[q] = R[tracer][q] · σ_stat[q].

    This is a fixed, fiducial-calibrated diagonal scaling applied AFTER the emulator;
    it does not vary with cosmology/N_tracers and does not touch the training data.
    The clean cosmology-varying quantity (σ_stat) is what the emulator predicts.

    `sigma_triplet` is either a dict keyed by {DH,DM,DV}_over_rs, or an array-like of
    shape (..., 3) in SIGMA_TARGET_NAMES order [DH, DM, DV]. Returns the same type.
    Unknown tracers / missing quantities scale by 1.0 (no-op).
    """
    R = (ratios or DESI_SYST_INFLATION).get(tracer, {})
    if isinstance(sigma_triplet, dict):
        return {k: sigma_triplet[k] * float(R.get(k, 1.0)) for k in sigma_triplet}
    factors = np.array([float(R.get(k, 1.0)) for k in _SIGMA_KEYS])
    return np.asarray(sigma_triplet, dtype=float) * factors


def compute_syst_inflation(tracer):
    """Recompute R[q] = σ(bao-recon_syst)/σ(bao-recon_stat-only) from the .h5 files
    (the (D/rd)_fid cancels in the ratio). Clamped to ≥1. Returns None if either
    file is missing."""
    unit = {"DH_over_rd_fid": 1.0, "DM_over_rd_fid": 1.0, "DV_over_rd_fid": 1.0}
    stat = _triplet_from_recon_cov(_LIK_DIR / _BAO_RECON_FILE[tracer], unit)
    syst = _triplet_from_recon_cov(_LIK_DIR / _BAO_RECON_SYST_FILE[tracer], unit)
    if any(np.isnan(stat[k]) or np.isnan(syst[k]) for k in _SIGMA_KEYS):
        return None
    R = {k: max(syst[k] / stat[k], 1.0) for k in _SIGMA_KEYS}
    if tracer in ("BGS", "QSO"):  # degenerate: single qiso factor on all three
        R = {k: R["DV_over_rs"] for k in _SIGMA_KEYS}
    return R


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DESI reference σ utilities")
    ap.add_argument("--regen-syst", action="store_true",
                    help="recompute DESI_SYST_INFLATION from the bao-recon .h5 files and "
                         "print it as a paste-ready dict (compare against the frozen table)")
    args = ap.parse_args()
    if args.regen_syst:
        print("DESI_SYST_INFLATION = {")
        for t in _TRACERS:
            R = compute_syst_inflation(t)
            if R is None:
                print(f'    # "{t}": missing _syst or stat-only .h5 — skipped')
                continue
            frozen = DESI_SYST_INFLATION.get(t, {})
            drift = max((abs(R[k] - frozen.get(k, R[k])) for k in _SIGMA_KEYS), default=0.0)
            flag = "" if drift < 5e-4 else f"   # DRIFT {drift:.4f} vs frozen"
            cells = ", ".join(f'"{k}": {R[k]:.4f}' for k in _SIGMA_KEYS)
            print(f'    "{t}": {{{cells}}},{flag}')
        print("}")
    else:
        ap.print_help()
