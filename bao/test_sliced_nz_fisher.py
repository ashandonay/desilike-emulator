"""Sliced-nz Fisher vs single-z_eff Fisher for ELG2 (and other tracers).

The bao/ pipeline default evaluates b1, Σ_post, nbar, σ_v at a single z_eff
(Fisher-info-weighted from n(z) slices). For wide redshift bins (ELG2 covers
z=1.1-1.6, with D(z) varying ~15% across the bin), this could be biasing σ.

The sliced-nz path (get_bao_fisher_covariance_sliced_nz) summed slice-wise
Fisher information instead, fully accounting for within-bin z-evolution.
This test runs both for each tracer and reports σ shifts.
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

import prep_covar as pc
from util import TRACER_CONFIGS
from plot_fisher_vs_desi import _get_ntracers, _dv_from_dh_dm_cov


_NZ_DIR = Path.home() / "data" / "desi" / "nz_slices"


def _sigmas_single(tracer, dataset, omega_m=0.3153, hrdrag=99.53):
    cfg = TRACER_CONFIGS[tracer]
    sample = {"N_tracers": _get_ntracers(dataset, tracer),
              "Om": omega_m, "hrdrag": hrdrag}
    area = {"dr1": 7500.0, "dr2": 14000.0}[dataset]
    return pc.run_fisher(
        sample, tracer_bin=tracer,
        zrange=tuple(cfg["zrange"]), z_eff=None,
        param_defaults=pc.PARAM_DEFAULTS, area=area,
    )


def _sigmas_sliced(tracer, dataset, omega_m=0.3153, hrdrag=99.53):
    cfg = TRACER_CONFIGS[tracer]
    nz_path = _NZ_DIR / f"{tracer}_nz_slices.csv"
    if not nz_path.exists():
        raise FileNotFoundError(nz_path)
    sample = {"N_tracers": _get_ntracers(dataset, tracer),
              "Om": omega_m, "hrdrag": hrdrag}
    area = {"dr1": 7500.0, "dr2": 14000.0}[dataset]
    return pc.run_fisher(
        sample, tracer_bin=tracer,
        zrange=tuple(cfg["zrange"]), z_eff=None,
        param_defaults=pc.PARAM_DEFAULTS, area=area,
        nz_slices_path=str(nz_path),
    )


def _to_sigmas(targets, tracer, omega_m, hrdrag):
    cfg = TRACER_CONFIGS[tracer]
    var_dh = targets["cov_DH_over_rd_DH_over_rd"]
    var_dm = targets["cov_DM_over_rd_DM_over_rd"]
    cov_dh_dm = targets["cov_DH_over_rd_DM_over_rd"]
    theta, hr = pc._to_bao_cosmo_params({**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
    rd = hr / pc._H_FID
    template = pc.BAOPowerSpectrumTemplate(
        z=float(cfg["z_eff"]), fiducial=("DESI", dict(theta)), apmode="qparqper"
    )
    dh = float(template.DH_fid) / rd
    dm = float(template.DM_fid) / rd
    sig_dv = _dv_from_dh_dm_cov(var_dh, cov_dh_dm, var_dm, float(cfg["z_eff"]), dh, dm)
    return float(np.sqrt(var_dh)), float(np.sqrt(var_dm)), float(sig_dv)


def main():
    print("\nSliced-nz vs single-z_eff Fisher comparison (DR1 + DR2).")
    print("Tests whether within-bin z-evolution is biasing σ for wide-z tracers.\n")
    print(f"{'Tracer':<12} {'DS':<4} {'σ(DH) Δ':>10} {'σ(DM) Δ':>10} {'σ(DV) Δ':>10}")
    print("-" * 54)
    import sys
    tracers = sys.argv[1:] if len(sys.argv) > 1 else ["ELG2"]
    for dataset in ["dr1", "dr2"]:
        for tracer in tracers:
            try:
                t_single = _sigmas_single(tracer, dataset)
                t_sliced = _sigmas_sliced(tracer, dataset)
            except Exception as e:
                print(f"  {tracer:<12} {dataset:<4} FAILED: {e}")
                continue
            s0 = _to_sigmas(t_single, tracer, 0.3153, 99.53)
            s1 = _to_sigmas(t_sliced, tracer, 0.3153, 99.53)
            dh = (s1[0]/s0[0] - 1) * 100.0
            dm = (s1[1]/s0[1] - 1) * 100.0
            dv = (s1[2]/s0[2] - 1) * 100.0
            print(f"{tracer:<12} {dataset:<4} {dh:>+9.2f}% {dm:>+9.2f}% {dv:>+9.2f}%")
        print()


if __name__ == "__main__":
    main()
