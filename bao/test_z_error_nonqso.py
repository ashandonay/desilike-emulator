"""Add literature-motivated spectroscopic z-errors to non-QSO tracers and
measure the σ(α) impact.

QSO already carries z_error_kms = 600 in tracers.yaml (broad-line catastrophic
z). For other tracers, the DESI spec pipeline (Lan+23 / Schlegel+25) reports:

  BGS:   σ_z ~ 30 km/s   (Hα + multiple absorption lines)
  LRG:   σ_z ~ 40 km/s   (Ca II triplet at moderate S/N)
  ELG:   σ_z ~ 20 km/s   ([OII] doublet, resolved)

σ_z enters σ_v² in quadrature with the satellite-virial dispersion. The σ_FoG
sensitivity diagnostic (computed above) showed 2.5–6.5% σ_FoG inflation per
tracer. §27's ALL-fix result caps the total nuisance-marginalization channel
at < 1% on σ(α), so the expected σ(α) effect here is < 0.5%.

Implementation: monkey-patch TRACER_CONFIGS at runtime to inject z_error_kms
during the Fisher call. No yaml edit unless the effect is meaningful.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

import prep_covar as pc
from util import TRACER_CONFIGS
from plot_fisher_vs_desi import _get_ntracers, _dv_from_dh_dm_cov


_Z_ERROR_LITERATURE_KMS = {
    "BGS":       30.0,
    "LRG1":      40.0,
    "LRG2":      40.0,
    "LRG3_ELG1": 30.0,   # bias-weighted blend (LRG3 ~40 + ELG1 ~20)
    "ELG2":      20.0,
    # QSO already 600 in yaml
}


def _sigmas(tracer, dataset, z_err_override=None,
            omega_m=0.3153, hrdrag=99.53):
    cfg = TRACER_CONFIGS[tracer]
    n_tracers = _get_ntracers(dataset, tracer)
    area = {"dr1": 7500.0, "dr2": 14000.0}[dataset]
    sample = {"N_tracers": n_tracers, "Om": omega_m, "hrdrag": hrdrag}

    # Inject z_error_kms via tracer_config override.
    if z_err_override is not None:
        cfg_override = {**cfg, "z_error_kms": float(z_err_override)}
        original = TRACER_CONFIGS[tracer]
        TRACER_CONFIGS[tracer] = cfg_override
        try:
            targets = pc.run_fisher(
                sample, tracer_bin=tracer,
                zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
                param_defaults=pc.PARAM_DEFAULTS, area=area,
            )
        finally:
            TRACER_CONFIGS[tracer] = original
    else:
        targets = pc.run_fisher(
            sample, tracer_bin=tracer,
            zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
            param_defaults=pc.PARAM_DEFAULTS, area=area,
        )

    var_dh = targets["cov_DH_over_rd_DH_over_rd"]
    var_dm = targets["cov_DM_over_rd_DM_over_rd"]
    cov_dh_dm = targets["cov_DH_over_rd_DM_over_rd"]
    theta, hrr = pc._to_bao_cosmo_params({**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
    rd = hrr / pc._H_FID
    template = pc.BAOPowerSpectrumTemplate(
        z=float(cfg["z_eff"]), fiducial=("DESI", dict(theta)), apmode="qparqper"
    )
    dh_mean = float(template.DH_fid) / rd
    dm_mean = float(template.DM_fid) / rd
    sig_dv = _dv_from_dh_dm_cov(var_dh, cov_dh_dm, var_dm,
                                 float(cfg["z_eff"]), dh_mean, dm_mean)
    return float(np.sqrt(var_dh)), float(np.sqrt(var_dm)), float(sig_dv)


def main():
    print(f"\nz-error sensitivity for non-QSO tracers (DR1).")
    print(f"Literature σ_z values from DESI spec pipeline (Lan+23/Schlegel+25).\n")
    print(f"{'Tracer':<12} {'z_err':>7} {'σ(DH) Δ':>12} {'σ(DM) Δ':>12} {'σ(DV) Δ':>12}")
    print("-" * 60)
    for tracer, z_err in _Z_ERROR_LITERATURE_KMS.items():
        sdh0, sdm0, sdv0 = _sigmas(tracer, "dr1", z_err_override=0.0)
        sdh1, sdm1, sdv1 = _sigmas(tracer, "dr1", z_err_override=z_err)
        dh = (sdh1 / sdh0 - 1.0) * 100.0
        dm = (sdm1 / sdm0 - 1.0) * 100.0
        dv = (sdv1 / sdv0 - 1.0) * 100.0
        print(f"{tracer:<12} {z_err:>5.0f}km {dh:>+11.3f}% {dm:>+11.3f}% {dv:>+11.3f}%")


if __name__ == "__main__":
    main()
