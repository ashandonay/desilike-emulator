"""Upper-bound test for the recon random-catalog noise contribution to σ(α).

The displacement field is reconstructed from a finite random catalog with
N_rand/N_data = 50 (`_N_RAND_OVER_N_DATA = 50` in prep_covar). The existing
_sigma_nl_post_from_recon uses 1/nbar for the shot-noise displacement
variance, treating the random catalog as effectively infinite. The correct
form uses (1 + 1/alpha)/nbar where alpha = N_rand/N_data.

The (1+1/50) = 1.02 correction is tiny — at most 1% on the shot term, which
is itself ~10% of Σ_post² for typical tracers. Net Σ_post change is < 0.5%,
σ(α) effect < 0.2%.

Rather than implement that exact +1% inflation, this test brackets the
displacement-field-noise channel by inflating Σ_post by ±10% (a generous
upper bound that covers grid resolution + finite-α + Wiener filter
truncation effects combined) and measures the σ(α) response. If even ±10%
on Σ_post produces < 1% σ change, the entire channel is irrelevant.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

import prep_covar as pc
from util import TRACER_CONFIGS
from plot_fisher_vs_desi import _get_ntracers, _dv_from_dh_dm_cov


def _sigmas(tracer, dataset, sigma_post_scale=1.0,
            omega_m=0.3153, hrdrag=99.53):
    cfg = TRACER_CONFIGS[tracer]
    n_tracers = _get_ntracers(dataset, tracer)
    area = {"dr1": 7500.0, "dr2": 14000.0}[dataset]

    # Use compute_pipeline_sigmas to get the baseline Σ_⊥, Σ_∥, then scale.
    sample = {"N_tracers": n_tracers, "Om": omega_m, "hrdrag": hrdrag}
    sper, spar = pc.compute_pipeline_sigmas(
        sample, tracer_bin=tracer,
        zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
        param_defaults=pc.PARAM_DEFAULTS, area=area,
    )
    override = (sper * sigma_post_scale, spar * sigma_post_scale)

    theta_cosmo, hr = pc._to_bao_cosmo_params(
        {**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag}
    )
    targets = pc.get_bao_fisher_covariance(
        N_tracers=n_tracers, theta_cosmo=theta_cosmo, hrdrag=hr,
        tracer_bin=tracer, zrange=tuple(cfg["zrange"]),
        z_eff=float(cfg["z_eff"]), area=area,
        override_sigmas=override,
    )

    var_dh = targets["cov_DH_over_rd_DH_over_rd"]
    var_dm = targets["cov_DM_over_rd_DM_over_rd"]
    cov_dh_dm = targets["cov_DH_over_rd_DM_over_rd"]
    template = pc.BAOPowerSpectrumTemplate(
        z=float(cfg["z_eff"]), fiducial=("DESI", dict(theta_cosmo)), apmode="qparqper"
    )
    rd = hr / pc._H_FID
    dh_mean = float(template.DH_fid) / rd
    dm_mean = float(template.DM_fid) / rd
    sig_dv = _dv_from_dh_dm_cov(var_dh, cov_dh_dm, var_dm,
                                 float(cfg["z_eff"]), dh_mean, dm_mean)
    return float(np.sqrt(var_dh)), float(np.sqrt(var_dm)), float(sig_dv), sper, spar


def main():
    print(f"\nΣ_post ±10% bracket — upper bound on recon random-catalog noise impact on σ(α).\n")
    print(f"{'Tracer':<12} {'Σ_⊥ fid':>9} {'Σ_∥ fid':>9}   "
          f"{'σ(DH) Δ':>12} {'σ(DM) Δ':>12} {'σ(DV) Δ':>12}  (Σ_post × 1.10 vs fid)")
    print("-" * 90)
    for tracer in ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]:
        try:
            sdh0, sdm0, sdv0, sper, spar = _sigmas(tracer, "dr1", 1.0)
            sdh1, sdm1, sdv1, _, _ = _sigmas(tracer, "dr1", 1.10)
        except Exception as e:
            print(f"  {tracer:<12} FAILED: {e}")
            continue
        dh = (sdh1 / sdh0 - 1.0) * 100.0
        dm = (sdm1 / sdm0 - 1.0) * 100.0
        dv = (sdv1 / sdv0 - 1.0) * 100.0
        print(f"{tracer:<12} {sper:>9.3f} {spar:>9.3f}   "
              f"{dh:>+11.3f}% {dm:>+11.3f}% {dv:>+11.3f}%")


if __name__ == "__main__":
    main()
