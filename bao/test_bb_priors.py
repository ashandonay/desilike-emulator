"""Test how QSO and LRG3+ELG1 sigma(DV/rd), sigma(DH/rd), sigma(DM/rd)
respond to fixing BB nuisance amplitudes (al0_-3 ... al2_1).

If BB-marginalization is dominant for QSO, fixing BB should drop sigma(QSO)
toward DESI's published value. For LRG2 (strong BAO peak) fixing BB should
have small effect.

This is an extreme test (fully fixed). If it shows BB is the dominant lever,
the next step is a physics-motivated Gaussian prior of finite width.
"""
import sys
import numpy as np

sys.path.insert(0, "..")

import prep_covar as pc  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402
from plot_fisher_vs_desi import _get_ntracers, _get_desi_std, _dv_from_dh_dm_cov  # noqa: E402


_BB_NAMES = [f"al0_{n}" for n in [-3, -2, -1, 0, 1]] + \
            [f"al2_{n}" for n in [-3, -2, -1, 0, 1]]
# Everything that's nuisance — fixing all of these gives the unmarginalized
# Fisher info on qpar/qper, i.e. the absolute tightest sigma the model can
# produce given its covariance.
_ALL_NUISANCE = _BB_NAMES + ["b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]


def _fisher(tracer, dataset, fix_bb=False, fix_all_nuisance=False,
            omega_m=0.3152, hrdrag=99.08):
    cfg = TRACER_CONFIGS[tracer]
    n_tracers = _get_ntracers(dataset, tracer)
    area = {"dr1": 7500.0, "dr2": 14000.0}[dataset]
    sample = {"N_tracers": n_tracers, "Om": omega_m, "hrdrag": hrdrag}

    if not (fix_bb or fix_all_nuisance):
        targets = pc.run_fisher(
            sample, tracer_bin=tracer,
            zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
            param_defaults=pc.PARAM_DEFAULTS, area=area,
        )
    else:
        # Build likelihood, then fix the BB params before computing Fisher.
        theta_cosmo, hr = pc._to_bao_cosmo_params(
            {**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag}
        )
        info = pc.build_bao_likelihood(
            N_tracers=n_tracers, theta_cosmo=theta_cosmo, hrdrag=hr,
            tracer_bin=tracer,
            zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
            area=area,
        )
        lik = info["likelihood"]
        names = _ALL_NUISANCE if fix_all_nuisance else _BB_NAMES
        for name in names:
            try:
                lik.all_params[name].update(fixed=True)
            except KeyError:
                pass
        F_q = pc._q_fisher_from_bao_likelihood_info(info)
        cov_q = np.linalg.inv(F_q)
        rd = info["rd"]
        template = info["template"]
        DH_fid = float(template.DH_fid) / rd
        DM_fid = float(template.DM_fid) / rd
        var_dh = cov_q[0, 0] * DH_fid ** 2
        var_dm = cov_q[1, 1] * DM_fid ** 2
        cov_dh_dm = cov_q[0, 1] * DH_fid * DM_fid
        targets = {
            "cov_DH_over_rd_DH_over_rd": var_dh,
            "cov_DM_over_rd_DM_over_rd": var_dm,
            "cov_DH_over_rd_DM_over_rd": cov_dh_dm,
        }

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
    print(f"\n{'tracer':<12} {'dataset':<5} {'BB':<8} "
          f"{'sig(DH)':>9} {'F/D':>6}   {'sig(DM)':>9} {'F/D':>6}   "
          f"{'sig(DV)':>9} {'F/D':>6}")
    print("-" * 100)
    for tracer in ["QSO", "LRG2", "LRG3_ELG1", "LRG1"]:
        for dataset in ["dr1", "dr2"]:
            desi = _get_desi_std(dataset, tracer)
            dh_d = desi.get("DH_over_rs", np.nan)
            dm_d = desi.get("DM_over_rs", np.nan)
            dv_d = desi.get("DV_over_rs", np.nan)
            for label, fix_bb, fix_all in [("floated", False, False),
                                            ("BB-fix", True, False),
                                            ("ALL-fix", False, True)]:
                try:
                    sdh, sdm, sdv = _fisher(tracer, dataset,
                                            fix_bb=fix_bb, fix_all_nuisance=fix_all)
                except Exception as e:
                    print(f"  {tracer:<12} {dataset:<5} {label:<8} FAILED: {e}")
                    continue
                rdh = sdh / dh_d if np.isfinite(dh_d) else np.nan
                rdm = sdm / dm_d if np.isfinite(dm_d) else np.nan
                rdv = sdv / dv_d if np.isfinite(dv_d) else np.nan
                print(f"  {tracer:<12} {dataset:<5} {label:<8} "
                      f"{sdh:>9.4f} {rdh:>6.3f}   "
                      f"{sdm:>9.4f} {rdm:>6.3f}   "
                      f"{sdv:>9.4f} {rdv:>6.3f}")


if __name__ == "__main__":
    main()
