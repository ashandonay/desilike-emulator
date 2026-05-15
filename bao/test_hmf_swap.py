"""Quick test: swap Tinker08 HMF for Despali+16 universal-virial HMF
and measure ELG2 σ shift.

Tinker+08 parameters (current, mean Δ=200):
    A0=0.186, a0=1.47, b0=2.57, c=1.19  with mild z-evolution.

Despali+16 universal virial (Despali et al. 2016, MNRAS 456, 2486 eq. 9):
    f(ν) = A (1 + 1/(aν)^p) (2aν/π)^(1/2) exp(-aν/2)
    (a, p, A) = (0.794, 0.247, 0.333)

Both are calibrated against N-body simulations at the few-percent level. The
difference between them in f_sigma for the relevant ν range (~1-3 for our
tracers) is typically <10%, which translates to small shifts in HOD b1 and
hence σ(α).

Implementation: monkey-patch prep_covar._tinker08_f_sigma to use Despali+16
form. Note the variable change: Tinker uses sigma directly, Despali uses ν=δc/σ.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

import prep_covar as pc
from util import TRACER_CONFIGS
from plot_fisher_vs_desi import _get_ntracers, _dv_from_dh_dm_cov


# Despali+16 universal virial params (z=0 best fit, Eq. 9)
_DESPALI_A = 0.333
_DESPALI_a = 0.794
_DESPALI_p = 0.247
_DELTA_C = 1.686


def _despali16_f_sigma(sigma, z, delta=200.0):
    """Despali+16 universal-virial f(σ).

    Despali show this is universal across z and cosmology to a few percent
    when ν is computed with δ_c(z). Their z-evolution is built into the
    ν(z, σ) variable, not into the (a, p, A) parameters.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    nu = _DELTA_C / np.maximum(sigma, 1.0e-30)
    f_nu = _DESPALI_A * (1.0 + (_DESPALI_a * nu * nu) ** (-_DESPALI_p)) \
           * np.sqrt(2.0 * _DESPALI_a * nu * nu / np.pi) \
           * np.exp(-_DESPALI_a * nu * nu / 2.0)
    # f(σ) = f(ν) × |dν/dlnσ| / ν, and dν/dlnσ = -ν → |..|/ν = 1
    # So f(σ) = f(ν) when both are dn/dlnM normalizations agree.
    # Tinker's form returns A*((sigma/b)^(-a)+1)*exp(-c/sigma^2) which is f(σ);
    # the calling code multiplies by (rho_m/M) * dln(1/σ)/dlnM to get dn/dlnM.
    # Despali's f(ν) above is the standard ν f(ν) form, which equals σ f(σ)
    # when the calling code uses dν/dlnσ = -ν. Convert to f(σ) by dividing by ν.
    return f_nu / np.maximum(nu, 1.0e-30)


def _sigmas(tracer, dataset, use_despali=False,
            omega_m=0.3153, hrdrag=99.53):
    cfg = TRACER_CONFIGS[tracer]
    sample = {"N_tracers": _get_ntracers(dataset, tracer),
              "Om": omega_m, "hrdrag": hrdrag}
    area = {"dr1": 7500.0, "dr2": 14000.0}[dataset]

    if use_despali:
        orig = pc._tinker08_f_sigma
        pc._tinker08_f_sigma = _despali16_f_sigma
    try:
        t = pc.run_fisher(
            sample, tracer_bin=tracer,
            zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
            param_defaults=pc.PARAM_DEFAULTS, area=area,
        )
    finally:
        if use_despali:
            pc._tinker08_f_sigma = orig

    var_dh = t["cov_DH_over_rd_DH_over_rd"]
    var_dm = t["cov_DM_over_rd_DM_over_rd"]
    cov_dh_dm = t["cov_DH_over_rd_DM_over_rd"]
    theta, hr = pc._to_bao_cosmo_params({**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
    rd = hr / pc._H_FID
    tmpl = pc.BAOPowerSpectrumTemplate(
        z=float(cfg["z_eff"]), fiducial=("DESI", dict(theta)), apmode="qparqper"
    )
    dh = float(tmpl.DH_fid) / rd
    dm = float(tmpl.DM_fid) / rd
    sig_dv = _dv_from_dh_dm_cov(var_dh, cov_dh_dm, var_dm, float(cfg["z_eff"]), dh, dm)
    return float(np.sqrt(var_dh)), float(np.sqrt(var_dm)), float(sig_dv)


def main():
    print("\nHMF swap test: Tinker+08 (current) vs Despali+16 universal virial.")
    print("Same f(σ) form structure but updated calibration parameters.\n")
    print(f"{'Tracer':<12} {'DS':<4} {'σ(DH) Δ':>10} {'σ(DM) Δ':>10} {'σ(DV) Δ':>10}")
    print("-" * 54)
    for dataset in ["dr1", "dr2"]:
        for tracer in ["LRG2", "LRG3_ELG1", "ELG2", "QSO"]:
            try:
                s0 = _sigmas(tracer, dataset, use_despali=False)
                s1 = _sigmas(tracer, dataset, use_despali=True)
            except Exception as e:
                print(f"  {tracer:<12} {dataset:<4} FAILED: {e}")
                continue
            dh = (s1[0]/s0[0] - 1) * 100.0
            dm = (s1[1]/s0[1] - 1) * 100.0
            dv = (s1[2]/s0[2] - 1) * 100.0
            print(f"{tracer:<12} {dataset:<4} {dh:>+9.2f}% {dm:>+9.2f}% {dv:>+9.2f}%")
        print()


if __name__ == "__main__":
    main()
