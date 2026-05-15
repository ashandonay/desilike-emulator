"""Compute the textbook Gaussian disconnected cov_P for LRG2 and compare
the per-ell sigma ratios to desilike's ObservablesCovarianceMatrix.

Textbook (Grieb+16 eq. 16):
    Cov(P_l(k), P_l'(k)) = 2 (2l+1)(2l'+1) / N_modes(k)
                         * int_{-1}^{1} dmu/2  L_l(mu) L_l'(mu) [P_g(k,mu) + 1/nbar]^2

with N_modes(k) = V * k^2 dk / (2 pi^2)  and
     P_g(k,mu) = (b1 + f mu^2)^2 P_lin(k)   [pre-recon Kaiser]

If textbook ratio_2/0 ~ 1.0 while desilike gives 1.26, the gap is in
desilike's implementation. If textbook also gives 1.26, the formula
itself does this and the discrepancy with DR1 is real physics
(EZmocks include effects we don't).
"""
import sys
import numpy as np

sys.path.insert(0, "..")

import compare_to_dr1_rigorous as cmp_rig  # noqa: E402
import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402


def _legendre(ell, mu):
    if ell == 0: return np.ones_like(mu)
    if ell == 2: return 0.5 * (3 * mu ** 2 - 1)
    raise ValueError(ell)


def _textbook_cov_p(k_arr, dk, V_survey, b1, f, nbar, P_lin_k, ells=(0, 2)):
    """Build the textbook Gaussian disconnected cov_P (block in ell, diagonal in k)."""
    n_k = len(k_arr)
    n_ells = len(ells)
    cov = np.zeros((n_k * n_ells, n_k * n_ells), dtype=np.float64)
    mu = np.linspace(-1.0, 1.0, 2049)
    L = {l: _legendre(l, mu) for l in ells}
    # N_modes(k) = V * k^2 dk / (2 pi^2)
    N_modes = V_survey * k_arr ** 2 * dk / (2.0 * np.pi ** 2)

    # P_g(k,mu) on (n_k, n_mu) grid
    P_g = (b1 + f * mu[None, :] ** 2) ** 2 * P_lin_k[:, None]
    integrand = (P_g + 1.0 / nbar) ** 2  # (n_k, n_mu)

    for i, l1 in enumerate(ells):
        for j, l2 in enumerate(ells):
            # int_{-1}^1 dmu/2 L_l1 L_l2 [P+1/n]^2
            kernel = L[l1] * L[l2]
            mu_int = np.trapz(integrand * kernel[None, :], mu, axis=1) * 0.5
            # Grieb+16 eq. 16: factor of 2 in the prefactor (equivalent to
            # using the full int_{-1}^{1} dmu rather than dmu/2). Matches
            # desilike's get_sigma_k convention.
            var_ll = 2.0 * (2 * l1 + 1) * (2 * l2 + 1) / N_modes * mu_int
            # Place on diagonal of (l1,l2) block
            for ki in range(n_k):
                cov[i * n_k + ki, j * n_k + ki] = var_ll[ki]
    return cov


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", default="LRG2")
    args = ap.parse_args()
    tracer = args.tracer
    cfg = TRACER_CONFIGS[tracer]
    z_eff = cfg["z_eff"]
    b1 = float(cfg["bias_recon"])

    n_tracers = cmp_rig._get_dr1_ntracers(tracer)
    theta_cosmo, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53}
    )
    info = prep_covar.build_bao_likelihood(
        N_tracers=n_tracers, theta_cosmo=theta_cosmo, hrdrag=hrdrag_eff,
        tracer_bin=tracer, zrange=tuple(cfg["zrange"]), z_eff=z_eff,
        area=cmp_rig._DR1_AREA,
    )
    obs = info["observable"]
    k_arr = np.asarray(obs.k[0], dtype=np.float64)
    dk = float(np.mean(np.diff(k_arr)))

    template = info["template"]
    cosmo = template.cosmo
    fo = cosmo.get_fourier()
    pk_lin = fo.pk_interpolator().to_1d(z=z_eff)
    P_lin_k = np.asarray(pk_lin(k_arr), dtype=np.float64)
    f_growth = float(fo.sigma8_z(z_eff, of='theta_cb') / fo.sigma8_z(z_eff, of='delta_cb'))

    # Get V_survey and nbar from the observable's footprint
    # (look at desilike ObservablesCovarianceMatrix internals)
    # Easier: pull from prep_covar's tracer_settings flow.
    # We have N_tracers and area; volume needs cosmology + zrange.
    # base_footprint is built inside build_bao_likelihood; not exposed.
    # As a proxy, compute V from area + zrange using the cosmology.
    z_min, z_max = cfg["zrange"]
    # cosmoprimo's comoving_radial_distance returns chi in Mpc/h already.
    chi_min = float(cosmo.comoving_radial_distance(z_min))
    chi_max = float(cosmo.comoving_radial_distance(z_max))
    V_shell = (4.0 / 3.0) * np.pi * (chi_max ** 3 - chi_min ** 3) * \
              (cmp_rig._DR1_AREA / 41252.96)  # area in deg^2 -> sr fraction
    nbar_eff = n_tracers / V_shell
    print(f"Tracer: {tracer}, b1={b1}, f={f_growth:.3f}, z_eff={z_eff}")
    print(f"V_shell ~ {V_shell:.3e} (Mpc/h)^3,  nbar_eff ~ {nbar_eff:.3e} h^3/Mpc^3")

    # Textbook cov_P
    cov_p_text = _textbook_cov_p(k_arr, dk, V_shell, b1, f_growth, nbar_eff,
                                 P_lin_k, ells=(0, 2))
    print(f"Textbook cov_P shape: {cov_p_text.shape}")

    # desilike cov_P
    cov_p_desi = np.asarray(info["cov_components"]["C_gauss"], dtype=np.float64)

    # Print per-ell DIAGONAL ratio textbook / desilike (should be ~1 if our
    # formula matches their implementation up to nbar uncertainty)
    n_k = len(k_arr)
    for io, l in enumerate((0, 2)):
        sl = slice(io * n_k, (io + 1) * n_k)
        d_text = np.diag(cov_p_text)[sl]
        d_desi = np.diag(cov_p_desi)[sl]
        r = d_text / d_desi
        print(f"\nell={l} P-space diagonal: textbook/desilike "
              f"median={np.median(r):.3f}  16-84=[{np.percentile(r,16):.3f}, "
              f"{np.percentile(r,84):.3f}]")

    # Now project both through M and compare per-ell sigma to DR1
    cov_xi_dr1, W, obs_ells, obs_s_per_ell, theory_ells, theory_s_per_ell = \
        cmp_rig._load_dr1(tracer)
    pipe_ells = (0, 2)
    M, _ = cmp_rig._build_M(W, theory_ells, theory_s_per_ell,
                            list(pipe_ells), k_arr, dk)
    n_per = len(obs_s_per_ell[0])
    sd = np.sqrt(np.maximum(np.diag(cov_xi_dr1), 0.0))

    print(f"\n=== Per-ell sigma_pipe / sigma_DR1 in xi-space ===")
    for label, cp in [("desilike", cov_p_desi), ("textbook", cov_p_text)]:
        cov_xi = M @ cp @ M.T
        sp = np.sqrt(np.maximum(np.diag(cov_xi), 0.0))
        rs = []
        for io in range(len(obs_ells)):
            sl = slice(io * n_per, (io + 1) * n_per)
            r = float(np.median(sp[sl] / sd[sl]))
            rs.append(r)
        parts = [f"  {label:10s}"]
        for io, ell in enumerate(obs_ells):
            parts.append(f"  ell={ell}: {rs[io]:.3f}")
        if len(rs) >= 2:
            parts.append(f"  ratio 2/0: {rs[1]/rs[0]:.3f}")
        print("".join(parts))


if __name__ == "__main__":
    main()
