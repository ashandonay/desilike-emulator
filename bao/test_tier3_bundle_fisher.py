"""Tier-3 end-to-end methodology check: full bundle likelihood, our theory.

Substitutes ALL DESI bundle products (data, cov, W) into the Fisher and uses
ONLY our pipeline's theory for derivatives. If σ matches DESI's published
value, the F/D ≈ 0.5 gap is fully isolated to "how we build the cov + window
in our pipeline" vs "the cov + window DESI ships." If σ still falls short,
there's a Fisher methodology piece DESI does that we don't.

Fisher construction:

  Observable model:  pred(θ) = W · ξ_th(s; θ) + BB(s; α_ℓ)
  Data vector:       d  = bundle ξ_obs
  Cov:               C  = bundle cov_ξ (52×52)

  F_ij = (∂pred/∂θ_i)^T · C^{-1} · (∂pred/∂θ_j)

  Schur-marginalize over 15 nuisances {b1, dbeta, σ_s, σ_par, σ_per, 10×α_ℓ}
  → 2×2 marginal Fisher for (qpar, qper).
  σ(DH/rd) = sqrt(cov[0,0]) · DH_over_rd_fid
  σ(DM/rd) = sqrt(cov[1,1]) · DM_over_rd_fid

The 10 BB coefficients enter LINEARLY in pred, so their Jacobian columns are
just the BB basis (5 powers of s, per ℓ). The 7 nonlinear params {qpar, qper,
b1, dbeta, σ_s, σ_par, σ_per} need finite-difference derivatives — we rebuild
the wide-k observable and FFTLog each time.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402
from cosmoprimo.fftlog import PowerToCorrelation  # noqa: E402
from desilike.observables.galaxy_clustering import (  # noqa: E402
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.likelihoods import ObservablesGaussianLikelihood  # noqa: E402

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_BUNDLE = "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5"
_TRACER = "LRG2"
_DR1_AREA = 7500.0
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1"}


def _load_bundle():
    with h5py.File(_LIK_DIR / _BUNDLE, "r") as f:
        cov = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        obs_s = [np.asarray(f[f"observable/{e}/s"][...], dtype=np.float64) for e in ["0", "2"]]
        obs_data = [np.asarray(f[f"observable/{e}/value"][...], dtype=np.float64) for e in ["0", "2"]]
        th_s = np.asarray(f["window/theory/0/s"][...], dtype=np.float64)
    return {"cov": cov, "W": W, "obs_s": obs_s, "obs_data": obs_data, "th_s": th_s}


def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP.get(tracer, tracer)]


def _get_desi_std(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    sub = df[df["tracer"] == _NAME_MAP.get(tracer, tracer)]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


def _build_wide_lik():
    """Build the full pipeline + a wide-k observable wrapping the same theory chain.

    Returns (info, wide_lik, wide_obs, k_lin) so the caller can repeatedly call
    wide_lik(**params_with_overrides) and read wide_obs.theory.
    """
    n_tr = _get_ntracers(_TRACER)
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
    info = prep_covar.build_bao_likelihood(
        N_tracers=n_tr, theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=_TRACER, zrange=TRACER_CONFIGS[_TRACER]["zrange"],
        z_eff=None, area=_DR1_AREA,
    )
    info["likelihood"](**info["params"])
    # Locate the outer BAO theory calculator (DampedBAOWigglesTracerPowerSpectrumMultipoles)
    theory = None
    for c in info["observable"].runtime_info.pipeline.calculators:
        if type(c).__name__.startswith("DampedBAOWigglesTracerPowerSpectrumMultipoles"):
            theory = c
            break
    k_min, k_max, nb = 0.0014, 0.5, 600
    dk = (k_max - k_min) / nb
    new_obs = TracerPowerSpectrumMultipolesObservable(
        data=info["params"],
        klim={0: [k_min, k_max, dk], 2: [k_min, k_max, dk]},
        theory=theory,
    )
    wide_lik = ObservablesGaussianLikelihood(
        observables=[new_obs], covariance=np.eye(2 * nb),
    )
    wide_lik(**info["params"])
    return info, wide_lik, new_obs, np.asarray(new_obs.k[0], dtype=np.float64)


def _xi_at_bundle_grid(wide_obs, k_lin, bundle, ells=(0, 2)):
    """FFTLog wide-k P_ℓ(k) → ξ_ℓ(s), interpolate to bundle's theory s-grid.
    Returns concatenated 600-vector matching bundle's [ℓ=0, ℓ=2, ℓ=4] layout
    (ℓ=4 set to zero — see test_l4_leakage.py: contribution <1e-3 σ)."""
    P_lk = np.asarray(wide_obs.theory, dtype=np.float64)
    k_log = np.logspace(np.log10(k_lin[0]), np.log10(k_lin[-1]), 1024)
    P_log = np.empty((len(ells), len(k_log)), dtype=np.float64)
    for ie in range(len(ells)):
        P_log[ie] = np.interp(k_log, k_lin, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=list(ells))
    s_out, xi_out = fft(P_log, extrap="log")
    s_out = np.asarray(s_out); xi_out = np.asarray(xi_out)
    target_s = bundle["th_s"]
    xi_th_blocks = []
    for ell in [0, 2, 4]:
        if ell in ells:
            ie = list(ells).index(ell)
            s_e = s_out[ie] if s_out.ndim == 2 else s_out
            order = np.argsort(s_e)
            xi_th_blocks.append(np.interp(target_s, s_e[order], xi_out[ie][order]))
        else:
            xi_th_blocks.append(np.zeros_like(target_s))
    return np.concatenate(xi_th_blocks)


def _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, overrides):
    """Evaluate the pipeline at fiducial + overrides, return 52-bin observable prediction (W·ξ_th)."""
    full_params = {**info["params"], **overrides}
    # info["params"] only has the *bias/damping* params; the qpar/qper/dbeta/al*
    # parameters are in the likelihood's all_params but not in info["params"].
    # Provide them explicitly via overrides; defaults will be used otherwise.
    wide_lik(**full_params)
    xi_th_vec = _xi_at_bundle_grid(wide_obs, k_lin, bundle)
    return bundle["W"] @ xi_th_vec


def _bb_basis(obs_s, n_total):
    """The 10 BB polynomial columns: {s^-2, s^-1, s^0, s^1, s^2} per ℓ ∈ {0, 2}."""
    cols = []
    cur = 0
    for s in obs_s:
        n = len(s)
        for p in [-2, -1, 0, 1, 2]:
            col = np.zeros(n_total)
            col[cur:cur + n] = s ** p
            cols.append(col)
        cur += n
    return np.array(cols).T  # (n_total, 10)


def _jacobian(wide_lik, wide_obs, k_lin, bundle, info, nl_params, steps):
    """Two-sided finite-difference Jacobian of pred(θ) for the 7 nonlinear params.
    Returns J_nl with shape (n_data, 7)."""
    n_data = bundle["W"].shape[0]
    J = np.zeros((n_data, len(nl_params)), dtype=np.float64)
    fid = {p: float(_get_fid_param_value(p, info)) for p in nl_params}
    for i, p in enumerate(nl_params):
        dp = steps[p]
        plus  = _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, {p: fid[p] + dp})
        minus = _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, {p: fid[p] - dp})
        J[:, i] = (plus - minus) / (2.0 * dp)
        print(f"    ∂pred/∂{p}: ‖J_col‖_∞ = {np.abs(J[:, i]).max():.3e}", flush=True)
    return J


def _get_fid_param_value(name, info):
    """Look up fiducial value: first info["params"] (bias/damping), then likelihood.all_params."""
    if name in info["params"]:
        return float(info["params"][name])
    for p in info["likelihood"].all_params:
        if p.name == name:
            return float(p.value)
    raise KeyError(name)


def _schur_marginalize(F, idx_keep):
    """Return marginal Fisher on the kept indices (Schur complement)."""
    n = F.shape[0]
    idx_keep = list(idx_keep)
    idx_marg = [i for i in range(n) if i not in idx_keep]
    F_kk = F[np.ix_(idx_keep, idx_keep)]
    if len(idx_marg) == 0:
        return F_kk
    F_km = F[np.ix_(idx_keep, idx_marg)]
    F_mm = F[np.ix_(idx_marg, idx_marg)]
    return F_kk - F_km @ np.linalg.solve(F_mm, F_km.T)


def _compute_sigmas(J_nl, BB, C_inv, nl_used, template, desi, fix_names):
    """Build Fisher from the given nonlinear-Jacobian columns + BB, marginalize over
    everything except (qpar, qper), return σ(DH/rd), σ(DM/rd), σ(DV/rd)."""
    keep_cols = [i for i, p in enumerate(nl_used) if p not in fix_names]
    J_kept = J_nl[:, keep_cols]
    kept_names = [nl_used[i] for i in keep_cols]
    J_full = np.concatenate([J_kept, BB], axis=1)
    param_names = kept_names + [f"al0_{p}" for p in [-2,-1,0,1,2]] + [f"al2_{p}" for p in [-2,-1,0,1,2]]
    F = J_full.T @ C_inv @ J_full
    idx_qpar = param_names.index("qpar")
    idx_qper = param_names.index("qper")
    F_marg = _schur_marginalize(F, idx_keep=[idx_qpar, idx_qper])
    cov_qq = np.linalg.inv(F_marg)
    sig_qpar = float(np.sqrt(cov_qq[0, 0]))
    sig_qper = float(np.sqrt(cov_qq[1, 1]))
    DH_over_rd = float(template.DH_over_rd_fid)
    DM_over_rd = float(template.DM_over_rd_fid)
    sig_DH = sig_qpar * DH_over_rd
    sig_DM = sig_qper * DM_over_rd
    grad = np.array([1.0/3.0, 2.0/3.0])
    var_lnDV = float(grad @ cov_qq @ grad)
    DV_over_rd = (DM_over_rd ** 2 * DH_over_rd) ** (1.0 / 3.0)
    sig_DV = float(np.sqrt(max(var_lnDV, 0.0))) * DV_over_rd
    return {
        "n_marg": F.shape[0] - 2, "F_cond": float(np.linalg.cond(F)),
        "sig_qpar": sig_qpar, "sig_qper": sig_qper,
        "sig_DH": sig_DH, "sig_DM": sig_DM, "sig_DV": sig_DV,
        "desi_DH": desi.get("DH_over_rs", float("nan")),
        "desi_DM": desi.get("DM_over_rs", float("nan")),
        "kept": kept_names,
    }


def main():
    print("Building pipeline + wide-k observable...")
    info, wide_lik, wide_obs, k_lin = _build_wide_lik()
    bundle = _load_bundle()

    pred_fid = _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, {})
    data = np.concatenate(bundle["obs_data"])
    cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
    C_inv = np.linalg.inv(cov_sym)
    chi2_fid = float((data - pred_fid) @ C_inv @ (data - pred_fid))
    print(f"  χ²(fiducial, no BB) = {chi2_fid:.2f} / 52\n")

    nl_params = ["qpar", "qper", "b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]
    # Smaller σ-step to avoid the FoG-damping FD blow-up at large perturbations
    steps = {
        "qpar": 0.01, "qper": 0.01, "b1": 0.05, "dbeta": 0.05,
        "sigmas": 0.05, "sigmapar": 0.1, "sigmaper": 0.1,
    }
    print("Computing nonlinear-param Jacobian (7 params, 2-sided FD)...")
    J_nl = _jacobian(wide_lik, wide_obs, k_lin, bundle, info, nl_params, steps)
    BB = _bb_basis(bundle["obs_s"], data.size)
    template = info["template"]
    desi = _get_desi_std(_TRACER)

    # Configurations to sweep:
    configs = [
        ("all 7 nl + 10 BB marginalized",       []),                                       # baseline
        ("fix σ_s",                              ["sigmas"]),
        ("fix Σ_par, Σ_per",                     ["sigmapar", "sigmaper"]),
        ("fix Σ_par, Σ_per, σ_s",                ["sigmapar", "sigmaper", "sigmas"]),
        ("fix all damping + dbeta (b1 + BB marg)", ["sigmapar", "sigmaper", "sigmas", "dbeta"]),
        ("fix all damping + dbeta + b1 (BB only marg)", ["sigmapar", "sigmaper", "sigmas", "dbeta", "b1"]),
    ]

    print()
    print("=" * 102)
    print(f"Tier-3 Fisher with progressively tighter priors on nuisances")
    print("(equivalent to fixing them in the marginalization — the σ_prior → 0 limit)")
    print("=" * 102)
    print(f"{'configuration':<48}  {'σ(DH/rd)':>9} {'/DESI':>7}  {'σ(DM/rd)':>9} {'/DESI':>7}  {'n_marg':>7}")
    print("-" * 102)
    for label, fix in configs:
        r = _compute_sigmas(J_nl, BB, C_inv, nl_params, template, desi, fix_names=fix)
        ratio_DH = r["sig_DH"] / r["desi_DH"] if r["desi_DH"] > 0 else float("nan")
        ratio_DM = r["sig_DM"] / r["desi_DM"] if r["desi_DM"] > 0 else float("nan")
        print(f"{label:<48}  {r['sig_DH']:>9.4f} {ratio_DH:>7.3f}  {r['sig_DM']:>9.4f} {ratio_DM:>7.3f}  {r['n_marg']:>7}")
    print("-" * 102)
    print(f"{'DESI published':<48}  {desi.get('DH_over_rs', float('nan')):>9.4f} {1.0:>7.3f}  "
          f"{desi.get('DM_over_rs', float('nan')):>9.4f} {1.0:>7.3f}        —")


if __name__ == "__main__":
    main()
