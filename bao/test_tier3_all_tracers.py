"""Tier-3 Fisher (FFTLog projection) for all six DR1 tracers.

Generalization of test_tier3_bundle_fisher.py (§31g LRG2-only) to the full
six-tracer set, now that all six bundle files are local.

The Tier-3 Fisher is the right test of the "leading theory" (cov is the
missing piece): it uses DESI's bundle data + cov + W, with pipeline theory
projected to ξ-space via FFTLog (avoiding the Riemann k-truncation bug from
§31c and the F2 precision-override numerical issues from earlier attempts).

For each tracer:
  1. Build the pipeline at fiducial cosmology
  2. Build a wide-k observable (klim=[0.0014, 0.5]) using the same theory chain
  3. FD-Jacobian: ∂(W · FFTLog(P_pipe))/∂θ for {qpar, qper, b1, dbeta, σ_s,
     Σ_par, Σ_per} + 10 (or 5 for DV-only) BB poly columns analytically
  4. Fisher = J^T · C_bundle^{-1} · J → cov(qpar, qper) via Schur over 15
     (or 10) nuisances
  5. σ(DH/rd), σ(DM/rd), σ(DV/rd) → ratio to DESI published

If the Tier-3 Fisher ratios cluster around 1.0 across all six tracers, the
leading theory ("cov is the missing piece") is supported and the §31g
sparse-tracer overshoots were precision-substitution numerical artifacts.
If sparse-tracer Tier-3 still overshoots, the leading theory is not
sufficient on its own.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
_DR1_AREA = 7500.0
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1"}

_BUNDLES = {
    "BGS":       "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}


def _load_bundle(tracer):
    fpath = _LIK_DIR / _BUNDLES[tracer]
    with h5py.File(fpath, "r") as f:
        cov = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        obs_ells, obs_s, obs_data = [], [], []
        for k in sorted(f["observable"].keys()):
            if k.isdigit():
                obs_ells.append(int(f[f"observable/{k}/meta/ell"][()]))
                obs_s.append(np.asarray(f[f"observable/{k}/s"][...], dtype=np.float64))
                obs_data.append(np.asarray(f[f"observable/{k}/value"][...], dtype=np.float64))
        th_ells, th_s = [], []
        for k in sorted(f["window/theory"].keys()):
            if k.isdigit():
                th_ells.append(int(f[f"window/theory/{k}/meta/ell"][()]))
                th_s.append(np.asarray(f[f"window/theory/{k}/s"][...], dtype=np.float64))
    return {"cov": cov, "W": W,
            "obs_ells": obs_ells, "obs_s": obs_s, "obs_data": obs_data,
            "th_ells": th_ells, "th_s": th_s}


def _get_ntracers(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    return {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}[_NAME_MAP.get(tracer, tracer)]


def _get_desi_std(tracer):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    sub = df[df["tracer"] == _NAME_MAP.get(tracer, tracer)]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


def _build_wide_lik(tracer):
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53})
    info = prep_covar.build_bao_likelihood(
        N_tracers=_get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag_eff,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_DR1_AREA,
    )
    info["likelihood"](**info["params"])
    theory = None
    for c in info["observable"].runtime_info.pipeline.calculators:
        if type(c).__name__.startswith("DampedBAOWigglesTracerPowerSpectrumMultipoles"):
            theory = c; break
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
    P_lk = np.asarray(wide_obs.theory, dtype=np.float64)
    k_log = np.logspace(np.log10(k_lin[0]), np.log10(k_lin[-1]), 1024)
    P_log = np.empty((len(ells), len(k_log)), dtype=np.float64)
    for ie in range(len(ells)):
        P_log[ie] = np.interp(k_log, k_lin, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=list(ells))
    s_out, xi_out = fft(P_log, extrap="log")
    s_out = np.asarray(s_out); xi_out = np.asarray(xi_out)
    target_s = bundle["th_s"][0]
    xi_th_blocks = []
    for ell in [0, 2, 4]:
        if ell in ells:
            ie = list(ells).index(ell)
            s_e = s_out[ie] if s_out.ndim == 2 else s_out
            order = np.argsort(s_e)
            xi_th_blocks.append(np.interp(target_s, s_e[order], xi_out[ie][order]))
        else:
            xi_th_blocks.append(np.zeros_like(target_s))
    # Order in bundle's th_ells layout
    by_ell = {0: xi_th_blocks[0], 2: xi_th_blocks[1], 4: xi_th_blocks[2]}
    return np.concatenate([by_ell[e] for e in bundle["th_ells"]])


def _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, overrides):
    full_params = {**info["params"], **overrides}
    wide_lik(**full_params)
    xi_th_vec = _xi_at_bundle_grid(wide_obs, k_lin, bundle)
    return bundle["W"] @ xi_th_vec


def _bb_basis(obs_ells, obs_s, n_total, powers=(0, 2)):
    """DESI Adame+24 ξ-space BB basis: {s^0, s^2} per ℓ by default
    (b_{ℓ,0} + b_{ℓ,2}(s·k_min/2π)²; see Adame+24 §4.3.2, eq 4.11-4.12).
    The legacy 5-power polynomial '-3,-2,-1,0,1' is also supported."""
    cols = []
    cur = 0
    for _, s in zip(obs_ells, obs_s):
        n = len(s)
        for p in powers:
            col = np.zeros(n_total)
            col[cur:cur + n] = s ** p
            cols.append(col)
        cur += n
    return np.array(cols).T


def _get_fid_param_value(name, info):
    if name in info["params"]:
        return float(info["params"][name])
    for p in info["likelihood"].all_params:
        if p.name == name:
            return float(p.value)
    raise KeyError(name)


def _schur_marginalize(F, idx_keep):
    n = F.shape[0]
    idx_keep = list(idx_keep)
    idx_marg = [i for i in range(n) if i not in idx_keep]
    F_kk = F[np.ix_(idx_keep, idx_keep)]
    if len(idx_marg) == 0:
        return F_kk
    F_km = F[np.ix_(idx_keep, idx_marg)]
    F_mm = F[np.ix_(idx_marg, idx_marg)]
    return F_kk - F_km @ np.linalg.solve(F_mm, F_km.T)


def run_tracer(tracer, bb_powers=(0, 2), sigma_override=None):
    print(f"\n{'=' * 70}\n  {tracer}  (BB powers={list(bb_powers)})\n{'=' * 70}", flush=True)
    info, wide_lik, wide_obs, k_lin = _build_wide_lik(tracer)
    bundle = _load_bundle(tracer)
    sig_par_pipe = info["params"].get("sigmapar")
    sig_per_pipe = info["params"].get("sigmaper")
    print(f"  pipeline fid: Σ_par={sig_par_pipe:.3f}, Σ_per={sig_per_pipe:.3f}", flush=True)
    if sigma_override is not None:
        for k, v in sigma_override.items():
            info["params"][k] = float(v)
        print(f"  override applied: Σ_par={info['params']['sigmapar']:.3f}, "
              f"Σ_per={info['params']['sigmaper']:.3f}", flush=True)

    pred_fid = _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, {})
    data = np.concatenate(bundle["obs_data"])
    n_data = data.size
    cov_sym = 0.5 * (bundle["cov"] + bundle["cov"].T)
    C_inv = np.linalg.inv(cov_sym)
    chi2_fid = float((data - pred_fid) @ C_inv @ (data - pred_fid))
    print(f"  χ²(fiducial, no BB) = {chi2_fid:.2f} / {n_data}", flush=True)

    nl_params = ["qpar", "qper", "b1", "dbeta", "sigmas", "sigmapar", "sigmaper"]
    steps = {"qpar": 0.01, "qper": 0.01, "b1": 0.05, "dbeta": 0.05,
             "sigmas": 0.05, "sigmapar": 0.1, "sigmaper": 0.1}

    print("  Jacobian:", flush=True)
    J_nl = np.zeros((n_data, len(nl_params)), dtype=np.float64)
    for i, p in enumerate(nl_params):
        fid = _get_fid_param_value(p, info)
        dp = steps[p]
        plus  = _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, {p: fid + dp})
        minus = _pred_at_params(wide_lik, wide_obs, k_lin, bundle, info, {p: fid - dp})
        J_nl[:, i] = (plus - minus) / (2.0 * dp)
        print(f"    ∂pred/∂{p}: ‖J‖_∞ = {np.abs(J_nl[:, i]).max():.3e}", flush=True)

    BB = _bb_basis(bundle["obs_ells"], bundle["obs_s"], n_data, powers=bb_powers)
    J_full = np.concatenate([J_nl, BB], axis=1)
    param_names = nl_params + [f"al{e}_{p}" for e in bundle["obs_ells"] for p in bb_powers]
    F = J_full.T @ C_inv @ J_full
    idx_qpar = param_names.index("qpar")
    idx_qper = param_names.index("qper")
    F_marg = _schur_marginalize(F, idx_keep=[idx_qpar, idx_qper])
    cov_qq = np.linalg.inv(F_marg)
    sig_qpar = float(np.sqrt(cov_qq[0, 0]))
    sig_qper = float(np.sqrt(cov_qq[1, 1]))
    template = info["template"]
    z_eff_local = float(TRACER_CONFIGS[tracer]["z_eff"])
    DH_over_rd = float(template.DH_over_rd_fid)
    DM_over_rd = float(template.DM_over_rd_fid)
    DV_over_rd = (z_eff_local * DM_over_rd ** 2 * DH_over_rd) ** (1.0 / 3.0)  # BAO definition includes z
    sig_DH = sig_qpar * DH_over_rd
    sig_DM = sig_qper * DM_over_rd
    grad = np.array([1.0 / 3.0, 2.0 / 3.0])
    var_lnDV = float(grad @ cov_qq @ grad)
    sig_DV = float(np.sqrt(max(var_lnDV, 0.0))) * DV_over_rd
    desi = _get_desi_std(tracer)
    return {
        "tracer": tracer,
        "sig_DH": sig_DH, "sig_DM": sig_DM, "sig_DV": sig_DV,
        "desi_DH": desi.get("DH_over_rs", float("nan")),
        "desi_DM": desi.get("DM_over_rs", float("nan")),
        "desi_DV": desi.get("DV_over_rs", float("nan")),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracers", default="BGS,LRG1,LRG2,LRG3_ELG1,ELG2,QSO",
                    help="comma-separated list of tracers to run")
    ap.add_argument("--bb", default="0,2",
                    help="comma-separated BB powers per ℓ; default '0,2' = DESI Adame+24 "
                         "ξ-space basis (b_{ℓ,0}+b_{ℓ,2}s²; §4.3.2). Legacy 5-power: '-3,-2,-1,0,1'")
    ap.add_argument("--sigma-par", type=float, default=None,
                    help="override Σ_par fiducial (applies to all tracers in run)")
    ap.add_argument("--sigma-per", type=float, default=None,
                    help="override Σ_per fiducial")
    ap.add_argument("--sigma-s", type=float, default=None,
                    help="override σ_s (FoG) fiducial")
    args = ap.parse_args()
    tracers = [t.strip() for t in args.tracers.split(",") if t.strip()]
    bb_powers = tuple(int(p.strip()) for p in args.bb.split(",") if p.strip())
    sigma_override = {}
    if args.sigma_par is not None:
        sigma_override["sigmapar"] = args.sigma_par
    if args.sigma_per is not None:
        sigma_override["sigmaper"] = args.sigma_per
    if args.sigma_s is not None:
        sigma_override["sigmas"] = args.sigma_s
    sigma_override = sigma_override or None

    rows = []
    for tracer in tracers:
        try:
            rows.append(run_tracer(tracer, bb_powers=bb_powers,
                                   sigma_override=sigma_override))
        except Exception as e:
            print(f"  [{tracer}] failed: {type(e).__name__}: {e}")

    print()
    print("=" * 102)
    print(f"Tier-3 Fisher (FFTLog) — bundle data+cov+W with pipeline theory; BB powers={list(bb_powers)}")
    print("=" * 102)
    print(f"{'Tracer':<11} {'σ(DH/rd)':>9} {'DESI':>9} {'F/D':>6}   "
          f"{'σ(DM/rd)':>9} {'DESI':>9} {'F/D':>6}   {'σ(DV/rd)':>9} {'DESI':>9} {'F/D':>6}")
    print("-" * 102)
    for r in rows:
        def _fmt(s, d):
            if not np.isfinite(d) or d <= 0:
                return f"{s:>9.4f} {'—':>9} {'—':>6}"
            return f"{s:>9.4f} {d:>9.4f} {s/d:>6.3f}"
        print(f"{r['tracer']:<11} "
              f"{_fmt(r['sig_DH'], r['desi_DH'])}   "
              f"{_fmt(r['sig_DM'], r['desi_DM'])}   "
              f"{_fmt(r['sig_DV'], r['desi_DV'])}")


if __name__ == "__main__":
    main()
