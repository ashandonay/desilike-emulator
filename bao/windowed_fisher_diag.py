"""Windowed-Fisher diagnostic.

Compute the BAO Fisher σ(α) for the worst tracers using the DR1 survey window
matrix W applied to the pipeline P(k) -> ξ_obs(s) basis, and compare to the
current (unwindowed P-space) Fisher.

Math
----
Unwindowed Fisher (current pipeline):
    F_unwin = J^T C_P^{-1} J
where J = ∂P_flat/∂θ and C_P is the pipeline's analytic Gaussian + SSC +
recon-shot covariance in flat-P space.

Windowed Fisher (this diagnostic):
    F_win = J^T M^T (M C_P M^T)^{-1} M J
where M = W @ H_block projects pipeline P(k) -> ξ_obs(s).

For square / full-rank M, F_win = F_unwin (Fisher is invariant under
invertible coordinate transformations). DESI's W is rectangular (n_obs ~ 52,
n_pipe ~ 112) and projects information out: F_win <= F_unwin, σ_win >= σ_unwin.

The Fisher computation reuses desilike's Fisher() machinery by overriding
likelihood.precision after build_bao_likelihood returns.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.special import spherical_jn

sys.path.insert(0, "..")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402
from desilike.fisher import Fisher  # noqa: E402


_DR1_FILES = {
    "BGS":       "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}
_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_DR1_AREA = 7500.0


def _load_dr1_window(tracer_key):
    fpath = _DR1_DIR / _DR1_FILES[tracer_key]
    with h5py.File(fpath, "r") as f:
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        theory_ells, theory_s = [], []
        for key in sorted(f["window/theory"].keys(), key=lambda k: (not k.isdigit(), k)):
            if key.isdigit():
                theory_ells.append(int(f[f"window/theory/{key}/meta/ell"][()]))
                theory_s.append(np.asarray(f[f"window/theory/{key}/s"][...], dtype=np.float64))
    return W, theory_ells, theory_s


def _hankel_matrix(ell, k, dk, s):
    sign = (1j ** ell).real
    H = np.empty((len(s), len(k)), dtype=np.float64)
    for i, si in enumerate(s):
        H[i, :] = sign * (k ** 2) * spherical_jn(ell, k * si) * dk / (2.0 * np.pi ** 2)
    return H


def _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk):
    n_k = len(k_centers)
    n_pipe = n_k * len(pipe_ells)
    n_theory_per = [len(s) for s in theory_s_per_ell]
    n_theory_total = sum(n_theory_per)
    H_block = np.zeros((n_theory_total, n_pipe), dtype=np.float64)
    row_offset = 0
    for it, te in enumerate(theory_ells):
        if te in pipe_ells:
            ip = pipe_ells.index(te)
            H_ell = _hankel_matrix(te, k_centers, dk, theory_s_per_ell[it])
            H_block[row_offset:row_offset + n_theory_per[it],
                    ip * n_k:(ip + 1) * n_k] = H_ell
        row_offset += n_theory_per[it]
    M = W @ H_block
    return M


def _get_dr1_ntracers(tracer_key):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    n_by = {row["tracer"]: float(row["passed"]) for _, row in df.iterrows()}
    name_map = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
    return n_by[name_map.get(tracer_key, tracer_key)]


def _windowed_precision(C_P, M, regularize=1e-12):
    """Return precision_win = M^T (M C_P M^T)^{-1} M (n_pipe x n_pipe).

    Rank deficiency in M (n_obs < n_pipe) is fine: the resulting precision is
    rank n_obs and acts as zero on the null space of M. The full Fisher
    becomes finite once Gaussian priors on Σ etc. are added.
    """
    C_xi = M @ C_P @ M.T
    C_xi = 0.5 * (C_xi + C_xi.T)
    # Add a tiny ridge for numerical stability before inversion.
    ridge = regularize * np.trace(C_xi) / C_xi.shape[0]
    C_xi_reg = C_xi + ridge * np.eye(C_xi.shape[0])
    Cxi_inv = np.linalg.inv(C_xi_reg)
    Cxi_inv = 0.5 * (Cxi_inv + Cxi_inv.T)
    P_win = M.T @ Cxi_inv @ M
    P_win = 0.5 * (P_win + P_win.T)
    return P_win


def _q_fisher_marginalized(info, override_precision=None, debug_tag=""):
    """Mirror prep_covar._q_fisher_from_bao_likelihood_info but with the
    optional override of likelihood.precision before differentiation.
    """
    likelihood = info["likelihood"]
    params = info["params"]
    if override_precision is not None:
        likelihood.precision = np.asarray(override_precision, dtype=np.float64)
        if hasattr(likelihood, "_precision_input"):
            likelihood._precision_input = likelihood.precision.copy()
        if hasattr(likelihood, "precision_hartlap2007"):
            likelihood.precision_hartlap2007 = likelihood.precision.copy()
    fisher = Fisher(likelihood)
    if debug_tag:
        # Confirm what precision the Fisher will see at finalize-time.
        ll = getattr(fisher.likelihood, "likelihoods", [fisher.likelihood])[0]
        prec = ll.precision
        print(f"  [{debug_tag}] likelihood.precision shape={prec.shape}, "
              f"trace={np.trace(prec):.4e}, rank≈{np.linalg.matrix_rank(prec, tol=1e-10)}")
    fisher_result = fisher(**params)
    F_full = -np.array(fisher_result._hessian)
    all_names = [str(p) for p in fisher_result.names()]
    cov_q = prep_covar._cov_q_from_fisher_matrix(
        F_full, all_names, bao_internal=["qpar", "qper"]
    )
    return cov_q, F_full, all_names


def _sigma_dh_dm_dv_from_cov_q(cov_q, info):
    """Convert 2x2 (qpar, qper) covariance to σ(DH/rd), σ(DM/rd), σ(DV/rd)
    using the same fid scaling that get_bao_fisher_covariance uses.
    Returns dict.
    """
    template = info["template"]
    rd = info["rd"]
    DH = float(template.DH_fid)
    DM = float(template.DM_fid)
    sig_qpar = np.sqrt(cov_q[0, 0])
    sig_qper = np.sqrt(cov_q[1, 1])
    rho = cov_q[0, 1] / (sig_qpar * sig_qper)
    sig_DH = DH / rd * sig_qpar
    sig_DM = DM / rd * sig_qper
    # DV ~ (DM^2 * DH)^(1/3). ln α_iso = (1/3) ln q_par + (2/3) ln q_per (fid q=1).
    grad = np.array([1.0 / 3.0, 2.0 / 3.0])
    var_alpha = grad @ cov_q @ grad
    DV = (DM ** 2 * DH) ** (1.0 / 3.0)
    sig_DV = DV / rd * np.sqrt(var_alpha)
    return {
        "sig_DH_rd": sig_DH,
        "sig_DM_rd": sig_DM,
        "sig_DV_rd": sig_DV,
        "sig_qpar": sig_qpar,
        "sig_qper": sig_qper,
        "rho_q": rho,
        "DH_fid": DH,
        "DM_fid": DM,
        "rd": rd,
    }


def run_one(tracer_key, omega_m=0.3153, hrdrag=99.53, kmax_cov=None):
    cfg = TRACER_CONFIGS[tracer_key]
    z_min, z_max = cfg["zrange"]
    n_tracers = _get_dr1_ntracers(tracer_key)
    theta_cosmo, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag}
    )

    info = prep_covar.build_bao_likelihood(
        N_tracers=n_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag_eff,
        tracer_bin=tracer_key,
        zrange=(z_min, z_max),
        z_eff=None,
        area=_DR1_AREA,
        kmax_cov=kmax_cov,
    )

    obs = info["observable"]
    k_centers = np.asarray(obs.k[0], dtype=np.float64)
    dk = float(np.mean(np.diff(k_centers))) if k_centers.size > 1 else 0.005
    pipe_ells = [int(l) for l in obs.ells]
    C_P = np.asarray(info["cov_components"]["C_total"], dtype=np.float64)

    W, theory_ells, theory_s_per_ell = _load_dr1_window(tracer_key)
    M = _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk)

    print(f"  pipe shape: cov_P {C_P.shape}, n_k={len(k_centers)}, ells={pipe_ells}, dk={dk:.4f}")
    print(f"  window:     M {M.shape} (n_obs={M.shape[0]}, n_pipe={M.shape[1]})")
    print(f"  rank-loss:  n_pipe - n_obs = {M.shape[1] - M.shape[0]} modes nominally lost")

    # Unwindowed (current) Fisher — uses default precision = inv(C_P)
    cov_q_unwin, F_unwin, names_unwin = _q_fisher_marginalized(info, debug_tag="unwin")

    # Re-build to get a fresh likelihood; precision override persists otherwise.
    info_win = prep_covar.build_bao_likelihood(
        N_tracers=n_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag_eff,
        tracer_bin=tracer_key,
        zrange=(z_min, z_max),
        z_eff=None,
        area=_DR1_AREA,
        kmax_cov=kmax_cov,
    )
    P_win = _windowed_precision(C_P, M)
    cov_q_win, F_win, names_win = _q_fisher_marginalized(
        info_win, override_precision=P_win, debug_tag="win"
    )

    # Sanity prints comparing precisions and the resulting Fisher q-blocks.
    P_unwin_direct = np.linalg.inv(C_P)
    qpar_i = names_unwin.index("qpar"); qper_i = names_unwin.index("qper")
    print(f"  precision_unwin: trace={np.trace(P_unwin_direct):.4e}, "
          f"||P_win - P_unwin|| / ||P_unwin|| = "
          f"{np.linalg.norm(P_win - P_unwin_direct) / np.linalg.norm(P_unwin_direct):.4e}")
    print(f"  Fisher_unwin (qpar,qper) block:\n{F_unwin[np.ix_([qpar_i, qper_i], [qpar_i, qper_i])]}")
    qpar_j = names_win.index("qpar"); qper_j = names_win.index("qper")
    print(f"  Fisher_win   (qpar,qper) block:\n{F_win[np.ix_([qpar_j, qper_j], [qpar_j, qper_j])]}")

    s_unwin = _sigma_dh_dm_dv_from_cov_q(cov_q_unwin, info)
    s_win   = _sigma_dh_dm_dv_from_cov_q(cov_q_win, info_win)
    return s_unwin, s_win


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracers", nargs="+",
                   default=["ELG2", "LRG2", "QSO", "LRG1", "BGS", "LRG3_ELG1"])
    p.add_argument("--omega-m", type=float, default=0.3153)
    p.add_argument("--hrdrag", type=float, default=99.53)
    p.add_argument("--kmax-cov", type=float, default=None)
    args = p.parse_args()

    print(f"\nWindowed-Fisher diagnostic @ Om={args.omega_m}, hrdrag={args.hrdrag}")
    print(f"Comparing Fisher σ before vs after applying DR1 window M = W @ H_Hankel\n")

    desi_csv = pd.read_csv(_DR1_DIR / "desi_data.csv")
    desi_map = {}
    for _, r in desi_csv.iterrows():
        t = r["tracer"]
        var = str(r["quantity"])  # DH_over_rs / DM_over_rs / DV_over_rs
        sig = float(r["std"])
        desi_map.setdefault(t, {})[var] = sig

    rows = []
    for tracer in args.tracers:
        print(f"=== {tracer} ===")
        try:
            s_unwin, s_win = run_one(tracer, args.omega_m, args.hrdrag, args.kmax_cov)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        name_for_desi = {"LRG3_ELG1": "LRG3+ELG1"}.get(tracer, tracer)
        desi = desi_map.get(name_for_desi, {})

        for q, key, desi_key in [
            ("DH/rd", "sig_DH_rd", "DH_over_rs"),
            ("DM/rd", "sig_DM_rd", "DM_over_rs"),
            ("DV/rd", "sig_DV_rd", "DV_over_rs"),
        ]:
            u = s_unwin[key]; w = s_win[key]
            d = desi.get(desi_key, None)
            print(f"  {q:5s}  unwin σ = {u:.4f}   win σ = {w:.4f}   ratio win/unwin = {w/u:.3f}"
                  + (f"   DESI = {d:.4f}   F_unwin/D = {u/d:.3f}   F_win/D = {w/d:.3f}" if d else ""))
            rows.append({
                "tracer": tracer, "quantity": q,
                "sig_unwin": u, "sig_win": w, "sig_desi": d,
                "ratio_win_unwin": w / u,
                "F_unwin_over_D": (u / d) if d else None,
                "F_win_over_D": (w / d) if d else None,
            })
        print()

    df = pd.DataFrame(rows)
    out_csv = "windowed_fisher_diag.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
