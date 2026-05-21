"""Cov-substitution Fisher diagnostic.

Asks: how much of the F/D ~ 0.5 gap is in the covariance vs in the
Fisher methodology DESI uses?

Three Fisher runs per tracer:

  F0 :  unwindowed pipeline                       (current σ)
        precision = inv(C_P^pipe)

  F1 :  pipeline cov projected to ξ_obs space     (§26 windowed)
        precision = M^T (M C_P^pipe M^T)^{-1} M
        — should reproduce F0 within ~percent; sanity check.

  F2 :  DESI's EZmock cov substituted in ξ_obs    (this diagnostic)
        precision = M^T (cov_ξ^DR1)^{-1} M

If σ_F2 ≈ DESI's published σ → the entire residual lives in C_P.
If σ_F2 still falls short  → there's a methodology piece DESI uses
that we don't.

M = W @ H_block, with H the per-ℓ Hankel matrix and W the DR1 survey
window matrix. Same construction as windowed_fisher_diag.py.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "..")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402
from desilike.fisher import Fisher  # noqa: E402
from scipy.special import spherical_jn  # noqa: E402  (after prep_covar's env setup)


_DR1_FILES = {
    "BGS":       "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}
_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_DR1_LIK_DIR = _DR1_DIR / "likelihoods"
_DR1_AREA = 7500.0
_NAME_MAP_TO_DATA = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}

# EZmock joint FS+ξ_post-recon covariances (DR1 v1.0 VAC).
# Same realizations as the local likelihood bundles; the difference is that
# these files ship cov only (no W matrix). For the F2 diagnostic we pair them
# with H_Hankel alone — §26 bounded the W contribution to σ at <2%.
_EZMOCK_DIR = _DR1_LIK_DIR / "covariance"
_EZMOCK_CORRREC_FILES = {
    "BGS":        "covariance_spectrum-poles+correlation-poles-recon_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.h5",
    "LRG1":       "covariance_spectrum-poles+correlation-poles-recon_LRG_GCcomb_z0.4-0.6_thetacut0.05.h5",
    "LRG2":       "covariance_spectrum-poles+correlation-poles-recon_LRG_GCcomb_z0.6-0.8_thetacut0.05.h5",
    "LRG3_ELG1":  "covariance_spectrum-poles+correlation-poles-recon_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5",
    "ELG2":       "covariance_spectrum-poles+correlation-poles-recon_ELG_LOPnotqso_GCcomb_z1.1-1.6_thetacut0.05.h5",
    "QSO":        "covariance_spectrum-poles+correlation-poles-recon_QSO_GCcomb_z0.8-2.1_thetacut0.05.h5",
}


def _load_dr1(tracer_key):
    """Load the local DESI likelihood bundle: post-recon ξ cov + window."""
    fpath = _DR1_LIK_DIR / _DR1_FILES[tracer_key]
    if not fpath.exists():
        raise FileNotFoundError(f"DR1 file not found: {fpath}")
    with h5py.File(fpath, "r") as f:
        cov_xi = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        theory_ells, theory_s = [], []
        for key in sorted(f["window/theory"].keys(), key=lambda k: (not k.isdigit(), k)):
            if key.isdigit():
                theory_ells.append(int(f[f"window/theory/{key}/meta/ell"][()]))
                theory_s.append(np.asarray(f[f"window/theory/{key}/s"][...], dtype=np.float64))
    return cov_xi, W, theory_ells, theory_s


def _load_ezmock_correlationrecon(tracer_key, keep_ells, s_range=(50.0, 150.0)):
    """Load the post-recon ξ_ℓ(s) block from the joint FS+ξ EZmock cov file,
    sliced to the requested ells AND restricted to the requested s-range.

    s_range defaults to [50, 150] Mpc/h to match Adame+24's BAO fit window.
    The raw EZmock cov is on a 50-bin s-grid from ~3 to 198 Mpc/h; including
    s outside [50, 150] gives the Fisher access to information DESI's BAO
    fit does not use, which makes the F2 result artificially tight. Pass
    s_range=None to use the full s-grid.

    No W matrix in these files — caller uses M = H_Hankel only (§26 bounds
    the W contribution to σ at <2% on LRG2).

    Returns (cov_xi, theory_ells, theory_s_per_ell).
    """
    fpath = _EZMOCK_DIR / _EZMOCK_CORRREC_FILES[tracer_key]
    if not fpath.exists():
        raise FileNotFoundError(f"EZmock file not found: {fpath}")
    with h5py.File(fpath, "r") as f:
        C_full = np.asarray(f["value"][...], dtype=np.float64)
        # Block layout: [spectrum (240), correlationrecon (150)] per labels_values.
        spec_grp = f["observable/spectrum"]
        n_spec = sum(int(spec_grp[f"{e}/k"].shape[0])
                     for e in spec_grp.keys() if e.isdigit())
        cr_grp = f["observable/correlationrecon"]
        cr_ells_in_file = [int(cr_grp[f"{e}/meta/ell"][()])
                           for e in sorted(cr_grp.keys()) if e.isdigit()]
        cr_n_per_ell = [int(cr_grp[f"{e}/s"].shape[0])
                        for e in sorted(cr_grp.keys()) if e.isdigit()]
        # Per-ell s grids in the order they appear in the file
        s_per_ell_in_file = [np.asarray(cr_grp[f"{e}/s"][...], dtype=np.float64)
                             for e in sorted(cr_grp.keys()) if e.isdigit()]

    # Slice out the correlationrecon block, then pick the requested ells +
    # restrict to the requested s-range per ell.
    cr_block = C_full[n_spec:, n_spec:]
    idx = []
    theory_ells_out = []
    theory_s_per_ell = []
    row_offset_by_ell = {}
    cur = 0
    for ell_f, n_ell in zip(cr_ells_in_file, cr_n_per_ell):
        row_offset_by_ell[ell_f] = (cur, cur + n_ell)
        cur += n_ell
    for want_ell in keep_ells:
        if want_ell not in row_offset_by_ell:
            continue
        lo, hi = row_offset_by_ell[want_ell]
        ell_idx_in_file = cr_ells_in_file.index(want_ell)
        s_full = s_per_ell_in_file[ell_idx_in_file]
        if s_range is None:
            s_mask = np.ones_like(s_full, dtype=bool)
        else:
            s_mask = (s_full >= s_range[0]) & (s_full <= s_range[1])
        kept_local = np.flatnonzero(s_mask)
        idx.extend((lo + j) for j in kept_local)
        theory_ells_out.append(want_ell)
        theory_s_per_ell.append(s_full[s_mask])
    idx = np.asarray(idx, dtype=np.int64)
    cov_xi_sub = cr_block[np.ix_(idx, idx)]
    return cov_xi_sub, theory_ells_out, theory_s_per_ell


def _get_dr1_ntracers(tracer_key):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    n_by = {row["tracer"]: float(row["passed"]) for _, row in df.iterrows()}
    return n_by[_NAME_MAP_TO_DATA.get(tracer_key, tracer_key)]


def _get_desi_std(tracer_key):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    sub = df[df["tracer"] == _NAME_MAP_TO_DATA.get(tracer_key, tracer_key)]
    return {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}


def _hankel_matrix(ell, k, dk, s):
    sign = (1j ** ell).real
    H = np.empty((len(s), len(k)), dtype=np.float64)
    for i, si in enumerate(s):
        H[i, :] = sign * (k ** 2) * spherical_jn(ell, k * si) * dk / (2.0 * np.pi ** 2)
    return H


def _build_H_block(theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk):
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
    return H_block


def _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk):
    """M = W @ H_block. Used with the local likelihood bundles which ship W."""
    H = _build_H_block(theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk)
    return W @ H


def _build_H_block_fftlog(theory_ells, theory_s_per_ell, pipe_ells, k_centers,
                          P_fid_per_ell=None):
    """FFTLog version of _build_H_block — built linearly via basis-vector
    evaluation using extrap='edge' (constant-value padding).

    Why extrap='edge' instead of 'log': cosmoprimo's log-log extrapolation is
    *nonlinear* in P (log of boundary values), so basis vectors with zero
    tails crash it (log(0) → NaN). A finite-difference Jacobian at the
    fiducial P is too noisy (cond → 10^20). 'edge' (constant padding) is
    linear in P, basis-vector evaluation is well-defined, and the result
    is a clean linear operator.

    Per §31c, the hand-Riemann Hankel on the pipeline's narrow linear k-grid
    [0.022, 0.30] catastrophically under-predicts ξ at the BAO peak (factor
    ~25× suppression at s=98 Mpc/h). FFTLog with log-spaced quadrature
    uses better integration weights than the linear Riemann sum and
    handles boundary effects more gracefully via the padding.

    P_fid_per_ell is kept for API compatibility but unused with extrap='edge'.
    """
    from cosmoprimo.fftlog import PowerToCorrelation

    n_k = len(k_centers)
    n_pipe = n_k * len(pipe_ells)
    n_per = [len(s) for s in theory_s_per_ell]
    n_theory_total = sum(n_per)

    # Log-k grid for FFTLog, covering the pipeline's k range.
    k_log = np.logspace(np.log10(k_centers[0]), np.log10(k_centers[-1]), 1024)
    ffts = {ell: PowerToCorrelation(k_log, ell=ell) for ell in pipe_ells}

    H_block = np.zeros((n_theory_total, n_pipe), dtype=np.float64)

    for i_p, p_ell in enumerate(pipe_ells):
        if p_ell not in theory_ells:
            continue
        i_th = theory_ells.index(p_ell)
        s_th = theory_s_per_ell[i_th]
        th_offset = sum(n_per[:i_th])
        fft = ffts[p_ell]

        # Build all basis-vector columns: P_lin[i] is the i-th basis vector.
        # After np.interp to log-k, this is a triangular "tent" at k_centers[i].
        basis_lin = np.eye(n_k)
        basis_log = np.zeros((n_k, len(k_log)))
        for ic in range(n_k):
            basis_log[ic] = np.interp(k_log, k_centers, basis_lin[ic])
        # extrap='edge' pads with boundary values (zero for tent basis vectors
        # whose boundaries are zero) → linear in input, no NaN issues.
        s_fft, xi_fft = fft(basis_log, extrap="edge")
        s_fft = np.asarray(s_fft); xi_fft = np.asarray(xi_fft)
        if s_fft.ndim == 2:
            s_fft = s_fft[0]
        order = np.argsort(s_fft)
        s_fft_sorted = s_fft[order]
        for ic in range(n_k):
            xi_at_th = np.interp(s_th, s_fft_sorted, xi_fft[ic][order])
            H_block[th_offset:th_offset + n_per[i_th], i_p * n_k + ic] = xi_at_th

    return H_block


def _build_M_fftlog(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers,
                    P_fid_per_ell):
    """M = W @ H_block built via FFTLog instead of hand-Riemann. See
    _build_H_block_fftlog for motivation (§31c showed Riemann is broken
    at s > 50 Mpc/h)."""
    H = _build_H_block_fftlog(theory_ells, theory_s_per_ell, pipe_ells, k_centers,
                              P_fid_per_ell)
    return W @ H


def _precision_from_cov_xi(cov_xi, M, regularize=1e-10):
    """precision_P = M^T (cov_xi + ε I)^{-1} M, with a tiny ridge for stability."""
    C = 0.5 * (cov_xi + cov_xi.T)
    ridge = regularize * np.trace(C) / C.shape[0]
    C_reg = C + ridge * np.eye(C.shape[0])
    C_inv = np.linalg.inv(C_reg)
    C_inv = 0.5 * (C_inv + C_inv.T)
    P = M.T @ C_inv @ M
    return 0.5 * (P + P.T)


def _q_fisher(info, override_precision=None):
    likelihood = info["likelihood"]
    params = info["params"]
    if override_precision is not None:
        likelihood.precision = np.asarray(override_precision, dtype=np.float64)
        if hasattr(likelihood, "_precision_input"):
            likelihood._precision_input = likelihood.precision.copy()
        if hasattr(likelihood, "precision_hartlap2007"):
            likelihood.precision_hartlap2007 = likelihood.precision.copy()
    fisher = Fisher(likelihood)
    res = fisher(**params)
    F_full = -np.array(res._hessian)
    names = [str(p) for p in res.names()]
    cov_q = prep_covar._cov_q_from_fisher_matrix(F_full, names, bao_internal=["qpar", "qper"])
    return cov_q


def _sigmas_from_cov_q(cov_q, info):
    # Use template.{DH,DM}_over_rd_fid directly — dimensionally correct.
    # The naive `template.DH_fid / info["rd"]` form is dimensionally mixed
    # (DH_fid is Mpc/h, info["rd"] is proper Mpc) and underreports σ by 1/h.
    template = info["template"]
    DH_over_rd = float(template.DH_over_rd_fid)
    DM_over_rd = float(template.DM_over_rd_fid)
    sig_DH = DH_over_rd * np.sqrt(cov_q[0, 0])
    sig_DM = DM_over_rd * np.sqrt(cov_q[1, 1])
    grad = np.array([1.0 / 3.0, 2.0 / 3.0])
    var_alpha = grad @ cov_q @ grad
    DV_over_rd = (DM_over_rd ** 2 * DH_over_rd) ** (1.0 / 3.0)
    sig_DV = DV_over_rd * np.sqrt(max(var_alpha, 0.0))
    return sig_DH, sig_DM, sig_DV


def _build_info(tracer_key, omega_m, hrdrag):
    cfg = TRACER_CONFIGS[tracer_key]
    z_min, z_max = cfg["zrange"]
    n_tracers = _get_dr1_ntracers(tracer_key)
    theta_cosmo, hrdrag_eff = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag}
    )
    return prep_covar.build_bao_likelihood(
        N_tracers=n_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag_eff,
        tracer_bin=tracer_key,
        zrange=(z_min, z_max),
        z_eff=None,
        area=_DR1_AREA,
    )


def run_one(tracer_key, omega_m, hrdrag, source="bundle", hankel="riemann"):
    print(f"\n=== {tracer_key}  (source={source}) ===")

    # Pipeline ingredients
    info_F0 = _build_info(tracer_key, omega_m, hrdrag)
    obs = info_F0["observable"]
    k_centers = np.asarray(obs.k[0], dtype=np.float64)
    dk = float(np.mean(np.diff(k_centers))) if k_centers.size > 1 else 0.005
    pipe_ells = [int(l) for l in obs.ells]
    C_P_pipe = np.asarray(info_F0["cov_components"]["C_total"], dtype=np.float64)

    # Fiducial P_ℓ(k) — needed for FFTLog M (Jacobian at this point).
    info_F0["likelihood"](**info_F0["params"])
    P_fid_per_ell = [np.asarray(obs.theory[i], dtype=np.float64) for i in range(len(pipe_ells))]

    # Cov source: local likelihood bundle (cov_ξ + W) or EZmock joint file
    # (cov_ξ only, no W). The §26 finding bounds the missing-W effect at <2%
    # on σ for LRG2, so dropping W is a clean approximation.
    if source == "bundle":
        cov_xi_dr1, W, theory_ells, theory_s_per_ell = _load_dr1(tracer_key)
        if hankel == "fftlog":
            M = _build_M_fftlog(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers,
                                P_fid_per_ell)
        else:
            M = _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk)
    elif source == "ezmock":
        cov_xi_dr1, theory_ells, theory_s_per_ell = _load_ezmock_correlationrecon(
            tracer_key, keep_ells=pipe_ells)
        if hankel == "fftlog":
            M = _build_H_block_fftlog(theory_ells, theory_s_per_ell, pipe_ells, k_centers,
                                       P_fid_per_ell)
        else:
            M = _build_H_block(theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk)
    else:
        raise ValueError(f"unknown source {source!r}")
    print(f"  M shape: {M.shape}   C_P_pipe: {C_P_pipe.shape}   cov_ξ_DR1: {cov_xi_dr1.shape}")

    # F0: unwindowed pipeline
    cov_q_F0 = _q_fisher(info_F0)

    # F1: pipeline cov projected to ξ
    info_F1 = _build_info(tracer_key, omega_m, hrdrag)
    prec_F1 = _precision_from_cov_xi(M @ C_P_pipe @ M.T, M)
    cov_q_F1 = _q_fisher(info_F1, override_precision=prec_F1)

    # F2: DR1 EZmock cov substituted in
    info_F2 = _build_info(tracer_key, omega_m, hrdrag)
    prec_F2 = _precision_from_cov_xi(cov_xi_dr1, M)
    cov_q_F2 = _q_fisher(info_F2, override_precision=prec_F2)

    s0 = _sigmas_from_cov_q(cov_q_F0, info_F0)
    s1 = _sigmas_from_cov_q(cov_q_F1, info_F1)
    s2 = _sigmas_from_cov_q(cov_q_F2, info_F2)
    desi = _get_desi_std(tracer_key)
    return {
        "F0_pipe_unwin":     s0,
        "F1_pipe_xi":        s1,
        "F2_DR1cov_xi":      s2,
        "DESI_published":    (desi.get("DH_over_rs", np.nan),
                              desi.get("DM_over_rs", np.nan),
                              desi.get("DV_over_rs", np.nan)),
    }


def _fmt(triple):
    return "  ".join(f"{x:8.4f}" if np.isfinite(x) else "    nan " for x in triple)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracers", nargs="+",
                    default=["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"])
    ap.add_argument("--omega-m", type=float, default=0.3153)
    ap.add_argument("--hrdrag", type=float, default=99.53)
    ap.add_argument("--source", choices=["bundle", "ezmock"], default="ezmock",
                    help="Where to get cov_ξ: 'bundle' (local likelihood h5, "
                         "LRG2 & QSO only, cov+W) or 'ezmock' (public DR1 "
                         "EZmock joint cov file, all six tracers, cov only).")
    ap.add_argument("--hankel", choices=["riemann", "fftlog"], default="riemann",
                    help="Method for the P→ξ Hankel transform inside M. "
                         "'riemann' (default for legacy comparison) uses a "
                         "hand-sum on the pipeline's narrow linear k-grid — "
                         "broken at s > 50 Mpc/h per §31c. 'fftlog' uses "
                         "cosmoprimo's PowerToCorrelation with log-log "
                         "extrapolation (the §31d fix).")
    args = ap.parse_args()

    print(f"Cov-substitution diagnostic @ Om={args.omega_m}, hrdrag={args.hrdrag}, source={args.source}")
    print("F0 = unwindowed pipeline cov_P")
    if args.source == "bundle":
        print("F1 = pipeline cov projected to ξ via M = W·H_Hankel")
        print("F2 = DR1 likelihood-bundle cov_ξ substituted in")
    else:
        print("F1 = pipeline cov projected to ξ via M = H_Hankel (no W)")
        print("F2 = DR1 EZmock correlationrecon cov_ξ substituted in")

    rows = []
    for tracer in args.tracers:
        try:
            rows.append((tracer, run_one(tracer, args.omega_m, args.hrdrag, source=args.source, hankel=args.hankel)))
        except FileNotFoundError as exc:
            print(f"  [{tracer}] skipped: {exc}")

    print("\n" + "=" * 88)
    print(f"{'tracer':<11} {'run':<18} {'σ(DH/rd)':>10} {'σ(DM/rd)':>10} {'σ(DV/rd)':>10}")
    print("-" * 88)
    for tracer, r in rows:
        for label in ("F0_pipe_unwin", "F1_pipe_xi", "F2_DR1cov_xi", "DESI_published"):
            sh, sm, sv = r[label]
            print(f"{tracer:<11} {label:<18} {sh:>10.4f} {sm:>10.4f} {sv:>10.4f}")
        # Ratios F2 / DESI — answers (b)
        f2 = r["F2_DR1cov_xi"]; ds = r["DESI_published"]
        ratios = tuple(f / d if (np.isfinite(f) and np.isfinite(d) and d > 0) else np.nan
                       for f, d in zip(f2, ds))
        print(f"{tracer:<11} {'F2/DESI':<18} {ratios[0]:>10.3f} {ratios[1]:>10.3f} {ratios[2]:>10.3f}")
        print("-" * 88)


if __name__ == "__main__":
    main()
