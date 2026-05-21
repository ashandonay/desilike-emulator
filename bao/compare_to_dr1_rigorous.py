"""Rigorous side-by-side comparison of pipeline covariance to DR1 BAO post-recon
covariance, in DR1's observable basis (xi_obs).

The chain is:
  Cov_P_theory  --[Hankel j_l]-->  Cov_xi_theory(s_grid_dense)
  Cov_xi_theory --[DR1 W]-->       Cov_xi_observed(s_grid_data)

where W is the 52x600 configuration-space window matrix shipped with the DR1
likelihood file (DESI 2024 BAO galaxy/QSO companion, mocks-derived per-bin
covariance; observables are post-recon xi_l(s)).

Then we directly compare the resulting 52x52 (or 26x26 for QSO) matrix to
DR1's published EZmock-derived covariance in identical units and identical
binning.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import spherical_jn

sys.path.insert(0, "..")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402

import pandas as pd  # noqa: E402


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

# Public DR1 EZmock joint FS+ξ_post-recon covariances (v1.0 VAC).
# Cover all six DR1 tracers. Cov-only — no W matrix. §26 bounds the missing-W
# contribution to σ at < 2%, so for visual structure comparison this is fine.
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
    """Return cov_xi_obs (n_obs,n_obs), W (n_obs,n_theory),
    obs_ells, obs_s_per_ell (list of arrays), theory_ells, theory_s_per_ell."""
    fpath = _DR1_LIK_DIR / _DR1_FILES[tracer_key]
    if not fpath.exists():
        raise FileNotFoundError(f"DR1 file not found: {fpath}")
    with h5py.File(fpath, "r") as f:
        cov = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        obs_ells, obs_s = [], []
        for key in sorted(f["covariance/observable"].keys(), key=lambda k: (not k.isdigit(), k)):
            if key.isdigit():
                obs_ells.append(int(f[f"covariance/observable/{key}/meta/ell"][()]))
                obs_s.append(np.asarray(f[f"covariance/observable/{key}/s"][...], dtype=np.float64))
        theory_ells, theory_s = [], []
        for key in sorted(f["window/theory"].keys(), key=lambda k: (not k.isdigit(), k)):
            if key.isdigit():
                theory_ells.append(int(f[f"window/theory/{key}/meta/ell"][()]))
                theory_s.append(np.asarray(f[f"window/theory/{key}/s"][...], dtype=np.float64))
    return cov, W, obs_ells, obs_s, theory_ells, theory_s


def _load_ezmock_correlationrecon(tracer_key, keep_ells, s_range=None):
    """Load the post-recon ξ_ℓ(s) block from the joint FS+ξ EZmock cov file
    (the canonical DR1 cov, available for all six tracers).

    Returns (cov_xi, obs_ells, obs_s_per_ell) restricted to keep_ells and the
    optional s_range (in Mpc/h). The raw block is ℓ ∈ {0,2,4} × 50 s-bins on
    s ∈ [2.97, 198.01] Mpc/h; the pipeline only models ℓ=0,2 so we slice to
    those by default.

    No window matrix in this file — caller treats M = H_Hankel only.
    """
    fpath = _EZMOCK_DIR / _EZMOCK_CORRREC_FILES[tracer_key]
    if not fpath.exists():
        raise FileNotFoundError(f"EZmock file not found: {fpath}")
    with h5py.File(fpath, "r") as f:
        C_full = np.asarray(f["value"][...], dtype=np.float64)
        # Block layout per labels_values: [spectrum (240), correlationrecon (150)]
        spec_grp = f["observable/spectrum"]
        n_spec = sum(int(spec_grp[f"{e}/k"].shape[0])
                     for e in spec_grp.keys() if e.isdigit())
        cr_grp = f["observable/correlationrecon"]
        cr_keys = sorted([k for k in cr_grp.keys() if k.isdigit()])
        cr_ells_in_file = [int(cr_grp[f"{e}/meta/ell"][()]) for e in cr_keys]
        cr_n_per_ell = [int(cr_grp[f"{e}/s"].shape[0]) for e in cr_keys]
        s_per_ell_in_file = [np.asarray(cr_grp[f"{e}/s"][...], dtype=np.float64)
                             for e in cr_keys]

    cr_block = C_full[n_spec:, n_spec:]
    row_offset_by_ell = {}
    cur = 0
    for ell_f, n_ell in zip(cr_ells_in_file, cr_n_per_ell):
        row_offset_by_ell[ell_f] = (cur, cur + n_ell)
        cur += n_ell
    idx, ells_out, s_per_ell_out = [], [], []
    for want_ell in keep_ells:
        if want_ell not in row_offset_by_ell:
            continue
        lo, hi = row_offset_by_ell[want_ell]
        s_full = s_per_ell_in_file[cr_ells_in_file.index(want_ell)]
        if s_range is None:
            mask = np.ones_like(s_full, dtype=bool)
        else:
            mask = (s_full >= s_range[0]) & (s_full <= s_range[1])
        kept_local = np.flatnonzero(mask)
        idx.extend((lo + j) for j in kept_local)
        ells_out.append(want_ell)
        s_per_ell_out.append(s_full[mask])
    idx = np.asarray(idx, dtype=np.int64)
    cov_xi = cr_block[np.ix_(idx, idx)]
    return cov_xi, ells_out, s_per_ell_out


def _get_dr1_ntracers(tracer_key):
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    n_by = {row["tracer"]: float(row["passed"]) for _, row in df.iterrows()}
    name_map = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
    return n_by[name_map.get(tracer_key, tracer_key)]


def _pipeline_pcov(tracer_key, omega_m, hrdrag, component="C_total",
                   kmax_cov=None):
    """Run the pipeline at DR1 fiducial cosmology+geometry.
    Returns selected component (n_pipe, n_pipe), k_centers, dk, ells."""
    cfg = TRACER_CONFIGS[tracer_key]
    z_min, z_max = cfg["zrange"]
    n_tracers = _get_dr1_ntracers(tracer_key)
    theta_cosmo, hrdrag_eff = prep_covar._to_bao_cosmo_params(  # noqa: SLF001
        {**prep_covar.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag}
    )
    # z_eff=None lets build_bao_likelihood derive it from the n(z) slices via
    # Fisher-info weighting (V × (nP/(1+nP))² per slice). Matches DESI's
    # Adame+24 Sec 2.5 convention and is cosmology-clean.
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
    if k_centers.size > 1:
        dk = float(np.mean(np.diff(k_centers)))
    else:
        dk = 0.005
    ells = tuple(int(l) for l in obs.ells)
    cov_p = np.asarray(info["cov_components"][component], dtype=np.float64)
    return cov_p, k_centers, dk, ells


def _hankel_matrix(ell, k, dk, s):
    """H[i, a] : xi_l(s_i) = sum_a H[i,a] P_l(k_a)
    using xi_l(s) = (i^l/2pi^2) integral k^2 j_l(ks) P_l(k) dk."""
    sign = (1j ** ell).real  # 1, -1, 1 for ell = 0, 2, 4
    H = np.empty((len(s), len(k)), dtype=np.float64)
    for i, si in enumerate(s):
        H[i, :] = sign * (k ** 2) * spherical_jn(ell, k * si) * dk / (2.0 * np.pi ** 2)
    return H


def _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk):
    """Build M of shape (n_obs, n_pipe) such that
        flat_xi_obs = M @ flat_pipeP_theory.

    Construct H_block (n_theory_total, n_pipe) mapping pipeline P to theory ξ;
    pipeline ells absent from theory get zero columns, theory ells absent
    from pipeline get zero rows. M = W @ H_block when W is provided;
    M = H_block when W is None (EZmock-source path — §26 bounds the missing
    W contribution at < 2% on σ).
    """
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
    M = H_block if W is None else W @ H_block
    return M, H_block


def _project_cov(cov_p, M):
    """Cov_xi_obs = M @ cov_p @ M^T."""
    return M @ cov_p @ M.T


def _bandpower_smooth(cov_p, k_centers, n_ells, sigma_dk):
    """Approximate finite-survey bandpower coupling by Gaussian-smoothing
    cov_P along k within each multipole block.

    Implements Cov_meas = K cov_p K^T where K is block-diagonal with one
    Gaussian-kernel block per ell. Within a block, K[a,b] propto
    exp(-(k_a - k_b)^2 / 2 sigma_k^2) with sigma_k = sigma_dk * dk, rows
    normalized to unit sum so K conserves total variance.

    sigma_dk = 0 => identity (no smoothing).
    """
    if sigma_dk <= 0:
        return cov_p
    n_k = len(k_centers)
    sigma_k = sigma_dk * float(np.mean(np.diff(k_centers)))
    diff = k_centers[:, None] - k_centers[None, :]
    K_block = np.exp(-0.5 * (diff / sigma_k) ** 2)
    K_block /= K_block.sum(axis=1, keepdims=True)
    K = np.zeros((n_k * n_ells, n_k * n_ells), dtype=np.float64)
    for i in range(n_ells):
        K[i * n_k:(i + 1) * n_k, i * n_k:(i + 1) * n_k] = K_block
    return K @ cov_p @ K.T


def _heatmap(ax, M, n_per_block, n_blocks, title, vmin, vmax):
    absM = np.abs(M)
    im = ax.imshow(absM, origin="upper", cmap="magma",
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    for i in range(1, n_blocks):
        ax.axhline(i * n_per_block - 0.5, color="cyan", lw=0.6, alpha=0.7)
        ax.axvline(i * n_per_block - 0.5, color="cyan", lw=0.6, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("flat index")
    ax.set_ylabel("flat index")
    return im


def _corr(M):
    diag = np.diag(M).copy()
    diag[diag <= 0] = np.nan
    norm = np.sqrt(np.outer(diag, diag))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = M / norm
    return np.where(np.isfinite(corr), corr, 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracer", required=True,
                   help=f"One of {sorted(_EZMOCK_CORRREC_FILES.keys())}.")
    p.add_argument("--omega-m", type=float, default=0.3153)
    p.add_argument("--hrdrag", type=float, default=99.53)
    p.add_argument("--source", choices=["ezmock", "bundle"], default="ezmock",
                   help="Where to load DR1 cov_ξ from. 'ezmock' (default): "
                        "DR1 EZmock joint cov file, available for all six "
                        "tracers, cov only (no W matrix). 'bundle': local "
                        "likelihood h5 file (cov + W), available locally for "
                        "LRG2 and QSO only.")
    p.add_argument("--s-range", type=float, nargs=2, default=None,
                   metavar=("S_MIN", "S_MAX"),
                   help="Restrict s-range (Mpc/h) when loading from EZmock "
                        "source. Default: use the full s-grid (s ∈ [3, 198]).")
    p.add_argument("--component", default="C_total",
                   choices=["C_total", "C_gauss", "C_SSC", "C_recon_shot"],
                   help="Pipeline covariance component to project (default: C_total).")
    p.add_argument("--bandpower-sigma-dk", type=float, default=0.0,
                   help="Bandpower-coupling kernel width in units of dk; "
                        "0 disables. Typical: 1-3 for DESI-scale boxes.")
    p.add_argument("--kmax-cov", type=float, default=None,
                   help="Override pipeline klim kmax for covariance "
                        "computation (default: 0.30 from prep_covar).")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    print(f"\n=== {args.tracer} (source={args.source}) ===")

    # Compute pipeline first so we know which ells to load from DR1.
    cov_p_pipe, k_centers, dk, pipe_ells = _pipeline_pcov(
        args.tracer, args.omega_m, args.hrdrag,
        component=args.component, kmax_cov=args.kmax_cov,
    )
    pipe_ells = list(pipe_ells)
    print(f"Pipeline cov_P: shape {cov_p_pipe.shape}, ells={pipe_ells}, "
          f"k in [{k_centers[0]:.4f}, {k_centers[-1]:.4f}] h/Mpc, dk={dk:.4f}")

    if args.source == "bundle":
        cov_xi_dr1, W, obs_ells, obs_s_per_ell, theory_ells, theory_s_per_ell = _load_dr1(args.tracer)
        print(f"DR1 cov (bundle): shape {cov_xi_dr1.shape}, obs ells={obs_ells}, "
              f"s ∈ [{obs_s_per_ell[0][0]:.1f}, {obs_s_per_ell[0][-1]:.1f}] Mpc/h")
        print(f"Window: {W.shape}, theory ells={theory_ells}")
    else:  # ezmock
        cov_xi_dr1, obs_ells, obs_s_per_ell = _load_ezmock_correlationrecon(
            args.tracer, keep_ells=pipe_ells, s_range=tuple(args.s_range) if args.s_range else None,
        )
        W = None
        # For ezmock + H-only projection, theory ells & s-grid equal obs ells & s-grid.
        theory_ells, theory_s_per_ell = obs_ells, obs_s_per_ell
        s_lo, s_hi = obs_s_per_ell[0][0], obs_s_per_ell[0][-1]
        print(f"DR1 cov (ezmock): shape {cov_xi_dr1.shape}, obs ells={obs_ells}, "
              f"n_s_per_ell={len(obs_s_per_ell[0])}, s ∈ [{s_lo:.1f}, {s_hi:.1f}] Mpc/h")
        print("Window: N/A (EZmock path uses M = H_Hankel only; §26 bounds W effect at < 2%)")
    n_obs_per = len(obs_s_per_ell[0])

    M, H_block = _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk)
    print(f"Projection matrix M: shape {M.shape}")

    if args.bandpower_sigma_dk > 0:
        cov_p_pipe = _bandpower_smooth(
            cov_p_pipe, k_centers, len(pipe_ells), args.bandpower_sigma_dk
        )
        print(f"Bandpower smoothing applied: sigma_k = {args.bandpower_sigma_dk:.2f}*dk "
              f"= {args.bandpower_sigma_dk * dk:.4f} h/Mpc")

    cov_xi_pipe = _project_cov(cov_p_pipe, M)
    print(f"Projected pipeline cov_xi_obs: shape {cov_xi_pipe.shape}")

    # Sanity-check shapes
    assert cov_xi_pipe.shape == cov_xi_dr1.shape, \
        f"Shape mismatch: pipeline {cov_xi_pipe.shape} vs DR1 {cov_xi_dr1.shape}"

    # Diagnostics on the comparison
    diag_dr1 = np.diag(cov_xi_dr1)
    diag_pipe = np.diag(cov_xi_pipe)
    sigma_dr1 = np.sqrt(np.maximum(diag_dr1, 0.0))
    sigma_pipe = np.sqrt(np.maximum(diag_pipe, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = sigma_pipe / sigma_dr1
    print("\nSigma per bin (pipeline / DR1):")
    print(f"  median ratio:  {np.nanmedian(ratio):.3f}")
    print(f"  16-84 pct:     [{np.nanpercentile(ratio, 16):.3f}, {np.nanpercentile(ratio, 84):.3f}]")
    print(f"  min, max:      [{np.nanmin(ratio):.3f}, {np.nanmax(ratio):.3f}]")

    for label, M_cov in [("DR1", cov_xi_dr1), ("pipeline_proj", cov_xi_pipe)]:
        C = _corr(M_cov)
        n = C.shape[0]
        off = C[~np.eye(n, dtype=bool)]
        print(f"  {label:14s}: median |rho_off|={np.nanmedian(np.abs(off)):.3f}  "
              f"max={np.nanmax(np.abs(off)):.3f}  "
              f"frac>0.1={np.mean(np.abs(off)>0.1):.3f}")

    # Plot: 2x2 = (heatmap, diagonal) x (DR1, pipeline)
    n_blocks = len(obs_ells)
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    abs_dr1 = np.abs(cov_xi_dr1)
    abs_pipe = np.abs(cov_xi_pipe)
    pos = np.concatenate([abs_dr1[abs_dr1 > 0], abs_pipe[abs_pipe > 0]])
    vmin, vmax = pos.min(), pos.max()

    ax = fig.add_subplot(gs[0, 0])
    im = _heatmap(ax, cov_xi_dr1, n_obs_per, n_blocks, r"DR1 $|\mathrm{Cov}(\xi_\mathrm{obs})|$", vmin, vmax)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = fig.add_subplot(gs[1, 0])
    im = _heatmap(ax, cov_xi_pipe, n_obs_per, n_blocks,
                  r"Pipeline projected $|\mathrm{Cov}(\xi_\mathrm{obs})|$", vmin, vmax)
    fig.colorbar(im, ax=ax, shrink=0.85)

    # Diagonals (sigma per bin) per ell
    ax = fig.add_subplot(gs[0:, 1])
    for io, oe in enumerate(obs_ells):
        sl = slice(io * n_obs_per, (io + 1) * n_obs_per)
        s_arr = obs_s_per_ell[io]
        ax.plot(s_arr, sigma_dr1[sl], "-",  color=f"C{io}", lw=1.6,
                label=fr"DR1 $\ell={oe}$")
        ax.plot(s_arr, sigma_pipe[sl], "--", color=f"C{io}", lw=1.6,
                label=fr"pipeline $\ell={oe}$")
    ax.set_xlabel(r"$s\,[\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$\sigma(\xi_\ell(s))$")
    ax.set_yscale("log")
    ax.set_title("Per-bin sigma — pipeline vs DR1 (same basis)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    comp_label = args.component.replace("C_", "")
    proj_desc = ("Hankel + DR1 window" if args.source == "bundle"
                 else "Hankel only (DR1 EZmock cov source)")
    fig.suptitle(
        f"DR1 vs pipeline covariance in $\\xi_\\mathrm{{obs}}$ basis "
        f"(pipeline P-cov projected through {proj_desc}) — "
        f"{args.tracer} [pipeline component: {args.component}]",
        fontsize=12,
    )

    out_suffix = "" if args.component == "C_total" else f"_{comp_label}"
    if args.source != "ezmock":
        out_suffix += f"_{args.source}"
    if args.bandpower_sigma_dk > 0:
        out_suffix += f"_bp{args.bandpower_sigma_dk:g}"
    if args.kmax_cov is not None:
        out_suffix += f"_kmax{args.kmax_cov:g}"
    if args.s_range is not None:
        out_suffix += f"_s{args.s_range[0]:g}-{args.s_range[1]:g}"
    out = args.output or f"compare_rigorous_{args.tracer}{out_suffix}.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
