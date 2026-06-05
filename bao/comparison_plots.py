"""Unified BAO comparison plots — forecast σ, ξ-covariance, and theory ξ.

Three plots, selected by a positional subcommand (default: forecast):

  comparison_plots.py [forecast]   Fisher + MCMC σ vs DESI bao-recon, EITHER frame
                                   (--space config|fourier). Three stacked panels
                                   σ(DH/rd, DM/rd, DV/rd); markers = estimator×cov.
  comparison_plots.py cov          Grieb Gaussian vs DESI bundle RascalC covariance
                                   (--matrix corr|cov|both). 3×N grid: Gaussian /
                                   bundle / (Gaussian − bundle).
  comparison_plots.py theory       pipeline native ξ-theory vs bundle data (2×3).
                                   s²·ξ_ℓ: data, theory (BB=0), theory + best-fit BB.

All three share the single fiducial config_space._FID and the validated pipeline
functions (build_bao_likelihood, gaussxi_cov_on_bundle_grid, bundle_fisher_sigmas,
build_native_theory_mcmc / _predict). For the α_SN covariance analysis see
alpha_sn_check.py.

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python comparison_plots.py            # forecast
    ... comparison_plots.py forecast --space fourier
    ... comparison_plots.py cov --matrix both
    ... comparison_plots.py theory
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import core
import config_space as cc
import fourier_space
import reference_sigmas as rs
from util import TRACER_CONFIGS

_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_HERE = Path(__file__).resolve().parent

# forecast
_DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
_QUANTITIES = [
    ("DH_over_rs", r"$\sigma(D_H/r_d)$"),
    ("DM_over_rs", r"$\sigma(D_M/r_d)$"),
    ("DV_over_rs", r"$\sigma(D_V/r_d)$"),
]
_QUANT_SHORT = {"DH_over_rs": "DH/rd", "DM_over_rs": "DM/rd", "DV_over_rs": "DV/rd"}
_MCMC_KEY = {"DH_over_rs": "DH", "DM_over_rs": "DM", "DV_over_rs": "DV"}
# theory
_BB_POWERS = (0, 2)  # DESI Adame+24 broadband basis (matches the forecast BB)


def _is_sparse(tracer):
    return tracer in core._ISO_TRACERS


# ===========================================================================
# forecast — Fisher + MCMC σ vs DESI bao-recon (config or fourier frame)
# ===========================================================================
def _mcmc_config(tracer, mcmc_raw, which):
    """Config-space MCMC σ. Prefer the seed sweep (mean ± σ-of-σ); fall back to
    the single-seed cache (err=0). which ∈ {gaussxi (analytic), bundle}."""
    sw = _HERE / "seed_sweep_xi" / f"{tracer}.json"
    if sw.exists():
        per_seed = json.loads(sw.read_text())[which]   # {seed: {DH,DM,DV}}
        point, err = {}, {}
        for q, _ in _QUANTITIES:
            vals = [s[_MCMC_KEY[q]] for s in per_seed.values()
                    if s.get(_MCMC_KEY[q]) is not None and np.isfinite(s.get(_MCMC_KEY[q]))]
            point[q] = float(np.mean(vals)) if vals else float("nan")
            err[q] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return point, err
    d = mcmc_raw.get(tracer, {}).get(which, {})
    point = {q: float(d.get(_MCMC_KEY[q], float("nan"))) for q, _ in _QUANTITIES}
    return point, {q: 0.0 for q, _ in _QUANTITIES}


def _mcmc_fourier(tracer, cov):
    """Fourier-space MCMC σ from mcmc_results_fourier/. cov ∈ {analytic, bundle}."""
    suffix = "_qiso" if _is_sparse(tracer) else ""
    p = _HERE / "mcmc_results_fourier" / f"{tracer}_dr1_{cov}{suffix}.json"
    if not p.exists():
        return {q: float("nan") for q, _ in _QUANTITIES}, {q: 0.0 for q, _ in _QUANTITIES}
    d = json.loads(p.read_text())
    point = {
        "DH_over_rs": float(d.get("sigma_DH_over_rd_mcmc", float("nan"))),
        "DM_over_rs": float(d.get("sigma_DM_over_rd_mcmc", float("nan"))),
        "DV_over_rs": float(d.get("sigma_DV_over_rd_mcmc", float("nan"))),
    }
    return point, {q: 0.0 for q, _ in _QUANTITIES}


def _fourier_bundle_fisher(tracer):
    """Fisher with the DESI bundle ξ-cov substituted as the Fourier precision
    (precision = Mᵀ cov_ξ⁻¹ M, M = W·H_Hankel — same path as mcmc_fourier --cov bundle)."""
    from mcmc_fourier import _load_bundle, _build_M, _precision_from_cov_xi

    cfg = TRACER_CONFIGS[tracer]
    apmode = "qiso" if _is_sparse(tracer) else "qparqper"
    theta, hrdrag = core._to_bao_cosmo_params({**core.PARAM_DEFAULTS, **cc._FID})
    info = core.build_bao_likelihood(
        N_tracers=rs._get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=cc._AREA, apmode=apmode)
    lik = info["likelihood"]
    lik(**info["params"])

    obs = info["observable"]
    cov_xi, W, th_ells, th_s = _load_bundle(tracer)
    k = np.asarray(obs.k[0], dtype=np.float64)
    dk = float(np.mean(np.diff(k)))
    pipe_ells = [int(l) for l in obs.ells]
    M = _build_M(W, th_ells, th_s, pipe_ells, k, dk)
    prec = _precision_from_cov_xi(cov_xi, M)
    lik.precision = np.asarray(prec, dtype=np.float64)
    if hasattr(lik, "_precision_input"):
        lik._precision_input = lik.precision.copy()
    if hasattr(lik, "precision_hartlap2007"):
        lik.precision_hartlap2007 = lik.precision.copy()

    F_q = fourier_space._q_fisher_from_bao_likelihood_info(info)
    try:
        cov_q = np.linalg.inv(F_q)
    except np.linalg.LinAlgError:
        cov_q = np.linalg.pinv(F_q, rcond=1e-12)

    tmpl = info["template"]
    DH, DM, DV = (float(tmpl.DH_over_rd_fid), float(tmpl.DM_over_rd_fid),
                  float(tmpl.DV_over_rd_fid))
    J = np.array([[DH], [DM]]) if apmode == "qiso" else np.diag([DH, DM])
    cov_phys = J @ cov_q @ J.T
    s = fourier_space._cov_to_sigma_triplet({
        "cov_DH_over_rd_DH_over_rd": cov_phys[0, 0],
        "cov_DH_over_rd_DM_over_rd": cov_phys[0, 1],
        "cov_DM_over_rd_DM_over_rd": cov_phys[1, 1],
        "DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV})
    return {"DH_over_rs": s[0], "DM_over_rs": s[1], "DV_over_rs": s[2],
            "DH_over_rd_fid": DH, "DM_over_rd_fid": DM, "DV_over_rd_fid": DV}


def _gather_config():
    mcmc_raw = json.loads((_HERE / "mcmc_config.json").read_text())
    out = {}
    for t in _TRACERS:
        print(f"== {t} (config) ==", flush=True)
        cfg = TRACER_CONFIGS[t]
        apmode = "qiso" if _is_sparse(t) else "qparqper"
        theta, hrdrag = core._to_bao_cosmo_params({**core.PARAM_DEFAULTS, **cc._FID})
        info = core.build_bao_likelihood(
            N_tracers=cc._get_ntracers(t), theta_cosmo=theta, hrdrag=hrdrag,
            tracer_bin=t, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
            area=cc._AREA, apmode=apmode)
        info["likelihood"](**info["params"])
        bundle = cc.load_bundle(t)

        fisher_bun = cc.bundle_fisher_sigmas(t)
        _, Cg = cc.gaussxi_cov_on_bundle_grid(t, info, bundle)
        fisher_ana = cc.bundle_fisher_sigmas(t, cov_override=Cg)
        recon = rs._recon_sigmas(t, fisher_bun)
        mcmc_ana, mcmc_ana_err = _mcmc_config(t, mcmc_raw, "gaussxi")
        mcmc_bun, mcmc_bun_err = _mcmc_config(t, mcmc_raw, "bundle")
        out[t] = _assemble(t, recon, fisher_ana, mcmc_ana, fisher_bun,
                           mcmc_bun, mcmc_ana_err, mcmc_bun_err)
    return out


def _gather_fourier():
    out = {}
    for t in _TRACERS:
        print(f"== {t} (fourier) ==", flush=True)
        fisher_ana = rs._prod_fisher_sigmas(t)
        recon = rs._recon_sigmas(t, fisher_ana)
        fisher_bun = _fourier_bundle_fisher(t)
        mcmc_ana, mcmc_ana_err = _mcmc_fourier(t, "analytic")
        mcmc_bun, mcmc_bun_err = _mcmc_fourier(t, "bundle")
        out[t] = _assemble(t, recon, fisher_ana, mcmc_ana, fisher_bun,
                           mcmc_bun, mcmc_ana_err, mcmc_bun_err)
    return out


def _assemble(t, recon, fisher_ana, mcmc_ana, fisher_bun, mcmc_bun,
              mcmc_ana_err, mcmc_bun_err):
    sparse = _is_sparse(t)
    mask = ["DH_over_rs", "DM_over_rs"] if sparse else ["DV_over_rs"]
    for src in (recon, fisher_ana, mcmc_ana, fisher_bun, mcmc_bun,
                mcmc_ana_err, mcmc_bun_err):
        for q in mask:
            src[q] = float("nan")
    d = {"recon": recon, "fisher_ana": fisher_ana, "mcmc_ana": mcmc_ana,
         "fisher_bun": fisher_bun, "mcmc_bun": mcmc_bun,
         "mcmc_ana_err": mcmc_ana_err, "mcmc_bun_err": mcmc_bun_err,
         "sparse": sparse}
    for nm in ("recon", "fisher_ana", "mcmc_ana", "fisher_bun", "mcmc_bun"):
        s = d[nm]
        print(f"  {nm:<11} DH={s['DH_over_rs']:.4f}  DM={s['DM_over_rs']:.4f}  DV={s['DV_over_rs']:.4f}")
    return d


def _print_table(data, space):
    print(f"\n=== DR1 {space}-space σ: Fisher(Fa/Fb) + MCMC(Ma/Mb) vs DESI (D) ===")
    header = (f"  {'tracer':<11} {'q':<6} {'σ_Fa':>8} {'σ_Ma':>8} {'σ_Fb':>8} "
              f"{'σ_Mb':>8} {'σ_D':>8} {'Fa/D':>6} {'Ma/D':>6} {'Fb/D':>6} {'Mb/D':>6}")
    print(header); print("  " + "-" * (len(header) - 2))
    for t in _TRACERS:
        d = data[t]
        for q, _ in _QUANTITIES:
            sD = d["recon"][q]
            if not np.isfinite(sD):
                continue
            sFa, sMa = d["fisher_ana"][q], d["mcmc_ana"][q]
            sFb, sMb = d["fisher_bun"][q], d["mcmc_bun"][q]
            r = lambda x: x / sD if sD > 0 else np.nan
            print(f"  {t:<11} {_QUANT_SHORT[q]:<6} {sFa:>8.4f} {sMa:>8.4f} "
                  f"{sFb:>8.4f} {sMb:>8.4f} {sD:>8.4f} "
                  f"{r(sFa):>6.3f} {r(sMa):>6.3f} {r(sFb):>6.3f} {r(sMb):>6.3f}")


def _plot_forecast(data, out_path, space):
    tracers = list(_TRACERS)
    display = [_DISPLAY.get(t, t) for t in tracers]
    x = np.arange(len(tracers), dtype=float)
    c_ana, c_bun = "tab:blue", "tab:orange"

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             constrained_layout=True)
    for ax, (q, ylabel) in zip(axes, _QUANTITIES):
        for xi in x:
            ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)

        def _xy(src):
            xs, ys = [], []
            for i, t in enumerate(tracers):
                v = data[t][src][q]
                if np.isfinite(v):
                    xs.append(i); ys.append(v)
            return xs, ys

        def _err(src, xs):
            ek = src + "_err"
            return [data[tracers[i]][ek][q] for i in xs]

        # Shape = estimator (circle/square Fisher, triangle/diamond MCMC);
        # color = cov source (blue analytic, orange bundle).
        ax.scatter(*_xy("recon"), marker="x", s=40, color="black",
                   linewidths=1.5, zorder=5)
        ax.scatter(*_xy("fisher_ana"), marker="o", s=28, color=c_ana,
                   linewidth=0, zorder=4)
        ax.scatter(*_xy("fisher_bun"), marker="s", s=24, color=c_bun,
                   linewidth=0, zorder=4)
        for src, mk, ms in (("mcmc_ana", "^", 32), ("mcmc_bun", "D", 22)):
            mx, my = _xy(src)
            me = _err(src, mx)
            if any(e > 0 for e in me):
                ax.errorbar(mx, my, yerr=me, fmt="none",
                            ecolor=c_ana if src == "mcmc_ana" else c_bun,
                            elinewidth=1.3, capsize=4, capthick=1.3, zorder=3)
            ax.scatter(mx, my, marker=mk, s=ms,
                       color=c_ana if src == "mcmc_ana" else c_bun,
                       linewidth=0, zorder=4)

        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, None)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")

    handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=6, markeredgewidth=1.5,
               color="black", label="DESI (bao-recon)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=c_ana, markeredgecolor="none", label="Fisher · analytic cov"),
        Line2D([0], [0], marker="^", linestyle="", markersize=7,
               markerfacecolor=c_ana, markeredgecolor="none", label="MCMC · analytic cov"),
        Line2D([0], [0], marker="s", linestyle="", markersize=7,
               markerfacecolor=c_bun, markeredgecolor="none", label="Fisher · bundle cov"),
        Line2D([0], [0], marker="D", linestyle="", markersize=6,
               markerfacecolor=c_bun, markeredgecolor="none", label="MCMC · bundle cov"),
    ]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   title=f"Source ({space}-space)", fontsize=9, title_fontsize=10)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(display, rotation=20)
    axes[-1].set_xlabel("Tracer bin")
    axes[0].set_title(
        f"{space.capitalize()}-space BAO σ — Fisher + MCMC, analytic vs bundle cov, "
        f"vs DESI bao-recon  |  DR1", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def run_forecast(space):
    data = _gather_config() if space == "config" else _gather_fourier()
    _print_table(data, space)
    _plot_forecast(data, _HERE / f"forecast_comparison_{space}_dr1.png", space)


# ===========================================================================
# cov — analytic Grieb Gaussian vs DESI bundle (RascalC) covariance
# ===========================================================================
def _corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    return C / np.outer(d, d)


def _covs(tracer):
    """(Cg, Cb): Grieb Gaussian and DESI bundle cov on the bundle observable grid."""
    cfg = TRACER_CONFIGS[tracer]
    apmode = "qiso" if _is_sparse(tracer) else "qparqper"
    theta, hrdrag = core._to_bao_cosmo_params({**core.PARAM_DEFAULTS, **cc._FID})
    info = core.build_bao_likelihood(
        N_tracers=cc._get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=cc._AREA, apmode=apmode, lean=True)
    bundle = cc.load_bundle(tracer)
    Cb = 0.5 * (np.asarray(bundle["cov"]) + np.asarray(bundle["cov"]).T)
    _, Cg = cc.gaussxi_cov_on_bundle_grid(tracer, info, bundle)  # windowed, matches Cb grid
    Cg = 0.5 * (Cg + Cg.T)
    return Cg, Cb


def _plot_cov(covs, matrix, out_path):
    tracers = list(covs.keys())
    ncol = len(tracers)
    fig, axes = plt.subplots(3, ncol, figsize=(2.6 * ncol, 8.2),
                             squeeze=False, constrained_layout=True)
    row_labels = ["Grieb Gaussian", "DESI bundle (RascalC)", "Gaussian − bundle"]

    im = None
    for j, t in enumerate(tracers):
        Cg, Cb = covs[t]
        if matrix == "corr":
            A, B = _corr(Cg), _corr(Cb)
            scale = None
        else:  # cov: per-tracer normalization so the column is self-comparable
            scale = max(float(np.max(np.abs(Cg))), float(np.max(np.abs(Cb))), 1e-300)
            A, B = Cg / scale, Cb / scale
        for i, M in enumerate([A, B, A - B]):
            ax = axes[i, j]
            im = ax.imshow(M, cmap="seismic", vmin=-1, vmax=1, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(t if scale is None else f"{t}\nscale={scale:.1e}", fontsize=10)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=10)

    label = "correlation" if matrix == "corr" else "C / max|C| (per tracer)"
    fig.colorbar(im, ax=axes, shrink=0.5, label=label)
    fig.suptitle(
        f"BAO ξ-covariance: analytic Grieb Gaussian vs DESI bundle RascalC "
        f"(DR1, {matrix})", fontsize=13)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def run_cov(matrix, tracers):
    covs = {}
    for t in tracers:
        print(f"building covs for {t} ...", flush=True)
        covs[t] = _covs(t)
    for m in (["corr", "cov"] if matrix == "both" else [matrix]):
        _plot_cov(covs, m, _HERE / f"cov_comparison_{m}_dr1.png")


# ===========================================================================
# theory — pipeline native ξ-theory vs DESI bundle data
#
# The pipeline includes broadband (BB) as Schur-MARGINALIZED nuisance params,
# not fixed values. The dashed (BB=0) curve is the raw theory shape; the gap to
# the data is the broadband the pipeline marginalizes away. The solid curve adds
# the maximum-likelihood BB (the visual stand-in for marginalization). The
# reported χ²/dof is AFTER marginalizing BB — only the BAO/shape residual.
# ===========================================================================
def _predict_tracer(tracer):
    """(bundle, pred, b1): pipeline native ξ theory (BB=0) windowed to the bundle grid."""
    apmode = "qiso" if _is_sparse(tracer) else "qparqper"
    info, native, bundle = cc.build_native_theory_mcmc(tracer, apmode)
    pred = cc._predict(native, bundle, info["params"])
    return bundle, pred, float(info["params"]["b1"])


def _best_fit_bb(data, pred, C_inv, BB):
    A = BB.T @ C_inv @ BB
    b = BB.T @ C_inv @ (data - pred)
    return BB @ np.linalg.solve(A, b)


def _chi2_marg(data, pred, C_inv, BB):
    A = BB.T @ C_inv @ BB
    b = BB.T @ C_inv @ (data - pred)
    chi2_full = float((data - pred) @ C_inv @ (data - pred))
    return chi2_full - float(b @ np.linalg.solve(A, b))


def run_theory(tracers):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = {"0": "C0", "2": "C3", "4": "C2"}

    for i, tracer in enumerate(tracers):
        ax = axes[i]
        try:
            bundle, pred, b1 = _predict_tracer(tracer)
        except Exception as e:  # keep the grid intact if one tracer fails
            ax.text(0.5, 0.5, f"{tracer}\nFAILED: {e}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(tracer)
            continue

        data = np.concatenate(bundle["obs_data"])
        cov = 0.5 * (bundle["cov"] + bundle["cov"].T)
        err = np.sqrt(np.diag(cov))
        C_inv = np.linalg.inv(cov)
        BB = cc.bb_basis(bundle["obs_ells"], bundle["obs_s"], data.size, powers=_BB_POWERS)
        chi2_m = _chi2_marg(data, pred, C_inv, BB)
        dof = data.size - BB.shape[1]
        pred_bb = pred + _best_fit_bb(data, pred, C_inv, BB)

        idx = 0
        for j, ell in enumerate(bundle["obs_ells"]):
            s = np.asarray(bundle["obs_s"][j]); s2 = s ** 2
            n = len(s); c = colors[str(ell)]; sl = slice(idx, idx + n)
            ax.errorbar(s, s2 * data[sl], yerr=s2 * err[sl], fmt="o", ms=3, capsize=2,
                        color=c, alpha=0.8, zorder=3, label=f"data ℓ={ell}")
            ax.plot(s, s2 * pred[sl], "--", color=c, lw=1.5, alpha=0.7,
                    label=f"theory ℓ={ell}")
            ax.plot(s, s2 * pred_bb[sl], "-", color=c, lw=2, alpha=0.9,
                    label=f"theory+BB ℓ={ell}")
            idx += n

        ax.set_xlabel(r"$s$ [$h^{-1}$ Mpc]")
        ax.set_ylabel(r"$s^2 \xi(s)$ [$h^{-2}$ Mpc$^2$]")
        ax.set_title(f"{tracer}   b1 = {b1:.2f}\n"
                     f"χ²(BB-marg)/dof = {chi2_m:.1f}/{dof} = {chi2_m / dof:.2f}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

    fig.suptitle("Pipeline native ξ-theory vs DESI DR1 bundle data\n"
                 "(dashed = theory, BB=0; solid = theory + best-fit DESI broadband)",
                 fontsize=12)
    fig.tight_layout()
    out = _HERE / "theory_comparison_dr1.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    pf = sub.add_parser("forecast", help="Fisher + MCMC σ vs DESI bao-recon (default)")
    pf.add_argument("--space", choices=["config", "fourier"], default="config",
                    help="Covariance frame for the four model series.")

    pc = sub.add_parser("cov", help="Grieb Gaussian vs bundle RascalC covariance")
    pc.add_argument("--tracers", nargs="+", default=_TRACERS, choices=_TRACERS)
    pc.add_argument("--matrix", choices=["corr", "cov", "both"], default="corr",
                    help="correlation, per-tracer-normalized covariance, or both")

    pt = sub.add_parser("theory", help="pipeline native ξ-theory vs bundle data")
    pt.add_argument("--tracers", nargs="+", default=_TRACERS, choices=_TRACERS)

    args = ap.parse_args()
    cmd = args.command or "forecast"
    if cmd == "forecast":
        run_forecast(getattr(args, "space", "config"))
    elif cmd == "cov":
        run_cov(args.matrix, args.tracers)
    elif cmd == "theory":
        run_theory(args.tracers)


if __name__ == "__main__":
    main()
