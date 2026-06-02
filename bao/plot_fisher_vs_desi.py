"""Money plot — BAO σ: Fisher + MCMC vs DESI bao-recon (DR1), EITHER frame.

Three stacked panels (σ_DH/rd, σ_DM/rd, σ_DV/rd) sharing the tracer x-axis.
Per (tracer, quantity), up to five markers:

  - black x        : DESI bao-recon σ (post-marg α-cov × OUR fid (D/rd))
  - blue circle    : Fisher, analytic cov
  - blue triangle  : MCMC,   analytic cov
  - orange square  : Fisher, bundle cov   (DESI DR1 RascalC ξ-cov substituted)
  - orange diamond : MCMC,   bundle cov

`--space` selects the frame; the four model series are computed in that frame:

  config  (default) : ξ(s) Grieb Gaussian cov (analytic) vs DESI bundle ξ-cov.
                      Fisher via config_space.bundle_fisher_sigmas (live);
                      MCMC from xi_money_mcmc.json (prefers seed_sweep_xi/).
  fourier            : P(k) analytic Gaussian Fisher (analytic) vs the same
                      bundle ξ-cov substituted as a Fourier precision (bundle).
                      Fisher live; MCMC from mcmc_results_fourier/.

DESI publishes DV for sparse tracers (qiso: BGS, QSO) and (DH, DM) for multipole
tracers (qparqper: LRG/ELG); the unused channels are masked. Reference σ is the
per-tracer bao-recon stat-only h5 (NOT desi_data.csv — see reference_sigmas.py).

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python plot_fisher_vs_desi.py --space config
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
_DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
_QUANTITIES = [
    ("DH_over_rs", r"$\sigma(D_H/r_d)$"),
    ("DM_over_rs", r"$\sigma(D_M/r_d)$"),
    ("DV_over_rs", r"$\sigma(D_V/r_d)$"),
]
_QUANT_SHORT = {"DH_over_rs": "DH/rd", "DM_over_rs": "DM/rd", "DV_over_rs": "DV/rd"}
_MCMC_KEY = {"DH_over_rs": "DH", "DM_over_rs": "DM", "DV_over_rs": "DV"}
_FID = {"Om": 0.3152, "hrdrag": 99.08}
_AREA = 7500.0
_HERE = Path(__file__).resolve().parent


def _is_sparse(tracer):
    return tracer in core._ISO_TRACERS


# ---------------------------------------------------------------------------
# MCMC loaders (one per frame)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Fourier bundle-cov Fisher: substitute the DESI bundle ξ-cov as the Fourier
# likelihood precision (precision = Mᵀ cov_ξ⁻¹ M, M = W·H_Hankel — same path as
# mcmc_fourier.py --cov bundle), then take the marginalized q-Fisher.
# ---------------------------------------------------------------------------
def _fourier_bundle_fisher(tracer):
    from mcmc_fourier import _load_bundle, _build_M, _precision_from_cov_xi

    cfg = TRACER_CONFIGS[tracer]
    apmode = "qiso" if _is_sparse(tracer) else "qparqper"
    theta, hrdrag = core._to_bao_cosmo_params({**core.PARAM_DEFAULTS, **_FID})
    info = core.build_bao_likelihood(
        N_tracers=rs._get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=_AREA, apmode=apmode)
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


# ---------------------------------------------------------------------------
# Per-frame gather → {tracer: {recon, fisher_ana, mcmc_ana, fisher_bun,
#                              mcmc_bun, *_err, sparse}}
# ---------------------------------------------------------------------------
def _gather_config():
    mcmc_raw = json.loads((_HERE / "xi_money_mcmc.json").read_text())
    out = {}
    for t in _TRACERS:
        print(f"== {t} (config) ==", flush=True)
        cfg = TRACER_CONFIGS[t]
        apmode = "qiso" if _is_sparse(t) else "qparqper"
        theta, hrdrag = core._to_bao_cosmo_params({**core.PARAM_DEFAULTS, **_FID})
        info = core.build_bao_likelihood(
            N_tracers=cc._get_ntracers(t), theta_cosmo=theta, hrdrag=hrdrag,
            tracer_bin=t, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
            area=_AREA, apmode=apmode)
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


# ---------------------------------------------------------------------------
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


def _plot(data, out_path, space):
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", choices=["config", "fourier"], default="config",
                    help="Covariance frame for the four model series.")
    args = ap.parse_args()

    data = _gather_config() if args.space == "config" else _gather_fourier()
    _print_table(data, args.space)
    _plot(data, _HERE / f"money_{args.space}_dr1.png", args.space)


if __name__ == "__main__":
    main()
