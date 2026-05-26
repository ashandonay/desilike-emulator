"""Money plot v2 — Fisher + MCMC vs DESI bao-recon (DR1).

Five markers per (tracer, quantity):
  - x          : DESI bao-recon σ (post-marg α-cov × OUR fid (D/rd))
  - filled dot : production Fisher (analytic Gaussian-only cov)
  - open dot   : production MCMC   (analytic cov, posterior σ)
  - filled sq  : bundle-cov Fisher (DESI bundle ξ-cov substituted in)
  - open sq    : bundle-cov MCMC   (bundle ξ-cov, posterior σ)

DESI reference is the per-tracer bao-recon h5 stat-only file. This removes
the ~5-10% fiducial-conversion mismatch in desi_data.csv space.

The open-vs-filled gap (same shape) visualizes Fisher-vs-MCMC posterior-width
gap. The square-vs-dot gap (same fill style) visualizes analytic-vs-bundle
cov-shape impact. Per §33t, the open markers (MCMC) close to ±7% of DESI
on all 10 real channels.

DESI publishes DV for sparse tracers (qiso: BGS, QSO) and (DH, DM) for
multipole tracers (qparqper: LRG/ELG); we mask the unused channels.

Usage (from bao/):
    ~/miniconda3/envs/emulator/bin/python plot_fisher_vs_desi_v2.py
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from plot_fisher_vs_desi_full import (
    _prod_fisher_sigmas, _bundle_fisher_sigmas, _recon_sigmas, _TRACERS,
    _profile_sigmas,
)
from util import TRACER_CONFIGS

_DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
_QUANTITIES = [
    ("DH_over_rs", "tab:blue",   r"$\sigma(D_H/r_d)$"),
    ("DM_over_rs", "tab:orange", r"$\sigma(D_M/r_d)$"),
    ("DV_over_rs", "tab:green",  r"$\sigma(D_V/r_d)$"),
]
_QUANT_SHORT = {"DH_over_rs": "DH/rd", "DM_over_rs": "DM/rd", "DV_over_rs": "DV/rd"}
_FID_OM, _FID_HRDRAG = 0.3152, 99.08
_HERE = Path(__file__).resolve().parent


def _load_prod_mcmc(tracer):
    """Production MCMC (analytic cov). Schema: sigma_*_over_rd_mcmc."""
    p = _HERE / "mcmc_results" / f"{tracer}_dr1.json"
    if not p.exists():
        return {q: float("nan") for q, _, _ in _QUANTITIES}
    d = json.loads(p.read_text())
    return {
        "DH_over_rs": float(d.get("sigma_DH_over_rd_mcmc", float("nan"))),
        "DM_over_rs": float(d.get("sigma_DM_over_rd_mcmc", float("nan"))),
        "DV_over_rs": float(d.get("sigma_DV_over_rd_mcmc", float("nan"))),
    }


def _load_bundle_mcmc(tracer):
    """Bundle-cov MCMC (native ξ-theory + bundle cov + BB-DESI). Schema: sig_*."""
    suffix = "_qiso" if tracer in ("BGS", "QSO") else ""
    p = _HERE / "mcmc_results_bundle_native" / f"{tracer}_dr1_native_theory{suffix}_bbdesi_adame24.json"
    if not p.exists():
        return {q: float("nan") for q, _, _ in _QUANTITIES}
    d = json.loads(p.read_text())
    return {
        "DH_over_rs": float(d.get("sig_DH", float("nan"))),
        "DM_over_rs": float(d.get("sig_DM", float("nan"))),
        "DV_over_rs": float(d.get("sig_DV", float("nan"))),
    }


def _gather():
    out = {}
    for t in _TRACERS:
        print(f"== {t} ==", flush=True)
        prod = _prod_fisher_sigmas(t)
        bundle = _bundle_fisher_sigmas(t)
        recon = _recon_sigmas(t, prod)
        prod_mcmc = _load_prod_mcmc(t)
        bundle_mcmc = _load_bundle_mcmc(t)
        profile = _profile_sigmas(t)

        is_sparse = bundle.get("apmode") == "qiso"
        if is_sparse:
            for src in (recon, prod, bundle, prod_mcmc, bundle_mcmc, profile):
                src["DH_over_rs"] = float("nan")
                src["DM_over_rs"] = float("nan")
        else:
            for src in (recon, prod, bundle, prod_mcmc, bundle_mcmc, profile):
                src["DV_over_rs"] = float("nan")

        out[t] = {"prod": prod, "bundle": bundle, "recon": recon,
                  "prod_mcmc": prod_mcmc, "bundle_mcmc": bundle_mcmc,
                  "profile": profile, "sparse": is_sparse}
        for src_name, src in [("prod", prod), ("bundle", bundle), ("recon", recon),
                              ("prod_mcmc", prod_mcmc), ("bundle_mcmc", bundle_mcmc),
                              ("profile", profile)]:
            print(f"  {src_name:<12} DH={src['DH_over_rs']:.4f}  DM={src['DM_over_rs']:.4f}  DV={src['DV_over_rs']:.4f}")
    return out


def _print_table(data):
    print(f"\n=== DR1 σ: Fisher (P/B) + MCMC (Pm/Bm) + Profile (Pf) vs DESI bao-recon (D) ===")
    header = (f"  {'tracer':<12} {'q':<6} {'σ_P':>8} {'σ_Pm':>8} {'σ_B':>8} "
              f"{'σ_Bm':>8} {'σ_Pf':>8} {'σ_D':>8} "
              f"{'P/D':>6} {'Pm/D':>6} {'B/D':>6} {'Bm/D':>6} {'Pf/D':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in _TRACERS:
        d = data[t]
        for q, _, _ in _QUANTITIES:
            sD = d["recon"][q]
            if not np.isfinite(sD): continue
            sP  = d["prod"][q]; sB = d["bundle"][q]
            sPm = d["prod_mcmc"][q]; sBm = d["bundle_mcmc"][q]
            sPf = d["profile"][q]
            r = lambda x: x / sD if sD > 0 else np.nan
            print(f"  {t:<12} {_QUANT_SHORT[q]:<6} "
                  f"{sP:>8.4f} {sPm:>8.4f} {sB:>8.4f} {sBm:>8.4f} {sPf:>8.4f} {sD:>8.4f} "
                  f"{r(sP):>6.3f} {r(sPm):>6.3f} {r(sB):>6.3f} {r(sBm):>6.3f} {r(sPf):>6.3f}")


def _plot(data, out_path):
    tracers = list(_TRACERS)
    display = [_DISPLAY.get(t, t) for t in tracers]
    x = np.arange(len(tracers), dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for xi in x:
        ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)

    for q, color, _ in _QUANTITIES:
        # Build per-source lists
        sources = {"recon": [], "prod": [], "bundle": [], "prod_mcmc": [],
                   "bundle_mcmc": [], "profile": []}
        for i, t in enumerate(tracers):
            for s in sources:
                v = data[t][s][q]
                if np.isfinite(v):
                    sources[s].append((i, v))

        def _xy(name):
            if not sources[name]: return [], []
            xs, ys = zip(*sources[name])
            return list(xs), list(ys)

        # DESI bao-recon (cross)
        dx, dy = _xy("recon")
        ax.scatter(dx, dy, marker="x", s=110, color=color,
                   linewidths=2.5, zorder=5)
        # Production Fisher (filled circle)
        px, py = _xy("prod")
        ax.scatter(px, py, marker="o", s=70, color=color,
                   edgecolor="black", linewidth=0.5, zorder=4)
        # Production MCMC (open circle, slightly larger)
        pmx, pmy = _xy("prod_mcmc")
        ax.scatter(pmx, pmy, marker="o", s=130, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=4)
        # Bundle Fisher (filled square)
        bx, by = _xy("bundle")
        ax.scatter(bx, by, marker="s", s=70, color=color,
                   edgecolor="black", linewidth=0.5, zorder=4)
        # Bundle MCMC (open square, slightly larger)
        bmx, bmy = _xy("bundle_mcmc")
        ax.scatter(bmx, bmy, marker="s", s=130, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=4)
        # Chi² profile (small "+") — bundle cov, nonlinear curvature
        prx, pry = _xy("profile")
        ax.scatter(prx, pry, marker="+", s=120, color=color,
                   linewidths=2.0, zorder=6)

    source_handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=11,
               markeredgewidth=2.5, color="gray", label="DESI (bao-recon)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor="gray", markeredgecolor="black",
               label="Fisher (analytic cov)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=11,
               markerfacecolor="none", markeredgecolor="gray", markeredgewidth=1.8,
               label="MCMC (analytic cov)"),
        Line2D([0], [0], marker="s", linestyle="", markersize=8,
               markerfacecolor="gray", markeredgecolor="black",
               label="Fisher (bundle cov)"),
        Line2D([0], [0], marker="s", linestyle="", markersize=11,
               markerfacecolor="none", markeredgecolor="gray", markeredgewidth=1.8,
               label="MCMC (bundle cov)"),
        Line2D([0], [0], marker="+", linestyle="", markersize=11,
               markeredgewidth=2.0, color="gray",
               label=r"$\chi^2$ profile (bundle, nonlin)"),
    ]
    quantity_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor=color, markeredgecolor="none", label=label)
        for _, color, label in _QUANTITIES
    ]
    leg_src = ax.legend(handles=source_handles, loc="upper left",
                        bbox_to_anchor=(0.0, 1.0),
                        title="Source", fontsize=9, title_fontsize=10)
    ax.add_artist(leg_src)
    ax.legend(handles=quantity_handles, loc="upper left",
              bbox_to_anchor=(0.0, 0.62),
              title="Quantity", fontsize=9, title_fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=20)
    ax.set_xlabel("Tracer bin")
    ax.set_ylabel(r"$\sigma$")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")
    ax.set_title(
        f"BAO σ — Fisher + MCMC vs DESI bao-recon  |  DR1  |  "
        f"Om={_FID_OM:.4g}, hrdrag={_FID_HRDRAG:.4g}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    data = _gather()
    _print_table(data)
    out_path = _HERE / f"fisher_vs_desi_v2_dr1_Om_{_FID_OM:.5f}_hrdrag_{_FID_HRDRAG:.5f}.png"
    _plot(data, out_path)


if __name__ == "__main__":
    main()
