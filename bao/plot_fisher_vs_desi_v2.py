"""Money plot v2 — Fisher vs DESI bao-recon (DR1).

Three markers per (tracer, quantity):
  - x     : DESI bao-recon σ (converted with OUR fid (D/rd) for consistency)
  - dot   : production Fisher (analytic Gaussian-only cov)
  - square: bundle-cov Fisher  (DESI bundle ξ-cov substituted in)

DESI reference is the per-tracer bao-recon h5 stat-only file (post-marg α-cov
× our (D/rd)_fid), NOT desi_data.csv. This removes the ~5-10% fiducial-
conversion mismatch that affects BGS / QSO in csv space, and gives an
apples-to-apples comparison.

The dot-vs-square gap visualizes the impact of cov shape: dot = analytic
Gaussian-only, square = DESI bundle (RascalC + window). All other pipeline
choices (theory model, BB basis, Σ priors, b1) are held fixed.

Style follows plot_fisher_vs_desi.py: DH/rd blue, DM/rd orange, DV/rd green.

Usage (from bao/):
    ~/miniconda3/envs/emulator/bin/python plot_fisher_vs_desi_v2.py
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

# Reuse the heavy lifting from the unified comparison script.
from plot_fisher_vs_desi_full import (
    _prod_fisher_sigmas, _bundle_fisher_sigmas, _recon_sigmas, _TRACERS,
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


def _gather():
    """Run all three sigma sources for all 6 DR1 tracers."""
    out = {}
    for t in _TRACERS:
        print(f"== {t} ==", flush=True)
        prod = _prod_fisher_sigmas(t)
        bundle = _bundle_fisher_sigmas(t)
        recon = _recon_sigmas(t, prod)
        # Mask DH/DM for sparse tracers — bao-recon only publishes qiso → DV.
        is_sparse = bundle.get("apmode") == "qiso"
        if is_sparse:
            for q in ("DH_over_rs", "DM_over_rs"):
                recon[q] = float("nan")
                # We still report bundle/prod σ(DH) and σ(DM) but suppress them
                # from the plot since there's no DESI reference for sparse DH/DM.
        out[t] = {"prod": prod, "bundle": bundle, "recon": recon, "sparse": is_sparse}
        print(f"  prod   DH={prod['DH_over_rs']:.4f}  DM={prod['DM_over_rs']:.4f}  DV={prod['DV_over_rs']:.4f}")
        print(f"  bundle DH={bundle['DH_over_rs']:.4f}  DM={bundle['DM_over_rs']:.4f}  DV={bundle['DV_over_rs']:.4f}")
        print(f"  recon  DH={recon['DH_over_rs']:.4f}  DM={recon['DM_over_rs']:.4f}  DV={recon['DV_over_rs']:.4f}")
    return out


def _print_table(data):
    print(f"\n=== DR1 σ: production Fisher (P), bundle-cov Fisher (B) vs DESI bao-recon (D) ===")
    header = f"  {'tracer':<12} {'q':<6} {'σ_P':>9} {'σ_B':>9} {'σ_D':>9} {'P/D':>7} {'B/D':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in _TRACERS:
        d = data[t]
        for q, _, _ in _QUANTITIES:
            sD = d["recon"][q]
            if not np.isfinite(sD):
                continue
            sP = d["prod"][q]
            sB = d["bundle"][q]
            rP = sP / sD if sD > 0 else np.nan
            rB = sB / sD if sD > 0 else np.nan
            print(f"  {t:<12} {_QUANT_SHORT[q]:<6} {sP:>9.4f} {sB:>9.4f} {sD:>9.4f} {rP:>7.3f} {rB:>7.3f}")


def _plot(data, out_path):
    tracers = list(_TRACERS)
    display = [_DISPLAY.get(t, t) for t in tracers]
    x = np.arange(len(tracers), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6.0), constrained_layout=True)
    for xi in x:
        ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)

    for q, color, _ in _QUANTITIES:
        p_xs, p_vs, b_xs, b_vs, d_xs, d_vs = [], [], [], [], [], []
        for i, t in enumerate(tracers):
            sD = data[t]["recon"][q]
            if not np.isfinite(sD):
                continue
            sP = data[t]["prod"][q]
            sB = data[t]["bundle"][q]
            d_xs.append(i); d_vs.append(sD)
            if np.isfinite(sP):
                p_xs.append(i); p_vs.append(sP)
            if np.isfinite(sB):
                b_xs.append(i); b_vs.append(sB)
        ax.scatter(d_xs, d_vs, marker="x", s=110, color=color,
                   linewidths=2.5, zorder=3)
        ax.scatter(p_xs, p_vs, marker="o", s=80, color=color,
                   edgecolor="black", linewidth=0.5, zorder=4)
        ax.scatter(b_xs, b_vs, marker="s", s=80, color=color,
                   edgecolor="black", linewidth=0.5, zorder=4)

    source_handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=11,
               markeredgewidth=2.5, color="gray", label="DESI (bao-recon)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="gray", markeredgecolor="black", label="Fisher (analytic cov)"),
        Line2D([0], [0], marker="s", linestyle="", markersize=9,
               markerfacecolor="gray", markeredgecolor="black", label="Fisher (bundle cov)"),
    ]
    quantity_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor=color, markeredgecolor="none", label=label)
        for _, color, label in _QUANTITIES
    ]
    leg_src = ax.legend(handles=source_handles, loc="upper left",
                        bbox_to_anchor=(0.0, 1.0),
                        title="Source", fontsize=10, title_fontsize=10)
    ax.add_artist(leg_src)
    ax.legend(handles=quantity_handles, loc="upper left",
              bbox_to_anchor=(0.0, 0.74),
              title="Quantity", fontsize=10, title_fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=20)
    ax.set_xlabel("Tracer bin")
    ax.set_ylabel(r"$\sigma$")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")
    ax.set_title(
        f"BAO σ — Fisher vs DESI bao-recon  |  DR1  |  Om={_FID_OM:.4g}, hrdrag={_FID_HRDRAG:.4g}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    data = _gather()
    _print_table(data)
    out_path = Path(__file__).resolve().parent / f"fisher_vs_desi_v2_dr1_Om_{_FID_OM:.5f}_hrdrag_{_FID_HRDRAG:.5f}.png"
    _plot(data, out_path)


if __name__ == "__main__":
    main()
