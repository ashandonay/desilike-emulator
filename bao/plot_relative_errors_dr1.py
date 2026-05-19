"""Relative-error scatter for DR1.

For each pair of (tracer_i, quantity_i) and (tracer_j, quantity_j) where
DESI quotes a sigma, compute

    r_F = sigma_F_i / sigma_F_j        (Fisher pipeline)
    r_D = sigma_D_i / sigma_D_j        (DESI Y1 BAO measurement)

and scatter r_F (x) vs r_D (y). The y = x diagonal indicates that the
*relative* size of the errors between bins/quantities matches between
pipeline and data, even when absolute F/D ratios differ. Points above the
diagonal mean DESI's i-vs-j contrast is larger than ours; below means
DESI is more uniform across that pair.

DR1 only (project policy: don't iterate on DR2 until DR1 is settled).
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

warnings.filterwarnings("ignore")

from plot_fisher_vs_desi import (
    QUANTITIES, _fisher_stds, _get_desi_std,
)
from util import TRACER_CONFIGS


_DISPLAY_NAME = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lyα QSO"}
_NAME_MAP_TO_DATA = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
_QUANT_SHORT = {"DH_over_rs": "DH/rd",
                "DM_over_rs": "DM/rd",
                "DV_over_rs": "DV/rd"}
_QUANT_COLOR = {"DH_over_rs": "tab:blue",
                "DM_over_rs": "tab:orange",
                "DV_over_rs": "tab:green"}


def _collect_cells(fisher, desi):
    """Flatten to a list of (tracer, quantity, sigma_F, sigma_D) entries
    where both DESI and Fisher report a finite sigma."""
    cells = []
    for key in TRACER_CONFIGS:
        if not TRACER_CONFIGS[key].get("supported", True):
            continue
        for q in QUANTITIES:
            dv = desi.get(key, {}).get(q, np.nan)
            fv = fisher.get(key, {}).get(q, np.nan)
            if np.isfinite(dv) and np.isfinite(fv) and dv > 0 and fv > 0:
                cells.append((key, q, float(fv), float(dv)))
    return cells


def _label(tracer, quantity):
    return f"({_QUANT_SHORT[quantity]} {_DISPLAY_NAME.get(tracer, tracer)})"


def _plot(cells, omega_m, hrdrag, out_path, log_axes=True,
          annotate_outliers=True):
    pairs = list(combinations(cells, 2))
    xs, ys, colors, labels = [], [], [], []
    for (t_i, q_i, fi, di), (t_j, q_j, fj, dj) in pairs:
        xs.append(fi / fj)
        ys.append(di / dj)
        colors.append(_QUANT_COLOR[q_i])
        labels.append(f"{_label(t_i, q_i)} / {_label(t_j, q_j)}")

    xs = np.asarray(xs); ys = np.asarray(ys)
    fig, ax = plt.subplots(figsize=(9.0, 8.5), constrained_layout=True)

    if log_axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        lo = float(min(xs.min(), ys.min())) * 0.85
        hi = float(max(xs.max(), ys.max())) * 1.15
    else:
        lo = float(min(xs.min(), ys.min())) * 0.95
        hi = float(max(xs.max(), ys.max())) * 1.05
    diag = np.array([lo, hi])
    ax.plot(diag, diag, color="black", linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=1)

    ax.scatter(xs, ys, s=55, c=colors, edgecolor="black",
               linewidth=0.4, alpha=0.85, zorder=3)

    if annotate_outliers:
        log_resid = np.log(ys / xs)
        order = np.argsort(np.abs(log_resid))[::-1]
        n_label = min(6, len(order))
        for i in order[:n_label]:
            ax.annotate(labels[i], (xs[i], ys[i]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, alpha=0.85)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Fisher ratio $\sigma_i / \sigma_j$ (pipeline)")
    ax.set_ylabel(r"DESI ratio $\sigma_i / \sigma_j$ (Y1 BAO)")

    ax.set_title("DR1 Tracer Bin Relative-Error", fontsize=12)

    quant_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor=c, markeredgecolor="black",
               label=_QUANT_SHORT[q])
        for q, c in _QUANT_COLOR.items()
    ]
    diag_handles = [
        Line2D([0], [0], color="black", linestyle="--", label="y = x"),
    ]
    leg_quant = ax.legend(handles=quant_handles, loc="upper left",
                          fontsize=9, title="Numerator quantity",
                          title_fontsize=9)
    ax.add_artist(leg_quant)
    ax.legend(handles=diag_handles, loc="lower right", fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _print_summary(cells, fisher, desi):
    print(f"\nDR1 cells ({len(cells)}):")
    for (t, q, fv, dv) in cells:
        print(f"  {_label(t, q):<22} sigma_F={fv:.5f}  sigma_D={dv:.5f}  "
              f"F/D={fv/dv:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omega-m", type=float, default=0.3152)
    ap.add_argument("--hrdrag", type=float, default=99.08)
    ap.add_argument(
        "--out-dir",
        default="/home/ashandonay/bedcosmo/src/bedcosmo/num_tracers/emulator/bao",
    )
    ap.add_argument("--linear-axes", action="store_true",
                    help="Use linear axes instead of log–log.")
    args = ap.parse_args()

    print("=== Computing Fisher sigmas for DR1 ===")
    fisher = _fisher_stds("dr1", args.omega_m, args.hrdrag)
    print("=== Loading DESI sigmas for DR1 ===")
    desi = {key: _get_desi_std("dr1", key) for key in TRACER_CONFIGS}

    cells = _collect_cells(fisher, desi)
    _print_summary(cells, fisher, desi)

    out_path = Path(args.out_dir) / (
        f"relative_errors_dr1_Om_{args.omega_m:.5f}_"
        f"hrdrag_{args.hrdrag:.5f}.png"
    )
    _plot(cells, args.omega_m, args.hrdrag, out_path,
          log_axes=not args.linear_axes)


if __name__ == "__main__":
    main()
