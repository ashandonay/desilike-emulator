"""Verify the DESI systematic-error layer moves the pipeline σ by the SAME
fractional amount that DESI's own σ move from stat-only to stat+syst.

The inflation factors (`DESI_SYST_INFLATION`) are DESI's measured
σ(bao-recon_syst)/σ(bao-recon_stat-only), so applying them to the pipeline σ
should reproduce DESI's own stat→syst jump exactly. This script extracts all
four series and plots them, plus the syst/stat ratio for pipeline vs DESI (which
should overlap). Any gap in the ratio panel = drift between the frozen table and
the live .h5 files.

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python verify_syst_inflation.py
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from util import plots_dir
warnings.filterwarnings("ignore")

import config_space as cc
import desi_reference as rs

_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
_HERE = Path(__file__).resolve().parent
_QUANTITIES = [
    ("DH_over_rs", r"$\sigma(D_H/r_d)$"),
    ("DM_over_rs", r"$\sigma(D_M/r_d)$"),
    ("DV_over_rs", r"$\sigma(D_V/r_d)$"),
]
_QSHORT = {"DH_over_rs": "DH/rd", "DM_over_rs": "DM/rd", "DV_over_rs": "DV/rd"}


def _is_iso(t):
    return t in cc._ISO_TRACERS if hasattr(cc, "_ISO_TRACERS") else t in ("BGS", "QSO")


def _gather():
    out = {}
    for t in _TRACERS:
        print(f"== {t} ==", flush=True)
        gen = cc.XiSigmaGenerator(t)
        pipe = gen.sigma_triplet(N_tracers=cc._get_ntracers(t))     # σ_stat (+ fid)

        pipe_stat = {k: pipe[k] for k in rs._SIGMA_KEYS}
        pipe_syst = rs.apply_desi_syst(pipe_stat, t)                # σ_stat → σ_tot
        desi_stat = rs._recon_sigmas(t, pipe)                       # DESI stat-only .h5
        desi_syst = rs._triplet_from_recon_cov(                     # DESI stat+syst .h5
            rs._LIK_DIR / rs._BAO_RECON_SYST_FILE[t], pipe)

        # Mask: iso tracers carry only DV; anisotropic carry DH+DM.
        mask = ["DH_over_rs", "DM_over_rs"] if _is_iso(t) else ["DV_over_rs"]
        for src in (pipe_stat, pipe_syst, desi_stat, desi_syst):
            for q in mask:
                src[q] = float("nan")

        out[t] = {"pipe_stat": pipe_stat, "pipe_syst": pipe_syst,
                  "desi_stat": desi_stat, "desi_syst": desi_syst}
    return out


def _print_table(data):
    print("\n=== inflation factor σ_syst/σ_stat: pipeline (applied) vs DESI (live .h5) ===")
    hdr = f"  {'tracer':<11} {'q':<6} {'pipe×':>8} {'DESI×':>8} {'Δ':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    max_drift = 0.0
    for t in _TRACERS:
        d = data[t]
        for q, _ in _QUANTITIES:
            ps, py = d["pipe_stat"][q], d["pipe_syst"][q]
            ds, dy = d["desi_stat"][q], d["desi_syst"][q]
            if not (np.isfinite(ps) and np.isfinite(ds)):
                continue
            rp, rd = py / ps, dy / ds
            drift = abs(rp - rd)
            max_drift = max(max_drift, drift)
            print(f"  {t:<11} {_QSHORT[q]:<6} {rp:>8.4f} {rd:>8.4f} {drift:>8.4f}")
    print(f"\n  max |pipeline× − DESI×| = {max_drift:.4f}  "
          f"({'PASS — frozen table matches live .h5' if max_drift < 5e-3 else 'DRIFT'})")


def _plot(data, out_path):
    tracers = list(_TRACERS)
    disp = [_DISPLAY.get(t, t) for t in tracers]
    x = np.arange(len(tracers), dtype=float)
    c_desi, c_pipe = "black", "tab:blue"
    dx = 0.12  # x-offset so pipeline / DESI don't overlap

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1.0]})

    for row, (q, ylabel) in enumerate(_QUANTITIES):
        ax_abs, ax_rat = axes[row]
        for xi in x:
            ax_abs.axvline(xi, color="gray", alpha=0.2, lw=0.6)
            ax_rat.axvline(xi, color="gray", alpha=0.2, lw=0.6)

        for i, t in enumerate(tracers):
            d = data[t]
            # --- absolute σ: stat (open) → syst (filled), connected ---
            for src_stat, src_syst, col, off in (
                ("desi_stat", "desi_syst", c_desi, +dx),
                ("pipe_stat", "pipe_syst", c_pipe, -dx)):
                s0, s1 = d[src_stat][q], d[src_syst][q]
                if not np.isfinite(s0):
                    continue
                xx = i + off
                ax_abs.plot([xx, xx], [s0, s1], color=col, lw=1.0, alpha=0.7, zorder=2)
                ax_abs.scatter([xx], [s0], marker="o", s=46, facecolor="none",
                               edgecolor=col, linewidths=1.5, zorder=3)
                ax_abs.scatter([xx], [s1], marker="o", s=30, color=col, zorder=4)
            # --- ratio σ_syst/σ_stat: pipeline vs DESI (should coincide) ---
            for src_stat, src_syst, col, off, mk in (
                ("desi_stat", "desi_syst", c_desi, +dx, "s"),
                ("pipe_stat", "pipe_syst", c_pipe, -dx, "o")):
                s0, s1 = d[src_stat][q], d[src_syst][q]
                if not np.isfinite(s0):
                    continue
                ax_rat.scatter([i + off], [s1 / s0], marker=mk, s=50, color=col, zorder=3)

        ax_abs.set_ylabel(ylabel)
        ax_abs.set_ylim(0.0, None)
        ax_rat.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax_rat.set_ylabel(r"$\sigma_{\rm syst}/\sigma_{\rm stat}$")
        for ax in (ax_abs, ax_rat):
            ax.set_xticks(x)
            ax.set_xticklabels(disp if row == 2 else [""] * len(x),
                               rotation=20 if row == 2 else 0)
            ax.grid(alpha=0.2, ls="--", lw=0.6, axis="y")

    axes[0, 0].set_title("Absolute σ: stat (open) → stat+syst (filled)", fontsize=11)
    axes[0, 1].set_title("Inflation factor (pipeline should overlap DESI)", fontsize=11)
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc="none", mec=c_desi, mew=1.5, ms=8,
               label="DESI stat-only"),
        Line2D([0], [0], marker="o", ls="", color=c_desi, ms=7, label="DESI stat+syst"),
        Line2D([0], [0], marker="o", ls="", mfc="none", mec=c_pipe, mew=1.5, ms=8,
               label="pipeline stat"),
        Line2D([0], [0], marker="o", ls="", color=c_pipe, ms=7, label="pipeline stat+syst"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("DESI systematic inflation: pipeline vs DESI  |  DR1", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    data = _gather()
    _print_table(data)
    _plot(data, str(plots_dir() / "bao_verify_syst_inflation_dr1.png"))


if __name__ == "__main__":
    main()
