"""ShapeFit comparison plots — forecast sigma, correlations, and mean values
against DESI DR1's published compressed constraints.

Three plots, selected by a positional subcommand (default: sigma):

  comparison_plots.py [sigma]   our sigma(qiso, qap, f_sigmar/f_sigmar, m) vs
                                DESI, 4 stacked panels, Kaiser and REPT markers.
  comparison_plots.py rho       the 6 pairwise correlations vs DESI, 6 panels.
                                These are 6 of the 10 emulator targets and had
                                no external check at all before desi_reference.
  comparison_plots.py mean      mean-pipeline values vs DESI's measurements.
                                READ THE CAVEAT -- most of this panel does not
                                test our pipeline (see below).

Reference is desi_reference.py: DESI 2024 V (arXiv:2411.12021) Appendix A,
ShapeFit-ALONE fits (not the tighter ShapeFit+BAO), transcribed per tracer with
full 4x4 covariances.

What the `mean` plot can and cannot say
---------------------------------------
Evaluated at the DESI fiducial cosmology, our mean pipeline returns qiso = 1,
qap = 1 and dm = 0 **by construction** -- the fiducial is its own reference. So
for those three, the plot shows whether DESI's DATA is consistent with the
fiducial cosmology. That is a real and interesting question, but it is a
statement about DESI, not about us.

The one genuinely predictive entry is f_sigmar: f(z) * sigma_r is an absolute
number our pipeline computes from the input cosmology, comparable to DESI's
measured f sigma_s8. Even there, our z_eff and DESI's differ (Fisher- vs
volume-weighted; bao CHANGELOG S18), and f*sigma8 evolves fast, so the panel
annotates both redshifts.

m is plotted as a DEVIATION on both sides. DESI's Eq. (4.9) m is already a
deviation (m = 0 means no shape change); ours is an absolute slope, so we plot
m - m_fid. Plotting the raw values against each other would show a spurious
~0.58 offset.

Usage (from shapefit/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python comparison_plots.py
    ... comparison_plots.py rho
    ... comparison_plots.py mean
    ... comparison_plots.py all --theory rept
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import desi_reference as desi_ref
import fourier_space
from compare_to_desi import FID_SAMPLE, our_forecast
from fourier_space import sf_core
from util import ntracers

_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
_DISPLAY = {}
_HERE = Path(__file__).resolve().parent

_SIGMA_PANELS = [
    ("sigma_qiso", "sigma_qiso", r"$\sigma(q_{\rm iso})$"),
    ("sigma_qap", "sigma_qap", r"$\sigma(q_{\rm ap})$"),
    ("fsr_frac", "sigma_f_sigmar_frac", r"$\sigma(f\sigma_r)/f\sigma_r$"),
    ("sigma_m", "sigma_m", r"$\sigma(m)$"),
]
_RHO_NAMES = [n for n in sf_core.TARGET_NAMES if n.startswith("rho_")]

_THEORY_STYLE = {
    "kaiser": dict(marker="v", color="tab:orange", label="ours (Kaiser)"),
    "rept": dict(marker="o", color="tab:blue", label="ours (REPT)"),
}

_CACHE: dict = {}


def _ours(tracer: str, theory: str) -> dict:
    key = (tracer, theory)
    if key not in _CACHE:
        _CACHE[key] = our_forecast(tracer, theory=theory)
    return _CACHE[key]


def _gather(tracers, theories):
    """Per-tracer: our targets for each theory, plus DESI's reference."""
    out = {}
    for t in tracers:
        try:
            ref = desi_ref.sigma_targets(t)
            z_desi, vec, _ = desi_ref.datavector(t)
        except KeyError:
            continue
        entry = {"desi": ref, "z_desi": z_desi, "desi_vec": vec, "theories": {}}
        for th in theories:
            o = _ours(t, th)
            tg = dict(o["targets"])
            tg["fsr_frac"] = (tg["sigma_f_sigmar"]
                              / float(o["info"]["f_sigmar_fid"]))
            entry["theories"][th] = {"targets": tg, "info": o["info"],
                                     "z": o["z_eff"]}
        out[t] = entry
    return out


def _xticks(ax, tracers):
    ax.set_xticks(range(len(tracers)))
    ax.set_xticklabels([_DISPLAY.get(t, t) for t in tracers], rotation=20)


def plot_sigma(data, tracers, out_path):
    fig, axes = plt.subplots(len(_SIGMA_PANELS), 1, figsize=(9, 12), sharex=True)
    x = np.arange(len(tracers))
    for ax, (ours_key, desi_key, ylabel) in zip(axes, _SIGMA_PANELS):
        d = [data[t]["desi"][desi_key] for t in tracers]
        ax.plot(x, d, "s", ms=10, mfc="none", color="k", label="DESI DR1")
        for th, st in _THEORY_STYLE.items():
            if th not in data[tracers[0]]["theories"]:
                continue
            y = [data[t]["theories"][th]["targets"][ours_key] for t in tracers]
            ax.plot(x, y, ls="none", ms=7, **st)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        # ratio annotation for the reference theory
        th = "rept" if "rept" in data[tracers[0]]["theories"] else "kaiser"
        for i, t in enumerate(tracers):
            r = data[t]["theories"][th]["targets"][ours_key] / data[t]["desi"][desi_key]
            ax.annotate(f"{r:.2f}", (i, data[t]["theories"][th]["targets"][ours_key]),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=7, color="tab:blue")
    axes[0].legend(fontsize=8, ncol=3)
    _xticks(axes[-1], tracers)
    axes[0].set_title("ShapeFit forecast $\\sigma$ vs DESI DR1 (2411.12021 App. A, "
                      "ShapeFit-alone)\nnumbers = ours/DESI for REPT; "
                      "LRG3+ELG1 is a different galaxy sample from DESI's bin",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def plot_rho(data, tracers, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    x = np.arange(len(tracers))
    for ax, name in zip(axes.ravel(), _RHO_NAMES):
        d = [data[t]["desi"][name] for t in tracers]
        ax.plot(x, d, "s", ms=10, mfc="none", color="k", label="DESI DR1")
        for th, st in _THEORY_STYLE.items():
            if th not in data[tracers[0]]["theories"]:
                continue
            y = [data[t]["theories"][th]["targets"][name] for t in tracers]
            ax.plot(x, y, ls="none", ms=7, **st)
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        ax.set_title(name[4:].replace("_", ", "), fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
        _xticks(ax, tracers)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("ShapeFit correlations vs DESI DR1 — 6 of the 10 emulator "
                 "targets, with no external check before this", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def plot_mean(data, tracers, out_path, theory="rept"):
    """DESI's measured compressed values against our mean-pipeline prediction.

    See the module docstring: only the f_sigmar panel actually tests us.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = np.arange(len(tracers))
    labels = [_DISPLAY.get(t, t) for t in tracers]

    # qiso, qap: ours is 1 by construction at the fiducial.
    for ax, (idx, key, ylabel) in zip(
            axes[:2], [(0, "sigma_qiso", r"$q_{\rm iso}$"),
                       (1, "sigma_qap", r"$q_{\rm ap}$")]):
        fid = [desi_ref.fiducial_dv_dhdm(data[t]["z_desi"])[idx] for t in tracers]
        meas = [data[t]["desi_vec"][idx] / f for t, f in zip(tracers, fid)]
        err = [data[t]["desi"][key] for t in tracers]
        ax.errorbar(x, meas, yerr=err, fmt="s", ms=8, mfc="none", color="k",
                    capsize=3, label="DESI DR1")
        ax.axhline(1.0, color="tab:blue", lw=1.6, label="ours (= 1 by constr.)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + "  — tests DESI vs fiducial,\nnot our pipeline",
                     fontsize=9)

    # f_sigmar: the one genuinely predictive panel.
    ax = axes[2]
    meas = [data[t]["desi_vec"][2] for t in tracers]
    err = [data[t]["desi"]["sigma_f_sigmar_frac"] * m for m, t in zip(meas, tracers)]
    ours = [float(data[t]["theories"][theory]["info"]["f_sigmar_fid"])
            for t in tracers]
    ax.errorbar(x, meas, yerr=err, fmt="s", ms=8, mfc="none", color="k",
                capsize=3, label="DESI DR1")
    ax.plot(x, ours, "o", ms=7, color="tab:blue", label="ours (predicted)")
    for i, t in enumerate(tracers):
        ax.annotate(f"z {data[t]['theories'][theory]['z']:.2f}/"
                    f"{data[t]['z_desi']:.2f}", (i, ours[i]),
                    textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=6)
    ax.set_ylabel(r"$f\sigma_r$")
    ax.set_title(r"$f\sigma_r$ — PREDICTIVE." "\n" r"z ours/DESI annotated",
                 fontsize=9)

    # m: deviation on both sides.
    ax = axes[3]
    meas = [data[t]["desi_vec"][3] for t in tracers]
    err = [data[t]["desi"]["sigma_m"] for t in tracers]
    ax.errorbar(x, meas, yerr=err, fmt="s", ms=8, mfc="none", color="k",
                capsize=3, label="DESI DR1")
    ax.axhline(0.0, color="tab:blue", lw=1.6, label="ours (dm = 0 by constr.)")
    ax.set_ylabel(r"$m - m_{\rm fid}$")
    ax.set_title("m as a DEVIATION both sides\n(ours is absolute, offset by "
                 r"$m_{\rm fid}$)", fontsize=9)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("ShapeFit mean values vs DESI DR1 — three of four panels test "
                 "whether DESI's data matches the fiducial, not our pipeline",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plot", nargs="?", default="sigma",
                   choices=["sigma", "rho", "mean", "all"])
    p.add_argument("--tracers", nargs="+", default=_TRACERS, choices=_TRACERS)
    p.add_argument("--theory", nargs="+", default=["kaiser", "rept"],
                   choices=["kaiser", "rept"])
    args = p.parse_args()

    print(f"Gathering {len(args.tracers)} tracers x {len(args.theory)} theories "
          f"(REPT ~12 s/tracer) ...")
    data = _gather(args.tracers, args.theory)
    tracers = [t for t in args.tracers if t in data]

    if args.plot in ("sigma", "all"):
        plot_sigma(data, tracers, _HERE / "shapefit_sigma_vs_desi.png")
    if args.plot in ("rho", "all"):
        plot_rho(data, tracers, _HERE / "shapefit_rho_vs_desi.png")
    if args.plot in ("mean", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_mean(data, tracers, _HERE / "shapefit_mean_vs_desi.png", theory=th)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
