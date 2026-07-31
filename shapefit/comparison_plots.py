"""ShapeFit comparison plots — forecast sigma, correlations, and mean values
against DESI DR1's published compressed constraints.

Three plots, selected by a positional subcommand (default: sigma):

  comparison_plots.py [sigma]   our sigma(qiso, qap, f_sigmar/f_sigmar, m) vs
                                DESI, 4 stacked panels. REPT only by default;
                                --theory kaiser rept to overlay Kaiser.
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
    "kaiser": dict(marker="v", color="tab:orange", label="generator (Kaiser)"),
    "rept": dict(marker="o", color="tab:blue", label="generator"),
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
                      "ShapeFit-alone)\nnumbers = generator/DESI",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


_RHO_PAIRS = [("qiso", "qap"), ("qiso", "f_sigmar"), ("qiso", "m"),
              ("qap", "f_sigmar"), ("qap", "m"), ("f_sigmar", "m")]
_RHO_LABELS = [r"$q_{\rm iso}$", r"$q_{\rm ap}$", r"$f\sigma_r$", r"$m$"]


def _rho_matrix(targets):
    """Strict lower triangle of the 4x4 correlation matrix, from the 6 rho_ targets.

    Diagonal and upper triangle left NaN. The 6 rho values ARE the whole matrix
    (diagonal is 1 by definition, upper is the mirror), so this is a relabelling
    of the pairwise numbers, not new information -- it just makes the sign and
    strength structure legible at a glance.
    """
    names = ["qiso", "qap", "f_sigmar", "m"]
    M = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(i):
            key = f"rho_{names[j]}_{names[i]}"
            if key in targets:
                M[i, j] = float(targets[key])
    return M


def _sigma_diag(targets, desi=False):
    """The 4 sigmas in matrix-diagonal order (qiso, qap, f_sigmar, m).

    f_sigmar is FRACTIONAL on both sides -- ours divided by f_sigmar_fid, DESI's
    by its own f sigma_s8 -- because the two are evaluated at slightly different
    z_eff and f*sigma8 evolves fast (desi_reference docstring). The other three
    are absolute and directly comparable.
    """
    if desi:
        return np.array([targets["sigma_qiso"], targets["sigma_qap"],
                         targets["sigma_f_sigmar_frac"], targets["sigma_m"]])
    return np.array([targets["sigma_qiso"], targets["sigma_qap"],
                     targets["fsr_frac"], targets["sigma_m"]])


def plot_rho_matrix(data, tracers, out_path, theory="rept"):
    """One 4x4 per tracer: sigma on the diagonal, correlations below it.

    Rows are ours / DESI / comparison. Off-diagonal cells carry rho (rows 1-2)
    and Delta-rho (row 3); diagonal cells carry sigma (rows 1-2) and the ratio
    ours/DESI (row 3).

    The diagonal and off-diagonal are on SEPARATE colour scales and separate
    colourbars, deliberately: sigma and rho are different quantities and a shared
    scale would invite reading a sigma cell against a rho cell. Only the ratio
    row's diagonal is coloured (centred on 1.0, which is agreement); the absolute
    sigma diagonals are left neutral since their magnitudes differ by tracer and
    the colour would carry no meaning.
    """
    n = len(tracers)
    fig, axes = plt.subplots(3, n, figsize=(2.55 * n + 1.9, 8.4), squeeze=False)
    rho_cmap = plt.get_cmap("RdBu_r").copy(); rho_cmap.set_bad("white")
    d_cmap = plt.get_cmap("PuOr_r").copy(); d_cmap.set_bad("white")
    rat_cmap = plt.get_cmap("BrBG").copy(); rat_cmap.set_bad("white")
    tri_lo = np.tril(np.ones((4, 4), bool), -1)
    diag = np.eye(4, dtype=bool)

    im_rho = im_drho = im_rat = None
    for col, t in enumerate(tracers):
        th = theory if theory in data[t]["theories"] else next(iter(data[t]["theories"]))
        ot = data[t]["theories"][th]["targets"]
        dt = data[t]["desi"]
        Ro, Rd = _rho_matrix(ot), _rho_matrix(dt)
        So, Sd = _sigma_diag(ot), _sigma_diag(dt, desi=True)
        rows = [(Ro, So, "abs"), (Rd, Sd, "abs"), (Ro - Rd, So / Sd, "ratio")]

        for row, (R, D, dmode) in enumerate(rows):
            ax = axes[row][col]
            cm, vlim = (rho_cmap, 1.0) if row < 2 else (d_cmap, 0.6)
            im = ax.imshow(np.ma.masked_where(~tri_lo, np.nan_to_num(R)),
                           cmap=cm, vmin=-vlim, vmax=vlim)
            if row < 2: im_rho = im
            else: im_drho = im

            # Diagonal on its own scale: neutral for absolute sigma, a
            # ratio-centred map for the comparison row.
            Dm = np.full((4, 4), np.nan)
            Dm[diag] = D
            if dmode == "ratio":
                imd = ax.imshow(np.ma.masked_invalid(Dm), cmap=rat_cmap,
                                vmin=0.5, vmax=1.5)
                im_rat = imd
            else:
                ax.imshow(np.ma.masked_invalid(Dm),
                          cmap=plt.get_cmap("Greys").copy(), vmin=0, vmax=1,
                          alpha=0.18)

            for i in range(4):
                for j in range(i + 1):
                    if i == j:
                        v = D[i]
                        txt = f"{v:.2f}" if dmode == "ratio" else f"{v:.3f}"
                        col_ = "black"
                        if dmode == "ratio" and abs(v - 1.0) > 0.35:
                            col_ = "white"
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=7.5, fontweight="bold", color=col_)
                    else:
                        v = R[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                                    fontsize=7.5,
                                    color="white" if abs(v) > 0.55 * vlim else "black")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(_RHO_LABELS, fontsize=8)
            ax.set_yticklabels(_RHO_LABELS, fontsize=8)
            ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
            if col == 0:
                ax.set_ylabel(("generator", "DESI DR1",
                               "generator - DESI  /  ratio")[row], fontsize=10)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(_DISPLAY.get(t, t), fontsize=11)
            if row < 2:
                ax.set_xticklabels([])

    fig.suptitle("ShapeFit vs DESI DR1 (2411.12021 App. A, ShapeFit-alone)\n"
                 "diagonal = $\\sigma$ (bold; $f\\sigma_r$ fractional), "
                 "below = $\\rho$; bottom row: diagonal = generator/DESI, "
                 "below = $\\Delta\\rho$", fontsize=11)
    fig.subplots_adjust(right=0.87, hspace=0.14, wspace=0.10, top=0.88)
    c1 = fig.add_axes([0.895, 0.50, 0.013, 0.36]); fig.colorbar(im_rho, cax=c1).set_label(r"$\rho$", fontsize=10)
    c2 = fig.add_axes([0.895, 0.28, 0.013, 0.17]); fig.colorbar(im_drho, cax=c2).set_label(r"$\Delta\rho$", fontsize=10)
    if im_rat is not None:
        c3 = fig.add_axes([0.895, 0.06, 0.013, 0.17]); fig.colorbar(im_rat, cax=c3).set_label(r"$\sigma$ generator/DESI", fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
        ax.axhline(1.0, color="tab:blue", lw=1.6, label="generator (= 1 by constr.)")
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
    ax.plot(x, ours, "o", ms=7, color="tab:blue", label="generator (predicted)")
    for i, t in enumerate(tracers):
        ax.annotate(f"z {data[t]['theories'][theory]['z']:.2f}/"
                    f"{data[t]['z_desi']:.2f}", (i, ours[i]),
                    textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=6)
    ax.set_ylabel(r"$f\sigma_r$")
    ax.set_title(r"$f\sigma_r$ — PREDICTIVE." "\n" r"z generator/DESI annotated",
                 fontsize=9)

    # m: deviation on both sides.
    ax = axes[3]
    meas = [data[t]["desi_vec"][3] for t in tracers]
    err = [data[t]["desi"]["sigma_m"] for t in tracers]
    ax.errorbar(x, meas, yerr=err, fmt="s", ms=8, mfc="none", color="k",
                capsize=3, label="DESI DR1")
    ax.axhline(0.0, color="tab:blue", lw=1.6, label="generator (dm = 0 by constr.)")
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
                   choices=["sigma", "rho", "rhomat", "mean", "all"])
    p.add_argument("--tracers", nargs="+", default=_TRACERS, choices=_TRACERS)
    p.add_argument("--theory", nargs="+", default=["rept"],
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
    if args.plot in ("rhomat", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_rho_matrix(data, tracers, _HERE / "shapefit_rho_matrix_vs_desi.png",
                        theory=th)
    if args.plot in ("mean", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_mean(data, tracers, _HERE / "shapefit_mean_vs_desi.png", theory=th)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
