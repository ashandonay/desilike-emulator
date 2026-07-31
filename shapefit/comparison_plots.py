"""ShapeFit comparison plots — forecast sigma, correlations, and mean values
against DESI DR1's published compressed constraints.

Three plots, selected by a positional subcommand (default: sigma):

  comparison_plots.py [sigma]   our sigma(qiso, qap, f_sigmar/f_sigmar, m) vs
                                DESI, 4 stacked panels. REPT only by default;
                                --theory kaiser rept to overlay Kaiser.
  comparison_plots.py covmat    4x4 covariance matrices per tracer, upper
                                triangular, generator / DESI / difference.
  comparison_plots.py corrmat   the same as correlation matrices.
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

m needs no conversion: the mean emulator now emits DESI's m -- the Eq. (4.9)
shape parameter, which multiplies the fiducial template so m = 0 is no shape
change (desilike calls that quantity `dm`; its own `m` is the absolute slope).
Before that change ours was the absolute slope and had to be offset by m_fid,
which is also why CHANGELOG S24's theory-dependent m_fid was a hazard.

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


def _corr_matrix(targets):
    """Full symmetric 4x4 CORRELATION matrix from the 6 rho_ targets.

    Diagonal is 1 by definition. The 6 rho values are the whole matrix, so this
    is a relabelling of the pairwise numbers rather than new information -- it
    just makes the sign and strength structure legible at a glance.
    """
    names = ["qiso", "qap", "f_sigmar", "m"]
    R = np.eye(4)
    for i in range(4):
        for j in range(i):
            key = f"rho_{names[j]}_{names[i]}"
            if key in targets:
                R[i, j] = R[j, i] = float(targets[key])
    return R


def _sigma_diag(targets, desi=False):
    """The 4 sigmas in matrix order (qiso, qap, f_sigmar, m).

    f_sigmar is FRACTIONAL on both sides -- ours divided by f_sigmar_fid, DESI's
    by its own f sigma_s8 -- because the two are evaluated at slightly different
    z_eff and f*sigma8 evolves fast (desi_reference docstring). The other three
    are absolute. This common basis is what makes the two covariances comparable
    element by element.
    """
    if desi:
        return np.array([targets["sigma_qiso"], targets["sigma_qap"],
                         targets["sigma_f_sigmar_frac"], targets["sigma_m"]])
    return np.array([targets["sigma_qiso"], targets["sigma_qap"],
                     targets["fsr_frac"], targets["sigma_m"]])


def _cov_matrix(targets, desi=False):
    """Full 4x4 COVARIANCE C = D R D, D = diag(sigma).

    Built rather than read off because the two sides arrive in different forms:
    DESI publishes a covariance in (D_V/r_d, D_H/D_M, f sigma_s8, m) units while
    the generator emits sigmas and correlations. Rebuilding both from
    (sigma, rho) in the SAME normalised basis (see _sigma_diag) is what makes
    them comparable -- DESI's raw covariance is in different units per entry and
    cannot be differenced against ours directly.

    Exact, not an approximation: C = D R D is the definition of the correlation
    matrix, so no information is lost either way.
    """
    D = np.diag(_sigma_diag(targets, desi=desi))
    return D @ _corr_matrix(targets) @ D


# Presentation differs between the two kinds because the quantities do:
# a covariance spans orders of magnitude and needs symlog, a correlation is
# bounded on [-1, 1] and does not (symlog would compress exactly the region
# where all the structure sits).
_MATRIX_KINDS = {
    "cov": dict(
        build=_cov_matrix, symlog=True, vmax=7e-2, dmax=7e-2, linthresh=1e-5,
        fmt=lambda v: "0" if abs(v) < 1e-6 else f"{v:+.1e}".replace("e-0", "e-"),
        textlim=8e-3, fontsize=6.4,
        label=r"$C$", dlabel=r"$\Delta C$",
        title="COVARIANCE", note="raw $C$ (diagonal = $\\sigma^2$)"),
    "corr": dict(
        build=lambda t, desi=False: _corr_matrix(t), symlog=False,
        vmax=1.0, dmax=0.4, linthresh=None,
        fmt=lambda v: f"{v:+.2f}", textlim=0.55, fontsize=7.5,
        label=r"$\rho$", dlabel=r"$\Delta\rho$",
        title="CORRELATION", note="diagonal = 1 by definition"),
}


def plot_covar_matrix(data, tracers, out_path, kind="cov", theory="rept"):
    """One 4x4 per tracer -- generator / DESI / difference -- upper triangular.

    ``kind='cov'``  covariance, so the diagonal is a real matrix element
                    (sigma^2) rather than a 1 by definition.
    ``kind='corr'`` correlation, when the sign/strength structure matters more
                    than the absolute scale.

    Values are identical either way (C = D R D); only the presentation differs.
    """
    from matplotlib.colors import SymLogNorm, Normalize
    K = _MATRIX_KINDS[kind]
    n = len(tracers)
    fig, axes = plt.subplots(3, n, figsize=(2.55 * n + 1.9, 8.4), squeeze=False)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("white")
    dcmap = plt.get_cmap("PuOr_r").copy(); dcmap.set_bad("white")
    keep = np.triu(np.ones((4, 4), bool))          # upper triangle + diagonal

    def _norm(vmax):
        if K["symlog"]:
            return SymLogNorm(linthresh=K["linthresh"], vmin=-vmax, vmax=vmax,
                              base=10)
        return Normalize(vmin=-vmax, vmax=vmax)

    nm_c, nm_d = _norm(K["vmax"]), _norm(K["dmax"])
    im_c = im_d = None
    for col, t in enumerate(tracers):
        th = theory if theory in data[t]["theories"] else next(iter(data[t]["theories"]))
        Mo = K["build"](data[t]["theories"][th]["targets"])
        Md = K["build"](data[t]["desi"], desi=True)
        for row, (M, cm, nm) in enumerate(((Mo, cmap, nm_c), (Md, cmap, nm_c),
                                           (Mo - Md, dcmap, nm_d))):
            ax = axes[row][col]
            im = ax.imshow(np.ma.masked_where(~keep, M), cmap=cm, norm=nm)
            if row < 2: im_c = im
            else: im_d = im
            lim = K["textlim"] if row < 2 else K["textlim"] * K["dmax"] / K["vmax"]
            for i in range(4):
                for j in range(i, 4):
                    v = M[i, j]
                    ax.text(j, i, K["fmt"](v), ha="center", va="center",
                            fontsize=K["fontsize"],
                            color="white" if abs(v) > lim else "black")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(_RHO_LABELS, fontsize=8)
            ax.set_yticklabels(_RHO_LABELS, fontsize=8)
            ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
            if col == 0:
                ax.set_ylabel(("generator", "DESI DR1",
                               "generator - DESI")[row], fontsize=10)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(_DISPLAY.get(t, t), fontsize=11)
            if row < 2:
                ax.set_xticklabels([])

    fig.suptitle(f"ShapeFit compressed-parameter {K['title']} vs DESI DR1 "
                 "(2411.12021 App. A, ShapeFit-alone)\n"
                 "basis $(q_{\\rm iso},\\, q_{\\rm ap},\\, f\\sigma_r/f\\sigma_r,\\, m)$ "
                 f"-- first three fractional, $m$ absolute; {K['note']}; "
                 "upper triangle", fontsize=11)
    fig.subplots_adjust(right=0.87, hspace=0.14, wspace=0.10, top=0.87)
    sfx = "  (symlog)" if K["symlog"] else ""
    c1 = fig.add_axes([0.895, 0.45, 0.013, 0.40])
    fig.colorbar(im_c, cax=c1).set_label(K["label"] + sfx, fontsize=9)
    c2 = fig.add_axes([0.895, 0.08, 0.013, 0.26])
    fig.colorbar(im_d, cax=c2).set_label(K["dlabel"] + sfx, fontsize=9)
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
    ax.set_ylabel(r"$m$")
    ax.set_title("m (DESI Eq. 4.9 convention)\nsame on both sides — no offset",
                 fontsize=9)

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
                   choices=["sigma", "rho", "covmat", "corrmat", "mean", "all"])
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
    if args.plot in ("covmat", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_covar_matrix(data, tracers,
                          _HERE / "shapefit_covar_matrix_vs_desi.png",
                          kind="cov", theory=th)
    if args.plot in ("corrmat", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_covar_matrix(data, tracers,
                          _HERE / "shapefit_corr_matrix_vs_desi.png",
                          kind="corr", theory=th)
    if args.plot in ("mean", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_mean(data, tracers, _HERE / "shapefit_mean_vs_desi.png", theory=th)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
