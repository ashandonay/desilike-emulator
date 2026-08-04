"""ShapeFit comparison plots — forecast sigma, correlations, and mean values
against DESI DR1's published compressed constraints.

Three plots, selected by a positional subcommand (default: sigma):

  comparison_plots.py [sigma]   our sigma(qiso, qap, f_sigmar/f_sigmar, m) vs
                                DESI, 4 stacked panels. REPT only by default;
                                --theory kaiser rept to overlay Kaiser.
  comparison_plots.py covmat    4x4 covariance matrices per tracer, upper
                                triangular: generator / DESI / % residual.
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

Because of that, the two AP panels plot D_V/r_d and D_H/D_M rather than qiso and
qap: the ratio is a flat line at 1, but its numerator (DESI's measurement) and
denominator (Table 11) are both real numbers worth seeing, and our own value at
our z_eff can be put on the same axis. See `plot_mean`.

The one genuinely predictive entry is f_sigmar: f(z) * sigma_r is an absolute
number our pipeline computes from the input cosmology, comparable to DESI's
measured f sigma_s8. Since the z_eff convention was corrected to DESI's
FKP-weighted definition (bao CHANGELOG S36) four of six tracers agree with
DESI's published z_eff to <0.5%; LRG3 and ELG2 still differ by ~2%, and
f*sigma8 evolves fast, so the panel labels those two with their Delta z.

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
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import desi_reference as desi_ref
import fourier_space
from compare_to_desi import FID_SAMPLE, our_forecast
from fourier_space import sf_core
from util import ntracers, plots_dir

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


# Percentage residuals are undefined where DESI's entry passes through zero, and
# an off-diagonal entry of either matrix is proportional to DESI's rho -- so a
# single cut on |rho_DESI| masks the ill-conditioned cells for BOTH kinds. The
# diagonal (rho = 1) is never masked.
_PCT_RHO_FLOOR = 0.05

# Minimum |z_eff/z_DESI - 1| worth labelling on the f sigma_r panel. Below this
# the two conventions agree and the label is clutter; above it the label is the
# explanation for a visible residual. See `plot_mean`.
_Z_EFF_LABEL_FLOOR = 0.005


def _pct_matrix(Mo, Md, Rd, floor=_PCT_RHO_FLOOR):
    """100 * (generator - DESI) / |DESI|, elementwise.

    |DESI| in the denominator rather than DESI, so the sign always reads
    "generator above/below DESI" instead of flipping with the sign of the
    reference entry.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        P = 100.0 * (Mo - Md) / np.abs(Md)
    P[np.abs(Rd) < floor] = np.nan
    return P


# Presentation differs between the two kinds because the quantities do:
# a covariance spans orders of magnitude and needs symlog, a correlation is
# bounded on [-1, 1] and does not (symlog would compress exactly the region
# where all the structure sits). The percentage row is linear for both.
_MATRIX_KINDS = {
    "cov": dict(
        build=_cov_matrix, symlog=True, vmax=7e-2, linthresh=1e-5,
        fmt=lambda v: "0" if abs(v) < 1e-6 else f"{v:+.1e}".replace("e-0", "e-"),
        textlim=8e-3, fontsize=6.4, pmax=100.0,
        label=r"$C$", title="COVARIANCE",
        note="raw $C$ (diagonal = $\\sigma^2$)"),
    "corr": dict(
        build=lambda t, desi=False: _corr_matrix(t), symlog=False,
        vmax=1.0, linthresh=None,
        fmt=lambda v: f"{v:+.2f}", textlim=0.55, fontsize=7.5, pmax=60.0,
        label=r"$\rho$", title="CORRELATION",
        note="diagonal = 1 by definition"),
}


def plot_covar_matrix(data, tracers, out_path, kind="cov", theory="rept"):
    """One 4x4 per tracer -- generator / DESI / percentage residual -- upper
    triangular.

    ``kind='cov'``  covariance, so the diagonal is a real matrix element
                    (sigma^2) rather than a 1 by definition.
    ``kind='corr'`` correlation, when the sign/strength structure matters more
                    than the absolute scale.

    Values are identical either way (C = D R D); only the presentation differs.
    The bottom row is a percentage of DESI rather than an absolute difference,
    so cells are comparable across tracers and across the two kinds -- an
    absolute Delta C is dominated by whichever tracer has the largest sigma.
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

    nm_c, nm_p = _norm(K["vmax"]), Normalize(vmin=-K["pmax"], vmax=K["pmax"])
    im_c = im_p = None
    for col, t in enumerate(tracers):
        th = theory if theory in data[t]["theories"] else next(iter(data[t]["theories"]))
        Mo = K["build"](data[t]["theories"][th]["targets"])
        Md = K["build"](data[t]["desi"], desi=True)
        Mp = _pct_matrix(Mo, Md, _corr_matrix(data[t]["desi"]))
        for row, (M, cm, nm) in enumerate(((Mo, cmap, nm_c), (Md, cmap, nm_c),
                                           (Mp, dcmap, nm_p))):
            ax = axes[row][col]
            bad = ~keep | ~np.isfinite(M)
            im = ax.imshow(np.ma.masked_where(bad, M), cmap=cm, norm=nm)
            if row < 2: im_c = im
            else: im_p = im
            fmt = K["fmt"] if row < 2 else (lambda v: f"{v:+.0f}%")
            lim = K["textlim"] if row < 2 else 0.55 * K["pmax"]
            for i in range(4):
                for j in range(i, 4):
                    v = M[i, j]
                    if not np.isfinite(v):
                        # grey only INSIDE the triangle: distinguishes "DESI is
                        # ~0 here" from the empty lower half, which stays white.
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                   facecolor="0.87",
                                                   edgecolor="none"))
                        ax.text(j, i, "--", ha="center", va="center",
                                fontsize=K["fontsize"] + 1, color="0.35")
                        continue
                    ax.text(j, i, fmt(v), ha="center", va="center",
                            fontsize=K["fontsize"],
                            color="white" if abs(v) > lim else "black")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(_RHO_LABELS, fontsize=8)
            ax.set_yticklabels(_RHO_LABELS, fontsize=8)
            ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
            if col == 0:
                ax.set_ylabel(("generator", "DESI DR1",
                               "(generator - DESI) / |DESI|")[row], fontsize=10)
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
                 "upper triangle\nbottom row is a percentage of DESI; "
                 f"'--' = undefined ($|\\rho_{{\\rm DESI}}| < {_PCT_RHO_FLOOR:g}$)",
                 fontsize=11)
    fig.subplots_adjust(right=0.87, hspace=0.14, wspace=0.10, top=0.85)
    sfx = "  (symlog)" if K["symlog"] else ""
    c1 = fig.add_axes([0.895, 0.45, 0.013, 0.40])
    fig.colorbar(im_c, cax=c1).set_label(K["label"] + sfx, fontsize=9)
    c2 = fig.add_axes([0.895, 0.08, 0.013, 0.26])
    fig.colorbar(im_p, cax=c2, extend="both").set_label(
        r"$(X_{\rm gen} - X_{\rm DESI})\,/\,|X_{\rm DESI}|$  [%]", fontsize=9)
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


# The two AP panels, as the physical distances rather than the ratios.
# (index into the DESI 4-vector, sigma key, y label, Table 11 key, q label,
#  legend corner -- residuals run one-signed in each panel, D_V/r_d negative and
#  D_H/D_M positive, so the free corner is opposite in each)
# Legend placement is "best", not a fixed corner, deliberately. These panels
# plot a residual whose SCALE keeps shrinking as the pipeline improves -- the
# D_H/D_M range went from +15.9% to -2.97% to -0.90% over three z_eff
# corrections in one day -- so any hardcoded position is chosen against a
# layout that no longer exists. "center left" was fine when the panel spanned
# -3%..+1%; at +-1% it sat on top of the BGS/LRG1/LRG2 markers and read as
# missing data.
_DIST_PANELS = [
    (0, "sigma_qiso", r"$D_V/r_d$", "DV_over_rd", r"$q_{\rm iso}$", "best"),
    (1, "sigma_qap", r"$D_H/D_M$", "DH_over_DM", r"$q_{\rm ap}$", "best"),
]


def plot_mean(data, tracers, out_path, theory="rept"):
    """DESI's measured compressed values against our mean-pipeline prediction.

    The AP panels compare the generator against DESI's PUBLISHED FIDUCIAL, not
    against DESI's DR1 measurement. Two reasons:

      - q = value/fiducial is 1 by construction on our side at the fiducial
        cosmology, so plotting q draws a flat line at 1 and hides the numbers.
      - The DR1 measurement is data. Whether the universe matches the fiducial
        is a statement about DESI, not about this pipeline, and putting it on
        the same axis invites reading a cosmological result as a code error.

    So the y-axis is the residual against Table 11 in percent, with the Table 11
    value itself annotated. Two generator markers separate the two effects: at
    DESI's z_eff the only difference is the distance/r_d convention (cosmoprimo
    vs the published table, <=0.13%); at our own z_eff the Fisher-weighted
    z_eff enters on top of that.

    Caveat: our distances come from the same cosmoprimo "DESI" cosmology the
    pipeline uses, so the open marker is a consistency check on conventions
    rather than an independent implementation.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = np.arange(len(tracers))
    labels = [_DISPLAY.get(t, t) for t in tracers]

    for ax, (idx, key, ylabel, t11, qlab, loc) in zip(axes[:2], _DIST_PANELS):
        fid = np.array([desi_ref.published_fiducial(t)[t11] for t in tracers])
        zf = np.array([data[t]["z_desi"] for t in tracers])
        zo = np.array([data[t]["theories"][theory]["z"] for t in tracers])
        # Two generator evaluations, which separates the two things that can
        # move us off the published fiducial:
        #   at DESI's z_eff  -> distance/r_d convention only (cosmoprimo vs T11)
        #   at our z_eff     -> that, plus our Fisher-weighted z_eff
        at_z_desi = np.array([desi_ref.fiducial_dv_dhdm(z)[idx] for z in zf])
        at_z_ours = np.array([desi_ref.fiducial_dv_dhdm(z)[idx] for z in zo])
        ax.axhline(0.0, color="0.45", lw=1.6, label="DESI fiducial (Table 11)")
        ax.plot(x, 100 * (at_z_desi / fid - 1), "o", ms=8, mfc="none", mew=1.4,
                color="tab:blue", label="generator @ DESI $z_{\\rm eff}$")
        ax.plot(x, 100 * (at_z_ours / fid - 1), "o", ms=6, color="tab:blue",
                label="generator @ own $z_{\\rm eff}$")
        # Table 11 values along the top, in axes coords -- anchoring them to the
        # y=0 line puts them straight through the markers.
        for i in range(len(tracers)):
            ax.text(i, 0.955, f"{fid[i]:.3g}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.0, color="0.35")
        r = np.concatenate([at_z_desi / fid - 1, at_z_ours / fid - 1, [0.0]]) * 100
        pad = max(r.ptp(), 1.0)
        ax.set_ylim(r.min() - 0.15 * pad, r.max() + 0.30 * pad)
        ax.set_ylabel(f"{ylabel}: generator / Table 11 $-1$  [%]")
        ax.legend(fontsize=7, loc=loc)
        ax.set_title(f"{ylabel} — the quantity behind {qlab}, against the "
                     "fiducial\nit is divided by (labels: Table 11 value)",
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
    # Under the DESI FKP z_eff convention four of six tracers now agree with
    # DESI's published z_eff to <0.5%, so annotating every point is noise.
    # Label only the two that still differ: f*sigma8 evolves fast enough that a
    # 2% shift in z is ~1.5% in f sigma_r, comparable to the residual plotted.
    for i, t in enumerate(tracers):
        dz = data[t]["theories"][theory]["z"] / data[t]["z_desi"] - 1.0
        if abs(dz) < _Z_EFF_LABEL_FLOOR:
            continue
        side = -1 if i == len(tracers) - 1 else 1     # last one points inward
        ax.annotate(rf"$\Delta z$ {100 * dz:+.1f}%", (i, ours[i]),
                    textcoords="offset points", xytext=(8 * side, -3),
                    ha="left" if side > 0 else "right", fontsize=6, color="0.3")
    ax.set_ylabel(r"$f\sigma_r$")
    ax.set_title(r"$f\sigma_r$ — PREDICTIVE." "\n"
                 r"$\Delta z$ vs DESI labelled where $>$0.5%",
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

    for i, ax in enumerate(axes):
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, fontsize=8)
        if i >= 2:                 # AP panels place their own legend (_DIST_PANELS)
            ax.legend(fontsize=7)
    fig.suptitle("ShapeFit mean values — AP panels: generator vs DESI's Table 11 "
                 "FIDUCIAL (no DR1 data; the ratio to it is 1 by construction).  "
                 "$f\\sigma_r$ and $m$: vs the DR1 measurement", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")



_FORECAST_PANELS = [
    ("sigma_qiso",        r"$\sigma(q_\mathrm{iso})$"),
    ("sigma_qap",         r"$\sigma(q_\mathrm{AP})$"),
    ("sigma_f_sigmar_frac", r"$\sigma(f\sigma_r)/f\sigma_r$"),
    ("sigma_m",           r"$\sigma(m)$"),
]


def _load_mcmc(tracers):
    """{tracer: {key: (mean, rms)}} merged over every mcmc.py --json output.

    The sweep runs one process per (tracer, seed) -- the log_prob closure is
    not picklable, so parallelism has to be across processes -- and each writes
    its own JSON. Seeds for the same tracer are therefore spread over files and
    must be unioned here before the rms means anything.

    Absent tracers simply get no MCMC marker; the run is expensive (hours per
    tracer) so the set on disk is often partial.
    """
    import json
    from util import logs_dir

    seeds = {}
    for path in sorted(Path(logs_dir("shapefit")).glob("shapefit_mcmc_*.json")):
        for t, v in json.loads(path.read_text()).items():
            if t in tracers:
                seeds.setdefault(t, {}).update(v["mcmc"])

    out = {}
    for t, runs in seeds.items():
        vals = list(runs.values())
        out[t] = {k: (float(np.mean([r[k] for r in vals])),
                      float(np.std([r[k] for r in vals])) if len(vals) > 1 else 0.0)
                  for k in vals[0]}
    return out


def plot_forecast(data, tracers, out_path, theory="rept"):
    """sigma per compressed parameter: DESI published vs Fisher vs MCMC.

    Modelled on bao/comparison_plots.py `_plot_forecast`, minus the bundle
    series -- shapefit has one covariance (ours), so colour carries nothing and
    only the ESTIMATOR varies: circle = Fisher, diamond = MCMC, cross = DESI.
    """
    mcmc = _load_mcmc(tracers)
    x = np.arange(len(tracers), dtype=float)
    c_ours, c_desi = "tab:blue", "black"

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                             constrained_layout=True)
    for ax, (key, ylabel) in zip(axes, _FORECAST_PANELS):
        for xi in x:
            ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)
        xd, yd, xf, yf, xm, ym, em = [], [], [], [], [], [], []
        for i, t in enumerate(tracers):
            d = data[t]["desi"].get(key, np.nan)
            if np.isfinite(d):
                xd.append(i); yd.append(d)
            tg = data[t]["theories"][theory]["targets"]
            f = tg["fsr_frac"] if key == "sigma_f_sigmar_frac" else tg.get(key, np.nan)
            if np.isfinite(f):
                xf.append(i); yf.append(f)
            if t in mcmc and key in mcmc[t]:
                mu, sd = mcmc[t][key]
                xm.append(i); ym.append(mu); em.append(sd)
        ax.scatter(xd, yd, marker="x", s=55, color=c_desi, linewidths=1.6,
                   zorder=5, label="DESI published")
        ax.scatter(xf, yf, marker="o", s=34, color=c_ours, linewidth=0,
                   zorder=4, label="Fisher (ours)")
        if xm:
            if any(e > 0 for e in em):
                ax.errorbar(xm, ym, yerr=em, fmt="none", ecolor=c_ours,
                            elinewidth=1.3, capsize=4, capthick=1.3, zorder=3)
            ax.scatter(xm, ym, marker="D", s=34, color=c_ours, linewidth=0,
                       zorder=4, label="MCMC (ours)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, None)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")

    handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=7,
               markeredgewidth=1.6, color=c_desi, label="DESI published (2411.12021 App. A)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=c_ours, markeredgecolor="none", label="Fisher (ours)"),
        Line2D([0], [0], marker="D", linestyle="", markersize=7,
               markerfacecolor=c_ours, markeredgecolor="none",
               label="MCMC (ours; error bar = seed rms)"),
    ]
    axes[0].legend(handles=handles, loc="best", frameon=True, fontsize=9)
    axes[0].set_title(f"ShapeFit forecast vs DESI DR1 published  ({theory.upper()})")
    _xticks(axes[-1], tracers)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plot", nargs="?", default="sigma",
                   choices=["sigma", "rho", "covmat", "corrmat", "mean", "forecast", "all"])
    p.add_argument("--tracers", nargs="+", default=_TRACERS, choices=_TRACERS)
    p.add_argument("--theory", nargs="+", default=["rept"],
                   choices=["kaiser", "rept"])
    args = p.parse_args()

    print(f"Gathering {len(args.tracers)} tracers x {len(args.theory)} theories "
          f"(REPT ~12 s/tracer) ...")
    data = _gather(args.tracers, args.theory)
    tracers = [t for t in args.tracers if t in data]

    if args.plot in ("sigma", "all"):
        plot_sigma(data, tracers, plots_dir() / "shapefit_sigma_vs_desi.png")
    if args.plot in ("rho", "all"):
        plot_rho(data, tracers, plots_dir() / "shapefit_rho_vs_desi.png")
    if args.plot in ("covmat", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_covar_matrix(data, tracers,
                          plots_dir() / "shapefit_covar_matrix_vs_desi.png",
                          kind="cov", theory=th)
    if args.plot in ("corrmat", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_covar_matrix(data, tracers,
                          plots_dir() / "shapefit_corr_matrix_vs_desi.png",
                          kind="corr", theory=th)
    if args.plot in ("forecast", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_forecast(data, tracers,
                      plots_dir() / "shapefit_forecast_comparison_dr1.png", theory=th)
    if args.plot in ("mean", "all"):
        th = "rept" if "rept" in args.theory else args.theory[0]
        plot_mean(data, tracers, plots_dir() / "shapefit_mean_vs_desi.png", theory=th)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
