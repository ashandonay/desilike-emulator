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
  comparison_plots.py mean      mean-pipeline values vs a DESI reference,
                                chosen with --reference:
                                  fiducial (default) Table 11 / Appendix C.
                                    A NULL TEST -- see below.
                                  dr1                the App. A measurement,
                                    with its 1-sigma band. A DESI RESULT, not
                                    a test of this code -- see below.

Reference is desi_reference.py: DESI 2024 V (arXiv:2411.12021) Appendix A,
ShapeFit-ALONE fits (not the tighter ShapeFit+BAO), transcribed per tracer with
full 4x4 covariances.

What the `mean` plot can and cannot say
---------------------------------------
Evaluated at the DESI fiducial cosmology, our mean pipeline returns qiso = 1,
qap = 1, f_sigmar = Table 11 and m = 0 **by construction** -- the fiducial is
its own reference. NEITHER reference mode tests how the pipeline responds to
cosmology, which is its actual content.

--reference fiducial is a null test of CONVENTIONS: a wrong r_d, a wrong z_eff
or a wrong de-wiggling engine shows up, nothing else does. That is how the
S53/S54 z_eff work was verified, so it earns its keep, but do not read
agreement there as validation of the emulator.

--reference dr1 asks whether DR1 agrees with Planck-LCDM. That is a DESI
result. It is on the same axes only because the basis is shared; a departure
there is not a code error. CHANGELOG S66 records the version of this plot that
mixed the two questions without saying so.

The AP panels plot D_V/r_d and D_H/D_M rather than qiso and qap because the
ratio is a flat line at 1 on our side, while its numerator and denominator are
both real numbers worth seeing. Those two entries are DISTANCES recomputed at
our z_eff; f_sigmar and m are the actual mean-pipeline outputs.

A genuine prediction test -- generator at DR1's own best-fit cosmology, against
DR1's measurement -- needs DESI's LCDM parameter posterior. desi_reference
carries compressed parameters only, so it is NOT BUILT.

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
from util import ntracers, plots_dir, tracer_area

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


_MEAN_CACHE: dict = {}


def _mean_targets(tracer):
    """Mean-pipeline (qiso, qap, f_sigmar, m) at the DESI fiducial cosmology.

    The actual generator output, not `f_sigmar_fid` read off the covar path's
    info dict. They agree to ~0.02% at the fiducial, but they are different
    objects (S64: the covar template's fiducial IS the sample, so its
    f_sigmar_fid is f*sigma_r(8) while the mean extractor returns
    f*sigma_r(8s)), and this plot is about the MEAN pipeline.
    """
    if tracer not in _MEAN_CACHE:
        from compare_to_desi import FID_SAMPLE
        _, vals, _ = fourier_space._worker_run_mean_targets(
            (dict(FID_SAMPLE), tracer, None, None,
             tracer_area(tracer, "dr1"), "dr1"))
        _MEAN_CACHE[tracer] = list(vals)
    return _MEAN_CACHE[tracer]


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


# The four mean panels. `kind` is how the residual is formed:
#   "ratio" -> 100 * (generator / reference - 1), in percent
#   "abs"   -> generator - reference, in the parameter's own units
# m is "abs" because in fiducial mode its reference is 0 (Eq. 4.9 is a
# definition, not a table entry), so no ratio exists. Keeping it "abs" in dr1
# mode too matches DESI's own convention, where m is an additive deviation.
_MEAN_PANELS = [
    (r"$D_V/r_d$",   "ratio", "DV_over_rd", "best"),
    (r"$D_H/D_M$",   "ratio", "DH_over_DM", "best"),
    (r"$f\sigma_r$", "ratio", "f_sigma_s8", "best"),
    (r"$m$",         "abs",   None,         "best"),
]

_REFERENCE_CHOICES = ("fiducial", "dr1")


def _mean_vectors(tracers, theory, data):
    """(generator, fiducial, dr1, dr1_sigma) as 4-vectors per tracer.

    Basis is DESI's: (D_V/r_d, D_H/D_M, f sigma_r, m).

    The first two generator entries are DISTANCES recomputed at our z_eff, not
    the mean pipeline's qiso/qap -- those are 1 to 1e-7 at the fiducial and
    carry no information. The last two are the actual mean-pipeline outputs.
    """
    gen, fid, dr1, sig = [], [], [], []
    for t in tracers:
        z_ours = data[t]["theories"][theory]["z"]
        dv, dhdm = desi_ref.fiducial_dv_dhdm(z_ours)
        mt = _mean_targets(t)
        gen.append([dv, dhdm, mt[2], mt[3]])
        pf = desi_ref.published_fiducial(t)
        fid.append([pf["DV_over_rd"], pf["DH_over_DM"], pf["f_sigma_s8"], 0.0])
        _z, vec, cov = desi_ref.datavector(t)
        dr1.append(list(vec))
        sig.append(list(np.sqrt(np.diag(cov))))
    return (np.array(gen), np.array(fid), np.array(dr1), np.array(sig))


def plot_mean(data, tracers, out_path, theory="rept", reference="fiducial"):
    """Mean-pipeline values against a chosen DESI reference.

    reference="fiducial" (default)
        Table 11 / Appendix C, with m = 0 by Eq. (4.9). A NULL TEST: at the
        fiducial cosmology the generator returns 1, 1, Table 11 and 0 by
        construction, so the only thing this can detect is a convention or
        implementation error -- a wrong r_d, a wrong z_eff, a wrong de-wiggling
        engine. It cannot test the pipeline's actual content, which is how the
        four outputs VARY with cosmology and N_tracers.

    reference="dr1"
        DESI's measured compressed values (Appendix A, Eqs. A.1-A.24) with
        their 1-sigma band. NOT a test of this code: the generator is
        evaluated at the FIDUCIAL cosmology, so the residual is "does DR1 agree
        with Planck-LCDM", a DESI result. Appendix A publishes a datavector
        plus a Gaussian covariance, so its centres are the mean and the MAP
        alike -- the two would only separate where the posterior is non-
        Gaussian, which S63 showed is the m row.

    A genuine prediction test would evaluate the generator at DR1's own
    best-fit cosmology and compare there. That needs DESI's LCDM parameter
    posterior, which desi_reference does not carry -- it holds compressed
    parameters only. Not built.

    The generator's AP entries are evaluated at OUR z_eff, so their offset
    mixes the distance/r_d convention (<=0.13%) with our Fisher-weighted z_eff
    differing from DESI's. An earlier version drew a second open marker at
    DESI's z_eff to separate the two; removed on request, restore it if the
    residual ever needs attributing.
    """
    if reference not in _REFERENCE_CHOICES:
        raise ValueError(f"reference must be one of {_REFERENCE_CHOICES}, "
                         f"got {reference!r}")
    gen, fid, dr1, dr1_sig = _mean_vectors(tracers, theory, data)
    ref = fid if reference == "fiducial" else dr1
    ref_label = ("DESI fiducial (Table 11)" if reference == "fiducial"
                 else "DESI DR1 (App. A)")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = np.arange(len(tracers))
    labels = [_DISPLAY.get(t, t) for t in tracers]

    for j, (ax, (ylabel, kind, _t11key, loc)) in enumerate(zip(axes, _MEAN_PANELS)):
        r_ref, r_gen, r_sig = ref[:, j], gen[:, j], dr1_sig[:, j]
        if kind == "ratio":
            resid = 100.0 * (r_gen / r_ref - 1.0)
            band = 100.0 * r_sig / np.abs(r_ref)
            unit = "  [%]"
            ylab = f"{ylabel}: generator / ref $-1${unit}"
        else:
            resid = r_gen - r_ref
            band = r_sig
            ylab = f"{ylabel}: generator $-$ ref"

        ax.axhline(0.0, color="0.45", lw=1.6, label=ref_label)
        spread = [resid]
        if reference == "dr1":
            ax.errorbar(x, np.zeros_like(x, dtype=float), yerr=band, fmt="none",
                        ecolor="0.45", elinewidth=1.2, capsize=4, capthick=1.2,
                        zorder=1, label=r"DESI $1\sigma$")
            spread.append(band)
            spread.append(-band)
        ax.plot(x, resid, "o", ms=7, color="tab:blue", zorder=3,
                label="generator")

        # Reference values along the top, anchored in axes coords.
        for i in range(len(tracers)):
            ax.text(i, 0.955, f"{r_ref[i]:.4g}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.0, color="0.35")

        allv = np.concatenate([np.atleast_1d(v) for v in spread] + [[0.0]])
        pad = max(allv.ptp(), 1.0 if kind == "ratio" else 1e-12)
        ax.set_ylim(allv.min() - 0.15 * pad, allv.max() + 0.30 * pad)
        ax.set_xlim(-0.5, len(tracers) - 0.5)
        ax.set_ylabel(ylab)
        ax.set_title(ylabel, fontsize=11)
        ax.legend(fontsize=7, loc=loc)

    if reference == "fiducial":
        # m's residual is ~1e-5 (CLASS shooting tolerance on the Omega_m route,
        # see CHANGELOG S66); state the scale rather than draw a sigma(m) band
        # that would fill the panel.
        axes[3].annotate(
            rf"$|m| \leq$ {np.abs(gen[:, 3] - ref[:, 3]).max():.1e}"
            "\n"
            rf"DESI $\sigma(m)$ = {dr1_sig[:, 3].min():.3f}–{dr1_sig[:, 3].max():.3f}",
            xy=(0.03, 0.03), xycoords="axes fraction", fontsize=7,
            color="0.3", va="bottom")

    for i, ax in enumerate(axes):
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, fontsize=8)
        # every panel now places its own legend
    if reference == "fiducial":
        sup = ("ShapeFit mean values vs DESI's FIDUCIAL (Table 11 / Appendix C; "
               "$m=0$ by Eq. 4.9).  At the fiducial cosmology the generator is "
               "1/1/Table 11/0 by construction, so this is a NULL TEST of "
               "conventions, not a measurement comparison.")
    else:
        sup = ("ShapeFit mean values vs DESI's DR1 MEASUREMENT (App. A, "
               "ShapeFit-alone), generator evaluated at the FIDUCIAL cosmology.  "
               "This is 'does DR1 agree with Planck-$\\Lambda$CDM' — a DESI "
               "result, not a test of this pipeline.")
    fig.suptitle(sup, fontsize=10)
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
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A sweep writes these from 24 processes that finish at different
            # times, so plotting mid-sweep will meet a half-written file. Skip
            # it rather than take the whole figure down over one unfinished job.
            print(f"  [mcmc] skipping unreadable {path.name}")
            continue
        for t, v in raw.items():
            if t in tracers and v.get("mcmc"):
                seeds.setdefault(t, {}).update(v["mcmc"])

    out = {}
    for t, runs in seeds.items():
        vals = list(runs.values())
        out[t] = {k: (float(np.mean([r[k] for r in vals])),
                      float(np.std([r[k] for r in vals])) if len(vals) > 1 else 0.0)
                  for k in vals[0]}
        if len(vals) > 1:
            print(f"  [mcmc] {t}: {len(vals)} seeds")
    return out


def plot_forecast(data, tracers, out_path, theory="rept"):
    """sigma per compressed parameter: DESI published vs Fisher vs MCMC.

    Modelled on bao/comparison_plots.py `_plot_forecast`, minus the bundle
    series -- shapefit has one covariance, so what varies is the ESTIMATOR:
    blue circle = Fisher, orange diamond = MCMC, black cross = DESI. Colour and
    marker both carry the estimator so the two series stay separable where they
    overlap (LRG2/QSO sit within a marker width of each other).
    """
    mcmc = _load_mcmc(tracers)
    x = np.arange(len(tracers), dtype=float)
    c_fisher, c_mcmc, c_desi = "tab:blue", "tab:orange", "black"

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
        ax.scatter(xf, yf, marker="o", s=34, color=c_fisher, linewidth=0,
                   zorder=4, label="Fisher")
        if xm:
            if any(e > 0 for e in em):
                ax.errorbar(xm, ym, yerr=em, fmt="none", ecolor=c_mcmc,
                            elinewidth=1.3, capsize=4, capthick=1.3, zorder=3)
            ax.scatter(xm, ym, marker="D", s=34, color=c_mcmc, linewidth=0,
                       zorder=4, label="MCMC")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, None)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")

    handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=7,
               markeredgewidth=1.6, color=c_desi, label="DESI published, 2411.12021 App. A"),
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=c_fisher, markeredgecolor="none", label="Fisher"),
        Line2D([0], [0], marker="D", linestyle="", markersize=7,
               markerfacecolor=c_mcmc, markeredgecolor="none",
               label="MCMC, error bar = seed rms"),
    ]
    axes[0].legend(handles=handles, loc="best", frameon=True, fontsize=9)
    axes[0].set_title(f"ShapeFit forecast vs DESI DR1 published  {theory.upper()}")
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
    p.add_argument("--reference", default="fiducial", choices=_REFERENCE_CHOICES,
                   help="`mean` plot only. fiducial: null test against Table 11 "
                        "(default). dr1: against DESI's measured compressed "
                        "values -- a DESI result, not a test of this code.")
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
        # Distinct filenames so the two references can coexist on disk; plots/
        # has no versioning and a rerun overwrites in place.
        stem = {"fiducial": "shapefit_mean_vs_fiducial",
                "dr1": "shapefit_mean_vs_dr1"}[args.reference]
        plot_mean(data, tracers, plots_dir() / f"{stem}.png", theory=th,
                  reference=args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
