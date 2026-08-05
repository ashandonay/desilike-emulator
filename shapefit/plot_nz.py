#!/usr/bin/env python
"""Plot n(z) per tracer bin -- the density the pipeline actually uses.

Not the raw table columns: this calls `bao_core.cov_nbar_per_slice`, so what is
drawn is what the covariance consumes, including the S85 split between DESI's
`NX` (single-tracer bins) and `N*frac/V` (the mixed LRG3_ELG1 bin, where <NX>
averages the two parents instead of summing them).

    python plot_nz.py                    # -> plots/nz_by_tracer.png
    python plot_nz.py --n-factor 0.5     # the design axis at half DR1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import core as sf_core  # noqa: E402
from util import ntracers, tracer_area  # noqa: E402

bao_core = sf_core.bao_core

# dataviz reference palette, categorical light slots 1-7, IN ORDER. Assigned by
# entity (fixed per tracer), never by rank, so a filtered rerun does not repaint.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7"]
BINS = ["BGS", "LRG1", "LRG2", "LRG3", "LRG3_ELG1", "ELG2", "QSO"]
COLOR = dict(zip(BINS, PALETTE))

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"

# Direct-label anchors, tuned once by looking at the render: (fraction along the
# bin, x-offset pt, y-offset pt). LRG2 and LRG3 peak within 0.01 in z of each
# other, so peak-anchoring collides -- LRG3 is moved onto its own descent.
_LABEL_AT = {
    "BGS":       (0.30, 0, 8),
    "LRG1":      (0.55, 0, 8),
    "LRG2":      (0.60, -14, 8),
    "LRG3":      (0.45, 16, 4),
    "LRG3_ELG1": (0.05, 8, 8),
    "ELG2":      (0.10, 6, 6),
    "QSO":       (0.42, 0, 8),
}
SURFACE = "#fcfcfb"


def gather(n_factor: float):
    from desilike.theories.primordial_cosmology import get_cosmo
    cosmo = get_cosmo("DESI")
    out = {}
    for t in BINS:
        bao_core._DESI_NX_CACHE.clear()
        z_mid, z_edges, frac, _ = bao_core._load_nz_slice_fractions(t, dataset="dr1")
        area = float(tracer_area(t, "dr1"))
        N = n_factor * float(ntracers(t, "dr1"))
        sky = area / 41252.96
        chi_lo = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 0]))
        chi_hi = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 1]))
        V = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * sky
        nbar, src = bao_core.cov_nbar_per_slice(t, frac, V, N, dataset="dr1")
        out[t] = dict(z_lo=z_edges[:, 0], z_hi=z_edges[:, 1], z_mid=z_mid,
                      nbar=np.asarray(nbar, float), src=src, area=area, N=N)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-factor", type=float, default=1.0,
                    help="multiple of the DR1 count (the design axis)")
    ap.add_argument("--out", type=Path,
                    default=_HERE.parent / "plots" / "nz_by_tracer.png")
    a = ap.parse_args()

    data = gather(a.n_factor)

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 11, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": INK_3, "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(11, 6.2))

    for t in BINS:
        d = data[t]
        # Piecewise-constant by construction -- draw it as steps rather than
        # implying an interpolation the pipeline never uses.
        z = np.empty(2 * len(d["z_lo"])); y = np.empty_like(z)
        z[0::2], z[1::2] = d["z_lo"], d["z_hi"]
        y[0::2], y[1::2] = d["nbar"], d["nbar"]
        mixed = d["src"].startswith("N*frac/V")
        ax.plot(z, y, color=COLOR[t], lw=2.0,
                ls="--" if mixed else "-",
                label=f"{t}" + ("  (mixed bin: N·frac/V)" if mixed else ""),
                zorder=3, solid_capstyle="round")
        # Direct labels: the bins tile z, so each lands in its own space and
        # identity never rests on colour alone.
        f, dx, dy = _LABEL_AT[t]
        i = min(int(round(f * (len(d["nbar"]) - 1))), len(d["nbar"]) - 1)
        ax.annotate(t, (d["z_mid"][i], d["nbar"][i]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=9.5, color=COLOR[t], zorder=4,
                    fontweight="medium")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1), numticks=12))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("redshift  $z$")
    ax.set_ylabel(r"$\bar{n}(z)$   [$h^3\,\mathrm{Mpc}^{-3}$]")
    ttl = "DESI DR1 n(z) per tracer bin — the density the covariance uses"
    if a.n_factor != 1.0:
        ttl += f"   (N = {a.n_factor:g} × DR1)"
    ax.set_title(ttl, color=INK, fontsize=13, pad=12, loc="left")
    ax.grid(True, which="major", color=INK_3, alpha=0.22, lw=0.6)
    ax.grid(True, which="minor", color=INK_3, alpha=0.10, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, labelcolor=INK_2,
              loc="lower left")

    src_note = ("solid = DESI NX·α(N)   ·   dashed = N·frac/V, used where ⟨NX⟩ "
                "averages a combined bin's parents instead of summing them (S85)")
    fig.text(0.008, 0.012, src_note, fontsize=8.5, color=INK_3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"wrote {a.out}")

    print(f"\n{'bin':11s} {'z range':>13s} {'slices':>7s} {'n̄ min':>10s} "
          f"{'n̄ max':>10s} {'source':>22s}")
    for t in BINS:
        d = data[t]
        print(f"{t:11s} {d['z_lo'][0]:5.2f}-{d['z_hi'][-1]:<7.2f} "
              f"{len(d['nbar']):7d} {d['nbar'].min():10.3e} "
              f"{d['nbar'].max():10.3e} {d['src']:>22s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
