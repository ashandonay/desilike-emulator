"""Visualize how the emulator inputs/outputs respond to N_tracers, at the fiducial
cosmology, for each DR1 tracer bin:

All figures land in plots/ (util.plots_dir) with a `bao_` prefix:

  bao_nz_vs_ntracers               nbar(z) = N_tracers * slice_fraction(z) /
                                   V_shell(z) (core.py:1201), per tracer (3x2),
                                   one colour per tracer, N in {0.5,1.0,1.5}x
                                   passed as linestyles.
  bao_nP_vs_z                      nbar*P(k_p, z) per slice -- the FKP
                                   signal-to-noise that decides which slices
                                   carry the constraint. Marks the V_eff-weighted
                                   nP and nP(z_eff); nP ~ 1 is the crossover from
                                   shot-noise- to sample-variance-dominated.
  bao_cov_scaling_vs_ntracers      2x2 (DH/rd, DM/rd) analytic-Gaussian Fisher cov
                                   swept over N_tracers (anisotropic tracers).
  bao_scaling_vs_ntracers_aniso    sigma(DH/rd), sigma(DM/rd), rho vs N_tracers,
                                   anisotropic tracers (qparqper).
  bao_scaling_vs_ntracers_iso      sigma(DV/rd) vs N_tracers for the isotropic
                                   (qiso) sparse tracers BGS & QSO.
  bao_scaling_vs_ntracers_ref      per-tracer sigma scaling against 1/sqrt(N) and
                                   1/N guides -- the shape comparison the two
                                   figures above deliberately omit
                                   (show_guides=False) to keep them readable.

Run (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        OMP_NUM_THREADS=1 ~/miniconda3/envs/emulator/bin/python plot_nz_cov_scaling.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import core
import config_space as cc
from fourier_space import _q_fisher_from_bao_likelihood_info
from util import TRACER_CONFIGS, ntracers, plots_dir
from desilike.theories.primordial_cosmology import get_cosmo

TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
ANISO = [t for t in TRACERS if t not in core._ISO_TRACERS]   # 2x2 (DH,DM)
SPARSE = [t for t in TRACERS if t in core._ISO_TRACERS]      # qiso -> DV only
DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}

# One colour per tracer, shared across all figures. Anisotropic = the 2x2 set;
# BGS/QSO get distinct colours.
TRACER_COLOR = {
    "LRG1": "tab:blue", "LRG2": "tab:orange", "LRG3_ELG1": "tab:green",
    "ELG2": "tab:red", "BGS": "tab:purple", "QSO": "tab:brown",
}
N_FACTORS = [0.5, 1.0, 1.5]                                  # v2 box edges + centre
N_LS = {0.5: ":", 1.0: "-", 1.5: "--"}                      # factor -> linestyle
_AREA = cc._AREA                                             # DR1 footprint (7500)
_SKY = 41252.96


def _fid_theta():
    theta, hrdrag = core._to_bao_cosmo_params({**core.PARAM_DEFAULTS, **cc._FID})
    return theta, hrdrag


def nbar_of_z(tracer, cosmo, N_tracers, dataset="dr1"):
    """nbar(z) per slice exactly as build_bao_likelihood does (core.py:1201)."""
    z_mid, z_edges, frac, _ = core._load_nz_slice_fractions(
        tracer, dataset=dataset)
    chi_lo = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 0]))
    chi_hi = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 1]))
    V_bin = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * (_AREA / _SKY)
    return z_mid, (float(N_tracers) * frac) / np.maximum(V_bin, 1.0)


K_PIVOT = 0.14  # h/Mpc, BAO Fisher kernel peak (matches core._compute_z_eff_from_nz)


def nP_of_z(tracer, cosmo, fo, b1, z_eff=None, dataset="dr1"):
    """n̄P(z) per slice at k=K_PIVOT, using the published nominal-design n̄(z).

    n̄ = nbar_file (the DESI design local density, from parse_desi_nz), and
    P = b1² P_lin(k_p, z). Returns (z_mid, nP, V_eff-weighted nP scalar,
    nP at z_eff). nP(z_eff) interpolates n̄ onto z_eff and evaluates P there;
    None if z_eff is not given.
    """
    z_mid, z_edges, frac, nbar_file = core._load_nz_slice_fractions(
        tracer, dataset=dataset)
    P_g = np.array([b1 ** 2 * float(core._linear_pk_1d(fo, z=float(z))(
        np.array([K_PIVOT]))[0]) for z in z_mid])
    nP = np.asarray(nbar_file, dtype=np.float64) * P_g
    # V_eff-weighted representative nP (FKP-weight² × shell volume).
    chi_lo = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 0]))
    chi_hi = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 1]))
    V_bin = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * (_AREA / _SKY)
    w = V_bin * (nP / (1.0 + nP)) ** 2
    nP_eff = float(np.sum(nP * w) / np.sum(w)) if w.sum() > 0 else float(np.max(nP))
    nP_zeff = None
    if z_eff is not None:
        ze = float(z_eff)
        nb_ze = float(np.interp(ze, z_mid, nbar_file))
        P_ze = b1 ** 2 * float(core._linear_pk_1d(fo, z=ze)(np.array([K_PIVOT]))[0])
        nP_zeff = nb_ze * P_ze
    return z_mid, nP, nP_eff, nP_zeff


def _build(tracer, theta, hrdrag, N_tracers, apmode):
    cfg = TRACER_CONFIGS[tracer]
    info = core.build_bao_likelihood(
        N_tracers=float(N_tracers), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
        area=_AREA, apmode=apmode)
    info["likelihood"](**info["params"])
    return info


def cov2x2(tracer, theta, hrdrag, N_tracers):
    """Analytic-Gaussian Fisher 2x2 cov in (DH/rd, DM/rd)."""
    info = _build(tracer, theta, hrdrag, N_tracers, "qparqper")
    cov_q = np.linalg.inv(_q_fisher_from_bao_likelihood_info(info))
    tmpl = info["template"]
    J = np.diag([float(tmpl.DH_over_rd_fid), float(tmpl.DM_over_rd_fid)])
    return J @ cov_q @ J.T


def sigma_dv_iso(tracer, theta, hrdrag, N_tracers):
    """σ(DV/rd) from the isotropic qiso fit (1 parameter)."""
    info = _build(tracer, theta, hrdrag, N_tracers, "qiso")
    cov_q = np.linalg.inv(_q_fisher_from_bao_likelihood_info(info))
    return float(np.sqrt(cov_q[0, 0])) * float(info["template"].DV_over_rd_fid)


# ---------------------------------------------------------------------------
def plot_nz(cosmo, out):
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for ax, t in zip(axes.ravel(), TRACERS):
        passed = ntracers(t, "dr1")
        for f in N_FACTORS:
            z, nb = nbar_of_z(t, cosmo, f * passed)
            ax.plot(z, nb, color=TRACER_COLOR[t], ls=N_LS[f], lw=1.6)
        ax.set_title(DISPLAY.get(t, t), color=TRACER_COLOR[t])
        ax.set_xlabel("z"); ax.set_ylabel(r"$\bar n(z)\ [(h/{\rm Mpc})^3]$")
        ax.set_yscale("log"); ax.grid(alpha=0.25, ls="--", lw=0.6)
    handles = [Line2D([0], [0], color="k", ls=N_LS[f], lw=1.6,
                      label=f"{f:g}× passed") for f in N_FACTORS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("n(z) vs N_tracers per DR1 tracer bin  (fiducial cosmology, DR1 area)",
                 fontsize=14, y=1.05)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_cov(theta, hrdrag, out, n_pts=8):
    xfac = np.linspace(N_FACTORS[0], N_FACTORS[-1], n_pts)
    data = {}
    for t in ANISO:
        passed = ntracers(t, "dr1")
        data[t] = np.array([cov2x2(t, theta, hrdrag, f * passed) for f in xfac])
        print(f"  {t}: 2x2 cov over {n_pts} N_tracers")
    labels = [[r"${\rm Var}(D_H/r_d)$", None],
              [r"${\rm Cov}(D_H/r_d, D_M/r_d)$", r"${\rm Var}(D_M/r_d)$"]]
    idx = [[(0, 0), None], [(0, 1), (1, 1)]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            if idx[r][c] is None:
                ax.axis("off"); continue
            i, j = idx[r][c]
            for t in ANISO:
                ax.plot(xfac, data[t][:, i, j], color=TRACER_COLOR[t], lw=1.6,
                        label=DISPLAY.get(t, t))
            ax.set_title(labels[r][c])
            ax.set_xlabel(r"$N_{\rm tracers} / {\rm passed}$")
            if i == j:
                ax.set_yscale("log")              # positive variances
            else:
                ax.axhline(0.0, color="gray", lw=0.8)  # signed (negative) cov
            ax.grid(alpha=0.25, ls="--", lw=0.6)
            ax.axvline(1.0, color="gray", lw=0.8, ls=":")
            if (r, c) == (0, 0):
                ax.legend(fontsize=9, title="tracer")
    fig.suptitle("2x2 (DH/rd, DM/rd) analytic-Gaussian covariance vs N_tracers\n"
                 "(dotted line = passed; anisotropic tracers only)", fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_corr(tracers, theta, hrdrag, out, n_pts=8, title=None, show_guides=True):
    """Corner-plot 2x2 triangle vs N_tracers for the given tracers: the diagonals
    carry σ(DH/rd) and σ(DM/rd) (amplitude, which scales with N), the lower
    off-diagonal the correlation coefficient ρ(DH/rd, DM/rd) (shape, which does
    not). A *pure* correlation matrix has 1s on the diagonal, so this mixes σ on
    the diagonal so all three panels are informative — same layout as plot_cov.

    The 2x2 (DH/rd, DM/rd) cov uses the anisotropic qparqper fit, so isotropic
    sparse tracers (BGS, QSO) are shown here as the degenerate ellipse they would
    yield if DH/DM were fit separately.
    """
    xfac = np.linspace(N_FACTORS[0], N_FACTORS[-1], n_pts)
    sig, rho = {}, {}
    for t in tracers:
        passed = ntracers(t, "dr1")
        C = np.array([cov2x2(t, theta, hrdrag, f * passed) for f in xfac])
        sig[t] = np.sqrt(C[:, [0, 1], [0, 1]])                     # (n_pts, 2)
        rho[t] = C[:, 0, 1] / np.sqrt(C[:, 0, 0] * C[:, 1, 1])
        print(f"  {t}: ρ(DH,DM)={rho[t][0]:.3f}->{rho[t][-1]:.3f} over {n_pts} N_tracers")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes[0, 1].axis("off")
    for t in tracers:
        lbl = DISPLAY.get(t, t)
        axes[0, 0].plot(xfac, sig[t][:, 0], color=TRACER_COLOR[t], lw=1.6, label=lbl)
        axes[1, 1].plot(xfac, sig[t][:, 1], color=TRACER_COLOR[t], lw=1.6, label=lbl)
        axes[1, 0].plot(xfac, rho[t], color=TRACER_COLOR[t], lw=1.6, label=lbl)
    axes[0, 0].set_title(r"$\sigma(D_H/r_d)$"); axes[0, 0].set_yscale("log")
    axes[1, 1].set_title(r"$\sigma(D_M/r_d)$"); axes[1, 1].set_yscale("log")
    axes[1, 0].set_title(r"$\rho(D_H/r_d,\ D_M/r_d)$")
    # Optional per-tracer slope guides on the σ panels: σ ∝ 1/√N (nP≈1 crossover,
    # dotted) and σ ∝ 1/N (nP≪1 shot-noise, long-dash), each anchored to the
    # tracer's own left-edge value. Off by default for the headline multi-tracer
    # plot (too busy); the per-tracer guides live in plot_scaling_ref instead.
    if show_guides:
        for ax, col in ((axes[0, 0], 0), (axes[1, 1], 1)):
            for i, t in enumerate(tracers):
                ax.plot(xfac, sig[t][0, col] * np.sqrt(xfac[0] / xfac),
                        color=TRACER_COLOR[t], lw=1.0, ls=":",
                        label=r"$\propto 1/\sqrt{N_{\rm tracers}}$" if i == 0 else None)
                ax.plot(xfac, sig[t][0, col] * (xfac[0] / xfac),
                        color=TRACER_COLOR[t], lw=1.0, ls=(0, (6, 2)),
                        label=r"$\propto 1/N_{\rm tracers}$" if i == 0 else None)
    # fully autoscaled (no ρ=0 reference line) so the ~1% N-dependence is visible
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.set_xlabel(r"$N_{\rm tracers} / {\rm passed}$")
        ax.grid(alpha=0.25, ls="--", lw=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # legend just to the right of the upper-left box (top-left of the empty quadrant)
    axes[0, 1].legend(handles, labels, fontsize=10, title="tracer", loc="upper left")
    fig.suptitle(title or "(DH/rd, DM/rd) corner view vs N_tracers", fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_dv_sparse(theta, hrdrag, out, n_pts=8, show_guides=True):
    xfac = np.linspace(N_FACTORS[0], N_FACTORS[-1], n_pts)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, t in enumerate(SPARSE):
        passed = ntracers(t, "dr1")
        sig = np.array([sigma_dv_iso(t, theta, hrdrag, f * passed) for f in xfac])
        ax.plot(xfac, sig, color=TRACER_COLOR[t], lw=1.6, label=DISPLAY.get(t, t))
        # crossover (σ∝1/√N, dotted) and shot-noise (σ∝1/N, long-dash) references,
        # anchored to this tracer's left-edge value. Off for the headline plot;
        # per-tracer guides live in plot_scaling_ref instead.
        if show_guides:
            ax.plot(xfac, sig[0] * np.sqrt(xfac[0] / xfac), color=TRACER_COLOR[t],
                    lw=1.0, ls=":", label=r"$\propto 1/\sqrt{N_{\rm tracers}}$" if i == 0 else None)
            ax.plot(xfac, sig[0] * (xfac[0] / xfac), color=TRACER_COLOR[t],
                    lw=1.0, ls=(0, (6, 2)), label=r"$\propto 1/N_{\rm tracers}$" if i == 0 else None)
        print(f"  {t}: σ(DV) over {n_pts} N_tracers")
    ax.set_xlabel(r"$N_{\rm tracers} / {\rm passed}$")
    ax.set_ylabel(r"$\sigma(D_V/r_d)$")
    ax.set_yscale("log"); ax.grid(alpha=0.25, ls="--", lw=0.6)
    # reorder so the slope-reference entries (1/√N, 1/N) sit last, after the tracers
    h, l = ax.get_legend_handles_labels()
    order = [i for i, lb in enumerate(l) if "propto" not in lb] + \
            [i for i, lb in enumerate(l) if "propto" in lb]
    ax.legend([h[i] for i in order], [l[i] for i in order],
              fontsize=10, title="isotropic tracer (qiso)",
              loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title("σ(DV/rd) vs N_tracers — isotropic tracers (qiso)", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# Distance-component colours for the per-tracer reference plot (one tracer per
# subplot, so the curves are coloured by component, not by tracer).
_COMP_COLOR = {"DH": "tab:blue", "DM": "tab:orange", "DV": "tab:green"}


def _curve_with_guides(ax, x, y, color, label):
    """Solid σ(N) curve plus faint per-curve slope guides: σ∝1/√N (dotted,
    crossover) and σ∝1/N (long-dash, shot-noise), anchored to the left edge."""
    ax.plot(x, y, color=color, lw=1.8, label=label)
    ax.plot(x, y[0] * np.sqrt(x[0] / x), color=color, lw=1.0, ls=":", alpha=0.5)
    ax.plot(x, y[0] * (x[0] / x), color=color, lw=1.0, ls=(0, (6, 2)), alpha=0.5)


def plot_scaling_ref(theta, hrdrag, out, n_pts=8):
    """Per-tracer σ-scaling reference: 3x2 subplots, one tracer each, with the
    σ∝1/√N (crossover) and σ∝1/N (shot-noise) slope guides drawn against each
    component separately. Anisotropic tracers show σ(DH/rd) and σ(DM/rd); the
    isotropic sparse tracers (BGS, QSO) show σ(DV/rd)."""
    xfac = np.linspace(N_FACTORS[0], N_FACTORS[-1], n_pts)
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for ax, t in zip(axes.ravel(), TRACERS):
        passed = ntracers(t, "dr1")
        if t in SPARSE:
            sig = np.array([sigma_dv_iso(t, theta, hrdrag, f * passed) for f in xfac])
            _curve_with_guides(ax, xfac, sig, _COMP_COLOR["DV"], r"$\sigma(D_V/r_d)$")
        else:
            C = np.array([cov2x2(t, theta, hrdrag, f * passed) for f in xfac])
            s = np.sqrt(C[:, [0, 1], [0, 1]])                       # (n_pts, 2)
            _curve_with_guides(ax, xfac, s[:, 0], _COMP_COLOR["DH"], r"$\sigma(D_H/r_d)$")
            _curve_with_guides(ax, xfac, s[:, 1], _COMP_COLOR["DM"], r"$\sigma(D_M/r_d)$")
        ax.set_title(DISPLAY.get(t, t))
        ax.set_xlabel(r"$N_{\rm tracers} / {\rm passed}$")
        ax.grid(alpha=0.25, ls="--", lw=0.6)  # linear y-axis
        print(f"  {t}: σ scaling over {n_pts} N_tracers")
    # single figure legend: components (solid, coloured) + the two slope guides.
    handles = [
        Line2D([0], [0], color=_COMP_COLOR["DH"], lw=1.8, label=r"$\sigma(D_H/r_d)$"),
        Line2D([0], [0], color=_COMP_COLOR["DM"], lw=1.8, label=r"$\sigma(D_M/r_d)$"),
        Line2D([0], [0], color=_COMP_COLOR["DV"], lw=1.8, label=r"$\sigma(D_V/r_d)$"),
        Line2D([0], [0], color="0.4", lw=1.0, ls=":", label=r"$\propto 1/\sqrt{N_{\rm tracers}}$"),
        Line2D([0], [0], color="0.4", lw=1.0, ls=(0, (6, 2)), label=r"$\propto 1/N_{\rm tracers}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Per-tracer σ(D/rd) scaling vs N_tracers with 1/√N and 1/N guides",
                 fontsize=14, y=1.06)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_nP(cosmo, theta, hrdrag, out):
    """n̄P(z) at k=K_PIVOT for the nominal design, all tracers on one axis."""
    fo = cosmo.get_fourier()
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    def _regime(x):
        # nP≫1 cosmic-variance limited; nP≈1 crossover (σ∝1/√N); nP≪1 shot-noise (σ∝1/N).
        return "CV-limited" if x >= 2.0 else ("crossover" if x >= 0.5 else "shot-noise")

    print(f"{'tracer':12s} {'b1':>6s} {'peak nP':>9s} {'Veff-wtd nP':>12s} "
          f"{'nP(z_eff)':>10s}  regime")
    for t in TRACERS:
        apmode = "qiso" if t in SPARSE else "qparqper"
        b1 = float(_build(t, theta, hrdrag, ntracers(t, "dr1"), apmode)["params"]["b1"])
        z_eff = float(TRACER_CONFIGS[t]["z_eff"])
        z, nP, nP_eff, nP_zeff = nP_of_z(t, cosmo, fo, b1, z_eff=z_eff)
        ax.plot(z, nP, color=TRACER_COLOR[t], lw=1.8,
                label=f"{DISPLAY.get(t, t)}  (nP$_{{\\rm eff}}$={nP_eff:.2f}, {_regime(nP_eff)})")
        # marker = n̄P evaluated at the single effective redshift z_eff
        ax.plot(z_eff, nP_zeff, marker="o", ms=9, ls="none", color=TRACER_COLOR[t],
                mec="black", mew=1.0, zorder=5)
        print(f"{t:12s} {b1:6.2f} {nP.max():9.2f} {nP_eff:12.2f} {nP_zeff:10.2f}  {_regime(nP_eff)}")
    # proxy legend entry for the z_eff markers
    ax.plot([], [], marker="o", ms=9, ls="none", color="0.7", mec="black", mew=1.0,
            label=r"value at $z_{\rm eff}$")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7)
    tr = ax.get_yaxis_transform()  # x in axes-fraction, y in data units
    ax.text(0.012, 1.18, "↑ cosmic-variance limited  (nP > 1,  σ flatter than 1/√N)",
            transform=tr, va="bottom", fontsize=8, color="0.3")
    ax.text(0.012, 0.85, "↓ shot-noise limited  (nP < 1,  σ steeper than 1/√N → 1/N)",
            transform=tr, va="top", fontsize=8, color="0.3")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\bar n P(k_{\rm p})$  at $k_{\rm p}=%.2f\ h/{\rm Mpc}$" % K_PIVOT)
    ax.set_yscale("log"); ax.grid(alpha=0.25, ls="--", lw=0.6)
    ax.legend(fontsize=9, ncol=2)
    fig.suptitle("Nominal-design "
                 r"$\bar n P$ per DR1 tracer bin  (fiducial cosmology)", fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    theta, hrdrag = _fid_theta()
    cosmo = get_cosmo(("DESI", dict(theta)))
    print("=== n(z) figure ===")
    plot_nz(cosmo, str(plots_dir() / "bao_nz_vs_ntracers.png"))
    print("=== nP(z) figure ===")
    plot_nP(cosmo, theta, hrdrag, str(plots_dir() / "bao_nP_vs_z.png"))
    print("=== covariance scaling figure (anisotropic) ===")
    plot_cov(theta, hrdrag, str(plots_dir() / "bao_cov_scaling_vs_ntracers.png"))
    print("=== σ/ρ scaling figure (anisotropic, no guides) ===")
    plot_corr(ANISO, theta, hrdrag, str(plots_dir() / "bao_scaling_vs_ntracers_anisotropic.png"),
              title="σ(DH/rd), σ(DM/rd), ρ vs N_tracers — anisotropic tracers (qparqper)",
              show_guides=False)
    print("=== σ(DV) scaling figure (isotropic, no guides) ===")
    plot_dv_sparse(theta, hrdrag, str(plots_dir() / "bao_scaling_vs_ntracers_isotropic.png"),
                   show_guides=False)
    print("=== per-tracer σ scaling reference (with 1/√N, 1/N guides) ===")
    plot_scaling_ref(theta, hrdrag, str(plots_dir() / "bao_scaling_vs_ntracers_ref.png"))


if __name__ == "__main__":
    main()
