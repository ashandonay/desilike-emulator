"""Visualize how the emulator inputs/outputs respond to N_tracers, at the fiducial
cosmology, for each DR1 tracer bin:

  Fig 1  nz_vs_ntracers.png        nbar(z) = N_tracers * slice_fraction(z) / V_shell(z)
                                   (core.py:1201), per tracer (3x2), one colour per
                                   tracer, N in {0.5,1.0,1.5}x passed as linestyles.
  Fig 2  cov_scaling_vs_ntracers   2x2 (DH/rd, DM/rd) analytic-Gaussian Fisher cov
                                   swept over N_tracers (anisotropic tracers).
  Fig 3  dv_scaling_sparse.png     σ(DV/rd) vs N_tracers for the isotropic (qiso)
                                   sparse tracers BGS & QSO (distinct colours).

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
from util import TRACER_CONFIGS, ntracers
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


def nbar_of_z(tracer, cosmo, N_tracers):
    """nbar(z) per slice exactly as build_bao_likelihood does (core.py:1201)."""
    z_mid, z_edges, frac, _ = core._load_nz_slice_fractions(tracer)
    chi_lo = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 0]))
    chi_hi = np.asarray(cosmo.comoving_radial_distance(z_edges[:, 1]))
    V_bin = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * (_AREA / _SKY)
    return z_mid, (float(N_tracers) * frac) / np.maximum(V_bin, 1.0)


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


def plot_dv_sparse(theta, hrdrag, out, n_pts=8):
    xfac = np.linspace(N_FACTORS[0], N_FACTORS[-1], n_pts)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for t in SPARSE:
        passed = ntracers(t, "dr1")
        sig = [sigma_dv_iso(t, theta, hrdrag, f * passed) for f in xfac]
        ax.plot(xfac, sig, color=TRACER_COLOR[t], lw=1.8, label=DISPLAY.get(t, t))
        print(f"  {t}: σ(DV) over {n_pts} N_tracers")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$N_{\rm tracers} / {\rm passed}$")
    ax.set_ylabel(r"$\sigma(D_V/r_d)$")
    ax.set_yscale("log"); ax.grid(alpha=0.25, ls="--", lw=0.6)
    ax.legend(fontsize=11, title="sparse tracer (qiso)")
    ax.set_title("σ(DV/rd) vs N_tracers — isotropic sparse tracers\n"
                 "(dotted line = passed)", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    theta, hrdrag = _fid_theta()
    cosmo = get_cosmo(("DESI", dict(theta)))
    print("=== n(z) figure ===")
    plot_nz(cosmo, str(_HERE / "nz_vs_ntracers.png"))
    print("=== covariance scaling figure (anisotropic) ===")
    plot_cov(theta, hrdrag, str(_HERE / "cov_scaling_vs_ntracers.png"))
    print("=== DV scaling figure (sparse) ===")
    plot_dv_sparse(theta, hrdrag, str(_HERE / "dv_scaling_sparse.png"))


if __name__ == "__main__":
    main()
