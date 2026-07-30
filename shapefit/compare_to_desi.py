"""Absolute comparison of the ShapeFit forecast against DESI DR1 products.

The forecast sigmas are a product of exactly two ingredients -- the covariance
C and the theory derivatives dP/dtheta -- so this script compares each against
the DESI DR1 full-shape data products shipped under
``~/data/desi/bao_dr1/likelihoods/``, rather than against internal
self-consistency (which is all `regress_sigmas.py` and `validate_forecast.py`
can establish).

Checks (select with --check, default all):

  shot : our FKP V_eff-matched n_eff vs DESI's MEASURED effective shot noise
         (num_shotnoise / norm) from the full-shape bundle. One scalar, no
         model dependence -- if this is wrong every sigma is wrong by the same
         factor and nothing downstream is interpretable. Bundle-only, so LRG2
         only (see _FS_BUNDLES).
  pk   : our fiducial theory P0, P2 vs DESI's measured multipoles on the same
         k bins. Tests b1 x Kaiser amplitude x FoG damping. Bundle-only.
         NOTE the measured spectra are window-convolved and ours are not, so
         this is a shape/amplitude sanity check, not an equality test.
  cov  : our analytic Gaussian+SSC covariance vs DESI's EZmock covariance,
         element by element on the identical (ell, k) grid. Available for all
         tracers with a covariance file.
         *** NOT apples-to-apples. *** DESI's covariance -- plain or rotated --
         is the covariance of a WINDOW-CONVOLVED estimator, and ours is not
         convolved, so the absolute level here is meaningless and the spread
         ACROSS tracers is not a density-response signal either (the window
         differs per footprint). This check is only good for k-SHAPE trends at
         fixed tracer.
         The apples-to-apples version is C_obs = M C_kin M^T, with M the
         window's 72x1047 value and C_kin our covariance on the window's own
         theory grid (CHANGELOG S19 has the recipe). Doing that on LRG2 turns
         a raw ratio of 1.586 into 0.815 with the right correlation structure
         (P0 nearest-neighbour 0.79 vs DESI's 0.67), i.e. our Gaussian
         covariance is ~18% LOW, not 60% high -- the familiar non-Gaussian
         deficit. Not wired in here: it needs a per-tracer bundle and a
         1047-bin auxiliary covariance (36 s).
  sigma: rebuild the Fisher with DESI's covariance substituted for ours
         (core.build_shapefit_likelihood(cov_override=...)) and report how far
         the four sigmas move. Attributes any sigma gap to the covariance vs
         the model.

Grid alignment
--------------
Our klim (0.02, 0.2, 0.005) x ells (0, 2) gives 72 entries. DESI's plain
covariance files are 240x240 (ells 0,2,4 x 80 bins, edges at multiples of
0.005 from 0), so ours is exactly their bins 4..39 for ells 0 and 2; the
rotated/thetacut products are already 36 bins. Rows are matched on k_EDGES,
never on k values: DESI stores mode-weighted effective bin centers (0.0227 for
the 0.020-0.025 bin), which do not equal bin midpoints.

Caveats that are physics, not bugs
----------------------------------
* DESI's baseline full-shape fit is a JOINT ``power+bao-recon`` fit with
  REPT-velocileptors, a rotated window and a theta-cut of 0.05. We are
  pre-recon power-only, Kaiser, windowless. Their published sigma(qiso) and
  sigma(qap) are therefore tighter than any power-only forecast can be.
* LRG3_ELG1 has NO DESI full-shape counterpart: the DR1 full-shape analysis
  splits LRG 0.8-1.1 as LRG-only, while our bin is the combined LRG+ELG1
  sample used by the BAO analysis. Densities differ, so that bin is reported
  but flagged.
* QSO z_eff differs from DESI's 1.491 by ~10% by design (bao CHANGELOG S18:
  ours is Fisher-weighted, DESI's is volume-weighted). f*sigma8 evolves fast,
  so never compare f_sigmar across that Delta z without matching z first.

Usage (from shapefit/, emulator env):
    python compare_to_desi.py                          # all checks, all tracers
    python compare_to_desi.py --check shot pk cov sigma --tracers LRG2
    python compare_to_desi.py --check cov --rotated --plot
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import h5py

import fourier_space
from fourier_space import sf_core
from util import ntracers

_LIK_DIR = Path.home() / "data" / "desi" / "bao_dr1" / "likelihoods"
_COV_DIR = _LIK_DIR / "covariance"

TRACERS_ALL = ("BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO")

# DESI full-shape sample name per tracer bin. LRG3_ELG1 maps to LRG-only
# because the DR1 full-shape analysis does not combine LRG+ELG1 (see caveats).
_DESI_SAMPLE = {
    "BGS": "BGS_BRIGHT-21.5_GCcomb_z0.1-0.4",
    "LRG1": "LRG_GCcomb_z0.4-0.6",
    "LRG2": "LRG_GCcomb_z0.6-0.8",
    "LRG3_ELG1": "LRG_GCcomb_z0.8-1.1",
    "ELG2": "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "QSO": "QSO_GCcomb_z0.8-2.1",
}

# Bins whose DESI sample is not the same galaxy sample as ours.
_SAMPLE_MISMATCH = {
    "LRG3_ELG1": "DESI full-shape uses LRG-only in 0.8-1.1; ours is LRG+ELG1",
}

# Full data bundles (data vector + covariance + window). DR1 ships these per
# tracer, but only LRG2 has been fetched locally so far.
_FS_BUNDLES = {
    "LRG2": _LIK_DIR / (
        "likelihood_spectrum-poles-rotated_syst-hod_"
        "LRG_GCcomb_z0.6-0.8_thetacut0.05.h5"
    ),
}

# DESI's published (volume-weighted) effective redshifts, for reference only.
# Table 1 of arXiv:2411.12021 -- the FULL-SHAPE z_eff. LRG3 and QSO previously
# carried 0.930 and 1.484, which are the BAO paper's values for those bins.
_DESI_ZEFF = {"BGS": 0.295, "LRG1": 0.510, "LRG2": 0.706,
              "LRG3_ELG1": 0.919, "ELG2": 1.317, "QSO": 1.491}

FID_SAMPLE = {
    "omega_cdm": 0.1200,
    "omega_b": 0.02237,
    "h": 0.6736,
    "ln10A_s": 3.036394,
    "n_s": 0.9649,
}

_OUR_ELLS = (0, 2)


# ---------------------------------------------------------------------------
# DESI product readers
# ---------------------------------------------------------------------------
def _cov_path(tracer: str, rotated: bool, thetacut: bool) -> Optional[Path]:
    sample = _DESI_SAMPLE[tracer]
    stem = "covariance_spectrum-poles-rotated" if rotated else "covariance_spectrum-poles"
    # The rotated products only exist with the theta-cut applied.
    suffix = "_thetacut0.05" if (thetacut or rotated) else ""
    path = _COV_DIR / f"{stem}_{sample}{suffix}.h5"
    return path if path.exists() else None


def _read_observable_grid(group) -> Tuple[np.ndarray, np.ndarray]:
    """Flattened (ell, k_edges) row labels of a DESI observable group, in the
    same order the covariance rows are stored."""
    ells, edges = [], []
    for key in sorted(group.keys(), key=lambda s: (not s.isdigit(), s)):
        if not key.isdigit():
            continue
        sub = group[key]
        ke = np.asarray(sub["k_edges"][:], dtype=np.float64)
        ells.append(np.full(ke.shape[0], int(key)))
        edges.append(ke)
    return np.concatenate(ells), np.concatenate(edges, axis=0)


def _spectrum_group(f: h5py.File):
    """DESI stores the multipoles either directly under observable/ (covariance
    files) or nested under observable/spectrum/ (joint bundles)."""
    obs = f["observable"]
    return obs["spectrum"] if "spectrum" in obs else obs


def _select_rows(
    file_ells: np.ndarray,
    file_edges: np.ndarray,
    target_edges: np.ndarray,
    atol: float = 1e-6,
) -> np.ndarray:
    """Indices of the DESI rows matching our (ell, k_edges) grid, in OUR order.

    Matching is on edges, not centers (DESI centers are mode-weighted).
    """
    idx = []
    for ell in _OUR_ELLS:
        for lo, hi in target_edges:
            hit = np.where(
                (file_ells == ell)
                & (np.abs(file_edges[:, 0] - lo) < atol)
                & (np.abs(file_edges[:, 1] - hi) < atol)
            )[0]
            if hit.size != 1:
                raise ValueError(
                    f"expected exactly 1 DESI row for ell={ell} "
                    f"k=[{lo:.4f},{hi:.4f}], found {hit.size}"
                )
            idx.append(int(hit[0]))
    return np.asarray(idx, dtype=int)


def read_desi_cov(path: Path, target_edges: np.ndarray) -> np.ndarray:
    """DESI covariance restricted to our (ells, k) grid, in our row order."""
    with h5py.File(path, "r") as f:
        cov = np.asarray(f["value" if "value" in f else "covariance/value"][:],
                         dtype=np.float64)
        grp = f["covariance/observable"] if "covariance" in f else f["observable"]
        grp = grp["spectrum"] if "spectrum" in grp else grp
        file_ells, file_edges = _read_observable_grid(grp)
    if cov.shape[0] != file_ells.size:
        raise ValueError(
            f"{path.name}: covariance is {cov.shape} but the observable grid "
            f"has {file_ells.size} rows"
        )
    idx = _select_rows(file_ells, file_edges, target_edges)
    return cov[np.ix_(idx, idx)]


def read_desi_bundle(path: Path, target_edges: np.ndarray) -> Dict:
    """Measured multipoles, effective shot noise and covariance from a bundle."""
    with h5py.File(path, "r") as f:
        spec = _spectrum_group(f)
        file_ells, file_edges = _read_observable_grid(spec)
        values, shot = [], []
        for key in sorted(spec.keys(), key=lambda s: (not s.isdigit(), s)):
            if not key.isdigit():
                continue
            sub = spec[key]
            values.append(np.asarray(sub["value"][:], dtype=np.float64))
            norm = np.asarray(sub["norm"][:], dtype=np.float64)
            num = np.asarray(sub["num_shotnoise"][:], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                shot.append(np.where(norm > 0, num / np.where(norm > 0, norm, 1.0),
                                     np.nan))
        values = np.concatenate(values)
        shot = np.concatenate(shot)
        cov = np.asarray(f["covariance/value"][:], dtype=np.float64)
        cov_grp = f["covariance/observable"]
        cov_grp = cov_grp["spectrum"] if "spectrum" in cov_grp else cov_grp
        cov_ells, cov_edges = _read_observable_grid(cov_grp)

    idx = _select_rows(file_ells, file_edges, target_edges)
    cidx = _select_rows(cov_ells, cov_edges, target_edges)
    finite_shot = shot[np.isfinite(shot) & (shot > 0)]
    return {
        "data": values[idx],
        "cov": cov[np.ix_(cidx, cidx)],
        "P_shot": float(np.median(finite_shot)) if finite_shot.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Our side
# ---------------------------------------------------------------------------
def our_forecast(tracer: str, cov_override: Optional[np.ndarray] = None,
                 theory: str = "kaiser") -> Dict:
    """build_shapefit_likelihood at the DESI fiducial cosmology and the DR1
    passed count, plus the marginalized sigmas."""
    theta = sf_core._to_shapefit_cosmo_params(
        {**FID_SAMPLE, "N_tracers": ntracers(tracer, "dr1")}
    )
    from desilike.theories.galaxy_clustering import (
        KaiserTracerPowerSpectrumMultipoles as _Kaiser,
        REPTVelocileptorsTracerPowerSpectrumMultipoles as _REPT)
    _cls = {"kaiser": _Kaiser, "rept": _REPT}.get(theory)
    if _cls is None:
        raise ValueError(f"theory must be 'kaiser' or 'rept', got {theory!r}")
    # ALWAYS pass theory_cls explicitly, for BOTH theories. Leaving the Kaiser
    # branch implicit made it inherit build_shapefit_likelihood's default, and
    # S22 flipped that default to REPT -- so "kaiser" silently became REPT and
    # every comparison plot showed two identical curves.
    # Route theory_kwargs through core's resolver too: the preset dict is keyed
    # by tracer TYPE (tracers.yaml), not by tracer BIN, and duplicating that
    # lookup here is what broke this for every bin but LRG2.
    kw = dict(theory_cls=_cls,
              theory_kwargs=sf_core.default_theory_kwargs(_cls, tracer))
    info = sf_core.build_shapefit_likelihood(
        N_tracers=float(ntracers(tracer, "dr1")),
        theta_cosmo=theta,
        tracer_bin=tracer,
        cov_override=cov_override,
        **kw,
    )
    cov_phys = fourier_space._sf_fisher_reduction(info)
    targets = dict(zip(fourier_space.TARGET_NAMES,
                       fourier_space.fisher_cov_to_emulator_targets(cov_phys)))
    obs = info["observable"]
    k = np.asarray(obs.k[0], dtype=np.float64)
    dk = float(np.median(np.diff(k)))
    edges = np.column_stack([k - dk / 2.0, k + dk / 2.0])
    return {
        "info": info,
        "targets": targets,
        "k": k,
        "k_edges": edges,
        "data": np.asarray(obs.flatdata, dtype=np.float64),
        "cov": info["cov_components"]["C_total"],
        "n_eff": info["n_eff"],
        "z_eff": info["z_eff"],
        # Kaiser names it b1, REPT's physical basis names it b1p (= b1 * sigma8).
        "b1": float(info["params"].get("b1", info["params"].get("b1p", float("nan")))),
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_shot(tracers: List[str]) -> None:
    print("\n=== 1. Effective shot noise: our FKP n_eff vs DESI measured ===")
    print("    (DESI: num_shotnoise/norm from the full-shape bundle)")
    print(f"{'tracer':>10s} {'z_eff':>6s} {'n_eff_ours':>12s} {'n_eff_DESI':>12s} "
          f"{'Pshot_ours':>11s} {'Pshot_DESI':>11s} {'ratio':>7s}")
    any_bundle = False
    for tracer in tracers:
        bundle = _FS_BUNDLES.get(tracer)
        if bundle is None or not bundle.exists():
            print(f"{tracer:>10s} {'':>6s} {'':>12s} {'no local bundle':>12s}")
            continue
        any_bundle = True
        ours = our_forecast(tracer)
        desi = read_desi_bundle(bundle, ours["k_edges"])
        n_ours = ours["n_eff"]
        p_ours = float("nan") if not n_ours else 1.0 / n_ours
        n_desi = 1.0 / desi["P_shot"] if desi["P_shot"] > 0 else float("nan")
        ratio = (p_ours / desi["P_shot"]) if desi["P_shot"] > 0 else float("nan")
        print(f"{tracer:>10s} {ours['z_eff']:>6.3f} {n_ours:>12.4e} {n_desi:>12.4e} "
              f"{p_ours:>11.1f} {desi['P_shot']:>11.1f} {ratio:>7.3f}")
    if not any_bundle:
        print("  No full-shape bundles found locally -- fetch the DR1 full-shape "
              "VAC to extend this beyond LRG2.")
    else:
        print("  ratio = Pshot_ours/Pshot_DESI. Away from 1 means the V_eff->n_eff")
        print("  mapping mis-sets the shot-noise floor, which scales every sigma.")


def check_pk(tracers: List[str]) -> None:
    print("\n=== 2. Fiducial multipoles: our theory vs DESI measured ===")
    print("    (DESI is window-convolved and theta-cut; ours is neither --")
    print("     read this as an amplitude/shape sanity check, not equality)")
    for tracer in tracers:
        bundle = _FS_BUNDLES.get(tracer)
        if bundle is None or not bundle.exists():
            continue
        ours = our_forecast(tracer)
        desi = read_desi_bundle(bundle, ours["k_edges"])
        nk = ours["k"].size
        print(f"\n  {tracer} (b1_fid = {ours['b1']:.3f}, z_eff = {ours['z_eff']:.3f})")
        print(f"    {'k':>7s} {'P0_ours':>10s} {'P0_DESI':>10s} {'ratio':>7s}   "
              f"{'P2_ours':>10s} {'P2_DESI':>10s} {'ratio':>7s}")
        for i in range(0, nk, max(1, nk // 8)):
            k = ours["k"][i]
            p0o, p0d = ours["data"][i], desi["data"][i]
            p2o, p2d = ours["data"][nk + i], desi["data"][nk + i]
            print(f"    {k:>7.4f} {p0o:>10.1f} {p0d:>10.1f} {p0o / p0d:>7.3f}   "
                  f"{p2o:>10.1f} {p2d:>10.1f} {p2o / p2d:>7.3f}")
        r0 = ours["data"][:nk] / desi["data"][:nk]
        print(f"    P0 ratio: median {np.median(r0):.3f}, "
              f"range [{r0.min():.3f}, {r0.max():.3f}]")


def check_cov(tracers: List[str], rotated: bool, thetacut: bool,
              plot: bool) -> Dict[str, np.ndarray]:
    label = ("rotated+thetacut" if rotated
             else ("thetacut" if thetacut else "plain"))
    print(f"\n=== 3. Covariance: our Gaussian+SSC vs DESI EZmock ({label}) ===")
    print(f"{'tracer':>10s} {'diag ratio ours/DESI':>34s}  {'offdiag':>18s}")
    print(f"{'':>10s} {'median':>9s} {'ell=0':>8s} {'ell=2':>8s} {'k-trend':>7s}"
          f"  {'|corr| ours':>9s} {'DESI':>7s}")
    results = {}
    for tracer in tracers:
        path = _cov_path(tracer, rotated, thetacut)
        if path is None:
            print(f"{tracer:>10s} {'no covariance file':>34s}")
            continue
        ours = our_forecast(tracer)
        try:
            C_desi = read_desi_cov(path, ours["k_edges"])
        except ValueError as exc:
            print(f"{tracer:>10s}  grid mismatch: {exc}")
            continue
        C_ours = ours["cov"]
        nk = ours["k"].size
        d_ours, d_desi = np.diag(C_ours), np.diag(C_desi)
        ratio = d_ours / d_desi
        # low-k half vs high-k half of the monopole, as a trend indicator
        lo = np.median(ratio[: nk // 2])
        hi = np.median(ratio[nk // 2: nk])
        trend = hi / lo if lo > 0 else float("nan")

        def offdiag_mean(C):
            s = np.sqrt(np.clip(np.diag(C), 1e-300, None))
            R = C / np.outer(s, s)
            return float(np.mean(np.abs(R[~np.eye(R.shape[0], dtype=bool)])))

        flag = " *" if tracer in _SAMPLE_MISMATCH else ""
        print(f"{tracer:>10s} {np.median(ratio):>9.3f} "
              f"{np.median(ratio[:nk]):>8.3f} {np.median(ratio[nk:]):>8.3f} "
              f"{trend:>7.2f}  {offdiag_mean(C_ours):>9.3f} "
              f"{offdiag_mean(C_desi):>7.3f}{flag}")
        results[tracer] = {"ours": C_ours, "desi": C_desi, "k": ours["k"]}

    for tracer, why in _SAMPLE_MISMATCH.items():
        if tracer in results:
            print(f"  * {tracer}: {why} -- densities differ, not apples-to-apples.")
    print("  *** These ratios are NOT apples-to-apples: DESI's covariance is of a")
    print("  window-convolved estimator and ours is unwindowed. On LRG2 the window")
    print("  moves the ratio 2.046 -> 0.512. Neither the absolute level nor the")
    print("  spread across tracers is interpretable -- the window differs per")
    print("  footprint, so the comparison error does too. Use --check window.")
    print("  What IS usable here: the k-trend at fixed tracer (1.0 = our shape")
    print("  matches DESI's), since the window is a milder function of k than of")
    print("  overall normalization.")

    if plot and results:
        _plot_cov(results, label)
    return results


def _plot_cov(results: Dict, label: str) -> None:
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.4), squeeze=False)
    for i, (tracer, r) in enumerate(results.items()):
        k, nk = r["k"], r["k"].size
        ax = axes[0][i]
        for j, ell in enumerate(_OUR_ELLS):
            sl = slice(j * nk, (j + 1) * nk)
            ax.plot(k, np.sqrt(np.diag(r["ours"])[sl]), "-",
                    label=rf"ours $\ell={ell}$")
            ax.plot(k, np.sqrt(np.diag(r["desi"])[sl]), "--",
                    label=rf"DESI $\ell={ell}$")
        ax.set_yscale("log")
        ax.set_title(tracer)
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        if i == 0:
            ax.set_ylabel(r"$\sigma(P_\ell)$")
            ax.legend(fontsize=7)
        ax = axes[1][i]
        for j, ell in enumerate(_OUR_ELLS):
            sl = slice(j * nk, (j + 1) * nk)
            ax.plot(k, (np.diag(r["ours"]) / np.diag(r["desi"]))[sl],
                    label=rf"$\ell={ell}$")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        if i == 0:
            ax.set_ylabel("diag ratio ours/DESI")
            ax.legend(fontsize=7)
    fig.suptitle(f"ShapeFit covariance vs DESI DR1 EZmock ({label})")
    fig.tight_layout()
    out = f"cov_vs_desi_{label.replace('+', '_')}.png"
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")


def read_desi_window(path: Path) -> Dict:
    """Window matrix and its theory-side grid from a full-shape bundle.

    DESI's model is ``measured = W @ theory``, W of shape (n_obs, n_theory).
    The theory side spans MORE multipoles and a wider, finer k range than the
    observable (ells 0,2,4 on 349 k from 0.001 to 0.349 for LRG2), because the
    window leaks ell=4 power and out-of-range k into the measured ell=0 and
    ell=2. Trailing columns beyond n_ells*n_k are DESI's rotation/photo
    systematic templates, which we have no counterpart for and set to zero.
    """
    with h5py.File(path, "r") as f:
        W = np.asarray(f["window/value"][:], dtype=np.float64)
        grp = f["window/theory"]
        grp = grp["spectrum"] if "spectrum" in grp else grp
        th_ells, th_k = [], None
        for key in sorted(grp.keys(), key=lambda s: (not s.isdigit(), s)):
            if not key.isdigit():
                continue
            th_ells.append(int(key))
            kk = np.asarray(grp[key]["k"][:], dtype=np.float64)
            if th_k is None:
                th_k = kk
            elif not np.allclose(th_k, kk):
                raise ValueError(f"{path.name}: window theory k differs per ell")
    n_extra = W.shape[1] - len(th_ells) * th_k.size
    if n_extra < 0:
        raise ValueError(f"{path.name}: W has {W.shape[1]} cols, theory grid "
                         f"needs {len(th_ells) * th_k.size}")
    return {"W": W, "th_ells": tuple(th_ells), "th_k": th_k, "n_extra": n_extra}


def our_theory_on_window_grid(tracer: str, win: Dict) -> Dict:
    """Our Kaiser multipoles on the window's theory grid, at the pipeline's
    fiducial nuisances. Returns the flat theory vector W expects plus the
    per-ell spectra (needed for the analytic Gaussian covariance)."""
    from desilike.theories.galaxy_clustering import (
        ShapeFitPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles)

    base = our_forecast(tracer)
    info = base["info"]
    theta = sf_core._to_shapefit_cosmo_params(
        {**FID_SAMPLE, "N_tracers": ntracers(tracer, "dr1")})
    template = ShapeFitPowerSpectrumTemplate(
        z=info["z_eff"], fiducial=("DESI", dict(theta)),
        apmode="qisoqap", with_now="wallish2018")
    theory = KaiserTracerPowerSpectrumMultipoles(
        template=template, k=win["th_k"], ells=win["th_ells"])
    theory(**{k: v for k, v in info["params"].items()
              if k in ("b1", "sn0", "sigmapar", "sigmaper")})
    P = np.asarray(theory.power, dtype=np.float64)          # (n_ells, n_k)
    vec = np.concatenate([P.ravel(), np.zeros(win["n_extra"])])
    return {"P_ells": P, "vec": vec, "base": base, "theta": theta}


def _windowed_analytic_cov(tracer: str, win: Dict, P_ells: np.ndarray,
                           theta: Dict) -> np.ndarray:
    """W @ C_theory @ W.T with C_theory the Grieb/FKP analytic Gaussian on the
    window's theory grid.

    Uses bao/fkp_analytic_cov.py rather than desilike's ObservablesCovarianceMatrix
    because the theory grid needs ells (0,2,4) at dk=0.001 down to k=0.001 --
    outside the range the observable path is built for. This is the same
    operation bao/config_space.py:416 performs in production for the
    correlation-function pipeline (C = W @ C_theory @ W.T), with the same
    justification: W comes from the RANDOM catalog, so it is survey geometry,
    not measured clustering.
    """
    from desilike.theories.primordial_cosmology import get_cosmo
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bao"))
    import fkp_analytic_cov as fac

    cosmo = get_cosmo(("DESI", dict(theta)))
    slices = fac.load_nz_slices(
        tracer, cosmo, area_deg2=sf_core.dataset_area("dr1"), N_design=float(ntracers(tracer, "dr1")))
    blocks = fac.fkp_analytic_cov(
        k=win["th_k"], P_ells_in=P_ells, ells_in=win["th_ells"],
        ells_obs=win["th_ells"], slices=slices)
    C_theory = fac.assemble_full_cov(blocks, win["th_ells"])
    n = C_theory.shape[0] + win["n_extra"]
    C_pad = np.zeros((n, n), dtype=np.float64)
    C_pad[:C_theory.shape[0], :C_theory.shape[0]] = C_theory
    W = win["W"]
    C_win = W @ C_pad @ W.T
    return 0.5 * (C_win + C_win.T)


def _analytic_cov_on_obs_grid(tracer: str, ours: Dict) -> np.ndarray:
    """The same analytic Gaussian engine as `_windowed_analytic_cov`, evaluated
    on the OBSERVABLE grid with no window applied. Control for the engine
    change (see check_window)."""
    from desilike.theories.galaxy_clustering import (
        ShapeFitPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles)
    from desilike.theories.primordial_cosmology import get_cosmo
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bao"))
    import fkp_analytic_cov as fac

    base = ours["base"]
    k = base["k"]
    template = ShapeFitPowerSpectrumTemplate(
        z=base["info"]["z_eff"], fiducial=("DESI", dict(ours["theta"])),
        apmode="qisoqap", with_now="wallish2018")
    # ells (0,2,4) as MODEL input even though the observable is (0,2): the
    # Grieb formula needs the hexadecapole of the model to build the (0,2) block.
    theory = KaiserTracerPowerSpectrumMultipoles(
        template=template, k=k, ells=(0, 2, 4))
    theory(**{kk: v for kk, v in base["info"]["params"].items()
              if kk in ("b1", "sn0", "sigmapar", "sigmaper")})
    cosmo = get_cosmo(("DESI", dict(ours["theta"])))
    slices = fac.load_nz_slices(
        tracer, cosmo, area_deg2=sf_core.dataset_area("dr1"),
        N_design=float(ntracers(tracer, "dr1")))
    blocks = fac.fkp_analytic_cov(
        k=k, P_ells_in=np.asarray(theory.power), ells_in=(0, 2, 4),
        ells_obs=_OUR_ELLS, slices=slices)
    return fac.assemble_full_cov(blocks, _OUR_ELLS)


def check_window(tracers: List[str]) -> None:
    """Does the missing survey window explain the theory and covariance gaps?"""
    print("\n=== 5. Window convolution: does it close the gap? ===")
    print("    measured = W @ theory. Our forecast applies NO window (the")
    print("    Fourier paths never have); bao/config_space.py:416 does apply")
    print("    one in production for the correlation pipeline.")
    for tracer in tracers:
        bundle = _FS_BUNDLES.get(tracer)
        if bundle is None or not bundle.exists():
            print(f"\n  {tracer}: no local full-shape bundle (window ships with it)")
            continue
        win = read_desi_window(bundle)
        ours = our_theory_on_window_grid(tracer, win)
        base = ours["base"]
        nk = base["k"].size
        desi = read_desi_bundle(bundle, base["k_edges"])
        C_desi = desi["cov"]

        conv = win["W"] @ ours["vec"]
        raw = base["data"]
        print(f"\n  {tracer}: W is {win['W'].shape}, theory grid ells "
              f"{win['th_ells']} x {win['th_k'].size} k "
              f"[{win['th_k'][0]:.3f}, {win['th_k'][-1]:.3f}], "
              f"{win['n_extra']} systematic columns zeroed")
        print(f"    {'k':>7s} {'P0 raw/DESI':>12s} {'P0 conv/DESI':>13s}   "
              f"{'P2 raw/DESI':>12s} {'P2 conv/DESI':>13s}")
        for i in range(0, nk, max(1, nk // 7)):
            print(f"    {base['k'][i]:>7.4f} "
                  f"{raw[i] / desi['data'][i]:>12.3f} "
                  f"{conv[i] / desi['data'][i]:>13.3f}   "
                  f"{raw[nk + i] / desi['data'][nk + i]:>12.3f} "
                  f"{conv[nk + i] / desi['data'][nk + i]:>13.3f}")
        r_raw = raw[:nk] / desi["data"][:nk]
        r_con = conv[:nk] / desi["data"][:nk]
        print(f"    P0 median ratio: raw {np.median(r_raw):.3f} -> "
              f"convolved {np.median(r_con):.3f}")

        try:
            C_win = _windowed_analytic_cov(tracer, win, ours["P_ells"],
                                           ours["theta"])
        except Exception as exc:
            print(f"    windowed covariance failed: {type(exc).__name__}: {exc}")
            continue

        # CONTROL, and it is not optional. The windowed covariance is built with
        # bao/fkp_analytic_cov.py while the unwindowed one comes from desilike's
        # ObservablesCovarianceMatrix, so a raw before/after comparison would
        # conflate "applied the window" with "changed covariance engine". Run the
        # analytic engine on the OBSERVABLE grid with no window: if it matches
        # desilike, the windowed/unwindowed difference is the window.
        ctrl = _analytic_cov_on_obs_grid(tracer, ours)
        r_ctrl = np.diag(ctrl) / np.diag(base["cov"])
        print(f"    engine control (same grid, no window, analytic/desilike): "
              f"median {np.median(r_ctrl):.3f}, "
              f"range [{r_ctrl.min():.3f}, {r_ctrl.max():.3f}]")
        if not 0.9 <= np.median(r_ctrl) <= 1.1:
            print("    ** control outside 10%: the window numbers below are "
                  "CONFOUNDED by the engine change, do not interpret them **")

        def offdiag(C):
            s = np.sqrt(np.clip(np.diag(C), 1e-300, None))
            R = C / np.outer(s, s)
            return float(np.mean(np.abs(R[~np.eye(R.shape[0], dtype=bool)])))

        d_ours = np.diag(base["cov"])
        d_win = np.diag(C_win)
        d_desi = np.diag(C_desi)
        print(f"    covariance diag ratio vs DESI: "
              f"unwindowed {np.median(d_ours / d_desi):.3f} -> "
              f"windowed {np.median(d_win / d_desi):.3f}")
        print(f"    mean |off-diag corr|:  unwindowed {offdiag(base['cov']):.3f}"
              f"   windowed {offdiag(C_win):.3f}   DESI {offdiag(C_desi):.3f}")



def check_compressed(tracers: List[str], theory: str = "rept") -> None:
    """Our forecast targets against DESI's PUBLISHED ShapeFit constraints.

    The end-to-end test: DESI 2024 V Appendix A gives per-tracer compressed
    constraints at DR1 volumes in a basis one division from ours, so this needs
    no window, no covariance surgery and no volume rescaling. See
    desi_reference.py for the transcription and its caveats.
    """
    import desi_reference as dr

    print(f"\n=== 6. Our targets vs DESI DR1 published ShapeFit ({theory}) ===")
    print("    DESI 2024 V (2411.12021) Appendix A, ShapeFit-alone fits.")
    print("    Ratio < 1 = we are TIGHTER than DESI. A Fisher forecast at the")
    print("    truth is expected somewhat tight against an MCMC posterior.")
    print(f"\n{'tracer':>10s} {'z_us':>6s} {'z_DESI':>7s} "
          f"{'qiso':>17s} {'qap':>17s} {'fsr/fsr':>17s} {'m':>17s}")
    rows = {}
    for tracer in tracers:
        try:
            ref = dr.sigma_targets(tracer)
            z_desi, _, _ = dr.datavector(tracer)
        except KeyError as exc:
            print(f"{tracer:>10s}  {exc}")
            continue
        ours = our_forecast(tracer, theory=theory)
        t = ours["targets"]
        fsr_frac = t["sigma_f_sigmar"] / float(ours["info"]["f_sigmar_fid"])
        pairs = [
            (t["sigma_qiso"], ref["sigma_qiso"]),
            (t["sigma_qap"], ref["sigma_qap"]),
            (fsr_frac, ref["sigma_f_sigmar_frac"]),
            (t["sigma_m"], ref["sigma_m"]),
        ]
        flag = " *" if dr.TRACER_MAP[tracer] in dr.SAMPLE_MISMATCH else ""
        cells = " ".join(f"{a:.4f}/{b:.4f}={a / b:>4.2f}" for a, b in pairs)
        print(f"{tracer:>10s} {ours['z_eff']:>6.3f} {z_desi:>7.2f} {cells}{flag}")
        rows[tracer] = (ours, ref)

    print("\n  correlations (ours / DESI), sign disagreements marked X:")
    rho_names = [n for n in fourier_space.TARGET_NAMES if n.startswith("rho_")]
    print(f"{'tracer':>10s} " + " ".join(f"{n[4:]:>17s}" for n in rho_names))
    for tracer, (ours, ref) in rows.items():
        cells = []
        for n in rho_names:
            a, b = ours["targets"][n], ref[n]
            bad = "X" if (a * b < 0) else " "
            cells.append(f"{a:+.2f}/{b:+.2f}{bad}")
        print(f"{tracer:>10s} " + " ".join(f"{c:>17s}" for c in cells))
    for tracer, why in dr.SAMPLE_MISMATCH.items():
        if any(dr.TRACER_MAP[t] == tracer for t in rows):
            print(f"  * {why}")
    print("  NB DESI warns BGS alpha_AP is prior-dominated (flat 0.8-1.2), so")
    print("  its sigma(qap) is a prior width, not a measurement.")


def check_sigma(tracers: List[str], rotated: bool, thetacut: bool) -> None:
    """Swap DESI's covariance into our Fisher and report the sigma shift."""
    print("\n=== 4. Sigmas with OUR covariance vs with DESI's covariance ===")
    print("    (theory, derivatives and marginalization held fixed)")
    names = fourier_space.TARGET_NAMES[:4]
    print(f"{'tracer':>10s} {'source':>8s} " + " ".join(f"{n[6:]:>11s}" for n in names))
    for tracer in tracers:
        path = _cov_path(tracer, rotated, thetacut)
        if path is None:
            print(f"{tracer:>10s} {'--':>8s}  no covariance file")
            continue
        ours = our_forecast(tracer)
        try:
            C_desi = read_desi_cov(path, ours["k_edges"])
        except ValueError as exc:
            print(f"{tracer:>10s}  grid mismatch: {exc}")
            continue
        swapped = our_forecast(tracer, cov_override=C_desi)
        a = [ours["targets"][n] for n in names]
        b = [swapped["targets"][n] for n in names]
        print(f"{tracer:>10s} {'ours':>8s} " + " ".join(f"{v:>11.5f}" for v in a))
        print(f"{tracer:>10s} {'DESI C':>8s} " + " ".join(f"{v:>11.5f}" for v in b))
        print(f"{tracer:>10s} {'ratio':>8s} "
              + " ".join(f"{x / y:>11.3f}" for x, y in zip(a, b)))
    print("  ratio < 1 means our covariance alone makes the forecast tighter.")
    print("  A ratio near 1 with a large gap to DESI's PUBLISHED sigmas would")
    print("  point at the theory/derivative side instead.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", nargs="*",
                   choices=["shot", "pk", "cov", "sigma", "window", "compressed", "all"],
                   default=["all"])
    p.add_argument("--tracers", nargs="*", default=None,
                   help=f"subset of {list(TRACERS_ALL)}; default all")
    p.add_argument("--rotated", action="store_true",
                   help="use DESI's rotated+thetacut covariance (their baseline) "
                        "instead of the plain one")
    p.add_argument("--thetacut", action="store_true",
                   help="use the theta-cut variant of the plain covariance")
    p.add_argument("--plot", action="store_true", help="write the covariance plot")
    p.add_argument("--theory", choices=["kaiser", "rept"], default="rept",
                   help="theory for --check compressed (default rept)")
    args = p.parse_args()

    tracers = args.tracers or list(TRACERS_ALL)
    bad = [t for t in tracers if t not in _DESI_SAMPLE]
    if bad:
        p.error(f"unknown tracers {bad}; choose from {list(TRACERS_ALL)}")
    checks = set(args.check)
    do = lambda name: "all" in checks or name in checks  # noqa: E731

    print(f"DESI DR1 products: {_LIK_DIR}")
    print(f"Tracers: {tracers}")

    if do("shot"):
        check_shot(tracers)
    if do("pk"):
        check_pk(tracers)
    if do("cov"):
        check_cov(tracers, args.rotated, args.thetacut, args.plot)
    if do("compressed"):
        check_compressed(tracers, args.theory)
    if do("window"):
        check_window(tracers)
    if do("sigma"):
        check_sigma(tracers, args.rotated, args.thetacut)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
