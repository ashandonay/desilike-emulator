#!/usr/bin/env python
"""Regenerate `{tracer}_nz_slices.csv` from the DR1 LSS clustering catalogues.

The shipped tables were built from DESI's published `*_nz.txt` files, which
describe the FINAL-SURVEY sample: 1.8-4.1x the DR1 counts over footprints of
11k-30k deg^2 (shapefit CHANGELOG S62). Only the SHAPE is consumed -- the
normalisation comes from `util.ntracers` -- but the shape is not DR1's, by
2.4% (LRG2) to 13.6% (BGS).

S62a measured the cost on the shapefit forecast at fixed N and found it
negligible (<=0.15%), because the shapefit covar path compresses the slice n(z)
through a single `n_eff` root-find. That argument does NOT cover:

  - the BAO CONFIG-SPACE path, where the per-slice n(z) enters the Gaussian xi
    covariance directly with no such compression (bao/config_space.py:696);
  - N_tracers away from the DR1 count, since the FKP weight 1/(1+nP0) is
    non-linear in nbar and the design box spans 0.5x-1.5x.

Either way the input should be right, so this rebuilds it from the catalogues.

    python make_nz_slices.py --out-dir /tmp/nz_new        # write elsewhere
    python make_nz_slices.py --install                    # overwrite, with backup

The SLICE EDGES are preserved exactly from the existing file. They are not free:
`bao/core._desi_nz_geometry` length-checks `{tracer}_desi_nx.csv` against the
slice count and silently falls back to `nbar_file` on a mismatch, which would
quietly revert z_eff to the pre-S53 convention.

`slice_fraction` is the WEIGHT-weighted galaxy fraction, not the raw count
fraction -- DESI's n(z) is weighted, and for BGS the two differ by 29% (S62).
`nbar_file` is the weighted count divided by the comoving shell volume at the
DESI fiducial cosmology and this tracer's footprint (util.tracer_area, S54/S58).

Catalogues are PUBLIC -- no NERSC, no MFA:
    https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/
Filenames use NGC/SGC, not N/S.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

warnings.filterwarnings("ignore")

from util import ntracers, tracer_area  # noqa: E402

CAT_DIR = Path.home() / "data" / "desi" / "lss_dr1"
# Output goes into the REPO (S80): these tables are version-controlled inputs,
# so regenerating them should produce a reviewable diff, not a silent change to
# a file outside the tree.
NZ_DIR = Path(__file__).resolve().parent.parent / "data"

# Release -> (catalogue dir, LSS run, LSS version). S62c item (3): everything
# else in this script -- STEM, the slice edges, the areas, the counts and the
# download URL -- is DR1-specific, so any other release must FAIL rather than
# quietly emit DR1 tables into a {dataset}/ directory that then looks populated.
# Filling this in is DR2 work, deferred by the DR1-first rule; the point of the
# table is that the failure is loud.
_RELEASES = {
    "dr1": (CAT_DIR, "iron", "v1.5"),
}

# tracer bin -> catalogue stem. LRG1/2/3 share one LRG catalogue and are split
# by the slice edges; LRG3_ELG1 is the BAO-only combined bin (S31/S32).
STEM = {
    "BGS": "BGS_BRIGHT-21.5",
    "LRG1": "LRG",
    "LRG2": "LRG",
    "LRG3": "LRG",
    "ELG2": "ELG_LOPnotqso",
    "QSO": "QSO",
    "LRG3_ELG1": "LRG+ELG_LOPnotqso",
}

_CACHE: dict = {}


def _load_catalogue(stem: str):
    """(Z, WEIGHT) over both caps."""
    if stem not in _CACHE:
        from astropy.io import fits
        z, w = [], []
        for cap in ("NGC", "SGC"):
            path = CAT_DIR / f"{stem}_{cap}_clustering.dat.fits"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path}\nDownload it (public, no NERSC needed):\n"
                    f"  curl -O https://data.desi.lbl.gov/public/dr1/survey/"
                    f"catalogs/dr1/LSS/iron/LSScats/v1.5/{path.name}")
            with fits.open(path) as h:
                d = h[1].data
                z.append(np.asarray(d["Z"], dtype=np.float64))
                w.append(np.asarray(d["WEIGHT"], dtype=np.float64))
        _CACHE[stem] = (np.concatenate(z), np.concatenate(w))
    return _CACHE[stem]


def rebuild(tracer: str, cosmo) -> pd.DataFrame:
    old_path = NZ_DIR / "dr1" / "nz_slices" / f"{tracer}_nz_slices.csv"
    if not old_path.exists():
        raise FileNotFoundError(f"No existing slice file to take edges from: {old_path}")
    old = pd.read_csv(old_path)
    z_lo = old["zlow"].to_numpy(dtype=np.float64)
    z_hi = old["zhigh"].to_numpy(dtype=np.float64)
    z_mid = old["zmid"].to_numpy(dtype=np.float64)

    z, w = _load_catalogue(STEM[tracer])
    wsum = np.array([w[(z >= a) & (z < b)].sum() for a, b in zip(z_lo, z_hi)])
    if not wsum.sum() > 0:
        raise ValueError(f"{tracer}: no catalogue galaxies in the slice range")
    frac = wsum / wsum.sum()

    area = tracer_area(tracer, "dr1")
    sky = float(area) / 41252.96
    chi_lo = np.asarray(cosmo.comoving_radial_distance(z_lo), dtype=np.float64)
    chi_hi = np.asarray(cosmo.comoving_radial_distance(z_hi), dtype=np.float64)
    V = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * sky

    out = pd.DataFrame({
        "zmid": z_mid, "zlow": z_lo, "zhigh": z_hi,
        "slice_fraction": frac,
        "nbar_file": wsum / np.maximum(V, 1.0),
        "Nbin_file": wsum,
        "Vol_bin_file": V,
        "source_file": f"{STEM[tracer]}_{{NGC,SGC}}_clustering.dat.fits",
        "file_area_deg2": area,
    })
    old_frac = old["slice_fraction"].to_numpy(dtype=np.float64)
    old_frac = old_frac / old_frac.sum()
    good = (old_frac > 0) & (frac > 0)
    r = frac[good] / old_frac[good]
    print(f"  {tracer:10s} {len(frac):3d} slices  N_cat={wsum.sum():10.0f}  "
          f"N_dr1={ntracers(tracer, 'dr1'):10.0f}  "
          f"shape change vs old: {(r.max()/r.min()-1)*100:5.1f}%")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", nargs="+", default=sorted(STEM))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write. Default: a sibling scratch dir; "
                         "--install writes to the live nz_slices dir.")
    ap.add_argument("--install", action="store_true",
                    help="Overwrite the live tables, backing up to *.prefinal.bak")
    ap.add_argument("--dataset", default="dr1",
                    help="Data release. Only dr1 is implemented; anything else "
                         "fails loudly rather than emitting DR1 tables under "
                         "another release's name (S62c).")
    a = ap.parse_args()

    if a.dataset not in _RELEASES:
        raise SystemExit(
            f"--dataset {a.dataset!r} is not implemented. This script is "
            "DR1-specific throughout: catalogue stems, slice edges, areas, "
            "counts and the download URL. Add an entry to _RELEASES and audit "
            "every one of those before using it for another release (shapefit "
            f"CHANGELOG S62c). Refusing to write DR1 tables into a "
            f"{a.dataset!r} directory.")

    from desilike.theories.primordial_cosmology import get_cosmo
    cosmo = get_cosmo("DESI")

    # Release-scoped layout (S62c/S80): data/{dataset}/nz_slices/{tracer}_*.csv
    out_dir = (Path(a.out_dir) if a.out_dir
               else (NZ_DIR / a.dataset / "nz_slices" if a.install
                     else NZ_DIR / "regenerated" / a.dataset / "nz_slices"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}\n")

    for t in a.tracers:
        try:
            df = rebuild(t, cosmo)
        except FileNotFoundError as exc:
            print(f"  {t:10s} SKIPPED: {exc}")
            continue
        dest = out_dir / f"{t}_nz_slices.csv"
        if a.install and dest.exists():
            bak = dest.with_suffix(".csv.prefinal.bak")
            if not bak.exists():
                shutil.copy2(dest, bak)
        df.to_csv(dest, index=False)
    print(f"\ndone -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
