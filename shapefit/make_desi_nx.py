#!/usr/bin/env python
"""Rebuild `{tracer}_desi_nx.csv` -- the reduction of DESI's randoms.

These tables are what z_eff is computed from since S53 (2411.12020 Eqs 8.1-8.3):

    nbar_desi_nx   selection-weighted <NX> per slice, NX = n(z) <C_assign>.
                   A DENSITY: scales with N_tracers via _nz_scale_factor.
    S1_weight      sum of WEIGHT per slice. Mask, tiling, per-region
                   normalisation -- survey GEOMETRY, fixed in N.

They are 20.9 GB of randoms reduced to ~10 KB, which is why the ~10 KB is
committed (data/{dataset}/nz_slices) and the 20.9 GB is not. This script exists
so the committed files are REPRODUCIBLE rather than inherited.

    python make_desi_nx.py --tracers LRG2 --check      # rebuild, diff, discard
    python make_desi_nx.py --install                   # rebuild in place

SOURCE: randoms, and it matters
-------------------------------
`--source randoms` (default) is the only one that reproduces the committed
tables. `--source data` reads the much smaller clustering.dat.fits instead and
is offered for a machine that cannot spare the download, but it is NOT
equivalent. Measured, z_eff vs DESI 2024 V Table 1 over six tracers (S79):

    S1 randoms + NX randoms   0.062% mean   <- the committed tables
    S1 data    + NX randoms   0.064%
    S1 randoms + NX data      0.117%
    S1 data    + NX data      0.111%

`S1` survives the swap because it is a SUM: data and randoms differ by a
constant (~1/13, the density ratio) that cancels in Eq. (2.1)'s normalised
ratio, and the residual 0.30% scatter sits at the 0.36% Poisson floor.

`NX` does not. It is a mean of a DENSITY weighted by the objects themselves, and
galaxies preferentially occupy high-NX regions, so the data-weighted mean is
<n^2>/<n> rather than <n> -- 4.8% high, with 0.41% z-DEPENDENT structure that
does not cancel. Randoms sample the SELECTION FUNCTION, which is exactly what
Eq. (2.1) means by n_ran. There is no reweighting that recovers the selection
function from the data alone, and a correction factor would be a fudge.

Get the randoms with:
    python init_desi_data.py --what randoms --nran 1
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
warnings.filterwarnings("ignore")

from util import nz_slices_path, reference_table_path  # noqa: E402

CAT_DIR = Path.home() / "data" / "desi" / "lss_dr1"
REPO_DATA = _HERE.parent / "data"

# Same mapping as make_nz_slices.STEM -- the parent catalogue each bin is cut
# from. LRG1/2/3 share the LRG catalogue and differ only by slice edges.
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


def _load(stem: str, source: str, nran: int):
    """(Z, WEIGHT, NX, WEIGHT_FKP) concatenated over both caps."""
    key = (stem, source, nran)
    if key in _CACHE:
        return _CACHE[key]
    from astropy.io import fits

    paths = []
    for cap in ("NGC", "SGC"):
        if source == "data":
            paths.append(CAT_DIR / f"{stem}_{cap}_clustering.dat.fits")
        else:
            for i in range(nran):
                paths.append(CAT_DIR / f"{stem}_{cap}_{i}_clustering.ran.fits")

    z, w, nx, wf = [], [], [], []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p}\nFetch it with:\n"
                f"  python init_desi_data.py --what "
                f"{'lss' if source == 'data' else 'randoms'} --nran {nran}")
        with fits.open(p) as h:
            d = h[1].data
            z.append(np.asarray(d["Z"], dtype=np.float64))
            w.append(np.asarray(d["WEIGHT"], dtype=np.float64))
            nx.append(np.asarray(d["NX"], dtype=np.float64))
            wf.append(np.asarray(d["WEIGHT_FKP"], dtype=np.float64))
    out = (np.concatenate(z), np.concatenate(w), np.concatenate(nx),
           np.concatenate(wf))
    _CACHE[key] = out
    return out


def rebuild(tracer: str, dataset: str, source: str, nran: int) -> pd.DataFrame:
    """Aggregate onto this tracer's EXISTING slice edges.

    The edges are not free: `bao/core._desi_nz_geometry` length-checks this file
    against the slice count and silently falls back to `nbar_file` on a
    mismatch, which would quietly revert z_eff to the pre-S53 convention.
    """
    sl = pd.read_csv(nz_slices_path(f"{tracer}_nz_slices.csv", dataset))
    sl = sl[sl["slice_fraction"] > 0.0].reset_index(drop=True)
    z_lo = sl["zlow"].to_numpy(dtype=np.float64)
    z_hi = sl["zhigh"].to_numpy(dtype=np.float64)

    z, w, nx, wf = _load(STEM[tracer], source, nran)
    edges = np.append(z_lo, z_hi[-1])
    idx = np.digitize(z, edges) - 1
    ok = (idx >= 0) & (idx < len(z_lo))

    S1 = np.zeros(len(z_lo))
    NX = np.zeros(len(z_lo))
    WF = np.zeros(len(z_lo))
    P0E = np.zeros(len(z_lo))
    for i in range(len(z_lo)):
        m = ok & (idx == i)
        if not m.any():
            continue
        S1[i] = w[m].sum()
        # Weighted by WEIGHT * WEIGHT_FKP, not WEIGHT alone. Eq. (2.1)'s random
        # density is n_ran = S1 * w_fkp, so <NX> must carry the same FKP
        # weighting that appears in the z_eff weight. Recovering this was the
        # point of S80: WEIGHT alone is 4.8% off, unweighted 1.3%, this 0.07%.
        NX[i] = np.average(nx[m], weights=w[m] * wf[m])
        # DESI's own FKP weight, and the per-slice pivot it implies (S82).
        #
        # p0_eff is back-solved PER OBJECT and then averaged, not derived from
        # the slice means: 1/(1+nP) is convex, so averaging first would import
        # the Jensen bias S49 measured at 6.5%.
        #
        # For a single-tracer sample this returns the yaml pivot exactly (LRG:
        # 10000.0 +- 0.0). For the LRG3_ELG1 combined bin it is NOT constant --
        # 11335 at z=0.81 rising to 18679 at z=1.09 as LRG gives way to ELG --
        # which no scalar `fkp_p0` can represent.
        WF[i] = np.average(wf[m], weights=w[m])
        P0E[i] = np.average((1.0 / wf[m] - 1.0) / nx[m], weights=w[m])

    if not np.all(NX > 0):
        bad = np.flatnonzero(NX <= 0)
        raise ValueError(f"{tracer}: empty slices {bad.tolist()} -- the "
                         "length check in _desi_nz_geometry would reject this")
    return pd.DataFrame({"zmid": sl["zmid"].to_numpy(dtype=np.float64),
                         "zlow": z_lo, "zhigh": z_hi,
                         "nbar_desi_nx": NX, "S1_weight": S1,
                         "w_fkp_mean": WF, "p0_eff": P0E})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", nargs="+", default=sorted(STEM))
    ap.add_argument("--dataset", default="dr1", choices=["dr1"])
    ap.add_argument("--source", choices=["randoms", "data"], default="randoms",
                    help="randoms (default) reproduces the committed tables; "
                         "data is a lossy fallback -- see the module docstring")
    ap.add_argument("--nran", type=int, default=2,
                    help="random files per cap. 2 reproduces the committed "
                         "tables; S1 is a SUM so it scales linearly with this")
    ap.add_argument("--check", action="store_true",
                    help="rebuild and diff against the committed table, "
                         "writing nothing")
    ap.add_argument("--install", action="store_true",
                    help="overwrite the committed table (produces a git diff)")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()

    out_dir = a.out_dir or (REPO_DATA / a.dataset / "nz_slices")
    worst = 0.0
    for t in a.tracers:
        try:
            df = rebuild(t, a.dataset, a.source, a.nran)
        except FileNotFoundError as exc:
            print(f"  {t:10s} SKIP -- {str(exc).splitlines()[0]}")
            continue

        cur = REPO_DATA / a.dataset / "nz_slices" / f"{t}_desi_nx.csv"
        if cur.exists():
            old = pd.read_csv(cur)
            if len(old) == len(df):
                dn = np.abs(df["nbar_desi_nx"].to_numpy()
                            / old["nbar_desi_nx"].to_numpy() - 1).max()
                ds = np.abs(df["S1_weight"].to_numpy()
                            / old["S1_weight"].to_numpy() - 1).max()
                worst = max(worst, dn, ds)
                print(f"  {t:10s} {len(df):3d} slices   max|dNX| {dn*100:7.3f}%"
                      f"   max|dS1| {ds*100:7.3f}%")
            else:
                print(f"  {t:10s} {len(df):3d} slices vs {len(old)} committed "
                      "-- SLICE COUNT CHANGED")
        else:
            print(f"  {t:10s} {len(df):3d} slices   (no committed table yet)")

        if a.check:
            continue
        if a.install or a.out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"{t}_desi_nx.csv", index=False)

    if a.check:
        print(f"\nworst deviation from committed: {worst*100:.3f}%  (nothing written)")
    elif not (a.install or a.out_dir):
        print("\nnothing written (pass --install or --out-dir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
