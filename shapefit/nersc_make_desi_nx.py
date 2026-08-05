#!/usr/bin/env python
"""Build a `{tracer}_desi_nx.csv` ON NERSC, where the randoms already live.

Self-contained on purpose: no imports from this repo, no desilike, no
cosmoprimo. Copy this one file to NERSC, run it, copy back the ~5 KB CSV.
That is 6.6 GB of reading turned into 5 KB of transfer.

    # on NERSC (any node that can read CFS; the desi conda env has astropy)
    source /global/common/software/desi/desi_environment.sh main
    python nersc_make_desi_nx.py --tracer LRG3_ELG1 -o LRG3_ELG1_desi_nx.csv

    # back on the workstation
    scp <nersc>:LRG3_ELG1_desi_nx.csv data/dr1/nz_slices/

WHAT IT COMPUTES (must match shapefit/make_desi_nx.py exactly)
--------------------------------------------------------------
Per redshift slice, over the DR1 v1.5 random catalogues, both caps pooled:

    nbar_desi_nx = average(NX, weights = WEIGHT * WEIGHT_FKP)
    S1_weight    = sum(WEIGHT)

The `WEIGHT_FKP` factor is NOT optional. DESI 2024 III Eq. (2.1) weights by the
random density n_ran = S1 * w_fkp, so <NX> has to carry the same FKP weighting
that appears in the z_eff weight. Measured against the committed LRG2 table,
this estimator lands at 0.071% while WEIGHT-alone is 4.8% off (shapefit
CHANGELOG S80).

`S1_weight` is a SUM, so it scales linearly with NFILES. The committed tables
used 2 files per cap; keep --nfiles 2 unless you are regenerating all of them.

The SLICE EDGES are hardcoded below and must match
`data/dr1/nz_slices/{tracer}_nz_slices.csv`. They are not free:
`bao/core._desi_nz_geometry` length-checks the two files against each other and
silently falls back to `nbar_file` on a mismatch -- which is exactly the
fallback this file exists to remove.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CFS = Path("/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5")

# Parent random catalogue per tracer bin, and that bin's slice edges.
# LRG3_ELG1 is DESI's BAO combined bin; full shape uses LRG3 instead.
TRACERS = {
    "LRG3_ELG1": {
        "stem": "LRG+ELG_LOPnotqso",
        "edges": [round(0.80 + 0.02 * i, 6) for i in range(16)],
        "zmid": [0.810025, 0.83003, 0.850008, 0.869977, 0.889957, 0.909974,
                 0.929945, 0.949957, 0.969959, 0.990007, 1.009957, 1.029822,
                 1.050078, 1.069957, 1.08993],
    },
    # Present so this script is SELF-VALIDATING: LRG2 already has a committed
    # table, so running --tracer LRG2 and diffing proves the arithmetic here
    # matches shapefit/make_desi_nx.py before trusting it on LRG3_ELG1.
    "LRG2": {
        "stem": "LRG",
        "edges": [round(0.60 + 0.02 * i, 6) for i in range(11)],
        "zmid": [0.610073, 0.630037, 0.650024, 0.66999, 0.690029, 0.710006,
                 0.730057, 0.750048, 0.770128, 0.79015],
    },
}


def build(tracer: str, nfiles: int, cat_dir: Path):
    from astropy.io import fits

    spec = TRACERS[tracer]
    edges = np.asarray(spec["edges"], dtype=np.float64)
    nslice = len(edges) - 1
    S1 = np.zeros(nslice)
    num = np.zeros(nslice)   # sum(NX * W * Wfkp)
    den = np.zeros(nslice)   # sum(W * Wfkp)

    for cap in ("NGC", "SGC"):
        for i in range(nfiles):
            path = cat_dir / f"{spec['stem']}_{cap}_{i}_clustering.ran.fits"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            print(f"  reading {path.name}", flush=True)
            with fits.open(path, memmap=True) as h:
                d = h[1].data
                z = np.asarray(d["Z"], dtype=np.float64)
                w = np.asarray(d["WEIGHT"], dtype=np.float64)
                nx = np.asarray(d["NX"], dtype=np.float64)
                wf = np.asarray(d["WEIGHT_FKP"], dtype=np.float64)
            idx = np.digitize(z, edges) - 1
            ok = (idx >= 0) & (idx < nslice)
            ww = w * wf
            # Accumulate across files rather than concatenating: these are
            # 1-2 GB each and there is no reason to hold them all.
            S1 += np.bincount(idx[ok], weights=w[ok], minlength=nslice)
            num += np.bincount(idx[ok], weights=(nx * ww)[ok], minlength=nslice)
            den += np.bincount(idx[ok], weights=ww[ok], minlength=nslice)

    if np.any(den <= 0):
        raise SystemExit(f"empty slices: {np.flatnonzero(den <= 0).tolist()}")
    return spec, num / den, S1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", default="LRG3_ELG1", choices=sorted(TRACERS))
    ap.add_argument("--nfiles", type=int, default=2,
                    help="random files per cap (default 2, matching the "
                         "committed tables; S1_weight scales linearly with it)")
    ap.add_argument("--cat-dir", type=Path, default=CFS)
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()

    spec, nx, s1 = build(a.tracer, a.nfiles, a.cat_dir)
    out = a.out or Path(f"{a.tracer}_desi_nx.csv")
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["zmid", "zlow", "zhigh", "nbar_desi_nx", "S1_weight"])
        for i, zm in enumerate(spec["zmid"]):
            wr.writerow([zm, spec["edges"][i], spec["edges"][i + 1],
                         f"{nx[i]:.10g}", f"{s1[i]:.10g}"])
    print(f"\nwrote {out}  ({len(spec['zmid'])} slices)")
    print(f"  nbar_desi_nx range {nx.min():.4e} .. {nx.max():.4e}")
    print(f"  S1_weight    total {s1.sum():.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
