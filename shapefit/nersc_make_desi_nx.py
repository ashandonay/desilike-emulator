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

# Parent random catalogue per tracer bin. Edges and zmid are NOT embedded --
# they come from that tracer's {tracer}_nz_slices.csv, which must be copied
# alongside this script (they are ~2-9 KB each). Embedding them would mean
# QSO's 65 slices transcribed by hand, and a silent mismatch there sends the
# bin down the nbar_file fallback via _desi_nz_geometry's length check.
STEMS = {
    "BGS": "BGS_BRIGHT-21.5",
    "LRG1": "LRG",
    "LRG2": "LRG",
    "LRG3": "LRG",
    "ELG2": "ELG_LOPnotqso",
    "QSO": "QSO",
    "LRG3_ELG1": "LRG+ELG_LOPnotqso",
}


def read_slices(path):
    """(edges, zmid) from a {tracer}_nz_slices.csv, dropping empty slices
    exactly as the pipeline does."""
    import csv as _csv
    lo, hi, zm = [], [], []
    with open(path) as f:
        for row in _csv.DictReader(f):
            if float(row["slice_fraction"]) <= 0.0:
                continue
            lo.append(float(row["zlow"])); hi.append(float(row["zhigh"]))
            zm.append(float(row["zmid"]))
    if not lo:
        raise SystemExit(f"no non-empty slices in {path}")
    return np.array(lo + [hi[-1]]), np.array(zm)


def build(tracer: str, nfiles: int, cat_dir: Path, edges, zmid):
    from astropy.io import fits

    nslice = len(edges) - 1
    S1 = np.zeros(nslice)
    num = np.zeros(nslice)   # sum(NX * W * Wfkp)
    den = np.zeros(nslice)   # sum(W * Wfkp)
    wsum = np.zeros(nslice)  # sum(W)             -- denominator for the two below
    wf_num = np.zeros(nslice)   # sum(Wfkp * W)
    p0_num = np.zeros(nslice)   # sum(((1/Wfkp)-1)/NX * W)

    for cap in ("NGC", "SGC"):
        for i in range(nfiles):
            path = cat_dir / f"{STEMS[tracer]}_{cap}_{i}_clustering.ran.fits"
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
            # DESI's own FKP weight and the per-slice pivot it implies (S82).
            # p0 is back-solved PER OBJECT then averaged: 1/(1+nP) is convex, so
            # deriving it from slice means would import the Jensen bias S49
            # measured at 6.5%.
            good = ok & (nx > 0) & (wf > 0)
            wsum += np.bincount(idx[good], weights=w[good], minlength=nslice)
            wf_num += np.bincount(idx[good], weights=(wf * w)[good], minlength=nslice)
            p0_num += np.bincount(
                idx[good], weights=(((1.0 / wf) - 1.0) / nx * w)[good], minlength=nslice)

    if np.any(den <= 0):
        raise SystemExit(f"empty slices: {np.flatnonzero(den <= 0).tolist()}")
    return num / den, S1, wf_num / wsum, p0_num / wsum


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", nargs="+", default=sorted(STEMS),
                    choices=sorted(STEMS))
    ap.add_argument("--slices-dir", type=Path, default=Path("."),
                    help="where the {tracer}_nz_slices.csv files are "
                         "(default: alongside this script)")
    ap.add_argument("--nfiles", type=int, default=2,
                    help="random files per cap (default 2, matching the "
                         "committed tables; S1_weight scales linearly with it)")
    ap.add_argument("--cat-dir", type=Path, default=CFS)
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()

    for tracer in a.tracers:
        print(f"\n== {tracer} ==")
        edges, zmid = read_slices(a.slices_dir / f"{tracer}_nz_slices.csv")
        nx, s1, wfm, p0e = build(tracer, a.nfiles, a.cat_dir, edges, zmid)
        out = a.out or Path(f"{tracer}_desi_nx.csv")
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["zmid", "zlow", "zhigh", "nbar_desi_nx", "S1_weight",
                         "w_fkp_mean", "p0_eff"])
            for i, zm in enumerate(zmid):
                wr.writerow([zm, edges[i], edges[i + 1],
                             f"{nx[i]:.10g}", f"{s1[i]:.10g}",
                             f"{wfm[i]:.10g}", f"{p0e[i]:.10g}"])
        print(f"  wrote {out}  ({len(zmid)} slices)")
        print(f"  nbar_desi_nx {nx.min():.4e} .. {nx.max():.4e}")
        print(f"  p0_eff       {p0e.min():.0f} .. {p0e.max():.0f}"
              + ("   (constant -> single-tracer)"
                 if p0e.max() / p0e.min() < 1.01 else
                 f"   ({p0e.max()/p0e.min():.2f}x -> MIXED, no scalar pivot works)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
