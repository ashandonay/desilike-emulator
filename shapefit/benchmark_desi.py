#!/usr/bin/env python
"""One-shot benchmark of the pipeline against DESI DR1 published values.

Every quantity we can check without a fit, in one table, so the S43-S54 fixes
are measured together rather than one at a time. Run before and after any
change that touches n(z), the FKP pivot, the footprint or z_eff.

    python shapefit/benchmark_desi.py            # current code
    python shapefit/benchmark_desi.py --json out.json

References, all primary:
  z_eff      DESI 2024 V (2411.12021) Table 1
  P_shot     the DR1 full-shape bundles, num_shotnoise / norm
  areas      DESI 2024 II (2411.12020) Table 2
  pivots     DESI 2024 II Eq. (8.4)

The three baselines quoted in `--help` output are:
  ORIGINAL   before S53 (nbar_file + Table-2 pivots + 7500 deg^2 everywhere)
  REFERENCE  Eq. (2.1) evaluated directly on DESI's own randoms (S51)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "bao"))

warnings.filterwarnings("ignore")

import core as bao_core  # noqa: E402  (bao/core.py)
from util import ntracers, tracer_area  # noqa: E402
from desilike.theories.primordial_cosmology import get_cosmo  # noqa: E402

TRACERS = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")

# DESI 2024 V Table 1
Z_EFF_PUB = {"BGS": 0.295, "LRG1": 0.510, "LRG2": 0.706,
             "LRG3": 0.919, "ELG2": 1.317, "QSO": 1.491}
# num_shotnoise / norm from the DR1 full-shape bundles (S43)
P_SHOT_PUB = {"BGS": 5723.11, "LRG1": 5081.53, "LRG2": 5229.50,
              "LRG3": 9573.55, "ELG2": 10691.98, "QSO": 47376.96}
# DESI 2024 II Table 2
AREA_PUB = {"BGS": 7473., "LRG1": 5740., "LRG2": 5740.,
            "LRG3": 5740., "ELG2": 5924., "QSO": 7249.}
# DESI 2024 II Eq. (8.4)
PIVOT_PUB = {"BGS": 7000., "LRG1": 10000., "LRG2": 10000.,
             "LRG3": 10000., "ELG2": 4000., "QSO": 6000.}

ORIGINAL_ZEFF = {"BGS": 0.2958, "LRG1": 0.5096, "LRG2": 0.7058,
                 "LRG3": 0.9224, "ELG2": 1.3256, "QSO": 1.4839}
REFERENCE_ZEFF = {"BGS": 0.2954, "LRG1": 0.5095, "LRG2": 0.7058,
                  "LRG3": 0.9185, "ELG2": 1.3169, "QSO": 1.4901}


def _pct(a, b):
    return 100.0 * (a / b - 1.0) if b else float("nan")


def collect():
    cosmo = get_cosmo("DESI")
    rows = {}
    for t in TRACERS:
        area = tracer_area(t, "dr1")
        z = bao_core._desi_z_eff_from_nz(t, cosmo, area, data_release="dr1")
        N = float(ntracers(t, "dr1"))
        zN = [bao_core._desi_z_eff_from_nz(t, cosmo, area, n_tracers=f * N,
                                           data_release="dr1")
              for f in (0.5, 1.0, 1.5)]
        rows[t] = {
            "z_eff": z,
            "z_eff_pub": Z_EFF_PUB[t],
            "z_eff_err_pct": _pct(z, Z_EFF_PUB[t]),
            "z_eff_orig_err_pct": _pct(ORIGINAL_ZEFF[t], Z_EFF_PUB[t]),
            "z_eff_ref_err_pct": _pct(REFERENCE_ZEFF[t], Z_EFF_PUB[t]),
            "N_span_pct": 100.0 * (zN[2] / zN[0] - 1.0),
            "area": area,
            "area_pub": AREA_PUB[t],
            "area_err_pct": _pct(area, AREA_PUB[t]),
            "pivot": bao_core._fkp_p0_for_tracer(t),
            "pivot_pub": PIVOT_PUB[t],
        }
    return rows


def report(rows, fh=sys.stdout):
    def w(s=""):
        print(s, file=fh)

    w("=" * 74)
    w("DESI DR1 benchmark".center(74))
    w("=" * 74)

    w("\n1. z_eff vs DESI 2024 V Table 1")
    w(f"   {'bin':6s} {'z_eff':>8s} {'pub':>7s} {'err %':>8s} "
      f"{'orig %':>8s} {'ref %':>7s} {'N span %':>9s}")
    e, e0, er = [], [], []
    for t in TRACERS:
        r = rows[t]
        e.append(abs(r["z_eff_err_pct"]))
        e0.append(abs(r["z_eff_orig_err_pct"]))
        er.append(abs(r["z_eff_ref_err_pct"]))
        w(f"   {t:6s} {r['z_eff']:8.4f} {r['z_eff_pub']:7.3f} "
          f"{r['z_eff_err_pct']:+8.3f} {r['z_eff_orig_err_pct']:+8.3f} "
          f"{r['z_eff_ref_err_pct']:+7.3f} {r['N_span_pct']:+9.2f}")
    w(f"   {'':6s} {'mean|err|':>17s} {np.mean(e):8.3f} {np.mean(e0):8.3f} "
      f"{np.mean(er):7.3f}")
    w(f"   {'':6s} {'max |err|':>17s} {np.max(e):8.3f} {np.max(e0):8.3f} "
      f"{np.max(er):7.3f}")
    w("   orig = pre-S53 (nbar_file, Table-2 pivots, 7500 deg^2 everywhere)")
    w("   ref  = Eq. (2.1) on DESI's own randoms (S51) -- the target")
    w("   N span = z_eff change across [0.5, 1.5] x N_dr1; nonzero is REQUIRED")
    w("            (S42: a constant bias cancels between two N, an N-dependent")
    w("             one does not, and comparing N values is the emulator's job)")

    w("\n2. Footprint vs DESI 2024 II Table 2")
    w(f"   {'bin':6s} {'area':>8s} {'pub':>8s} {'err %':>8s} {'vs 7500':>9s}")
    for t in TRACERS:
        r = rows[t]
        w(f"   {t:6s} {r['area']:8.0f} {r['area_pub']:8.0f} "
          f"{r['area_err_pct']:+8.3f} {7500.0 / r['area']:9.3f}")
    w("   'vs 7500' is the nbar correction relative to the nominal footprint")

    w("\n3. FKP pivot vs DESI 2024 II Eq. (8.4)")
    w(f"   {'bin':6s} {'pivot':>9s} {'pub':>9s} {'match':>7s}")
    for t in TRACERS:
        r = rows[t]
        ok = "OK" if abs(r["pivot"] - r["pivot_pub"]) < 1e-6 else "MISMATCH"
        w(f"   {t:6s} {r['pivot']:9.0f} {r['pivot_pub']:9.0f} {ok:>7s}")

    w("\nNot covered here (needs a fit or NERSC): sigma vs DESI's published")
    w("compressed constraints (compare_to_desi.py --check compressed), the")
    w("covariance level (S45, quarantined), and P_shot (S46, norm not yet")
    w("implemented in the covariance path).")
    w("=" * 74)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the raw numbers here")
    a = ap.parse_args()
    rows = collect()
    report(rows)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
