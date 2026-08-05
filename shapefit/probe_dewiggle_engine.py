"""S76: is `peakaverage` stable across the mean pipeline's prior box?

S74 measured peakaverage-vs-wallish2018 at ONE cosmology (DESI's MAP) and found
+0.004 in dm, 0.02-0.08 sigma. Its explicit caveat was that nothing showed the
offset holds across `omega_cdm` U[0.01, 0.99] x `h` U[0.2, 1.0] x `ln10A_s`
U[1.61, 3.91] -- and `peakaverage` is the engine that crashed chaotically and
mislabelled sigma ~2x in the BAO path (project_bao_dewiggling_engine).

S76 switched the mean path to `peakaverage`, so that caveat is now load-bearing:
it sits under every mean training label. This probes it before regeneration.

Two things are being asked, and they are NOT the same question:

  1. ROBUSTNESS -- does peakaverage return finite, non-crazy labels everywhere
     in the box? A failure here invalidates the S76 switch outright.
  2. AGREEMENT -- how far do the engines drift apart away from the MAP? A large
     drift does not by itself condemn either engine (we cannot say which is
     right from this), but it bounds how much S74's "0.02-0.08 sigma" understates
     the systematic on training labels.

The chaotic-crash signature from the BAO path is specifically NON-SMOOTHNESS:
labels that jump under a 1e-9 parameter nudge. Finiteness alone would not catch
it, so the box scan is followed by a jitter test at each corner.

Run (mean path only -- no covariance, so this is cheap):
    python probe_dewiggle_engine.py --tracer LRG2
"""
from __future__ import annotations

import argparse
import itertools
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))

import core as sf_core          # noqa: E402
import fourier_space as fs      # noqa: E402
from desilike_emulator.util import ntracers, tracer_area  # noqa: E402

_ENGINES = ("peakaverage", "wallish2018")
_LABELS = ("qiso", "qap", "f_sigmar", "m")

# The MAP, plus the box corners and edge midpoints. omega_b and n_s are Gaussian
# priors and are held at their means -- they are not what the de-wiggling filter
# is sensitive to, and varying them would multiply the grid for nothing.
_FID_B, _FID_NS = 0.02218, 0.9649


def _box_samples(n_tracers: float) -> Dict[str, Dict[str, float]]:
    import desi_reference as dr

    out: Dict[str, Dict[str, float]] = {}
    m = dr.dr1_bestfit_cosmology()
    out["MAP"] = {**m, "N_tracers": n_tracers}

    # Corners of (omega_cdm, h, ln10A_s). 0.01 is the low edge of the omega_cdm
    # prior; note S-era finding that the low-Om end of the box is UNPHYSICAL in
    # the Omega_m basis -- here inputs are omega_cdm directly, so the edge is
    # reachable, but treat low-omega_cdm failures as box pathology, not engine.
    ocdm = (0.01, 0.99)
    hh = (0.2, 1.0)
    lnA = (1.61, 3.91)
    for oc, h, la in itertools.product(ocdm, hh, lnA):
        out[f"oc{oc:g}_h{h:g}_lnA{la:g}"] = {
            "omega_cdm": oc, "omega_b": _FID_B, "h": h,
            "ln10A_s": la, "n_s": _FID_NS, "N_tracers": n_tracers}

    # Interior points, to catch anything that is only pathological mid-box.
    for oc in (0.05, 0.12, 0.30, 0.60):
        out[f"oc{oc:g}_mid"] = {
            "omega_cdm": oc, "omega_b": _FID_B, "h": 0.6736,
            "ln10A_s": 3.044, "n_s": _FID_NS, "N_tracers": n_tracers}
    return out


def _labels_for(sample, tracer, engine, area, z_eff=None) -> List[float] | None:
    """Mean targets under a forced engine. Clears the extractor cache so the
    engine actually changes -- the cache keys on (tracer, z_eff) only."""
    fs._MEAN_EXTRACTOR_CACHE.clear()
    real = fs._get_mean_extractor

    def patched(tracer_bin, z):
        from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumExtractor
        return ShapeFitPowerSpectrumExtractor(z=float(z), fiducial="DESI",
                                              with_now=engine)
    fs._get_mean_extractor = patched
    try:
        s, vals, err = fs._worker_run_mean_targets(
            (dict(sample), tracer, z_eff, None, area, "dr1"))
        return None if vals is None else [float(v) for v in vals]
    finally:
        fs._get_mean_extractor = real
        fs._MEAN_EXTRACTOR_CACHE.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", default="LRG2")
    ap.add_argument("--jitter", type=float, default=1e-9,
                    help="relative nudge on omega_cdm for the smoothness test")
    args = ap.parse_args()

    tracer = args.tracer
    area = tracer_area(tracer, "dr1")
    n_tr = float(ntracers(tracer, "dr1"))
    samples = _box_samples(n_tr)

    print(f"\n{tracer}: {len(samples)} points x {len(_ENGINES)} engines "
          f"(area {area:g} deg2, N {n_tr:.3g})\n")
    hdr = f"{'point':22s} " + " ".join(f"{l:>12s}" for l in _LABELS)
    print(hdr + f" {'engine':>12s}")
    print("-" * len(hdr + " " * 13))

    fails = {e: [] for e in _ENGINES}
    diffs = {l: [] for l in _LABELS}

    for name, sample in samples.items():
        row = {}
        for eng in _ENGINES:
            try:
                vals = _labels_for(sample, tracer, eng, area)
            except Exception as exc:
                vals = None
                print(f"  {name}: {eng} RAISED {type(exc).__name__}: {exc}")
            row[eng] = vals
            if vals is None:
                fails[eng].append(name)
            print(f"{name:22s} " + " ".join(
                f"{v:12.6f}" for v in (vals or [float('nan')] * 4))
                + f" {eng:>12s}")
        a, b = row["peakaverage"], row["wallish2018"]
        if a and b:
            for i, l in enumerate(_LABELS):
                diffs[l].append(a[i] - b[i])
        print()

    print("=" * 72)
    print("ROBUSTNESS")
    for eng in _ENGINES:
        f = fails[eng]
        print(f"  {eng:12s}: {len(samples) - len(f)}/{len(samples)} ok"
              + (f"   FAILED at {f}" if f else ""))

    print("\nAGREEMENT (peakaverage - wallish2018), over points where both ran")
    for l in _LABELS:
        d = np.asarray(diffs[l], dtype=float)
        if d.size:
            print(f"  {l:10s} n={d.size:3d}  mean {d.mean():+.6f}  "
                  f"min {d.min():+.6f}  max {d.max():+.6f}  "
                  f"absmax {np.abs(d).max():.6f}")

    # ---- smoothness: the actual BAO-path failure signature ----
    print("\n" + "=" * 72)
    print(f"SMOOTHNESS  (omega_cdm nudged by {args.jitter:g} relative;")
    print("             a chaotic engine moves labels far more than the nudge)")
    for name in ("MAP", "oc0.01_h0.2_lnA1.61", "oc0.99_h1_lnA3.91", "oc0.05_mid"):
        if name not in samples:
            continue
        base = samples[name]
        nudged = dict(base)
        nudged["omega_cdm"] = base["omega_cdm"] * (1.0 + args.jitter)
        for eng in _ENGINES:
            v0 = _labels_for(base, tracer, eng, area)
            v1 = _labels_for(nudged, tracer, eng, area)
            if v0 is None or v1 is None:
                print(f"  {name:22s} {eng:12s}  n/a (failed)")
                continue
            rel = max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip(v1, v0))
            flag = "  <-- CHAOTIC" if rel > 1e-4 else ""
            print(f"  {name:22s} {eng:12s}  max rel move {rel:.3e}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
