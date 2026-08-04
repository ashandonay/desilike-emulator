"""Regression harness for the ShapeFit error/mean generators.

Dumps the 10 emulator targets (4 sigmas + 6 rhos) -- plus the raw covariance
blocks, observable k-grid, marginalized 4x4 physical covariance, template
fiducials and the mean-pipeline (qiso, qap, f_sigmar, m) -- over a fixed grid,
and exact-compares two dumps. Run it before and after any change that could
perturb the numbers without changing this repo: a desilike, cosmoprimo, scipy
or numpy upgrade (same rationale as bao/regress_sigmas.py: these outputs are
emulator training labels; a silent numeric shift mislabels every dataset
generated afterwards).

Usage
-----
    python regress_sigmas.py dump    --out before.npz     # then change the dep
    python regress_sigmas.py dump    --out after.npz
    python regress_sigmas.py compare before.npz after.npz

`compare` tests exact equality and exits non-zero on any inequality; it
reports max absolute and relative deltas so a real regression is separable
from last-bit noise, but deliberately auto-accepts neither.

Run from the ``shapefit/`` directory.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import fourier_space
from fourier_space import sf_core
from util import ntracers, tracer_area


# ---------------------------------------------------------------------------
# Fixed evaluation grid
# ---------------------------------------------------------------------------
# Deliberately hard-coded rather than sampled: the point is byte-
# reproducibility across two interpreters, so nothing here may depend on RNG,
# dict ordering, or an env-dependent default. Every point stays inside
# DEFAULT_PRIORS (omega basis) and respects high_z_matter_dom (w0 + wa <= 0).

TRACERS: Tuple[str, ...] = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")

# (label, omega_cdm, omega_b, h, ln10A_s, n_s, w0, wa, N_factor)
#   N = N_factor * dr1 passed count
COSMO_GRID: Tuple[Tuple[str, float, float, float, float, float, float, float, float], ...] = (
    ("fid",     0.1200, 0.02237, 0.6736, 3.036394, 0.9649, -1.00,  0.00, 1.00),
    ("lowOc",   0.0500, 0.02237, 0.6736, 3.036394, 0.9649, -1.00,  0.00, 1.00),
    ("highOc",  0.3000, 0.02237, 0.6736, 3.036394, 0.9649, -1.00,  0.00, 1.00),
    ("lowH",    0.1200, 0.02180, 0.4500, 3.036394, 0.9649, -1.00,  0.00, 1.00),
    ("highH",   0.1200, 0.02290, 0.9000, 3.036394, 0.9649, -1.00,  0.00, 1.00),
    ("lowA",    0.1200, 0.02237, 0.6736, 2.300, 0.9000, -1.00,  0.00, 1.00),
    ("highA",   0.1250, 0.02237, 0.6736, 3.700, 1.0300, -1.00,  0.00, 0.55),
    ("w0wa",    0.1200, 0.02237, 0.6736, 3.036394, 0.9649, -0.80, -0.60, 1.45),
)

# The footprint is per TRACER, not per release (S54), so this cannot be a
# module constant at all. It was `dataset_area("dr1")` = 7500 for every tracer,
# which is 1.31x the true LRG area and 1.27x the ELG area -- and because
# build_shapefit_likelihood only falls back to tracer_area when `area is None`,
# passing it explicitly OVERRODE the corrected geometry. The golden was
# therefore pinning a footprint production does not use, and would have gone on
# passing while the real path changed underneath it.
#
# Same failure as the bao golden pinning z_eff (commit 19dc4b3): a harness that
# freezes an input stops testing the code that derives it.
def _area(tracer: str) -> float:
    return tracer_area(tracer, "dr1")


def _sample_for(row) -> Dict[str, float]:
    """Grid row -> full sample dict (without N_tracers)."""
    _, oc, ob, h, lnA, ns, w0, wa, _ = row
    return {"omega_cdm": oc, "omega_b": ob, "h": h,
            "ln10A_s": lnA, "n_s": ns, "w0": w0, "wa": wa}


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------
def _record(out: Dict[str, np.ndarray], key: str, value) -> None:
    """Store as float64 so equality is checked at full precision."""
    out[key] = np.asarray(value, dtype=np.float64)


def _dump_covar(out: Dict[str, np.ndarray], tracer: str) -> None:
    """Error-pipeline path: build the likelihood, capture the raw desilike
    surfaces (C_gauss / C_SSC / C_total, observable k-grid and data vector)
    and the marginalized physical 4x4 + the 10 emulator targets on top."""
    N_fid = float(ntracers(tracer, "dr1"))
    for row in COSMO_GRID:
        label, N_factor = row[0], row[8]
        sample = _sample_for(row)
        theta = sf_core._to_shapefit_cosmo_params(sample)
        info = sf_core.build_shapefit_likelihood(
            N_tracers=N_factor * N_fid,
            theta_cosmo=theta,
            tracer_bin=tracer,
            area=_area(tracer),
        )
        pfx = f"covar/{tracer}/{label}"

        # Raw desilike surfaces.
        for name, arr in info["cov_components"].items():
            _record(out, f"{pfx}/cov/{name}", arr)
        observable = info["observable"]
        _record(out, f"{pfx}/obs_k",
                np.concatenate([np.asarray(k) for k in observable.k]))
        _record(out, f"{pfx}/obs_ells", np.asarray(observable.ells))
        _record(out, f"{pfx}/obs_flatdata", observable.flatdata)

        # Template fiducials + derived z_eff.
        _record(out, f"{pfx}/z_eff", info["z_eff"])
        _record(out, f"{pfx}/f_sigmar_fid", info["f_sigmar_fid"])
        _record(out, f"{pfx}/m_fid", info["m_fid"])

        # Marginalized physical 4x4 and the emulator targets.
        cov_phys = fourier_space._sf_fisher_reduction(info)
        _record(out, f"{pfx}/cov_phys", cov_phys)
        targets = fourier_space.fisher_cov_to_emulator_targets(cov_phys)
        for name, val in zip(fourier_space.TARGET_NAMES, targets):
            _record(out, f"{pfx}/{name}", val)


def _dump_mean(out: Dict[str, np.ndarray], tracer: str) -> None:
    """Mean-pipeline path: per-tracer extractor at the fiducial-cosmology
    z_eff derived per sample, the same convention generate_mean_data.py uses."""
    fid_sample = _sample_for(COSMO_GRID[0])
    theta_fid = sf_core._to_shapefit_cosmo_params(fid_sample)
    from desilike.theories.primordial_cosmology import get_cosmo

    cosmo_fid = get_cosmo(("DESI", dict(theta_fid)))
    fo_fid = cosmo_fid.get_fourier()
    from util import get_tracer_config, ntracers

    cfg = get_tracer_config(tracer)
    # Record the fiducial-cosmology, DR1-count z_eff as a diagnostic anchor
    # only. Production no longer uses a single frozen z_eff (S42), so this is
    # NOT the z the rows below are evaluated at -- passing z_eff=None makes the
    # worker derive it per sample from that sample's cosmology AND N_tracers,
    # which is what generate_mean_data.py does. Pinning it here would leave the
    # harness blind to the very dependence S42 added.
    try:
        z_eff = sf_core._fs_compute_z_eff(
            tracer_bin=tracer, cosmo=cosmo_fid, fo=fo_fid,
            area_deg2=_area(tracer), b1=float(cfg.get("bias_recon", 2.0)),
            n_tracers=ntracers(tracer, "dr1"), dataset="dr1",
        )
    except (FileNotFoundError, ValueError):
        z_eff = float(cfg["z_eff"])
    _record(out, f"mean/{tracer}/z_eff", z_eff)

    N_fid = float(ntracers(tracer, "dr1"))
    for row in COSMO_GRID:
        label, N_factor = row[0], row[8]
        # N_tracers is a mean-emulator INPUT since S42 and it moves z_eff, so
        # the grid's lowN/highN rows must reach the mean dump too -- otherwise
        # the golden pins a dependence it never exercises.
        sample = {**_sample_for(row), "N_tracers": N_factor * N_fid}
        _s, vals, tb = fourier_space._worker_run_mean_targets(
            (sample, tracer, None, None, _area(tracer), "dr1")
        )
        if vals is None:
            raise RuntimeError(f"mean extractor failed for {tracer}/{label}:\n{tb}")
        for name, val in zip(sf_core.MEAN_TARGET_NAMES, vals):
            _record(out, f"mean/{tracer}/{label}/{name}", val)


def cmd_dump(args: argparse.Namespace) -> int:
    out: Dict[str, np.ndarray] = {}
    tracers = args.tracers or list(TRACERS)
    for tracer in tracers:
        print(f"[dump] {tracer} mean ...", flush=True)
        _dump_mean(out, tracer)
        if not args.mean_only:
            print(f"[dump] {tracer} covar ...", flush=True)
            _dump_covar(out, tracer)
    np.savez(args.out, **out)
    print(f"[dump] wrote {len(out)} arrays -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Compare (verbatim logic from bao/regress_sigmas.py)
# ---------------------------------------------------------------------------
def cmd_compare(args: argparse.Namespace) -> int:
    a = np.load(args.a)
    b = np.load(args.b)
    keys_a, keys_b = set(a.files), set(b.files)

    rc = 0
    only_a, only_b = sorted(keys_a - keys_b), sorted(keys_b - keys_a)
    for label, keys in (("only in A", only_a), ("only in B", only_b)):
        if keys:
            rc = 1
            print(f"KEY MISMATCH ({label}): {len(keys)} keys, e.g. {keys[:5]}")

    diffs: List[Tuple[str, float, float]] = []
    for key in sorted(keys_a & keys_b):
        xa, xb = a[key], b[key]
        if xa.shape != xb.shape:
            rc = 1
            print(f"SHAPE  {key}: {xa.shape} vs {xb.shape}")
            continue
        if np.array_equal(xa, xb):
            continue
        adiff = np.abs(xa - xb)
        scale = np.maximum(np.abs(xa), np.abs(xb))
        with np.errstate(divide="ignore", invalid="ignore"):
            rdiff = np.where(scale > 0, adiff / scale, 0.0)
        diffs.append((key, float(np.nanmax(adiff)), float(np.nanmax(rdiff))))

    if diffs:
        rc = 1
        diffs.sort(key=lambda t: -t[2])
        print(f"\n{len(diffs)} / {len(keys_a & keys_b)} arrays differ "
              f"(sorted by max relative delta):\n")
        for key, amax, rmax in diffs[:args.max_report]:
            print(f"  {rmax:10.3e} rel  {amax:12.5e} abs   {key}")
        if len(diffs) > args.max_report:
            print(f"  ... and {len(diffs) - args.max_report} more")
        print("\nFAIL: outputs are not identical. Trace each delta to a "
              "specific line before adopting the new env -- these are "
              "training labels, so 'it's only 1e-16' is not a pass.")
    elif rc == 0:
        print(f"OK: all {len(keys_a & keys_b)} arrays bit-identical.")

    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="evaluate the fixed grid and save to .npz")
    d.add_argument("--out", required=True)
    d.add_argument("--tracers", nargs="*", default=None,
                   help=f"subset of {list(TRACERS)}; default all")
    d.add_argument("--mean-only", action="store_true",
                   help="skip the covar path (much faster smoke run)")
    d.set_defaults(func=cmd_dump)

    c = sub.add_parser("compare", help="exact-equality compare of two dumps")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--max-report", type=int, default=25)
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
