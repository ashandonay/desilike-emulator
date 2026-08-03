"""Regression harness for the BAO error generator.

Dumps the sigma-triplets -- plus the raw covariance blocks, observable k-grid
and marginalized Fisher underneath them -- over a fixed grid, and exact-compares
two dumps. Run it before and after any change that could perturb the numbers
without changing this repo: a desilike, cosmoprimo, scipy or numpy upgrade.

Worth the ceremony because these outputs become emulator training *labels*. A
silent numerical shift does not announce itself; it mislabels every dataset
generated afterwards, and the cost surfaces much later as a mis-trained model.

Usage
-----
    python regress_sigmas.py dump    --out before.npz     # then change the dep
    python regress_sigmas.py dump    --out after.npz
    python regress_sigmas.py compare before.npz after.npz

`compare` tests exact equality and exits non-zero on any inequality; it reports
max absolute and relative deltas so a real regression is separable from
last-bit noise, but deliberately auto-accepts neither.

A dump takes ~6 min for all 6 tracers; `--config-only` skips the slower Fourier
path. Run from the ``bao/`` directory -- DESI bundle paths resolve relative to
it.
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

import core
import config_space
import fourier_space
from util import ntracers


# ---------------------------------------------------------------------------
# Fixed evaluation grid
# ---------------------------------------------------------------------------
# Deliberately hard-coded rather than sampled: the point is byte-reproducibility
# across two interpreters, so nothing here may depend on RNG, dict ordering, or
# an env-dependent default.
#
# Every point stays inside the PHYSICAL region of DEFAULT_PRIORS. The Om prior
# runs down to 0.01, but omega_cdm goes negative below Om ~= 0.0507 -- roughly
# 4% of the prior box is unphysical, and a CLASS failure there would be
# pre-existing noise rather than a migration signal. Constraints respected:
# valid_densities (0 <= Om + Ok <= 1) and high_z_matter_dom (w0 + wa <= 0).

TRACERS: Tuple[str, ...] = ("BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO")

# (label, Om, Ok, w0, wa, hrdrag, N_factor)  -- N = N_factor * dr1 passed count
COSMO_GRID: Tuple[Tuple[str, float, float, float, float, float, float], ...] = (
    ("fid",       0.3152,  0.00,  -1.00,   0.00,   99.08, 1.00),
    ("lowOm",     0.1500,  0.00,  -1.00,   0.00,   99.08, 1.00),
    ("highOm",    0.6000,  0.00,  -1.00,   0.00,   99.08, 1.00),
    ("openK",     0.3000,  0.20,  -1.00,   0.00,   99.08, 1.00),
    ("closedK",   0.3500, -0.15,  -1.00,   0.00,   99.08, 1.00),
    ("w0wa",      0.3152,  0.00,  -0.80,  -0.60,   99.08, 1.00),
    ("lowN",      0.2800,  0.05,  -1.20,   0.20,   85.00, 0.55),
    ("highN",     0.4000, -0.05,  -0.90,  -0.30,  130.00, 1.45),
)

_AREA = config_space._AREA


def _sample_for(row) -> Tuple[Dict[str, float], float]:
    """Grid row -> (cosmology overrides, absolute N_tracers) for a tracer."""
    _, Om, Ok, w0, wa, hrdrag, _ = row
    return {"Om": Om, "Ok": Ok, "w0": w0, "wa": wa, "hrdrag": hrdrag}, None


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------
def _record(out: Dict[str, np.ndarray], key: str, value) -> None:
    """Store as float64 so equality is checked at full precision."""
    out[key] = np.asarray(value, dtype=np.float64)


def _dump_config(out: Dict[str, np.ndarray], tracer: str) -> None:
    """Config-space sigma-triplets -- the production training-data driver.

    Touches desilike only via BAOPowerSpectrumTemplate / DampedBAOWiggles* /
    get_cosmo, none of which changed numerically, so this is the path that
    should migrate for free.
    """
    gen = config_space.XiSigmaGenerator(tracer)
    N_fid = float(ntracers(tracer, "dr1"))
    for row in COSMO_GRID:
        label, N_factor = row[0], row[6]
        cosmo, _ = _sample_for(row)
        res = gen.sigma_triplet(N_tracers=N_factor * N_fid, **cosmo)
        pfx = f"config/{tracer}/{label}"
        for name in ("DH_over_rs", "DM_over_rs", "DV_over_rs",
                     "DH_over_rd_fid", "DM_over_rd_fid", "DV_over_rd_fid"):
            _record(out, f"{pfx}/{name}", res[name])
        _record(out, f"{pfx}/cov_q", res["cov_q"])
        if "rho_DH_DM" in res:
            _record(out, f"{pfx}/rho_DH_DM", res["rho_DH_DM"])
            _record(out, f"{pfx}/cov_DH_DM", res["cov_DH_DM"])


def _dump_fourier(out: Dict[str, np.ndarray], tracer: str) -> None:
    """Fourier-space path -- where the changed desilike types actually live.

    Captures the raw ObservablesCovarianceMatrix output (C_gauss) and the
    observable's k-grid, since those are exactly what the lsstypes /
    2-D-bin-edge refactor could perturb, then the marginalized Fisher on top.
    """
    cfg = core.TRACER_CONFIGS[tracer]
    apmode = "qiso" if core.is_iso_tracer_bin(tracer, "dr1") else "qparqper"
    N_fid = float(ntracers(tracer, "dr1"))
    for row in COSMO_GRID:
        label, N_factor = row[0], row[6]
        cosmo, _ = _sample_for(row)
        sample = {**core.PARAM_DEFAULTS, **config_space._FID, **cosmo}
        theta_cosmo, hrdrag = core._to_bao_cosmo_params(sample)
        # z_eff is DERIVED here, not pinned from the yaml. Pinning it made the
        # golden blind to the entire z_eff code path: it never reached
        # _compute_z_eff_from_nz, so §36 and §37 both changed production
        # labels while the regression stayed green. The lowN/highN grid rows
        # now also exercise the N dependence (§37a), since z_eff moves with
        # the sample size.
        #
        # NOTE the config dump above deliberately does the opposite -- config
        # space pins z_eff from the DESI bundle by design (§36), so pinning
        # there IS production behaviour.
        info = core.build_bao_likelihood(
            N_tracers=N_factor * N_fid, theta_cosmo=theta_cosmo,
            hrdrag=hrdrag, tracer_bin=tracer, zrange=cfg["zrange"],
            area=_AREA, apmode=apmode,
        )
        pfx = f"fourier/{tracer}/{label}"
        _record(out, f"{pfx}/z_eff", float(info["z_eff"]))

        # Raw desilike surfaces.
        for name, arr in info["cov_components"].items():
            _record(out, f"{pfx}/cov/{name}", arr)
        observable = info["observable"]
        _record(out, f"{pfx}/obs_k", np.concatenate([np.asarray(k) for k in observable.k]))
        _record(out, f"{pfx}/obs_ells", np.asarray(observable.ells))
        _record(out, f"{pfx}/obs_flatdata", observable.flatdata)
        _record(out, f"{pfx}/lik_covariance", info["likelihood"].covariance)

        # Marginalized Fisher on the dilation parameters.
        _record(out, f"{pfx}/F_q", fourier_space._q_fisher_from_bao_likelihood_info(info))

        template = info["template"]
        for name in ("DH_over_rd_fid", "DM_over_rd_fid", "DV_over_rd_fid"):
            _record(out, f"{pfx}/{name}", float(getattr(template, name)))


def cmd_dump(args: argparse.Namespace) -> int:
    out: Dict[str, np.ndarray] = {}
    tracers = args.tracers or list(TRACERS)
    for tracer in tracers:
        print(f"[dump] {tracer} config ...", flush=True)
        _dump_config(out, tracer)
        if not args.config_only:
            print(f"[dump] {tracer} fourier ...", flush=True)
            _dump_fourier(out, tracer)
    np.savez(args.out, **out)
    print(f"[dump] wrote {len(out)} arrays -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Compare
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
        # Not bit-identical -- quantify so a real regression is separable from
        # last-bit noise. Both are reported; neither is auto-accepted.
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
    d.add_argument("--config-only", action="store_true",
                   help="skip the Fourier path (much faster smoke run)")
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
