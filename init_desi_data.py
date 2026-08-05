"""Fetch the DESI data products this pipeline needs, once, on any machine.

Everything here is PUBLIC and anonymous over HTTPS. No NERSC account, no MFA,
no DTN pull. (The recipe recorded in the notes -- scp from dtn01 because
data.desi.lbl.gov was Spin-backed and down -- was an outage workaround. The
public endpoints are back; use them.)

    python init_desi_data.py --dry-run          # what would be fetched, and how big
    python init_desi_data.py                    # the default set
    python init_desi_data.py --what randoms --nran 4

Idempotent: a file already present with the right byte count is skipped, so
re-running is cheap and safe. Partial downloads are resumed (curl -C -) and a
file whose final size disagrees with the server is deleted rather than left to
be mistaken for good data.

WHAT EACH GROUP IS FOR
----------------------
  lss         {tracer}_{cap}_clustering.dat.fits -- the galaxy catalogues.
              shapefit/make_nz_slices.py reads (Z, WEIGHT) to build the n(z)
              slice tables.                                       1.12 GB

  randoms     {tracer}_{cap}_{i}_clustering.ran.fits -- the random catalogues.
              OPTIONAL, and rarely worth it. 0.5-2.2 GB EACH; 20.9 GB at
              nran=2, hence the flag and the exclusion from the default set.

              The {tracer}_desi_nx.csv tables (NX and S1_weight, 2411.12020
              Eqs. 8.1-8.3) are what z_eff is computed from since S53, and they
              can be built EITHER from these randoms or from the `lss` data
              catalogues you already have -- `NX` and `WEIGHT` are columns in
              both. Measured, all six tracers, z_eff vs DESI 2024 V Table 1:

                  from randoms   0.062% mean   0.121% max
                  from data      0.111% mean   0.204% max
                  (pre-S53, for scale:  0.313% mean, 0.653% max)

              The two differ because `S1` from the data is a constant 0.0772x
              the randoms version (0.30% spread -- the data/random density
              ratio, which CANCELS in Eq. 2.1's normalised ratio) and `NX` is
              4.8% high (0.41% spread, nearly z-independent, so it mostly
              cancels too).

              So the randoms buy a factor ~1.8 on a quantity where S62a measured
              n(z) shape errors costing <=0.15% in sigma. Fetch them only to
              reproduce the shipped tables exactly, or for LRG3_ELG1 (the BAO
              combined bin, the one tracer with no table) -- and for that you
              need only the LRG+ELG stem, ~3.3 GB at --nran 1.

              Transient either way: ~20 GB in, ~10 KB of CSV out.

  bao         likelihood_correlation-recon-poles_*.h5 -- window matrix and
              theory s-grid for the config-space path, plus the bao-recon
              post-marginalisation bundles used as the sigma reference.  3 MB

  fs          likelihood_spectrum-poles-rotated_syst-hod_*_thetacut0.05.h5 --
              the DR1 full-shape bundles. These carry the real `norm` and
              `num_shotnoise`; the covariance files ship 0/1 placeholders
              (S43/S46), so nothing else answers the norm question.    4 MB

  cov         DESI's EZmock and RascalC covariances, for compare_to_desi's
              cov check. NOT used to build anything -- ours is analytic. 13 MB

Sizes are measured, not estimated: `--dry-run` HEADs every URL and totals only
what is actually missing.

HOW MANY RANDOMS
----------------
Default 1, and that is a defensible default rather than a lazy one. The two
quantities taken from the randoms behave differently:

  - `nbar_desi_nx` is an OBJECT-weighted mean of NX, which is occupancy-
    independent and converges immediately (S50 measured 4.0083/4.0086/4.0084e-4
    at z=0.60 across NRAN 2/4/8).
  - `S1_weight` is a SUM, so it grows with the number of files -- but it enters
    Eq. (2.1) as (S1 w_fkp)^2 / V inside a normalised ratio, where any constant
    factor cancels.

What does NOT converge quickly is a VOLUME-weighted mean on the mesh (S50:
~0.6% drift per doubling, because a 10 Mpc/h cell straddling the patchy mask
counts as fully occupied and each extra random file finds more edge cells). If
you ever compute one of those, raise --nran and say so in the CHANGELOG.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

DESI = "https://data.desi.lbl.gov/public/dr1"
LSS_DIR = f"{DESI}/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"
VAC_LIK = f"{DESI}/vac/dr1/full-shape-bao-clustering/v1.0/data/likelihood"

VAC_COV_EZ = f"{DESI}/vac/dr1/full-shape-bao-clustering/v1.0/data/covariance/EZmock"
VAC_COV_RC = f"{DESI}/vac/dr1/full-shape-bao-clustering/v1.0/data/covariance/RascalC"

DATA_ROOT = Path.home() / "data" / "desi"
LSS_LOCAL = DATA_ROOT / "lss_dr1"
BAO_LOCAL = DATA_ROOT / "bao_dr1" / "likelihoods"
# compare_to_desi._COV_DIR / config_space flatten EZmock and RascalC into two
# local dirs; the VAC keeps them in separate remote subdirectories.
COV_LOCAL = BAO_LOCAL / "covariance"
COV_RC_LOCAL = BAO_LOCAL / "covariance_rascalc"

# Catalogue stems, matching shapefit/make_nz_slices.STEM. LRG+ELG_LOPnotqso is
# the BAO combined bin (LRG3_ELG1) -- the one tracer with no _desi_nx table, and
# therefore the only reason `nbar_file` still has a fallback to be.
STEMS = ["BGS_BRIGHT-21.5", "LRG", "ELG_LOPnotqso", "QSO", "LRG+ELG_LOPnotqso"]

# Randoms are per PARENT sample; the combined bin reuses LRG+ELG.
RANDOM_STEMS = ["BGS_BRIGHT-21.5", "LRG", "ELG_LOPnotqso", "QSO", "LRG+ELG_LOPnotqso"]

CAPS = ["NGC", "SGC"]

_BAO_SAMPLES = [
    "BGS_BRIGHT-21.5_GCcomb_z0.1-0.4",
    "LRG_GCcomb_z0.4-0.6",
    "LRG_GCcomb_z0.6-0.8",
    "LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1",
    "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "QSO_GCcomb_z0.8-2.1",
]
# Mirrors compare_to_desi._DESI_SAMPLE. LRG3 uses the LRG-only 0.8-1.1 sample
# here, NOT the combined bin the BAO path uses -- full shape and BAO split that
# redshift range differently.
_FS_SAMPLES = [
    "BGS_BRIGHT-21.5_GCcomb_z0.1-0.4",
    "LRG_GCcomb_z0.4-0.6",
    "LRG_GCcomb_z0.6-0.8",
    "LRG_GCcomb_z0.8-1.1",
    "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "QSO_GCcomb_z0.8-2.1",
]


class Item(NamedTuple):
    url: str
    dest: Path
    group: str


def _url(base: str, name: str) -> str:
    """DESI serves `+` percent-encoded. The LOCAL filename keeps the literal
    `+` (that is what every consumer in this repo globs for), so the encoding
    must happen on the URL side only -- encoding both, or neither, silently
    yields a 404 or an unfindable file."""
    return f"{base}/{urllib.parse.quote(name, safe='')}"


def manifest(nran: int) -> List[Item]:
    items: List[Item] = []
    for stem in STEMS:
        for cap in CAPS:
            n = f"{stem}_{cap}_clustering.dat.fits"
            items.append(Item(_url(LSS_DIR, n), LSS_LOCAL / n, "lss"))
    for stem in RANDOM_STEMS:
        for cap in CAPS:
            for i in range(nran):
                n = f"{stem}_{cap}_{i}_clustering.ran.fits"
                items.append(Item(_url(LSS_DIR, n), LSS_LOCAL / n, "randoms"))
    for s in _BAO_SAMPLES:
        for kind in ("correlation-recon-poles", "bao-recon_stat-only",
                     "bao-recon_syst"):
            n = f"likelihood_{kind}_{s}.h5"
            items.append(Item(_url(VAC_LIK, n), BAO_LOCAL / n, "bao"))
    for s in _FS_SAMPLES:
        # Mirrors compare_to_desi._FS_BUNDLES exactly. These carry the real
        # `norm` and `num_shotnoise`; the covariance files ship 0/1 placeholders
        # (S43/S46), so this is the only product that answers the norm question.
        n = f"likelihood_spectrum-poles-rotated_syst-hod_{s}_thetacut0.05.h5"
        items.append(Item(_url(VAC_LIK, n), BAO_LOCAL / n, "fs"))
        # DESI's EZmock covariance, both cuts -- compare_to_desi._cov_path
        # wants plain and _thetacut0.05, and the rotated products exist only
        # with the cut.
        for stem in ("covariance_spectrum-poles+bao-recon",
                     "covariance_spectrum-poles-rotated+bao-recon"):
            for suffix in ("", "_thetacut0.05"):
                if "rotated" in stem and not suffix:
                    continue
                n = f"{stem}_{s}{suffix}.h5"
                items.append(Item(_url(VAC_COV_EZ, n), COV_LOCAL / n, "cov"))
    for s in _BAO_SAMPLES:
        n = f"covariance_correlation-recsym-poles_{s}.h5"
        items.append(Item(_url(VAC_COV_RC, n), COV_RC_LOCAL / n, "cov"))
    return items


def remote_size(url: str, timeout: int = 30) -> Optional[int]:
    """Content-Length, or None if the server will not say."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n is not None else None
    except Exception:
        return None


def fetch(item: Item, timeout: int, force: bool) -> str:
    """-> 'skip' | 'ok' | 'fail'. Verifies the final size against the server."""
    item.dest.parent.mkdir(parents=True, exist_ok=True)
    want = remote_size(item.url, timeout)

    if item.dest.exists():
        if force:
            # `curl -C -` against a COMPLETE local file resumes from the end,
            # finds nothing to do and exits 0 -- so without this unlink, --force
            # would silently be a no-op, which is the opposite of what it says.
            item.dest.unlink()
        else:
            have = item.dest.stat().st_size
            if want is None or have == want:
                return "skip"
            # Short file: resume. `-C -` appends from the current end, which is
            # correct for an interrupted download (the bytes already written are
            # the file's real prefix) and is what makes a dropped 2 GB random
            # cheap to finish. It would be WRONG for a file corrupted in the
            # middle, but nothing here produces that -- curl only ever appends.
            print(f"    size {have} != remote {want}; resuming", flush=True)

    cmd = ["curl", "-fsSL", "-C", "-", "--retry", "3", "--retry-delay", "5",
           "--max-time", str(timeout * 60), "-o", str(item.dest), item.url]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"    FAILED ({exc.returncode})", flush=True)
        return "fail"

    if want is not None and item.dest.exists():
        got = item.dest.stat().st_size
        if got != want:
            # Never leave a short file behind: it would pass the `.exists()`
            # check on the next run and be read as though it were complete.
            print(f"    TRUNCATED {got} != {want}; removing", flush=True)
            item.dest.unlink()
            return "fail"
    return "ok"


def human(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", default=["lss", "bao", "fs", "cov"],
                    choices=["lss", "randoms", "bao", "fs", "cov", "all"],
                    help="groups to fetch. Default omits `randoms`: they are "
                         "0.5-2.2 GB each and buy only 0.062%% vs 0.111%% on "
                         "z_eff over the data catalogues (see module docstring).")
    ap.add_argument("--nran", type=int, default=1,
                    help="random files per tracer per cap (default 1: the "
                         "object-weighted NX mean is occupancy-independent and "
                         "converges immediately, S50)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="refetch even if present at the right size")
    ap.add_argument("--timeout", type=int, default=60,
                    help="per-file timeout in MINUTES (default 60)")
    a = ap.parse_args()

    groups = ({"lss", "randoms", "bao", "fs", "cov"} if "all" in a.what
              else set(a.what))
    items = [i for i in manifest(a.nran) if i.group in groups]

    print(f"DESI DR1 data init -> {DATA_ROOT}")
    print(f"groups: {', '.join(sorted(groups))}"
          + (f"   (nran={a.nran})" if "randoms" in groups else ""))
    print(f"{len(items)} files\n")

    if a.dry_run:
        total, missing = 0, 0
        for i in items:
            have = i.dest.exists()
            sz = remote_size(i.url)
            if not have:
                missing += 1
                total += sz or 0
            print(f"  [{'have' if have else ' -- '}] {i.group:8s} "
                  f"{human(sz):>9s}  {i.dest.name}")
        print(f"\nwould fetch {missing} file(s), {human(total)}")
        return 0

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed: List[str] = []
    for n, i in enumerate(items, 1):
        print(f"[{n}/{len(items)}] {i.dest.name}", flush=True)
        r = fetch(i, a.timeout, a.force)
        counts[r] += 1
        if r == "fail":
            failed.append(i.dest.name)

    print(f"\n{counts['ok']} fetched, {counts['skip']} already present, "
          f"{counts['fail']} failed")
    if failed:
        print("\nFAILED:")
        for f in failed:
            print(f"  {f}")
        return 1

    print("\nNext:")
    print("  python shapefit/make_nz_slices.py --install    # n(z) slice tables")
    print("  python shapefit/benchmark_desi.py              # verify vs DESI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
