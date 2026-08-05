#!/usr/bin/env python3
"""Generate LRG-only n(z) slices for the full-shape 0.8-1.1 bin.

DESI's BAO analysis combines LRG3 + ELG1 there, but the full-shape analysis
does NOT: ELG1 failed the pre-unblinding fibre-collision tests for growth-rate
measurements (DESI 2024 V Sec 2; shapefit CHANGELOG S31). `bao/parse_desi_nz.py`
only knows the BAO bin definitions, and `bao/` is regression-frozen, so this
driver injects an `LRG3` bin into the module's tables at runtime and reuses the
same slicing code rather than forking it.

Writes ``data/dr1/nz_slices/LRG3_nz_slices.csv`` in the repo (S80). Does not touch the existing
LRG3_ELG1 slice file, which `bao/` still needs.

    python make_lrg3_nz_slices.py [--nz-dir DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

_BIN = "LRG3"
_ZRANGE = (0.8, 1.1)
_TARGETS = ["LRG"]          # LRG-only: the whole point


def _load_parser_module():
    """Import bao/parse_desi_nz.py by path (never `import parse_desi_nz`, which
    would be ambiguous once both bao/ and shapefit/ are on sys.path)."""
    path = _HERE.parent / "bao" / "parse_desi_nz.py"
    spec = importlib.util.spec_from_file_location("bao_parse_desi_nz", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bao_parse_desi_nz"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nz-dir", default=str(Path.home() / "data" / "desi" / "nz_data"))
    p.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parent.parent / "data" / "dr1" / "nz_slices"))
    p.add_argument("--caps", nargs="+", default=["NGC", "SGC"])
    p.add_argument("--coarsen-dz", type=float, default=0.02)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    mod = _load_parser_module()

    # Inject the full-shape bin. Restricted to LRG3 alone so the shared slicing
    # code runs exactly as it does for the BAO bins -- same trimming, same
    # coarsening, same shape rescaling.
    mod.BAO_BINS = {_BIN: _ZRANGE}
    targets = mod.build_targets_by_bin(elg_target="ELG_LOPnotqso", bgs_target="BGS_ANY")
    targets[_BIN] = list(_TARGETS)

    # N_design: LRG3's own passed count, not the LRG3+ELG1 combination.
    from util import ntracers
    n_design = float(ntracers(_BIN, "dr1"))
    print(f"{_BIN}: zrange={_ZRANGE}  targets={_TARGETS}  N_design={n_design:.0f}")

    slices = mod.build_all_tracer_slices(
        nz_dir=Path(args.nz_dir),
        targets_by_bin=targets,
        caps=args.caps,
        design_counts={_BIN: n_design},
        coarsen_dz=args.coarsen_dz,
    )
    df = slices[_BIN]
    print(f"  {len(df)} slices, sum(slice_fraction) = {df['slice_fraction'].sum():.6f}")
    print(f"  z coverage {df['zlow'].min():.3f} - {df['zhigh'].max():.3f}")
    areas = sorted(set(df["file_area_deg2"].round(1)))
    print(f"  file_area_deg2 distinct values: {areas}"
          + ("   <-- non-uniform, see CHANGELOG S28" if len(areas) > 1 else ""))

    out = Path(args.out_dir) / f"{_BIN}_nz_slices.csv"
    if args.dry_run:
        print(f"  [dry-run] would write {out}")
        return 0
    if out.exists():
        print(f"  REFUSING to overwrite existing {out}")
        return 1
    df.to_csv(out, index=False)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
