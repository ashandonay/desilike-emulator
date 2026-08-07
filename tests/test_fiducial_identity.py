"""Unit test: cosmoprimo's DESI fiducial has not moved.

This is the one check in the suite whose value is catching a change nobody in
this repo made. Every ShapeFit mean training label is defined RELATIVE to
whatever `fiducial="DESI"` resolves to inside desilike's
ShapeFitPowerSpectrumExtractor, so if that object moves, every label silently
changes meaning -- and every ratio-based check still passes, because the
fiducial sits in both numerator and denominator (shapefit CHANGELOG S38-S40).

`DESI` is an alias upstream (`DESI = AbacusSummitBase` in cosmoprimo/fiducial.py),
and the same module already ships DESIDR2Flatw0waCDM with materially different
values. cosmoprimo is pinned by SHA but has been upgraded in place before, and
trusting an upstream default has already cost us once: `with_now='peakaverage'`
mislabelled sigma ~2x.

The other checks in validate_forecast.py stay manual -- they each need CLASS
runs over six tracers. This one is a few seconds and asserts on constants,
which is what makes it worth automating.

Run:  pytest tests/ -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "shapefit"))

cosmoprimo = pytest.importorskip(
    "cosmoprimo",
    reason="needs the emulator env (~/miniconda3/envs/emulator/bin/python)",
)


@pytest.fixture(scope="module")
def check():
    """The assertion lives in validate_forecast so the manual CLI and the test
    can never disagree about what the fiducial is supposed to be."""
    import validate_forecast

    return validate_forecast


def test_fiducial_identity_unchanged(check):
    """Raises SystemExit listing every parameter that moved."""
    check.check_fiducial_identity()


def test_fiducial_reference_values_are_self_consistent(check):
    """The recorded ln10A_s/omega_cdm must be the ones the mean pipeline's own
    fiducial sample uses, up to the rounding S40 documented.

    Guards against someone editing one encoding of the fiducial and not the
    other. The tolerance is 1e-6, which admits the known 8.4e-8 ln10A_s
    rounding in FID_SAMPLE and nothing larger.
    """
    for key, recorded in check._FIDUCIAL_PARAMS.items():
        if key not in check.FID_SAMPLE:
            continue
        ours = check.FID_SAMPLE[key]
        assert abs(ours / recorded - 1.0) < 1e-6, (
            f"{key}: FID_SAMPLE has {ours!r}, cosmoprimo's DESI has "
            f"{recorded!r} -- the two encodings of the fiducial disagree"
        )
