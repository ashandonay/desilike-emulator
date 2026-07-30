"""DESI DR1 published ShapeFit compressed-parameter constraints.

Source: DESI 2024 V, "Full-Shape Galaxy Clustering from Galaxies and Quasars"
(arXiv:2411.12021), Appendix A, "Datavectors and covariances for the compressed
ShapeFit parameters". Transcribed from Eqs. (A.1)-(A.24).

This is the reference the whole shapefit validation effort was missing: DESI's
own per-tracer compressed constraints, at DR1 volumes, in a basis one division
away from our emulator targets. The analogue of bao/desi_reference.py for the
full-shape side.

Basis
-----
DESI reports the 4-vector

    D_SF = [D_V/r_d,  D_H/D_M,  f sigma_s8,  m + n]

with covariance in units of 1e-4. Our emulator targets map onto it as

    qiso     = (D_V/r_d)  / (D_V/r_d)_fid       -> sigma(qiso) = sigma / value
    qap      = (D_H/D_M)  / (D_H/D_M)_fid       -> sigma(qap)  = sigma / value
    f_sigmar ~ f sigma_s8                       -> compare FRACTIONAL errors
    m        ~ m + n                            (our dn is fixed, so n does not vary)

Ratios of a quantity to its fiducial leave correlations unchanged, so the six
rho_* targets compare directly with no conversion at all.

Dividing sigma by the MEASURED central value rather than the fiducial is a ~1%
approximation (DR1 is consistent with the fiducial at that level) and is the
reason `sigma_targets` returns fractional errors rather than pretending to a
precision it does not have.

Two fit variants are published and both are recorded here:

  "sf"      ShapeFit alone. **Use this one** for comparison against our
            forecast, which is pre-recon power-spectrum only.
  "sf_bao"  ShapeFit combined with the post-reconstruction BAO fits. Tighter,
            especially on qiso -- DESI's headline full-shape numbers. Comparing
            our power-only forecast against these would understate us.

Caveats
-------
* DESI's z_eff differ slightly from ours (they are volume-weighted, ours are
  Fisher-weighted; bao CHANGELOG S18). Sigmas are not strongly z-sensitive over
  these offsets, but f sigma_s8 evolves fast -- do not compare its ABSOLUTE
  value across a z offset, only the fractional error.
* **LRG3 is not our LRG3_ELG1.** DESI's full-shape 0.8-1.1 bin is LRG-only; our
  bin is the combined LRG+ELG1 sample used by the BAO analysis. Different
  galaxy sample, different density. Flagged by `SAMPLE_MISMATCH`.
* The BGS entry carries DESI's own warning that alpha_AP is not well
  constrained there and D_H/D_M is "highly affected by the flat prior between
  0.8 and 1.2". Its sigma(qap) is a prior width, not a measurement.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Order of the DESI 4-vector, and the emulator target each entry maps to.
DESI_ORDER = ("DV_over_rd", "DH_over_DM", "f_sigma_s8", "m_plus_n")
TARGET_ORDER = ("qiso", "qap", "f_sigmar", "m")

SAMPLE_MISMATCH = {
    "LRG3": "DESI full-shape 0.8-1.1 is LRG-only; our LRG3_ELG1 is LRG+ELG1",
}

# Our tracer-bin names -> DESI's appendix labels.
TRACER_MAP = {
    "BGS": "BGS", "LRG1": "LRG1", "LRG2": "LRG2",
    "LRG3_ELG1": "LRG3", "ELG2": "ELG2", "QSO": "QSO",
}

_COV_SCALE = 1e-4


def _sym(upper: List[List[float]]) -> np.ndarray:
    """Build a symmetric 4x4 from the upper triangle DESI prints."""
    C = np.array(upper, dtype=np.float64)
    C = np.triu(C) + np.triu(C, k=1).T
    return C * _COV_SCALE


# ---------------------------------------------------------------------------
# ShapeFit alone -- Eqs. (A.1)-(A.12)
# ---------------------------------------------------------------------------
_SF = {
    "BGS": (0.30, [7.788174, 3.053800, 0.377174, -0.031370], [
        [1314.664401, -142.669742, 109.427163, -217.509715],
        [0.0, 829.170940, -158.101737, -61.084426],
        [0.0, 0.0, 88.510877, -2.272457],
        [0.0, 0.0, 0.0, 279.702360]]),
    "LRG1": (0.51, [12.514437, 1.637266, 0.513635, 0.027840], [
        [541.309833, 48.425593, -4.652853, -37.707751],
        [0.0, 97.249820, -34.923265, -14.418597],
        [0.0, 0.0, 41.295470, 15.405508],
        [0.0, 0.0, 0.0, 48.918910]]),
    "LRG2": (0.71, [15.675560, 1.165867, 0.483623, 0.046650], [
        [762.457717, 39.781004, -1.896006, -62.812849],
        [0.0, 36.344098, -17.341350, -8.333900],
        [0.0, 0.0, 28.119682, 8.865225],
        [0.0, 0.0, 0.0, 47.624520]]),
    "LRG3": (0.92, [19.676985, 0.844996, 0.422164, -0.024690], [
        [847.499793, 26.038900, 3.257324, -37.016091],
        [0.0, 16.251088, -10.044074, -3.698790],
        [0.0, 0.0, 22.370314, 6.467510],
        [0.0, 0.0, 0.0, 34.883220]]),
    "ELG2": (1.32, [23.861806, 0.470709, 0.376715, 0.059960], [
        [2342.506886, 26.159601, 13.521001, -95.336060],
        [0.0, 10.309303, -6.654663, -5.183903],
        [0.0, 0.0, 13.997473, 9.109619],
        [0.0, 0.0, 0.0, 43.575710]]),
    "QSO": (1.49, [25.708520, 0.426508, 0.434858, 0.064550], [
        [3013.788566, -2.205101, 36.332110, -98.167826],
        [0.0, 5.845806, -6.747133, -1.913326],
        [0.0, 0.0, 19.785658, 5.357546],
        [0.0, 0.0, 0.0, 26.266260]]),
}


def datavector(tracer_bin: str, variant: str = "sf") -> Tuple[float, np.ndarray, np.ndarray]:
    """(z_eff, 4-vector, 4x4 covariance) for a tracer, in DESI's own basis."""
    if variant != "sf":
        raise ValueError(
            f"variant {variant!r} not transcribed. Only the ShapeFit-alone fits "
            f"(Eqs. A.1-A.12) are here; the ShapeFit+BAO fits (A.13-A.24) are "
            f"tighter and are the wrong comparison for a power-only forecast.")
    key = TRACER_MAP.get(tracer_bin, tracer_bin)
    if key not in _SF:
        raise KeyError(f"No DESI ShapeFit entry for {tracer_bin!r} "
                       f"(mapped to {key!r}); have {sorted(_SF)}")
    z, vec, upper = _SF[key]
    return float(z), np.array(vec, dtype=np.float64), _sym(upper)


def sigma_targets(tracer_bin: str, variant: str = "sf") -> Dict[str, float]:
    """DESI's constraints expressed as our emulator targets.

    sigma_qiso and sigma_qap are fractional (sigma divided by the measured
    central value -- see the module docstring on why that is a ~1%
    approximation). sigma_f_sigmar is returned as a FRACTION of f sigma_s8,
    since our f_sigmar and DESI's f sigma_s8 share a definition but are
    evaluated at slightly different z_eff. sigma_m is absolute.
    """
    _z, vec, C = datavector(tracer_bin, variant)
    sig = np.sqrt(np.diag(C))
    out = {
        "sigma_qiso": float(sig[0] / vec[0]),
        "sigma_qap": float(sig[1] / vec[1]),
        "sigma_f_sigmar_frac": float(sig[2] / vec[2]),
        "sigma_m": float(sig[3]),
    }
    names = TARGET_ORDER
    for i in range(4):
        for j in range(i + 1, 4):
            out[f"rho_{names[i]}_{names[j]}"] = float(
                C[i, j] / (sig[i] * sig[j]))
    return out
