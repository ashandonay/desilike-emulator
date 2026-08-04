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
    m        <-> m + n                          -> compare SIGMA ONLY, see below

Ratios of a quantity to its fiducial leave correlations unchanged, so the six
rho_* targets compare directly with no conversion at all.

On m and its zero point. DESI never states "m_fid = 0" -- the claim rests on
Eq. (4.9) being a definition: m multiplies P^fid_lin, so m = 0 recovers the
template exactly. m is a deviation FROM WHATEVER TEMPLATE WAS USED. For the DR1
data fits transcribed here the template IS the fiducial cosmology, so these
published values are directly comparable to our deviation, and they sit near
zero (-0.031 to +0.065) against a Table 4 prior of U[-0.8, 0.8].

⚠ m_fid is NOT always zero in DESI's own usage. Figure 6 plots `m - m_fid` for
Abacus-2 mock fits, where m_fid is the m expected for the mock's known true
cosmology -- generally non-zero. Anything comparing against mock-based results
rather than Appendix A has to carry that reference explicitly.

On m vs (m + n). DESI varies m with n FIXED, exactly as we do (our `dn` is
fixed): section 4.9, "we will only consider varying one of the two shape
variables: we will vary m keeping n fixed. Later, in the interpretation step m
can be seen as if it were m + n." So the (m+n) label is an interpretive relabel
of a single varied parameter, and sigma(m+n) = sigma(m) -- the ERROR comparison
is exact and the parameterisations match.

The CENTRAL VALUES now compare directly. Since CHANGELOG S35 the mean emulator
emits DESI's convention -- the Eq. (4.9) deviation, verified as -4.5e-05 at the
DESI fiducial cosmology -- so no offset conversion is needed by bedcosmo or
anything else. (Before S35 our target was the ABSOLUTE slope, -0.5775 at the
fiducial, and every comparison had to subtract m_fid. Note desilike still calls
the absolute slope `m`; its `dm` is what equals DESI's m, and the mean worker
reads `extractor.dm` into a target named `m`.)

The denominator is the FIDUCIAL value from DESI's own Table 11 (Appendix C),
not the measured central value. An earlier version divided by the
measurement on the assumption that DR1 sits within ~1% of the fiducial. It does
not: measured/fiducial runs to 0.948 for LRG2 D_V/r_d and 0.923 for LRG1
D_H/r_d (the latter being DR1's well-known low point). Those are real data
deviations, so using the measurement inflated DESI's sigma(qiso) and sigma(qap)
by up to 5% and correspondingly deflated every ratio computed against them.

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
* The 0.8-1.1 bin is `LRG3` (LRG-only), matching DESI's full-shape sample.
  ELG1 is excluded there because it failed the pre-unblinding fibre-collision
  tests for growth-rate measurements, though it stays in the BAO analysis
  (DESI 2024 V Sec 2). The BAO-side combined bin is `LRG3_ELG1`, a separate
  tracers.yaml block. Before CHANGELOG S31 this module compared our combined
  bin against DESI's LRG-only one -- a factor 2.18 in N.
* The BGS entry carries DESI's own warning that alpha_AP is not well
  constrained there and D_H/D_M is "highly affected by the flat prior between
  0.8 and 1.2". Its sigma(qap) is a prior width, not a measurement.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

_FID_CACHE: Dict[float, Tuple[float, float]] = {}

# ---------------------------------------------------------------------------
# Table 11 (Appendix C): DESI's OWN fiducial values, at the Table 1 z_eff.
#   label: (z_eff, DM/rd, DH/rd, DV/rd, DH/DM, sigma_s8, f sigma_s8)
# The fiducial cosmology is AbacusSummit c000 "Planck LCDM" (Table 6, row 1):
#   omega_b 0.02237, omega_cdm 0.1200, h 0.6736, 1e9 A_s 2.0830, n_s 0.9649,
#   N_ur 2.0328 (one 0.06 eV neutrino), w0 -1, wa 0.
# Section 4.7 item 10: this same cosmology is BOTH the grid cosmology (z ->
# comoving distance) and the ShapeFit template cosmology. It is exactly
# cosmoprimo's "DESI", which is what our pipeline uses -- fiducial_dv_dhdm
# below reproduces this table to <=0.13%.
# rd = 99.0792 Mpc. (The paper's caption says "Mpc/h"; that is a slip -- in
# Mpc/h it would be 66.74. Our cosmoprimo value is 99.0844 Mpc, 0.005% away.)
# ---------------------------------------------------------------------------
_T11 = {
    "BGS":  (0.295,  8.2908, 25.8506,  8.0663, 3.1180, 0.6936, 0.4723),
    "LRG1": (0.510, 13.4928, 22.7462, 12.8269, 1.6858, 0.6210, 0.4733),
    "LRG2": (0.706, 17.6976, 20.1727, 16.4597, 1.1399, 0.5638, 0.4608),
    "LRG3": (0.919, 21.7238, 17.7321, 19.7356, 0.8162, 0.5108, 0.4398),
    "ELG2": (1.317, 28.0276, 14.0956, 24.4318, 0.5029, 0.4320, 0.3944),
    "QSO":  (1.491, 30.3606, 12.8359, 26.0292, 0.4228, 0.4042, 0.3750),
}


def published_fiducial(tracer_bin: str) -> Dict[str, float]:
    """DESI's Table 11 fiducial row for a tracer. The denominators of record."""
    key = TRACER_MAP.get(tracer_bin, tracer_bin)
    if key not in _T11:
        raise KeyError(f"No Table 11 entry for {tracer_bin!r} (mapped {key!r})")
    z, dm, dh, dv, dhdm, s8, fs8 = _T11[key]
    return {"z_eff": z, "DM_over_rd": dm, "DH_over_rd": dh, "DV_over_rd": dv,
            "DH_over_DM": dhdm, "sigma_s8": s8, "f_sigma_s8": fs8}


def fiducial_dv_dhdm(z: float) -> Tuple[float, float]:
    """(D_V/r_d, D_H/D_M) recomputed from cosmoprimo at the DESI fiducial.

    Cross-check on `published_fiducial`, which is what sigma_targets actually
    uses -- there is no reason to recompute a number DESI printed. Agrees with
    Table 11 to <=0.13% on every tracer.

    Conventions match desilike's BAOExtractor._set_base:
        D_H = (c/1e3) / (100 * efunc(z)),  D_M = comoving_angular_distance(z),
        D_V = D_H^(1/3) * D_M^(2/3) * z^(1/3),  all already Mpc/h.

    Feed it the Table 1 z_eff, not a rounded one: at z=0.30 instead of 0.295
    BGS moves +1.4% in D_V/r_d and -1.7% in D_H/D_M, which is a bigger error
    than anything it is being used to measure.
    """
    key = round(float(z), 6)
    if key not in _FID_CACHE:
        from desilike.theories.primordial_cosmology import get_cosmo
        c = get_cosmo(("DESI", {}))
        dm = float(c.comoving_angular_distance(z))
        dh = 299792.458 / (100.0 * float(c.efunc(z)))
        rd = float(c.rs_drag)
        dv = dh ** (1.0 / 3.0) * dm ** (2.0 / 3.0) * float(z) ** (1.0 / 3.0)
        _FID_CACHE[key] = (dv / rd, dh / dm)
    return _FID_CACHE[key]

def dv_dhdm_at(z: float, params: Dict[str, float] | None = None) -> Tuple[float, float]:
    """(D_V/r_d, D_H/D_M) at an arbitrary cosmology, same conventions as above.

    `params` is an omega-basis sample ({omega_cdm, omega_b, h, ln10A_s, n_s});
    None means the DESI fiducial, in which case this is `fiducial_dv_dhdm`.
    """
    if not params:
        return fiducial_dv_dhdm(z)
    from desilike.theories.primordial_cosmology import get_cosmo
    theta = {"omega_cdm": float(params["omega_cdm"]),
             "omega_b": float(params["omega_b"]),
             "h": float(params["h"]),
             "logA": float(params["ln10A_s"]),
             "n_s": float(params["n_s"])}
    key = (round(float(z), 6),) + tuple(round(v, 10) for v in theta.values())
    if key not in _FID_CACHE:
        c = get_cosmo(("DESI", theta))
        dm = float(c.comoving_angular_distance(z))
        dh = 299792.458 / (100.0 * float(c.efunc(z)))
        rd = float(c.rs_drag)
        dv = dh ** (1.0 / 3.0) * dm ** (2.0 / 3.0) * float(z) ** (1.0 / 3.0)
        _FID_CACHE[key] = (dv / rd, dh / dm)
    return _FID_CACHE[key]


# ---------------------------------------------------------------------------
# DESI DR1's OWN best-fit LCDM cosmology.
#
# DESI 2024 VII (arXiv:2411.12022) Eq. (3.1), dataset DESI (FS+BAO)+BBN+ns10:
#     Omega_m = 0.2962 +- 0.0095
#     sigma8  = 0.842  +- 0.034
#     H0      = 68.56  +- 0.75  km/s/Mpc
#
# omega_b and n_s are NOT measured by DESI full-shape; they are priors, so the
# prior centres are used: BBN omega_b = 0.02218, ns10 n_s = 0.9649 (2024 VII
# Table 1). Those are the same numbers as core.DEFAULT_PRIORS, by construction
# -- our priors were taken from that table.
#
# ⚠ THIS IS NOT A JOINT MAP. Eq. (3.1) reports MARGINALISED means, one
# parameter at a time. Assembling a vector from them lands on the posterior's
# centre-of-mass, which coincides with the best-fit POINT only for a Gaussian
# posterior. DESI does not publish a full LCDM chain here, so this is the best
# available stand-in, not the true maximum-likelihood cosmology.
#
# ⚠ NOT INDEPENDENT DATA. Eq. (3.1) was inferred FROM the same compressed
# measurements this module transcribes (plus BAO). Comparing our prediction at
# this cosmology against those measurements is a CLOSURE test -- does the
# cosmology DESI extracted reproduce the per-tracer numbers it was extracted
# from -- not an independent validation.
#
# ⚠ FS+BAO, not FS-alone. Our compressed comparison is ShapeFit-ALONE. DESI do
# not quote an FS-alone LCDM constraint in this equation, so the cosmology
# carries BAO information the compressed vectors do not.
# ---------------------------------------------------------------------------
DR1_BESTFIT_INPUTS = {"Omega_m": 0.2962, "sigma8": 0.842, "H0": 68.56,
                      "omega_b": 0.02218, "n_s": 0.9649,
                      "source": "DESI 2024 VII (2411.12022) Eq. (3.1), "
                                "FS+BAO+BBN+ns10"}
_DR1_BESTFIT_CACHE: Dict[str, float] = {}


def dr1_bestfit_cosmology() -> Dict[str, float]:
    """DR1's best-fit LCDM as an omega-basis sample. See the caveats above.

    ln10A_s is solved for, not published: DESI quote sigma8, and the mean
    pipeline takes A_s. Linear sigma8 scales exactly as sqrt(A_s), so one
    Boltzmann call fixes the normalisation and a second verifies it (agreement
    is exact to the printed digits).

    omega_cdm is assembled as Omega_m h^2 - omega_b - omega_ncdm, with
    omega_ncdm from the DESI fiducial's single 0.06 eV neutrino -- the same
    convention core._to_mean_extractor_params uses, so the mean and covar
    pipelines stay on one definition (CHANGELOG S66).
    """
    if not _DR1_BESTFIT_CACHE:
        from desilike.theories.primordial_cosmology import get_cosmo
        d = DR1_BESTFIT_INPUTS
        h = float(d["H0"]) / 100.0
        fid = get_cosmo(("DESI", {}))
        omega_ncdm = float(np.sum(np.atleast_1d(fid.Omega_ncdm(0.0)))) * fid.h ** 2
        omega_cdm = float(d["Omega_m"]) * h ** 2 - float(d["omega_b"]) - omega_ncdm
        base = {"omega_cdm": omega_cdm, "omega_b": float(d["omega_b"]),
                "h": h, "n_s": float(d["n_s"])}
        s8_ref = float(get_cosmo(("DESI", {**base, "logA": 3.0}))
                       .get_fourier().sigma8_z(0.0, of="delta_m"))
        ln10A_s = 3.0 + 2.0 * np.log(float(d["sigma8"]) / s8_ref)
        _DR1_BESTFIT_CACHE.update({**base, "ln10A_s": float(ln10A_s)})
    return dict(_DR1_BESTFIT_CACHE)


# Order of the DESI 4-vector, and the emulator target each entry maps to.
DESI_ORDER = ("DV_over_rd", "DH_over_DM", "f_sigma_s8", "m_plus_n")
TARGET_ORDER = ("qiso", "qap", "f_sigmar", "m")

# Our tracer-bin names -> DESI's appendix labels.
TRACER_MAP = {
    "BGS": "BGS", "LRG1": "LRG1", "LRG2": "LRG2",
    "LRG3": "LRG3", "ELG2": "ELG2", "QSO": "QSO",
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
    "BGS": (0.295, [7.788174, 3.053800, 0.377174, -0.031370], [
        [1314.664401, -142.669742, 109.427163, -217.509715],
        [0.0, 829.170940, -158.101737, -61.084426],
        [0.0, 0.0, 88.510877, -2.272457],
        [0.0, 0.0, 0.0, 279.702360]]),
    "LRG1": (0.510, [12.514437, 1.637266, 0.513635, 0.027840], [
        [541.309833, 48.425593, -4.652853, -37.707751],
        [0.0, 97.249820, -34.923265, -14.418597],
        [0.0, 0.0, 41.295470, 15.405508],
        [0.0, 0.0, 0.0, 48.918910]]),
    "LRG2": (0.706, [15.675560, 1.165867, 0.483623, 0.046650], [
        [762.457717, 39.781004, -1.896006, -62.812849],
        [0.0, 36.344098, -17.341350, -8.333900],
        [0.0, 0.0, 28.119682, 8.865225],
        [0.0, 0.0, 0.0, 47.624520]]),
    "LRG3": (0.919, [19.676985, 0.844996, 0.422164, -0.024690], [
        [847.499793, 26.038900, 3.257324, -37.016091],
        [0.0, 16.251088, -10.044074, -3.698790],
        [0.0, 0.0, 22.370314, 6.467510],
        [0.0, 0.0, 0.0, 34.883220]]),
    "ELG2": (1.317, [23.861806, 0.470709, 0.376715, 0.059960], [
        [2342.506886, 26.159601, 13.521001, -95.336060],
        [0.0, 10.309303, -6.654663, -5.183903],
        [0.0, 0.0, 13.997473, 9.109619],
        [0.0, 0.0, 0.0, 43.575710]]),
    "QSO": (1.491, [25.708520, 0.426508, 0.434858, 0.064550], [
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

    sigma_qiso and sigma_qap are divided by the FIDUCIAL D_V/r_d and D_H/D_M
    as published in Table 11 (see published_fiducial), which is what the
    definitions of qiso and qap require. sigma_f_sigmar is returned as a
    FRACTION of the FIDUCIAL f sigma_s8 (Table 11) for the same reason -- our
    f_sigmar and DESI's f sigma_s8 share a definition but are evaluated at
    slightly different z_eff, and normalising each side by its own fiducial
    absorbs that offset without importing DR1's data fluctuation. sigma_m is
    absolute.
    """
    z, vec, C = datavector(tracer_bin, variant)
    sig = np.sqrt(np.diag(C))
    fid = published_fiducial(tracer_bin)
    fid_dv, fid_dhdm = fid["DV_over_rd"], fid["DH_over_DM"]
    out = {
        "sigma_qiso": float(sig[0] / fid_dv),
        "sigma_qap": float(sig[1] / fid_dhdm),
        # FIDUCIAL f sigma_s8 (Table 11), not the measured central value.
        # S17 moved qiso and qap off the measurement and left this one on it,
        # so the fractional f_sigmar comparison stayed skewed by exactly the
        # error S17 diagnosed -- and by more than it was for q: measured/fiducial
        # runs 0.799 (BGS) to 1.160 (QSO), against 0.94-1.04 there.
        #
        # Our side divides by OUR f_sigmar_fid (compare_to_desi.py), so the two
        # denominators have to be the same KIND of quantity or the ratio carries
        # a data fluctuation. They are also numerically the same quantity: our
        # mean pipeline returns f_sigmar = 0.460725 at the LRG2 fiducial against
        # Table 11's 0.4608, agreeing to 0.02%. Dividing each side by its own
        # fiducial therefore compares sigma against sigma, and additionally
        # absorbs the small z_eff difference in a controlled way -- which
        # dividing by the measurement does not, since that mixes the z offset
        # with DR1's fluctuation.
        "sigma_f_sigmar_frac": float(sig[2] / fid["f_sigma_s8"]),
        "sigma_m": float(sig[3]),
    }
    names = TARGET_ORDER
    for i in range(4):
        for j in range(i + 1, 4):
            out[f"rho_{names[i]}_{names[j]}"] = float(
                C[i, j] / (sig[i] * sig[j]))
    return out
