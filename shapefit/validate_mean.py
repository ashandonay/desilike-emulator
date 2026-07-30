"""Independent checks on the ShapeFit MEAN pipeline (cosmology -> qiso, qap,
f_sigmar, m).

Motivation: every validation the mean pipeline has ever had is the fiducial
identity -- at the DESI fiducial cosmology the extractor returns qiso = qap = 1.
That is true BY CONSTRUCTION and proves nothing. Meanwhile the mean targets are
half of what bedcosmo consumes, and the two bugs this pipeline has actually
suffered (`w0_fde` never reaching cosmoprimo, and an Omega_m that dropped the
neutrino density) both lived in the cosmology MAPPING layer, which the fiducial
identity cannot see: at the fiducial there is nothing to map wrongly.

So the checks below target the mapping and the compression separately.

  mapping : build the cosmology our mapping produces, read its parameters BACK
            out of cosmoprimo, and compare against the sample we asked for.
            Catches a parameter that silently never arrives (the w0_fde class of
            bug) -- which a value comparison downstream can easily mask, since a
            wrong-but-plausible cosmology still returns finite numbers.
  ap      : qiso, qap against distances computed directly from cosmoprimo.
            FULLY independent of desilike's extractor -- qiso and qap are pure
            background quantities (D_M, D_H, r_d), so this is a real second
            implementation, not a paraphrase.
  shape   : f_sigmar and m against a direct transcription of the definitions in
            desilike's ShapeFitPowerSpectrumExtractor._set_base:
                s  = rs_drag / rs_drag_fid,      kp_eff = kp / s
                m  = dln P_nw(k) / dln k  at kp_eff          (n_varied=False)
                f_sigmar = f * sigma_r(r*s)
                           * exp(dm/(2a) * tanh(a * rs_drag_fid / r))
            This is a CROSS-IMPLEMENTATION check, not an independent derivation:
            it re-derives the same formulas from cosmoprimo primitives. What it
            can catch is our usage -- wrong z, wrong with_now, wrong fiducial,
            wrong mapping. It cannot catch an error in the formula itself.
  covar   : the covar pipeline's f_sigmar_fid / m_fid against the mean
            pipeline's f_sigmar / m at the same cosmology and z_eff. These are
            built by different code paths and should describe the same universe.
            NOTE neither m nor f_sigmar is expected to match in general, and
            that is a definitional difference rather than an error. The covar
            template sets fiducial=theta_cosmo, so s = r_d(theta)/r_d(fid) = 1
            and it evaluates the ShapeFit slope at kp = 0.03 exactly; the mean
            extractor keeps the DESI fiducial, so s != 1 and it evaluates at
            kp/s. The two therefore report the slope at DIFFERENT pivots. Only
            cosmologies that leave r_d alone (amplitude, n_s, late-time w0/wa)
            agree; anything moving r_d (omega_cdm, h, omega_b) does not. The
            check reports the numbers rather than asserting equality.

Usage (from shapefit/, emulator env):
    python validate_mean.py                       # all checks
    python validate_mean.py --check ap shape --tracers LRG2
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import numpy as np

import fourier_space
from fourier_space import sf_core
from util import ntracers

TRACERS_ALL = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")

FID = {"omega_cdm": 0.1200, "omega_b": 0.02237, "h": 0.6736,
       "ln10A_s": 3.036394, "n_s": 0.9649}

# Cosmology grid. Every point stays inside DEFAULT_PRIORS and satisfies the
# Omega_m domain guard; the w0/wa point respects high_z_matter_dom (w0+wa <= 0).
GRID = {
    "fid":      {},
    "lowOc":    {"omega_cdm": 0.080},
    "highOc":   {"omega_cdm": 0.170},
    "lowh":     {"h": 0.60},
    "highh":    {"h": 0.75},
    "lowA":     {"ln10A_s": 2.80},
    "highA":    {"ln10A_s": 3.25},
    "lowns":    {"n_s": 0.92},
    "highns":   {"n_s": 1.00},
    "lowob":    {"omega_b": 0.0205},
    "w0wa":     {"w0": -0.85, "wa": -0.35},
}

# ShapeFit conventions (desilike defaults; must match the extractor's).
_KP = sf_core._SHAPEFIT_KP     # 0.03
_A = sf_core._SHAPEFIT_A       # 0.6
_R = 8.0


def _sample(label: str, tracer: str) -> Dict[str, float]:
    s = dict(FID)
    s.update(GRID[label])
    s["N_tracers"] = float(ntracers(tracer, "dr1"))
    return s


def _cosmo_from_sample(sample: Dict[str, float]):
    """The cosmology the MEAN pipeline actually hands to desilike."""
    from desilike.theories.primordial_cosmology import get_cosmo
    theta = sf_core._to_mean_extractor_params(sample)
    # Cosmoprimo's own parameterisation, matching what the extractor's
    # calculator is driven with.
    return get_cosmo(("DESI", {
        "h": theta["h"], "Omega_m": theta["Omega_m"], "omega_b": theta["omega_b"],
        "logA": theta["logA"], "n_s": theta["n_s"],
        "w0_fld": theta["w0_fld"], "wa_fld": theta["wa_fld"],
    })), theta


# ---------------------------------------------------------------------------
def check_mapping(tracers: List[str]) -> None:
    print("\n=== 1. Cosmology mapping round trip ===")
    print("    build the cosmology our mapping produces, read the parameters")
    print("    back out, compare to what we asked for. A parameter that never")
    print("    arrives (w0_fde) shows up here and nowhere else.")
    bad = 0
    for label in GRID:
        sample = _sample(label, tracers[0])
        cosmo, theta = _cosmo_from_sample(sample)
        # cosmoprimo exposes present-day densities as Omega0_*, and no
        # omega_cdm at all, so the physical densities are reconstructed as
        # Omega0_x * h^2. That is the point of the check: omega_cdm is what our
        # mapping has to rebuild via Omega_m, and it is exactly where the
        # neutrino density was dropped before.
        h_got = float(cosmo.h)
        want = {
            "omega_b": float(sample["omega_b"]),
            "h": float(sample["h"]),
            "n_s": float(sample["n_s"]),
            "w0_fld": float(sample.get("w0", sf_core.PARAM_DEFAULTS["w0"])),
            "wa_fld": float(sample.get("wa", sf_core.PARAM_DEFAULTS["wa"])),
        }
        got = {
            "omega_b": float(cosmo.Omega0_b) * h_got**2,
            "h": h_got,
            "n_s": float(cosmo.n_s),
            "w0_fld": float(cosmo.w0_fld),
            "wa_fld": float(cosmo.wa_fld),
        }
        deltas = [abs(got[n] - want[n]) / max(abs(want[n]), 1e-12) for n in want]
        oc_want = float(sample["omega_cdm"])
        oc_got = float(cosmo.Omega0_cdm) * h_got**2
        oc_rel = abs(oc_got - oc_want) / oc_want
        worst = max(max(deltas), oc_rel)
        flag = "" if worst < 1e-6 else "   <-- MISMATCH"
        if worst >= 1e-6:
            bad += 1
        print(f"  {label:8s} worst rel. delta {worst:.2e}"
              f"  (omega_cdm {oc_got:.6f} vs {oc_want:.6f}){flag}")
        if worst >= 1e-6:
            for name in want:
                if abs(got[name] - want[name]) / max(abs(want[name]), 1e-12) >= 1e-6:
                    print(f"      {name}: asked {want[name]:.6g}, got {got[name]:.6g}")
    print(f"  {len(GRID) - bad}/{len(GRID)} cosmologies round-trip exactly.")


def check_ap(tracers: List[str]) -> None:
    """qiso, qap from cosmoprimo distances -- genuinely independent."""
    print("\n=== 2. qiso / qap vs distances computed directly ===")
    print("    qiso = [(D_H/r_d)(D_M/r_d)^2]^(1/3) / fid,  qap = (D_H/D_M)/fid")
    from desilike.theories.primordial_cosmology import get_cosmo
    fid_cosmo = get_cosmo(("DESI", {}))
    for tracer in tracers:
        z = _z_eff(tracer)
        print(f"\n  {tracer} (z_eff = {z:.4f})")
        print(f"    {'cosmology':>9s} {'qiso ext':>10s} {'qiso calc':>10s} "
              f"{'rel':>9s}   {'qap ext':>10s} {'qap calc':>10s} {'rel':>9s}")
        for label in GRID:
            sample = _sample(label, tracer)
            ext = _extractor_values(tracer, z, sample)
            cosmo, _ = _cosmo_from_sample(sample)

            def triple(c):
                # Conventions transcribed from desilike BAOExtractor._set_base:
                #   DH = (c/1e3) / (100 * efunc(z)),  DM = comoving_angular_distance
                # Both are already Mpc/h, as is rs_drag (99.08 at the fiducial),
                # so NO explicit h factors belong here. Multiplying by h cancels
                # in DM/rd but leaves a spurious 1/h in DH/rd, which corrupts
                # only the rows where h is varied -- how this was caught.
                dm = float(c.comoving_angular_distance(z))
                dh = 299792.458 / (100.0 * float(c.efunc(z)))
                rd = float(c.rs_drag)
                return dm / rd, dh / rd

            dm_rd, dh_rd = triple(cosmo)
            f_dm_rd, f_dh_rd = triple(fid_cosmo)
            # DV = DH^eta * DM^(1-eta) * z^(1/3) with eta = 1/3; the z factor
            # cancels in the ratio. qap is DH/DM, not DM/DH.
            qiso = ((dh_rd * dm_rd**2) / (f_dh_rd * f_dm_rd**2)) ** (1.0 / 3.0)
            qap = (dh_rd / dm_rd) / (f_dh_rd / f_dm_rd)
            r1 = abs(ext["qiso"] - qiso) / qiso
            r2 = abs(ext["qap"] - qap) / qap
            print(f"    {label:>9s} {ext['qiso']:>10.6f} {qiso:>10.6f} {r1:>9.2e}"
                  f"   {ext['qap']:>10.6f} {qap:>10.6f} {r2:>9.2e}")


def check_shape(tracers: List[str]) -> None:
    """f_sigmar and m against a transcription of the extractor's definitions."""
    print("\n=== 3. f_sigmar / m vs a direct transcription of the definitions ===")
    print("    cross-implementation (same formulas, cosmoprimo primitives):")
    print("    m = dlnP_nw/dlnk at kp/s ; f_sigmar = f*sigma_r(r s)*exp(dm/2a tanh(a rd_fid/r))")
    from desilike.theories.primordial_cosmology import get_cosmo
    fid_cosmo = get_cosmo(("DESI", {}))
    rd_fid = float(fid_cosmo.rs_drag)
    for tracer in tracers:
        z = _z_eff(tracer)
        m_fid_calc = _m_direct(fid_cosmo, z, rd_fid)
        print(f"\n  {tracer} (z_eff = {z:.4f}, m_fid_calc = {m_fid_calc:.6f})")
        print(f"    {'cosmology':>9s} {'m ext':>10s} {'m calc':>10s} {'rel':>9s}"
              f"   {'fsr ext':>10s} {'fsr calc':>10s} {'rel':>9s}")
        for label in GRID:
            sample = _sample(label, tracer)
            ext = _extractor_values(tracer, z, sample)
            cosmo, _ = _cosmo_from_sample(sample)
            m_calc = _m_direct(cosmo, z, rd_fid)
            fsr_calc = _f_sigmar_direct(cosmo, z, rd_fid, m_calc, m_fid_calc)
            r1 = abs(ext["m"] - m_calc) / max(abs(m_calc), 1e-12)
            r2 = abs(ext["f_sigmar"] - fsr_calc) / max(abs(fsr_calc), 1e-12)
            print(f"    {label:>9s} {ext['m']:>10.6f} {m_calc:>10.6f} {r1:>9.2e}"
                  f"   {ext['f_sigmar']:>10.6f} {fsr_calc:>10.6f} {r2:>9.2e}")


def _pknow(cosmo, z):
    """Linear no-wiggle P(k) for delta_cb at z, wallish2018 -- the same
    de-wiggling engine the template and extractor are pinned to."""
    fo = cosmo.get_fourier()
    pk = fo.pk_interpolator(of="delta_cb").to_1d(z=z)
    from cosmoprimo import PowerSpectrumBAOFilter
    return PowerSpectrumBAOFilter(pk, engine="wallish2018").smooth_pk_interpolator()


def _m_direct(cosmo, z, rd_fid):
    s = float(cosmo.rs_drag) / rd_fid
    kp = _KP / s
    dk = 1e-2
    k = kp * np.array([1.0 - dk, 1.0 + dk])
    pnw = _pknow(cosmo, z)(k)
    return float(np.diff(np.log(pnw))[0] / np.diff(np.log(k))[0])


def _f_sigmar_direct(cosmo, z, rd_fid, m, m_fid):
    s = float(cosmo.rs_drag) / rd_fid
    fo = cosmo.get_fourier()
    f = float(fo.sigma8_z(z, of="theta_cb") / fo.sigma8_z(z, of="delta_cb"))
    sig_r = float(_pknow(cosmo, z).sigma_r(_R * s))
    dm = m - m_fid
    return f * sig_r * np.exp(dm / (2.0 * _A) * np.tanh(_A * rd_fid / _R))


_Z_CACHE: Dict[str, float] = {}


def _z_eff(tracer: str) -> float:
    if tracer not in _Z_CACHE:
        info = sf_core.build_shapefit_likelihood(
            N_tracers=float(ntracers(tracer, "dr1")),
            theta_cosmo=sf_core._to_shapefit_cosmo_params(
                {**FID, "N_tracers": ntracers(tracer, "dr1")}),
            tracer_bin=tracer)
        _Z_CACHE[tracer] = float(info["z_eff"])
    return _Z_CACHE[tracer]


def _extractor_values(tracer: str, z: float, sample: Dict[str, float]) -> Dict:
    """Exactly what the mean worker produces, via the same code path."""
    extractor = fourier_space._get_mean_extractor(tracer, z)
    theta = sf_core._to_mean_extractor_params(sample)
    extractor(**theta)
    extractor.get()
    return {"qiso": float(extractor.qiso), "qap": float(extractor.qap),
            "f_sigmar": float(extractor.f_sigmar), "m": float(extractor.m)}


def check_covar(tracers: List[str]) -> None:
    print("\n=== 4. covar pipeline fids vs mean pipeline, same cosmology ===")
    print("    NOT expected to be equal: the covar template sets fiducial=theta")
    print("    so s=1 and it evaluates the slope at kp=0.03, while the mean")
    print("    extractor keeps the DESI fiducial and evaluates at kp/s. Rows that")
    print("    leave r_d alone (A_s, n_s, w0/wa) agree; rows that move it do not.")
    for tracer in tracers:
        z = _z_eff(tracer)
        print(f"\n  {tracer}")
        print(f"    {'cosmology':>9s} {'m covar':>10s} {'m mean':>10s} {'rel':>9s}"
              f"   {'fsr covar':>10s} {'fsr mean':>10s} {'ratio':>8s}")
        for label in GRID:
            sample = _sample(label, tracer)
            info = sf_core.build_shapefit_likelihood(
                N_tracers=float(sample["N_tracers"]),
                theta_cosmo=sf_core._to_shapefit_cosmo_params(sample),
                tracer_bin=tracer, z_eff=z)
            ext = _extractor_values(tracer, z, sample)
            rel_m = abs(info["m_fid"] - ext["m"]) / max(abs(ext["m"]), 1e-12)
            print(f"    {label:>9s} {info['m_fid']:>10.6f} {ext['m']:>10.6f} "
                  f"{rel_m:>9.2e}   {info['f_sigmar_fid']:>10.6f} "
                  f"{ext['f_sigmar']:>10.6f} "
                  f"{info['f_sigmar_fid'] / ext['f_sigmar']:>8.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", nargs="*",
                   choices=["mapping", "ap", "shape", "covar", "all"],
                   default=["all"])
    p.add_argument("--tracers", nargs="*", default=["LRG2"],
                   help=f"subset of {list(TRACERS_ALL)}; default LRG2")
    args = p.parse_args()
    checks = set(args.check)
    do = lambda n: "all" in checks or n in checks  # noqa: E731

    if do("mapping"):
        check_mapping(args.tracers)
    if do("ap"):
        check_ap(args.tracers)
    if do("shape"):
        check_shape(args.tracers)
    if do("covar"):
        check_covar(args.tracers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
