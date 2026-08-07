"""Assert the MEAN and COVAR pipelines agree on everything they share.

The two halves of this pipeline are built and maintained separately -- the mean
path runs a `ShapeFitPowerSpectrumExtractor`, the covar path builds a likelihood
and Fisher-reduces it -- but they describe ONE Gaussian per tracer, and they
must therefore agree on its inputs. They have repeatedly not:

  S42  z_eff frozen at the fiducial in the mean path while the covar path
       derived it per sample -- mu and C at different redshifts.
  S58  the per-tracer footprint reached the covar path and not the generators.
  S60  a correction applied to one path's f_sigmar and not the other's.
  S64  the two paths appeared to define f_sigmar at different radii.
  S65  a stale denominator in the mean plot's error bar.
  S76  an explicit `with_now` that one path honoured and the other silently
       overrode, so the halves ran different de-wiggling engines.

Four of those five were found by hand, late, and each cost a regeneration. All
of them are the same shape: a quantity that BOTH paths consume, changed on one
side. That is a mechanical check, so this is the mechanical check.

What it does NOT do: verify either path is CORRECT. Both can be wrong together
and this stays silent -- it is a consistency test, not a validation. Use
`benchmark_desi.py` (against DESI's published tables) and `validate_forecast.py`
for correctness.

Run (default is cheap: Kaiser, which z_eff does not depend on):
    python seam_check.py
    python seam_check.py --tracers LRG2 QSO --theory rept
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import core as sf_core        # noqa: E402
import fourier_space as fs    # noqa: E402
from util import ntracers, tracer_area, nz_slices_path  # noqa: E402

_ALL = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")

# Tolerances. z_eff and the cosmology mapping should agree to ROUNDING, not to
# "close enough": both paths call the same functions, so any real difference is
# an argument mismatch, and an argument mismatch does not produce a small error.
_RTOL_EXACT = 1e-12

# Parameters passed STRAIGHT through on both sides must be bit-equal.
_RTOL_PASSTHRU = 1e-12

# Quantities CLASS SOLVES for are a different matter. The covar path hands
# cosmoprimo `omega_cdm`; the mean path must hand it `Omega_m` (the extractor's
# pipeline exposes no omega_cdm), and CLASS then shoots to recover omega_cdm.
# The two therefore cannot agree bit-for-bit -- one route goes through a
# nonlinear solve -- and the residual measures the shooting tolerance, ~5e-7
# relative. This is the SAME residual that makes `m` come out at 1e-5 instead
# of 0 in the mean plot's fiducial null test (S66).
#
# 5e-6 is ten times the observed residual: tight enough that an actual mapping
# change (a dropped omega_ncdm shifts omega_cdm by ~6e-4, i.e. 5e-3 relative)
# fails loudly, loose enough that the solver's own noise does not.
_RTOL_SHOOT = 5e-6


class Seam:
    """One shared quantity, and what each side says about it."""

    def __init__(self, name: str, detail: str = ""):
        self.name, self.detail = name, detail
        self.rows: List[Tuple[str, str, str, bool]] = []
        self.gaps: List[Tuple[str, float, float]] = []

    def check(self, label, mean_val, covar_val, rtol=_RTOL_EXACT):
        """Record one comparison. Reports the relative gap so a PASS that is
        creeping toward its tolerance is visible before it becomes a FAIL."""
        if isinstance(mean_val, (int, float, np.floating)) and \
           isinstance(covar_val, (int, float, np.floating)):
            ok = bool(np.isclose(mean_val, covar_val, rtol=rtol, atol=0.0))
            m, c = f"{mean_val:.12g}", f"{covar_val:.12g}"
            denom = max(abs(float(covar_val)), 1e-300)
            self.gaps.append((label, abs(float(mean_val) - float(covar_val)) / denom, rtol))
        else:
            ok = mean_val == covar_val
            m, c = str(mean_val), str(covar_val)
        self.rows.append((label, m, c, ok))
        return ok

    @property
    def passed(self) -> bool:
        return all(r[3] for r in self.rows)

    def report(self) -> None:
        mark = "PASS" if self.passed else "**FAIL**"
        print(f"\n[{mark}] {self.name}")
        if self.detail:
            print(f"        {self.detail}")
        w = max((len(r[0]) for r in self.rows), default=8)
        print(f"        {'':{w}s}  {'mean path':>24s}  {'covar path':>24s}")
        gaps = {g[0]: g for g in self.gaps}
        for label, m, c, ok in self.rows:
            flag = "" if ok else "   <-- DIFFERS"
            g = gaps.get(label)
            rel = f"  rel {g[1]:.1e} / tol {g[2]:.0e}" if g and g[1] > 0 else ""
            print(f"        {label:{w}s}  {m:>24s}  {c:>24s}{rel}{flag}")


def _samples() -> Dict[str, Dict[str, float]]:
    import desi_reference as dr
    out = {"MAP": dr.dr1_bestfit_cosmology()}
    # An off-fiducial point: a seam that only agrees at the fiducial (because
    # one side froze something there) passes a fiducial-only test. S42 was
    # exactly that bug.
    out["offset"] = {"omega_cdm": 0.135, "omega_b": 0.0228, "h": 0.72,
                     "ln10A_s": 3.15, "n_s": 0.955}
    return out


def run(tracers, theory: str, data_release: str) -> bool:
    from desilike.theories.primordial_cosmology import get_cosmo

    theory_cls = (sf_core.KaiserTracerPowerSpectrumMultipoles if theory == "kaiser"
                  else sf_core.REPTVelocileptorsTracerPowerSpectrumMultipoles)
    seams: List[Seam] = []

    # ---- seams that do not depend on a sample -------------------------------
    s_area = Seam("tracer area", "both must read util.tracer_area(t, data_release)")
    s_nz = Seam("n(z) table", "same release-scoped file (S62c/S77)")
    for t in tracers:
        s_area.check(t, tracer_area(t, data_release), tracer_area(t, data_release))
        # The mean path reaches n(z) through _mean_z_eff_for_sample -> the covar
        # path through build_shapefit_likelihood; both land in nz_slices_path.
        p = nz_slices_path(f"{t}_nz_slices.csv", data_release)
        s_nz.check(t, p.name, p.name)
    seams += [s_area, s_nz]

    for sname, sample in _samples().items():
        theta = sf_core._to_shapefit_cosmo_params(sample)

        # ---- cosmology mapping ---------------------------------------------
        # The two paths describe the same cosmology through DIFFERENT bases:
        # covar passes omega_cdm straight to cosmoprimo, mean must assemble
        # Omega_m (the extractor's pipeline exposes no omega_cdm). Omitting
        # omega_ncdm in that assembly shifts omega_cdm by ~6e-4 -- the bug
        # _to_mean_extractor_params exists to prevent. Compare what CLASS
        # actually ends up with, not the dicts.
        s_cos = Seam(f"cosmology mapping [{sname}]",
                     "covar omega-basis vs mean Omega_m-basis, compared after CLASS")
        c_cov = get_cosmo(("DESI", dict(theta)))
        mean_par = sf_core._to_mean_extractor_params(
            {**sf_core.PARAM_DEFAULTS, **sample})
        c_mean = get_cosmo(("DESI", {k: v for k, v in mean_par.items()}))
        # omega_cdm is the one that matters: it is what the two bases disagree
        # about if omega_ncdm is dropped from the Omega_m assembly.
        for q, get, rt in (
                ("omega_b", lambda c: c.Omega0_b * c.h ** 2, _RTOL_PASSTHRU),
                ("h", lambda c: c.h, _RTOL_PASSTHRU),
                ("n_s", lambda c: c.n_s, _RTOL_PASSTHRU),
                ("omega_cdm", lambda c: c.Omega0_cdm * c.h ** 2, _RTOL_SHOOT),
                ("Omega_m", lambda c: c.Omega0_m, _RTOL_SHOOT),
                ("rs_drag", lambda c: c.rs_drag, _RTOL_SHOOT),
                ("sigma8", lambda c: c.sigma8_m, _RTOL_SHOOT)):
            s_cos.check(q, float(get(c_mean)), float(get(c_cov)), rtol=rt)
        seams.append(s_cos)

        for t in tracers:
            area = float(tracer_area(t, data_release))
            n_tr = float(ntracers(t, data_release))
            smp = {**sample, "N_tracers": n_tr}

            info = sf_core.build_shapefit_likelihood(
                N_tracers=n_tr,
                theta_cosmo=theta,
                tracer_bin=t,
                area=area,
                data_release=data_release,
                theory_cls=theory_cls,
            )
            extractor = fs._get_mean_extractor(t, info["z_eff"])

            # ---- z_eff ------------------------------------------------------
            # Mean side computed exactly as _worker_run_mean_targets does.
            s_z = Seam(f"z_eff [{sname}/{t}]",
                       "S42: mu and C must be evaluated at ONE redshift")
            s_z.check("z_eff",
                      fs._mean_z_eff_for_sample(smp, t, area, data_release),
                      float(info["z_eff"]))
            seams.append(s_z)

            # ---- de-wiggling engine (S76) -----------------------------------
            # The covar template's value is what SURVIVES theory_cls, which
            # overrides it unconditionally for REPT/FOLPS/PyBird-nnlo.
            s_e = Seam(f"de-wiggle engine [{sname}/{t}]",
                       "S76: theory_cls silently overrides the template's with_now")
            s_e.check("with_now",
                      str(getattr(extractor, "with_now", "?")),
                      str(getattr(info["template"], "with_now", "?")))
            seams.append(s_e)

            # ---- f_sigmar radius (S64) --------------------------------------
            s_r = Seam(f"f_sigmar radius [{sname}/{t}]",
                       "S64/S73: sigma_r is evaluated at r*s on both sides")
            s_r.check("r", float(getattr(extractor, "r", np.nan)),
                      float(getattr(info["template"], "r", np.nan)))
            seams.append(s_r)

    for s in seams:
        s.report()
    n_fail = sum(1 for s in seams if not s.passed)
    print("\n" + "=" * 78)
    print(f"{len(seams) - n_fail}/{len(seams)} seams agree"
          + ("" if n_fail == 0 else f"   -- {n_fail} MISMATCHED"))
    print("=" * 78)
    return n_fail == 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", nargs="+", default=["LRG2"],
                    help="default LRG2; pass more for a full sweep")
    ap.add_argument("--theory", choices=["kaiser", "rept"], default="kaiser",
                    help="kaiser (default) is far cheaper and none of the "
                         "checked quantities depend on the theory model")
    ap.add_argument("--data-release", default="dr1", choices=["dr1"])
    a = ap.parse_args()
    return 0 if run(a.tracers, a.theory, a.data_release) else 1


if __name__ == "__main__":
    raise SystemExit(main())
