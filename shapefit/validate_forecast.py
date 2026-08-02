"""Physical sanity checks for the ShapeFit forecast pipeline.

Checks (select with --check, default all):

  fiducial : per-tracer sigma/rho table at the DESI fiducial cosmology and the
             DR1 passed counts (via util.ntracers — never hardcoded, bao
             CHANGELOG §33n). Prints fractional errors so they can be compared
             qualitatively against DESI DR1 full-shape per-tracer precision
             (sigma(fsigma8)/fsigma8 ~ 4-10%; direct-fit velocileptors, so
             order-of-magnitude only — do NOT tune to match).
  scaling  : sigma vs N_tracers over each tracer's emulator box — must be
             monotone decreasing and saturate toward high N (sample-variance
             regime), ~1/sqrt(N) in the shot-noise regime. Writes
             scaling_vs_ntracers_shapefit.png.
  damping  : sensitivity of the fiducial sigmas to floating vs fixing the
             Gaussian damping scales (float_sigma_damp), documenting the
             Kaiser FoG-in-quadrature approximation.
  kmax     : sigmas vs the fit kmax, plus the k where the Kaiser quadrupole
             crosses zero. compare_to_desi.py (CHANGELOG S4) found our P2 going
             negative above k ~ 0.16 for LRG2 where DESI measures positive
             power, so the top of the band contributes derivatives from a model
             that has broken down. This quantifies how much information is
             actually being drawn from there: if sigma(kmax=0.10) is close to
             sigma(kmax=0.20), the broken region carries little weight and the
             labels are safer than the P2 comparison alone suggests.

Usage (from shapefit/, emulator env):
    python validate_forecast.py --check fiducial --tracers LRG2 QSO
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fourier_space
from fourier_space import sf_core
from util import (TRACER_TYPE_CHOICES, get_tracer_config, ntracers,
                  ntracers_range, plots_dir)

TRACERS_ALL = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")

FID_SAMPLE = {
    "omega_cdm": 0.1200,
    "omega_b": 0.02237,
    "h": 0.6736,
    "ln10A_s": 3.036394,
    "n_s": 0.9649,
}


def _fid_sample_for(tracer: str, n_factor: float = 1.0) -> dict:
    sample = dict(FID_SAMPLE)
    sample["N_tracers"] = float(ntracers(tracer, "dr1")) * n_factor
    return sample


def check_fiducial(tracers) -> None:
    print("\n=== Fiducial sigma table (DESI fiducial cosmology, DR1 passed counts) ===")
    header = (f"{'tracer':>10s} {'z_eff':>6s} {'N_dr1':>9s} "
              f"{'s(qiso)':>8s} {'s(qap)':>8s} {'s(fsr)':>8s} {'s(m)':>8s} "
              f"{'s(fsr)/fsr':>10s}")
    print(header)
    rows = {}
    for tracer in tracers:
        out = fourier_space.run_fisher(_fid_sample_for(tracer), tracer_bin=tracer)
        rows[tracer] = out
        frac_fsr = out["sigma_f_sigmar"] / out["f_sigmar_fid"]
        print(f"{tracer:>10s} {out['z_eff']:>6.3f} "
              f"{ntracers(tracer, 'dr1'):>9.0f} "
              f"{out['sigma_qiso']:>8.4f} {out['sigma_qap']:>8.4f} "
              f"{out['sigma_f_sigmar']:>8.4f} {out['sigma_m']:>8.4f} "
              f"{frac_fsr:>9.1%}")
    print("\nCorrelations:")
    rho_names = [t for t in fourier_space.TARGET_NAMES if t.startswith("rho_")]
    print(f"{'tracer':>10s} " + " ".join(f"{n[4:]:>14s}" for n in rho_names))
    for tracer in tracers:
        out = rows[tracer]
        print(f"{tracer:>10s} " + " ".join(f"{out[n]:>14.3f}" for n in rho_names))
    print("\nNote: DESI DR1 full-shape per-tracer precision is "
          "sigma(fsigma8)/fsigma8 ~ 4-10% (direct-fit velocileptors + window + "
          "systematics; qualitative anchor only — a Kaiser Gaussian-cov Fisher "
          "is expected to come out somewhat tighter).")


def check_scaling(tracers) -> None:
    import matplotlib.pyplot as plt

    n_factors = None  # per-tracer box from tracers.yaml
    fig, axes = plt.subplots(1, len(tracers), figsize=(4.2 * len(tracers), 3.6),
                             squeeze=False)
    print("\n=== sigma vs N_tracers (monotone decrease, saturation at high N) ===")
    failures = []
    for iax, tracer in enumerate(tracers):
        lo, hi = ntracers_range(tracer, "dr1")
        Ns = np.geomspace(lo, hi, 7)
        sigmas = {n: [] for n in fourier_space.TARGET_NAMES[:4]}
        for N in Ns:
            sample = dict(FID_SAMPLE)
            sample["N_tracers"] = float(N)
            out = fourier_space.run_fisher(sample, tracer_bin=tracer)
            for n in sigmas:
                sigmas[n].append(out[n])
        ax = axes[0][iax]
        for n, vals in sigmas.items():
            vals = np.asarray(vals)
            ax.loglog(Ns, vals, marker="o", label=n[6:])
            if not np.all(np.diff(vals) < 0):
                failures.append((tracer, n))
        # 1/sqrt(N) guide anchored at the lowest N of sigma_qiso.
        guide = sigmas["sigma_qiso"][0] * np.sqrt(Ns[0] / Ns)
        ax.loglog(Ns, guide, "k--", lw=0.8, label=r"$1/\sqrt{N}$")
        ax.set_title(tracer)
        ax.set_xlabel(r"$N_{\rm tracers}$")
        if iax == 0:
            ax.set_ylabel(r"$\sigma$")
            ax.legend(fontsize=7)
        print(f"  {tracer}: sigma_qiso {sigmas['sigma_qiso'][0]:.4f} -> "
              f"{sigmas['sigma_qiso'][-1]:.4f} over N=[{lo:.3g}, {hi:.3g}]")
    fig.tight_layout()
    out_png = plots_dir() / "shapefit_scaling_vs_ntracers.png"
    fig.savefig(out_png, dpi=140)
    print(f"  wrote {out_png}")
    if failures:
        print("  NON-MONOTONE sigma(N) at:", failures)
    else:
        print("  all sigmas monotone decreasing in N. "
              "(Saturation shows as flattening vs the 1/sqrt(N) guide.)")


def check_damping(tracers) -> None:
    print("\n=== Damping-float sensitivity (float_sigma_damp True vs False) ===")
    print(f"{'tracer':>10s} " + " ".join(
        f"{n[6:]:>12s}" for n in fourier_space.TARGET_NAMES[:4]))
    for tracer in tracers:
        outs = {}
        for flag in (True, False):
            outs[flag] = fourier_space.run_fisher(
                _fid_sample_for(tracer), tracer_bin=tracer, float_sigma_damp=flag)
        deltas = []
        for n in fourier_space.TARGET_NAMES[:4]:
            a, b = outs[True][n], outs[False][n]
            deltas.append(f"{(a - b) / b:>+11.1%}")
        print(f"{tracer:>10s} " + " ".join(deltas) + "   (float vs fixed)")


_KMAX_GRID = (0.10, 0.125, 0.15, 0.175, 0.20)


def _forecast_at_kmax(tracer: str, kmax: float) -> dict:
    """Marginalized sigmas at a given fit kmax, plus the fiducial multipoles.

    run_fisher does not expose klim_spec (the generators never vary it), so this
    drives core.build_shapefit_likelihood directly.
    """
    sample = _fid_sample_for(tracer)
    theta = sf_core._to_shapefit_cosmo_params(sample)
    info = sf_core.build_shapefit_likelihood(
        N_tracers=float(sample["N_tracers"]),
        theta_cosmo=theta,
        tracer_bin=tracer,
        klim_spec=(0.02, float(kmax), 0.005),
    )
    cov_phys = fourier_space._sf_fisher_reduction(info)
    out = dict(zip(fourier_space.TARGET_NAMES,
                   fourier_space.fisher_cov_to_emulator_targets(cov_phys)))
    out["k"] = np.asarray(info["observable"].k[0], dtype=np.float64)
    out["flatdata"] = np.asarray(info["observable"].flatdata, dtype=np.float64)
    return out


def check_kmax(tracers) -> None:
    print("\n=== sigma vs fit kmax (is the broken high-k region informative?) ===")
    names = fourier_space.TARGET_NAMES[:4]
    for tracer in tracers:
        print(f"\n  {tracer}")
        print(f"    {'kmax':>6s} " + " ".join(f"{n[6:]:>11s}" for n in names)
              + f"   {'vs kmax=0.20':>12s}")
        rows = {}
        for kmax in _KMAX_GRID:
            rows[kmax] = _forecast_at_kmax(tracer, kmax)
        ref = rows[_KMAX_GRID[-1]]
        for kmax in _KMAX_GRID:
            r = rows[kmax]
            infl = np.mean([r[n] / ref[n] for n in names])
            print(f"    {kmax:>6.3f} " + " ".join(f"{r[n]:>11.5f}" for n in names)
                  + f"   {infl:>11.2f}x")
        # Zero-crossing of the Kaiser quadrupole on the widest band.
        k, flat = ref["k"], ref["flatdata"]
        nk = k.size
        P2 = flat[nk:2 * nk]
        neg = np.where(P2 < 0)[0]
        if neg.size:
            i = int(neg[0])
            if i == 0:
                kz = float(k[0])
            else:  # linear interpolation between the bracketing bins
                kz = float(k[i - 1] + (k[i] - k[i - 1]) * P2[i - 1]
                           / (P2[i - 1] - P2[i]))
            print(f"    P2 crosses zero at k = {kz:.3f} "
                  f"({(k[-1] - kz) / (k[-1] - k[0]):.0%} of the band is past it)")
        else:
            print(f"    P2 stays positive to k = {k[-1]:.3f}")
    print("\n  'vs kmax=0.20' is the mean sigma inflation from truncating the band.")
    print("  Close to 1.0 => the high-k region carries little information, so the")
    print("  Kaiser breakdown there costs little. Much above 1.0 => the labels")
    print("  depend on a regime where the model is known to be wrong.")


# q = 1 should hold to machine precision; the residual ~1e-7 is the
# omega_cdm -> Omega_m -> omega_cdm round trip in _to_mean_extractor_params.
# 1e-5 is loose enough not to flag that and tight enough to catch the bug it
# exists for: dropping omega_ncdm shifts omega_cdm by ~0.0006, which shows up
# here at the 1e-4 level.
_Q_UNITY_TOL = 1e-5
# Agreement required between the extractor's q and the same ratio built
# straight from cosmoprimo. Both are cosmoprimo underneath, so this is a
# convention test (eta, DH/DM orientation, which cosmology is the fiducial),
# not a numerical one -- it passes at <1e-6 or fails outright.
_Q_AP_TOL = 1e-5


def check_mean_ap(tracers) -> None:
    """Self-consistency of the mean pipeline's AP outputs.

    Needs no DESI reference data: qiso and qap are ratios of the SAME
    extractor's varying cosmology to its fixed fiducial, both at the same z.

      (a) at fiducial input, q must be exactly 1 -- the numerator and
          denominator are then the same cosmology. This is the only test that
          exercises the mean path's cosmology mapping (_to_mean_extractor_params
          assembles Omega_m from the omega basis, including omega_ncdm; the
          covar path passes omega_cdm straight through). If those two ever
          disagree, the mean and covar emulators are trained on different
          cosmologies and nothing else in the suite would notice.

      (b) off-fiducial, q must equal the distance ratio computed directly from
          cosmoprimo. This is the AP machinery proper, in the regime the
          emulator is actually used in. Note the comparison plots CANNOT test
          this: their AP panels compare a fiducial to a fiducial, so the
          extractor's numerator never enters.
    """
    print("\n=== Mean-pipeline AP self-consistency ===")
    from cosmoprimo.fiducial import DESI

    def _q_from_cosmoprimo(theta, z):
        c = sf_core.get_cosmo(("DESI", dict(theta)))
        DH = (299792.458 / 1e3) / (100.0 * c.efunc(z))
        DM = c.comoving_angular_distance(z)
        # eta = 1/3, matching BAOExtractor._set_base.
        DV = DH ** (1.0 / 3.0) * DM ** (2.0 / 3.0) * z ** (1.0 / 3.0)
        return DV / c.rs_drag, DH / DM      # (DV/rd, DH/DM) -- NOT DM/DH

    def _run(sample, tracer, z):
        _s, vals, err = fourier_space._worker_run_mean_targets(
            (sample, tracer, z, sf_core.PARAM_DEFAULTS))
        if vals is None:
            raise RuntimeError(f"{tracer}: mean worker failed\n{err}")
        return float(vals[0]), float(vals[1])

    perturbations = [("omega_cdm +5%", {"omega_cdm": FID_SAMPLE["omega_cdm"] * 1.05}),
                     ("h -3%", {"h": FID_SAMPLE["h"] * 0.97})]
    fails = 0
    print(f"{'tracer':>8s} {'case':>14s} {'qiso':>12s} {'qap':>12s} {'max dev':>10s}")
    for t in tracers:
        cosmo = DESI()
        cfg = get_tracer_config(t, analysis="shapefit", dataset="dr1")
        z = sf_core._fs_compute_z_eff(
            tracer_bin=t, cosmo=cosmo, fo=cosmo.get_fourier(),
            area_deg2=float(sf_core.dataset_area("dr1")),
            b1=float(cfg.get("bias_recon", 2.0)))

        qiso, qap = _run(dict(FID_SAMPLE), t, z)
        dev = max(abs(qiso - 1.0), abs(qap - 1.0))
        ok = dev < _Q_UNITY_TOL
        fails += not ok
        print(f"{t:>8s} {'fiducial':>14s} {qiso:12.8f} {qap:12.8f} "
              f"{dev:10.1e} {'' if ok else '  <-- FAIL'}")

        for name, pert in perturbations:
            s = {**FID_SAMPLE, **pert}
            qiso, qap = _run(s, t, z)
            theta_p = sf_core._to_shapefit_cosmo_params(s)
            theta_f = sf_core._to_shapefit_cosmo_params(dict(FID_SAMPLE))
            dv_p, dhdm_p = _q_from_cosmoprimo(theta_p, z)
            dv_f, dhdm_f = _q_from_cosmoprimo(theta_f, z)
            dev = max(abs(qiso / (dv_p / dv_f) - 1.0),
                      abs(qap / (dhdm_p / dhdm_f) - 1.0))
            ok = dev < _Q_AP_TOL
            fails += not ok
            print(f"{'':>8s} {name:>14s} {qiso:12.8f} {qap:12.8f} "
                  f"{dev:10.1e} {'' if ok else '  <-- FAIL'}")

    print(f"\n  'max dev' is |q - 1| for the fiducial row and |q/q_cosmoprimo - 1|")
    print(f"  for the perturbed rows. Tolerances {_Q_UNITY_TOL:g} / {_Q_AP_TOL:g}.")
    if fails:
        raise SystemExit(f"  {fails} AP self-consistency check(s) FAILED")
    print("  all rows pass.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check",
                   choices=["fiducial", "scaling", "damping", "kmax",
                            "mean-ap", "all"],
                   default="all")
    p.add_argument("--tracers", nargs="*", default=None,
                   choices=list(TRACER_TYPE_CHOICES),
                   help=f"subset of {list(TRACERS_ALL)}; default all 6")
    args = p.parse_args()

    tracers = args.tracers or list(TRACERS_ALL)
    if args.check in ("fiducial", "all"):
        check_fiducial(tracers)
    if args.check in ("scaling", "all"):
        check_scaling(tracers)
    if args.check in ("damping", "all"):
        check_damping(tracers)
    if args.check in ("kmax", "all"):
        check_kmax(tracers)
    if args.check in ("mean-ap", "all"):
        check_mean_ap(tracers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
