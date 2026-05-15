"""Sweep the kmin floor on LRG1 and LRG2 (the tracers where k_IC > kmin_eff)
to quantify how much σ(α) depends on the low-k cutoff.

If σ is flat between kmin = 0.02 and kmin = k_IC, the radial IC suppression
of modes just above the existing floor is sub-percent. If σ drops noticeably
when kmin → k_IC, then the IC carries real σ-budget for those tracers.

Implementation: monkey-patch `build_bao_likelihood` to expose the kmin floor
via a `kmin_floor_override` argument injected through TRACER_CONFIGS, then
sweep.

Simpler approach used here: directly edit the kmin_eff line in prep_covar
via runtime monkey-patching of the constant value.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

import prep_covar as pc
from util import TRACER_CONFIGS
from plot_fisher_vs_desi import _get_ntracers


# Trick: replace prep_covar.build_bao_likelihood internals by importing and
# rewriting the module source line — too brittle. Instead, override `max` in
# the prep_covar namespace temporarily during the call so the
# `kmin_eff = max(0.02, kmin_window)` line picks up a custom floor.
import builtins

class _MaxFloorOverride:
    """Context manager that injects a patched `max` into the prep_covar
    module namespace; Python's LOAD_GLOBAL looks in module globals before
    builtins, so the patched version is picked up by every call inside
    prep_covar.build_bao_likelihood. The patched function only intervenes
    when called as `max(0.02, kmin_window)` — i.e. only for the kmin floor
    line — and is transparent for every other use of max() inside the
    module."""
    def __init__(self, new_floor):
        self.new_floor = new_floor
        self._had_attr = False
        self._orig = None

    def __enter__(self):
        self._had_attr = "max" in pc.__dict__
        if self._had_attr:
            self._orig = pc.__dict__["max"]
        new_floor = self.new_floor
        builtin_max = builtins.max
        def patched_max(a, b, *args, **kwargs):
            if isinstance(a, float) and abs(a - 0.02) < 1e-12:
                return builtin_max(new_floor, b, *args, **kwargs)
            return builtin_max(a, b, *args, **kwargs)
        pc.max = patched_max

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._had_attr:
            pc.max = self._orig
        else:
            del pc.max


def _sigma_at_kmin(tracer, dataset, kmin_floor, omega_m=0.3153, hrdrag=99.53):
    cfg = TRACER_CONFIGS[tracer]
    n_tracers = _get_ntracers(dataset, tracer)
    area = {"dr1": 7500.0, "dr2": 14000.0}[dataset]
    sample = {"N_tracers": n_tracers, "Om": omega_m, "hrdrag": hrdrag}
    with _MaxFloorOverride(kmin_floor):
        targets = pc.run_fisher(
            sample, tracer_bin=tracer,
            zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
            param_defaults=pc.PARAM_DEFAULTS, area=area,
        )
    var_dh = targets["cov_DH_over_rd_DH_over_rd"]
    var_dm = targets["cov_DM_over_rd_DM_over_rd"]
    cov_dh_dm = targets["cov_DH_over_rd_DM_over_rd"]
    sig_dh = float(np.sqrt(var_dh))
    sig_dm = float(np.sqrt(var_dm))
    # σ(DV) via fid DH/DM and α_iso propagation
    theta, hrr = pc._to_bao_cosmo_params({**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
    rd = hrr / pc._H_FID
    template = pc.BAOPowerSpectrumTemplate(
        z=float(cfg["z_eff"]), fiducial=("DESI", dict(theta)), apmode="qparqper"
    )
    dh_mean = float(template.DH_fid) / rd
    dm_mean = float(template.DM_fid) / rd
    # σ_DV from prep_covar's convention
    from plot_fisher_vs_desi import _dv_from_dh_dm_cov
    sig_dv = float(_dv_from_dh_dm_cov(var_dh, cov_dh_dm, var_dm,
                                       float(cfg["z_eff"]), dh_mean, dm_mean))
    return sig_dh, sig_dm, sig_dv


def main():
    print(f"\nkmin sweep on tracers where k_IC > kmin_eff (LRG1, LRG2).")
    print(f"kmin = 0.02 is the production floor; higher kmin simulates explicit IC removal.\n")
    for tracer, k_IC in [("LRG1", 0.0206), ("LRG2", 0.0232)]:
        print(f"=== {tracer}  (k_IC = {k_IC:.4f}) ===")
        print(f"{'kmin':>8} {'σ(DH)':>9} {'σ(DM)':>9} {'σ(DV)':>9}  rel.Δσ(DH) vs kmin=0.02")
        sigs_ref = None
        for kmin in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040]:
            sdh, sdm, sdv = _sigma_at_kmin(tracer, "dr1", kmin)
            if abs(kmin - 0.020) < 1e-12:
                sigs_ref = (sdh, sdm, sdv)
            mark = "  (ref)" if abs(kmin - 0.020) < 1e-12 else \
                   (f"  {(sdh/sigs_ref[0]-1)*100:+.2f}%" if sigs_ref else "")
            print(f"{kmin:>8.4f} {sdh:>9.4f} {sdm:>9.4f} {sdv:>9.4f}{mark}")
        print()


if __name__ == "__main__":
    main()
