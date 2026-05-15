"""Radial integral-constraint (IC) audit.

The radial IC arises because the random catalog is built with the same n̄(z)
as the data, which zeros the LOS-averaged mode by construction. The effect
is a suppression of modes with k_∥ below k_IC ~ 2π/L_z, where L_z is the
comoving LOS extent of the redshift slice.

If k_IC sits BELOW our pipeline's kmin_eff floor (0.02 h/Mpc), the IC modes
are already excluded by the existing k cutoff and adding the IC explicitly
gives zero impact. Computed at DR1 fiducial cosmology.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

import prep_covar as pc
from util import TRACER_CONFIGS


KMIN_EFF_FLOOR = 0.02


def main():
    theta, _hr = pc._to_bao_cosmo_params(
        {**pc.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53}
    )
    cosmo = pc.get_cosmo(("DESI", dict(theta)))

    print(f"\nRadial IC scale k_IC = 2π / L_z   vs   pipeline kmin_eff = {KMIN_EFF_FLOOR}")
    print(f"At DR1 fiducial Om={0.3153}, h={float(cosmo.h):.4f}\n")
    hdr = ("Tracer", "zrange", "L_z [Mpc/h]", "k_IC [h/Mpc]", "kmin_eff", "k_IC<kmin?")
    print(f"{hdr[0]:<12} {hdr[1]:<13} {hdr[2]:>13} {hdr[3]:>14} {hdr[4]:>10} {hdr[5]:>12}")
    print("-" * 79)
    rows = []
    for tracer in ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]:
        z_min, z_max = TRACER_CONFIGS[tracer]["zrange"]
        chi_min = float(cosmo.comoving_radial_distance(z_min)) * float(cosmo.h)
        chi_max = float(cosmo.comoving_radial_distance(z_max)) * float(cosmo.h)
        L_z = chi_max - chi_min
        k_IC = 2.0 * np.pi / L_z
        flag = "YES" if k_IC < KMIN_EFF_FLOOR else "no"
        zr = f"[{z_min:.1f},{z_max:.1f}]"
        print(f"{tracer:<12} {zr:<13} {L_z:>13.1f} {k_IC:>14.5f} "
              f"{KMIN_EFF_FLOOR:>10.4f} {flag:>12}")
        rows.append((tracer, L_z, k_IC))

    print("\nInterpretation:")
    n_below = sum(1 for _, _, k in rows if k < KMIN_EFF_FLOOR)
    if n_below == len(rows):
        print(f"  All {len(rows)} tracers have k_IC < kmin_eff. The radial IC suppresses")
        print(f"  only modes that the pipeline already excludes — adding an explicit IC")
        print(f"  treatment changes nothing for the BAO Fisher.")
    else:
        print(f"  {n_below}/{len(rows)} tracers have k_IC < kmin_eff. The remaining tracers")
        print(f"  might see a small effect from explicit IC modeling.")


if __name__ == "__main__":
    main()
