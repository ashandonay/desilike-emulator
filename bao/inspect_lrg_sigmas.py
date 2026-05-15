"""Print Sigma_post breakdown for LRG1 vs LRG2 side by side."""
import sys
sys.path.insert(0, "..")
import numpy as np
import prep_covar as P
from util import TRACER_CONFIGS

# DESI fiducial DR1 cosmology
sample_base = {**P.PARAM_DEFAULTS, "Om": 0.3153, "hrdrag": 99.53}
area = 7500.0

# DR1 N_tracers (from DR1 data csv, "passed" column)
import pandas as pd
from pathlib import Path
desi_dir = Path.home() / "data" / "desi" / "bao_dr1"
df = pd.read_csv(desi_dir / "desi_data.csv")[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
n_by = {row["tracer"]: float(row["passed"]) for _, row in df.iterrows()}
print(f"DR1 N_tracers: LRG1={n_by['LRG1']:.0f}  LRG2={n_by['LRG2']:.0f}")

results = {}
for tb in ("LRG1", "LRG2"):
    cfg = TRACER_CONFIGS[tb]
    s = dict(sample_base)
    s["N_tracers"] = n_by[tb]
    sigma_perp, sigma_par = P.compute_pipeline_sigmas(
        sample=s, tracer_bin=tb,
        zrange=tuple(cfg["zrange"]), z_eff=float(cfg["z_eff"]),
        param_defaults=P.PARAM_DEFAULTS, area=area, n_iter=1,
    )
    results[tb] = dict(sigma_perp=sigma_perp, sigma_par=sigma_par, cfg=cfg, N=n_by[tb])

print()
print(f"{'':<22}{'LRG1':>14}{'LRG2':>14}{'ratio L2/L1':>14}")
for k in ("z_eff", "bias_recon", "smoothing_scale"):
    v1 = results["LRG1"]["cfg"][k]; v2 = results["LRG2"]["cfg"][k]
    print(f"  {k:<20}{v1:>14}{v2:>14}{'':>14}")
print(f"  {'N_tracers':<20}{results['LRG1']['N']:>14.0f}{results['LRG2']['N']:>14.0f}{results['LRG2']['N']/results['LRG1']['N']:>14.3f}")
for k in ("sigma_perp", "sigma_par"):
    v1 = results["LRG1"][k]; v2 = results["LRG2"][k]
    print(f"  {k:<20}{v1:>14.4f}{v2:>14.4f}{v2/v1:>14.3f}")

# Volume / nbar
from desilike.observables.galaxy_clustering import CutskyFootprint
from cosmoprimo.fiducial import DESI
cosmo = DESI(**P._to_bao_cosmo_params(sample_base)[0])
print()
print(f"{'':<22}{'LRG1':>14}{'LRG2':>14}{'ratio L2/L1':>14}")
for tb in ("LRG1", "LRG2"):
    cfg = results[tb]["cfg"]
    foot = CutskyFootprint(area=area, zrange=tuple(cfg["zrange"]),
                            nbar=results[tb]["N"]/area, cosmo=cosmo)
    results[tb]["V"] = float(foot.volume)
    results[tb]["nbar"] = results[tb]["N"] / float(foot.volume)
v1, v2 = results["LRG1"]["V"], results["LRG2"]["V"]
print(f"  {'V (Gpc/h)^3':<20}{v1/1e9:>14.3f}{v2/1e9:>14.3f}{v2/v1:>14.3f}")
v1, v2 = results["LRG1"]["nbar"], results["LRG2"]["nbar"]
print(f"  {'nbar (h/Mpc)^3':<20}{v1:>14.3e}{v2:>14.3e}{v2/v1:>14.3f}")
