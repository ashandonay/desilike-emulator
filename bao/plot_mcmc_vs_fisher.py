"""Compare pipeline Fisher, MCMC, and DESI σ values per tracer × quantity.

Reads MCMC JSON outputs from mcmc_results/{tracer}_dr1.json (produced by
mcmc_bao.py --save-json), recomputes the matching Fisher results inline via
run_fisher, pulls DESI published σ from desi_data.csv, and produces a single
PNG with one subplot per quantity (DH/rd, DM/rd, DV/rd).

Run from bao/:
    ~/miniconda3/envs/emulator/bin/python plot_mcmc_vs_fisher.py
"""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")

import prep_covar as pc
from util import TRACER_CONFIGS


_DR1_AREA = 7500.0
_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_MCMC_DIR = Path(__file__).resolve().parent / "mcmc_results"
_DESI_FID = {"Om": 0.3152, "hrdrag": 99.08}  # match mcmc_bao.py default cosmology so Fisher and MCMC use the same fid


def _load_dr1_ntracers():
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")[["tracer", "passed"]].drop_duplicates("tracer")
    out = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return out


def _load_dr1_sigmas():
    df = pd.read_csv(_DR1_DIR / "desi_data.csv")
    out = {}
    for _, row in df.iterrows():
        out.setdefault(row["tracer"], {})[row["quantity"]] = float(row["std"])
    return out


def _fisher_sigmas(tracer_bin: str, n_tracers: float):
    """Return Fisher sigma(DH/rd), sigma(DM/rd), sigma(DV/rd) for one tracer."""
    cfg = TRACER_CONFIGS[tracer_bin]
    sample = {"N_tracers": n_tracers, **_DESI_FID}
    targets = pc.run_fisher(
        sample, tracer_bin=tracer_bin, zrange=tuple(cfg["zrange"]),
        z_eff=None, param_defaults=pc.PARAM_DEFAULTS, area=_DR1_AREA,
    )
    var_dh = float(targets["cov_DH_over_rd_DH_over_rd"])
    cov_dhdm = float(targets["cov_DH_over_rd_DM_over_rd"])
    var_dm = float(targets["cov_DM_over_rd_DM_over_rd"])

    # Need z_eff that was actually used for the template
    theta, _ = pc._to_bao_cosmo_params({**pc.PARAM_DEFAULTS, **_DESI_FID})
    cosmo_z = pc.get_cosmo(("DESI", dict(theta)))
    fo_z = cosmo_z.get_fourier()
    try:
        z_eff = pc._compute_z_eff_from_nz(
            tracer_bin=tracer_bin, cosmo=cosmo_z, fo=fo_z,
            area_deg2=_DR1_AREA, b1=float(cfg.get("bias_recon", 2.0)),
        )
    except Exception:
        z_eff = float(cfg["z_eff"])

    template = pc.BAOPowerSpectrumTemplate(z=z_eff, fiducial=("DESI", dict(theta)), apmode="qparqper")
    # Use template.{DH,DM}_over_rd_fid directly — dimensionally correct.
    # The `template.DH_fid / rd` form is dimensionally mixed and underreports σ by 1/h.
    dh = float(template.DH_over_rd_fid)
    dm = float(template.DM_over_rd_fid)
    sd_dh = math.sqrt(max(var_dh, 0.0))
    sd_dm = math.sqrt(max(var_dm, 0.0))
    dv = (z_eff * dm * dm * dh) ** (1.0 / 3.0)
    g_dh = dv / (3.0 * dh)
    g_dm = 2.0 * dv / (3.0 * dm)
    var_dv = g_dh ** 2 * var_dh + g_dm ** 2 * var_dm + 2.0 * g_dh * g_dm * cov_dhdm
    sd_dv = math.sqrt(max(var_dv, 0.0))
    return {"DH_over_rs": sd_dh, "DM_over_rs": sd_dm, "DV_over_rs": sd_dv, "z_eff": z_eff}


def main():
    tracers = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
    n_by = _load_dr1_ntracers()
    name_map = {"LRG3_ELG1": "LRG3+ELG1"}
    desi_sigmas = _load_dr1_sigmas()

    rows = []
    for tb in tracers:
        desi_name = name_map.get(tb, tb)
        N = n_by.get(desi_name)
        if N is None:
            print(f"[skip] {tb}: no DR1 N_tracers")
            continue
        # Fisher
        fish = _fisher_sigmas(tb, N)
        # MCMC: JSONs were re-generated post-§31g.duodecimus z-factor/unit
        # fix, so sigma_*_over_rd_mcmc values are already dimensionally
        # correct (use mcmc_bao.py's DH_fid_over_rd directly). No 1/h
        # correction needed.
        _h_corr = 1.0
        mcmc_path = _MCMC_DIR / f"{tb}_dr1.json"
        if not mcmc_path.exists():
            print(f"[skip] {tb}: missing {mcmc_path}")
            mcmc = {"sigma_DH_over_rd_mcmc": float("nan"),
                    "sigma_DM_over_rd_mcmc": float("nan"),
                    "sigma_DV_over_rd_mcmc": float("nan")}
        else:
            mcmc = json.loads(mcmc_path.read_text())
        # DESI
        desi = desi_sigmas.get(desi_name, {})
        row = {
            "tracer": tb,
            "z_eff": fish["z_eff"],
            "Fisher_DH": fish["DH_over_rs"],
            "Fisher_DM": fish["DM_over_rs"],
            "Fisher_DV": fish["DV_over_rs"],
            "MCMC_DH": float(mcmc.get("sigma_DH_over_rd_mcmc", float("nan"))) * _h_corr,
            "MCMC_DM": float(mcmc.get("sigma_DM_over_rd_mcmc", float("nan"))) * _h_corr,
            "MCMC_DV": float(mcmc.get("sigma_DV_over_rd_mcmc", float("nan"))) * _h_corr,
            "DESI_DH": desi.get("DH_over_rs", float("nan")),
            "DESI_DM": desi.get("DM_over_rs", float("nan")),
            "DESI_DV": desi.get("DV_over_rs", float("nan")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot: 3 subplots (DH, DM, DV). Each: bar groups by tracer with
    # Fisher/DESI and MCMC/DESI ratios.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    quantities = [("DH", "DH/rd"), ("DM", "DM/rd"), ("DV", "DV/rd")]
    x = np.arange(len(df))
    width = 0.35

    for ax, (q_short, q_label) in zip(axes, quantities):
        fisher_ratio = df[f"Fisher_{q_short}"] / df[f"DESI_{q_short}"]
        mcmc_ratio = df[f"MCMC_{q_short}"] / df[f"DESI_{q_short}"]
        ax.bar(x - width/2, fisher_ratio, width, label="Fisher / DESI", color="C0", alpha=0.85)
        ax.bar(x + width/2, mcmc_ratio, width, label="MCMC / DESI", color="C1", alpha=0.85)
        ax.axhline(1.0, color="k", lw=0.8, ls="--", label="DESI" if q_short == "DH" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(df["tracer"], rotation=30, ha="right")
        ax.set_title(f"σ({q_label})")
        if q_short == "DH":
            ax.set_ylabel("σ_pipeline / σ_DESI")
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate gap-vs-1
        for i, (fr, mr) in enumerate(zip(fisher_ratio, mcmc_ratio)):
            if not math.isnan(fr):
                ax.text(i - width/2, fr + 0.02, f"{fr:.2f}", ha="center", fontsize=8)
            if not math.isnan(mr):
                ax.text(i + width/2, mr + 0.02, f"{mr:.2f}", ha="center", fontsize=8)

    axes[0].legend(loc="lower right")
    fig.suptitle("Fisher vs MCMC vs DESI σ — DR1 (cosmology Ωm=0.3153, hrdrag=99.53)")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fisher_vs_mcmc_vs_desi_dr1.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved: {out}")

    # CSV companion
    csv_out = out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
