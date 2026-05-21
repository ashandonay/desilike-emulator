"""Plot Fisher-pipeline vs DESI-observed sigma(DH/rd), sigma(DM/rd),
sigma(DV/rd) for all tracer bins on a single panel per dataset (dr1, dr2).

DH/rd = blue, DM/rd = orange, DV/rd = green. Fisher = filled dots,
DESI = x markers, MCMC (with --with-mcmc) = open diamonds. Tracers that
DESI quotes only DV/rd for (DR1 BGS+QSO, DR2 BGS) are shown only in
green. Computes Fisher on-the-fly via prep_covar.run_fisher; MCMC values
are read from cached JSON in mcmc_results/{tracer}_{dataset}.json
(produced by `mcmc_bao.py --save-json`).
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import prep_covar as pc
from util import TRACER_CONFIGS


_DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}
_NAME_MAP_TO_DATA = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
_DISPLAY_NAME = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lyα QSO"}
_MCMC_DIR = Path(__file__).resolve().parent / "mcmc_results"
_MCMC_KEY = {"DH_over_rs": "sigma_DH_over_rd_mcmc",
             "DM_over_rs": "sigma_DM_over_rd_mcmc",
             "DV_over_rs": "sigma_DV_over_rd_mcmc"}
QUANTITIES = ["DH_over_rs", "DM_over_rs", "DV_over_rs"]
QUANT_LABEL = {"DH_over_rs": r"$\sigma(D_H/r_d)$",
               "DM_over_rs": r"$\sigma(D_M/r_d)$",
               "DV_over_rs": r"$\sigma(D_V/r_d)$"}


def _get_ntracers(dataset, tracer_key):
    df = pd.read_csv(Path.home() / "data" / "desi" / f"bao_{dataset}" / "desi_data.csv")
    df = df[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by.get(_NAME_MAP_TO_DATA.get(tracer_key, tracer_key))


def _get_desi_std(dataset, tracer_key):
    df = pd.read_csv(Path.home() / "data" / "desi" / f"bao_{dataset}" / "desi_data.csv")
    sub = df[df["tracer"] == _NAME_MAP_TO_DATA.get(tracer_key, tracer_key)]
    out = {row["quantity"]: float(row["std"]) for _, row in sub.iterrows()}
    return out


def _load_mcmc_stds(dataset):
    """Read cached MCMC sigmas from mcmc_results/{tracer}_{dataset}.json.

    JSON files were written before the prep_covar unit fix, so the cached
    sigma_*_over_rd_mcmc values were computed with the dimensionally mixed
    `template.DH_fid / info["rd"]` form and are deflated by 1/h. Multiply
    by 1/h_fid to get the correct dimensionless σ.

    Missing files are silently skipped — re-run mcmc_bao.py --save-json
    for the tracers you want overlaid.
    """
    h_corr = 1.0 / pc._H_FID
    out = {}
    for tracer_key in TRACER_CONFIGS:
        path = _MCMC_DIR / f"{tracer_key}_{dataset}.json"
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text())
        except Exception as e:
            print(f"[mcmc] {tracer_key}: failed to read {path}: {e}")
            continue
        out[tracer_key] = {
            q: float(blob.get(_MCMC_KEY[q], np.nan)) * h_corr for q in QUANTITIES
        }
    return out


def _dv_from_dh_dm_cov(var_dh, cov_dh_dm, var_dm, z_eff, dh_mean, dm_mean):
    dv_mean = (z_eff * dm_mean * dm_mean * dh_mean) ** (1.0 / 3.0)
    d_dh = (1.0 / 3.0) * dv_mean / dh_mean
    d_dm = (2.0 / 3.0) * dv_mean / dm_mean
    var_dv = d_dh * d_dh * var_dh + 2.0 * d_dh * d_dm * cov_dh_dm + d_dm * d_dm * var_dm
    return np.sqrt(max(var_dv, 0.0))


def _fisher_stds(dataset, omega_m, hrdrag):
    area = _DATASET_AREAS[dataset.lower()]
    out = {}
    # z_eff is now derived from the n(z) slices via Fisher-info weighting
    # inside build_bao_likelihood (cosmology-clean). We pass None into
    # run_fisher and recover the derived value below to evaluate the template
    # at the same z_eff.
    theta_for_z, _ = pc._to_bao_cosmo_params({**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
    cosmo_for_z = pc.get_cosmo(("DESI", dict(theta_for_z)))
    fo_for_z = cosmo_for_z.get_fourier()
    for key, cfg in TRACER_CONFIGS.items():
        if not cfg.get("supported", True):
            continue
        zrange = tuple(cfg["zrange"])
        try:
            z_eff = pc._compute_z_eff_from_nz(
                tracer_bin=key, cosmo=cosmo_for_z, fo=fo_for_z,
                area_deg2=float(area), b1=float(cfg.get("bias_recon", 2.0)),
            )
        except (FileNotFoundError, ValueError):
            z_eff = float(cfg["z_eff"])
        n_tracers = _get_ntracers(dataset, key)
        if n_tracers is None:
            continue
        sample = {"N_tracers": n_tracers, "Om": omega_m, "hrdrag": hrdrag}
        try:
            targets = pc.run_fisher(
                sample, tracer_bin=key, zrange=zrange, z_eff=z_eff,
                param_defaults=pc.PARAM_DEFAULTS, area=area,
            )
        except Exception as e:
            print(f"[fisher] {dataset} {key} FAILED: {e}")
            continue
        var_dh = targets["cov_DH_over_rd_DH_over_rd"]
        cov_dh_dm = targets["cov_DH_over_rd_DM_over_rd"]
        var_dm = targets["cov_DM_over_rd_DM_over_rd"]
        theta, hr = pc._to_bao_cosmo_params({**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
        template = pc.BAOPowerSpectrumTemplate(z=z_eff, fiducial=("DESI", dict(theta)), apmode="qparqper")
        # Use template.{DH,DM}_over_rd_fid directly — dimensionally correct.
        # The `template.DH_fid / rd` form is dimensionally mixed and underreports σ by 1/h.
        dh_mean = float(template.DH_over_rd_fid)
        dm_mean = float(template.DM_over_rd_fid)
        out[key] = {
            "DH_over_rs": float(np.sqrt(max(var_dh, 0.0))),
            "DM_over_rs": float(np.sqrt(max(var_dm, 0.0))),
            "DV_over_rs": float(_dv_from_dh_dm_cov(var_dh, cov_dh_dm, var_dm, z_eff, dh_mean, dm_mean)),
        }
    return out


_QUANTITY_STYLES = [
    ("DH_over_rs", "tab:blue",   r"$\sigma(D_H/r_d)$"),
    ("DM_over_rs", "tab:orange", r"$\sigma(D_M/r_d)$"),
    ("DV_over_rs", "tab:green",  r"$\sigma(D_V/r_d)$"),
]
_QUANT_SHORT = {"DH_over_rs": "DH/rd",
                "DM_over_rs": "DM/rd",
                "DV_over_rs": "DV/rd"}


def _print_table(dataset, fisher, desi, mcmc=None):
    supported = [k for k, cfg in TRACER_CONFIGS.items() if cfg.get("supported", True)]
    tracers = [k for k in supported
               if any(np.isfinite(desi.get(k, {}).get(q, np.nan)) for q in QUANTITIES)]
    has_mcmc = mcmc is not None
    print(f"\n=== {dataset.upper()} sigmas: Fisher vs DESI"
          f"{' vs MCMC' if has_mcmc else ''} ===")
    if has_mcmc:
        header = (f"  {'tracer':<12} {'quantity':<7} {'Fisher':>10} {'MCMC':>10} "
                  f"{'DESI':>10} {'F/D':>7} {'M/D':>7}")
    else:
        header = (f"  {'tracer':<12} {'quantity':<7} {'Fisher':>10} {'DESI':>10} "
                  f"{'F/D':>7} {'(F-D)/D':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in tracers:
        for q in QUANTITIES:
            dv = desi.get(key, {}).get(q, np.nan)
            fv = fisher.get(key, {}).get(q, np.nan)
            if not np.isfinite(dv):
                continue
            ratio = fv / dv if np.isfinite(fv) and dv != 0 else np.nan
            fv_s = f"{fv:.4f}" if np.isfinite(fv) else "  --  "
            ratio_s = f"{ratio:.3f}" if np.isfinite(ratio) else "  --  "
            if has_mcmc:
                mv = (mcmc.get(key, {}) or {}).get(q, np.nan)
                mv_s = f"{mv:.4f}" if np.isfinite(mv) else "  --  "
                mratio = mv / dv if np.isfinite(mv) and dv != 0 else np.nan
                mratio_s = f"{mratio:.3f}" if np.isfinite(mratio) else "  --  "
                print(f"  {key:<12} {_QUANT_SHORT[q]:<7} {fv_s:>10} {mv_s:>10} "
                      f"{dv:>10.4f} {ratio_s:>7} {mratio_s:>7}")
            else:
                pct = (fv - dv) / dv * 100 if np.isfinite(fv) and dv != 0 else np.nan
                pct_s = f"{pct:+6.1f}%" if np.isfinite(pct) else "  --  "
                print(f"  {key:<12} {_QUANT_SHORT[q]:<7} {fv_s:>10} "
                      f"{dv:>10.4f} {ratio_s:>7} {pct_s:>10}")


def _plot(dataset, fisher, desi, out_path, omega_m, hrdrag, mcmc=None):
    """One-panel: DH/rd blue, DM/rd orange, DV/rd green; Fisher=dot, DESI=x,
    MCMC=open diamond (only drawn when mcmc dict is provided and contains
    finite values for that tracer × quantity).

    Plots a Fisher point only for (tracer, quantity) pairs that DESI also
    quotes -- so DR1 BGS/QSO and DR2 BGS show only DV/rd.
    """
    supported = [k for k, cfg in TRACER_CONFIGS.items() if cfg.get("supported", True)]
    tracers = [k for k in supported
               if any(np.isfinite(desi.get(k, {}).get(q, np.nan)) for q in QUANTITIES)]
    display = [_DISPLAY_NAME.get(k, k) for k in tracers]
    x = np.arange(len(tracers), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6.0), constrained_layout=True)
    for xi in x:
        ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)

    for q, color, _ in _QUANTITY_STYLES:
        f_vals, f_xs, d_vals, d_xs, m_vals, m_xs = [], [], [], [], [], []
        for i, key in enumerate(tracers):
            dv = desi.get(key, {}).get(q, np.nan)
            fv = fisher.get(key, {}).get(q, np.nan)
            if np.isfinite(dv):
                d_vals.append(dv); d_xs.append(i)
                if np.isfinite(fv):
                    f_vals.append(fv); f_xs.append(i)
                if mcmc is not None:
                    mv = (mcmc.get(key, {}) or {}).get(q, np.nan)
                    if np.isfinite(mv):
                        m_vals.append(mv); m_xs.append(i)
        ax.scatter(f_xs, f_vals, marker="o", s=80, color=color,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter(d_xs, d_vals, marker="x", s=90, color=color,
                   linewidths=2.5, zorder=3)
        if mcmc is not None and m_vals:
            ax.scatter(m_xs, m_vals, marker="D", s=70,
                       facecolors="none", edgecolors=color,
                       linewidths=1.8, zorder=4)

    # Two separate legends: one for source markers, one for quantity colors.
    source_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor="gray", markeredgecolor="black", label="Fisher"),
        Line2D([0], [0], marker="x", linestyle="", markersize=10,
               markeredgewidth=2.5, color="gray", label="DESI"),
    ]
    if mcmc is not None:
        source_handles.append(
            Line2D([0], [0], marker="D", linestyle="", markersize=9,
                   markerfacecolor="none", markeredgecolor="gray",
                   markeredgewidth=1.8, label="MCMC")
        )
    quantity_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor=color, markeredgecolor="none", label=label)
        for _, color, label in _QUANTITY_STYLES
    ]
    leg_src = ax.legend(handles=source_handles, loc="upper left",
                        bbox_to_anchor=(0.0, 1.0),
                        title="Source", fontsize=10, title_fontsize=10)
    ax.add_artist(leg_src)
    # Drop the quantity legend down enough to clear the (possibly 3-row) source legend.
    quant_y = 0.74 if mcmc is not None else 0.80
    ax.legend(handles=quantity_handles, loc="upper left",
              bbox_to_anchor=(0.0, quant_y),
              title="Quantity", fontsize=10, title_fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=20)
    ax.set_xlabel("Tracer bin")
    ax.set_ylabel(r"$\sigma$")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")
    ax.set_title(
        f"BAO sigmas: Fisher vs DESI  |  {dataset.upper()}  |  "
        f"Om={omega_m:.4g}, hrdrag={hrdrag:.4g}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omega-m", type=float, default=0.3152)
    ap.add_argument("--hrdrag", type=float, default=99.08)
    ap.add_argument("--out-dir", default="/home/ashandonay/bedcosmo/src/bedcosmo/num_tracers/emulator/bao")
    ap.add_argument("--datasets", nargs="+", default=["dr1", "dr2"],
                    choices=["dr1", "dr2"])
    ap.add_argument("--with-mcmc", action="store_true",
                    help="Overlay cached MCMC sigmas from mcmc_results/"
                         "{tracer}_{dataset}.json as open diamonds. Does NOT "
                         "re-run MCMC -- run mcmc_bao.py --save-json first.")
    args = ap.parse_args()

    for dataset in args.datasets:
        print(f"\n=== Computing Fisher for {dataset} ===")
        fisher = _fisher_stds(dataset, args.omega_m, args.hrdrag)
        print(f"=== Loading DESI stds for {dataset} ===")
        desi = {key: _get_desi_std(dataset, key) for key in TRACER_CONFIGS}

        mcmc = None
        if args.with_mcmc:
            mcmc = _load_mcmc_stds(dataset)
            if not mcmc:
                print(f"[mcmc] no cached results found in {_MCMC_DIR} "
                      f"for {dataset}; skipping overlay")
                mcmc = None

        _print_table(dataset, fisher, desi, mcmc=mcmc)

        mcmc_suffix = "_mcmc" if mcmc is not None else ""
        out_path = Path(args.out_dir) / (
            f"fisher_vs_desi{mcmc_suffix}_{dataset}_"
            f"Om_{args.omega_m:.5f}_hrdrag_{args.hrdrag:.5f}.png"
        )
        _plot(dataset, fisher, desi, out_path, args.omega_m, args.hrdrag, mcmc=mcmc)


if __name__ == "__main__":
    main()
