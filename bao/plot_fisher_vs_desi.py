"""Money plot — Fisher + MCMC vs DESI bao-recon (DR1).

Three stacked panels (DH/rd, DM/rd, DV/rd) sharing the tracer x-axis.
Per (tracer, quantity), five markers:
  - black x       : DESI bao-recon σ (post-marg α-cov × OUR fid (D/rd))
  - filled circle : production Fisher  (blue, analytic Gaussian-only cov)
  - open circle   : production MCMC    (blue, analytic cov, posterior σ)
  - filled square : bundle-cov Fisher  (orange, DESI bundle ξ-cov substituted)
  - open square   : bundle-cov MCMC    (orange, bundle ξ-cov, posterior σ)

DESI reference is the per-tracer bao-recon h5 stat-only file. This removes
the ~5-10% fiducial-conversion mismatch in desi_data.csv space.

The open-vs-filled gap (same shape) visualizes Fisher-vs-MCMC posterior-width
gap. The orange-vs-blue gap (same fill style) visualizes analytic-vs-bundle
cov-shape impact. Per §33t, the open markers (MCMC) close to ±7% of DESI
on all 10 real channels.

DESI publishes DV for sparse tracers (qiso: BGS, QSO) and (DH, DM) for
multipole tracers (qparqper: LRG/ELG); we mask the unused channels.

Source helpers come from fisher_sigmas.py.

Usage (from bao/):
    ~/miniconda3/envs/emulator/bin/python plot_fisher_vs_desi.py
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from fisher_sigmas import (
    _prod_fisher_sigmas, _bundle_fisher_sigmas, _recon_sigmas, _TRACERS,
)
from util import TRACER_CONFIGS

_DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
_QUANTITIES = [
    ("DH_over_rs", "tab:blue",   r"$\sigma(D_H/r_d)$"),
    ("DM_over_rs", "tab:orange", r"$\sigma(D_M/r_d)$"),
    ("DV_over_rs", "tab:green",  r"$\sigma(D_V/r_d)$"),
]
_QUANT_SHORT = {"DH_over_rs": "DH/rd", "DM_over_rs": "DM/rd", "DV_over_rs": "DV/rd"}
_FID_OM, _FID_HRDRAG = 0.3152, 99.08
_HERE = Path(__file__).resolve().parent


def _load_prod_mcmc(tracer):
    """Production MCMC (analytic cov). Schema: sigma_*_over_rd_mcmc."""
    p = _HERE / "mcmc_results" / f"{tracer}_dr1.json"
    if not p.exists():
        return {q: float("nan") for q, _, _ in _QUANTITIES}
    d = json.loads(p.read_text())
    return {
        "DH_over_rs": float(d.get("sigma_DH_over_rd_mcmc", float("nan"))),
        "DM_over_rs": float(d.get("sigma_DM_over_rd_mcmc", float("nan"))),
        "DV_over_rs": float(d.get("sigma_DV_over_rd_mcmc", float("nan"))),
    }


def _load_bundle_mcmc(tracer):
    """Bundle-cov MCMC (native ξ-theory + bundle cov + BB-DESI). Schema: sig_*."""
    suffix = "_qiso" if tracer in ("BGS", "QSO") else ""
    p = _HERE / "mcmc_results_bundle_native" / f"{tracer}_dr1_native_theory{suffix}_bbdesi_adame24.json"
    if not p.exists():
        return {q: float("nan") for q, _, _ in _QUANTITIES}
    d = json.loads(p.read_text())
    return {
        "DH_over_rs": float(d.get("sig_DH", float("nan"))),
        "DM_over_rs": float(d.get("sig_DM", float("nan"))),
        "DV_over_rs": float(d.get("sig_DV", float("nan"))),
    }


_Q_KEY = {"DH_over_rs": "DH", "DM_over_rs": "DM", "DV_over_rs": "DV"}


def _load_chain_noise():
    p = _HERE / "seed_sweep" / "sigma_of_sigma.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out = {}
    for tracer, by_cov in raw.items():
        out[tracer] = {}
        for cov, by_q in by_cov.items():
            out[tracer][cov] = {q: (entry["mean"], entry["std"]) for q, entry in by_q.items()}
    return out


def _gather():
    out = {}
    noise = _load_chain_noise()
    for t in _TRACERS:
        print(f"== {t} ==", flush=True)
        prod = _prod_fisher_sigmas(t)
        bundle = _bundle_fisher_sigmas(t)
        recon = _recon_sigmas(t, prod)
        prod_mcmc = _load_prod_mcmc(t)
        bundle_mcmc = _load_bundle_mcmc(t)
        is_sparse = bundle.get("apmode") == "qiso"

        # If seed-sweep results exist, replace MCMC σ with the seed-mean
        # and collect per-quantity σ-of-σ for error bars.
        prod_mcmc_err = {q: 0.0 for q, _, _ in _QUANTITIES}
        bundle_mcmc_err = {q: 0.0 for q, _, _ in _QUANTITIES}
        for q, _, _ in _QUANTITIES:
            qk = _Q_KEY[q]
            if t in noise and "prod" in noise[t] and qk in noise[t]["prod"]:
                mu, sd = noise[t]["prod"][qk]
                prod_mcmc[q] = mu
                prod_mcmc_err[q] = sd
            if t in noise and "bundle" in noise[t] and qk in noise[t]["bundle"]:
                mu, sd = noise[t]["bundle"][qk]
                bundle_mcmc[q] = mu
                bundle_mcmc_err[q] = sd

        if is_sparse:
            for src in (recon, prod, bundle, prod_mcmc, bundle_mcmc,
                        prod_mcmc_err, bundle_mcmc_err):
                src["DH_over_rs"] = float("nan")
                src["DM_over_rs"] = float("nan")
        else:
            for src in (recon, prod, bundle, prod_mcmc, bundle_mcmc,
                        prod_mcmc_err, bundle_mcmc_err):
                src["DV_over_rs"] = float("nan")

        out[t] = {"prod": prod, "bundle": bundle, "recon": recon,
                  "prod_mcmc": prod_mcmc, "bundle_mcmc": bundle_mcmc,
                  "prod_mcmc_err": prod_mcmc_err,
                  "bundle_mcmc_err": bundle_mcmc_err,
                  "sparse": is_sparse}
        for src_name, src in [("prod", prod), ("bundle", bundle), ("recon", recon),
                              ("prod_mcmc", prod_mcmc), ("bundle_mcmc", bundle_mcmc)]:
            print(f"  {src_name:<12} DH={src['DH_over_rs']:.4f}  DM={src['DM_over_rs']:.4f}  DV={src['DV_over_rs']:.4f}")
    return out


def _print_table(data):
    print(f"\n=== DR1 σ: Fisher (P/B) + MCMC (Pm/Bm) vs DESI bao-recon (D) ===")
    header = (f"  {'tracer':<12} {'q':<6} {'σ_P':>9} {'σ_Pm':>9} {'σ_B':>9} "
              f"{'σ_Bm':>9} {'σ_D':>9} {'P/D':>7} {'Pm/D':>7} {'B/D':>7} {'Bm/D':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in _TRACERS:
        d = data[t]
        for q, _, _ in _QUANTITIES:
            sD = d["recon"][q]
            if not np.isfinite(sD): continue
            sP  = d["prod"][q]; sB = d["bundle"][q]
            sPm = d["prod_mcmc"][q]; sBm = d["bundle_mcmc"][q]
            r = lambda x: x / sD if sD > 0 else np.nan
            print(f"  {t:<12} {_QUANT_SHORT[q]:<6} "
                  f"{sP:>9.4f} {sPm:>9.4f} {sB:>9.4f} {sBm:>9.4f} {sD:>9.4f} "
                  f"{r(sP):>7.3f} {r(sPm):>7.3f} {r(sB):>7.3f} {r(sBm):>7.3f}")


def _plot(data, out_path):
    tracers = list(_TRACERS)
    display = [_DISPLAY.get(t, t) for t in tracers]
    x = np.arange(len(tracers), dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             constrained_layout=True)

    for ax, (q, _, ylabel) in zip(axes, _QUANTITIES):
        for xi in x:
            ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)

        sources = {"recon": [], "prod": [], "bundle": [], "prod_mcmc": [],
                   "bundle_mcmc": []}
        errors = {"prod_mcmc": [], "bundle_mcmc": []}
        for i, t in enumerate(tracers):
            for s in sources:
                v = data[t][s][q]
                if np.isfinite(v):
                    sources[s].append((i, v))
            for s in errors:
                err_key = s + "_err"
                if err_key in data[t]:
                    e = data[t][err_key][q]
                    if np.isfinite(e):
                        errors[s].append((i, e))

        def _xy(name):
            if not sources[name]: return [], []
            xs, ys = zip(*sources[name])
            return list(xs), list(ys)

        def _err(name):
            if not errors[name]: return []
            return [e for _, e in errors[name]]

        c_ana, c_bun = "tab:blue", "tab:orange"
        dx, dy = _xy("recon")
        ax.scatter(dx, dy, marker="x", s=40, color="black",
                   linewidths=1.5, zorder=2)
        px, py = _xy("prod")
        ax.scatter(px, py, marker="o", s=20, color=c_ana,
                   linewidth=0, zorder=4)
        pmx, pmy = _xy("prod_mcmc")
        pme = _err("prod_mcmc")
        if pme:
            ax.errorbar(pmx, pmy, yerr=pme, fmt="none",
                        ecolor=c_ana, elinewidth=1.5, capsize=4, capthick=1.5,
                        zorder=3)
        ax.scatter(pmx, pmy, marker="^", s=20, color=c_ana,
                   linewidth=0, zorder=4)
        bx, by = _xy("bundle")
        ax.scatter(bx, by, marker="s", s=20, color=c_bun,
                   linewidth=0, zorder=4)
        bmx, bmy = _xy("bundle_mcmc")
        bme = _err("bundle_mcmc")
        if bme:
            ax.errorbar(bmx, bmy, yerr=bme, fmt="none",
                        ecolor=c_bun, elinewidth=1.5, capsize=4, capthick=1.5,
                        zorder=3)
        ax.scatter(bmx, bmy, marker="D", s=16, color=c_bun,
                   linewidth=0, zorder=4)

        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 0.9)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")

    source_handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=6,
               markeredgewidth=1.5, color="black", label="DESI (bao-recon)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor="tab:blue", markeredgecolor="none",
               label="Fisher (analytic cov)"),
        Line2D([0], [0], marker="^", linestyle="", markersize=7,
               markerfacecolor="tab:blue", markeredgecolor="none",
               label="MCMC (analytic cov)"),
        Line2D([0], [0], marker="s", linestyle="", markersize=7,
               markerfacecolor="tab:orange", markeredgecolor="none",
               label="Fisher (bundle cov)"),
        Line2D([0], [0], marker="D", linestyle="", markersize=6,
               markerfacecolor="tab:orange", markeredgecolor="none",
               label="MCMC (bundle cov)"),
    ]
    axes[0].legend(handles=source_handles, loc="upper left",
                   bbox_to_anchor=(1.01, 1.0),
                   title="Source", fontsize=9, title_fontsize=10)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(display, rotation=20)
    axes[-1].set_xlabel("Tracer bin")
    axes[0].set_title(
        f"BAO σ — Fisher + MCMC vs DESI bao-recon  |  DR1  |  "
        f"Om={_FID_OM:.4g}, hrdrag={_FID_HRDRAG:.4g}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    data = _gather()
    _print_table(data)
    out_path = _HERE / f"fisher_vs_desi_dr1_Om_{_FID_OM:.5f}_hrdrag_{_FID_HRDRAG:.5f}.png"
    _plot(data, out_path)


if __name__ == "__main__":
    main()
