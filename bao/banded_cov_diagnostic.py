"""Diagnostic: how much of the bundle ξ-cov off-diagonal structure is needed
to recover the full-cov σ_q?

For each DR1 tracer, we re-run the bundle-cov Fisher under a *Gaussian
taper* of the off-diagonals:

    C_tapered[i, j] = C_bundle[i, j] * exp(-(i - j)^2 / (2 ℓ^2))

ℓ → 0  : diagonal-only (PSD: variances only)
ℓ → ∞  : recovers the full bundle cov

A Gaussian kernel is PSD, and the Hadamard product of PSD with PSD is PSD
(Schur's product theorem), so every tapered C is positive-definite. We
sweep ℓ ∈ [0.3, ∞], record σ(DH/rd), σ(DM/rd), σ(DV/rd), and plot
σ_ℓ / σ_full vs ℓ.

If σ saturates at small ℓ (≲2 bins), the missing structure is short-range
(likely window mode coupling). If it requires large ℓ (≳10 bins), the
structure is long-range (likely reconstruction-induced bin-to-bin
correlations from R_sm-smoothed displacement field).

Output:
  bao/banded_cov_diagnostic_dr1.csv   (per-tracer per-ℓ σ values)
  bao/banded_cov_diagnostic_dr1.png   (per-tracer σ/σ_full vs ℓ)

Usage (from bao/):
    ~/miniconda3/envs/emulator/bin/python banded_cov_diagnostic.py
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from compare_pipeline_theory_to_bundle import load_bundle
from fisher_sigmas import _bundle_fisher_sigmas, _TRACERS

_HERE = Path(__file__).resolve().parent


def _taper_cov(cov, length_scale):
    """Hadamard-multiply cov by a Gaussian kernel exp(-(i-j)^2 / (2 ℓ^2)).

    length_scale=0       → diagonal only (kernel = Kronecker δ_ij)
    length_scale=np.inf  → returns cov unchanged
    """
    if not np.isfinite(length_scale):
        return cov.copy()
    n = cov.shape[0]
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    if length_scale <= 0:
        K = (i == j).astype(float)
    else:
        K = np.exp(-((i - j) ** 2) / (2.0 * length_scale ** 2))
    return cov * K


def _length_scales():
    return [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 35.0,
            60.0, np.inf]


def main():
    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    axes = axes.ravel()

    scales = _length_scales()
    for ax, tracer in zip(axes, _TRACERS):
        print(f"\n=== {tracer} ===", flush=True)
        bundle = load_bundle(tracer)
        n_data = bundle["cov"].shape[0]
        print(f"  data vector size = {n_data}", flush=True)

        ref = _bundle_fisher_sigmas(tracer)
        apmode = ref["apmode"]
        print(f"  apmode = {apmode}", flush=True)
        keys = (("DV_over_rs",) if apmode == "qiso"
                else ("DH_over_rs", "DM_over_rs"))
        ref_sigmas = {k: ref[k] for k in keys}
        print("  full-cov σ:", {k: f"{v:.4f}" for k, v in ref_sigmas.items()}, flush=True)

        ratios_by_key = {k: [] for k in keys}
        for ls in scales:
            try:
                cov_t = _taper_cov(bundle["cov"], ls)
                res = _bundle_fisher_sigmas(tracer, cov_override=cov_t)
                for k in keys:
                    r = res[k] / ref_sigmas[k]
                    ratios_by_key[k].append(r)
                    rows.append({"tracer": tracer, "length_scale": ls, "key": k,
                                 "sigma": res[k], "ratio_vs_full": r})
                tag = "full" if not np.isfinite(ls) else f"ℓ={ls:>5.2f}"
                print(f"  {tag}  " + "  ".join(
                    f"{k.split('_')[0]}/full={res[k]/ref_sigmas[k]:.3f}"
                    for k in keys), flush=True)
            except np.linalg.LinAlgError as e:
                print(f"  ℓ={ls}  LinAlgError ({e})", flush=True)
                for k in keys:
                    ratios_by_key[k].append(float("nan"))
                    rows.append({"tracer": tracer, "length_scale": ls, "key": k,
                                 "sigma": float("nan"), "ratio_vs_full": float("nan")})

        # Plot vs ℓ (replace ∞ with a finite value for plotting at the right edge)
        x = np.array([1e3 if not np.isfinite(s) else s for s in scales])
        colors = {"DH_over_rs": "tab:blue", "DM_over_rs": "tab:orange",
                  "DV_over_rs": "tab:green"}
        labels = {"DH_over_rs": r"$\sigma(D_H/r_d)$",
                  "DM_over_rs": r"$\sigma(D_M/r_d)$",
                  "DV_over_rs": r"$\sigma(D_V/r_d)$"}
        for k in keys:
            ax.plot(x, ratios_by_key[k], "o-", color=colors[k],
                    linewidth=1.5, label=labels[k])
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xscale("symlog", linthresh=0.3)
        ax.set_xlabel("Gaussian taper ℓ (data-vector bin units; rightmost = full)")
        ax.set_ylabel(r"$\sigma_\ell / \sigma_{\rm full}$")
        ax.set_title(tracer)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        ax.legend(fontsize=8, loc="best")

    out_csv = _HERE / "banded_cov_diagnostic_dr1.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved {out_csv}")
    out_png = _HERE / "banded_cov_diagnostic_dr1.png"
    fig.suptitle("Bundle-cov σ_q vs off-diagonal bandwidth — DR1", fontsize=13)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
