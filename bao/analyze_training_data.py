"""Physical sanity checks for the config_space emulator training data + a plot.

For each tracer:
  1. Power-law fit  log σ = β0 + βN·logN + βOm·logOm + βh·log(hrd).  βN is the
     N_tracers scaling exponent (σ ∝ N^βN): 0 = cosmic-variance-limited, -1 =
     deep shot-noise, -0.5 the midpoint. βh≈0 means hrd carries no σ signal.
  2. Fiducial check: predict σ at (passed, Om=0.3152, hrd=99.08) via a local
     log-linear fit over the nearest LHS samples, vs the config-space MCMC at the
     same fiducial (mcmc_config.json gaussxi) and DESI bao-recon. With the v2
     passed-centred N box, the fiducial is interpolation for every tracer.

Run (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python analyze_training_data.py [--data-dir DIR]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

_BAO = Path("/home/ashandonay/desilike-emulator/bao")
sys.path.insert(0, str(_BAO))
sys.path.insert(0, str(_BAO.parent))

import config_space as cc          # noqa: E402
import desi_reference as desi_ref      # noqa: E402
from util import TRACER_CONFIGS    # noqa: E402

_DEFAULT_DIR = "/home/ashandonay/scratch/bedcosmo/num_tracers/emulator/training_data/bao/dr1/base/config/v1"
TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
FID_OM, FID_HRD = cc._FID["Om"], cc._FID["hrdrag"]
TGT = ["sigma_DH_over_rd", "sigma_DM_over_rd", "sigma_DV_over_rd"]
_KEY = {"sigma_DH_over_rd": "DH", "sigma_DM_over_rd": "DM", "sigma_DV_over_rd": "DV"}
_C = 299792.458  # km/s


def _load(data_dir, tracer):
    tr = np.load(Path(data_dir) / f"{tracer}_train.npz", allow_pickle=True)
    te = np.load(Path(data_dir) / f"{tracer}_test.npz", allow_pickle=True)
    x = np.vstack([tr["x"], te["x"]])
    y = np.vstack([tr["y"], te["y"]])
    return x, y, list(tr["target_names"])


def _powerlaw(x, y):
    """log y = β0 + βN logN + βOm logOm + βh log hrd -> (βN, βOm, βh, R²)."""
    L = np.log(x)
    A = np.column_stack([np.ones(len(L)), L[:, 0], L[:, 1], L[:, 2]])
    ly = np.log(y)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    r2 = 1.0 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return coef[1], coef[2], coef[3], r2


def _predict_fiducial(x, y, fid, k):
    """Local log-linear prediction of y at `fid` using the k nearest samples."""
    L = np.log(x)
    mu, sd = L.mean(0), L.std(0)
    Z = (L - mu) / sd
    zf = (np.log(fid) - mu) / sd
    idx = np.argsort(np.linalg.norm(Z - zf, axis=1))[:k]
    A = np.column_stack([np.ones(k), L[idx]])
    coef, *_ = np.linalg.lstsq(A, np.log(y[idx]), rcond=None)
    return np.exp(np.array([1.0, *np.log(fid)]) @ coef)


def _fid_distances(z, Om, hrdrag):
    """Fiducial DH/rd, DM/rd, DV/rd for flat LCDM (base model)."""
    E = lambda zz: np.sqrt(Om * (1 + zz) ** 3 + (1 - Om))
    DH = (_C / 100.0) / (E(z) * hrdrag)
    I, _ = quad(lambda zz: 1.0 / E(zz), 0.0, z)
    DM = (_C / 100.0) * I / hrdrag
    DV = (z * DM * DM * DH) ** (1.0 / 3.0)
    return DH, DM, DV


def analyze(data_dir):
    mcmc = json.loads((_BAO / "mcmc_config.json").read_text())
    rows = {}   # tracer -> dict of results
    for t in TRACERS:
        x, y, tnames = _load(data_dir, t)
        k = min(200, len(x) // 3)
        passed = cc._get_ntracers(t)
        z = float(TRACER_CONFIGS[t]["z_eff"])
        lo, hi = x[:, 0].min(), x[:, 0].max()
        coeffs = {tg: _powerlaw(x, y[:, j]) for j, tg in enumerate(tnames)}
        fid = np.array([passed, FID_OM, FID_HRD])
        pred = _predict_fiducial(x, y, fid, k)
        dv_emu = pred[tnames.index("sigma_DV_over_rd")]
        # config MCMC gaussxi (mean over seeds) at the same fiducial
        per = mcmc.get(t, {}).get("gaussxi", {})
        dvs = [s["DV"] for s in per.values() if s.get("DV") and np.isfinite(s["DV"])]
        dv_mcmc = float(np.mean(dvs)) if dvs else np.nan
        # DESI recon at the analytic fiducial distances
        DHf, DMf, DVf = _fid_distances(z, FID_OM, FID_HRD)
        recon = desi_ref._recon_sigmas(t, {"DH_over_rd_fid": DHf, "DM_over_rd_fid": DMf,
                                     "DV_over_rd_fid": DVf})
        dv_recon = recon.get("DV_over_rs", np.nan)
        rows[t] = dict(coeffs=coeffs, passed=passed, pos=(passed - lo) / (hi - lo),
                       dv_emu=dv_emu, dv_mcmc=dv_mcmc, dv_recon=dv_recon, n=len(x))
    return rows


def report(rows):
    print("=" * 72)
    print("POWER-LAW  log σ = β0 + βN·logN + βOm·logOm + βh·log(hrd)")
    print(f"{'tracer':11s} {'tgt':3s} {'βN':>7s} {'βOm':>7s} {'βhrd':>7s} {'R²':>6s}")
    print("-" * 44)
    for t in TRACERS:
        for tg in TGT:
            bN, bOm, bh, r2 = rows[t]["coeffs"][tg]
            print(f"{t:11s} {_KEY[tg]:3s} {bN:7.3f} {bOm:7.3f} {bh:7.3f} {r2:6.3f}")
        print()
    print("=" * 72)
    print("FIDUCIAL σ(DV/rd) at passed (box pos) — emu vs config-MCMC vs recon")
    print(f"{'tracer':11s} {'pos':>5s} {'emu':>8s} {'mcmc':>8s} {'emu/mcmc':>8s} {'recon':>8s}")
    print("-" * 52)
    for t in TRACERS:
        r = rows[t]
        ratio = r["dv_emu"] / r["dv_mcmc"] if np.isfinite(r["dv_mcmc"]) else np.nan
        print(f"{t:11s} {r['pos']:5.2f} {r['dv_emu']:8.4f} {r['dv_mcmc']:8.4f} "
              f"{ratio:8.3f} {r['dv_recon']:8.4f}")


def plot(rows, out_path):
    xs = np.arange(len(TRACERS))
    labels = [DISPLAY.get(t, t) for t in TRACERS]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    # Panel 1: N-scaling exponent βN (DH/DM/DV grouped) + physical band.
    ax = axes[0]
    ax.axhspan(-1.0, 0.0, color="green", alpha=0.06)
    ax.axhline(0.0, color="gray", lw=0.8); ax.axhline(-1.0, color="gray", lw=0.8)
    ax.axhline(-0.5, color="k", ls="--", lw=1.0, label=r"$-0.5$ (shot/CV midpoint)")
    w = 0.25
    for i, tg in enumerate(TGT):
        vals = [rows[t]["coeffs"][tg][0] for t in TRACERS]
        ax.bar(xs + (i - 1) * w, vals, w, label=_KEY[tg])
    ax.set_ylabel(r"$\beta_N$   ($\sigma \propto N_{\rm tracers}^{\beta_N}$)")
    ax.set_title("N-tracers scaling: 0 = cosmic-variance-limited,  −1 = deep shot-noise")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.legend(ncol=4, fontsize=9, loc="lower left")

    # Panel 2: βOm vs βhrd (hrd ≈ null) — DV target.
    ax = axes[1]
    bOm = [rows[t]["coeffs"]["sigma_DV_over_rd"][1] for t in TRACERS]
    bh = [rows[t]["coeffs"]["sigma_DV_over_rd"][2] for t in TRACERS]
    ax.bar(xs - 0.2, bOm, 0.4, label=r"$\beta_{\Omega_m}$", color="tab:purple")
    ax.bar(xs + 0.2, bh, 0.4, label=r"$\beta_{h r_d}$", color="tab:gray")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("fit coefficient (DV target)")
    ax.set_title(r"$\Omega_m$ dominates;  $h r_d$ carries ~no signal ($\beta_{hr_d}\approx 0$)")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.legend(fontsize=10)

    # Panel 3: fiducial σ(DV/rd) emulator vs config-MCMC vs DESI recon.
    ax = axes[2]
    emu = [rows[t]["dv_emu"] for t in TRACERS]
    mc = [rows[t]["dv_mcmc"] for t in TRACERS]
    rc = [rows[t]["dv_recon"] for t in TRACERS]
    ax.scatter(xs, emu, s=70, marker="o", color="tab:blue", label="emulator data (v1, new)", zorder=3)
    ax.scatter(xs, mc, s=70, marker="s", color="tab:orange", label="config MCMC (gaussxi)", zorder=3)
    ax.scatter(xs, rc, s=70, marker="x", color="black", label="DESI bao-recon", zorder=3)
    for i, t in enumerate(TRACERS):
        if np.isfinite(mc[i]) and mc[i] > 0:
            ax.annotate(f"{emu[i]/mc[i]:.2f}", (xs[i], max(emu[i], mc[i])),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_ylabel(r"$\sigma(D_V/r_d)$ at fiducial")
    ax.set_title("Fiducial validation (passed-centred box): emu/MCMC ratio annotated")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylim(0, None)
    ax.legend(fontsize=9)

    fig.suptitle("Config-space emulator training data — physical sanity checks", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=_DEFAULT_DIR)
    ap.add_argument("--out", default=str(_BAO / "training_data_diagnostics.png"))
    args = ap.parse_args()
    print(f"Data: {args.data_dir}")
    rows = analyze(args.data_dir)
    report(rows)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
