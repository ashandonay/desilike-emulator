"""Emulator σ vs DESI bao-recon vs re-computed config-space pipeline σ.

A stripped-down sibling of comparison_plots.py:forecast. For each DR1 tracer it
plots three σ(D/rd) series at the NOMINAL DESI input (N_tracers from the DR1 box,
fiducial Om / h·r_d):

  - DESI (bao-recon)   : the published stat-only σ (rs._recon_sigmas), black ×
  - Fisher analytic cov: the config-space pipeline re-run here from scratch
                         (XiSigmaGenerator.sigma_triplet — the emulator's training
                         target), blue ○
  - emulator           : the trained NN's prediction at the same input, read from
                         the downloaded .pt checkpoints, red ◆

The emulator and the analytic-cov dots should sit on top of each other (the NN is
a surrogate for that exact pipeline); the gap to the DESI × is the physical
non-Gaussian / α_SN residual the Gaussian forecast does not carry.

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python emulator_vs_desi.py
    ... emulator_vs_desi.py --models-dir /path/to/models/dr1/base
"""
from __future__ import annotations
import argparse, os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import config_space as cc
import desi_reference as rs
from util import build_model, decode_emulator_outputs

_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
_HERE = Path(__file__).resolve().parent
_DEFAULT_MODELS = Path.home() / "scratch/bedcosmo/num_tracers/emulator/models/dr1/base/config/v4"

_QUANTITIES = [
    ("DH_over_rs", r"$\sigma(D_H/r_d)$"),
    ("DM_over_rs", r"$\sigma(D_M/r_d)$"),
    ("DV_over_rs", r"$\sigma(D_V/r_d)$"),
]
_QUANT_SHORT = {"DH_over_rs": "DH/rd", "DM_over_rs": "DM/rd", "DV_over_rs": "DV/rd",
                "rho_DH_DM": "rho"}
# Panels, top→bottom: σ(DH), σ(DM), ρ(DH,DM), σ(DV). ρ is the off-diagonal
# correlation carried only by the anisotropic tracers (is_corr → signed y-axis).
_PANELS = [
    ("DH_over_rs", r"$\sigma(D_H/r_d)$", False),
    ("DM_over_rs", r"$\sigma(D_M/r_d)$", False),
    ("rho_DH_DM", r"$\rho(D_H, D_M)$", True),
    ("DV_over_rs", r"$\sigma(D_V/r_d)$", False),
]
# emulator output names → our quantity keys (per-tracer targets, incl. ρ for aniso)
_TGT_TO_Q = {"sigma_DH_over_rd": "DH_over_rs",
             "sigma_DM_over_rd": "DM_over_rs",
             "sigma_DV_over_rd": "DV_over_rs",
             "rho_DH_DM": "rho_DH_DM"}


def _is_sparse(tracer):
    return tracer in cc._ISO_TRACERS if hasattr(cc, "_ISO_TRACERS") else tracer in ("BGS", "QSO")


def _emulator_predict(model_path, tracer):
    """Run the trained NN at the nominal DESI input for this tracer. Mirrors the
    standardize -> predict -> unstandardize (+ optional symlog inverse) path in
    eval.run_eval. Returns {DH_over_rs, DM_over_rs, DV_over_rs}."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(
        analysis=ckpt.get("analysis", "bao"),
        architecture=ckpt.get("architecture", "resnet"),
        in_dim=len(ckpt["param_names"]),
        out_dim=len(ckpt["target_names"]),
        hidden_dim=ckpt["hidden_dim"],
        n_hidden=ckpt["n_hidden"],
        dropout=ckpt.get("dropout", 0.0),
        expand=ckpt.get("expand", 4),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build the input row in the checkpoint's param order from the nominal input:
    # N_tracers from the DR1 box, everything else at the fiducial (_FID/defaults).
    src = {"N_tracers": float(cc._get_ntracers(tracer)), **cc._FID}
    x_raw = np.array([[src[p] for p in ckpt["param_names"]]], dtype=np.float32)

    x_mu = ckpt["x_mu"].cpu().numpy(); x_sigma = ckpt["x_sigma"].cpu().numpy()
    y_mu = ckpt["y_mu"].cpu().numpy(); y_sigma = ckpt["y_sigma"].cpu().numpy()
    x_norm = (x_raw - x_mu) / x_sigma
    with torch.no_grad():
        y_pred_norm = model(torch.from_numpy(x_norm)).cpu().numpy()
    # Invert z-score + per-target transforms (tanh for rho, exp/symlog for sigma)
    # via the shared helper so this stays consistent with eval.run_eval.
    y_linthresh = ckpt.get("y_linthresh")
    if y_linthresh is not None:
        y_linthresh = y_linthresh.cpu().numpy()
    y_pred = decode_emulator_outputs(
        y_pred_norm, y_mu, y_sigma, list(ckpt["target_names"]),
        log_normalize=ckpt.get("log_normalize", False),
        y_linthresh=y_linthresh,
    ).ravel()
    return {_TGT_TO_Q[t]: float(v) for t, v in zip(ckpt["target_names"], y_pred)}


def _gather(models_dir):
    out = {}
    for t in _TRACERS:
        print(f"== {t} ==", flush=True)
        gen = cc.XiSigmaGenerator(t)
        # Re-compute the pipeline at the nominal DESI input (fiducial cosmology,
        # DR1 N_tracers). This also gives the (D/rd)_fid used to scale the recon σ.
        pipeline = gen.sigma_triplet(N_tracers=cc._get_ntracers(t))
        recon = rs._recon_sigmas(t, pipeline)
        recon["rho_DH_DM"] = rs._recon_rho(t)

        mp = Path(models_dir) / f"{t}.pt"
        emu = _emulator_predict(mp, t) if mp.exists() else {q: float("nan") for q, _ in _QUANTITIES}
        if not mp.exists():
            print(f"  WARNING: no checkpoint at {mp}")

        # Display mask: sparse tracers carry only DV (no DH/DM, no ρ); anisotropic
        # carry DH+DM+ρ (no standalone DV emulator output).
        mask = ["DH_over_rs", "DM_over_rs", "rho_DH_DM"] if _is_sparse(t) else ["DV_over_rs"]
        for src in (recon, pipeline, emu):
            for q in mask:
                src[q] = float("nan")
            src.setdefault("rho_DH_DM", float("nan"))

        out[t] = {"recon": recon, "pipeline": pipeline, "emu": emu}
        for nm in ("recon", "pipeline", "emu"):
            s = out[t][nm]
            print(f"  {nm:<9} DH={s['DH_over_rs']:.4f}  DM={s['DM_over_rs']:.4f}  "
                  f"DV={s['DV_over_rs']:.4f}  rho={s['rho_DH_DM']:.4f}")
    return out


def _print_table(data):
    print("\n=== DR1 config-space σ: pipeline (P) + emulator (E) vs DESI (D) ===")
    header = (f"  {'tracer':<11} {'q':<6} {'σ_P':>9} {'σ_E':>9} {'σ_D':>9} "
              f"{'P/D':>6} {'E/D':>6} {'E/P':>6}")
    print(header); print("  " + "-" * (len(header) - 2))
    for t in _TRACERS:
        d = data[t]
        for q in ("DH_over_rs", "DM_over_rs", "rho_DH_DM", "DV_over_rs"):
            sD = d["recon"][q]
            if not np.isfinite(sD):
                continue
            sP, sE = d["pipeline"][q], d["emu"][q]
            # ρ is signed → ratios are meaningless; show the absolute difference instead.
            if q == "rho_DH_DM":
                r = lambda x, y: x - y if (np.isfinite(x) and np.isfinite(y)) else np.nan
            else:
                r = lambda x, y: x / y if (np.isfinite(x) and y > 0) else np.nan
            print(f"  {t:<11} {_QUANT_SHORT[q]:<6} {sP:>9.4f} {sE:>9.4f} {sD:>9.4f} "
                  f"{r(sP, sD):>6.3f} {r(sE, sD):>6.3f} {r(sE, sP):>6.3f}")


def _plot(data, out_path):
    tracers = list(_TRACERS)
    display = [_DISPLAY.get(t, t) for t in tracers]
    x = np.arange(len(tracers), dtype=float)
    c_pipe, c_emu = "tab:blue", "tab:red"

    fig, axes = plt.subplots(len(_PANELS), 1, figsize=(11, 14), sharex=True,
                             constrained_layout=True)
    for ax, (q, ylabel, is_corr) in zip(axes, _PANELS):
        for xi in x:
            ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)

        def _xy(src):
            xs, ys = [], []
            for i, t in enumerate(tracers):
                v = data[t][src][q]
                if np.isfinite(v):
                    xs.append(i); ys.append(v)
            return xs, ys

        ax.scatter(*_xy("recon"), marker="x", s=45, color="black",
                   linewidths=1.6, zorder=5)
        # Fisher pipeline and emulator share the marker (circle); color is the
        # only distinction — open blue = pipeline, filled red = emulator.
        ax.scatter(*_xy("pipeline"), marker="o", s=52, facecolor="none",
                   edgecolor=c_pipe, linewidths=1.6, zorder=3)
        ax.scatter(*_xy("emu"), marker="o", s=22, color=c_emu,
                   linewidth=0, zorder=4)

        ax.set_ylabel(ylabel)
        if not is_corr:
            ax.set_ylim(0.0, None)
        # ρ (is_corr) is signed and clustered well away from 0; let it autoscale
        # tightly around the data rather than padding down to zero.
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")

    handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=7, markeredgewidth=1.6,
               color="black", label="DESI (bao-recon)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor="none",
               markeredgecolor=c_pipe, markeredgewidth=1.6, label="Fisher · analytic cov (pipeline)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=c_emu, markeredgecolor="none", label="Emulator (NN)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   title="Source (config-space)", fontsize=9, title_fontsize=10)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(display, rotation=20)
    axes[-1].set_xlabel("Tracer bin")
    axes[0].set_title(
        "Config-space BAO σ — emulator vs re-computed analytic-cov pipeline "
        "vs DESI bao-recon  |  DR1", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(_DEFAULT_MODELS),
                    help="directory of per-tracer .pt checkpoints (default: dr1/base)")
    ap.add_argument("--out", default=str(_HERE / "emulator_vs_desi_dr1.png"))
    args = ap.parse_args()

    data = _gather(args.models_dir)
    _print_table(data)
    _plot(data, args.out)


if __name__ == "__main__":
    main()
