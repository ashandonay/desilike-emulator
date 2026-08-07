"""Emulator σ vs DESI bao-recon vs re-computed config-space pipeline σ.

A stripped-down sibling of comparison_plots.py:forecast. For each DR1 tracer it
plots three σ(D/rd) series at the NOMINAL DESI input (N_tracers from the DR1 box,
fiducial Om / h·r_d):

  - DESI (bao-recon)   : the published stat-only σ (rs._recon_sigmas), black ×
  - pipeline NOW       : the config-space pipeline re-run here from scratch
                         (XiSigmaGenerator.sigma_triplet — the emulator's training
                         target), blue ○, dodged RIGHT
  - pipeline BEFORE    : the same quantity from an archived regress_sigmas dump
                         (--baseline), orange ○, dodged LEFT
  - emulator           : the trained NN at the same input, from the .pt
                         checkpoints, green ●, drawn ON the BEFORE position

Reading it (S94) -- and the labels matter, because two very different
intervals are on the same figure:

  orange -> blue   is S81 -> S92 ONLY (one day's work: the NX volume-ratio fix
                   and the config-space density move). Small except ELG2.
  green            is emulator v4, trained on data generated 2026-06-19 --
                   155 commits and S52-S81 older than the orange series. Its
                   offset is therefore NOT the pending regeneration's delta;
                   the regeneration has to close green -> blue, which spans
                   S52-S92, not just the visible orange->blue gap.

The residual gap to the DESI × is the physical non-Gaussian / α_SN term the
Gaussian forecast does not carry, and no regeneration closes it.

The BEFORE series is read from a dump rather than recomputed: `config/{tracer}/
fid` in a regress dump IS this quantity (N_factor 1.0 = the DR1 passed count, at
the fiducial), so no old checkout is needed to draw history.

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
from util import build_model, decode_emulator_outputs, plots_dir

_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}
_HERE = Path(__file__).resolve().parent
_DEFAULT_MODELS = Path.home() / "scratch/bedcosmo/num_tracers/emulator/bao/models/dr1/base/config/v4"

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


def _load_baseline(path):
    """Pre-change pipeline sigma from a regress_sigmas dump (S94).

    The dump's `config/{tracer}/fid` cell is EXACTLY what this plot recomputes:
    `XiSigmaGenerator.sigma_triplet` at N_factor 1.0 (= the DR1 passed count) and
    the fiducial cosmology. So an archived dump is a drop-in "before" series and
    no old code has to be checked out to draw one.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: baseline dump not found at {p} -- drawing without it")
        return None
    d = np.load(p)
    out = {}
    for t in _TRACERS:
        vals = {}
        for q, _ in _QUANTITIES:
            k = f"config/{t}/fid/{q}"
            vals[q] = float(d[k]) if k in d.files else float("nan")
        k_rho = f"config/{t}/fid/rho_DH_DM"
        vals["rho_DH_DM"] = float(d[k_rho]) if k_rho in d.files else float("nan")
        out[t] = vals
    return out


def _gather(models_dir, baseline=None):
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
        prev = dict(baseline[t]) if baseline is not None else {
            q: float("nan") for q, _ in _QUANTITIES}
        for src in (recon, pipeline, emu, prev):
            for q in mask:
                src[q] = float("nan")
            src.setdefault("rho_DH_DM", float("nan"))

        out[t] = {"recon": recon, "pipeline": pipeline, "emu": emu, "prev": prev}
        for nm in ("recon", "prev", "pipeline", "emu"):
            s = out[t][nm]
            print(f"  {nm:<9} DH={s['DH_over_rs']:.4f}  DM={s['DM_over_rs']:.4f}  "
                  f"DV={s['DV_over_rs']:.4f}  rho={s['rho_DH_DM']:.4f}")
    return out


def _print_table(data):
    print("\n=== DR1 config-space σ: pipeline (P) + emulator (E) vs DESI (D) ===")
    header = (f"  {'tracer':<11} {'q':<6} {'σ_pre':>9} {'σ_now':>9} {'σ_E':>9} "
              f"{'σ_D':>9} {'now/pre':>8} {'now/D':>6} {'E/pre':>6}")
    print(header); print("  " + "-" * (len(header) - 2))
    for t in _TRACERS:
        d = data[t]
        for q in ("DH_over_rs", "DM_over_rs", "rho_DH_DM", "DV_over_rs"):
            sD = d["recon"][q]
            if not np.isfinite(sD):
                continue
            sP, sE, s0 = d["pipeline"][q], d["emu"][q], d["prev"][q]
            # ρ is signed → ratios are meaningless; show the absolute difference instead.
            if q == "rho_DH_DM":
                r = lambda x, y: x - y if (np.isfinite(x) and np.isfinite(y)) else np.nan
            else:
                r = lambda x, y: x / y if (np.isfinite(x) and y > 0) else np.nan
            print(f"  {t:<11} {_QUANT_SHORT[q]:<6} {s0:>9.4f} {sP:>9.4f} {sE:>9.4f} "
                  f"{sD:>9.4f} {r(sP, s0):>8.3f} {r(sP, sD):>6.3f} {r(sE, s0):>6.3f}")


def _plot(data, out_path):
    tracers = list(_TRACERS)
    display = [_DISPLAY.get(t, t) for t in tracers]
    x = np.arange(len(tracers), dtype=float)
    # dataviz reference palette, categorical slots 1-3 (validated all-pairs in
    # both modes). Assigned by ENTITY, so adding/removing the baseline series
    # never repaints the others. DESI stays black: it is the reference the
    # series are measured against, not one of them.
    c_prev, c_pipe, c_emu = "#eb6834", "#2a78d6", "#1baf7a"
    # Small horizontal dodge: pre- and post-change sigma differ by 0.07% (LRG1)
    # to 10% (ELG2), so at a shared x the near-identical bins would occlude and
    # look like a single point. Dodging shows "unchanged" as two adjacent marks
    # rather than as one ambiguous one.
    DX = 0.13

    fig, axes = plt.subplots(len(_PANELS), 1, figsize=(11, 14), sharex=True,
                             constrained_layout=True)
    for ax, (q, ylabel, is_corr) in zip(axes, _PANELS):
        for xi in x:
            ax.axvline(xi, color="gray", alpha=0.25, linewidth=0.6)

        def _xy(src, dx=0.0):
            xs, ys = [], []
            for i, t in enumerate(tracers):
                v = data[t][src][q]
                if np.isfinite(v):
                    xs.append(i + dx); ys.append(v)
            return xs, ys

        ax.scatter(*_xy("recon"), marker="x", s=45, color="black",
                   linewidths=1.6, zorder=5)
        # Pre-change pipeline + the emulator trained against it, dodged LEFT and
        # sharing an x on purpose: the NN is a surrogate for that pipeline, so
        # the two sitting on top of each other is the check, and their offset
        # from the right-hand marker is what the regeneration will close.
        ax.scatter(*_xy("prev", -DX), marker="o", s=52, facecolor="none",
                   edgecolor=c_prev, linewidths=1.6, zorder=3)
        ax.scatter(*_xy("emu", -DX), marker="o", s=20, color=c_emu,
                   linewidth=0, zorder=4)
        ax.scatter(*_xy("pipeline", DX), marker="o", s=52, facecolor="none",
                   edgecolor=c_pipe, linewidths=1.6, zorder=3)

        ax.set_ylabel(ylabel)
        if not is_corr:
            # Explicit headroom: matplotlib's autoscale clipped the tallest
            # markers (LRG1 sigma(DH) at 0.653) against the axis top once the
            # dodge widened the point cloud.
            vals = [data[tr][s][q] for tr in tracers
                    for s in ("recon", "prev", "pipeline", "emu")]
            vals = [v for v in vals if np.isfinite(v)]
            ax.set_ylim(0.0, max(vals) * 1.12 if vals else None)
        # ρ (is_corr) is signed and clustered well away from 0; let it autoscale
        # tightly around the data rather than padding down to zero.
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, axis="y")

    handles = [
        Line2D([0], [0], marker="x", linestyle="", markersize=7, markeredgewidth=1.6,
               color="black", label="DESI (bao-recon)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor="none",
               markeredgecolor=c_prev, markeredgewidth=1.6,
               label="pipeline at S81 (today 11:47, pre-S82..92)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=c_emu, markeredgecolor="none",
               label="Emulator v4 (trained 2026-06-19,\n155 commits / S52-S81 behind)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor="none",
               markeredgecolor=c_pipe, markeredgewidth=1.6,
               label="pipeline NOW (S92)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   title="Source (config-space)", fontsize=9, title_fontsize=10)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(display, rotation=20)
    axes[-1].set_xlabel("Tracer bin")
    axes[0].set_title(
        "Config-space BAO σ vs DESI bao-recon  |  DR1\n"
        "orange = pipeline at S81 (today), blue = NOW (S92): the last day's work.\n"
        "The NN predates BOTH by 155 commits, so its offset is NOT the S81->S92 delta",
        fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(_DEFAULT_MODELS),
                    help="directory of per-tracer .pt checkpoints (default: dr1/base)")
    ap.add_argument("--out", default=str(plots_dir() / "bao_emulator_vs_desi_dr1.png"))
    ap.add_argument("--baseline", default=str(Path(__file__).resolve().parent
                                              / "regress_dumps" / "pre_S81.npz"),
                    help="regress_sigmas dump to draw as the BEFORE series "
                         "(its config/{tracer}/fid cell is this exact quantity). "
                         "Pass '' to omit.")
    args = ap.parse_args()

    baseline = _load_baseline(args.baseline or None)
    data = _gather(args.models_dir, baseline=baseline)
    _print_table(data)
    _plot(data, args.out)


if __name__ == "__main__":
    main()
