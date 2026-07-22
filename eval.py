import argparse
import json
import math
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
try:
    # Package context (deployed as bedcosmo.num_tracers.emulator).
    from .util import (
        latin_hypercube_samples,
        get_default_save_path,
        get_pipeline,
        TRACER_TYPE_CHOICES,
        build_model,
        decode_emulator_outputs,
    )
except ImportError:
    # Script context (run directly from this dir; sys.path.insert above puts it first).
    from util import (
        latin_hypercube_samples,
        get_default_save_path,
        get_pipeline,
        TRACER_TYPE_CHOICES,
        build_model,
        decode_emulator_outputs,
    )

def _pick_scale(vals: np.ndarray, min_decades: float = 2.0) -> str:
    """Choose 'log', 'symlog', or 'linear' from the data's dynamic range, not its sign.

    A log/symlog axis is used only when the values genuinely span at least
    ``min_decades`` orders of magnitude, measured robustly from the 2.5-97.5 percentile
    range of |value|. This prevents a single sample near zero from collapsing the symlog
    linthresh and forcing a spurious log axis, so bounded O(1)-range parameters (e.g. the
    Ok/w0/wa uniform priors) render linear while genuine multi-decade quantities (the
    sigma targets) stay log.
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "linear"
    abs_nonzero = np.abs(v[v != 0])
    if abs_nonzero.size == 0:
        return "linear"
    lo, hi = np.percentile(abs_nonzero, [2.5, 97.5])
    if lo <= 0 or hi / lo < 10.0 ** min_decades:
        return "linear"
    return "log" if np.all(v > 0) else "symlog"


def _robust_linthresh(vals: np.ndarray) -> float:
    """Symlog linthresh from the 10th percentile of |nonzero value| (not the min, which a
    single near-zero sample would collapse into ~5 fake decades of axis)."""
    abs_nonzero = np.abs(vals[np.isfinite(vals) & (vals != 0)])
    if abs_nonzero.size == 0:
        return 1e-6
    return max(float(np.percentile(abs_nonzero, 10)), 1e-12)


def _log_bins(vals: np.ndarray, n_bins: int = 30, min_decades: float = 2.0) -> np.ndarray:
    """Return histogram bin edges matching the axis scale chosen by _pick_scale."""
    scale = _pick_scale(vals, min_decades)
    if scale == "log":
        return np.geomspace(vals.min(), vals.max(), n_bins + 1)
    if scale == "symlog":
        linthresh = _robust_linthresh(vals)
        def fwd(x):
            return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)
        def inv(y):
            return np.sign(y) * linthresh * (10 ** np.abs(y) - 1)
        t_min, t_max = fwd(vals.min()), fwd(vals.max())
        return inv(np.linspace(t_min, t_max, n_bins + 1))
    return np.linspace(vals.min(), vals.max(), n_bins + 1)


def _set_plain_formatter(ax) -> None:
    """Ensure tick labels use compact text, not LaTeX scientific notation."""
    for axis in (ax.xaxis, ax.yaxis):
        scale = axis.get_scale()
        if scale == "linear":
            axis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=3))
        elif scale == "log":
            vmin, vmax = axis.get_view_interval()
            if vmin > 0 and vmax > 0 and vmax / vmin < 30:
                axis.set_major_locator(mticker.LogLocator(subs=[1, 2, 5], numticks=8))
        # Use plain '1e-3' style, not '$1 \times 10^{-3}$'
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: "" if v == 0 else f"{v:.3g}"
        ))
        axis.set_minor_formatter(mticker.NullFormatter())
        axis.get_offset_text().set_visible(False)


def _set_log_or_symlog(ax, axis: str, vals: np.ndarray, min_decades: float = 2.0) -> None:
    """Set an axis scale from the data's dynamic range (see _pick_scale). Bounded,
    low-dynamic-range data (e.g. Ok/w0/wa) stays linear even under --log-scale."""
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    scale = _pick_scale(vals, min_decades)
    if scale == "log":
        setter("log")
    elif scale == "symlog":
        setter("symlog", linthresh=_robust_linthresh(vals))
    # linear: leave the matplotlib default


def _tol_normalized_error(y_pred, y_true, target_names, rtol, atol):
    """Per-sample coloring inputs for the eval corner plots.

    Returns
      severity        (N,) worst-target, scale-agnostic error: relative
                      |Δ|/|true| for sigma_* targets, absolute |Δ| for the rest
                      (rho_*, where a relative error is ill-defined near 0). This
                      is the continuous, threshold-free quantity the primary
                      ("relerr") plot shades by; it does NOT depend on atol/rtol.
      is_outlier      (N,) bool: sample fails np.isclose on ANY target — the
                      legacy pass/fail split, kept for the fallback colorings.
      regime_is_rtol  (N,) bool: for a failing sample, True if the target driving
                      the failure is rtol-dominated (|true| > atol/rtol), False if
                      atol-dominated. Only meaningful where is_outlier is True.
                      The crossover σ = atol/rtol is where the isclose budget
                      switches from an absolute floor (small σ, generous in
                      relative terms) to a pure rtol test (large σ, strict).
    """
    tiny = 1e-12
    diff = np.abs(y_pred - y_true)
    per_tgt = np.empty_like(diff)
    for c, name in enumerate(target_names):
        if name.startswith("sigma_"):
            per_tgt[:, c] = diff[:, c] / np.maximum(np.abs(y_true[:, c]), tiny)
        else:
            per_tgt[:, c] = diff[:, c]
    severity = per_tgt.max(axis=1)

    is_close = np.isclose(y_pred, y_true, rtol=rtol, atol=atol)
    is_outlier = ~np.all(is_close, axis=1)

    crossover = atol / rtol if rtol > 0 else np.inf
    excess = diff - (atol + rtol * np.abs(y_true))  # >0 on a target that fails
    drive = np.argmax(excess, axis=1)               # target driving the failure
    regime_is_rtol = np.abs(y_true[np.arange(len(y_true)), drive]) > crossover
    return severity, is_outlier, regime_is_rtol


def _corner_plot(save_file, data, names, *, severity, is_outlier, regime_is_rtol,
                 mode, log_scale, min_decades, title, panel_size):
    """Triangle plot over ``data`` columns, colored per ``mode``:

      "relerr"  (default) — off-diagonals shade every point by continuous
                worst-target relative error (viridis, log color scale), with a
                full-height colorbar; the diagonal is the conventional marginal.
                No pass/fail threshold, so no atol/rtol artifact.
      "regime"  (fallback) — pass (blue) vs fail, with fails split into
                atol-dominated (orange, σ<atol/rtol) and rtol-dominated
                (red, σ>atol/rtol) so strictness fails are visually separable.
      "passfail" (legacy) — single-color pass/fail, the original behavior.
    """
    from matplotlib.colors import LogNorm
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(panel_size * n, panel_size * n), squeeze=False)

    sev_lo, sev_hi = 1e-4, 1.0          # 0.01% .. 100% rel error, then saturate
    norm = LogNorm(vmin=sev_lo, vmax=sev_hi)
    sev_c = np.clip(severity, sev_lo, sev_hi)
    ok = ~is_outlier
    fail_atol = is_outlier & ~regime_is_rtol
    fail_rtol = is_outlier & regime_is_rtol
    # named by the error symptom, not the isclose regime (they invert): a small-sigma
    # fail (atol term dominates the budget) is one whose *relative* error |Δ|/σ blew up,
    # while a large-sigma fail (rtol term dominates) is a large *absolute* miss.
    lab_a = "relative-error fail (small σ, σ<atol/rtol)"
    lab_r = "absolute-error fail (large σ, σ>atol/rtol)"
    mappable = None

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                vals = data[:, i]
                bins = _log_bins(vals, min_decades=min_decades) if log_scale else np.linspace(vals.min(), vals.max(), 31)
                if mode == "relerr":
                    # conventional marginal; the error signal lives in the off-diagonal color
                    ax.hist(vals, bins=bins, color="tab:blue", alpha=0.6)
                else:
                    ax.hist(vals[ok], bins=bins, color="tab:blue", alpha=0.6, label="pass")
                    if mode == "regime":
                        ax.hist(vals[fail_atol], bins=bins, color="tab:orange", alpha=0.6, label=lab_a)
                        ax.hist(vals[fail_rtol], bins=bins, color="tab:red", alpha=0.6, label=lab_r)
                    else:
                        ax.hist(vals[is_outlier], bins=bins, color="tab:red", alpha=0.6, label="fail")
                if log_scale:
                    _set_log_or_symlog(ax, "x", vals, min_decades=min_decades)
            else:
                xv, yv = data[:, j], data[:, i]
                if mode == "relerr":
                    order = np.argsort(severity)  # draw worst points on top
                    mappable = ax.scatter(xv[order], yv[order], c=sev_c[order], s=5,
                                          cmap="viridis", norm=norm, alpha=0.7, linewidths=0)
                elif mode == "regime":
                    ax.scatter(xv[ok], yv[ok], s=4, alpha=0.35, color="tab:blue", label="pass")
                    ax.scatter(xv[fail_atol], yv[fail_atol], s=6, alpha=0.7, color="tab:orange", label=lab_a)
                    ax.scatter(xv[fail_rtol], yv[fail_rtol], s=6, alpha=0.7, color="tab:red", label=lab_r)
                else:
                    ax.scatter(xv[ok], yv[ok], s=4, alpha=0.4, color="tab:blue", label="pass")
                    ax.scatter(xv[is_outlier], yv[is_outlier], s=4, alpha=0.6, color="tab:red", label="fail")
                if log_scale:
                    _set_log_or_symlog(ax, "x", xv, min_decades=min_decades)
                    _set_log_or_symlog(ax, "y", yv, min_decades=min_decades)
            _set_plain_formatter(ax)
            ax.tick_params(axis="both", labelsize=8)
            if i == n - 1:
                ax.set_xlabel(names[j])
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(names[i])
            else:
                ax.tick_params(labelleft=False)

    if mode != "relerr":
        handles, labels = axes[n - 1, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=11)
    fig.suptitle(title, fontsize=22, y=1.01)
    fig.tight_layout()
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if not ax.get_visible():
                continue
            _set_plain_formatter(ax)
            if i != n - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)
    if mode == "relerr" and mappable is not None:
        # full-height colorbar to the right of the grid (added after tight_layout so
        # the panel positions are final); spans row 0 top to row n-1 bottom. Width is a
        # fixed fraction of a single panel so it stays proportionate as n grows, rather
        # than a fixed figure fraction (which reads as a thin sliver on a wide grid).
        top = axes[0, 0].get_position().y1
        bottom = axes[n - 1, 0].get_position().y0
        right = max(axes[i, i].get_position().x1 for i in range(n))
        panel_w = axes[n - 1, n - 1].get_position().width
        cax = fig.add_axes([right + 0.4 * panel_w, bottom, 0.28 * panel_w, top - bottom])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("worst-target rel. error  (σ: |Δ|/|true|)", fontsize=16)
        cb.ax.tick_params(labelsize=13)
    fig.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved triangle plot to: {save_file}")



def run_eval(model_path: str, save_path: str, analysis: str = "shapefit", quantity: str | None = None, n_samples: int = 500, seed: int = 42, hist_xlims: dict[str, tuple[float, float]] | None = None, rtol: float = 5e-3, atol: float = 1e-4, log_scale: bool = False, tracer_bin: str | None = None, min_decades: float = 2.0, triangle_color: str = "relerr") -> None:
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.random.seed(seed)

    # Load trained model
    # Prefer GPU 1 when 2+ are visible (leaves GPU 0 free); fall back to cuda:0 under
    # a masked/single-GPU set (e.g. CUDA_VISIBLE_DEVICES=1). Override via EMULATOR_DEVICE.
    if torch.cuda.is_available():
        _default_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda"
        device = torch.device(os.environ.get("EMULATOR_DEVICE", _default_dev))
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(
        analysis=ckpt.get("analysis", "bao"),
        architecture=ckpt.get("architecture", "resnet"),
        in_dim=len(ckpt["param_names"]),
        out_dim=len(ckpt["target_names"]),
        hidden_dim=ckpt["hidden_dim"],
        n_hidden=ckpt["n_hidden"],
        dropout=ckpt.get("dropout", 0.0),
        expand=ckpt.get("expand", 4),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mu = ckpt["x_mu"].cpu().numpy()
    x_sigma = ckpt["x_sigma"].cpu().numpy()
    y_mu = ckpt["y_mu"].cpu().numpy()
    y_sigma = ckpt["y_sigma"].cpu().numpy()
    log_normalize = ckpt.get("log_normalize", False)
    y_linthresh = ckpt["y_linthresh"].cpu().numpy() if ckpt.get("y_linthresh") is not None else None
    param_names = ckpt["param_names"]
    ckpt_target_names = ckpt["target_names"]

    # Resolve the ground-truth quantity. An explicit --quantity wins; otherwise use the
    # generator the model was trained against (recorded in the checkpoint), falling back
    # to the per-analysis default. This prevents silently scoring a config-space emulator
    # against the Fourier-space ('covar') ground truth (they differ ~10%).
    if quantity is None:
        quantity = ckpt.get("quantity") or ("config" if analysis == "bao" else "covar")
    print(f"Using ground-truth quantity: {quantity}"
          + (f"  (from checkpoint)" if ckpt.get("quantity") == quantity else "  (default/CLI)"))

    default_priors, target_names, ground_truth_fn, setup = get_pipeline(analysis, quantity, tracer_bin=tracer_bin, param_names=param_names)

    true_rows = []
    param_rows = []
    skipped = 0
    batch_seed = seed
    while len(true_rows) < n_samples:
        remaining = n_samples - len(true_rows)
        draws = latin_hypercube_samples(default_priors, n_samples=remaining, seed=batch_seed)
        batch_seed += 1
        for sample in draws:
            try:
                targets = ground_truth_fn(setup, sample)
                vals = [targets[t] for t in target_names]
                if not all(np.isfinite(vals)):
                    skipped += 1
                    continue
                true_rows.append(vals)
                param_rows.append([sample[p] for p in param_names])
            except Exception:
                skipped += 1
                continue
            if len(true_rows) >= n_samples:
                break

    print(f"Collected {len(true_rows)} accepted samples ({skipped} rejected).")

    y_true = np.array(true_rows, dtype=np.float32)
    x_raw = np.array(param_rows, dtype=np.float32)

    # NN predictions (standardize -> predict -> unstandardize)
    x_norm = (x_raw - x_mu) / x_sigma
    with torch.no_grad():
        y_pred_norm = model(torch.from_numpy(x_norm).to(device)).cpu().numpy()
    y_pred = decode_emulator_outputs(
        y_pred_norm,
        y_mu,
        y_sigma,
        ckpt_target_names,
        log_normalize=log_normalize,
        y_linthresh=y_linthresh,
    )

    # Absolute error for covariance elements (off-diagonal can be near zero),
    # percentage error for other targets.
    is_cov = target_names[0].startswith("cov_")
    if is_cov:
        deltas = y_pred - y_true
        delta_label = "absolute error"
        delta_unit = ""
    else:
        deltas = (y_pred - y_true) / y_true * 100.0
        delta_label = "% error"
        delta_unit = " [%]"

    # Plot histograms (dynamic grid)
    n_targets = len(target_names)
    ncols = min(n_targets, 4)
    nrows = math.ceil(n_targets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(target_names):
        ax = axes[i]
        col = deltas[:, i]
        if hist_xlims and name in hist_xlims:
            lo, hi = hist_xlims[name]
            ax.hist(col, bins=np.linspace(lo, hi, 41), range=(lo, hi),
                    edgecolor="black", linewidth=0.5)
        elif log_scale:
            # symlog x when the signed error genuinely spans min_decades (self-gated
            # by _pick_scale); a narrow, well-behaved distribution stays linear.
            ax.hist(col, bins=_log_bins(col, min_decades=min_decades),
                    edgecolor="black", linewidth=0.5)
            _set_log_or_symlog(ax, "x", col, min_decades=min_decades)
            _set_plain_formatter(ax)
        else:
            ax.hist(col, bins=40, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        mean = np.mean(col)
        std = np.std(col)
        ax.set_xlabel(f"$\\Delta$ {name}{delta_unit}")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nmean={mean:.2e}, std={std:.2e}")
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f"NN prediction {delta_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"eval_{timestamp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation plot to: {os.path.join(save_path, f'eval_{timestamp}.png')}")

    # --- Triangle plots, colored per triangle_color (see _corner_plot) ---
    # The pass/fail and severity inputs are threshold/coloring choices only; they are
    # computed once from the single sampled (y_pred, y_true) set and reused for every
    # requested mode, so no mode triggers a fresh ground-truth pass.
    severity, is_outlier, regime_is_rtol = _tol_normalized_error(
        y_pred, y_true, list(ckpt_target_names), rtol, atol)
    n_outlier = int(is_outlier.sum())
    panel_size = 4 if log_scale else 3
    modes = [triangle_color] if isinstance(triangle_color, str) else list(triangle_color)
    for mode in modes:
        if mode == "relerr":
            subtitle = f"median rel err {np.median(severity) * 100:.2f}%  (n={len(severity)})"
        else:
            subtitle = f"{n_outlier}/{len(is_outlier)} fail (rtol={rtol}, atol={atol})"

        _corner_plot(
            os.path.join(save_path, f"eval_triangle_params_{mode}_{timestamp}.png"),
            x_raw, list(param_names), severity=severity, is_outlier=is_outlier,
            regime_is_rtol=regime_is_rtol, mode=mode, log_scale=log_scale,
            min_decades=min_decades, title=f"Cosmo inputs — {subtitle}", panel_size=panel_size)

        _corner_plot(
            os.path.join(save_path, f"eval_triangle_targets_{mode}_{timestamp}.png"),
            y_true, list(ckpt_target_names), severity=severity, is_outlier=is_outlier,
            regime_is_rtol=regime_is_rtol, mode=mode, log_scale=log_scale,
            min_decades=min_decades, title=f"Target outputs — {subtitle}", panel_size=panel_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ShapeFit NN against the extractor.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID. Resolves model.pt and save path from the run's artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory from a training run (contains model.pt). Plots are saved here.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model.pt (ignored if --run-id or --run-dir is set).",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hist-xlims",
        type=str,
        default=None,
        help='JSON dict mapping target name to [lo, hi], e.g. \'{"qpar": [-1, 1], "qper": [-2, 2]}\'',
    )
    parser.add_argument("--rtol", type=float, default=2e-3, help="Relative tolerance for allclose outlier check (default: 2e-3).")
    parser.add_argument("--atol", type=float, default=2e-3, help="Absolute tolerance for allclose outlier check (default: 2e-3).")
    parser.add_argument("--log-scale", action="store_true", help="Use log/symlog scale on the triangle plot axes and the error-histogram x-axis.")
    parser.add_argument("--min-decades", type=float, default=2.0,
                        help="With --log-scale, only use a log/symlog axis when the data spans at "
                             "least this many orders of magnitude (robust 2.5-97.5 pct of |value|); "
                             "otherwise linear. Prevents bounded params (Ok/w0/wa) from getting a "
                             "spurious symlog axis. Default: 2.0.")
    parser.add_argument(
        "--analysis",
        type=str,
        default="shapefit",
        help="Analysis type: 'shapefit' or 'bao'.",
    )
    parser.add_argument(
        "--quantity",
        type=str,
        default=None,
        help="Ground-truth quantity. If omitted, resolved from the checkpoint's recorded "
             "'quantity', else the per-analysis default (bao -> 'config', the emulator's "
             "training generator; shapefit -> 'covar'). For bao: 'config' (config-space "
             "XiSigmaGenerator, matches training) or 'covar' (Fourier-space Fisher). For "
             "shapefit: 'covar' or 'mean'. Pass explicitly only to override.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save plots (default: resolved from run-id/run-dir, else get_default_save_path()).",
    )
    parser.add_argument(
        "--tracer-bin",
        dest="tracer_bin",
        type=str,
        default=None,
        choices=TRACER_TYPE_CHOICES,
        help="DESI tracer bin (e.g. BGS, LRG1). Sets zrange, z_eff, and N_tracers bounds for evaluation.",
    )
    parser.add_argument(
        "--triangle-color",
        dest="triangle_color",
        type=str,
        default="relerr,regime",
        help="Triangle-plot coloring mode(s), comma-separated. Rendering each extra "
             "mode is near-free once the sample is drawn, so the default renders both "
             "'relerr' (continuous rel-error shading, threshold-free) and 'regime' "
             "(pass/fail with relative- vs absolute-error fails split). Also available: "
             "'passfail' (legacy single-color). Pass e.g. 'relerr' for just one.",
    )
    args = parser.parse_args()
    triangle_modes = [m.strip() for m in args.triangle_color.split(",") if m.strip()]
    _valid = {"relerr", "regime", "passfail"}
    _bad = set(triangle_modes) - _valid
    if _bad:
        parser.error(f"--triangle-color: unknown mode(s) {sorted(_bad)}; choose from {sorted(_valid)}")

    if args.run_id is not None:
        import mlflow
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
        mlflow.set_tracking_uri(f"file:{scratch}/bedcosmo/num_tracers/emulator/mlruns")
        run = mlflow.get_run(args.run_id)
        artifact_uri = run.info.artifact_uri
        if artifact_uri.startswith("file://"):
            artifact_uri = artifact_uri[7:]
        model_path = os.path.join(artifact_uri, "checkpoints", "model_best.pt")
        save_path = args.save_path or artifact_uri
    elif args.run_dir is not None:
        model_path = os.path.join(args.run_dir, "checkpoints", "model_best.pt")
        save_path = args.save_path or args.run_dir
    else:
        if args.model_path is None:
            raise ValueError("Either --run-id, --run-dir, or --model-path must be set.")
        model_path = args.model_path
        save_path = args.save_path or get_default_save_path(analysis=args.analysis, quantity=args.quantity)

    hist_xlims = None
    if args.hist_xlims is not None:
        raw = json.loads(args.hist_xlims)
        hist_xlims = {k: tuple(v) for k, v in raw.items()}

    run_eval(
        model_path,
        save_path,
        analysis=args.analysis,
        quantity=args.quantity,
        n_samples=args.n_samples,
        seed=args.seed,
        hist_xlims=hist_xlims,
        rtol=args.rtol,
        atol=args.atol,
        log_scale=args.log_scale,
        tracer_bin=args.tracer_bin,
        min_decades=args.min_decades,
        triangle_color=triangle_modes,
    )


if __name__ == "__main__":
    main()
