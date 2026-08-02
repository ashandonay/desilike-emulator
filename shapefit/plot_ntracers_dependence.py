"""How the ShapeFit full-shape emulator data responds to the N_tracers input.

The ShapeFit emulator has two datasets. Only the *covar* one takes N_tracers:
generate_mean_data.py drops it from the prior (the fiducial qiso/qap/f_sigmar/m
do not depend on how many objects you counted), while generate_covar_data.py
maps (N_tracers, cosmology) -> the 10 covariance targets, TARGET_NAMES =
4 sigmas + 6 correlations. So "how the full-shape data varies with N_tracers"
is a statement about those 10 targets, and this script draws it two ways:

  fiducial  shapefit_ntracers_fiducial.png
            Live Fisher sweep at the DESI fiducial cosmology over each
            tracer's own emulator box (util.ntracers_range). Clean curves --
            the pure N_tracers response with cosmology held fixed. sigma
            panels carry a 1/sqrt(N) guide: the shot-noise-limited slope, so
            flattening below it is the sample-variance regime where extra
            objects stop buying precision.

  training  shapefit_ntracers_training.png
            The same 10 targets read straight off the generated covar
            training set (median + 16-84 band per N bin). This is what the
            network actually sees: the cosmology prior is marginalised over,
            so the band width is the part of the label that N_tracers does
            NOT explain.

x is N / N_DR1 (the tracer's DR1 passed count, util.ntracers) so all six
tracers share one axis and the DR1 anchor sits at 1.0.

Run (from shapefit/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        OMP_NUM_THREADS=1 ~/miniconda3/envs/emulator/bin/python \
        plot_ntracers_dependence.py --figure both
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from util import get_default_save_path, ntracers, ntracers_range, plots_dir

# Fiducial-sweep bins are the shapefit samples (LRG3, not the BAO LRG3+ELG1).
TRACERS = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")
TRACER_COLOR = {
    "BGS": "tab:red", "LRG1": "tab:blue", "LRG2": "tab:orange",
    "LRG3": "tab:green", "LRG3_ELG1": "tab:green", "ELG2": "tab:purple",
    "QSO": "tab:brown",
}
DISPLAY = {"LRG3_ELG1": "LRG3+ELG1"}

FID_SAMPLE = {
    "omega_cdm": 0.1200,
    "omega_b": 0.02237,
    "h": 0.6736,
    "ln10A_s": 3.036394,
    "n_s": 0.9649,
}

TARGET_LABEL = {
    "sigma_qiso": r"$\sigma(q_{\rm iso})$",
    "sigma_qap": r"$\sigma(q_{\rm AP})$",
    "sigma_f_sigmar": r"$\sigma(f\sigma_r)$",
    "sigma_m": r"$\sigma(m)$",
    "rho_qiso_qap": r"$\rho(q_{\rm iso},q_{\rm AP})$",
    "rho_qiso_f_sigmar": r"$\rho(q_{\rm iso},f\sigma_r)$",
    "rho_qiso_m": r"$\rho(q_{\rm iso},m)$",
    "rho_qap_f_sigmar": r"$\rho(q_{\rm AP},f\sigma_r)$",
    "rho_qap_m": r"$\rho(q_{\rm AP},m)$",
    "rho_f_sigmar_m": r"$\rho(f\sigma_r,m)$",
}


def _label(tracer: str) -> str:
    return DISPLAY.get(tracer, tracer)


# ===========================================================================
# Fiducial-cosmology sweep
# ===========================================================================
def _sweep_worker(task):
    """(tracer, N) -> (tracer, N, [10 target values]).  Top-level for spawn."""
    tracer, N = task
    import fourier_space
    sample = dict(FID_SAMPLE)
    sample["N_tracers"] = float(N)
    try:
        out = fourier_space.run_fisher(sample, tracer_bin=tracer)
        vals = [float(out[n]) for n in fourier_space.TARGET_NAMES]
    except Exception as exc:                                  # noqa: BLE001
        print(f"  FAILED {tracer} N={N:.4g}: {exc}", file=sys.stderr)
        vals = [np.nan] * 10
    return tracer, float(N), vals


def run_sweep(tracers, n_points: int, workers: int) -> dict:
    """{tracer: (N array, target array [n_points, 10])} at fixed cosmology."""
    import fourier_space

    grids = {t: np.geomspace(*ntracers_range(t, "dr1"), n_points)
             for t in tracers}
    tasks = [(t, N) for t in tracers for N in grids[t]]
    print(f"Fiducial sweep: {len(tasks)} Fisher runs on {workers} workers "
          f"(~35 s each serially)")

    ctx = mp.get_context("spawn")
    # maxtasksperchild: the full-shape likelihood leaks per build (bao
    # CHANGELOG; a covar run OOM'd the box on 2026-07-29). Recycle often --
    # the sweep is small enough that respawn cost is irrelevant.
    with ctx.Pool(workers, maxtasksperchild=8) as pool:
        results = pool.map(_sweep_worker, tasks, chunksize=1)

    out = {}
    for t in tracers:
        rows = [r for r in results if r[0] == t]
        rows.sort(key=lambda r: r[1])
        out[t] = (np.array([r[1] for r in rows]),
                  np.array([r[2] for r in rows]))
    return out, list(fourier_space.TARGET_NAMES)


# ===========================================================================
# Training-set read
# ===========================================================================
def load_training(version: str, dataset: str = "dr1",
                  cosmo_model: str = "base") -> dict:
    """{tracer: (N, y[n,10])} pooled over the train and test splits."""
    root = Path(get_default_save_path(analysis="shapefit", quantity="covar",
                                      cosmo_model=cosmo_model,
                                      dataset=dataset)) / version
    out, names, pnames = {}, None, None
    for tracer in ("BGS", "LRG1", "LRG2", "LRG3", "LRG3_ELG1", "ELG2", "QSO"):
        xs, ys = [], []
        for split in ("train", "test"):
            path = root / f"{tracer}_{split}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=True)
            xs.append(d["x"])
            ys.append(d["y"])
            names = [str(s) for s in d["target_names"]]
            pnames = [str(s) for s in d["param_names"]]
        if not xs:
            continue
        x, y = np.vstack(xs), np.vstack(ys)
        iN = pnames.index("N_tracers")
        out[tracer] = (x[:, iN], y)
    if names is None:
        raise SystemExit(
            f"No shapefit covar training data found for version {version!r}.")
    return out, names


def _binned(xr, y, nbins=10):
    """Median and 16-84 band of y in nbins log-spaced bins of xr."""
    edges = np.geomspace(xr.min(), xr.max() * (1 + 1e-9), nbins + 1)
    idx = np.clip(np.digitize(xr, edges) - 1, 0, nbins - 1)
    xc, med, lo, hi = [], [], [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 5:
            continue
        xc.append(np.median(xr[m]))
        q = np.nanpercentile(y[m], [16, 50, 84])
        lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    return (np.array(xc), np.array(med), np.array(lo), np.array(hi))


# ===========================================================================
# Figures
# ===========================================================================
def _grid(names, title, ylog_sigma=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(21, 8), squeeze=False)
    fig.suptitle(title, fontsize=13)
    axarr = axes.ravel()
    for ax, name in zip(axarr, names):
        ax.set_title(TARGET_LABEL.get(name, name), fontsize=11)
        ax.set_xscale("log")
        if name.startswith("sigma_") and ylog_sigma:
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color="0.7", lw=0.6, zorder=0)
        ax.axvline(1.0, color="0.5", lw=0.8, ls=":", zorder=0)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axarr[5:]:
        ax.set_xlabel(r"$N_{\rm tracers}\,/\,N_{\rm DR1}$")
    return fig, axarr


def figure_fiducial(sweep, names, out_png):
    import matplotlib.pyplot as plt
    fig, axarr = _grid(
        names,
        "ShapeFit covar targets vs $N_{\\rm tracers}$ — DESI fiducial "
        "cosmology, each tracer swept over its own emulator box")
    for tracer, (N, y) in sweep.items():
        xr = N / ntracers(tracer, "dr1")
        c = TRACER_COLOR[tracer]
        for i, name in enumerate(names):
            axarr[i].plot(xr, y[:, i], marker="o", ms=3.5, lw=1.4, color=c,
                          label=_label(tracer))
    # 1/sqrt(N) guides on the sigma panels, anchored at each tracer's low end.
    for i, name in enumerate(names):
        if not name.startswith("sigma_"):
            continue
        for tracer, (N, y) in sweep.items():
            xr = N / ntracers(tracer, "dr1")
            axarr[i].plot(xr, y[0, i] * np.sqrt(xr[0] / xr), color="k",
                          ls="--", lw=0.7, alpha=0.5, zorder=0)
    axarr[0].plot([], [], "k--", lw=0.7, label=r"$1/\sqrt{N}$")
    axarr[0].legend(fontsize=7.5, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")


def figure_training(train, names, out_png, version):
    import matplotlib.pyplot as plt
    fig, axarr = _grid(
        names,
        f"ShapeFit covar training set {version} — targets vs "
        r"$N_{\rm tracers}$, marginalised over the 5-parameter cosmology "
        "prior (median, 16–84%)")
    for tracer, (N, y) in train.items():
        xr = N / ntracers(tracer, "dr1")
        c = TRACER_COLOR[tracer]
        for i, name in enumerate(names):
            xc, med, lo, hi = _binned(xr, y[:, i])
            if xc.size == 0:
                continue
            axarr[i].fill_between(xc, lo, hi, color=c, alpha=0.16, lw=0)
            axarr[i].plot(xc, med, lw=1.6, color=c, label=_label(tracer))
    axarr[0].legend(fontsize=7.5, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", choices=["fiducial", "training", "both"],
                   default="both")
    p.add_argument("--tracers", nargs="+", default=list(TRACERS))
    p.add_argument("--n-points", type=int, default=9,
                   help="N_tracers grid points per tracer (fiducial sweep).")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--version", default="v100",
                   help="covar training-set version for the training figure.")
    args = p.parse_args()

    pd = plots_dir()
    if args.figure in ("fiducial", "both"):
        sweep, names = run_sweep(args.tracers, args.n_points, args.workers)
        np.savez(pd / "shapefit_ntracers_fiducial.npz",
                 target_names=np.array(names),
                 **{f"{t}_N": v[0] for t, v in sweep.items()},
                 **{f"{t}_y": v[1] for t, v in sweep.items()})
        figure_fiducial(sweep, names, pd / "shapefit_ntracers_fiducial.png")
        print("\n  sigma(qiso) across each box (low N -> high N):")
        for t, (N, y) in sweep.items():
            print(f"    {_label(t):>10s}  {y[0,0]:.5f} -> {y[-1,0]:.5f}  "
                  f"(ratio {y[0,0]/y[-1,0]:.2f}x, "
                  f"1/sqrt(N) would give {np.sqrt(N[-1]/N[0]):.2f}x)")

    if args.figure in ("training", "both"):
        train, names = load_training(args.version)
        figure_training(train, names, pd / "shapefit_ntracers_training.png",
                        args.version)


if __name__ == "__main__":
    main()
