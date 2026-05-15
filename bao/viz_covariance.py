"""Visualize the BAO data-covariance components used in prep_covar.py.

For a single tracer bin, calls build_bao_likelihood (which now returns
cov_components = {C_gauss, C_SSC, C_recon_shot, C_total}), then produces:

  - One row of heatmaps of |C_X| (shared log color scale) for each component
    plus the total. Multipole-block boundaries are marked.

Run from bao/ with:
    ~/miniconda3/envs/emulator/bin/python viz_covariance.py --tracer QSO
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "..")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402

import pandas as pd  # noqa: E402


_DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}


def _get_ntracers(dataset: str, tracer_key: str) -> float:
    desi_dir = Path.home() / "data" / "desi" / f"bao_{dataset}"
    data_path = desi_dir / "desi_data.csv"
    df = pd.read_csv(data_path)[["tracer", "passed"]].drop_duplicates(subset=["tracer"])
    n_by_tracer = {row["tracer"]: float(row["passed"]) for _, row in df.iterrows()}
    name_map = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
    return n_by_tracer[name_map.get(tracer_key, tracer_key)]


def _build_components(tracer_key: str, dataset: str, omega_m: float, hrdrag: float):
    cfg = TRACER_CONFIGS[tracer_key]
    z_min, z_max = cfg["zrange"]
    z_eff = cfg["z_eff"]
    area = _DATASET_AREAS[dataset.lower()]
    n_tracers = _get_ntracers(dataset, tracer_key)

    theta_cosmo, hrdrag_eff = prep_covar._to_bao_cosmo_params(  # noqa: SLF001
        {**prep_covar.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag}
    )
    info = prep_covar.build_bao_likelihood(
        N_tracers=n_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag_eff,
        tracer_bin=tracer_key,
        zrange=(z_min, z_max),
        z_eff=z_eff,
        area=area,
    )
    obs = info["observable"]
    k_centers = np.asarray(obs.k[0], dtype=np.float64)
    ells = tuple(int(l) for l in obs.ells)
    return info["cov_components"], k_centers, ells, info["is_lya"]


def _plot_heatmap(ax, M, k_centers, ells, title, norm):
    n_k = len(k_centers)
    absM = np.abs(np.asarray(M))
    im = ax.imshow(absM, origin="upper", cmap="magma", norm=norm)
    for i in range(1, len(ells)):
        ax.axhline(i * n_k - 0.5, color="cyan", linewidth=0.6, alpha=0.7)
        ax.axvline(i * n_k - 0.5, color="cyan", linewidth=0.6, alpha=0.7)
    tick_pos = [i * n_k + n_k // 2 for i in range(len(ells))]
    tick_lbl = [f"$\\ell={l}$" for l in ells]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lbl)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracer", required=True,
                   help="Tracer key, e.g. LRG2, ELG2, QSO, BGS, Lya_QSO.")
    p.add_argument("--dataset", default="dr2", help="DESI dataset tag (dr1 or dr2).")
    p.add_argument("--omega-m", type=float, default=0.3153)
    p.add_argument("--hrdrag", type=float, default=99.53)
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: viz_cov_<tracer>_<dataset>.png).")
    args = p.parse_args()

    components, k_centers, ells, is_lya = _build_components(
        args.tracer, args.dataset, args.omega_m, args.hrdrag
    )

    if is_lya:
        print(f"[info] Lya tracer — only C_gauss is added; "
              f"NG components are skipped in the pipeline.")

    keys_present = [k for k in ("C_gauss", "C_SSC", "C_recon_shot", "C_total") if k in components]
    n_components = len(keys_present)

    pretty_label = {
        "C_gauss": r"$|C_\mathrm{gauss}|$",
        "C_SSC":   r"$|C_\mathrm{SSC}|$",
        "C_recon_shot": r"$|C_\mathrm{recon\_shot}|$",
        "C_total": r"$|C_\mathrm{total}|$",
    }

    # Shared log color scale across all heatmaps.
    all_pos = np.concatenate([
        np.abs(np.asarray(components[k])).ravel() for k in keys_present
    ])
    all_pos = all_pos[all_pos > 0]
    shared_norm = LogNorm(vmin=all_pos.min(), vmax=all_pos.max())

    fig, axes = plt.subplots(
        1, n_components,
        figsize=(3.6 * n_components + 0.8, 4.0),
        constrained_layout=True,
    )
    if n_components == 1:
        axes = [axes]
    last_im = None
    for j, name in enumerate(keys_present):
        last_im = _plot_heatmap(axes[j], components[name], k_centers, ells,
                                pretty_label.get(name, name), shared_norm)
    fig.colorbar(last_im, ax=list(axes), shrink=0.9, location="right",
                 label=r"$|C|$")

    suptitle = (f"BAO covariance components — {args.tracer} / {args.dataset.upper()}  "
                f"(Om={args.omega_m:.4g}, hrdrag={args.hrdrag:.4g})\n"
                f"k in [{k_centers[0]:.3f}, {k_centers[-1]:.3f}] h/Mpc, "
                f"n_k={len(k_centers)}, ells={list(ells)}")
    fig.suptitle(suptitle, fontsize=11)

    out = args.output or f"viz_cov_{args.tracer}_{args.dataset}.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")

    print("\nComponent summary (Frobenius norms):")
    for name in keys_present:
        M = np.asarray(components[name])
        print(f"  {name:10s}  ||M||_F = {np.linalg.norm(M):.4e}   "
              f"diag.range=[{np.min(np.abs(np.diag(M))):.3e}, "
              f"{np.max(np.abs(np.diag(M))):.3e}]   shape={M.shape}")


if __name__ == "__main__":
    main()
