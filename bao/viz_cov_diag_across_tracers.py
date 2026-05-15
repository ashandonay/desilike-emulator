"""Compare the per-multipole diagonal of the covariance components across tracers.

For each requested tracer, runs build_bao_likelihood and pulls
cov_components = {C_gauss, C_SSC, C_recon_shot}. Plots one horizontal panel
per component, with shared log y-axis. Color encodes tracer.

Run from bao/:
    ~/miniconda3/envs/emulator/bin/python viz_cov_diag_across_tracers.py \\
        --tracers BGS,LRG1,LRG2,LRG3_ELG1,ELG2,QSO --dataset dr2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "..")

import prep_covar  # noqa: E402
from util import TRACER_CONFIGS  # noqa: E402

import pandas as pd  # noqa: E402


_DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}


def _get_ntracers(dataset: str, tracer_key: str) -> float:
    desi_dir = Path.home() / "data" / "desi" / f"bao_{dataset}"
    df = pd.read_csv(desi_dir / "desi_data.csv")[["tracer", "passed"]] \
        .drop_duplicates(subset=["tracer"])
    n_by = {row["tracer"]: float(row["passed"]) for _, row in df.iterrows()}
    name_map = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
    return n_by[name_map.get(tracer_key, tracer_key)]


def _build_components(tracer_key, dataset, omega_m, hrdrag):
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
    return (
        info["cov_components"],
        np.asarray(obs.k[0], dtype=np.float64),
        tuple(int(l) for l in obs.ells),
        info["is_lya"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracers", required=True,
                   help="Comma-separated tracer keys, e.g. BGS,LRG1,LRG2,LRG3_ELG1,ELG2,QSO.")
    p.add_argument("--dataset", default="dr2")
    p.add_argument("--omega-m", type=float, default=0.3153)
    p.add_argument("--hrdrag", type=float, default=99.53)
    p.add_argument("--components", default="C_gauss,C_SSC,C_recon_shot",
                   help="Comma-separated components to plot (one panel each).")
    p.add_argument("--ell", type=int, default=0,
                   help="Which multipole's diagonal to plot (0, 2).")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    tracers = [t.strip() for t in args.tracers.split(",") if t.strip()]
    components_to_plot = [c.strip() for c in args.components.split(",") if c.strip()]

    cmap = plt.get_cmap("tab10")
    tracer_colors = {t: cmap(i % 10) for i, t in enumerate(tracers)}
    component_label = {
        "C_gauss":      r"$C_\mathrm{gauss}$",
        "C_SSC":        r"$C_\mathrm{SSC}$",
        "C_recon_shot": r"$C_\mathrm{recon\_shot}$",
        "C_total":      r"$C_\mathrm{total}$",
    }

    per_tracer = {}
    for t in tracers:
        comps, k, ells, is_lya = _build_components(t, args.dataset, args.omega_m, args.hrdrag)
        if is_lya:
            print(f"[warn] {t} is Lya — only C_gauss available; other components will be skipped.")
        per_tracer[t] = (comps, k, ells)

    ells_ref = per_tracer[tracers[0]][2]
    if args.ell not in ells_ref:
        raise ValueError(f"--ell {args.ell} not in pipeline ells {ells_ref}")
    ell_idx = ells_ref.index(args.ell)

    n_panels = len(components_to_plot)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.0 * n_panels, 4.0),
        sharey=True, sharex=True, constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for j, cname in enumerate(components_to_plot):
        ax = axes[j]
        for t in tracers:
            comps, k, _ = per_tracer[t]
            n_k = len(k)
            sl = slice(ell_idx * n_k, (ell_idx + 1) * n_k)
            if cname not in comps:
                continue
            d = np.abs(np.diag(np.asarray(comps[cname]))[sl])
            d = np.where(d > 0, d, np.nan)
            ax.plot(k, d,
                    color=tracer_colors[t],
                    linestyle="-", linewidth=1.6, label=t)
        ax.set_yscale("log")
        ax.set_xlabel(r"$k\,[h/\mathrm{Mpc}]$")
        ax.set_title(component_label.get(cname, cname), fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(rf"$|C_{{\ell\ell}}(k,k)|$, $\ell={args.ell}$")

    tracer_handles = [plt.Line2D([0], [0], color=tracer_colors[t], lw=2, label=t)
                      for t in tracers]
    fig.legend(handles=tracer_handles, title="tracer",
               loc="center right", fontsize=9, frameon=True)
    fig.suptitle(
        f"Diagonal of covariance components across tracers, "
        f"$\\ell={args.ell}$ — {args.dataset.upper()} "
        f"(Om={args.omega_m:.4g}, hrdrag={args.hrdrag:.4g})",
        fontsize=11,
    )

    out = args.output or f"viz_cov_diag_tracers_{args.dataset}_ell{args.ell}.png"
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
