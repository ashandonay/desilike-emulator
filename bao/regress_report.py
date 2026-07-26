"""Turn `regress_sigmas.py` dumps into an inspectable report.

`regress_sigmas.py compare` answers "identical or not" with an exit code. That
is the right acceptance gate, but it shows none of the numbers, so it cannot be
checked by eye. This prints the sigma-triplets themselves for every
(tracer, cosmology) in both spaces, side by side across desilike versions, plus
the max |delta| over *all* recorded arrays (covariances, k-grids, Fisher
matrices) -- not just the sigmas shown.

    python regress_report.py golden_41f082f0.npz new_4cfd6bec.npz
    python regress_report.py golden_41f082f0.npz new_4cfd6bec.npz --md report.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from regress_sigmas import COSMO_GRID, TRACERS

SIGMA_KEYS = ("DH_over_rs", "DM_over_rs", "DV_over_rs")
COSMO_LABELS = [row[0] for row in COSMO_GRID]
COSMO_DESC = {
    row[0]: f"Om={row[1]:.4g} Ok={row[2]:+.2f} w0={row[3]:+.2f} "
            f"wa={row[4]:+.2f} hrd={row[5]:g} N={row[6]:.2f}x"
    for row in COSMO_GRID
}


def _delta(xa: np.ndarray, xb: np.ndarray) -> Tuple[float, float]:
    """(max abs, max rel) between two arrays, NaN-safe."""
    adiff = np.abs(xa - xb)
    scale = np.maximum(np.abs(xa), np.abs(xb))
    with np.errstate(divide="ignore", invalid="ignore"):
        rdiff = np.where(scale > 0, adiff / scale, 0.0)
    return float(np.nanmax(adiff)), float(np.nanmax(rdiff))


def _fourier_sigmas(a: np.lib.npyio.NpzFile, tracer: str, label: str) -> Dict[str, float]:
    """sigma-triplet from the stored Fisher, in the same D/rd units as config.

    The Fourier dump records F_q (the dilation-parameter Fisher, marginalized)
    rather than the sigmas, so invert it here and scale by the template's
    fiducial D/rd -- the same conversion `fourier_space` does downstream.
    """
    pfx = f"fourier/{tracer}/{label}"
    if f"{pfx}/F_q" not in a.files:
        return {}
    F = np.atleast_2d(a[f"{pfx}/F_q"])
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return {}
    sig_q = np.sqrt(np.diag(cov))
    if F.shape[0] == 1:                      # qiso -> DV only
        return {"DV_over_rs": sig_q[0] * float(a[f"{pfx}/DV_over_rd_fid"])}
    return {                                  # qparqper -> (qpar, qper) = (DH, DM)
        "DH_over_rs": sig_q[0] * float(a[f"{pfx}/DH_over_rd_fid"]),
        "DM_over_rs": sig_q[1] * float(a[f"{pfx}/DM_over_rd_fid"]),
    }


def _config_sigmas(a: np.lib.npyio.NpzFile, tracer: str, label: str) -> Dict[str, float]:
    pfx = f"config/{tracer}/{label}"
    return {k: float(a[f"{pfx}/{k}"]) for k in SIGMA_KEYS if f"{pfx}/{k}" in a.files}


def _plot(A, B, path: str, name_a: str, name_b: str) -> None:
    """Overlay old vs new sigmas per tracer.

    A table of zeros proves equality but says nothing about *coverage*, which is
    the other half of the question: sigma spans ~2 orders of magnitude across
    this grid (lowOm to highOm), so the log y-axis is doing real work here --
    it shows the agreement holds across the whole range, not just near fiducial.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(COSMO_LABELS))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    styles = {  # quantity -> (colour, marker)
        "DH_over_rs": ("#1f77b4", "o"),
        "DM_over_rs": ("#d62728", "s"),
        "DV_over_rs": ("#2ca02c", "^"),
    }

    for ax, tracer in zip(axes.ravel(), TRACERS):
        for space, getter, ls in (("config", _config_sigmas, "-"),
                                  ("fourier", _fourier_sigmas, "--")):
            for q, (colour, marker) in styles.items():
                old = [getter(A, tracer, c).get(q, np.nan) for c in COSMO_LABELS]
                new = [getter(B, tracer, c).get(q, np.nan) for c in COSMO_LABELS]
                if np.all(np.isnan(old)):
                    continue
                # Old drawn as a fat pale line, new as small solid markers on
                # top: any disagreement shows as a marker off the ribbon.
                ax.plot(x, old, ls, color=colour, lw=4, alpha=0.28,
                        label=f"{q.split('_')[0]} {space} (old)")
                ax.plot(x, new, marker, color=colour, ms=4.5,
                        mfc="none" if space == "fourier" else colour,
                        label=f"{q.split('_')[0]} {space} (new)")
        ax.set_yscale("log")
        ax.set_title(tracer, fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.set_xticks(x)
        ax.set_xticklabels(COSMO_LABELS, rotation=45, ha="right", fontsize=8)

    axes[0, 0].set_ylabel(r"$\sigma(D/r_d)$")
    axes[1, 0].set_ylabel(r"$\sigma(D/r_d)$")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"desilike upgrade regression: {name_a} (old) vs {name_b} (new)\n"
                 "old = thick pale line, new = markers; solid = config, "
                 "open = Fourier", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="baseline dump (old desilike)")
    p.add_argument("b", help="comparison dump (new desilike)")
    p.add_argument("--md", help="also write a markdown copy here")
    p.add_argument("--plot", help="also write a PNG overlay here")
    args = p.parse_args()

    A, B = np.load(args.a), np.load(args.b)
    shared = sorted(set(A.files) & set(B.files))

    # --- per-space delta rollup over every recorded array, not just sigmas ---
    by_space: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    n_ident = 0
    for key in shared:
        xa, xb = A[key], B[key]
        if xa.shape == xb.shape and np.array_equal(xa, xb):
            n_ident += 1
            by_space[key.split("/")[0]].append((0.0, 0.0))
        else:
            by_space[key.split("/")[0]].append(_delta(xa, xb))

    lines: List[str] = []
    w = lines.append

    w(f"# BAO sigma regression: `{args.a}`  vs  `{args.b}`")
    w("")
    w(f"Arrays compared: **{len(shared)}**  |  bit-identical: **{n_ident}**  |  "
      f"differing: **{len(shared) - n_ident}**")
    if set(A.files) ^ set(B.files):
        w(f"WARNING: {len(set(A.files) ^ set(B.files))} keys present in only one dump.")
    w("")

    w("## Coverage and max deviation, by space")
    w("")
    w("| space | arrays | max abs delta | max rel delta |")
    w("|---|---:|---:|---:|")
    for space in sorted(by_space):
        deltas = by_space[space]
        w(f"| {space} | {len(deltas)} | {max(d[0] for d in deltas):.3e} "
          f"| {max(d[1] for d in deltas):.3e} |")
    w("")

    w("## Cosmology grid")
    w("")
    w("| label | parameters |")
    w("|---|---|")
    for label in COSMO_LABELS:
        w(f"| `{label}` | {COSMO_DESC[label]} |")
    w("")

    # --- the sigmas themselves ---
    for space, getter in (("config", _config_sigmas), ("fourier", _fourier_sigmas)):
        w(f"## {space}-space sigma(D/rd), old -> new")
        w("")
        w("| tracer | cosmology | quantity | old | new | delta |")
        w("|---|---|---|---:|---:|---:|")
        for tracer in TRACERS:
            for label in COSMO_LABELS:
                sa, sb = getter(A, tracer, label), getter(B, tracer, label)
                for k in SIGMA_KEYS:
                    if k not in sa:
                        continue
                    d = sb[k] - sa[k]
                    flag = "0" if d == 0.0 else f"**{d:+.3e}**"
                    w(f"| {tracer} | `{label}` | {k} | {sa[k]:.10f} "
                      f"| {sb[k]:.10f} | {flag} |")
        w("")

    if args.plot:
        _plot(A, B, args.plot, args.a, args.b)

    text = "\n".join(lines)
    print(text)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text + "\n")
        print(f"\n[wrote] {args.md}")
    return 0 if n_ident == len(shared) else 1


if __name__ == "__main__":
    raise SystemExit(main())
