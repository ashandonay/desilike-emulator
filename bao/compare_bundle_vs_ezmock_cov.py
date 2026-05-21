"""Bundle vs EZmock cov_ξ head-to-head.

We have two DR1 cov_ξ products for LRG2 and QSO:

  bundle :  likelihood_correlation-recon-poles_*.h5   (the local likelihood h5)
            cov_ξ on BAO-fit grid only: ℓ ∈ {0,2}, s ∈ [50,150] Mpc/h, Δs=4

  ezmock :  covariance_spectrum-poles+correlation-poles-recon_*.h5
            correlationrecon block: ℓ ∈ {0,2,4}, s ∈ [3,198] Mpc/h, Δs≈4

The bundle s-grid is a strict subset of the ezmock s-grid. This script slices
ezmock to the bundle's (s, ℓ) and compares element-wise:

  - diagonal σ(s, ℓ) ratio
  - full Frobenius norm of the difference
  - correlation matrices side-by-side
  - per-ℓ σ overplots

LRG2 F2/DESI was 0.538 (bundle) vs 0.480 (ezmock), a ~10% gap. If the covs are
identical here, the gap is some other code path. If they differ, this localizes it.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_EZMOCK_DIR = _LIK_DIR / "covariance"

_BUNDLES = {
    "LRG2": "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "QSO":  "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}
_EZMOCK_FILES = {
    "LRG2": "covariance_spectrum-poles+correlation-poles-recon_LRG_GCcomb_z0.6-0.8_thetacut0.05.h5",
    "QSO":  "covariance_spectrum-poles+correlation-poles-recon_QSO_GCcomb_z0.8-2.1_thetacut0.05.h5",
}


def _load_bundle(tracer):
    fpath = _LIK_DIR / _BUNDLES[tracer]
    with h5py.File(fpath, "r") as f:
        cov = np.asarray(f["covariance/value"][...], dtype=np.float64)
        ells, s_per_ell = [], []
        for k in sorted(f["covariance/observable"].keys()):
            if k.isdigit():
                ells.append(int(f[f"covariance/observable/{k}/meta/ell"][()]))
                s_per_ell.append(np.asarray(f[f"covariance/observable/{k}/s"][...], dtype=np.float64))
    return cov, ells, s_per_ell


def _load_ezmock_corrrec_block(tracer):
    """Return the FULL 150×150 correlationrecon cov + its ells and s-grids."""
    fpath = _EZMOCK_DIR / _EZMOCK_FILES[tracer]
    with h5py.File(fpath, "r") as f:
        C_full = np.asarray(f["value"][...], dtype=np.float64)
        spec_grp = f["observable/spectrum"]
        n_spec = sum(int(spec_grp[f"{e}/k"].shape[0])
                     for e in spec_grp.keys() if e.isdigit())
        cr_grp = f["observable/correlationrecon"]
        ells, s_per_ell = [], []
        for k in sorted(cr_grp.keys()):
            if k.isdigit():
                ells.append(int(cr_grp[f"{k}/meta/ell"][()]))
                s_per_ell.append(np.asarray(cr_grp[f"{k}/s"][...], dtype=np.float64))
    cr_block = C_full[n_spec:, n_spec:]
    return cr_block, ells, s_per_ell


def _slice_ezmock_to_bundle(ez_block, ez_ells, ez_s, b_ells, b_s):
    """Build global indices into ez_block that match the bundle's (ℓ, s) layout."""
    # Per-ℓ row offsets in ez_block
    offsets = {}
    cur = 0
    for e, sgrid in zip(ez_ells, ez_s):
        offsets[e] = (cur, cur + len(sgrid))
        cur += len(sgrid)

    idx = []
    for b_e, b_sgrid in zip(b_ells, b_s):
        if b_e not in offsets:
            raise RuntimeError(f"bundle ell={b_e} not in ezmock ells {ez_ells}")
        lo, hi = offsets[b_e]
        ez_sgrid = ez_s[ez_ells.index(b_e)]
        # Find nearest ezmock bin per bundle bin (s-grids match to ~floating-point precision)
        for b_si in b_sgrid:
            j = int(np.argmin(np.abs(ez_sgrid - b_si)))
            if abs(ez_sgrid[j] - b_si) > 0.1:
                raise RuntimeError(f"no ezmock bin within 0.1 Mpc/h of s={b_si} (ell={b_e})")
            idx.append(lo + j)
    idx = np.asarray(idx, dtype=np.int64)
    return ez_block[np.ix_(idx, idx)]


def _correlation_from_cov(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def _summary(tracer, C_b, C_e):
    """Compare bundle (C_b) and ezmock-sliced (C_e) cov on the same 52×52 grid."""
    diag_b = np.sqrt(np.diag(C_b))
    diag_e = np.sqrt(np.diag(C_e))
    diag_ratio = diag_b / diag_e

    diff = C_b - C_e
    frob_diff = np.linalg.norm(diff, ord="fro")
    frob_b = np.linalg.norm(C_b, ord="fro")
    frob_e = np.linalg.norm(C_e, ord="fro")

    R_b = _correlation_from_cov(C_b)
    R_e = _correlation_from_cov(C_e)

    print(f"\n=== {tracer} ===")
    print(f"  cov shape: bundle {C_b.shape}, ezmock-sliced {C_e.shape}")
    print(f"  σ ratio (bundle / ezmock):  min {diag_ratio.min():.4f}  "
          f"median {np.median(diag_ratio):.4f}  max {diag_ratio.max():.4f}")
    print(f"  ||C_b - C_e||_F / ||C_b||_F  = {frob_diff / frob_b:.4f}")
    print(f"  ||C_b||_F  = {frob_b:.4e}")
    print(f"  ||C_e||_F  = {frob_e:.4e}")
    print(f"  trace(C_b) = {np.trace(C_b):.4e}")
    print(f"  trace(C_e) = {np.trace(C_e):.4e}")
    print(f"  max |R_b - R_e| (off-diag) = {np.abs(R_b - R_e)[~np.eye(R_b.shape[0], dtype=bool)].max():.4f}")
    return {"diag_ratio": diag_ratio, "R_b": R_b, "R_e": R_e}


def _plot(tracer, C_b, C_e, b_ells, b_s, info, out_path):
    R_b, R_e = info["R_b"], info["R_e"]
    diag_b = np.sqrt(np.diag(C_b))
    diag_e = np.sqrt(np.diag(C_e))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{tracer}: bundle vs EZmock-sliced cov_ξ (DR1)", fontsize=13)

    # Row 0: correlation matrices + difference
    for ax, R, label in [(axes[0, 0], R_b, "bundle R"),
                          (axes[0, 1], R_e, "ezmock R"),
                          (axes[0, 2], R_b - R_e, "R_bundle − R_ezmock")]:
        vmin, vmax = (-1, 1) if "−" not in label else (-0.1, 0.1)
        cmap = "RdBu_r"
        im = ax.imshow(R, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046)
        # Mark ell boundary
        n_ell0 = len(b_s[0])
        ax.axhline(n_ell0 - 0.5, color="k", lw=0.5)
        ax.axvline(n_ell0 - 0.5, color="k", lw=0.5)

    # Row 1: per-ℓ σ overplot, σ ratio, full diag ratio
    n_per_ell = [len(s) for s in b_s]
    cur = 0
    for ie, (ell, n_ell, s_ell) in enumerate(zip(b_ells, n_per_ell, b_s)):
        ax = axes[1, 0]
        ax.plot(s_ell, diag_b[cur:cur + n_ell], "o-", label=f"bundle ℓ={ell}", color=f"C{ie}")
        ax.plot(s_ell, diag_e[cur:cur + n_ell], "x--", label=f"ezmock ℓ={ell}", color=f"C{ie}")
        ax.set_xlabel("s [Mpc/h]")
        ax.set_ylabel("σ(ξ_ℓ)")
        ax.set_yscale("log")
        ax.set_title("σ per s, per ℓ")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        ratio = diag_b[cur:cur + n_ell] / diag_e[cur:cur + n_ell]
        ax.plot(s_ell, ratio, "o-", label=f"ℓ={ell}", color=f"C{ie}")
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_xlabel("s [Mpc/h]")
        ax.set_ylabel("σ_bundle / σ_ezmock")
        ax.set_title("σ ratio")
        ax.legend(fontsize=8)
        cur += n_ell

    ax = axes[1, 2]
    # Off-diagonal correlation difference, flattened
    mask = ~np.eye(R_b.shape[0], dtype=bool)
    ax.scatter(R_e[mask], R_b[mask], s=4, alpha=0.4)
    lim = max(abs(R_b[mask]).max(), abs(R_e[mask]).max(), 0.1)
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.5)
    ax.set_xlabel("R_ezmock (off-diagonal)")
    ax.set_ylabel("R_bundle (off-diagonal)")
    ax.set_title("R off-diagonal: bundle vs ezmock")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved {out_path}")


def main():
    out_dir = Path(__file__).parent
    for tracer in ["LRG2", "QSO"]:
        C_b, b_ells, b_s = _load_bundle(tracer)
        C_e_full, ez_ells, ez_s = _load_ezmock_corrrec_block(tracer)
        C_e = _slice_ezmock_to_bundle(C_e_full, ez_ells, ez_s, b_ells, b_s)
        info = _summary(tracer, C_b, C_e)
        out_path = out_dir / f"compare_bundle_vs_ezmock_{tracer}.png"
        _plot(tracer, C_b, C_e, b_ells, b_s, info, out_path)


if __name__ == "__main__":
    main()
