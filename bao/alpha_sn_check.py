"""ONE-OFF PROOF (not a pipeline component) — show that the gap between our
first-principles Gaussian ξ-cov and the DESI bundle (RascalC) cov IS the
non-Gaussian / shot-noise piece that RascalC's α_SN rescaling (arXiv:2411.12027
eq 2.7) parametrizes.

Mechanism: the analytic Gaussian per-mode variance (Grieb 2016 / FKP) splits into
three physical channels,
    σ²(k) = [mean_LLP2·I44]  +  [2·mean_LLP·I34]  +  [mean_LL·I24]
              sample variance      P·shot cross        pure shot (1/n̄²)
RascalC's α_SN rescales the shot density (eq 2.7: ⟨δ²⟩ → α_SN ⟨δ²⟩), so it
multiplies the cross term by α_SN and the pure-shot term by α_SN². We fit ONE
α_SN per tracer to the bundle cov (Frobenius), with NO reference to σ, then check:
  (a) does α_SN land in RascalC's published ~1.0–1.4 range?
  (b) does the inflated cov match the bundle cov far better than the Gaussian?
  (c) does it independently bring σ(DH/DM/DV) onto DESI?

This is a fudge (α_SN is fit to DESI data) and deliberately NOT cosmology/N_tracers
transferable — it only demonstrates the missing piece. See feedback memories.

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python alpha_sn_check.py
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import core
import config_space as cc
import reference_sigmas as rs
from util import TRACER_CONFIGS

_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_QS = [("DH_over_rs", "DH/rd"), ("DM_over_rs", "DM/rd"), ("DV_over_rs", "DV/rd")]


def _rel_fro(C, Cb):
    return float(np.linalg.norm(C - Cb) / np.linalg.norm(Cb))


def _analyze(tracer):
    cfg = TRACER_CONFIGS[tracer]
    apmode = "qiso" if tracer in core._ISO_TRACERS else "qparqper"
    theta, hrdrag = core._to_bao_cosmo_params({**core.PARAM_DEFAULTS, **cc._FID})
    info = core.build_bao_likelihood(
        N_tracers=cc._get_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]),
        area=cc._AREA, apmode=apmode, lean=True)

    cosmo = core.get_cosmo(("DESI", dict(theta)))
    slices = cc.load_nz_slices(tracer_bin=tracer, cosmo=cosmo, area_deg2=cc._AREA,
                               N_design=float(cc._get_ntracers(tracer)))
    P_wide = cc._wide_Pell(tracer, info)

    bundle = cc.load_bundle(tracer)
    Cb = 0.5 * (bundle["cov"] + bundle["cov"].T)
    W = bundle["W"]
    th_ells = list(bundle["th_ells"]); s_th = np.asarray(bundle["th_s"][0])
    ds_th = float(np.mean(np.diff(s_th)))
    jbar = cc.build_jbar_cache(th_ells, cc.K_WIDE, s_th, ds_th)

    sv, cross, shot = cc.per_mode_sigma2(cc.K_WIDE, P_wide, (0, 2, 4),
                                         tuple(th_ells), slices, return_channels=True)
    Csv = W @ cc.xi_cov_from_sigma2(s_th, th_ells, cc.K_WIDE, sv, jbar_cache=jbar) @ W.T
    Ccr = W @ cc.xi_cov_from_sigma2(s_th, th_ells, cc.K_WIDE, cross, jbar_cache=jbar) @ W.T
    Csh = W @ cc.xi_cov_from_sigma2(s_th, th_ells, cc.K_WIDE, shot, jbar_cache=jbar) @ W.T

    def C_of_alpha(a):
        M = Csv + a * Ccr + a * a * Csh
        return 0.5 * (M + M.T)

    Cg = C_of_alpha(1.0)  # = the first-principles Gaussian cov

    # (1) fit α_SN to the bundle cov (Frobenius, NO σ involved).
    res = minimize_scalar(lambda a: _rel_fro(C_of_alpha(a), Cb),
                          bounds=(0.3, 5.0), method="bounded")
    a_fit = float(res.x)
    # (2) best single global scalar (whole-matrix amplitude) for comparison.
    s_glob = float(np.sum(Cg * Cb) / np.sum(Cg * Cg))

    # σ(DH/DM/DV) for Gaussian, α_SN-inflated, global-scalar, and bundle; DESI recon ref.
    sig_g = cc.bundle_fisher_sigmas(tracer, cov_override=Cg)
    sig_a = cc.bundle_fisher_sigmas(tracer, cov_override=C_of_alpha(a_fit))
    sig_s = cc.bundle_fisher_sigmas(tracer, cov_override=s_glob * Cg)
    sig_b = cc.bundle_fisher_sigmas(tracer)            # DESI bundle cov
    recon = rs._recon_sigmas(tracer, sig_b)

    shot_frac = float(np.mean(np.diag(Csh) / np.diag(Cg)))  # how shot-dominated
    Ca = C_of_alpha(a_fit)
    return {
        "a_fit": a_fit, "s_glob": s_glob,
        "fro_gauss": _rel_fro(Cg, Cb), "fro_alpha": _rel_fro(Ca, Cb),
        "fro_glob": _rel_fro(s_glob * Cg, Cb), "shot_frac": shot_frac,
        "sig_g": sig_g, "sig_a": sig_a, "sig_s": sig_s, "sig_b": sig_b, "recon": recon,
        "sparse": apmode == "qiso",
        "Cg": Cg, "Ca": Ca, "Cb": Cb, "n_ell": len(th_ells) if False else len(bundle["obs_ells"]),
    }


def _corr(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def main():
    R = {}
    for t in _TRACERS:
        print(f"== {t} ==", flush=True)
        R[t] = _analyze(t)

    print("\n=== Fudge fit to the DESI bundle cov  (‖ΔC‖_F / ‖C_bundle‖_F) ===")
    print(f"  {'tracer':<11} {'shot-frac':>9} | {'α_SN':>6} {'resid':>6} | "
          f"{'scalar s':>9} {'√s':>5} {'resid':>6}")
    print("  " + "-" * 64)
    for t in _TRACERS:
        d = R[t]
        print(f"  {t:<11} {d['shot_frac']:>9.2f} | {d['a_fit']:>6.3f} {d['fro_alpha']:>6.3f} | "
              f"{d['s_glob']:>9.3f} {np.sqrt(d['s_glob']):>5.3f} {d['fro_glob']:>6.3f}   "
              f"(gauss resid {d['fro_gauss']:.3f})")

    print("\n=== σ recovered: Gaussian → α_SN → scalar·C, vs DESI bundle / recon ===")
    print(f"  {'tracer':<11} {'q':<6} {'σ_gauss':>8} {'σ_αSN':>8} {'σ_scal':>8} "
          f"{'σ_bundle':>9} {'σ_recon':>8} {'g/D':>6} {'αSN/D':>6} {'scal/D':>6}")
    print("  " + "-" * 88)
    for t in _TRACERS:
        d = R[t]
        for q, qs in _QS:
            sD = d["recon"][q]
            if not np.isfinite(sD):
                continue
            g, a, s, b = d["sig_g"][q], d["sig_a"][q], d["sig_s"][q], d["sig_b"][q]
            print(f"  {t:<11} {qs:<6} {g:>8.4f} {a:>8.4f} {s:>8.4f} {b:>9.4f} {sD:>8.4f} "
                  f"{g / sD:>6.3f} {a / sD:>6.3f} {s / sD:>6.3f}")

    # --- decompose the cov mismatch into amplitude (diagonal) vs structure (corr) ---
    print("\n=== matrix mismatch decomposed: diagonal AMPLITUDE vs CORRELATION structure ===")
    print(f"  {'tracer':<11} {'shot-frac':>9} | {'√diag ratio→bundle':>20} | "
          f"{'corr resid ‖R-Rb‖/‖Rb‖':>24}")
    print(f"  {'':<11} {'':>9} | {'gauss':>9} {'α_SN':>9} | {'gauss':>11} {'α_SN':>11}")
    print("  " + "-" * 76)
    for t in _TRACERS:
        d = R[t]
        Cg, Ca, Cb = d["Cg"], d["Ca"], d["Cb"]
        amp_g = float(np.median(np.sqrt(np.diag(Cg)) / np.sqrt(np.diag(Cb))))
        amp_a = float(np.median(np.sqrt(np.diag(Ca)) / np.sqrt(np.diag(Cb))))
        Rg, Ra, Rb = _corr(Cg), _corr(Ca), _corr(Cb)
        cr_g = float(np.linalg.norm(Rg - Rb) / np.linalg.norm(Rb))
        cr_a = float(np.linalg.norm(Ra - Rb) / np.linalg.norm(Rb))
        flag = "  <-- α_SN worsens structure" if cr_a > cr_g + 1e-3 else ""
        print(f"  {t:<11} {d['shot_frac']:>9.2f} | {amp_g:>9.3f} {amp_a:>9.3f} | "
              f"{cr_g:>11.3f} {cr_a:>11.3f}{flag}")

    _plot_corr(R)


def _plot_corr(R):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(_TRACERS), 3, figsize=(11, 3.1 * len(_TRACERS)),
                             constrained_layout=True)
    for row, t in enumerate(_TRACERS):
        d = R[t]
        mats = [("DESI bundle", _corr(d["Cb"])),
                ("Gaussian (α=1)", _corr(d["Cg"])),
                (f"α_SN={d['a_fit']:.2f}", _corr(d["Ca"]))]
        for col, (label, Rm) in enumerate(mats):
            ax = axes[row, col]
            im = ax.imshow(Rm, cmap="seismic", vmin=-1, vmax=1, origin="upper")
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{t}\n(shot {d['shot_frac']:.0%})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.4, label="correlation")
    out = Path(__file__).resolve().parent / "alpha_sn_cov_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved correlation-matrix comparison to: {out}")


if __name__ == "__main__":
    main()
