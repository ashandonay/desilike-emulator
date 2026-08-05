#!/usr/bin/env python
"""MCMC on the ShapeFit likelihood, to test the Fisher approximation.

Our forecast targets come from a Fisher matrix -- the Gaussian approximation of
the likelihood AT THE PEAK. DESI's published 4x4 (2411.12021 Appendix A) is a
"Gaussian approximation of the ShapeFit covariances" fitted to an MCMC MARGINAL
posterior. Those are different objects whenever the likelihood is non-Gaussian
in the nuisance directions, and DESI say so explicitly (Section 4.5):

  - PWE (prior weight effect): the prior pulls away from what the data prefer.
  - PVE (prior volume effect): "for non-Gaussian likelihoods with partially
    degenerate parameters this can shift the peak of the marginal posterior away
    from the most-likely value".

A Fisher matrix cannot produce either -- it is by construction the curvature at
the peak. So a mismatch in the compressed covariance is EXPECTED at some level,
and this driver measures how much.

Motivation (shapefit CHANGELOG S57): every rho involving `m` is under-predicted
against DESI (mean |rho| ratio 0.356) while every rho not involving `m` is
over-predicted (1.399), with sigma(m) itself the best-matching of the four.
Three explanations were checked and eliminated first -- we already run
velocileptors REPT (S22), our counterterm priors are DESI's N(0, 12.5) exactly,
and `dn` is already fixed as DESI fix it. Projection effects are what is left.

Usage (from shapefit/, emulator env):
    python mcmc.py --tracers LRG2                     # one tracer, one seed
    python mcmc.py --tracers LRG2 ELG2 --seeds 1 2 3  # seed sweep
    python mcmc.py --tracers LRG2 --save-chain out.npz

Seed-sweep before trusting any single number: bao CHANGELOG and
`feedback_mcmc_chain_noise_sparse_tracers` both record that per-seed scatter is
the thing that bites, and it is worst for the sparse tracers.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import core as sf_core
import fourier_space as fs
import desi_reference as dr
from util import ntracers

_TRACERS = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")
_PARAMS = ("qiso", "qap", "f_sigmar", "m")
_PAIRS = ("qiso_qap", "qiso_f_sigmar", "qiso_m",
          "qap_f_sigmar", "qap_m", "f_sigmar_m")

FID = {"omega_cdm": 0.1200, "omega_b": 0.02237, "h": 0.6736,
       "ln10A_s": 3.036394, "n_s": 0.9649}


def cosmology(which: str) -> Dict[str, float]:
    """Input cosmology for the likelihood build.

    "fiducial"  DESI's AbacusSummit c000, what S57/S63 ran at.
    "dr1_map"   DESI's published ShapeFit-alone MAP (S70). Better motivated --
                the data was taken in the real universe, not the fiducial one --
                and worth +1.4 to +3.4% on sigma, uniformly toward DESI (S71).

    ⚠ Neither is exactly DESI's configuration. Their sigma come from an EZmock
    covariance built at the FIDUCIAL, with only the theory at the best fit;
    `build_shapefit_likelihood` uses one cosmology for both. See S71's table --
    the covariance-at-fiducial / derivatives-at-MAP case is unmeasured and its
    sign is not predictable from the other two.
    """
    if which == "fiducial":
        return dict(FID)
    if which == "dr1_map":
        return dr.dr1_bestfit_cosmology()
    raise ValueError(f"cosmology must be 'fiducial' or 'dr1_map', got {which!r}")


# ---------------------------------------------------------------------------
def build(tracer: str, theory: str = "rept", cosmo: str = "dr1_map"):
    """Likelihood at the chosen cosmology (see `cosmology`) and this tracer's
    DR1 N."""
    from desilike.theories.galaxy_clustering import (
        KaiserTracerPowerSpectrumMultipoles as _Kaiser)
    from desilike.theories.galaxy_clustering import (
        REPTVelocileptorsTracerPowerSpectrumMultipoles as _REPT)
    cls = {"kaiser": _Kaiser, "rept": _REPT}[theory]
    return sf_core.build_shapefit_likelihood(
        N_tracers=float(ntracers(tracer, "dr1")),
        theta_cosmo=sf_core._to_shapefit_cosmo_params(
            {**cosmology(cosmo), "N_tracers": float(ntracers(tracer, "dr1"))}),
        tracer_bin=tracer,
        theory_cls=cls,
        theory_kwargs=sf_core.default_theory_kwargs(cls, tracer),
    )


def _varied(likelihood):
    return [p for p in likelihood.all_params if p.varied and not p.derived]


def make_log_prob(likelihood):
    """(log_prob, names, fid) over the likelihood's varied parameters.

    Priors are taken from the parameters themselves, so the MCMC explores
    exactly the space the Fisher linearises -- otherwise the comparison
    measures a prior change rather than the Gaussian approximation.
    """
    params = _varied(likelihood)
    names = [str(p.name) for p in params]
    fid = {str(p.name): float(p.value) for p in params}
    priors = [p.prior for p in params]

    def log_prob(x):
        lp = 0.0
        for xi, pr in zip(x, priors):
            try:
                lpi = pr(xi)
            except Exception:
                return -np.inf
            if not np.isfinite(lpi):
                return -np.inf
            lp += lpi
        try:
            ll = float(likelihood(**dict(zip(names, x))))
        except Exception:
            return -np.inf
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    return log_prob, names, fid


def run_emcee(log_prob, names, fid, seed, nwalkers, niter, burn,
              checkpoint=None):
    """Returns (chain, acceptance, tau_max).

    `tau_max` is the worst integrated autocorrelation time over the varied
    parameters, computed with tol=0 so it is returned even when the chain is
    too short to trust it. Compare it against niter: emcee's own rule of thumb
    is niter > 50*tau. It is the only convergence handle here, and an overnight
    run that turns out to be 10 tau long is worth knowing about.
    """
    import emcee
    ndim = len(names)
    rng = np.random.default_rng(seed)
    scale = {"qiso": 0.01, "qap": 0.02, "df": 0.03, "dm": 0.03, "b1p": 0.05}
    p0 = np.empty((nwalkers, ndim))
    for i, nm in enumerate(names):
        s = scale.get(nm, 0.1)
        p0[:, i] = fid[nm] + s * rng.standard_normal(nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    # Heartbeat: a multi-hour run with no output is indistinguishable from a
    # hung one. Every 5% is cheap and makes the log auditable after the fact.
    #
    # CONVERGENCE STOP (S72). S63 ran a fixed 2500 and landed at 9-12 tau,
    # ~5x short of emcee's niter > 50*tau. Guessing a bigger fixed number is
    # the same mistake with a different constant: tau is not known in advance
    # and its ESTIMATE grows as the chain lengthens. So check periodically and
    # stop on emcee's standard two-part criterion -- long enough AND the tau
    # estimate has itself stopped moving.
    t0 = time.time()
    every = max(1, niter // 20)
    # Scales with the budget but stays small enough that a short smoke
    # run exercises the convergence branch instead of skipping it.
    check = max(50, niter // 40)
    tau_prev = np.inf
    tau = float("nan")
    converged_at = None
    n_done = 0
    for i, _ in enumerate(sampler.sample(p0, iterations=niter), start=1):
        n_done = i
        if i in (10, 25, 50) or i % every == 0 or i == niter:
            el = time.time() - t0
            print(f"    seed {seed}: {i}/{niter} ({i / niter:.0%})  "
                  f"{el / 60:.1f} min elapsed, {el / i * (niter - i) / 60:.1f} "
                  f"min left, acc {np.mean(sampler.acceptance_fraction):.3f}, "
                  f"tau {tau:.0f} ({i / tau:.1f}x)" if tau == tau else
                  f"    seed {seed}: {i}/{niter} ({i / niter:.0%})  "
                  f"{el / 60:.1f} min elapsed, "
                  f"acc {np.mean(sampler.acceptance_fraction):.3f}", flush=True)
        # Checkpoint the raw chain. A run with a 20k ceiling is ~2 days; a
        # reboot or a stray kill at hour 40 would otherwise lose everything,
        # since the JSON is only written when mcmc.py exits.
        if checkpoint is not None and i % (2 * check) == 0:
            np.savez(checkpoint, chain=sampler.get_chain(),
                     names=np.array(names), n_done=i, seed=seed,
                     acceptance=float(np.mean(sampler.acceptance_fraction)))
        if i % check or i < check:
            continue
        try:
            tau_now = float(np.max(sampler.get_autocorr_time(tol=0)))
        except Exception:
            continue
        if not np.isfinite(tau_now) or tau_now <= 0:
            continue
        tau = tau_now
        long_enough = i > 50.0 * tau
        settled = abs(tau_prev - tau) / tau < 0.01
        if long_enough and settled:
            converged_at = i
            print(f"    seed {seed}: CONVERGED at {i} iters "
                  f"(tau {tau:.0f}, {i / tau:.0f}x, d_tau "
                  f"{abs(tau_prev - tau) / tau:.4f})", flush=True)
            break
        tau_prev = tau

    # Burn-in is a fraction of what actually ran, not of the requested budget --
    # an early stop would otherwise discard the wrong slice.
    chain = sampler.get_chain()[int(burn * n_done):].reshape(-1, ndim)
    try:
        tau = float(np.max(sampler.get_autocorr_time(tol=0)))
    except Exception:
        tau = float("nan")
    return (chain, float(np.mean(sampler.acceptance_fraction)), tau,
            n_done, converged_at)


def compress(chain, names, info):
    """Chain -> the four physical targets, matching _sf_fisher_reduction:
        f_sigmar = df * f_sigmar_fid,  m = m_fid + dm  (unit Jacobian)."""
    idx = {n: i for i, n in enumerate(names)}
    fsr = float(info["f_sigmar_fid"])
    out = {
        "qiso": chain[:, idx["qiso"]],
        "qap": chain[:, idx["qap"]],
        "f_sigmar": chain[:, idx["df"]] * fsr,
        "m": chain[:, idx["dm"]],           # dm IS DESI's m (CHANGELOG S35)
    }
    return np.column_stack([out[p] for p in _PARAMS])


def summarize(samples, fsr_fid):
    C = np.cov(samples, rowvar=False)
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    out = {f"sigma_{p}": s[i] for i, p in enumerate(_PARAMS)}
    out["sigma_f_sigmar_frac"] = s[2] / fsr_fid
    for a in range(4):
        for b in range(a + 1, 4):
            out[f"rho_{_PARAMS[a]}_{_PARAMS[b]}"] = R[a, b]
    return out


# ---------------------------------------------------------------------------
def sweep(tracer, seeds, nwalkers, niter, burn, theory="rept", save=None,
          cosmo="dr1_map"):
    info = build(tracer, theory, cosmo=cosmo)
    fsr = float(info["f_sigmar_fid"])

    cov_f = fs._sf_fisher_reduction(info)
    sf = np.sqrt(np.diag(cov_f))
    Rf = cov_f / np.outer(sf, sf)
    fisher = {f"sigma_{p}": sf[i] for i, p in enumerate(_PARAMS)}
    fisher["sigma_f_sigmar_frac"] = sf[2] / fsr
    for a in range(4):
        for b in range(a + 1, 4):
            fisher[f"rho_{_PARAMS[a]}_{_PARAMS[b]}"] = Rf[a, b]

    log_prob, names, fid = make_log_prob(info["likelihood"])
    nw = nwalkers or max(2 * len(names) + 2, 32)
    print(f"\n{'=' * 72}\n  {tracer}  ({theory}, {cosmo}, {len(names)} varied, "
          f"{nw} walkers x <={niter} iters, burn {burn:.0%})\n{'=' * 72}")
    print(f"  varied: {', '.join(names)}")

    runs, diag = {}, {}
    for seed in seeds:
        ckpt = None
        if save:
            _p = Path(save)
            ckpt = _p.with_name(f"{_p.stem}_seed{seed}_partial{_p.suffix}")
        chain, acc, tau, n_ran, conv = run_emcee(
            log_prob, names, fid, seed, nw, niter, burn, checkpoint=ckpt)
        samp = compress(chain, names, info)
        runs[seed] = summarize(samp, fsr)
        diag[seed] = {"acceptance": acc, "tau_max": tau,
                      "n_samples": len(chain), "niter": n_ran,
                      "niter_max": niter, "converged_at": conv,
                      "converged": conv is not None, "cosmology": cosmo,
                      "nwalkers": nw,
                      "iter_per_tau": n_ran / tau if tau else np.nan}
        print(f"  seed {seed}: {len(chain)} samples, acceptance {acc:.2f}, "
              f"tau_max {tau:.1f} ({n_ran / tau:.0f} tau of chain), "
              f"{'CONVERGED' if conv else 'NOT converged'} after {n_ran} iters"
              if np.isfinite(tau) else
              f"  seed {seed}: {len(chain)} samples, acceptance {acc:.2f}")
        if save:
            # One file per seed. The loop previously wrote every seed to the
            # same path, so a sweep kept only the last chain.
            p = Path(save)
            p = p.with_name(f"{p.stem}_seed{seed}{p.suffix}")
            np.savez(p, chain=chain, names=np.array(names), samples=samp)
            print(f"    wrote {p}")
            if ckpt is not None and ckpt.exists():
                ckpt.unlink()      # the final file supersedes the checkpoint

    ref = dr.sigma_targets(tracer)
    keys = ([f"sigma_{p}" for p in ("qiso", "qap")] + ["sigma_f_sigmar_frac",
            "sigma_m"] + [f"rho_{p}" for p in _PAIRS])
    print(f"\n  {'quantity':22s} {'Fisher':>9s} {'MCMC':>9s} "
          f"{'seed rms':>9s} {'DESI':>9s} {'F/DESI':>8s} {'M/DESI':>8s}")
    for k in keys:
        vals = np.array([runs[s][k] for s in seeds])
        m, sd = vals.mean(), (vals.std() if len(vals) > 1 else 0.0)
        d = ref.get(k, np.nan)
        star = " *" if k.endswith("_m") and k.startswith("rho") else ""
        print(f"  {k:22s} {fisher[k]:9.4f} {m:9.4f} {sd:9.4f} {d:9.4f} "
              f"{fisher[k]/d:8.3f} {m/d:8.3f}{star}")
    return {"fisher": fisher, "mcmc": runs, "desi": ref, "diag": diag}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", nargs="+", default=["LRG2"], choices=_TRACERS)
    ap.add_argument("--theory", choices=["kaiser", "rept"], default="rept")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--nwalkers", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=4000)
    ap.add_argument("--burnin-frac", type=float, default=0.4)
    ap.add_argument("--save-chain", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--cosmology", choices=["dr1_map", "fiducial"],
                    default="dr1_map",
                    help="Input cosmology. dr1_map (default) is DESI's "
                         "published ShapeFit-alone MAP (S70/S71); fiducial is "
                         "AbacusSummit c000, what S57/S63 ran at.")
    a = ap.parse_args()

    out = {}
    for t in a.tracers:
        out[t] = sweep(t, a.seeds, a.nwalkers, a.max_iterations,
                       a.burnin_frac, a.theory, a.save_chain,
                       cosmo=a.cosmology)
    if a.json:
        import json
        ser = {t: {"fisher": v["fisher"],
                   "mcmc": {str(s): d for s, d in v["mcmc"].items()},
                   "desi": v["desi"],
                   "diag": {str(s): d for s, d in v["diag"].items()}}
               for t, v in out.items()}
        a.json.write_text(json.dumps(ser, indent=2, default=float))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
