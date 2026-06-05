"""BAO MCMC — one CLI for both the config-space (ξ) and Fourier-space (P)
forecasts. Pick the frame with ``--space {config,fourier}``.

  config  : native ξ-theory + Grieb/bundle covariance, sampled with raw emcee on
            the log-prob from config_space.make_log_prob.  --cov gaussxi|bundle|both.
  fourier : the desilike P-likelihood from core.build_bao_likelihood, sampled with
            desilike's EmceeSampler; bundle cov is substituted as the precision via
            precision = Mᵀ cov_ξ⁻¹ M, M = W·H_Hankel.  --cov analytic|bundle|both.

Both frames reduce each chain to σ(DH/rd), σ(DM/rd), σ(DV/rd) and write the same
combined seed-sweep file:

  mcmc_{space}.json   {tracer: {cov: {seed: {DH,DM,DV}}}}   — the full --seeds
          sweep, merge-updated per (tracer, cov) so partial --tracers/--cov runs
          accumulate safely. Consumed by comparison_plots.py forecast for both the
          scatter points (mean over seeds) and the σ-of-σ error bars.

Sparse tracers (BGS, QSO) use the isotropic qiso fit — DH/DM are degenerate and
reported NaN; DV is the constrained quantity. Reference σ is DESI bao-recon
stat-only (reference_sigmas.py), NOT desi_data.csv.

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python mcmc.py --space config
    # fourier, bundle cov, seed sweep (mean ± σ-of-σ error bars in the plot):
    ... mcmc.py --space fourier --cov bundle --seeds 42 43 44 45 46
    # one tracer (merge-updates just LRG1 in the combined file):
    ... mcmc.py --space config --tracers LRG1 --seeds 42 43
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

print = functools.partial(print, flush=True)

import h5py
import numpy as np
import pandas as pd
from scipy.special import spherical_jn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from reference_sigmas import _recon_sigmas
from util import TRACER_CONFIGS

_HERE = Path(__file__).resolve().parent
_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_ISO = ("BGS", "QSO")

# config-space sampler default
_BB_BASIS = "desi_adame24"

# fourier-space constants
_FID = {"Om": 0.3152, "hrdrag": 99.08}            # canonical fiducial (reconciled)
_DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}
_DR1_DIR = Path.home() / "data" / "desi" / "bao_dr1"
_LIK_DIR = _DR1_DIR / "likelihoods"
_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
_CHAIN_DIR = _HERE / "chains_fourier"   # only used with --save-chain (fourier)
_BUNDLES = {
    "BGS":       "likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    "LRG1":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5",
    "LRG2":      "likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5",
    "LRG3_ELG1": "likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    "ELG2":      "likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    "QSO":       "likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5",
}

# cov labels valid per space; "both" expands to the two real labels.
_COVS = {"config": ["gaussxi", "bundle"], "fourier": ["analytic", "bundle"]}


# ===========================================================================
# shared
# ===========================================================================
def _reduce_sigmas(samples, apmode, DH, DM, DV):
    """σ(DH/rd), σ(DM/rd), σ(DV/rd) from per-parameter sample arrays. `samples`
    maps param name → 1-D chain (post burn-in). qiso reports DH/DM as NaN."""
    if apmode == "qiso":
        sq = float(np.asarray(samples["qiso"]).std(ddof=1))
        return {"DH": float("nan"), "DM": float("nan"), "DV": sq * DV}
    qp = np.asarray(samples["qpar"])
    qq = np.asarray(samples["qper"])
    cov = np.cov(qp, qq, ddof=1)
    g = np.array([1.0 / 3.0, 2.0 / 3.0])
    return {"DH": float(qp.std(ddof=1) * DH), "DM": float(qq.std(ddof=1) * DM),
            "DV": float(np.sqrt(max(g @ cov @ g, 0.0)) * DV)}


def _recon_ref(tracer, DH, DM, DV):
    """DESI bao-recon stat-only reference σ; print and return (rDH, rDM, rDV)."""
    ref = _recon_sigmas(tracer, {"DH_over_rd_fid": DH, "DM_over_rd_fid": DM,
                                 "DV_over_rd_fid": DV})
    rDH, rDM, rDV = (ref.get("DH_over_rs", np.nan), ref.get("DM_over_rs", np.nan),
                     ref.get("DV_over_rs", np.nan))
    print(f"  recon σ: DH={rDH:.4f}  DM={rDM:.4f}  DV={rDV:.4f}")
    return rDH, rDM, rDV


def _print_seed(tracer, cov, seed, sig, ref):
    rDH, rDM, rDV = ref
    print(f"  [{tracer} {cov} seed={seed}] "
          f"DH={sig['DH']:.4f} (M/D={sig['DH']/rDH:.3f})  "
          f"DM={sig['DM']:.4f} (M/D={sig['DM']/rDM:.3f})  "
          f"DV={sig['DV']:.4f} (M/D={sig['DV']/rDV:.3f})")


# ===========================================================================
# config-space sweep
# ===========================================================================
def _sweep_config(tracer, covs, seeds, nwalkers, niter, burn):
    """{cov: {seed: {DH,DM,DV}}} for the config-space (native-ξ) frame."""
    import config_space as M

    apmode = "qiso" if tracer in _ISO else "qparqper"
    info, native, bundle = M.build_native_theory_mcmc(tracer, apmode)
    Cb = np.asarray(bundle["cov"]).copy()
    _, Cg = M.gaussxi_cov_on_bundle_grid(tracer, info, bundle)
    cov_map = {"gaussxi": Cg, "bundle": Cb}

    tmpl = info["template"]
    z = float(TRACER_CONFIGS[tracer]["z_eff"])
    DH, DM = float(tmpl.DH_over_rd_fid), float(tmpl.DM_over_rd_fid)
    DV = (z * DM * DM * DH) ** (1.0 / 3.0)
    print(f"\n{'=' * 70}\n  {tracer}  (config, apmode={apmode})\n{'=' * 70}")
    ref = _recon_ref(tracer, DH, DM, DV)

    out = {}
    for label in covs:
        bundle["cov"] = cov_map[label]
        log_prob, names, fid = M.make_log_prob(info, native, bundle, apmode, tracer,
                                               bb_basis=_BB_BASIS)
        out[label] = {}
        for seed in seeds:
            flat = _run_emcee(log_prob, names, fid, seed, nwalkers, niter, burn)
            samples = {p: flat[:, i] for i, p in enumerate(names)}
            sig = _reduce_sigmas(samples, apmode, DH, DM, DV)
            out[label][str(seed)] = sig
            _print_seed(tracer, label, seed, sig, ref)
    return out


def _run_emcee(log_prob, names, fid, seed, nwalkers, niter, burn):
    import emcee
    ndim = len(names)
    rng = np.random.default_rng(seed)
    p0 = np.empty((nwalkers, ndim))
    for i, p in enumerate(names):
        if p in ("qiso", "qpar", "qper"):
            p0[:, i] = 1.0 + 0.02 * rng.standard_normal(nwalkers)
        elif p in ("sigmas", "sigmapar", "sigmaper"):
            p0[:, i] = np.maximum(fid[p] + 0.2 * rng.standard_normal(nwalkers), 0.2)
        else:
            p0[:, i] = fid[p] + 0.05 * rng.standard_normal(nwalkers)
    s = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    s.run_mcmc(p0, niter, progress=False)
    return s.get_chain()[int(burn * niter):].reshape(-1, ndim)


# ===========================================================================
# fourier-space sweep (+ bundle cov → precision machinery)
# ===========================================================================
def _get_ntracers(dataset, tracer):
    df = pd.read_csv(Path.home() / "data" / "desi" / f"bao_{dataset}" / "desi_data.csv")
    df = df[["tracer", "passed"]].drop_duplicates("tracer")
    n_by = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    return n_by[_NAME_MAP.get(tracer, tracer)]


def _load_bundle(tracer):
    p = _LIK_DIR / _BUNDLES[tracer]
    with h5py.File(p, "r") as f:
        cov_xi = np.asarray(f["covariance/value"][...], dtype=np.float64)
        W = np.asarray(f["window/value"][...], dtype=np.float64)
        theory_ells, theory_s = [], []
        for k in sorted(f["window/theory"].keys(), key=lambda k: (not k.isdigit(), k)):
            if k.isdigit():
                theory_ells.append(int(f[f"window/theory/{k}/meta/ell"][()]))
                theory_s.append(np.asarray(f[f"window/theory/{k}/s"][...], dtype=np.float64))
    return cov_xi, W, theory_ells, theory_s


def _hankel_matrix(ell, k, dk, s):
    sign = (1j ** ell).real
    H = np.empty((len(s), len(k)), dtype=np.float64)
    for i, si in enumerate(s):
        H[i, :] = sign * (k ** 2) * spherical_jn(ell, k * si) * dk / (2.0 * np.pi ** 2)
    return H


def _build_M(W, theory_ells, theory_s_per_ell, pipe_ells, k_centers, dk):
    n_k = len(k_centers)
    n_pipe = n_k * len(pipe_ells)
    n_per = [len(s) for s in theory_s_per_ell]
    H_block = np.zeros((sum(n_per), n_pipe), dtype=np.float64)
    row = 0
    for it, te in enumerate(theory_ells):
        if te in pipe_ells:
            ip = pipe_ells.index(te)
            H_ell = _hankel_matrix(te, k_centers, dk, theory_s_per_ell[it])
            H_block[row:row + n_per[it], ip * n_k:(ip + 1) * n_k] = H_ell
        row += n_per[it]
    return W @ H_block


def _precision_from_cov_xi(cov_xi, M, regularize=1e-10):
    C = 0.5 * (cov_xi + cov_xi.T)
    ridge = regularize * np.trace(C) / C.shape[0]
    C_inv = np.linalg.inv(C + ridge * np.eye(C.shape[0]))
    C_inv = 0.5 * (C_inv + C_inv.T)
    P = M.T @ C_inv @ M
    return 0.5 * (P + P.T)


def _sweep_fourier(tracer, covs, seeds, nwalkers, niter, burn, *,
                   dataset, omega_m, hrdrag, alpha_low, alpha_high, apmode_arg,
                   save_chain):
    """{cov: {seed: {DH,DM,DV}}} for the Fourier-space (desilike-P) frame."""
    import core as pc
    from desilike.samplers import EmceeSampler

    apmode = ("qiso" if tracer in _ISO else "qparqper") if apmode_arg == "auto" else apmode_arg
    out = {}
    for cov in covs:
        area = _DATASET_AREAS["dr1"] if cov == "bundle" else _DATASET_AREAS[dataset]
        print(f"\n{'=' * 70}\n  {tracer}  (fourier, cov={cov}, apmode={apmode}, "
              f"area={area:.0f})\n{'=' * 70}")

        theta, hrdrag_eff = pc._to_bao_cosmo_params(
            {**pc.PARAM_DEFAULTS, "Om": omega_m, "hrdrag": hrdrag})
        cfg = TRACER_CONFIGS[tracer]
        info = pc.build_bao_likelihood(
            N_tracers=_get_ntracers("dr1" if cov == "bundle" else dataset, tracer),
            theta_cosmo=theta, hrdrag=hrdrag_eff, tracer_bin=tracer,
            zrange=cfg["zrange"], z_eff=float(cfg["z_eff"]), area=area, apmode=apmode)
        likelihood = info["likelihood"]
        template = info["template"]

        if cov == "bundle":
            obs = info["observable"]
            cov_xi, W, th_ells, th_s = _load_bundle(tracer)
            k = np.asarray(obs.k[0], dtype=np.float64)
            dk = float(np.mean(np.diff(k)))
            pipe_ells = [int(l) for l in obs.ells]
            M = _build_M(W, th_ells, th_s, pipe_ells, k, dk)
            prec = _precision_from_cov_xi(cov_xi, M)
            print(f"  M {M.shape}, cov_ξ {cov_xi.shape}, precision {prec.shape}; "
                  f"bundle ells {th_ells}, pipe ells {pipe_ells}")
            likelihood.precision = np.asarray(prec, dtype=np.float64)
            if hasattr(likelihood, "_precision_input"):
                likelihood._precision_input = likelihood.precision.copy()
            if hasattr(likelihood, "precision_hartlap2007"):
                likelihood.precision_hartlap2007 = likelihood.precision.copy()

        ap_params = ["qiso"] if apmode == "qiso" else ["qpar", "qper"]
        for p in ap_params:
            likelihood.all_params[p].update(
                prior={"dist": "uniform", "limits": [alpha_low, alpha_high]},
                ref={"dist": "norm", "loc": 1.0, "scale": 0.02})

        DH = float(template.DH_over_rd_fid)
        DM = float(template.DM_over_rd_fid)
        DV = float(template.DV_over_rd_fid)
        ref = _recon_ref(tracer, DH, DM, DV)

        out[cov] = {}
        for seed in seeds:
            sampler_kwargs = dict(nwalkers=nwalkers, seed=seed)
            if save_chain:
                _CHAIN_DIR.mkdir(exist_ok=True)
                sampler_kwargs["save_fn"] = str(_CHAIN_DIR / f"{tracer}_{dataset}_{cov}_seed{seed}.h5")
            sampler = EmceeSampler(likelihood, **sampler_kwargs)
            print(f"  emcee: {nwalkers} walkers × {niter} iter, seed={seed}")
            chain = sampler.run(max_iterations=niter, check=False)[0]
            n_burn = int(burn * chain.shape[0])
            samples = {p: np.asarray(chain[p][n_burn:]).ravel() for p in ap_params}
            sig = _reduce_sigmas(samples, apmode, DH, DM, DV)
            out[cov][str(seed)] = sig
            _print_seed(tracer, cov, seed, sig, ref)
    return out


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", choices=["config", "fourier"], required=True,
                    help="config = native-ξ frame; fourier = desilike-P frame.")
    ap.add_argument("--tracers", nargs="+", default=_TRACERS, choices=_TRACERS)
    ap.add_argument("--cov", default="both",
                    choices=["both", "gaussxi", "analytic", "bundle"],
                    help="config: gaussxi|bundle|both;  fourier: analytic|bundle|both.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--nwalkers", type=int, default=None,
                    help="default 48 (config) / 64 (fourier).")
    ap.add_argument("--max-iterations", type=int, default=3000)
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    # fourier-only knobs (ignored for --space config)
    ap.add_argument("--dataset", choices=list(_DATASET_AREAS.keys()), default="dr1",
                    help="[fourier] survey area for analytic cov (bundle forces dr1).")
    ap.add_argument("--apmode", choices=["auto", "qparqper", "qiso"], default="auto",
                    help="[fourier] auto = qiso for BGS/QSO, qparqper otherwise.")
    ap.add_argument("--omega-m", type=float, default=_FID["Om"], help="[fourier]")
    ap.add_argument("--hrdrag", type=float, default=_FID["hrdrag"], help="[fourier]")
    ap.add_argument("--alpha-low", type=float, default=0.8, help="[fourier] AP prior.")
    ap.add_argument("--alpha-high", type=float, default=1.2, help="[fourier] AP prior.")
    ap.add_argument("--save-chain", action="store_true",
                    help="[fourier] dump chains to chains_fourier/{tracer}_{dataset}_{cov}_seed{seed}.h5.")
    args = ap.parse_args()

    valid = _COVS[args.space]
    covs = list(valid) if args.cov == "both" else [args.cov]
    bad = [c for c in covs if c not in valid]
    if bad:
        ap.error(f"--cov {bad[0]} invalid for --space {args.space} "
                 f"(choose from {valid + ['both']}).")
    nwalkers = args.nwalkers if args.nwalkers is not None else (48 if args.space == "config" else 64)

    out_path = _HERE / f"mcmc_{args.space}.json"
    combined = json.loads(out_path.read_text()) if out_path.exists() else {}
    combined["_meta"] = {
        "source": f"mcmc.py --space {args.space} (full seed sweep)",
        "settings": f"seeds={args.seeds}, nwalkers={nwalkers}, "
                    f"max_iter={args.max_iterations}, burnin={args.burnin_frac}"
                    + (f", bb_basis={_BB_BASIS}" if args.space == "config" else ""),
        "note": ("config: gaussxi = analytic Grieb cov, bundle = DESI RascalC. "
                 if args.space == "config" else
                 "fourier: analytic = pipeline Gaussian cov, bundle = DESI RascalC cov_ξ substitution. ")
                + "qiso (BGS,QSO): DV only. Per tracer: {cov: {seed: {DH,DM,DV}}}.",
    }
    for t in args.tracers:
        if args.space == "config":
            sweep = _sweep_config(t, covs, args.seeds, nwalkers,
                                  args.max_iterations, args.burnin_frac)
        else:
            sweep = _sweep_fourier(t, covs, args.seeds, nwalkers,
                                   args.max_iterations, args.burnin_frac,
                                   dataset=args.dataset, omega_m=args.omega_m,
                                   hrdrag=args.hrdrag, alpha_low=args.alpha_low,
                                   alpha_high=args.alpha_high, apmode_arg=args.apmode,
                                   save_chain=args.save_chain)
        # merge per cov so a partial --cov run doesn't wipe the other cov's sweep
        combined.setdefault(t, {}).update(sweep)
        out_path.write_text(json.dumps(combined, indent=2))
        print(f"wrote {t} -> mcmc_{args.space}.json")


if __name__ == "__main__":
    main()
