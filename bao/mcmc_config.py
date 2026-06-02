"""Config-space (ξ) BAO MCMC — native ξ-theory + Grieb/bundle covariance, via
config_space.py §4 (make_log_prob + emcee on the bundle data vector).

Covariance (``--cov``):
  gaussxi : first-principles analytic Grieb Gaussian ξ-cov (the forecast cov).
  bundle  : DESI DR1 RascalC bundle cov.
  both    : run both (default) — the cov is the only thing that differs.

Output (``--out``):
  money : one seed (seeds[0]), all --tracers, both covs ->
          mcmc_config.json   {tracer: {cov: {DH,DM,DV}}}
          (consumed by plot_xi_money.py for the scatter points)
  sweep : all --seeds, per tracer ->
          seed_sweep_xi/{tracer}.json   {cov: {seed: {DH,DM,DV}}}
          (consumed by aggregate_seed_sweep.py and plot_xi_money.py error bars)

Sparse tracers (BGS, QSO) use the isotropic qiso fit (DH/DM reported NaN; DV is
the constrained quantity). The Fourier-space analog is mcmc_fourier.py.

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python mcmc_config.py --out money
    # seed sweep for one tracer:
    ... mcmc_config.py --out sweep --tracers LRG1 --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import functools
print = functools.partial(print, flush=True)

import numpy as np
import emcee

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import config_space as M
from util import TRACER_CONFIGS

_HERE = Path(__file__).resolve().parent
NW = 48
NIT = 3000
BURN = 0.3
BB_BASIS = "desi_adame24"
_TRACERS = ["BGS", "LRG1", "LRG2", "LRG3_ELG1", "ELG2", "QSO"]
_ISO = ("BGS", "QSO")


def _sigmas(flat, names, apmode, tmpl, z):
    """Posterior σ(DH/rd), σ(DM/rd), σ(DV/rd) from a flattened chain. qiso
    tracers report DH/DM as NaN (degenerate); DV is the constrained direction."""
    DH = float(tmpl.DH_over_rd_fid)
    DM = float(tmpl.DM_over_rd_fid)
    DV = (z * DM * DM * DH) ** (1.0 / 3.0)
    if apmode == "qiso":
        sq = float(flat[:, names.index("qiso")].std(ddof=1))
        return {"DH": float("nan"), "DM": float("nan"), "DV": sq * DV}
    qp = flat[:, names.index("qpar")]
    qq = flat[:, names.index("qper")]
    cov = np.cov(qp, qq, ddof=1)
    g = np.array([1.0 / 3.0, 2.0 / 3.0])
    return {"DH": float(qp.std(ddof=1) * DH), "DM": float(qq.std(ddof=1) * DM),
            "DV": float(np.sqrt(max(g @ cov @ g, 0.0)) * DV)}


def _run_chain(log_prob, names, fid, seed):
    ndim = len(names)
    rng = np.random.default_rng(seed)
    p0 = np.empty((NW, ndim))
    for i, p in enumerate(names):
        if p in ("qiso", "qpar", "qper"):
            p0[:, i] = 1.0 + 0.02 * rng.standard_normal(NW)
        elif p in ("sigmas", "sigmapar", "sigmaper"):
            p0[:, i] = np.maximum(fid[p] + 0.2 * rng.standard_normal(NW), 0.2)
        else:
            p0[:, i] = fid[p] + 0.05 * rng.standard_normal(NW)
    s = emcee.EnsembleSampler(NW, ndim, log_prob)
    s.run_mcmc(p0, NIT, progress=False)
    return s.get_chain()[int(BURN * NIT):].reshape(-1, ndim)


def _build(tracer):
    """Build (info, native, bundle, {gaussxi, bundle} covs, tmpl, z, apmode) once."""
    apmode = "qiso" if tracer in _ISO else "qparqper"
    info, native, bundle = M.build_native_theory_mcmc(tracer, apmode)
    Cb = np.asarray(bundle["cov"]).copy()
    _, Cg = M.gaussxi_cov_on_bundle_grid(tracer, info, bundle)
    cov_map = {"gaussxi": Cg, "bundle": Cb}
    return info, native, bundle, cov_map, info["template"], float(TRACER_CONFIGS[tracer]["z_eff"]), apmode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", nargs="+", default=_TRACERS, choices=_TRACERS)
    ap.add_argument("--cov", choices=["gaussxi", "bundle", "both"], default="both")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", choices=["money", "sweep"], default="money",
                    help="money: 1 seed, all tracers -> mcmc_config.json; "
                         "sweep: all seeds, per tracer -> seed_sweep_xi/{tracer}.json")
    args = ap.parse_args()
    covs = ["gaussxi", "bundle"] if args.cov == "both" else [args.cov]

    if args.out == "money":
        seed = args.seeds[0]
        out = {"_meta": {
            "source": "mcmc_config.py --out money (config_space native-ξ MCMC, single seed)",
            "settings": f"SEED={seed}, NW={NW}, NIT={NIT}, BURN={BURN}, bb_basis={BB_BASIS}",
            "note": "gaussxi = analytic Grieb cov; bundle = DESI RascalC. qiso (BGS,QSO): DV only.",
        }}
        for t in args.tracers:
            info, native, bundle, cov_map, tmpl, z, apmode = _build(t)
            rec = {}
            for label in covs:
                bundle["cov"] = cov_map[label]
                log_prob, names, fid = M.make_log_prob(info, native, bundle, apmode, t,
                                                       bb_basis=BB_BASIS)
                rec[label] = _sigmas(_run_chain(log_prob, names, fid, seed), names, apmode, tmpl, z)
                print(f"[{t} {label}] {rec[label]}")
            out[t] = rec
        (_HERE / "mcmc_config.json").write_text(json.dumps(out, indent=2))
        print("\nwrote mcmc_config.json")
    else:  # sweep
        odir = _HERE / "seed_sweep_xi"
        odir.mkdir(exist_ok=True)
        for t in args.tracers:
            info, native, bundle, cov_map, tmpl, z, apmode = _build(t)
            out = {label: {} for label in covs}
            for label in covs:
                bundle["cov"] = cov_map[label]
                log_prob, names, fid = M.make_log_prob(info, native, bundle, apmode, t,
                                                       bb_basis=BB_BASIS)
                for seed in args.seeds:
                    res = _sigmas(_run_chain(log_prob, names, fid, seed), names, apmode, tmpl, z)
                    out[label][str(seed)] = res
                    print(f"[{t} {label} seed={seed}] {res}")
            (odir / f"{t}.json").write_text(json.dumps(out, indent=2))
            print(f"wrote seed_sweep_xi/{t}.json")


if __name__ == "__main__":
    main()
