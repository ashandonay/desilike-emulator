"""Parallel generator of ShapeFit error-emulator training data.

Maps emulator INPUTS  (N_tracers + cosmology θ, omega basis)
  to emulator OUTPUTS (the 10 targets: sigma_qiso, sigma_qap, sigma_f_sigmar,
                       sigma_m + the 6 pairwise rho_*).

Single backend: the Fourier-space pre-recon Fisher with the Gaussian
(FKP + SSC) P-space covariance (fourier_space.run_fisher). The sampling
(constrained Latin-hypercube), spawn worker Pool, progress/ETA loop and the
train/test .npz writer are shared with the BAO pipeline (bao/core.py
generate_dataset); only the per-sample compute lives here.

Usage (from shapefit/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python generate_covar_data.py \
        --tracer-bin LRG2 --cosmo-model base --n-samples 5000 --workers 16

Notes
-----
* z_eff defaults to the per-sample n(z)-derived value (cosmology-clean);
  --z-eff pins it (sensitivity checks only). This differs from the BAO
  Fourier CLI, which pins the yaml value.
* The N_tracers box is ALWAYS anchored via util.ntracers_range (tracers.yaml
  low/high factors x the --data-release passed count) unless --ntracers-range
  overrides it with absolute bounds. Never hardcode N (CHANGELOG bao §33n).
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fourier_space
from fourier_space import sf_core
from util import (
    TRACER_TYPE_CHOICES,
    get_default_save_path,
    get_tracer_config,
    ntracers_range,
    parse_priors,
    save_dataset,
    tracer_area,
)

CONSTRAINTS = sf_core.CONSTRAINTS
COSMO_MODELS = sf_core.COSMO_MODELS
DEFAULT_PRIORS = sf_core.DEFAULT_PRIORS
PARAM_DEFAULTS = sf_core.PARAM_DEFAULTS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate ShapeFit error-emulator training data "
                    "(cosmo + N_tracers -> sigma/rho of qiso, qap, f_sigmar, m)."
    )
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--sigma-clip", type=float, default=4.0)
    p.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = serial).",
    )
    p.add_argument(
        "--maxtasksperchild", type=int, default=50,
        help="Recycle each worker after this many tasks. The covar path leaks "
             "~50 MB per accepted sample inside desilike/cosmoprimo, so "
             "long-lived workers pass 4.5 GB in 10 h and a wide pool OOMs the "
             "box on multi-hour runs. 0 disables recycling.",
    )
    p.add_argument(
        "--tracer-bin", dest="tracer_bin", type=str, default="LRG2",
        choices=TRACER_TYPE_CHOICES,
        help="DESI tracer bin key (must match tracers.yaml).",
    )
    p.add_argument(
        "--z-eff", type=float, default=None,
        help="Pin the effective redshift. Default: derived per sample from the "
             "n(z) slices with the FS Fisher weight (cosmology-clean).",
    )
    p.add_argument(
        "--zrange", type=float, nargs=2, default=None, metavar=("Z_MIN", "Z_MAX"),
        help="Redshift bin edges for the footprint volume. Defaults to "
             "tracers.yaml.",
    )
    p.add_argument(
        "--name", type=str, default=None,
        help="Tracer name prefix for saved files. Defaults to --tracer-bin.",
    )
    p.add_argument(
        "--version", type=int, default=None,
        help="Explicit version for the training_data/v{N} dir. "
             "If omitted, auto-increments.",
    )
    p.add_argument(
        "--data-release", type=str, default="dr1", choices=["dr1"],
        help="DESI release anchoring the N_tracers box (dr1 only for now).",
    )
    p.add_argument(
        "--ntracers-range", type=float, nargs=2, default=None,
        metavar=("NTRACERS_LOW", "NTRACERS_HIGH"),
        help="Override the N_tracers prior with ABSOLUTE bounds. Default is "
             "tracers.yaml low/high factors x the --data-release passed count.",
    )
    p.add_argument(
        "--priors-json", type=str, default="",
        help='JSON dict of priors, e.g. '
             '\'{"N_tracers":{"dist":"uniform","low":1e5,"high":1e7}}\'',
    )
    p.add_argument(
        "--cosmo-model", type=str, default="base", choices=list(COSMO_MODELS.keys()),
        help="Cosmology model determining which params are varied (default: base).",
    )
    p.add_argument(
        "--area", type=float, default=None,
        help="Effective survey area in deg^2 used by CutskyFootprint. "
             "Default: this TRACER's footprint (util.tracer_area), not the "
             "release's -- BGS 7473, LRG 5740, ELG 5924, QSO 7249.",
    )
    return p


def main() -> None:
    sys.argv = [a for a in sys.argv if a.strip()]
    args = _build_arg_parser().parse_args()

    tracer_bin_cfg = get_tracer_config(args.tracer_bin, data_release=args.data_release)
    zrange = (tuple(args.zrange) if args.zrange is not None
              else tuple(tracer_bin_cfg["zrange"]))
    z_eff = args.z_eff  # None -> derived per sample inside the likelihood build

    cosmo_model = args.cosmo_model
    model_params = COSMO_MODELS[cosmo_model]

    if args.priors_json:
        priors = parse_priors(args.priors_json)
    else:
        varied_keys = ["N_tracers"] + model_params
        priors = {k: dict(DEFAULT_PRIORS[k]) for k in varied_keys}

    if args.ntracers_range is not None:
        nt_low, nt_high = args.ntracers_range
    else:
        nt_low, nt_high = ntracers_range(args.tracer_bin, args.data_release)
    priors["N_tracers"] = {"dist": "uniform", "low": nt_low, "high": nt_high}

    all_cosmo_keys = set(DEFAULT_PRIORS) - {"N_tracers"}
    fixed_keys = all_cosmo_keys - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys if k in PARAM_DEFAULTS}

    constraints = {
        name: spec for name, spec in CONSTRAINTS.items()
        if all(p in model_params for p in spec["params"])
    }

    save_path = os.path.abspath(
        args.save_path if args.save_path else
        get_default_save_path(analysis="shapefit", quantity="covar",
                              cosmo_model=cosmo_model, data_release=args.data_release)
    )

    # Footprint for THIS tracer, not for the release. S54 established the area
    # is per tracer class (DESI 2024 II Table 2: BGS 7473, LRG 5740, ELG 5924,
    # QSO 7249) because priority and imaging vetoes remove different sky from
    # different samples. This used to resolve DATASET_AREAS[data_release] = 7500 and
    # pass it in explicitly, which OVERRODE the per-tracer default that
    # build_shapefit_likelihood only applies when area is None -- so the S54 fix
    # reached every validation path (our_forecast, mcmc.py, benchmark_desi.py)
    # but never the training data. Measured cost on the generated sigma, at
    # fixed N: LRG -6..-10%, ELG2 -0.4..-4.5%, QSO/BGS <0.5%. The larger area
    # wins over the diluted nbar, so the emulator was being taught that DESI is
    # MORE constraining than it is, worst exactly on the LRG bins.
    area = (float(args.area) if args.area is not None
            else tracer_area(args.tracer_bin, args.data_release))

    worker_fn = fourier_space._worker_run_fisher_targets
    make_task = lambda s: (  # noqa: E731
        s, args.tracer_bin, zrange, z_eff, param_defaults, area, args.data_release,
    )

    target_names = sf_core.emulator_target_names(args.tracer_bin, args.data_release)

    print(f"Tracer bin: {args.tracer_bin}")
    print(f"Tracer bin config: {tracer_bin_cfg}")
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
    print(f"N_tracers prior: [{priors['N_tracers']['low']:.3g}, "
          f"{priors['N_tracers']['high']:.3g}]")
    print("Using priors:", priors)
    print(f"Active constraints: {list(constraints.keys())}")
    print(f"Redshift range: {zrange}, z_eff = "
          f"{'derived per sample' if z_eff is None else f'{z_eff:.3f} (pinned)'}")
    print(f"Area: {area:.3f} deg^2 ({args.data_release} footprint)")
    print(f"Target: {target_names}")
    print("Writing dataset to:", save_path)

    try:
        param_names, X, y = sf_core.generate_dataset(
            priors=priors,
            n_samples=args.n_samples,
            tracer_bin=args.tracer_bin,
            zrange=zrange,
            z_eff=z_eff,
            batch_size=args.batch_size,
            seed=args.seed,
            sigma_clip=args.sigma_clip,
            workers=args.workers,
            param_defaults=param_defaults,
            constraints=constraints,
            area=area,
            worker_fn=worker_fn,
            make_task=make_task,
            maxtasksperchild=args.maxtasksperchild or None,
        )
        print(f"Generated dataset with shape X={X.shape}, y={y.shape}")
        save_dataset(
            save_path=save_path,
            param_names=param_names,
            X=X,
            y=y,
            test_size=args.test_size,
            target_names=target_names,
            name=args.name if args.name is not None else args.tracer_bin,
            version=args.version,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
