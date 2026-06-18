"""Unified parallel generator of BAO emulator training data.

Maps emulator INPUTS  (N_tracers + cosmology θ)
  to emulator OUTPUTS (distance-error triplet σ(DH/rd), σ(DM/rd), σ(DV/rd)).

Two interchangeable backends, selected with ``--space``:

  fourier : Fourier-space single-bin Fisher with the analytic Gaussian P-space
            covariance (fourier_space.run_fisher). The 2x2 (DH/rd, DM/rd) covariance
            is converted to the σ-triplet via fourier_space._cov_to_sigma_triplet.

  config  : Config-space (ξ) Fisher with the first-principles Grieb Gaussian
            ξ-covariance (config_space.XiSigmaGenerator), the emulator σ-driver.

Both spaces emit per-tracer emulator targets (see core.emulator_target_names):
  isotropic (BGS; BGS+QSO for dr1): [sigma_DV_over_rd]
  anisotropic: [sigma_DH_over_rd, sigma_DM_over_rd, rho_DH_DM]
The sampling (constrained Latin-hypercube), the spawn
worker Pool, the progress/ETA loop, and the train/test .npz writer are all shared
(reused from core) — only the per-sample compute differs by space and lives
in its own module (core for fourier, config_space for config).

Usage (from bao/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python generate_emulator_data.py \
        --space config --tracer-bin LRG2 --cosmo-model base \
        --n-samples 5000 --workers 16

Notes
-----
* ``--space config`` survey geometry (window + footprint area) is frame-fixed by
  the DESI DR1 bundle, so ``--area`` and ``--nz-slices-path`` apply to the
  fourier backend only (the config backend uses its own fixed DR1 frame).
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

from core import (
    CONSTRAINTS,
    COSMO_MODELS,
    DEFAULT_PRIORS,
    PARAM_DEFAULTS,
    emulator_target_names,
    generate_dataset,
)
from util import (
    TRACER_TYPE_CHOICES,
    get_default_save_path,
    get_tracer_config,
    ntracers_range,
    parse_priors,
    save_dataset,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate BAO emulator training data (cosmo + N_tracers -> "
                    "distance-error triplet) in Fourier or config space."
    )
    p.add_argument(
        "--space",
        choices=["fourier", "config"],
        default="fourier",
        help="Covariance backend: 'fourier' (analytic P-space Gaussian Fisher) "
             "or 'config' (first-principles Grieb ξ-space Fisher).",
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
        "--tracer-bin", dest="tracer_bin", type=str, default="LRG2",
        choices=TRACER_TYPE_CHOICES,
        help="DESI tracer bin key (must match tracers.yaml).",
    )
    p.add_argument(
        "--z-eff", type=float, default=None,
        help="Effective redshift. If omitted, defaults to tracer-bin config. "
             "(fourier only; config uses the DR1 bundle z_eff)",
    )
    p.add_argument(
        "--zrange", type=float, nargs=2, default=None, metavar=("Z_MIN", "Z_MAX"),
        help="Redshift bin edges for the footprint volume. fourier only.",
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
        "--dataset", type=str, default="dr1", choices=["dr1", "dr2"],
        help="DESI release whose 'passed' counts anchor the N_tracers box "
             "(box = tracers.yaml low/high FACTORS x passed[dataset]).",
    )
    p.add_argument(
        "--ntracers-range", type=float, nargs=2, default=None,
        metavar=("NTRACERS_LOW", "NTRACERS_HIGH"),
        help="Override the N_tracers prior with ABSOLUTE bounds. Default is "
             "tracers.yaml low/high factors x the --dataset passed count.",
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
        "--area", type=float, default=14000.0,
        help="Effective survey area in deg^2 (fourier only; config is fixed by "
             "the DR1 bundle footprint).",
    )
    p.add_argument(
        "--nz-slices-path", type=str, default=None,
        help="Optional parsed *_nz_slices.csv for redshift-sliced Fisher info "
             "(fourier only).",
    )
    return p


def main() -> None:
    sys.argv = [a for a in sys.argv if a.strip()]
    args = _build_arg_parser().parse_args()

    tracer_bin_cfg = get_tracer_config(args.tracer_bin)
    zrange = tuple(args.zrange) if args.zrange is not None else tuple(tracer_bin_cfg["zrange"])
    z_eff = args.z_eff if args.z_eff is not None else float(tracer_bin_cfg["z_eff"])

    cosmo_model = args.cosmo_model
    model_params = COSMO_MODELS[cosmo_model]

    # Priors: N_tracers + the varied cosmology params for this model.
    if args.priors_json:
        priors = parse_priors(args.priors_json)
    else:
        varied_keys = ["N_tracers"] + model_params
        priors = {k: dict(DEFAULT_PRIORS[k]) for k in varied_keys}

    if args.ntracers_range is not None:
        nt_low, nt_high = args.ntracers_range
    else:
        nt_low, nt_high = ntracers_range(args.tracer_bin, args.dataset)
    priors["N_tracers"] = {"dist": "uniform", "low": nt_low, "high": nt_high}

    all_cosmo_keys = {"Om", "Ok", "w0", "wa", "hrdrag"}
    fixed_keys = all_cosmo_keys - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys if k in PARAM_DEFAULTS}

    constraints = {
        name: spec for name, spec in CONSTRAINTS.items()
        if all(p in model_params for p in spec["params"])
    }

    save_path = os.path.abspath(
        args.save_path if args.save_path else
        get_default_save_path(analysis="bao", quantity=args.space,
                              cosmo_model=cosmo_model, dataset=args.dataset)
    )

    # Backend selection: pick the picklable worker (defined in its space's module)
    # and the per-sample task payload. generate_dataset routes every sample
    # through worker_fn(make_task(sample)) in both the serial and parallel paths.
    if args.space == "fourier":
        import fourier_space
        worker_fn = fourier_space._worker_run_fisher_sigma
        make_task = lambda s: (  # noqa: E731
            s, args.tracer_bin, zrange, z_eff, param_defaults, args.area,
            args.nz_slices_path, args.dataset,
        )
    else:
        import config_space
        tracer = args.tracer_bin
        worker_fn = config_space._worker_xi_sigma
        make_task = lambda s: (s, tracer, args.dataset)  # noqa: E731

    target_names = emulator_target_names(args.tracer_bin, args.dataset)

    print(f"Space: {args.space}")
    print(f"Tracer bin: {args.tracer_bin}")
    print(f"Tracer bin config: {tracer_bin_cfg}")
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
    print(f"N_tracers prior: [{priors['N_tracers']['low']:.3g}, "
          f"{priors['N_tracers']['high']:.3g}]")
    print("Using priors:", priors)
    print(f"Active constraints: {list(constraints.keys())}")
    if args.space == "fourier":
        print(f"Redshift range: {zrange}, z_eff = {z_eff:.3f}")
        print(f"Area: {args.area:.3f} deg^2")
        if args.nz_slices_path:
            print(f"Using n(z) slices: {args.nz_slices_path}")
    else:
        print("Config space: survey frame (window + area + z_eff) fixed by the "
              "DESI DR1 bundle.")
    print(f"Target: {target_names}")
    print("Writing dataset to:", save_path)

    try:
        param_names, X, y = generate_dataset(
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
            area=args.area,
            nz_slices_path=args.nz_slices_path,
            worker_fn=worker_fn,
            make_task=make_task,
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
