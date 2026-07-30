"""Parallel generator of ShapeFit mean-emulator training data.

Maps emulator INPUTS  (cosmology θ, omega basis — no N_tracers)
  to emulator OUTPUTS (qiso, qap, f_sigmar, m)

via desilike's ShapeFitPowerSpectrumExtractor at the tracer's effective
redshift, compared against the DESI fiducial template. This is the "mean"
side of the per-tracer ShapeFit Gaussian likelihood in bedcosmo (which has
no differentiable f_sigmar/m of its own); the "covar" side is
generate_covar_data.py.

Per-tracer because z_eff differs per tracer. z_eff is computed ONCE at the
DESI fiducial cosmology with the FS Fisher weight (the extractor's z is an
init-time argument; the residual cosmology dependence of z_eff enters the
labels only through slowly-varying volume weights — documented approximation;
--z-eff overrides for sensitivity checks).

Usage (from shapefit/, emulator env):
    LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
        ~/miniconda3/envs/emulator/bin/python generate_mean_data.py \
        --tracer-bin LRG2 --cosmo-model base --n-samples 10000 --workers 8
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
    parse_priors,
    save_dataset,
)

COSMO_MODELS = sf_core.COSMO_MODELS
CONSTRAINTS = sf_core.CONSTRAINTS
DEFAULT_PRIORS = sf_core.DEFAULT_PRIORS
PARAM_DEFAULTS = sf_core.PARAM_DEFAULTS

_FID_SAMPLE = {
    "omega_cdm": 0.1200,
    "omega_b": 0.02237,
    "h": 0.6736,
    "ln10A_s": 3.036394,
    "n_s": 0.9649,
}


def _fiducial_z_eff(tracer_bin: str, area: float) -> float:
    """Tracer z_eff at the DESI fiducial cosmology with the FS band weight."""
    from desilike.theories.primordial_cosmology import get_cosmo

    cfg = get_tracer_config(tracer_bin)
    theta = sf_core._to_shapefit_cosmo_params(_FID_SAMPLE)
    cosmo = get_cosmo(("DESI", dict(theta)))
    fo = cosmo.get_fourier()
    try:
        return sf_core._fs_compute_z_eff(
            tracer_bin=tracer_bin, cosmo=cosmo, fo=fo,
            area_deg2=float(area), b1=float(cfg.get("bias_recon", 2.0)),
        )
    except (FileNotFoundError, ValueError):
        return float(cfg["z_eff"])


def main() -> None:
    sys.argv = [a for a in sys.argv if a.strip()]
    p = argparse.ArgumentParser(
        description="Generate ShapeFit mean-emulator training data "
                    "(cosmology -> qiso, qap, f_sigmar, m per tracer)."
    )
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--sigma-clip", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes (default: 1 = serial).")
    p.add_argument("--maxtasksperchild", type=int, default=0,
                   help="Recycle each worker after this many tasks (0 = never). "
                        "The mean path holds flat RSS (~0.5 GB/worker measured "
                        "over 10 h), so recycling is off by default here.")
    p.add_argument("--tracer-bin", dest="tracer_bin", type=str, default="LRG2",
                   choices=TRACER_TYPE_CHOICES,
                   help="DESI tracer bin key (must match tracers.yaml).")
    p.add_argument("--z-eff", type=float, default=None,
                   help="Pin the effective redshift. Default: derived at the "
                        "DESI fiducial cosmology with the FS Fisher weight.")
    p.add_argument("--area", type=float, default=None,
                   help="Effective survey area in deg^2 (z_eff volume weights). "
                        "Default: the dataset footprint (dr1 7500, dr2 14000).")
    p.add_argument("--name", type=str, default=None,
                   help="Tracer name prefix for saved files. Defaults to "
                        "--tracer-bin.")
    p.add_argument("--version", type=int, default=None,
                   help="Explicit version for the training_data/v{N} dir. "
                        "If omitted, auto-increments.")
    p.add_argument("--dataset", type=str, default="dr1", choices=["dr1"],
                   help="Dataset path segment (dr1 only for now).")
    p.add_argument("--priors-json", type=str, default="",
                   help="JSON dict of priors overriding the cosmo-model set.")
    p.add_argument("--cosmo-model", type=str, default="base",
                   choices=list(COSMO_MODELS.keys()),
                   help="Cosmology model determining which params are varied.")
    args = p.parse_args()

    cosmo_model = args.cosmo_model
    model_params = COSMO_MODELS[cosmo_model]

    if args.priors_json:
        priors = parse_priors(args.priors_json)
    else:
        priors = {k: dict(DEFAULT_PRIORS[k]) for k in model_params}

    all_cosmo_keys = set(DEFAULT_PRIORS) - {"N_tracers"}
    fixed_keys = all_cosmo_keys - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys if k in PARAM_DEFAULTS}

    constraints = {
        name: spec for name, spec in CONSTRAINTS.items()
        if all(p in model_params for p in spec["params"])
    }

    z_eff = args.z_eff if args.z_eff is not None else _fiducial_z_eff(
        args.tracer_bin,
        float(args.area) if args.area is not None
        else sf_core.dataset_area(args.dataset))

    save_path = os.path.abspath(
        args.save_path if args.save_path else
        get_default_save_path(analysis="shapefit", quantity="mean",
                              cosmo_model=cosmo_model, dataset=args.dataset)
    )

    worker_fn = fourier_space._worker_run_mean_targets
    make_task = lambda s: (s, args.tracer_bin, z_eff, param_defaults)  # noqa: E731

    print(f"Tracer bin: {args.tracer_bin}")
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
    print("Using priors:", priors)
    print(f"Active constraints: {list(constraints.keys())}")
    print(f"z_eff = {z_eff:.4f} "
          f"({'pinned' if args.z_eff is not None else 'fiducial-derived'})")
    print(f"Target: {sf_core.MEAN_TARGET_NAMES}")
    print("Writing dataset to:", save_path)

    try:
        param_names, X, y = sf_core.generate_dataset(
            priors=priors,
            n_samples=args.n_samples,
            tracer_bin=args.tracer_bin,
            batch_size=args.batch_size,
            seed=args.seed,
            sigma_clip=args.sigma_clip,
            workers=args.workers,
            param_defaults=param_defaults,
            constraints=constraints,
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
            target_names=sf_core.MEAN_TARGET_NAMES,
            name=args.name if args.name is not None else args.tracer_bin,
            version=args.version,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
