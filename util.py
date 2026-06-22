import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import json
import yaml
import torch.nn as nn
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split

from scipy.stats import qmc
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

try:
    # Preferred when installed as the desilike_emulator package (pip install -e).
    from desilike_emulator.bao import model as bao_model
    from desilike_emulator.shapefit import model as shapefit_model
except ModuleNotFoundError:
    try:
        # Legacy location when this code lived inside the bedcosmo package.
        from bedcosmo.num_tracers.emulator.bao import model as bao_model
        from bedcosmo.num_tracers.emulator.shapefit import model as shapefit_model
    except ModuleNotFoundError:
        # Backward-compatible fallback for running from emulator/ as a script.
        from bao import model as bao_model  # type: ignore[reportMissingImports]
        from shapefit import model as shapefit_model  # type: ignore[reportMissingImports]

# analysis -> architecture name -> nn.Module subclass (hyperparameters passed at build time).
ARCHITECTURE_REGISTRY: Dict[str, Dict[str, Type[nn.Module]]] = {
    "bao": {
        "resnet": bao_model.ResNetRegressor,
    },
    "shapefit": {
        "resnet": shapefit_model.base_regressor,
    },
}

_THIS_DIR = Path(__file__).resolve().parent
_TRACER_CONFIG_PATH = _THIS_DIR / "tracers.yaml"
_REQUIRED_TRACER_KEYS = {
    "zrange",
    "z_eff",
    "low",
    "high",
}


def _load_tracer_configs(path: Path) -> Dict[str, Dict[str, object]]:
    """Load and validate ``tracers.yaml``; normalise keys to uppercase."""
    if not path.exists():
        raise FileNotFoundError(f"Tracer config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid or empty tracer config YAML: {path}")

    cleaned: Dict[str, Dict[str, object]] = {}
    for key, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Tracer config for {key!r} must be a mapping")

        missing = _REQUIRED_TRACER_KEYS - set(cfg)
        if missing:
            raise ValueError(f"Tracer config for {key!r} missing keys: {sorted(missing)}")

        zrange = cfg["zrange"]
        if not isinstance(zrange, (list, tuple)) or len(zrange) != 2:
            raise ValueError(f"Tracer config for {key!r} must have zrange as length-2 list")

        # Start from all yaml fields (preserves per-tracer extras like Lyα's
        # bF0, gamma_bF, Sigma_perp_fid, etc.), then coerce the mandatory ones.
        entry: Dict[str, object] = dict(cfg)
        entry["zrange"] = (float(zrange[0]), float(zrange[1]))
        entry["z_eff"] = float(cfg["z_eff"])
        entry["low"] = float(cfg["low"])
        entry["high"] = float(cfg["high"])
        if "bias_recon" in cfg:
            entry["bias_recon"] = float(cfg["bias_recon"])
        if "smoothing_scale" in cfg:
            entry["smoothing_scale"] = float(cfg["smoothing_scale"])
        entry["supported"] = bool(cfg.get("supported", True))
        cleaned[key] = entry

    return cleaned


TRACER_CONFIGS: Dict[str, Dict[str, object]] = _load_tracer_configs(_TRACER_CONFIG_PATH)
TRACER_TYPE_CHOICES = list(TRACER_CONFIGS.keys())


_HOD_CONFIG_PATH = _THIS_DIR / "hod.yaml"
_REQUIRED_HOD_KEYS = {
    "sigma_logM",
    "f_cen",
    "log_Mquench_over_Mcut",
    "eta_quench",
    "log_M_sat_over_Mcut",
    "log_M1_over_Mcut",
    "alpha_sat",
}
# Optional HOD keys with their defaults. Float-coerced and passed through to
# `_hod_halo_props` in prep_covar.py.
_OPTIONAL_HOD_FLOAT_KEYS = {
    "assembly_bias_factor": 1.0,  # decorated-HOD multiplier on b1 (Hadzhiyska+22)
}
# Optional HOD keys that are strings (left as-is by the loader).
_OPTIONAL_HOD_STR_KEYS = {
    "central_form",               # "erf" (default in _hod_halo_props) or "gaussian"
}


def _load_hod_configs(path: Path) -> Dict[str, Dict[str, object]]:
    """Load and validate ``hod.yaml``.

    Returns a dict keyed by tracer_type ("BGS", "LRG", "ELG", "MIX", "QSO")
    with float-coerced shape parameters plus any optional fields
    (``central_form``, ``assembly_bias_factor``). Cosmology-independent by
    construction; M_cut is solved per cosmology inside `_hod_halo_props`.
    """
    if not path.exists():
        raise FileNotFoundError(f"HOD config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid or empty HOD config YAML: {path}")

    cleaned: Dict[str, Dict[str, object]] = {}
    for key, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"HOD config for {key!r} must be a mapping")
        missing = _REQUIRED_HOD_KEYS - set(cfg)
        if missing:
            raise ValueError(f"HOD config for {key!r} missing keys: {sorted(missing)}")
        entry: Dict[str, object] = {k: float(cfg[k]) for k in _REQUIRED_HOD_KEYS}
        for k, default in _OPTIONAL_HOD_FLOAT_KEYS.items():
            entry[k] = float(cfg.get(k, default))
        for k in _OPTIONAL_HOD_STR_KEYS:
            if k in cfg:
                entry[k] = str(cfg[k])
        cleaned[key] = entry
    return cleaned


HOD_CONFIGS: Dict[str, Dict[str, float]] = _load_hod_configs(_HOD_CONFIG_PATH)


def get_tracer_config(tracer_bin: str) -> Dict[str, object]:
    """Return validated tracer config dict."""
    key = tracer_bin.strip()
    if key not in TRACER_CONFIGS:
        raise ValueError(f"Unknown tracer bin {tracer_bin!r}. Choices: {TRACER_TYPE_CHOICES}")
    cfg = dict(TRACER_CONFIGS[key])
    if not cfg.get("supported", True):
        raise ValueError(f"Tracer bin {tracer_bin!r} is marked unsupported in tracers.yaml")
    return cfg


_CSV_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
_NTRACERS_CACHE: Dict[str, Dict[str, float]] = {}  # dataset -> {csv_label: passed}


def ntracers(tracer_bin: str, dataset: str = "dr1") -> float:
    """Return the DESI 'passed' N_tracers for ``tracer_bin`` from
    ``~/data/desi/bao_{dataset}/desi_data.csv`` (dataset in {dr1, dr2}).

    The HOD M_cut root-find depends on nbar = N_tracers / V_eff, so any
    pipeline configuration that compares predictions against bundle data must
    use the actual sample size for that tracer/release — not a generic default.
    Using a mismatched N silently shifts the HOD-weighted b1 and invalidates
    downstream b1/f_AB calibration.
    """
    if dataset not in _NTRACERS_CACHE:
        import pandas as pd
        path = Path.home() / "data" / "desi" / f"bao_{dataset}" / "desi_data.csv"
        df = pd.read_csv(path)[["tracer", "passed"]].drop_duplicates("tracer")
        _NTRACERS_CACHE[dataset] = {r["tracer"]: float(r["passed"]) for _, r in df.iterrows()}
    cache = _NTRACERS_CACHE[dataset]
    key = _CSV_NAME_MAP.get(tracer_bin, tracer_bin)
    if key not in cache:
        raise KeyError(f"No {dataset} N_tracers for {tracer_bin!r} (looked up as {key!r}). "
                       f"Available: {sorted(cache)}")
    return cache[key]


def dr1_ntracers(tracer_bin: str) -> float:
    """Back-compat DR1 wrapper for :func:`ntracers`."""
    return ntracers(tracer_bin, "dr1")


def ntracers_range(tracer_bin: str, dataset: str = "dr1") -> Tuple[float, float]:
    """Absolute N_tracers LHS bounds for ``tracer_bin``: the per-tracer
    ``low``/``high`` *multiplicative factors* in tracers.yaml times the DESI
    ``passed`` count for ``dataset``. E.g. factors (0.5, 1.5) -> [0.5*passed,
    1.5*passed], a box centred on the passed count."""
    cfg = get_tracer_config(tracer_bin)
    p = ntracers(tracer_bin, dataset)
    return float(cfg["low"]) * p, float(cfg["high"]) * p


def build_model(analysis: str, architecture: str, **kwargs) -> nn.Module:
    """Instantiate a model from ARCHITECTURE_REGISTRY using the YAML ``architecture`` field."""
    if analysis not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown analysis '{analysis}'. Choose from: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    registry = ARCHITECTURE_REGISTRY[analysis]
    if architecture not in registry:
        raise ValueError(
            f"Unknown architecture '{architecture}' for analysis '{analysis}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[architecture](**kwargs)

def latin_hypercube_samples(
    priors: Dict[str, Dict[str, float]],
    n_samples: int,
    seed: int,
    sigma_clip: float = 4.0,
) -> List[Dict[str, float]]:
    keys = list(priors.keys())
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit_samples = sampler.random(n=n_samples)

    rows: List[Dict[str, float]] = []
    for urow in unit_samples:
        out: Dict[str, float] = {}
        for key, u in zip(keys, urow):
            spec = priors[key]
            dist = spec["dist"]
            if dist == "uniform":
                low = float(spec["low"])
                high = float(spec["high"])
                out[key] = low + (high - low) * float(u)
            elif dist == "normal":
                mu = float(spec["mu"])
                sigma = float(spec["sigma"])
                a = -sigma_clip
                b = sigma_clip
                out[key] = float(truncnorm.ppf(u, a, b, loc=mu, scale=sigma))
            else:
                raise ValueError(f"Unsupported dist '{dist}' for '{key}'")
        rows.append(out)
    return rows


def to_extractor_params(sample: Dict[str, float]) -> Dict[str, float]:
    # omega_* are physical densities: omega_x = Omega_x * h^2
    # so Omega_m = (omega_cdm + omega_b) / h^2.
    # assumes fiducial values for omega_cdm, omega_b, h, ln10A_s, n_s
    omega_cdm = sample.get("omega_cdm", 0.12069)
    omega_b = sample.get("omega_b", 0.02218)
    h = sample.get("h", 0.6736)
    if h <= 0.0:
        raise ValueError("h must be > 0 to compute Omega_m")
    omega_m = (omega_cdm + omega_b) / (h * h)
    result = {
        "h": float(h),
        "Omega_m": float(omega_m),
        "omega_b": float(omega_b),
        "logA": float(sample.get("ln10A_s", 3.036394)),
        "n_s": float(sample.get("n_s", 0.9649)),
    }
    if "w0" in sample:
        result["w0_fde"] = float(sample["w0"])
    if "wa" in sample:
        result["wa_fde"] = float(sample["wa"])
    return result


def save_dataset(
    save_path: str,
    param_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 1,
    target_names: List[str] | None = None,
    name: str = "",
    version: int | None = None,
) -> str:
    if target_names is None:
        target_names = TARGET_NAMES
    if version is None:
        version = _next_version(save_path)
    versioned_path = os.path.join(save_path, f"v{version}")
    os.makedirs(versioned_path, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    prefix = f"{name}_" if name else ""
    np.savez(
        f"{versioned_path}/{prefix}train.npz",
        x=X_train,
        y=y_train,
        param_names=np.array(param_names),
        target_names=np.array(target_names),
    )
    np.savez(
        f"{versioned_path}/{prefix}test.npz",
        x=X_test,
        y=y_test,
        param_names=np.array(param_names),
        target_names=np.array(target_names),
    )
    print(f"Saved {prefix}train/test files to: {versioned_path} (version {version})")
    return versioned_path

def _next_version(save_path: str) -> int:
    """Find the next available version number in save_path/v{N} directories."""
    if not os.path.isdir(save_path):
        return 1
    existing = [
        int(d[1:])
        for d in os.listdir(save_path)
        if d.startswith("v") and d[1:].isdigit()
    ]
    return max(existing, default=0) + 1

def parse_priors(priors_json: str) -> Dict[str, Dict[str, float]]:
    raw = json.loads(priors_json)
    if not isinstance(raw, dict):
        raise ValueError("Priors JSON must be a dictionary")
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "dist" not in spec:
            raise ValueError(f"Prior '{name}' must be a dictionary with a 'dist' key")
        if spec["dist"] == "uniform":
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"Uniform prior '{name}' needs 'low' and 'high'")
            if float(spec["low"]) >= float(spec["high"]):
                raise ValueError(f"Uniform prior '{name}' must satisfy low < high")
        elif spec["dist"] == "normal":
            if "mu" not in spec or "sigma" not in spec:
                raise ValueError(f"Normal prior '{name}' needs 'mu' and 'sigma'")
            if float(spec["sigma"]) <= 0.0:
                raise ValueError(f"Normal prior '{name}' must have sigma > 0")
        else:
            raise ValueError(f"Unsupported dist '{spec['dist']}' for '{name}'")
    return raw


def _load_module(name: str, path: str):
    """Load a Python module from an explicit file path (avoids name collisions)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_pipeline(analysis: str, quantity: str, tracer_bin: str | None = None, param_names=None):
    """Return (default_priors, target_names, ground_truth_fn, setup) for the given analysis/quantity.

    If *tracer_bin* is given, overrides N_tracers prior bounds and passes the
    corresponding zrange/z_eff to the ground truth function. If *param_names* is
    given (e.g. a checkpoint's), the priors are restricted to exactly those
    parameters so the live ground truth varies the same params the model was
    trained on (the rest stay at their fiducial defaults).
    """
    _here = os.path.dirname(os.path.abspath(__file__))

    tracer_bin_cfg = get_tracer_config(tracer_bin) if tracer_bin else None

    if analysis == "shapefit":
        if quantity == "covar":
            mod = _load_module("shapefit_prep_covar", os.path.join(_here, "shapefit", "prep_covar.py"))
            kw = {}
            if tracer_bin_cfg:
                kw = {"zrange": tracer_bin_cfg["zrange"], "z_eff": tracer_bin_cfg["z_eff"]}
            def ground_truth_fn(_setup, sample, _kw=kw):
                return mod.run_fisher(sample, **_kw)
            priors = dict(mod.DEFAULT_PRIORS)
            if tracer_bin_cfg:
                _nt_lo, _nt_hi = ntracers_range(tracer_bin)
                priors["N_tracers"] = {"dist": "uniform", "low": _nt_lo, "high": _nt_hi}
            return priors, mod.TARGET_NAMES, ground_truth_fn, None
        elif quantity == "mean":
            mod = _load_module("shapefit_prep_mean", os.path.join(_here, "shapefit", "prep_mean.py"))
            from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumExtractor
            extractor = ShapeFitPowerSpectrumExtractor()
            extractor()
            extractor.get()
            def ground_truth_fn(setup, sample):
                return mod.run_extractor(setup, sample)
            priors = dict(mod.DEFAULT_PRIORS)
            return priors, mod.TARGET_NAMES, ground_truth_fn, extractor
        else:
            raise ValueError(f"Unknown quantity for shapefit: {quantity}")

    elif analysis == "bao":
        if quantity not in ("config", "covar"):
            raise ValueError(
                f"Unknown quantity for bao: {quantity!r} (expected 'config' or 'covar')"
            )
        if not tracer_bin:
            raise ValueError(
                "bao eval requires a tracer_bin (the σ generator needs the tracer geometry)."
            )
        # Lazy + heavy: pulls desilike + the DESI bundles. The bao modules use
        # sibling imports (e.g. `import core`), so put bao/ on sys.path like the
        # bao runtime does (cwd=bao/) before importing them.
        import sys
        _bao_dir = os.path.join(_here, "bao")
        if _bao_dir not in sys.path:
            sys.path.insert(0, _bao_dir)
        import core as bao_core

        target_names = list(bao_core.emulator_target_names(tracer_bin, dataset="dr1"))

        # Restrict priors to the trained model's varied params (param_names),
        # else default to N_tracers + the base cosmo set. N_tracers bounds come
        # from the tracer's DR1 box; non-varied cosmo params stay at fiducial
        # defaults inside the generator.
        varied = [str(p) for p in param_names] if param_names is not None \
            else (["N_tracers"] + bao_core.COSMO_MODELS["base"])
        priors = {p: dict(bao_core.DEFAULT_PRIORS[p]) for p in varied}
        _nt_lo, _nt_hi = ntracers_range(tracer_bin)
        priors["N_tracers"] = {"dist": "uniform", "low": _nt_lo, "high": _nt_hi}

        if quantity == "config":
            # Config-space ξ-covariance σ — the emulator's training-data generator.
            import config_space
            gen = config_space.XiSigmaGenerator(tracer_bin)  # built once, reused per sample
            def ground_truth_fn(_setup, sample, _gen=gen):
                cosmo = {k: v for k, v in sample.items() if k != "N_tracers"}
                s = _gen.sigma_triplet(N_tracers=sample["N_tracers"], **cosmo)
                vals = bao_core.fisher_sigmas_to_emulator_targets(s, tracer_bin, "dr1")
                return dict(zip(target_names, vals))
            return priors, target_names, ground_truth_fn, gen

        # quantity == "covar": Fourier-space Fisher σ.
        import fourier_space
        kw = {"tracer_bin": tracer_bin}
        if tracer_bin_cfg:
            kw.update(zrange=tracer_bin_cfg["zrange"], z_eff=tracer_bin_cfg["z_eff"])
        def ground_truth_fn(_setup, sample, _kw=kw):
            targets = fourier_space.run_fisher(sample, **_kw)
            vals = fourier_space._fisher_targets_to_emulator_targets(
                targets, tracer_bin, "dr1",
            )
            return dict(zip(target_names, vals))
        return priors, target_names, ground_truth_fn, None

    else:
        raise ValueError(f"Unknown analysis: {analysis}")


def get_default_save_path(analysis: str = "shapefit", quantity: str = "mean",
                          cosmo_model: str | None = None,
                          dataset: str | None = None) -> str:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH is not set; please pass --save-path explicitly.")
    # training_data/{analysis}/[{dataset}/][{cosmo_model}/]{quantity}
    parts = [scratch, "bedcosmo", "num_tracers", "emulator", "training_data", analysis]
    if dataset is not None:
        parts.append(dataset)
    if cosmo_model is not None:
        parts.append(cosmo_model)
    parts.append(quantity)
    return os.path.join(*parts)


def deploy_checkpoint_path(
    data_path: str,
    analysis: str,
    tracer_bin: str | None,
) -> str | None:
    """Deploy path for a per-tracer .pt mirroring training_data under models/.

    ``training_data/bao/dr1/base/config/v2`` + ``LRG1`` →
    ``models/dr1/base/config/v2/LRG1.pt`` (under SCRATCH/bedcosmo/num_tracers/emulator).
    Returns None if ``tracer_bin`` is unset or ``data_path`` is not under
    ``training_data/{analysis}/``.
    """
    if not tracer_bin:
        return None
    marker = os.path.join("training_data", analysis) + os.sep
    norm = os.path.normpath(data_path)
    idx = norm.find(marker)
    if idx < 0:
        return None
    rel = norm[idx + len(marker):]
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    deploy_dir = os.path.join(
        scratch, "bedcosmo", "num_tracers", "emulator", "models", rel,
    )
    return os.path.join(deploy_dir, f"{tracer_bin}.pt")


def compare_losses(
        run_ids: list,
        labels: list | None = None,
        log_scale: bool = True,
        y_lim: tuple | None = None,
        per_step: bool = False,
        ) -> None:
    """Compare train/test loss curves across multiple MLflow runs.

    Args:
        run_ids: List of MLflow run IDs to compare.
        labels: Optional display labels for each run. Defaults to run IDs.
        log_scale: Use log scale for y-axis.
        y_lim: Tuple of (min, max) y-axis limits.
        per_step: If True, plot per-batch losses instead of epoch-averaged.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    mlflow.set_tracking_uri(f"file:{scratch}/bedcosmo/num_tracers/emulator/mlruns")
    client = MlflowClient()

    if labels is None:
        labels = run_ids

    if per_step:
        train_metric, test_metric = "batch_train_loss", "batch_test_loss"
        x_label = "Step"
    else:
        train_metric, test_metric = "epoch_train_loss", "epoch_test_loss"
        x_label = "Epoch"

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for run_id, label in zip(run_ids, labels):
        train_hist = client.get_metric_history(run_id, train_metric)
        test_hist = client.get_metric_history(run_id, test_metric)

        color = None

        if train_hist:
            steps, vals = zip(
                *[(m.step, m.value) for m in train_hist if np.isfinite(m.value)]
            )
            line = ax_train.plot(
                steps,
                vals,
                label=label,
                alpha=0.8,
            )[0]
            color = line.get_color()

        if test_hist:
            steps, vals = zip(
                *[(m.step, m.value) for m in test_hist if np.isfinite(m.value)]
            )
            ax_test.plot(
                steps,
                vals,
                label=label,
                alpha=0.8,
                color=color,
            )

    ax_train.set_xlabel(x_label)
    ax_test.set_xlabel(x_label)
    ax_train.set_ylabel("MSE Loss")
    ax_train.set_title("Train")
    ax_test.set_title("Test")
    if log_scale:
        ax_train.set_yscale("log")
    if y_lim:
        ax_train.set_ylim(y_lim)
    ax_train.legend()
    ax_test.legend()

    fig.tight_layout()
    plt.show()


# Minimum σ(D/rd) applied at decode time for sigma_* emulator outputs.
DEFAULT_SIGMA_FLOOR = 1e-4
_RHO_CLIP = 1.0 - 1e-6


def _symlog_transform_np(y: np.ndarray, linthresh: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.log1p(np.abs(y) / linthresh)


def _symlog_inverse_np(y_log: np.ndarray, linthresh: np.ndarray) -> np.ndarray:
    return np.sign(y_log) * linthresh * np.expm1(np.abs(y_log))


def transform_emulator_targets_forward(
    y: np.ndarray,
    target_names: List[str],
    *,
    log_normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Map physical emulator targets to training space (before z-score)."""
    y = np.array(y, dtype=np.float64, copy=True)
    y_linthresh = None
    if log_normalize:
        y_linthresh = np.empty((1, y.shape[1]), dtype=np.float64)
        for col, name in enumerate(target_names):
            if not name.startswith("sigma_"):
                y_linthresh[0, col] = 1.0
                continue
            abs_nz = np.abs(y[:, col][y[:, col] != 0])
            y_linthresh[0, col] = float(np.min(abs_nz)) if len(abs_nz) > 0 else 1e-8

    for col, name in enumerate(target_names):
        if name.startswith("sigma_"):
            if log_normalize and y_linthresh is not None:
                y[:, col] = _symlog_transform_np(y[:, col, None], y_linthresh[:, col, None])[:, 0]
        elif name.startswith("rho_"):
            y[:, col] = np.arctanh(np.clip(y[:, col], -_RHO_CLIP, _RHO_CLIP))
    return y.astype(np.float32), y_linthresh.astype(np.float32) if y_linthresh is not None else None


def transform_emulator_targets_inverse(
    y: np.ndarray,
    target_names: List[str],
    *,
    log_normalize: bool = False,
    y_linthresh: np.ndarray | None = None,
    sigma_floor: float = DEFAULT_SIGMA_FLOOR,
    apply_sigma_floor: bool = True,
) -> np.ndarray:
    """Map denormalized training-space targets back to physical units."""
    import torch

    if isinstance(y, torch.Tensor):
        out = y.clone()
        for col, name in enumerate(target_names):
            if name.startswith("rho_"):
                out[..., col] = torch.tanh(out[..., col])
            elif name.startswith("sigma_"):
                if log_normalize and y_linthresh is not None:
                    lt = y_linthresh[..., col]
                    out[..., col] = torch.sign(out[..., col]) * lt * torch.expm1(torch.abs(out[..., col]))
                if apply_sigma_floor and sigma_floor > 0:
                    out[..., col] = torch.clamp(out[..., col], min=sigma_floor)
        return out

    out = np.array(y, dtype=np.float64, copy=True)
    for col, name in enumerate(target_names):
        if name.startswith("rho_"):
            out[..., col] = np.tanh(out[..., col])
        elif name.startswith("sigma_"):
            if log_normalize and y_linthresh is not None:
                lt = y_linthresh[..., col]
                out[..., col] = np.sign(out[..., col]) * lt * np.expm1(np.abs(out[..., col]))
            if apply_sigma_floor and sigma_floor > 0:
                out[..., col] = np.maximum(out[..., col], sigma_floor)
    return out


def cov_block_from_marginals(s_DH, s_DM, rho):
    """2×2 (DH/rd, DM/rd) covariance from marginal σ and correlation (PSD if |ρ|≤1)."""
    import torch

    if isinstance(s_DH, torch.Tensor):
        c11 = s_DH * s_DH
        c22 = s_DM * s_DM
        c12 = rho * s_DH * s_DM
        return c11, c12, c22
    c11 = s_DH * s_DH
    c22 = s_DM * s_DM
    c12 = rho * s_DH * s_DM
    return c11, c12, c22


def decode_emulator_outputs(
    y_norm,
    y_mu,
    y_sigma,
    target_names: List[str],
    *,
    log_normalize: bool = False,
    y_linthresh=None,
    sigma_floor: float = DEFAULT_SIGMA_FLOOR,
    apply_sigma_floor: bool = True,
):
    """Invert z-score and per-target transforms to physical emulator targets."""
    y = y_norm * y_sigma + y_mu
    return transform_emulator_targets_inverse(
        y,
        target_names,
        log_normalize=log_normalize,
        y_linthresh=y_linthresh,
        sigma_floor=sigma_floor,
        apply_sigma_floor=apply_sigma_floor,
    )
