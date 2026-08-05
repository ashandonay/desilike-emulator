import warnings
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


def tracers_for(analysis: str) -> List[str]:
    """Tracer bins belonging to ``analysis``, in tracers.yaml order.

    A block with no ``analyses`` key belongs to every analysis. Exists because
    DESI's 0.8-1.1 bin differs between analyses: LRG3+ELG1 for BAO, LRG3-only
    for full shape (see the tracers.yaml header and shapefit CHANGELOG S31).
    """
    out = []
    for key, cfg in TRACER_CONFIGS.items():
        an = cfg.get("analyses")
        if an is None or analysis in an:
            out.append(key)
    return out


def get_tracer_config(tracer_bin: str, analysis: str | None = None,
                      dataset: str | None = None) -> Dict[str, object]:
    """Return validated tracer config dict.

    With ``analysis``/``dataset`` omitted this is exactly the historical
    behaviour — the base yaml block, minus bookkeeping keys — so existing
    callers (all of ``bao/``, which is regression-frozen) are untouched.

    Passing ``analysis`` validates the bin against the block's ``analyses``
    list and merges any ``overrides`` on top, in the order ``<analysis>`` then
    ``<analysis>/<dataset>``. The override mechanism is wired but only DR1
    values are populated (DR1-first rule).
    """
    key = tracer_bin.strip()
    if key not in TRACER_CONFIGS:
        raise ValueError(f"Unknown tracer bin {tracer_bin!r}. Choices: {TRACER_TYPE_CHOICES}")
    cfg = dict(TRACER_CONFIGS[key])
    if not cfg.get("supported", True):
        raise ValueError(f"Tracer bin {tracer_bin!r} is marked unsupported in tracers.yaml")

    overrides = cfg.pop("overrides", None) or {}
    if analysis is not None:
        allowed = cfg.get("analyses")
        if allowed is not None and analysis not in allowed:
            raise ValueError(
                f"Tracer bin {tracer_bin!r} is not part of the {analysis!r} analysis "
                f"(tracers.yaml declares analyses={list(allowed)}). "
                f"Valid bins: {tracers_for(analysis)}")
        for okey in (analysis, f"{analysis}/{dataset}" if dataset else None):
            if okey and okey in overrides:
                cfg.update(dict(overrides[okey]))
    return cfg


_CSV_NAME_MAP = {"LRG3_ELG1": "LRG3+ELG1", "Lya_QSO": "Lya QSO"}
_NTRACERS_CACHE: Dict[str, Dict[str, float]] = {}  # dataset -> {csv_label: passed}
_COMPONENTS_CACHE: Dict[str, Dict[str, float]] = {}  # dataset -> {component: passed}

_DATASET_AREA_FALLBACK = {"dr1": 7500.0, "dr2": 14000.0}

# Cosmology-independent n(z) slice tables. Release-scoped since S62c; see
# nz_slices_path.
NZ_SLICES_DIR = Path.home() / "data" / "desi" / "nz_slices"

# Small DESI-derived reference tables VENDORED into the repo (S79), so a fresh
# checkout runs the forecast with no downloads at all. 39 KB total: the n(z)
# slice tables, the {tracer}_desi_nx.csv summaries reduced from DESI's randoms
# (which are 20.9 GB and exist only to produce these ~10 KB), and the two
# N_tracers tables.
#
# ~/data/desi WINS when present. The vendored copy is a FALLBACK, not an
# override, so that `make_nz_slices.py --install` keeps taking effect
# immediately and this box's behaviour is unchanged. See data/dr1/PROVENANCE.md.
REPO_DATA_DIR = Path(__file__).resolve().parent / "data"


def _repo_fallback(local: Path, dataset: str, *parts: str) -> Optional[Path]:
    """The vendored twin of `local`, if `local` is absent and the twin exists."""
    if local.exists():
        return local
    cand = REPO_DATA_DIR.joinpath(str(dataset), *parts)
    return cand if cand.exists() else None


def nz_slices_path(filename: str, dataset: str,
                   base_dir=None) -> Path:
    """Resolve a release-scoped n(z) file (shapefit CHANGELOG S62c).

    Layout is ``{base}/{dataset}/{filename}``. The flat ``{base}/{filename}``
    is the pre-S62c layout and is accepted ONLY for dr1, with a warning,
    because flat *is* dr1 -- those files came from a make_nz_slices.py
    hardcoded to the DR1 catalogues, areas, counts and download URL.

    For any other release a missing scoped directory RAISES rather than
    falling back. That asymmetry is the entire point: `ntracers`,
    `tracer_area` and `get_default_save_path` all switch on `dataset`, so a
    release-mixing bug in the n(z) layer would surface only as a subtly wrong
    covariance -- the same silent class as S58 (a fallback that never fired)
    and S59 (a caller never updated).

    Lives here, beside `tracer_area`/`ntracers`/`get_default_save_path`,
    because it is the same kind of release-scoped lookup. It must NOT live in
    bao/core.py: `bao/fkp_analytic_cov.py` needs it too, and a bare
    ``import core`` there resolves to shapefit/core.py whenever cwd is
    shapefit/.
    """
    base = Path(base_dir) if base_dir is not None else NZ_SLICES_DIR
    scoped = base / str(dataset) / filename
    if scoped.exists():
        return scoped

    # Vendored fallback (S79): a fresh checkout has no ~/data at all.
    if base_dir is None:
        vendored = REPO_DATA_DIR / str(dataset) / "nz_slices" / filename
        if vendored.exists():
            return vendored

    flat = base / filename
    if str(dataset) == "dr1" and flat.exists():
        warnings.warn(
            f"n(z) file {filename!r} found only in the flat pre-S62c layout "
            f"({flat}); treating it as dr1. Move these under {base / 'dr1'} -- "
            "the flat fallback is dr1-only and will not serve another release.",
            DeprecationWarning, stacklevel=2)
        return flat

    raise FileNotFoundError(
        f"No n(z) file {filename!r} for dataset {dataset!r}. Looked for {scoped}"
        + (f" (and the dr1-only flat fallback {flat})"
           if str(dataset) == "dr1" else
           f"; the flat fallback {flat} is NOT used for {dataset!r} because "
           "those files are DR1")
        + f". Generate them with shapefit/make_nz_slices.py --dataset {dataset}.")


def tracer_area(tracer_bin: str, dataset: str = "dr1") -> float:
    """Footprint in deg^2 for ``tracer_bin``: `area_deg2` from tracers.yaml.

    The area is NOT one number per release. DESI 2024 II Table 2 gives it per
    tracer CLASS, because priority vetoes remove sky from lower-priority
    samples (a QSO target can veto an LRG) and the imaging vetoes differ:

        BGS 7473    LRG 5740    ELG 5924    QSO 7249

    against the ~7500 nominal DR1 footprint. Using 7500 everywhere inflates V
    by 1.31x for LRG and 1.27x for ELG, which depresses nbar = N/V by the same
    factor -- most of the n(z) gap measured in shapefit CHANGELOG S50 -- and
    inflates the covariance mode count. BGS (0.996) and QSO (0.967) are barely
    affected, which is why they behaved differently from LRG/ELG throughout
    S46-S51.

    ``area`` is survey GEOMETRY and is held fixed: `N_tracers` is the design
    variable and scales the DENSITY within this footprint. Scaling area with N
    instead would leave nbar invariant and the design axis inert.

    Falls back to the release footprint, with a warning, for bins carrying no
    `area_deg2` (e.g. Lya_QSO).
    """
    cfg = TRACER_CONFIGS.get(tracer_bin.strip(), {})
    area = cfg.get("area_deg2")
    if area is not None:
        return float(area)
    try:
        fallback = _DATASET_AREA_FALLBACK[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {dataset!r}; "
            f"known: {sorted(_DATASET_AREA_FALLBACK)}") from None
    warnings.warn(
        f"tracers.yaml has no `area_deg2` for {tracer_bin!r}; falling back to "
        f"the {dataset} footprint ({fallback:.0f} deg^2). Per-tracer areas are "
        "DESI 2024 II Table 2 -- see shapefit CHANGELOG S54.")
    return float(fallback)


def ntracers(tracer_bin: str, dataset: str = "dr1") -> float:
    """Return the DESI 'passed' N_tracers for ``tracer_bin`` from
    ``~/data/desi/bao_{dataset}/desi_data.csv`` (dataset in {dr1, dr2}).

    The HOD M_cut root-find depends on nbar = N_tracers / V_eff, so any
    pipeline configuration that compares predictions against bundle data must
    use the actual sample size for that tracer/release — not a generic default.
    Using a mismatched N silently shifts the HOD-weighted b1 and invalidates
    downstream b1/f_AB calibration.
    """
    # Bins declaring `components` are not in desi_data.csv, which only carries
    # the BAO combinations (there is no LRG3 row, only LRG3+ELG1). Sum the
    # per-component passed counts from desi_tracers.csv instead:
    #     passed = targets x comp x efficiency
    # verified to reproduce DESI 2024 V Table 1 for every tracer.
    cfg = TRACER_CONFIGS.get(tracer_bin.strip(), {})
    components = cfg.get("components")
    if components:
        if dataset not in _COMPONENTS_CACHE:
            import pandas as pd
            tpath = _repo_fallback(
                Path.home() / "data" / "desi" / f"bao_{dataset}" / "desi_tracers.csv",
                dataset, "desi_tracers.csv")
            if tpath is None:
                raise FileNotFoundError(
                    f"No desi_tracers.csv for {dataset!r} in ~/data/desi/bao_{dataset}/ "
                    f"or {REPO_DATA_DIR / dataset}")
            tdf = pd.read_csv(tpath)
            _COMPONENTS_CACHE[dataset] = {
                str(r["tracer"]): float(r["targets"]) * float(r["comp"]) * float(r["efficiency"])
                for _, r in tdf.iterrows()}
        ccache = _COMPONENTS_CACHE[dataset]
        missing = [c for c in components if c not in ccache]
        if missing:
            raise KeyError(f"desi_tracers.csv ({dataset}) has no rows {missing} for "
                           f"{tracer_bin!r}; available: {sorted(ccache)}")
        return float(sum(ccache[c] for c in components))

    if dataset not in _NTRACERS_CACHE:
        import pandas as pd
        path = _repo_fallback(
            Path.home() / "data" / "desi" / f"bao_{dataset}" / "desi_data.csv",
            dataset, "desi_data.csv")
        if path is None:
            raise FileNotFoundError(
                f"No desi_data.csv for {dataset!r} in ~/data/desi/bao_{dataset}/ "
                f"or {REPO_DATA_DIR / dataset}")
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
        # NB: historically emitted "w0_fde"/"wa_fde", which cosmoprimo does not
        # accept (the fluid params are w0_fld/wa_fld) — the old base_w_wa
        # shapefit path was silently broken.
        result["w0_fld"] = float(sample["w0"])
    if "wa" in sample:
        result["wa_fld"] = float(sample["wa"])
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
        if quantity not in ("covar", "mean"):
            raise ValueError(
                f"Unknown quantity for shapefit: {quantity!r} (expected 'covar' or 'mean')"
            )
        if not tracer_bin:
            raise ValueError(
                "shapefit eval requires a tracer_bin (per-tracer z_eff / N box)."
            )
        # Load via explicit file paths with distinct module names: both bao/
        # and shapefit/ carry core.py + fourier_space.py, so a bare
        # `import core` here would collide with the bao branch in-process.
        sf_fs = _load_module(
            "shapefit_fourier", os.path.join(_here, "shapefit", "fourier_space.py"))
        sf_core = sf_fs.sf_core

        if quantity == "covar":
            target_names = list(sf_core.TARGET_NAMES)
            varied = [str(p) for p in param_names] if param_names is not None \
                else (["N_tracers"] + sf_core.COSMO_MODELS["base"])
            priors = {p: dict(sf_core.DEFAULT_PRIORS[p]) for p in varied}
            _nt_lo, _nt_hi = ntracers_range(tracer_bin)
            priors["N_tracers"] = {"dist": "uniform", "low": _nt_lo, "high": _nt_hi}

            # z_eff stays None -> derived per sample inside the likelihood
            # build (matches generate_covar_data.py's default).
            def ground_truth_fn(_setup, sample, _tracer=tracer_bin):
                return sf_fs.run_fisher(sample, tracer_bin=_tracer)
            return priors, target_names, ground_truth_fn, None

        # quantity == "mean": per-tracer extractor, z_eff derived PER SAMPLE.
        target_names = list(sf_core.MEAN_TARGET_NAMES)
        varied = [str(p) for p in param_names] if param_names is not None \
            else list(sf_core.COSMO_MODELS["base"])
        priors = {p: dict(sf_core.DEFAULT_PRIORS[p])
                  for p in varied if p != "N_tracers"}
        # This call site had drifted twice out of step with the worker it calls,
        # and the second one made shapefit mean eval raise on every sample:
        #
        #   1. the task tuple is 6 fields (sample, tracer, z_eff,
        #      param_defaults, area, dataset). It was passing 4, so
        #      _worker_run_mean_targets died on the unpack -- which happens
        #      BEFORE its try/except, so the failure was a hard ValueError, not
        #      the (None, None, traceback) the contract promises. Verified:
        #      "not enough values to unpack (expected 6, got 4)".
        #
        #   2. it pinned z_eff to a single fiducial value. S42 made z_eff depend
        #      on the sampled cosmology AND on N_tracers, and the generator
        #      passes z_eff=None so the worker derives it per sample. Pinning it
        #      here would score the emulator against labels evaluated at a
        #      different redshift from the ones it was trained on -- reviving
        #      exactly the frozen-z_eff bug S42 removed.
        #
        # The area is this TRACER's footprint (S54/S58), not the release's; it
        # is the argument the worker hands to _fs_compute_z_eff. shapefit is
        # DR1-only (generate_covar_data.py restricts --dataset), so pin DR1
        # rather than thread a dataset argument through get_pipeline.
        _area = tracer_area(tracer_bin, "dr1")

        def ground_truth_fn(_setup, sample, _tracer=tracer_bin, _area=_area):
            _s, vals, tb = sf_fs._worker_run_mean_targets(
                (sample, _tracer, None, None, _area, "dr1"))
            if vals is None:
                raise RuntimeError(f"mean extractor failed:\n{tb}")
            return dict(zip(target_names, vals))
        return priors, target_names, ground_truth_fn, None

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


def plots_dir() -> Path:
    """Repo-level `plots/` directory, created on demand.

    Every diagnostic figure lands here rather than beside the script that made
    it, so the analysis directories stay code-only. The whole tree is
    gitignored (both `*.png` and `plots/`) — figures are regenerable, and
    large binaries in git history are not worth the pull cost on a public repo.

    Because the directory is shared across analyses, a figure name that names a
    concept more than one analysis has (covariance, forecast, scaling, training
    data, emulator-vs-DESI, ...) must carry its analysis as a prefix:
    `bao_forecast_comparison_*.png` vs `shapefit_sigma_vs_desi.png`. Names that
    are already unambiguous (`alpha_sn_cov_compare`, the `normalized_nz_*`
    catalogue parses) are left alone.
    """
    d = _THIS_DIR / "plots"
    d.mkdir(exist_ok=True)
    return d


def mlflow_tracking_dir(analysis: str) -> str:
    """Filesystem path of the per-analysis MLflow store.

    Each analysis owns its own store ({analysis}/mlruns) alongside its
    training_data/, models/ and logs/. Note the consequence: run IDs are only
    resolvable within one analysis, so cross-analysis run comparison needs two
    stores opened separately (two `mlflow ui` invocations).
    """
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    return os.path.join(
        scratch, "bedcosmo", "num_tracers", "emulator", analysis, "mlruns")


def mlflow_tracking_uri(analysis: str) -> str:
    """``file:`` URI of the per-analysis MLflow store."""
    return f"file:{mlflow_tracking_dir(analysis)}"


def logs_dir(analysis: str) -> str:
    """Filesystem path of the per-analysis log directory, created on demand.

    Sibling of training_data/, models/ and mlruns/ under
    emulator/{analysis}/. Exists so generation drivers write their logs
    somewhere discoverable instead of wherever the caller happened to
    redirect -- a run whose log lands in a scratch dir is a run nobody can
    audit later.
    """
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    d = os.path.join(
        scratch, "bedcosmo", "num_tracers", "emulator", analysis, "logs")
    os.makedirs(d, exist_ok=True)
    return d


def get_default_save_path(analysis: str = "shapefit", quantity: str = "mean",
                          cosmo_model: str | None = None,
                          dataset: str | None = None) -> str:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH is not set; please pass --save-path explicitly.")
    # {analysis}/training_data/[{dataset}/][{cosmo_model}/]{quantity}
    # The analysis segment is the TOP level under emulator/ so that each
    # pipeline owns its own training_data/, models/ and logs/ subtree. It must
    # stay above the quantity: 'covar' is a valid quantity for both bao (the
    # Fourier Fisher backend) and shapefit, so an analysis-last layout collides.
    parts = [scratch, "bedcosmo", "num_tracers", "emulator", analysis, "training_data"]
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

    ``bao/training_data/dr1/base/config/v2`` + ``LRG1`` →
    ``bao/models/dr1/base/config/v2/LRG1.pt`` (under
    SCRATCH/bedcosmo/num_tracers/emulator). The analysis segment is kept in the
    deploy path — dropping it made bao and shapefit collide whenever they shared
    a quantity name ('covar' is valid for both). Returns None if ``tracer_bin``
    is unset or ``data_path`` is not under ``{analysis}/training_data/``.
    """
    if not tracer_bin:
        return None
    marker = os.path.join(analysis, "training_data") + os.sep
    norm = os.path.normpath(data_path)
    idx = norm.find(marker)
    if idx < 0:
        return None
    rel = norm[idx + len(marker):]
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    deploy_dir = os.path.join(
        scratch, "bedcosmo", "num_tracers", "emulator", analysis, "models", rel,
    )
    return os.path.join(deploy_dir, f"{tracer_bin}.pt")


def compare_losses(
        run_ids: list,
        labels: list | None = None,
        log_scale: bool = True,
        y_lim: tuple | None = None,
        per_step: bool = False,
        analysis: str = "bao",
        ) -> None:
    """Compare train/test loss curves across multiple MLflow runs.

    Args:
        run_ids: List of MLflow run IDs to compare.
        labels: Optional display labels for each run. Defaults to run IDs.
        log_scale: Use log scale for y-axis.
        y_lim: Tuple of (min, max) y-axis limits.
        per_step: If True, plot per-batch losses instead of epoch-averaged.
        analysis: Which per-analysis MLflow store the run IDs live in. Stores
            are separate, so all run_ids must come from the same analysis.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri(analysis))
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


def transform_emulator_targets_forward(
    y: np.ndarray,
    target_names: List[str],
    *,
    log_normalize: bool = False,
    y_linthresh: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Map physical emulator targets to training space (before z-score).

    ``y_linthresh`` (the symlog scale) is derived from ``y`` when None, and returned
    so the caller can reuse it. Pass the TRAIN split's value when transforming any
    other split: linthresh is ``min|nonzero|``, an extreme order statistic that
    differs between splits, so re-deriving it per split puts each split on its own
    scale -- the model then predicts in train-space while the target sits in
    test-space, adding a spurious ~ln(lt_other/lt_train) offset to every large
    target. The inverse (``transform_emulator_targets_inverse``) already takes
    linthresh as an argument for the same reason.
    """
    y = np.array(y, dtype=np.float64, copy=True)
    if log_normalize and y_linthresh is None:
        y_linthresh = np.empty((1, y.shape[1]), dtype=np.float64)
        for col, name in enumerate(target_names):
            if not name.startswith("sigma_"):
                y_linthresh[0, col] = 1.0
                continue
            abs_nz = np.abs(y[:, col][y[:, col] != 0])
            y_linthresh[0, col] = float(np.min(abs_nz)) if len(abs_nz) > 0 else 1e-8
    elif not log_normalize:
        y_linthresh = None
    else:
        y_linthresh = np.asarray(y_linthresh, dtype=np.float64).reshape(1, -1)

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
    sigma_floor: float | None = DEFAULT_SIGMA_FLOOR,
    sigma_ceiling: float | None = None,
) -> np.ndarray:
    """Map denormalized training-space targets back to physical units.

    sigma bounds are two optional clamps, each disabled by passing ``None``:
    ``sigma_floor`` clamps from below (default ``DEFAULT_SIGMA_FLOOR``);
    ``sigma_ceiling`` clamps from above and maps inf/nan -> ceiling (default
    ``None`` = off, so training/eval see the raw, possibly overflowing tail). A
    covariance consumer that needs finite, well-conditioned blocks passes a
    finite ceiling to encode "no information" instead of +inf (see
    ``cov_block_from_marginals``). rho is always pulled strictly inside (-1, 1)
    via ``_RHO_CLIP`` -- mirroring the forward transform's arctanh clip -- so the
    decode never returns a saturated +-1 that would make a 2x2 block singular.
    """
    import torch

    # -inf sigma (from a negative symlog pre-image) collapses to the floor, or
    # to 0 when the floor is disabled -- sigma is a standard deviation, never < 0.
    neg_fill = sigma_floor if sigma_floor is not None else 0.0

    if isinstance(y, torch.Tensor):
        out = y.clone()
        for col, name in enumerate(target_names):
            if name.startswith("rho_"):
                out[..., col] = torch.clamp(
                    torch.tanh(out[..., col]), min=-_RHO_CLIP, max=_RHO_CLIP
                )
            elif name.startswith("sigma_"):
                if log_normalize and y_linthresh is not None:
                    lt = y_linthresh[..., col]
                    out[..., col] = torch.sign(out[..., col]) * lt * torch.expm1(torch.abs(out[..., col]))
                if sigma_ceiling is not None:
                    out[..., col] = torch.nan_to_num(
                        out[..., col],
                        nan=sigma_ceiling,
                        posinf=sigma_ceiling,
                        neginf=neg_fill,
                    )
                if sigma_floor is not None or sigma_ceiling is not None:
                    out[..., col] = torch.clamp(out[..., col], min=sigma_floor, max=sigma_ceiling)
        return out

    out = np.array(y, dtype=np.float64, copy=True)
    for col, name in enumerate(target_names):
        if name.startswith("rho_"):
            out[..., col] = np.clip(np.tanh(out[..., col]), -_RHO_CLIP, _RHO_CLIP)
        elif name.startswith("sigma_"):
            if log_normalize and y_linthresh is not None:
                lt = y_linthresh[..., col]
                out[..., col] = np.sign(out[..., col]) * lt * np.expm1(np.abs(out[..., col]))
            if sigma_ceiling is not None:
                out[..., col] = np.nan_to_num(
                    out[..., col], nan=sigma_ceiling, posinf=sigma_ceiling, neginf=neg_fill
                )
            if sigma_floor is not None:
                out[..., col] = np.maximum(out[..., col], sigma_floor)
            if sigma_ceiling is not None:
                out[..., col] = np.minimum(out[..., col], sigma_ceiling)
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
    sigma_floor: float | None = DEFAULT_SIGMA_FLOOR,
    sigma_ceiling: float | None = None,
):
    """Invert z-score and per-target transforms to physical emulator targets."""
    y = y_norm * y_sigma + y_mu
    return transform_emulator_targets_inverse(
        y,
        target_names,
        log_normalize=log_normalize,
        y_linthresh=y_linthresh,
        sigma_floor=sigma_floor,
        sigma_ceiling=sigma_ceiling,
    )
