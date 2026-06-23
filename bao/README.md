# BAO forecast pipeline

Forecasts DESI BAO distance errors — `σ(D_H/r_d)`, `σ(D_M/r_d)`, `σ(D_V/r_d)` —
as a function of cosmology and tracer count, in **two independent covariance
frames** (Fourier-space and configuration-space), and generates training data
for an emulator that maps

```
(cosmology θ, N_tracers)  ──►  [ σ(D_H/r_d), σ(D_M/r_d), σ(D_V/r_d) ]
```

The same σ-triplet is produced by both frames, so they are directly comparable.

---

## Module layout

```
core.py                  shared base: BAO likelihood + cosmology + sampling engine
├─ fourier_space.py      Fourier-space Fisher backend  (P(k) analytic Gaussian cov)
├─ config_space.py       config-space backend          (ξ(s) Grieb Gaussian cov) + MCMC engine
fkp_analytic_cov.py      FKP / Grieb cov building blocks (used by config_space)

generate_emulator_data.py   unified Fisher training-data runner   (--space fourier|config)
generate_training_data.sh   all-tracer driver -> shared auto-versioned folder
mcmc.py                 unified BAO MCMC                      (--space fourier|config)

reference_sigmas.py      DESI reference σ (bao-recon, desi_data.csv) — frame-agnostic
comparison_plots.py      comparison plots: [forecast|cov|theory] (default forecast)
alpha_sn_check.py        α_SN fit to the bundle cov + amplitude/structure decomposition

model.py                 production emulator architecture(s) (autoresearch promotion target)
analyze_training_data.py physical sanity checks + plot for the generated training data
parse_desi_nz.py         parse DESI _nz.txt -> BAO-bin redshift slices (input prep)
plot_nz_cov_scaling.py   diagnostic: emulator inputs/outputs vs N_tracers, fiducial cosmo
```

The N_tracers box itself is anchored by `ntracers_range` in the repo-root
`util.py`, reading repo-root `tracers.yaml` / `hod.yaml` (shared with `../train.py`).

### Dependency direction (acyclic)

```
              core.py
        (build_bao_likelihood, _to_bao_cosmo_params,
         priors/constraints, generate_dataset engine,
         SIGMA_TARGET_NAMES, _ISO_TRACERS)
                │                │
        ┌───────┘                └───────┐
        ▼                                ▼
  fourier_space.py                 config_space.py
        │         │                ┌──────┘     │
        ▼         └────────┐       ▼            ▼
        └──────► mcmc.py ◄─────┘   (XiSigmaGenerator)
       (--space fourier|config)
                  generate_emulator_data.py  (imports all three)
```

`core` imports neither backend; the two backends are independent siblings; the
runners sit on top. `core` is a library (no CLI).

---

## The two covariance frames

| | **Fourier** (`fourier_space.py`) | **config** (`config_space.py`) |
|---|---|---|
| Observable | P(k) multipoles | ξ(s) multipoles |
| Covariance | analytic Gaussian (FKP + SSC + recon-shot) | first-principles Grieb 2016 Gaussian ξ-cov |
| Fisher | desilike likelihood → q-Fisher → σ | native-ξ Jacobian + Schur-marg → σ |
| Cosmology-varying cov | yes | yes |
| DESI-data dependence | none | none (bundle window/grid only as a fixed frame) |

**Sparse tracers (BGS, QSO; `core._ISO_TRACERS`)** are fit isotropically
(`qiso`, monopole only) in *both* frames, matching DESI Adame+24. `σ(D_V/r_d)`
is the constrained quantity; `σ(D_H/r_d)` and `σ(D_M/r_d)` are the degenerate
projections of the single `qiso` dilation. All other tracers use the anisotropic
`qpar/qper` fit.

At the fiducial cosmology + DR1 `N_tracers`, the two frames agree on every
distance to ~1–2% (a cross-check of the two independent cov pipelines).

---

## Environment

All scripts run from `bao/` with the emulator conda env:

```bash
cd ~/desilike-emulator/bao
LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
  ~/miniconda3/envs/emulator/bin/python <script.py> [args]
```

`SCRATCH` must be set for the default training-data save path.

---

## 1. Emulator training data (Fisher)

`generate_emulator_data.py` is the per-tracer entry point. It draws constrained
Latin-hypercube samples of `(N_tracers, cosmology)`, runs the chosen frame's
Fisher per sample in a spawn `Pool`, and writes versioned `train`/`test` `.npz`.
**`generate_training_data.sh` is the all-tracer driver** — it runs every tracer
into one shared, auto-incremented version folder (no overwrite):

```bash
# all 6 tracers -> next free v{N} (DR1, config space, base model)
bao/generate_training_data.sh --space config --cosmo-model base --n-samples 10000

# DR2 counts — same factors, different anchor (lands in its own v{N})
bao/generate_training_data.sh --dataset dr2 --cosmo-model base --n-samples 10000

# single tracer, by hand (auto-versions per save() call — prefer the driver)
python generate_emulator_data.py --space config --dataset dr1 --tracer-bin LRG2 \
    --cosmo-model base --n-samples 5000 --workers 16
```

- Output: `$SCRATCH/.../training_data/bao/{dataset}/{cosmo_model}/{space}/v{N}/{tracer}_{train,test}.npz`
  with `target_names = [sigma_DH_over_rd, sigma_DM_over_rd, sigma_DV_over_rd]`
  and the cosmo + `N_tracers` inputs in `x`.
- `--dataset` ∈ {`dr1`, `dr2`} picks the DESI release whose `passed` count anchors
  the `N_tracers` box (`tracers.yaml low/high` are factors x that count) and forms
  the `{dataset}` path segment. Train with the matching `train.py --dataset`.
- `--cosmo-model` ∈ {`base`, `base_w`, `base_w_wa`, `base_omegak`, `base_omegak_w_wa`}
  selects which cosmology parameters vary.
- **`--space config` ignores `--area`/`--zrange`/`--z-eff`/`--nz-slices-path`** — its
  survey frame (window + footprint) is fixed by the DESI DR1 bundle. Those flags
  only affect `--space fourier`.

### Legacy Fourier covariance-triplet CLI

`fourier_space.py` has its own `main()` that emits the older **2×2 covariance**
target (`cov_DH_DH, cov_DH_DM, cov_DM_DM`) instead of the σ-triplet — used only
if you specifically need that contract:

```bash
python fourier_space.py --tracer-bin LRG2 --cosmo-model base --n-samples 5000
```

---

## 2. MCMC sampling

One CLI (`mcmc.py`), pick the frame with `--space`; both reduce the posterior
to the same σ-triplet.

```bash
# one CLI, pick the frame with --space {config|fourier}

# Fourier — sample the desilike likelihood object
python mcmc.py --space fourier --tracers LRG2 QSO --cov analytic --dataset dr1
python mcmc.py --space fourier --cov bundle              # DR1 RascalC cov_ξ as precision
python mcmc.py --space fourier --cov bundle --seeds 42 43 44 45 46   # sweep → error bars

# config — native-ξ theory + Grieb/bundle cov (config_space §4)
python mcmc.py --space config                           # all tracers → mcmc_config.json
python mcmc.py --space config --tracers LRG1 --seeds 42 43 44 45 46  # merge-updates LRG1 only
```

- `mcmc.py --space {config|fourier}` is the single MCMC CLI.
  - `--space fourier --cov {analytic|bundle|both}`: `analytic` = pipeline Gaussian
    cov; `bundle` = DESI DR1 cov_ξ substituted via `precision = Mᵀ cov_ξ⁻¹ M`,
    `M = W·H_Hankel`. `--apmode auto` → `qiso` for BGS/QSO. `nwalkers` default 64
    (needs `≥ 2·ndim`). `--save-chain` dumps to `chains_fourier/` (debug).
  - `--space config --cov {gaussxi|bundle|both}`: native-ξ theory + Grieb/bundle
    cov via raw emcee on `config_space.make_log_prob`. `nwalkers` default 48.
- Each space writes one combined seed-sweep file — `mcmc_config.json` /
  `mcmc_fourier.json`, schema `{tracer: {cov: {seed: {DH,DM,DV}}}}`. Each run
  **merge-updates** only the (tracer, cov) it ran over `--seeds`, leaving the rest
  intact; `comparison_plots.py forecast` reads the sweep for both the scatter
  points (mean over seeds) and the σ-of-σ error bars.

Reference σ for both is **DESI bao-recon stat-only** (not `desi_data.csv`, whose
published σ folds in the systematic budget); see `reference_sigmas.py`.

## 3. Comparison plots (`comparison_plots.py`)

One script, three plots, selected by a positional subcommand (default `forecast`):

```bash
# forecast σ: Fisher + MCMC vs DESI bao-recon (default; bare run == forecast config)
python comparison_plots.py                       # → forecast_comparison_config_dr1.png
python comparison_plots.py forecast --space fourier

# ξ-covariance: Grieb Gaussian vs DESI bundle RascalC, 3×6 grid
python comparison_plots.py cov --matrix both     # → cov_comparison_{corr,cov}_dr1.png

# theory ξ: pipeline native ξ(s) vs bundle data
python comparison_plots.py theory                # → theory_comparison_dr1.png
```

- `comparison_plots.py [forecast] --space {config|fourier}` — 3-panel σ(DH/DM/DV)
  comparing, per tracer, Fisher and MCMC in both the analytic and bundle covariance
  against DESI bao-recon. Fisher live; MCMC from the combined seed-sweep json
  (`mcmc_config.json` / `mcmc_fourier.json`) — mean ± σ-of-σ over seeds. Marker
  **color = cov** (analytic/bundle), **shape = estimator** (○ Fisher, ◇ MCMC). →
  `forecast_comparison_{space}_dr1.png`.
- `comparison_plots.py cov --matrix {corr|cov|both}` — 3×6 grid of analytic Grieb
  Gaussian vs DESI bundle RascalC covariance and their difference. →
  `cov_comparison_{matrix}_dr1.png`.
- `comparison_plots.py theory` — 2×3 grid of s²·ξ_ℓ: bundle data vs pipeline native
  ξ-theory (BB=0) vs theory + best-fit broadband. → `theory_comparison_dr1.png`.

For the α_SN covariance analysis (shot-noise rescaling fit to the bundle cov), see
`alpha_sn_check.py`.

The Fourier-frame **bundle** Fisher substitutes the DESI bundle ξ-cov as a Fourier
precision (`Mᵀ cov_ξ⁻¹ M`); the config frame's native-ξ bundle Fisher is the
faithful one (the Fourier M-projection loses information and overshoots).

---

## 4. DESI systematic-error layer (post-emulator)

The emulator predicts **statistical** σ (matched to `bao-recon stat-only`). DESI's
published total adds a small, mock-calibrated systematic budget in quadrature,
`σ_tot² = σ_stat² + σ_sys²`. That budget is *not* first-principles and cannot be
emulated across cosmology/N_tracers, so it is applied as a fixed, opt-in scaling
**after** the emulator — the training data stays clean σ_stat.

```python
from reference_sigmas import apply_desi_syst, DESI_SYST_INFLATION
sigma_tot = apply_desi_syst(sigma_stat_triplet, "LRG2")   # array (...,3) or dict
```

- `DESI_SYST_INFLATION[tracer][q] = σ(bao-recon_syst)/σ(bao-recon_stat-only)` — DESI's
  *measured* per-tracer, per-distance inflation (not a tuned coefficient). Range
  ~1.00–1.07; BGS/QSO ≈ +0.6–0.7 %, largest are LRG3+ELG1 q∥ and ELG2 q⊥ (~+6 %).
- Diagonal scaling on `[DH, DM, DV]` (SIGMA_TARGET_NAMES order); iso tracers (BGS, QSO)
  carry the single qiso factor on all three. Fixed at the DESI fiducial / DR1 n̄ — the
  one assumption is that `σ_sys/σ_stat` is roughly cosmology- and n̄-independent (a
  second-order error since the budget is ≤7 %).
- Regenerate / verify the table from the `_syst` .h5 files (must be downloaded):
  `python reference_sigmas.py --regen-syst` (flags any drift vs the frozen table).

---

## Conventions

- **Fiducial cosmology:** `Ω_m = 0.3152`, `h·r_drag = 99.08` (`core.PARAM_DEFAULTS`
  + the frame `_FID`). `h` is fixed; `r_drag` enters via `r_d = h·r_drag / h_fid`.
- **σ(D_V/r_d)** from the Fourier 2×2 cov uses the AP combination
  `D_V/r_d = (z · (D_M/r_d)² · (D_H/r_d))^(1/3)`
  (`fourier_space._cov_to_sigma_triplet`); the config frame produces it natively.
- **Broadband:** `pcs` (B-spline) for the full likelihood; the analytic-BB Fisher
  uses a separate `power` basis with the theory BB zeroed (identical ξ at BB=0).

---

## Key API (in `core.py` unless noted)

| symbol | role |
|---|---|
| `build_bao_likelihood(..., lean=, apmode=)` | core model build; `lean=True` returns just template + nuisances (config path) |
| `generate_dataset(..., worker_fn, make_task)` | backend-agnostic sampling + parallel engine (requires `worker_fn`) |
| `SIGMA_TARGET_NAMES` | unified emulator target `[sigma_DH_over_rd, sigma_DM_over_rd, sigma_DV_over_rd]` |
| `_ISO_TRACERS` | `("BGS", "QSO")` — isotropic (qiso) tracers |
| `fourier_space.run_fisher` / `_worker_run_fisher_sigma` | Fourier Fisher + σ-triplet worker |
| `config_space.XiSigmaGenerator` | per-tracer config σ-triplet generator (varies cosmology + N_tracers) |
| `config_space.build_native_theory_mcmc` / `make_log_prob` | config-space MCMC engine (§4); emcee driver `_run_emcee` is in `mcmc.py` |
