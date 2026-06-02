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
mcmc_fourier.py             Fourier-space MCMC                    (--cov analytic|bundle)
mcmc_config.py              config-space MCMC                     (--cov gaussxi|bundle)

reference_sigmas.py      DESI reference σ (bao-recon, desi_data.csv) — frame-agnostic
plot_fisher_vs_desi.py   unified money plot  (--space config|fourier) — Fisher + MCMC vs DESI
```

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
  mcmc_fourier.py          ▼   mcmc_config.py   (XiSigmaGenerator)
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

`generate_emulator_data.py` is the single entry point. It draws constrained
Latin-hypercube samples of `(N_tracers, cosmology)`, runs the chosen frame's
Fisher per sample in a spawn `Pool`, and writes versioned `train`/`test` `.npz`.

```bash
# config-space (first-principles Grieb ξ-cov — the σ driver)
python generate_emulator_data.py --space config  --tracer-bin LRG2 \
    --cosmo-model base --n-samples 5000 --workers 16

# Fourier-space (analytic P-space Gaussian Fisher — same target)
python generate_emulator_data.py --space fourier --tracer-bin LRG2 \
    --cosmo-model base --n-samples 5000 --workers 16
```

- Output: `$SCRATCH/.../training_data/bao/{cosmo_model}/sigma_{space}/v{N}/{tracer}_{train,test}.npz`
  with `target_names = [sigma_DH_over_rd, sigma_DM_over_rd, sigma_DV_over_rd]`
  and the cosmo + `N_tracers` inputs in `x`.
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

One script per frame; both reduce the posterior to the same σ-triplet.

```bash
# Fourier — sample the desilike likelihood object
python mcmc_fourier.py --tracers LRG2 QSO --cov analytic --dataset dr1
python mcmc_fourier.py --cov bundle        # DR1 RascalC cov_ξ substituted as precision

# config — native-ξ theory + Grieb/bundle cov (config_space §4)
python mcmc_config.py --out money          # 1 seed, all tracers  → xi_money_mcmc.json
python mcmc_config.py --out sweep --tracers LRG1 --seeds 42 43 44 45 46
                                           # → seed_sweep_xi/LRG1.json  (chain-noise)
```

- `mcmc_fourier.py --cov {analytic|bundle}`: `analytic` = pipeline Gaussian cov;
  `bundle` = DESI DR1 cov_ξ substituted via `precision = Mᵀ cov_ξ⁻¹ M`, `M = W·H_Hankel`.
  `--apmode auto` → `qiso` for BGS/QSO. Writes `mcmc_results_fourier/`.
  Needs `nwalkers ≥ 2·ndim` (default 64).
- `mcmc_config.py --out {money|sweep}`: `money` writes `xi_money_mcmc.json`
  (one combined file, single seed); `sweep` writes per-tracer
  `seed_sweep_xi/{tracer}.json` over `--seeds` (for chain-noise error bars).
  Both consumed by `plot_fisher_vs_desi.py` / `aggregate_seed_sweep.py`.

Reference σ for both is **DESI bao-recon stat-only** (not `desi_data.csv`, whose
published σ folds in the systematic budget); see `reference_sigmas.py`.

## 3. Money plot

`plot_fisher_vs_desi.py --space {config|fourier}` draws the 3-panel money plot
(σ_DH/rd, σ_DM/rd, σ_DV/rd) comparing, per tracer, Fisher and MCMC in both the
analytic and bundle covariance against DESI bao-recon. It computes the Fisher
series live and loads the MCMC series (config: `xi_money_mcmc.json`, preferring
`seed_sweep_xi/`; fourier: `mcmc_results_fourier/`). Writes `money_{space}_dr1.png`.

The Fourier-frame **bundle** Fisher substitutes the DESI bundle ξ-cov as a Fourier
precision (`Mᵀ cov_ξ⁻¹ M`); the config frame's native-ξ bundle Fisher is the
faithful one (the Fourier M-projection loses information and overshoots).

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
| `config_space.make_log_prob` / `run_mcmc` | config-space MCMC engine (§4) |
