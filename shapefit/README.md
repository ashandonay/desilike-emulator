# ShapeFit (full-shape) forecast pipeline

Forecasts DESI full-shape errors on the ShapeFit compressed parameters —
`σ(qiso)`, `σ(qap)`, `σ(f_sigmar)`, `σ(m)` plus their 6 pairwise correlations —
as a function of cosmology and tracer count, and generates training data for
two per-tracer emulators consumed by bedcosmo's `num_tracers` experiment:

```
covar : (cosmology θ, N_tracers) ──► [σ(qiso), σ(qap), σ(f_sigmar), σ(m), ρ×6]
mean  : (cosmology θ)            ──► [qiso, qap, f_sigmar, m]
```

The mean side exists because bedcosmo has no differentiable `f_sigmar`/`m`;
together the two emulators define a per-tracer Gaussian ShapeFit likelihood.

Parallel to the production BAO pipeline in `../bao/` and built on its shared
survey-physics machinery (imported from `bao/core.py`, which stays untouched
and regression-frozen).

---

## Module layout

```
core.py                  constants + cosmology mapping + FS band kernel +
                         build_shapefit_likelihood (imports bao/core.py machinery)
fourier_space.py         Fisher → 4×4 (qiso,qap,df,dm) → physical basis →
                         σ/ρ targets; run_fisher; spawn workers (covar + mean)
generate_emulator_data.py   covar training-data CLI (per tracer)
generate_mean_data.py       mean training-data CLI (per tracer)
generate_training_data.sh   all-tracer driver, --quantity covar|mean,
                            one shared auto-versioned v{N} folder
regress_sigmas.py        bit-exact dump/compare regression harness
validate_forecast.py     fiducial σ table, σ(N) scaling, damping sensitivity
model.py                 emulator architecture (registered for analysis=shapefit)
model_config.yaml        NN hyperparameters (key "default" = all cosmo models)
```

Dependency direction: `core.py` → (`bao/core.py`, `util.py`); `fourier_space.py`
→ `core.py`; the generators/harnesses sit on top. The legacy
`prep_covar.py` / `prep_mean.py` / `run_single_fisher.py` / `run_prep_covar.sh`
were deleted 2026-07-27 (single-bin toys; wrong de-wiggling engine, no
HOD/n(z)/V_eff, guard-less `cov_*` targets).

---

## Analysis configuration (decisions)

| choice | value | why |
|---|---|---|
| Template | `ShapeFitPowerSpectrumTemplate(apmode="qisoqap", with_now="wallish2018")` | `with_now` MUST be explicit; desilike's `'peakaverage'` default mislabels σ ~2× and crashes chaotically across wide priors (bao/core.py:1641 comment). `dn` stays fixed — un-fixing it changes the definition of `m`. |
| Theory | `KaiserTracerPowerSpectrumMultipoles` (pluggable `theory_cls`) | Fast first pass; velocileptors LPT (DESI KP4.5 reference) is the planned drop-in upgrade. |
| Fit range | ells (0, 2), klim `[0.02, 0.2, 0.005]` | DESI KP4.5 full-shape reference config. |
| Tracers | all 6 DR1 bins, all anisotropic | The iso/aniso split is a BAO-recon convention, not a full-shape one. No Lya. |
| Recon | none (pre-recon everywhere) | Full shape is a pre-recon analysis: no Σ_post, no smoothing_scale/bias_recon, no shifted-random shot term. |
| Damping fiducials | pre-recon (linear + 1-loop) Σ⊥, Σ∥; HOD FoG added in quadrature to Σ∥ | Kaiser's damping is a single Gaussian — there is no separate Lorentzian streaming parameter. Floated with N(center, 2.0) priors (`float_sigma_damp`). |
| Broadband | none — `sn0` is the only stochastic freedom | Kaiser has no broadband basis and DESI full-shape has no polynomial broadband either (EFT counterterms/stochastics arrive with velocileptors). Do not graft the BAO `pcs` basis onto FS, and do not tune nuisance choices to close F/D (bao CHANGELOG §33r error-cancellation lesson). |
| Cosmology basis | `omega_cdm, omega_b, h, ln10A_s, n_s` (+ `w0`, `wa`) | Full shape constrains the P(k) shape/amplitude — no (Om, hrdrag) compression. Matches bedcosmo `prior_args_fs.yaml`. No Ω_k models yet (bedcosmo fullshape only has `base`). |
| Domain constraint | derived Ω_m ∈ [0.01, 0.99] (`core._check_omega_m`, fail-fast) | The raw omega box reaches Ω_m ~ 17 (high ω_cdm, low h), where cosmoprimo's wallish2018 filter produces non-finite pknow — ~65% of the raw LHS box is out-of-domain and is rejected before any CLASS init; in-domain failure rate is 0. Mirrors the BAO `Om ∈ [0.01, 0.99]` box. **The bedcosmo prior must carry the same constraint** or it will query the emulator outside its training domain. |
| FS band kernel | `dF/dk ∝ k²` over [0.02, 0.2] | Replaces the BAO Silk kernel `k⁴e^{-k²Σ²}` in the FKP V_eff → n_eff mapping and the z_eff weight; to leading order every mode in the band carries shape/growth information. |
| z_eff | derived per sample from the n(z) slices (covar); fiducial-derived once (mean) | Cosmology-clean; the extractor's `z` is init-time so the mean pipeline freezes it at the fiducial-cosmology value (documented approximation, `--z-eff` to test). |
| Cov | Gaussian (FKP-effective volume) + SSC | Shared `bao/core.py` machinery; recon-shot term dropped (pre-recon). |

**Target contract.** `TARGET_NAMES` = 4 `sigma_*` + 6 `rho_*`. The name
prefixes are load-bearing: `util.transform_emulator_targets_*` and the
bedcosmo decode guards (σ floor/ceiling, tanh ρ-clamp) dispatch on them.
Pairwise ρ-clamps only guarantee PSD per 2×2 block — the bedcosmo-side 4×4
assembly must add an eigenvalue-floor / nearest-PSD projection (follow-up).

**Physical basis.** desilike varies `(qiso, qap, df, dm)`;
`f_sigmar = df · f_sigmar_fid` and `m = m_fid + dm`, so the Jacobian is
`diag(1, 1, f_sigmar_fid, 1)`. `m_fid` is the absolute fiducial slope at the
pivot (≈ −0.58 at the DESI fiducial), not 0. Mean and covar pipelines are on
the identical cosmology mapping (`_to_shapefit_cosmo_params` /
`_to_mean_extractor_params`, which includes the neutrino density the legacy
`util.to_extractor_params` dropped).

---

## Environment

Same as `bao/`: run from `shapefit/` with the emulator conda env,

```bash
cd ~/desilike-emulator/shapefit
LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
  ~/miniconda3/envs/emulator/bin/python <script.py> [args]
```

`SCRATCH` must be set. Pinned deps: desilike @ `4cfd6bec`, cosmoprimo @
`1b100803`, `lsstypes` (install from SHAs, never bare `main` — see
`bao/README.md`). `velocileptors` is used for the 1-loop displacement
variances.

---

## 1. Emulator training data

```bash
# covar (errors), all 6 tracers -> next free v{N}
shapefit/generate_training_data.sh --quantity covar --cosmo-model base --n-samples 5000

# mean, all 6 tracers
shapefit/generate_training_data.sh --quantity mean --cosmo-model base --n-samples 10000

# single tracer by hand (prefer the driver — it pins one shared version)
python generate_emulator_data.py --tracer-bin LRG2 --cosmo-model base \
    --n-samples 5000 --workers 16
```

- Output: `$SCRATCH/.../training_data/shapefit/dr1/{cosmo_model}/{covar|mean}/v{N}/{tracer}_{train,test}.npz`.
- The `N_tracers` box is anchored via `util.ntracers_range` (tracers.yaml
  low/high factors × DR1 `passed` counts). **Never hardcode N** — production,
  validation and plotting must all draw from the util helpers (bao §33n).
- Train with `train.py --analysis shapefit --quantity {covar|mean}
  --cosmo-model base --dataset dr1 --tracer-bin <T>`; eval auto-selects the
  matching ground-truth generator from the checkpoint's recorded quantity.

## 2. Validation

```bash
python validate_forecast.py --check fiducial   # per-tracer σ/ρ table
python validate_forecast.py --check scaling    # σ(N) monotone/saturation plot
python validate_forecast.py --check damping    # float vs fixed damping deltas
```

Fiducial anchor (2026-07-27, Kaiser, DR1 counts): σ(f_sigmar)/f_sigmar spans
3.7% (LRG3+ELG1) – 9.6% (BGS), the same range as DESI DR1 full-shape
σ(fσ8)/fσ8 ≈ 4–10% (direct-fit velocileptors; qualitative anchor only — an
un-windowed Kaiser Gaussian-cov Fisher is expected to sit on the tight side).
ρ(qap, f_sigmar) ≈ −0.66…−0.72 for every tracer (the AP–RSD degeneracy).

## 3. Regression harness

```bash
python regress_sigmas.py dump --out before.npz    # then change the dep
python regress_sigmas.py dump --out after.npz
python regress_sigmas.py compare before.npz after.npz
```

Exact-equality compare over a fixed 6-tracer × 8-cosmology grid (mean +
covar surfaces). Run before/after any desilike/cosmoprimo/scipy/numpy change —
these outputs are emulator training labels. Baseline dump at the current pins:
`golden_4cfd6bec.npz` (gitignored; regenerate with the dump command).

---

## Follow-ups (documented, not built)

- velocileptors LPT theory swap (`theory_cls` is ready; needs the physical
  prior basis + analytic marginalization over `alpha*/sn*`).
- bedcosmo integration: `models.yaml` `shapefit:` block, extend
  `_build_emulator_input`'s feature whitelist to the omega basis, 4×4 block
  assembly with a PSD guard, shapefit reference data files, and a
  `sample_parameters` decision for the shapefit basis.
- Ω_k cosmology models; DR2 anchoring; a DESI FS systematic-error layer.
- MCMC cross-check of the Fisher σ (bao §33t lesson: validate against
  posterior widths before trusting Fisher labels tracer-by-tracer).
