# ShapeFit forecast pipeline — Changelog

Running log for the full-shape (ShapeFit) forecast pipeline. Conventions and
policies are inherited from `bao/CHANGELOG.md` (no fudge factors; physics
correctness beats reference-matching; every number that becomes a training
label gets a regression harness).

## 1. 2026-07-27 — Ground-up rebuild on the production BAO machinery

The pre-existing `shapefit/` content (`prep_covar.py`, `prep_mean.py`,
`run_single_fisher.py`, `run_prep_covar.sh`) predated all production BAO work
and was deleted: single z-bin toy with a flat `area=14000` footprint, a
hardcoded `b0=0.84` amplitude-scaled bias, no n(z)/HOD/V_eff, the
`peakaverage` de-wiggling default (mislabels σ ~2× across wide priors, bao
§"de-wiggling"), `cov_*` targets that bypass every name-prefix inference
guard, and a `w0_fde`/`wa_fde` typo that meant the w0/wa cosmology never
reached cosmoprimo.

New pipeline (see README.md for the full decision table):

- `core.py::build_shapefit_likelihood` — the bao 21-step spine minus every
  reconstruction step. Shared survey physics imported from `bao/core.py`
  unchanged (HOD b1 + assembly bias + interlopers, n(z) slices, FKP V_eff →
  n_eff Brent mapping, 1-loop pre-recon Σ, SSC, `_cov_to_array`, sampling
  engine). Full-shape specifics: `ShapeFitPowerSpectrumTemplate(
  apmode="qisoqap", with_now="wallish2018")` with `dn` fixed; Kaiser theory
  (pluggable `theory_cls`); FS band kernel `dF/dk ∝ k²` over the fit band
  replacing the BAO Silk kernel in the V_eff mapping and the z_eff weight;
  ells (0,2), klim [0.02, 0.2, 0.005]; FoG in quadrature with Σ∥ (Kaiser has
  a single Gaussian damping, no Lorentzian); no broadband (sn0 only), no
  recon-shot cov term.
- `fourier_space.py` — Fisher → Schur-marginalize all nuisances → 4×4
  (qiso, qap, df, dm) → physical basis via `J = diag(1, 1, f_sigmar_fid, 1)`,
  `m = m_fid + dm` → targets `[sigma_qiso, sigma_qap, sigma_f_sigmar,
  sigma_m] + 6 rho_*` (prefix-compatible with the util/bedcosmo guards).
  Mean worker: `ShapeFitPowerSpectrumExtractor(z=z_eff_tracer,
  with_now="wallish2018")`, runtime params via `_to_mean_extractor_params`
  (Omega_m includes the neutrino density the legacy mapping dropped —
  verified: at the DESI fiducial the extractor returns qiso=qap=1 exactly and
  f_sigmar/m bit-match the covar template's fids).
- Generators + `generate_training_data.sh` (`--quantity covar|mean`, version
  resolved once for all tracers), `regress_sigmas.py` (6 tracers × 8
  cosmologies, exact-equality compare; baseline `golden_4cfd6bec.npz`, 1158
  arrays, all finite), `validate_forecast.py`.
- `util.py`: `get_pipeline` shapefit branches rewired to the new modules
  (explicit-path module loading to avoid the bao/shapefit `core` name
  collision); `to_extractor_params` `w0_fde→w0_fld` fixed in place.
  `model_config.yaml` key `base` → `default` so every `--cosmo-model` finds
  hyperparameters.

Fiducial validation (Kaiser, DR1 passed counts, DESI fiducial cosmology):

| tracer | z_eff | σ(qiso) | σ(qap) | σ(f_sigmar) | σ(m) | σ(fsr)/fsr |
|---|---|---|---|---|---|---|
| BGS       | 0.292 | 0.0149 | 0.0462 | 0.0453 | 0.0500 | 9.6% |
| LRG1      | 0.508 | 0.0096 | 0.0302 | 0.0352 | 0.0345 | 7.4% |
| LRG2      | 0.704 | 0.0077 | 0.0241 | 0.0279 | 0.0276 | 6.0% |
| LRG3_ELG1 | 0.945 | 0.0069 | 0.0208 | 0.0163 | 0.0218 | 3.7% |
| ELG2      | 1.303 | 0.0111 | 0.0312 | 0.0185 | 0.0239 | 4.6% |
| QSO       | 1.343 | 0.0172 | 0.0504 | 0.0303 | 0.0315 | 7.7% |

σ ordering tracks V_eff·nP; fractional f_sigmar errors sit in the DESI DR1
full-shape 4–10% band (qualitative anchor; expected on the tight side —
Kaiser, Gaussian cov, no window/systematics). ρ(qap, f_sigmar) ≈ −0.66…−0.72
across all tracers (AP–RSD degeneracy); ρ(f_sigmar, m) mildly negative.
The 8-point cosmology grid (ω_cdm ∈ [0.05, 0.30], h ∈ [0.45, 0.90],
ln10A_s ∈ [2.3, 3.7], w0wa) produced finite, smoothly-varying σ everywhere —
no wallish2018 instabilities.

## 2. 2026-07-27 — Ω_m domain guard (wallish2018 fails at Ω_m ≳ 3)

The first 256-sample LRG2 generation run showed a 41% failure rate (37% for
the mean pipeline). Root cause: the omega-basis LHS box (`omega_cdm` ∈
[0.01, 0.99] × `h` ∈ [0.2, 1.0], inherited from the legacy shapefit priors
and bedcosmo's `prior_args_fs.yaml`) reaches derived Ω_m =
(ω_cdm+ω_b+ω_ν)/h² ≈ 17 in the high-ω_cdm/low-h corner. There cosmoprimo's
**wallish2018** de-wiggling filter itself produces non-finite `pknow`
(`CubicSpline: y must contain only finite values`, `bao_filter.py:420`) —
every observed failure had Ω_m ≳ 2.9. This is a *different* corner from the
BAO de-wiggling saga (whose (Om, hrdrag) box capped Om ≤ 0.99 by
construction): wallish2018 is the right engine, but no engine is expected to
be meaningful at Ω_m ~ 10.

Fix: `core._check_omega_m` — a fail-fast domain constraint Ω_m ∈
[0.01, 0.99] raised inside both cosmology mappings (covar + mean workers),
mirroring the BAO Om box. Rejected draws cost microseconds (no CLASS init);
the sampling rejection loop refills. Measured after the guard: ~65% of the
raw LHS box is out-of-domain (fast-rejected); **in-domain failure rate
0/21**; the 6×8 regress grid is entirely in-domain and bit-identical to
`golden_4cfd6bec.npz` (193/193 arrays checked on LRG2).

Consequences to carry forward:
- The emulator's valid domain is Ω_m ∈ [0.01, 0.99]; **bedcosmo's fullshape
  prior must enforce the same constraint** (its `constraints:` machinery in
  models.yaml is the natural place) or it will query outside the training
  domain.
- The LHS acceptance rate against the raw box is ~35% by construction —
  generation-time "failed" counts are dominated by the fast domain
  rejection, not compute.

## 3. 2026-07-27 — End-to-end round trip verified

Smoke datasets (LRG2, v99: covar n=256, mean n=512, both regenerated
in-domain after §2) through `train.py --analysis shapefit --quantity
{covar|mean} --cosmo-model base --dataset dr1 --tracer-bin LRG2` →
`eval.py --model-path <deploy .pt>`:

- `model_config.yaml` `default` key resolves for the cosmo model; checkpoint
  records analysis/quantity/param_names/target_names; deploy checkpoint
  lands at `models/dr1/base/{covar|mean}/v99/LRG2.pt`.
- eval auto-selects the ground-truth quantity from the checkpoint, draws
  from the checkpoint's param_names, skips domain-rejected draws (the
  rejection counts in eval logs are the Ω_m guard), and the
  `sigma_`/`rho_` name-prefix transforms apply end to end.
- v99 is smoke-scale only (accuracy numbers are meaningless); production
  training needs the full `generate_training_data.sh` runs.
