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

## 4. 2026-07-29 — First absolute comparison against DESI DR1 (`compare_to_desi.py`)

Everything above validates the pipeline against *itself*: `regress_sigmas.py`
pins the numbers across a dependency upgrade (that is its stated trigger — not
repo edits), and `validate_forecast.py` checks orderings, scaling laws and
damping sensitivity. None of it can say whether the absolute σ are right.

A Fisher σ has exactly two ingredients — the covariance C and the derivatives
∂P/∂θ — so `compare_to_desi.py` compares each against the DESI DR1 full-shape
products under `~/data/desi/bao_dr1/likelihoods/`. Our klim (0.02, 0.2, 0.005)
× ells (0, 2) is **exactly** DESI's bins 4..39, so nothing is interpolated.
Rows match on `k_edges`, never on centers: DESI stores mode-weighted effective
centers (0.0227 for the 0.020–0.025 bin).

`core.py` additions used by the comparison — `n_eff`/`nbar_comoving`/
`V_survey` in the return dict, and `cov_override` — leave every existing
number untouched (193/193 shared arrays bit-identical to
`golden_4cfd6bec.npz`).

**Sound:** inverting our own covariance through the Gaussian formula returns a
*flat* effective volume 4.94e9 against `V_survey` = 5.18e9 (5%, no k-trend).
The covariance engine does exactly what it claims.

**Finding 1 — the theory model breaks down over the upper half of the fit
band.** LRG2, ours/DESI-measured:

| k | P₀ | P₂ |
|---|---|---|
| 0.023 | 1.285 | 0.932 |
| 0.103 | 1.019 | 0.528 |
| 0.143 | 0.880 | 0.171 |
| 0.183 | 0.658 | **−0.312** |

The Kaiser quadrupole **goes negative above k ≈ 0.16** where DESI measures
≈ +2000. Kaiser × Gaussian FoG cannot hold to k = 0.2. The quadrupole carries
the AP and RSD information, so σ(qap) and σ(f_sigmar) are built from
derivatives of a model that has failed over roughly half the band. Part of the
low-k P₀ excess is window convolution (DESI's spectra are convolved, ours are
not) — but the window does not produce a sign flip.

**Finding 2 — our covariance is ≈1.95× DESI's EZmock** (BGS 1.32, LRG1 2.03,
LRG2 1.98, ELG2 1.95, QSO 1.92). Note the sign: Gaussian+SSC comes out
*looser*, not tighter, than a covariance containing the full non-Gaussian
term. It decomposes cleanly — the (P + P_shot)² amplitude error accounts for
**all** of the k-dependence, leaving a **flat 1.72 with no k-trend**, i.e. a
normalization/volume factor. Four tracers of very different density landing at
~1.95 indicates a common cause, not per-tracer physics.
LRG3_ELG1 reads 0.40 and is **excluded**: DESI's full-shape 0.8–1.1 bin is
LRG-only while ours is the combined LRG+ELG1 BAO bin.

**Finding 3 — n_eff is 23% low**: ours 1.465e-4 vs DESI's measured 1.912e-4
(P_shot 6825 vs 5229). Caveat: DESI's FKP weights use a fixed P₀ while our
n_eff is V_eff-matched to the actual P_g, so the definitions are not
identical — but it is the same quantity in the covariance's role.

**Where the σ actually come from.** Substituting DESI's covariance into our
Fisher moves the σ by only 7–20% (qiso 1.074, qap 1.095, f_sigmar 1.061, m
1.197) — far less than the √2 a 2× covariance naively implies, because DESI's
larger off-diagonal correlations (⟨|corr|⟩ 0.070 vs our 0.012) remove the
information their smaller diagonal adds back. **The covariance is not the
dominant lever on σ; the derivatives are.** Effort belongs on Finding 1.

**Open — the `area` convention.** `area = 14000` deg² is a code default
inherited from `bao/core.py` (the DESI 5-year footprint) used with DR1 counts;
`desi_data.csv` carries no area column. It is the obvious suspect for the flat
1.72, but **the sign is wrong**: 14000 exceeds DESI DR1's actual footprint, so
our volume is too *large*, which makes our covariance *smaller*. Correcting
area alone widens the gap to ~4×. Not changed pending analysis — and it is not
shapefit-specific, since `bao/core.py` carries the same default and lands at a
uniform 0.72–0.80× DESI, which an over-large volume would also produce.

Cheapest next test for Finding 1: refit at `kmax = 0.10` instead of 0.20. If
the σ barely move, the broken high-k region was contributing little
information and the labels are safer than they look. Otherwise the
velocileptors swap (already pluggable via `theory_cls`) is a prerequisite for
trustworthy labels, not a refinement.

Still missing for a full absolute check: DESI's **recorded** per-tracer
compressed values and their σ (qiso, qap, f_sigmar, m). Only the data bundles
and covariances are local — and only LRG2 has a full bundle. The rest needs the
`dr1_full-shape-bao-clustering v1.0` VAC.

## 5. 2026-07-29 — The negative quadrupole is our damping, not Kaiser

§4 attributed the P₂ sign flip to Kaiser breaking down at high k. That was
wrong, and the correction matters because it makes the problem fixable without
swapping the theory. **Bare Kaiser gives a positive, well-behaved quadrupole**
(P₂ = 2645, 2185, 1986 at k = 0.16, 0.18, 0.195). The sign flip comes from the
damping parameters *we* supply.

`KaiserPowerSpectrumMultipoles.calculate` (desilike `full_shape.py:492-497`)
applies

    damping = exp(-k²(σ∥²μ² + σ⊥²(1-μ²))/2)

to the **entire** `pktable`, not to the wiggle component. §1 fed it
`sigmapar = sqrt(Σ∥_pre² + σ_FoG²)` and `sigmaper = Σ⊥_pre`, i.e. the BAO
*de-wiggling* scales. For LRG2 that is σ∥ = 8.18, σ⊥ = 4.28 Mpc/h.

Two things are wrong with that:

1. **Σ∥/Σ⊥ do not belong here at all.** They describe smearing of the BAO
   *feature* by large-scale displacements. In a BAO template the damping
   multiplies only the oscillatory part (`bao.py:126,138` — same algebra,
   applied to the wiggles). Applied to the full spectrum it suppresses the
   broadband, which no physics asks for.
2. **σ⊥ should be 0.** Finger-of-God suppression is purely line-of-sight; a
   transverse Gaussian damping suppresses the monopole isotropically for no
   reason.

Measured effect on the LRG2 quadrupole (DESI = measured, window-convolved):

| k | as-built | no damping | DESI |
|---|---|---|---|
| 0.0225 | 32726 | 33765 | 35129 |
| 0.0975 | 4216 | 8369 | 7417 |
| 0.1475 | 334 | 4761 | 3196 |
| 0.1975 | **−813** | 2927 | 1311 |

The truth sits between the two: real FoG does suppress, but the as-built value
over-damps by enough to flip the sign above k ≈ 0.155, and undershoots P₀ by 34%
at k = 0.18. The indicated fix is `sigmapar = σ_FoG`, `sigmaper = 0` — a removal
of an incorrectly-applied term, not a tuned parameter.

**And the broken region is informative** (`validate_forecast.py --check kmax`,
new). Mean σ inflation from truncating the fit band, vs kmax = 0.20:

| kmax | LRG2 | QSO |
|---|---|---|
| 0.100 | 2.56× | 1.75× |
| 0.150 | 1.23× | 1.11× |
| 0.200 | 1.00× | 1.00× |

P₂ crosses zero at k = 0.156 (LRG2) and 0.142 (QSO), so 24% and 32% of the band
sits past it, and truncating at 0.15 still costs 11–23%. The labels do depend on
the region where the model is misbehaving. This is not a "restrict kmax and move
on" situation.

**Not changed pending a decision.** Fixing the damping alters every training
label and invalidates `golden_4cfd6bec.npz`; §1 recorded the quadrature choice
deliberately, so it is reversed on purpose or not at all. Note the direction:
over-damping *loses* high-k information, so correcting it should tighten σ
further and widen the gap to DESI's published values — the covariance and
damping errors are currently partially cancelling, which is exactly the §33r
failure mode.

**velocileptors status.** REPT is the honest long-term theory (DESI's baseline
is `reptvelocileptors`) and `theory_cls` already accepts it, but it is blocked:
velocileptors 2.3 calls `np.trapezoid`, which is numpy 2.0's rename of
`np.trapz`, and this env is pinned at numpy 1.26.4 for the frozen desilike
`4cfd6bec` / cosmoprimo `1b100803`. Two call sites
(`velocileptors/Utils/qfuncfft.py:181`, `LPT/gaussian_streaming_model_fftw.py:150`).
Measured with a local shim: REPT costs **0.267 s/call vs Kaiser's 0.001 s**
(~270×) with 11 varied nuisances instead of 6, and gives positive high-k P₂.
Pin velocileptors to a numpy-1.x-compatible version rather than upgrading numpy
(which would invalidate the regression baseline) or patching site-packages.

## 6. 2026-07-29 — Damping fixed: `sigmapar = σ_FoG`, `sigmaper = 0`

Applied the §5 fix. `params` now passes the Finger-of-God dispersion alone as
the line-of-sight damping and **zero** transverse damping; the pre-recon BAO
scales Σ∥/Σ⊥ no longer enter the broadband (they are still computed, and still
recorded in the footprint attrs, as diagnostics). `float_sigma_damp` now floats
only `sigmapar` — `sigmaper` is not an uncertain quantity but an absent one, and
floating a physically-zero parameter opens a marginalization direction that
removes real information.

**The quadrupole is positive everywhere.** LRG2 vs DESI measured:

| k | P₂ before | P₂ after | DESI | P₀ ratio before | P₀ ratio after |
|---|---|---|---|---|---|
| 0.0225 | 32726 | 33738 | 35129 | 1.285 | 1.297 |
| 0.1025 | 4216 | 7553 | 6700 | 1.019 | 1.231 |
| 0.1625 | −187 | 3675 | 2319 | 0.736 | 1.170 |
| 0.1825 | −611 | 3067 | 1961 | 0.658 | 1.174 |

The important change is not the sign but the **shape**: the P₀ ratio went from a
2× swing across the band (1.285 → 0.658) to nearly flat (median 1.250, range
1.144–1.353). A flat offset is what a window-convolution difference plus a
modest amplitude offset looks like; the running ratio was a genuine shape error.
The covariance k-trend likewise flattened, 0.63 → 0.89.

Regression partition (`golden_4cfd6bec.npz` vs a fresh LRG2 dump) came out
exactly as designed: **73 damping-independent arrays unchanged, 120
damping-dependent arrays moved, zero exceptions.** Nothing leaked into `z_eff`,
`f_sigmar_fid`, `m_fid`, the observable grid or the mean pipeline.

New fiducial table (DESI fiducial cosmology, DR1 passed counts):

| tracer | z_eff | σ(qiso) | σ(qap) | σ(f_sigmar) | σ(m) | σ(fsr)/fsr |
|---|---|---|---|---|---|---|
| BGS       | 0.292 | 0.0119 | 0.0351 | 0.0395 | 0.0391 | 8.3% |
| LRG1      | 0.508 | 0.0080 | 0.0240 | 0.0313 | 0.0275 | 6.6% |
| LRG2      | 0.704 | 0.0066 | 0.0196 | 0.0251 | 0.0223 | 5.4% |
| LRG3_ELG1 | 0.945 | 0.0060 | 0.0172 | 0.0148 | 0.0179 | 3.4% |
| ELG2      | 1.303 | 0.0097 | 0.0254 | 0.0170 | 0.0213 | 4.3% |
| QSO       | 1.343 | 0.0152 | 0.0434 | 0.0280 | 0.0269 | 7.1% |

All σ tightened 9–19% (LRG2: qiso 0.00769 → 0.00658, m 0.02762 → 0.02231), as
§5 predicted — over-damping was destroying high-k information. ρ(qap, f_sigmar)
≈ −0.73…−0.80 still, and ρ(qap, m) collapsed to ≈ 0, which is sensible: with no
transverse damping there is no longer a spurious isotropic suppression coupling
the AP ratio to the shape slope.

**Two things remain open, and they now point in the same direction.**

1. Our P is ~1.25× DESI's *measured* multipoles, near-uniformly. DESI's spectra
   are window-convolved and ours are not, and the bundle ships the 72×1050
   window matrix — so this can be settled properly by convolving our theory
   rather than argued about. That is the next step.
2. The covariance excess grew from 1.98× to 2.92× (expected: C ∝ (P+P_shot)²,
   and P is no longer suppressed) and is now nearly flat in k, i.e. clearly a
   normalization rather than a shape problem. With the shape error removed, the
   flat factor is the whole story, and 1 may well explain part of it.

Because §5's prediction held — the σ tightened, widening the gap to DESI — the
earlier apparent agreement with DESI's 4–10% band was partly two errors
cancelling. The current table is *more* correct and looks *less* reassuring.
That is the expected direction of travel, not a regression.

## 7. 2026-07-29 — Window convolution: settles one gap, not the other

`compare_to_desi.py --check window` applies DESI's window matrix to our theory
and our covariance, following the precedent already in production on the BAO
side: `bao/config_space.py:416` computes `C = W @ C_theory @ W.T` for the
correlation pipeline, justified at `config_space.py:399-400` because W is
"computed from the RANDOM catalog — geometry, not data". That is what
distinguishes it from the rejected data-derived rescalings: a randoms-derived
window describes the survey mask, not the measured clustering. It is also
frame-fixed there, held at the fiducial frame together with FKP volume/n̄.

Neither Fourier path has ever had one — `grep` finds nothing in
`bao/fourier_space.py`, and `bao/core.py` carries only
`kmin_window = 2π/L_survey`, commented as "a simplified approximation to the
full DESI window-function". shapefit inherited that. So the missing window is an
**inherited asymmetry between the config-space and Fourier lineages**, not a
policy choice.

W for LRG2 is 72 × 1050: 72 measured points out, and a theory side of ells
(0, 2, 4) × 349 k over [0.001, 0.349] plus 3 rotation/photo systematic columns
we zero. The ell=4 input matters — the window leaks hexadecapole power into the
measured monopole and quadrupole, so the convolution needs a multipole our
forecast does not otherwise compute.

**Engine control, run as part of the check and not optional.** The windowed
covariance is built with `bao/fkp_analytic_cov.py` (the theory grid needs ells
(0,2,4) at dk = 0.001 down to k = 0.001, outside what the desilike observable
path is built for), while the unwindowed one comes from desilike's
`ObservablesCovarianceMatrix`. A naive before/after would conflate the window
with the engine swap. Running the analytic engine on the observable grid with no
window: **median ratio 0.989, range [0.909, 1.030]** against desilike+SSC. The
engines agree to ~1%, so the window numbers below are the window.

**Result 1 — the window does NOT explain the theory amplitude offset.**

    P0 median ratio vs DESI measured:  raw 1.250  ->  convolved 1.286

It gets marginally *worse*. The reason is that DESI's baseline uses a **rotated**
window (`wmatrix='rotated'` in the KP fit config), which is constructed to be
compact and near-diagonal precisely so it barely redistributes the mean. So the
~25-29% excess in our P₀ is a **genuine amplitude discrepancy** — b₁, n_eff or
growth — and not a comparison artifact. Combined with §4's finding that our
n_eff is 23% low (P_shot 30% high), the b₁/n_eff normalization is now the prime
suspect, and it is the same normalization that sets the covariance.

**Result 2 — the window dominates the covariance, and overshoots.**

| | diag ratio vs DESI | mean \|off-diag corr\| |
|---|---|---|
| ours, unwindowed | 1.930 | 0.016 |
| ours, windowed | **0.489** | 0.049 |
| DESI | 1.000 | 0.095 |

Applying the window moves us from 2× too large to 2× too small, and closes
about half the off-diagonal correlation deficit. Direction is now physically
sensible — a Gaussian covariance *should* fall below an EZmock covariance
carrying the non-Gaussian term — and the θ-cut in the bundle product removes
modes, inflating DESI's side further. But a factor 2 is larger than the BAO
experience (config/bundle 0.66-0.88, `project_gaussian_xi_cov_findings`).

Note the comparison baseline differs between checks: §4/§6 quote the **plain**
covariance file (2.92× after the damping fix), this section the **bundle's**
rotated+θ-cut covariance (1.93× unwindowed). Different DESI products, not a
changed result.

**What this establishes.** The window is a first-order effect on the covariance
that shapefit omits entirely, so the Fourier path is missing a term the
config-space path has had in production all along. That is a real modelling gap,
not a comparison detail. It does not by itself close the gap — it overshoots —
which means there is a second normalization issue, and Result 1 says the same
thing from the theory side.

**Open, and now sharpened to one question:** an amplitude/normalization error
common to the theory and the covariance, ~1.25× in P and ~2× in C (C ∝ P², so
1.25² ≈ 1.56 of the 2× would follow from the P offset alone). n_eff being 23%
low is the leading candidate, and `area = 14000` deg² — the DESI 5-year
footprint used with DR1 counts — remains an unexamined geometry input feeding
both. `fkp_analytic_cov`'s `P_FKP = 1e4` default matches DESI's LRG FKP weight
choice, but would need per-tracer values beyond LRG2.

## 8. 2026-07-29 — DR1 area is 7500 deg², not 14000

`shapefit` defaulted to `area = 14000.0`, inherited from `bao/core.py`. The repo
already had an explicit convention saying that is wrong for DR1 —
`bao/mcmc.py:66`:

    _DATASET_AREAS = {"dr1": 7500.0, "dr2": 14000.0}

and it is honoured everywhere the production BAO code touches DR1:
`bao/config_space.py:56` (`_AREA = 7500`, the config-space σ-triplet driver),
`bao/desi_reference.py:33`, and `bao/plot_nz_cov_scaling.py:51` (commented
"DR1 footprint (7500)"). Only the Fourier lineage carries 14000, so a DR1-only
pipeline was forecasting DR1 galaxy counts spread over the **DR2** footprint —
nearly twice the sky.

Fixed (completed in §9): `core.DATASET_AREAS` drives the
`build_shapefit_likelihood` and `run_fisher` defaults, and
`generate_covar_data.py --area` now defaults to `None` and resolves from
`--dataset` (printing which footprint it used) rather than hardcoding.

Effect on LRG2 at the fiducial:

| | area 14000 | area 7500 |
|---|---|---|
| b1 | 2.4178 | 2.1705 |
| V_survey | 5.181e9 | 2.775e9 |
| n_eff | 1.465e-4 | 2.734e-4 |
| **P₀ / DESI measured** | **1.250** | **1.032** |

The theory amplitude discrepancy is gone. Larger area at fixed N depresses
n̄ = N/V, and the HOD responds by raising b1 for the rarer sample; since P ∝ b1²,
a 11% b1 error is a 25% power error. Landing at 1.032 is strong independent
evidence that 7500 is the right footprint — a wrong area would not put P₀ on top
of DESI's measured monopole.

New fiducial table (all σ loosened 6–27%, as expected from the smaller volume):

| tracer | z_eff | σ(qiso) | σ(qap) | σ(f_sigmar) | σ(m) | σ(fsr)/fsr |
|---|---|---|---|---|---|---|
| BGS       | 0.292 | 0.0140 | 0.0419 | 0.0448 | 0.0494 | 9.5% |
| LRG1      | 0.508 | 0.0097 | 0.0291 | 0.0356 | 0.0350 | 7.5% |
| LRG2      | 0.704 | 0.0079 | 0.0237 | 0.0285 | 0.0284 | 6.2% |
| LRG3_ELG1 | 0.945 | 0.0066 | 0.0194 | 0.0164 | 0.0217 | 3.7% |
| ELG2      | 1.303 | 0.0099 | 0.0264 | 0.0167 | 0.0235 | 4.2% |
| QSO       | 1.343 | 0.0146 | 0.0417 | 0.0253 | 0.0272 | 6.4% |

**§7's "one common normalization" hypothesis was wrong.** The area fix closed
the theory gap and left the covariance essentially untouched: unwindowed
1.93 → 2.05, windowed 0.489 → 0.512 against the bundle covariance. The reason is
cancellation — halving the area halves V (raising the covariance) but doubles n̄
(lowering the shot-noise term), and the smaller b1 lowers P as well. These are
two independent problems, not one.

**What remains, in order.**

1. **Covariance is ~2× too small after windowing** (0.512). The direction is now
   physically sensible — an analytic Gaussian *should* fall below an EZmock
   covariance carrying the non-Gaussian term, and the bundle's θ-cut removes
   modes, inflating DESI's side — but a factor 2 exceeds the BAO experience
   (config/bundle 0.66–0.88, `project_gaussian_xi_cov_findings`). Off-diagonal
   correlation is also still short: 0.051 windowed vs DESI's 0.095.
2. **n_eff now overshoots the other way.** P_shot ours 3657 vs DESI's measured
   5229, a ratio of 0.699 (it was 1.305 at area 14000). Our sample is ~43% too
   dense, which *suppresses* the shot term and is a live candidate for item 1.
   Caveat as in §4: DESI's FKP weights use a fixed P₀ = 1e4 and include the
   random-catalog term, while our n_eff is V_eff-matched to the actual P_g, so
   the two are not the same functional and need not agree exactly.
3. **P₂ is over-predicted at high k** — 1.32× at k = 0.12 rising to 1.91× at
   k = 0.198. This is Kaiser without EFT counterterms and is a genuine model
   limitation rather than a bug; it is the argument for REPT (blocked on the
   numpy pin, §5), not something to tune.

The Fourier path in `bao/core.py` carries the same `area = 14000.0` default and
was **not** touched here — it is regression-frozen, and config-space is the
production BAO driver. Worth checking separately whether anything still consumes
the BAO Fourier path for DR1.

## 9. 2026-07-29 — Area is now DERIVED from the dataset, not a frozen default

§8's fix was incomplete. It replaced one hardcoded number with another
(`_DEFAULT_AREA = DATASET_AREAS["dr1"]`) and only `generate_covar_data.py`
actually resolved the footprint from `--dataset`. Three places still pinned
14000, and one of them mattered a lot:

- `generate_mean_data.py:103` — `--area` still defaulted to 14000, so the whole
  **mean** pipeline was still computing z_eff volume weights on the DR2
  footprint.
- `regress_sigmas.py:68` — `_AREA = 14000.0`, pinned independently of `core.py`.
  The regression harness would have gone on validating the DR2 footprint
  *after* the pipeline was corrected — i.e. the one tool whose job is to catch
  configuration drift was itself the source of it.
- `build_shapefit_likelihood` / `run_fisher` took no `dataset` at all, so the
  dataset→area relationship was not encoded anywhere in the library; it was a
  constant that happened to be right for DR1.

Now:

    core.dataset_area(dataset) -> DATASET_AREAS[dataset], raising on unknown

`build_shapefit_likelihood(area=None, dataset="dr1")` and `run_fisher(...)`
resolve `area = float(area) if area is not None else dataset_area(dataset)`, so
an explicit override still wins but there is no frozen default to inherit. Both
generators default `--area` to `None` and resolve from `--dataset`.
`regress_sigmas._AREA` and `compare_to_desi`'s analytic-covariance helpers now
call `dataset_area("dr1")` rather than carrying their own copy.

Verified end to end:

    dataset_area dr1 = 7500.0 | dr2 = 14000.0
    regress_sigmas._AREA = 7500.0
    dataset_area('dr3') -> ValueError: No footprint area for dataset 'dr3'
    run_fisher default (dr1) : sigma_qiso=0.00787  sigma_m=0.02836
    run_fisher dataset='dr2' : sigma_qiso=0.00658  sigma_m=0.02231

The dr2 line reproduces the old (wrong-for-DR1) numbers exactly, which confirms
the only thing §8 changed was the footprint, and that DR2 forecasts will now get
the right area for free when shapefit is extended past DR1.

Lesson worth keeping: the original bug was a default constant carried into a
context where it did not apply, and the first fix repeated the same pattern one
level down. The dataset is the input; the area is a function of it.

## 10. 2026-07-29 — P_shot is NOT the covariance bug (retracts §8's lead)

§8 named the n_eff overshoot as the leading candidate for the remaining
covariance deficit. That was wrong, and the test is cheap enough that it should
have been run before the claim.

DESI's shot noise is pypower's `shotnoise_nonorm / wnorm`, i.e. the FKP
estimator's S/A = ∫dV n̄w² / ∫dV n̄²w² with w = 1/(1 + n̄P_FKP) — an n̄w²-weighted
mean of 1/n̄, not 1/n̄ anywhere. In `fkp_analytic_cov`'s notation that is
**I12/I22**, computable from the very slices the config-space pipeline uses.

LRG2, area 7500, P_FKP = 1e4:

| method | P_shot | vs DESI |
|---|---|---|
| DESI's own formula I12/I22 (config-space machinery) | 3598 | 0.688 |
| volume-weighted 1/⟨n⟩ | 3595 | 0.688 |
| shapefit's Brent n_eff collapse | 3657 | 0.699 |
| **DESI measured** | **5229** | 1.000 |

**The three reductions agree to 2%.** Two consequences:

1. The Brent V_eff→n_eff collapse is not lossy in any way that matters. It
   reproduces DESI's own functional on the same inputs.
2. **`bao/config_space.py` carries the identical offset** — the first row was
   computed with `fkp_analytic_cov._fkp_integrals` at config-space's own
   `_AREA = 7500` via the shared `load_nz_slices`. So this is a property of the
   shared n(z) inputs, not a shapefit defect. Since config-space is the
   validated production driver, an offset it also carries is not the bug.

The likelier explanation is that DESI's measured value is not the same quantity:
`num_shotnoise` sums w²_tot over data plus α²-scaled randoms, with completeness
weights inside w_tot. Both inflate it relative to ∫dV n̄w² from a smooth n(z),
and both are absent from our side by construction. Not measured here, so not
claimed as the full explanation — only as the right order.

**Loose end, flagged not resolved.** P_shot ∝ area, so matching 5229 would want
an effective area of ~10900 deg² (7500 × 5229/3598 = 10900; 14000 × 5229/6713 =
10905 — consistent from either end). But §8's theory-amplitude test independently
pinned 7500 through the HOD b₁, landing P₀/DESI at 1.032. Those point at
different areas. P₀ is a direct comparison against DESI's measured monopole while
P_shot compares across a known weighting-convention mismatch, so P₀ is the one to
trust — but this is unresolved.

**Where that leaves the covariance.** Back to unexplained, but better posed:
config-space sits at 0.66–0.88 against DESI's covariance
(`project_gaussian_xi_cov_findings`), the same direction as our windowed 0.512.
So part of the deficit is the known Gaussian-vs-non-Gaussian shortfall shared
with the validated pipeline, and only the excess beyond ~0.7 needs its own
explanation.

## 11. 2026-07-29 — Correction: velocileptors is NOT blocked

§5 reported the REPT swap as blocked because velocileptors 2.3 calls
`np.trapezoid` (numpy 2.0's rename of `np.trapz`) against a numpy-1.26.4 pin.
That is wrong. `bao/core.py:27-29` already installs the shim, with a comment
naming velocileptors as the reason:

    # velocileptors uses np.trapezoid (numpy >= 2.0); shim for numpy 1.x.
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz

The §5 probe failed only because it was a standalone script that never imported
`bao_core`. Re-run inside the pipeline's own import context, REPT builds and
evaluates with no manual shim, returning a positive high-k quadrupole
(P₂ = 4196, 3629 at k = 0.16, 0.195).

So REPT is available now. The real cost is the measured one: **0.267 s/call vs
Kaiser's 0.001 s** with 11 varied nuisances instead of 6, which is a generation
-budget question, not an environment blocker.

Separately, this surfaced a latent fragility: `shapefit/core.py` calls
`np.trapezoid` at module level (line 273) and only worked because importing
`bao_core` above it patched numpy globally. Any reordering or deferral of that
import would have broken shapefit at import time. It now guards locally instead
of relying on another module's side effect.

## 12. 2026-07-29 — Mean pipeline validated (`validate_mean.py`)

The mean pipeline's only prior validation was the fiducial identity — at the
DESI fiducial the extractor returns qiso = qap = 1. That holds **by
construction** and proves nothing, which is why the two bugs this pipeline has
actually suffered (`w0_fde` never reaching cosmoprimo; an Ω_m that dropped the
neutrino density) both survived it: at the fiducial there is nothing to map
wrongly. `validate_mean.py` targets the mapping and the compression separately,
over an 11-point cosmology grid.

**1. Cosmology mapping round trip — 11/11 exact.** Build the cosmology our
mapping produces, read the parameters back out of cosmoprimo, compare to what we
asked for. Worst relative delta 7e-7 (CLASS's own precision), including
`omega_cdm` reconstructed via Ω_m — the quantity where the neutrino density was
previously dropped — and `w0_fld`/`wa_fld`, the `w0_fde` failure mode. Note
cosmoprimo exposes `Omega0_*` and no `omega_cdm`, so physical densities are
recovered as `Omega0_x · h²`.

**2. qiso / qap vs distances computed directly — agree to 1e-7 and 1e-11.**
Genuinely independent: qiso and qap are pure background quantities, so this is a
second implementation rather than a paraphrase. The qiso residual is CLASS
interpolation noise.

**3. f_sigmar / m vs a transcription of the extractor's definitions — 5e-6 to
1.4e-4.** Cross-implementation, not independent: it re-derives desilike's own
formulas from cosmoprimo primitives, so it validates our *usage* (z, with_now,
fiducial, mapping) and not the formulas. Largest residuals sit at the
cosmologies furthest from fiducial in shape, consistent with de-wiggling plus
a two-point finite difference.

Two convention traps were caught while writing the check, both in the check
rather than the pipeline, and both worth recording because they are easy to
repeat:

- `qap = DH_over_DM / fid`, **not** DM/DH. An inverted qap shows up as exact
  reciprocals, which is a recognisable signature.
- desilike's `DH = (c/1e3)/(100·efunc(z))` and `DM = comoving_angular_distance`
  are **already Mpc/h**, as is `rs_drag` (99.08 at the fiducial). Adding explicit
  `·h` factors cancels in DM/r_d but leaves a spurious 1/h in DH/r_d — so it
  corrupts only the rows where h is varied, and looks like a physics
  disagreement rather than a units bug.

**4. covar-vs-mean fiducials: a definitional difference, not an error.** The
covar template sets `fiducial=theta_cosmo`, so s = r_d(θ)/r_d(fid) = 1 and it
evaluates the ShapeFit slope at kp = 0.03 exactly. The mean extractor keeps the
DESI fiducial, so s ≠ 1 and it evaluates at kp/s. **The two report the slope at
different pivots.** Exactly the rows that move r_d disagree (lowOc 11%, highOc
22%, lowh 21%, highh 19%, lowob 2.7%) while those that do not (lowA, highA,
lowns, highns, w0wa) agree to ~1e-4. The sign confirms the mechanism: lower
ω_cdm raises the sound horizon, s > 1, smaller pivot, less negative slope
(−0.864 vs −0.959).

Consequence for bedcosmo, flagged not fixed: the mean pipeline supplies the
central value of m in the DESI-fiducial convention while the covar pipeline
supplies σ(m) in θ's own convention. σ(m) is the error on a shape slope and is
only weakly pivot-dependent, so pairing them is approximately consistent — but
it is an approximation nobody has quantified, and it should be before the two
emulators are combined in a likelihood.

**Verdict: the mean pipeline is sound.** The mapping is exact, the background
compression is exact, and the shape compression reproduces to 1e-4. Nothing here
blocks training the mean emulator.

## 13. 2026-07-29 — External anchor: Lai et al. (arXiv:2404.07283)

The DESI KP5 ShapeFit-vs-Full-Modelling paper (Lai et al., PyBird) gives the
first external reference for our σ that does not require the DR1 VAC. It is a
better match to our setup than the DR1 data paper in three ways: **cubic boxes,
so no window** (we have none either); **kmax = 0.20**, our fit range; and the
template cosmology set to the truth, so α⊥ = α∥ = rA = 1 — *the same convention
as our covar pipeline* (`fiducial=theta_cosmo`).

Reported at kmax = 0.20, single-box covariance 8 (Gpc/h)³:

| tracer | σ(α⊥) | σ(α∥) | σ(rA) | σ(m) |
|---|---|---|---|---|
| LRG z=0.8 | ~0.86% | ~1.75% | ~4.8% | ~0.029 |
| ELG z=1.1 | ~0.79% | ~1.6% | ~3.2% | ~0.033 |
| QSO z=1.4 | ~1.6% | ~2.6% | ~6.4% | ~0.037 |

Our LRG2 has V = 2.775 (Gpc/h)³, so σ ∝ 1/√V gives √(8/2.775) = 1.70. Converting
their basis approximately (qiso = α∥^⅓ α⊥^⅔, qap = α∥/α⊥):

| | ours | Lai, rescaled | ratio |
|---|---|---|---|
| σ(qiso) | 0.0079 | ~0.0136 | 0.58 |
| σ(qap) | 0.0237 | ~0.033 | 0.72 |
| σ(f_sigmar)/f_sigmar | 6.2% | ~8.2% | 0.76 |
| σ(m) | 0.0284 | ~0.049 | 0.58 |

**0.58–0.76×**, the same direction and rough size as the BAO pipeline's 0.72–0.80
against DESI. Qualitative only: their LRG is z = 0.8 at a cubic-box number
density rather than DR1 LRG2's, the intervals are asymmetric MCMC quantiles
symmetrised here, and volume rescaling wrongly assumes the shot-noise term scales
with V.

**This reframes REPT, and the reframing is the point.** PyBird marginalizes a
full EFT counterterm and stochastic set. Our Kaiser varies **6** parameters
(qiso, qap, dm, df, b1, sn0); REPT varies **11** (+b2p, bsp, alpha0p, alpha2p,
sn2p). Marginalizing more nuisances loosens σ, and that is plausibly most of the
0.58–0.76 gap — more than the covariance normalization is worth.

So the theory choice is not a cosmetic repair of the high-k quadrupole (§5, §6).
**The nuisance count is a first-order effect on σ itself**, which is precisely
the quantity the emulator exists to deliver for experimental design. A Kaiser
forecast will systematically under-report errors regardless of how well the
covariance is fixed. Combined with §11 (REPT is not blocked — the shim already
exists) and its measured 0.267 s/call, this is arguably now the highest-leverage
remaining item, ahead of the covariance normalization.

Caveat in the other direction, for honesty: some of the gap is real physics we
should NOT close. A Fisher forecast at the truth is optimistic relative to an
MCMC posterior with prior volume effects, and the BAO pipeline already
established that Fisher under-predicts MCMC for a simpler likelihood. Do not
tune nuisance choices to close the ratio — the §33r error-cancellation lesson.

## 14. 2026-07-29 — REPT costs 1.6×, not 270×, and it closes the σ gap

Measured end to end on LRG2 at the fiducial, DR1 counts (`theory_fiducial_params`
added to `core.py` so fiducial nuisances dispatch on the theory class):

| | build | Fisher | total | varied |
|---|---|---|---|---|
| Kaiser | 3.62 s | 3.75 s | **7.37 s** | 7 |
| REPT | 5.34 s | 6.22 s | **11.55 s** | 11 |

**1.57× per sample.** The 270× in §5/§11 was the per-theory-call ratio (0.267 s
vs 0.001 s) and was badly misleading as a budget number: a sample is dominated
by CLASS and the covariance build, not by theory calls. Generation cost is not a
constraint on this decision.

The σ move much further than the runtime:

| | Kaiser | REPT | ratio |
|---|---|---|---|
| σ(qiso) | 0.00787 | 0.01363 | 1.73 |
| σ(qap) | 0.02370 | 0.04504 | 1.90 |
| σ(f_sigmar) | 0.02854 | 0.04641 | 1.63 |
| σ(m) | 0.02836 | 0.06025 | 2.12 |
| σ(fsr)/fsr | 6.17% | 10.03% | 1.63 |

Against the §13 anchor (Lai et al., rescaled to our volume):

| | Kaiser vs Lai | REPT vs Lai |
|---|---|---|
| σ(qiso) | 0.58 | **1.00** |
| σ(qap) | 0.72 | 1.36 |
| σ(f_sigmar)/fsr | 0.76 | 1.22 |
| σ(m) | 0.58 | 1.23 |

σ(qiso) lands on 0.01363 against Lai's rescaled 0.0136. **This confirms §13: the
nuisance count, not the covariance, was the dominant cause of the σ deficit.**
Kaiser under-reports the errors by 1.6–2.1×. For a pipeline whose whole output
is σ feeding an experimental-design likelihood, that is the largest error this
validation has found — bigger than the damping bug (§5) and bigger than the
footprint bug (§8).

Coming out 1.2–1.4× *looser* than Lai on three of four is unexplained and left
untuned. Candidates: §13's crude α⊥/α∥ → qiso/qap conversion ignores their
correlation; their LRG is z = 0.8 at a cubic-box density, not DR1 LRG2's; and
every counterterm sits at its desilike prior centre rather than an HOD-informed
value.

`theory_fiducial_params` raises for an unrecognised theory instead of falling
back. Deliberate: an under-parameterised forecast does not error, it silently
marginalises over too little and returns σ that are too tight — exactly the
failure this section documents.

Kaiser is behaviour-preserved by the refactor: σ = 0.00787 / 0.02370 / 0.02854 /
0.02836 reproduces §8 exactly.

## 15. 2026-07-29 — DESI's recorded ShapeFit constraints (`desi_reference.py`)

The reference this whole effort was missing. DESI 2024 V (arXiv:2411.12021)
Appendix A, "Datavectors and covariances for the compressed ShapeFit
parameters", publishes per-tracer 4-vectors and full 4×4 covariances for all six
DR1 bins, in **both** ShapeFit-alone (Eqs. A.1–A.12) and ShapeFit+BAO
(A.13–A.24) variants. Transcribed into `shapefit/desi_reference.py` — the
full-shape analogue of `bao/desi_reference.py`.

Obtained by downloading the PDF and running `pdftotext -layout`; every web
fetcher truncates around §4.4, well before the appendix. `data.desi.lbl.gov` was
simultaneously down ("power outage maintenance of the underlying Spin service at
NERSC"), so the VAC route was unavailable anyway — the paper turned out to be
the faster path and needs no NERSC access at all.

DESI reports `[D_V/r_d, D_H/D_M, f σ_s8, m+n]`, one division from our basis:
qiso = (D_V/r_d)/fid, qap = (D_H/D_M)/fid. Ratios leave correlations untouched,
so **the six rho targets compare with no conversion whatsoever** — which matters,
because nothing had ever validated them.

We use the **ShapeFit-alone** variant. The +BAO fits are tighter, particularly
on qiso, and comparing our pre-recon power-only forecast against them would
flatter us.

### LRG2, ours (REPT) vs DESI

| | ours | DESI | ratio |
|---|---|---|---|
| σ(qiso) | 0.0136 | 0.01762 | 0.77 |
| σ(qap) | 0.0450 | 0.05171 | 0.87 |
| σ(f_sigmar)/f_sigmar | 10.03% | 10.96% | 0.92 |
| σ(m) | 0.0602 | 0.06901 | 0.87 |

**0.77–0.92**, in the direction a Fisher forecast at the truth should sit
relative to an MCMC posterior carrying prior-volume effects, and comparable to
the BAO pipeline's 0.72–0.80.

### Correlations — first validation of 6 of the 10 targets

| | ours (REPT) | DESI | Kaiser |
|---|---|---|---|
| ρ(qiso,qap) | +0.269 | +0.239 | **−0.110** ✗ |
| ρ(qiso,f_sigmar) | +0.056 | −0.013 | +0.248 |
| ρ(qiso,m) | −0.169 | −0.330 | −0.373 |
| ρ(qap,f_sigmar) | −0.694 | −0.542 | −0.754 |
| ρ(qap,m) | −0.052 | −0.200 | +0.004 |
| ρ(f_sigmar,m) | +0.074 | +0.242 | **−0.244** ✗ |

**REPT reproduces five of six signs; Kaiser gets two backwards** — ρ(qiso,qap)
and ρ(f_sigmar,m), both of which are LARGE in DESI. REPT's one miss,
ρ(qiso,f_sigmar) at +0.056 vs −0.013, is a pair where both values are consistent
with zero. (An earlier draft of this section said "all six" — wrong; corrected,
and see §16 for the all-tracer picture, which is worse.) Independent of the σ
argument in §14, and it bears on the correlation targets specifically, which the
Lai comparison could not reach.

### Full DESI reference, in our target basis

| tracer | z_DESI | σ(qiso) | σ(qap) | σ(fsr)/fsr | σ(m) |
|---|---|---|---|---|---|
| BGS | 0.30 | 0.04656 | 0.09429 | 24.94% | 0.16724 |
| LRG1 | 0.51 | 0.01859 | 0.06023 | 12.51% | 0.06994 |
| LRG2 | 0.71 | 0.01762 | 0.05171 | 10.96% | 0.06901 |
| LRG3 | 0.92 | 0.01479 | 0.04771 | 11.20% | 0.05906 * |
| ELG2 | 1.32 | 0.02028 | 0.06821 | 9.93% | 0.06601 |
| QSO | 1.49 | 0.02135 | 0.05669 | 10.23% | 0.05125 |

`*` DESI's 0.8–1.1 bin is **LRG-only**; our LRG3_ELG1 is LRG+ELG1. Different
sample, different density — not comparable.

ρ(qap, f_sigmar) sits at −0.53 to −0.63 for every tracer, so the AP–RSD
degeneracy is a robust structural feature and a good target to reproduce. BGS is
an outlier throughout, and DESI flags why: its α_AP is prior-dominated, "highly
affected by the flat prior between 0.8 and 1.2", so its σ(qap) is a prior width
rather than a measurement.

**This answers the original question.** For LRG2 on the covar side, with REPT:
yes, we recover DESI's recorded ShapeFit errors to 0.77–0.92 and their
correlation structure with the right signs. With Kaiser we did not — 0.41–0.56
on the σ, and two correlations inverted.

## 16. 2026-07-29 — All six tracers vs DESI (`--check compressed`)

§15 read LRG2 off by hand. `compare_to_desi.py --check compressed [--theory
kaiser|rept]` now scores every tracer against `desi_reference.py`. LRG2 was
flattering.

**Transcription validated independently.** DESI's ShapeFit+BAO BGS entry
(Eq. A.13/A.14) gives D_V/r_d = 7.920703 ± 0.15489; `desi_data.csv`'s BAO-only
value is 7.925129 ± 0.15074 — 0.06% on the centre, 2.8% on σ. BAO dominates D_V,
so those should nearly coincide, and they do. Appendix A came across correctly.

σ ratios, ours (REPT) / DESI, ShapeFit-alone:

| tracer | qiso | qap | fsr/fsr | m |
|---|---|---|---|---|
| BGS | 0.59 | 0.93 | 0.67 | 0.68 |
| LRG1 | 0.97 | 0.98 | 1.03 | 1.07 |
| LRG2 | 0.77 | 0.87 | 0.91 | 0.87 |
| LRG3_ELG1 | 0.69 | 0.68 | 0.48 | 0.96 * |
| ELG2 | 0.64 | 0.55 | 0.54 | 0.87 |
| QSO | 0.79 | 0.83 | 0.73 | 0.99 |

`*` sample mismatch, not comparable (ours is LRG+ELG1, denser, hence tighter).

**LRG1 is essentially exact (0.97–1.07). Agreement then degrades with redshift**,
with ELG2 worst at 0.54–0.64. That is a trend across tracers, not scatter, and it
is the thing to chase next: a uniform offset would be a normalisation, a
z-dependent one is physics. Candidates: our z_eff being Fisher- rather than
volume-weighted (QSO 1.343 vs 1.484), the FoG/counterterm treatment at higher z,
and the n̄P regime differing most for the sparse high-z tracers.

**Correlations: 6 sign disagreements out of 36.** §15's "five of six" was LRG2
only. Most sit on pairs where DESI's own value is near zero (ρ(qiso,f_sigmar)
spans −0.03 to +0.15), so their sign carries no information. Two are
substantive: **ELG2 ρ(qiso,qap)** ours −0.08 vs DESI +0.17, and **QSO
ρ(f_sigmar,m)** ours −0.01 vs DESI +0.24.

Two systematic biases hold for *every* tracer:

- ρ(qap, f_sigmar) consistently **too strong**: ours −0.65…−0.73, DESI −0.53…−0.63
- ρ(f_sigmar, m) consistently **too weak**: ours +0.01…+0.07, DESI +0.23…+0.37

Structural, not noise, and they matter: ρ is 6 of the 10 emulator targets, and
bedcosmo assembles the 4×4 from them.

BGS remains a special case — DESI flags its α_AP as prior-dominated, so its
σ(qap) is a prior width and the 0.93 ratio there is not a physics result.

## 17. 2026-07-29 — Correction: σ(qiso)/σ(qap) denominators must be the fiducial

§15–§16 divided DESI's σ(D_V/r_d) and σ(D_H/D_M) by the **measured** central
value to get σ(qiso) and σ(qap), justified as a "~1% approximation". That
justification was asserted, not measured, and it is wrong.

qiso ≡ (D_V/r_d)/(D_V/r_d)_fid is a ratio to the FIDUCIAL, so
σ(qiso) = σ(D_V/r_d)/(D_V/r_d)_fid exactly. Measured/fiducial actually runs:

| tracer | D_V/r_d meas/fid | D_H/D_M meas/fid |
|---|---|---|
| BGS | 0.952 | 0.996 |
| LRG1 | 0.975 | 0.972 |
| LRG2 | 0.948 | 1.030 |
| LRG3 | 0.996 | 1.037 |
| ELG2 | 0.976 | 0.939 |
| QSO | 0.988 | 1.008 |

Up to 5–6%, not 1%. These are genuine DR1 deviations, not an error in the
fiducial: checked against `desi_data.csv`, our fiducial reproduces DESI's
published BAO to 1–2% for most quantities, with LRG1 D_H/r_d at 0.923 being
DR1's documented low point.

Using the measurement inflated DESI's σ and so deflated every ratio.
`desi_reference.fiducial_dv_dhdm(z)` now computes the fiducial from cosmoprimo
at DESI's z_eff, with desilike's BAOExtractor conventions.

Corrected ratios (ours REPT / DESI):

| tracer | qiso | qap | fsr/fsr | m |
|---|---|---|---|---|
| BGS | 0.62 | 0.94 | 0.67 | 0.68 |
| LRG1 | 0.99 | 1.01 | 1.03 | 1.07 |
| LRG2 | 0.82 | 0.85 | 0.91 | 0.87 |
| LRG3_ELG1 | 0.69 | 0.66 | 0.48 | 0.96 * |
| ELG2 | 0.65 | 0.58 | 0.54 | 0.87 |

Conclusions from §16 are unchanged: LRG1 essentially exact, agreement degrading
with redshift. Only the numbers shift, by up to 5%.

Correlations are unaffected — a ratio to a constant leaves ρ invariant, which is
why the ρ comparison needed no fiducial in the first place.

## 18. 2026-07-29 — Comparison plots (`comparison_plots.py`)

Modelled on `bao/comparison_plots.py`: positional subcommand, default `sigma`,
plus `rho`, `mean` and `all`. Reference throughout is `desi_reference.py`
(ShapeFit-alone). Outputs `shapefit_{sigma,rho,mean}_vs_desi.png`.

**`sigma`** — four stacked panels, σ(qiso), σ(qap), σ(fσ_r)/fσ_r, σ(m), with
Kaiser and REPT markers against DESI and the ours/DESI ratio annotated. The
Kaiser–REPT separation is the visual headline: Kaiser sits roughly half of DESI
on every panel and every tracer, REPT sits on top of it for LRG1 and within
10–20% for LRG2/QSO.

**`rho`** — the six correlations, which had no external check before this. Two
Kaiser failures are now visible as systematic, not incidental:

- ρ(qiso,qap): Kaiser is **negative for every tracer**, DESI positive for four
  of six. A sign error across the board, not a per-tracer fluctuation.
- ρ(f_sigmar,m): Kaiser sits at −0.25…−0.35 throughout, DESI at +0.2…+0.37.
  Same story.

REPT fixes both in sign. Residual REPT issues, also systematic:

- ρ(f_sigmar,m) is right in sign but **too weak** — near 0 against DESI's +0.24
- ρ(qiso,qap) and ρ(qiso,f_sigmar) go wrong for **ELG2 and QSO specifically**
  (ρ(qiso,qap) dips negative where DESI is positive; ρ(qiso,f_sigmar) reaches
  +0.41 against DESI's +0.08/+0.15)

ρ(qap,f_sigmar), the dominant AP–RSD degeneracy, is the best-behaved panel:
both theories track DESI's −0.53…−0.63, REPT slightly over-strong.

That ELG2 and QSO are the offenders in ρ matches §16's finding that σ agreement
degrades with redshift. Two independent symptoms pointing at the same place is
worth more than either alone, and makes the high-z tracers the next thing to
chase.

**`mean`** — carries a caveat that is the point of the panel rather than a
footnote. At the DESI fiducial cosmology our mean pipeline returns qiso = 1,
qap = 1 and dm = 0 **by construction**, so three of the four panels test whether
DESI's *data* is consistent with the fiducial, not whether our pipeline is
right. Only f_sigmar is predictive: f(z)·σ_r is an absolute number we compute
from the input cosmology. Both z_eff are annotated there, since fσ8 evolves fast
and ours are Fisher-weighted against DESI's volume-weighted. m is plotted as a
deviation on both sides — DESI's is already one, ours is absolute, and plotting
them raw would show a spurious ~0.58 offset (§17).

---

## §19 — The DESI survey window, done consistently, is a no-op for ShapeFit

§16/§18 left "we never apply DESI's survey window" as the leading suspect for
the residual σ gap. Tested on LRG2, the only tracer whose full-shape bundle we
have locally (`likelihood_spectrum-poles-rotated_syst-hod_LRG_GCcomb_z0.6-0.8_thetacut0.05.h5`).
**It is not the explanation.**

### Getting the window into desilike at all

Two separate obstacles, both now cleared:

1. Passing the bundle *path* makes lsstypes return the whole
   `GaussianLikelihood`, and desilike falls through to its deprecated pypower
   branch → `AttributeError: 'GaussianLikelihood' object has no attribute
   'deepcopy'`. Pass `bundle.window`, not the path.
2. `bundle.window` still fails, at `window.py:340`, `wmatrix.theory.ells`. This
   was previously written up here as a desilike-4cfd6bec-vs-lsstypes-1.1.0 API
   mismatch. **That was wrong.** The DR1 bundles ship the *rotated* window,
   whose theory axis is an `ObservableTree` — the P0/P2/P4 spectrum block
   (3×349) plus two `rotation` and one `photo` nuisance-template columns — and
   a tree legitimately has no `.ells`. Select the spectrum block on both axes:

   ```python
   W = (bundle.window.at.theory.get(observables='spectrum')
                     .at.observable.get(observables='spectrum'))   # (72, 1047)
   ```

   No desilike fork, no version bump. This does drop the 3 systematic-template
   columns DESI marginalizes over, which makes the result mildly optimistic.

### The covariance does not follow the window

`ObservablesCovarianceMatrix` has **no window handling at all** — grep
`wmatrix|window` in `observables/galaxy_clustering/covariance.py` returns
nothing. So `wmatrix=W` convolves the theory and its derivatives and leaves the
covariance unconvolved. Measured on LRG2:

| | diag ours/DESI | P0 nearest-neighbour corr |
|---|---|---|
| no window | 1.586 | 0.059 |
| `wmatrix=W`, desilike's covariance | 1.594 | 0.063 |
| `C_obs = M C_kin M^T` | **0.815** | **0.788** |
| DESI (EZmock, rotated) | 1 | 0.666 |

That middle row is a trap: smoothed derivatives against an unsmoothed
covariance double-counts the window's information loss and inflates σ by
1.38×/1.30× on qiso/qap. Any number produced that way is meaningless.

The two outer rows are the real finding. Our analytic Gaussian covariance was
never 1.6× too large — that ratio was comparing an unconvolved covariance to a
convolved one. Rotated properly it lands at **0.815** of DESI's, with the right
correlation structure, which is exactly the modest non-Gaussian deficit the BAO
side already documents (config/bundle 0.66–0.88, `bao` CHANGELOG). `compare_to_desi.py --check cov` compares the *unrotated* covariances and so
still reports ~1.6; its docstring now carries the corrected reading.

`C_kin` is built by calling `build_shapefit_likelihood` again on the window's
own theory grid — `ells=(0,2,4)`, `klim_spec=(0.0005, 0.3495, 0.001)`,
`skip_kmin_guard=True` (new kwarg; the guard would clamp kmin to 0.0052 and
break the grid match). 1047×1047, 36 s. The k grids agree to 2.8e-16.

### Result: the window barely moves the compressed parameters

LRG2, REPT, ours/DESI:

| | no window | W, derivs only | **W, consistent** | DESI |
|---|---|---|---|---|
| σ(qiso) | 0.82 | 1.12 | **0.81** | — |
| σ(qap) | 0.85 | 1.10 | **0.79** | — |
| σ(fσ_r)/fσ_r | 0.91 | 1.00 | **0.85** | — |
| σ(m) | 0.87 | 0.89 | **0.80** | — |

Every ρ moves by less than 0.03 between the unwindowed and consistently-windowed
runs. Derivative smoothing and covariance correlation cancel almost exactly, and
what survives makes the fit *slightly tighter*, not looser.

### Consequences

- **Production stays windowless.** It costs 36 s/cosmology (≈15× the current
  per-sample cost) to change σ by ~5% in the wrong direction. The hook and
  `skip_kmin_guard` stay for diagnostics.
- **The high-z degradation is not geometry.** ELG2/QSO at 0.54–0.65 (§16) has
  to be physics or nuisance freedom. The remaining full-shape bundles drop off
  the critical path — worth having, no longer blocking.
- **The residual ~20% is accounted for.** Our covariance sits at 0.815 of
  DESI's, worth ~0.90 on σ on its own; the rest is Fisher-vs-MCMC, the effect
  already established on the BAO side. Neither is a knob to tune — the first is
  the known non-Gaussian deficit, the second is expected from a Gaussianised
  posterior.

  **It is NOT a thin nuisance set** — an earlier draft of this section said we
  marginalize b1p/sn0/σ_par against DESI's full EFT basis. Wrong, and worth
  recording so it does not get re-derived. Dumping `likelihood.varied_params`
  on the REPT path gives 11: qiso, qap, dm, df plus **b1p U[0,3], b2p N(0,5),
  bsp N(0,5), alpha0p N(0,12.5), alpha2p N(0,12.5), sn0p N(0,2), sn2p
  N(0,5)** — `prior_basis='physical'` ships the whole set with the KP5 prior
  widths, and all 7 are Schur-marginalized. The `{b1, sn0, σ_par}` set is the
  *Kaiser* path, retired in §15. `float_sigma_damp` is a no-op under REPT,
  since `theory_fiducial_params` hands it only `b1p`.

  **And there are no model differences left either** — checked line by line
  against arXiv:2411.12021 §4.7 and Table 4:

  | DESI baseline | ours |
  |---|---|
  | ℓ = 0, 2 — hexadecapole explicitly dropped (§4.7 item 3: "it causes stronger prior weight effects") | ℓ = (0, 2) ✓ |
  | 0.02 < k < 0.20, Δk = 0.005 | `_KLIM = (0.02, 0.2, 0.005)` ✓ |
  | velocileptors **EPT** (§4.7 item 2; Fig. 10 legend "velocileptors EPT (baseline)") | `REPTVelocileptors...` ✓ |
  | b3 **fixed null** ("quite degenerate with the counterterms") | not varied ✓ |
  | α4, SN4 — absent, follow ℓ=4 | absent ✓ |
  | α_iso, α_AP U[0.8,1.2]; f/f_fid U[0,2] | qiso, qap, df ✓ |
  | (1+b1)σ8 U[0,3]; b2σ8², bsσ8² N[0,5²] | b1p, b2p, bsp ✓ |
  | α0, α2 N[0,12.5²] | alpha0p, alpha2p ✓ |
  | SN0 N[0,2²]×1/n̄; SN2 N[0,5²]×f_sat σ_v²/n̄ | sn0p, sn2p ✓ (that normalisation *is* `prior_basis='physical'`) |
  | m U[−0.8, 0.8] | dm U[−3, 3] — **the only difference, and inert** |

  The `dm` width does not matter: a uniform prior contributes nothing to a
  Fisher matrix, and DESI's own σ(m) ≈ 0.069 is nowhere near ±0.8, so it is not
  truncating their posterior either. Table 4's caption calling the biases
  "Lagrangian basis" is the parameterisation, not the code — the baseline is
  Eulerian EPT.

  So the modelling side is settled and should not be reopened as an explanation
  for the residual.

---

## §20 — DESI's fiducial cosmology, and two transcription errors it exposed

"Which cosmology are the published ShapeFit numbers tied to?" — §4.7 item 10
and Table 6, row 1: **AbacusSummit c000, "Planck ΛCDM"**.

| ω_b | ω_cdm | h | 10⁹A_s | n_s | N_ur | w₀ | w_a |
|---|---|---|---|---|---|---|---|
| 0.02237 | 0.1200 | 0.6736 | 2.0830 | 0.9649 | 2.0328 | −1 | 0 |

N_ur = 2.0328 is one massive neutrino at 0.06 eV. One cosmology does both jobs:
the **grid** cosmology that turns redshifts into comoving distances, and the
ShapeFit **template** cosmology generating the reference P_lin(k) — *"For
simplicity, this cosmology, namely template cosmology, is chosen to be the same
as the grid cosmology."* This is exactly cosmoprimo's `("DESI", {})`, which the
pipeline already uses.

### Table 11 validates our fiducial distances

Appendix C publishes the fiducial D_M/r_d, D_H/r_d, D_V/r_d, D_H/D_M, σ_s8 and
fσ_s8 per bin. `fiducial_dv_dhdm` reproduces them to **≤0.13%** on all six
tracers, and our r_d = 99.0844 Mpc against their 99.0792 (0.005%). Independent
confirmation that our fiducial cosmology *and* our D_H/D_M/D_V conventions are
DESI's. (Their caption labels r_d "Mpc/h"; that is a slip — in Mpc/h it is
66.74.)

Table 11 is now transcribed into `desi_reference._T11` and
`sigma_targets` divides by **those** published values rather than recomputing
them. There is no reason to recompute a number DESI printed.
`fiducial_dv_dhdm` stays as the cross-check.

### Two errors of ours that this caught

1. **Rounded z_eff.** The Appendix A headings round to 2 d.p. ("at z_eff =
   0.30"); Table 1 has 0.295, 0.510, 0.706, 0.919, 1.317, 1.491. We transcribed
   the headings, and those z were feeding the fiducial denominators in
   `sigma_targets`. Corrected shifts in DESI's σ:

   | | σ(qiso) | σ(qap) |
   |---|---|---|
   | BGS | +1.39% | −1.69% |
   | LRG2 | +0.43% | −0.74% |
   | ELG2 | +0.11% | −0.31% |
   | LRG1 / LRG3 / QSO | ≤0.09% | ≤0.19% |

   BGS was the worst by an order of magnitude — a ~1.5% error on the tracer
   whose scorecard entry (0.62/0.94/0.67/0.68) is already the odd one out.
   Also fixed `_DESI_ZEFF` in `compare_to_desi.py`, which carried the **BAO**
   paper's 0.930 and 1.484 for LRG3 and QSO instead of full-shape's 0.919 and
   1.491.

2. **Wrong fiducial A_s.** Every fiducial sample in `shapefit/` used
   `ln10A_s = 3.044`, the Planck 2018 TT,TE,EE+lowE+lensing value. DESI's
   fiducial is A_s = 2.083e-9, i.e. **ln10A_s = 3.036394**. We were 0.76% high
   in A_s, 0.38% in σ8, and therefore 0.38% high in every mean-pipeline
   f_sigmar. Negligible for the σ targets (a fiducial shift, not a derivative
   one) but wrong, and free to correct. Changed in `compare_to_desi.py`,
   `validate_forecast.py`, `validate_mean.py`, `generate_mean_data.py` and the
   `regress_sigmas.py` probe grid.

The regress grid change means the golden baseline must be regenerated — it was
already stale from the §14 damping fix, the §15 theory dispatch and the §18
area fix, so this costs nothing extra. **Any σ ratio quoted before this entry
carries the z-rounding error**, largest for BGS.

---

## §21 — Golden regression baseline regenerated

`golden_4cfd6bec.npz` had been stale since §14 and was invalidated again by
§20's probe-grid change. Regenerated, 1158 arrays, ~11 min.

**Reproducibility verified.** Two independent dumps from the same tree compare
**bit-identical on all 1158 arrays** — which is the harness's entire purpose,
since these are emulator training labels and it exact-compares.

Old vs new differs on **1050 / 1158** arrays. Largest relative deltas are QSO
`C_gauss`/`C_total` at ~1.95, consistent with §18's footprint fix (14000 →
7500 h⁻²Mpc² is 1.867 in a covariance). At the fiducial cosmology, the ten
emulator targets moved:

| new/old | σ(qiso) | σ(qap) | σ(fσ_r) | σ(m) |
|---|---|---|---|---|
| BGS | 0.945 | 0.907 | 0.987 | 0.988 |
| LRG1 | 1.005 | 0.963 | 1.012 | 1.014 |
| LRG2 | 1.023 | 0.984 | 1.022 | 1.027 |
| LRG3_ELG1 | 0.960 | 0.932 | 1.009 | 0.998 |
| ELG2 | 0.891 | 0.848 | 0.903 | 0.984 |
| QSO | 0.852 | 0.829 | 0.834 | 0.863 |

ρ shifts are larger than the σ ones and systematic in sign across all six
tracers: ρ(f_sigmar,m) −0.15…−0.25, ρ(qiso,m) −0.01…−0.19, ρ(qiso,qap)
−0.09…−0.14, ρ(qiso,f_sigmar) +0.00…+0.06.

These are the *cumulative* effect of §14 (damping), §18 (area) and §20 (A_s),
and this comparison does not separate them — the old dump predates all three
and no intermediate baselines were kept. It is recorded as a magnitude check,
not an attribution.

`z_eff` also moved, which is correct and not a bug: the FKP band weight runs
through P_g(k), so a change in A_s shifts the weighting and hence the
Fisher-weighted z_eff.

### Two caveats on this baseline

1. **It is a Kaiser baseline.** `build_shapefit_likelihood` still defaults to
   `KaiserTracerPowerSpectrumMultipoles` (core.py:391) and the harness does not
   override it, so none of §15/§16's REPT work is exercised here. Promoting
   REPT to the production default invalidates this file and requires another
   regeneration.
2. It is not evidence that any of §14/§18/§20 is *right* — only that the
   pipeline is now self-consistent and reproducible. Correctness against DESI
   is §16/§19/§20's business.

---

## §22 — REPT promoted to the production theory

The open question was never whether REPT models better — §15/§16 settled that,
and §20 confirmed it *is* DESI's baseline. It was robustness: Kaiser is
analytic and cannot fail, REPT is a loop integrator, and the `base` box runs to
ω_cdm ∈ [0.01, 0.99], h ∈ [0.2, 1.0]. The ≥95% acceptance target had only ever
been measured against Kaiser.

Audit: 64 Latin-hypercube draws over the real `base` box, LRG2, both theories,
same draw, single process.

**REPT and Kaiser fail on exactly the same samples.** Identical failure lists,
and every failure is `unphysical Omega_m outside [0.01, 0.99]` — the prior box
rejecting itself before any theory is evaluated. REPT adds **zero** failures.
That is the whole robustness question, answered.

Two things the audit turned up that are worth recording:

- **Acceptance is 42.2%, and that is the prior box, not the theory.** Ω_m =
  (ω_cdm + ω_b)/h² exceeds 0.99 over most of the box. Consistent with the known
  "Ω_m box keeps ~38% and skews high". Costs attempts, not samples: the
  generator loops until `n_samples` accepted rows exist.
- **Cost is ~3.2×, not the 1.57× carried since §15.** 27 accepted samples took
  1.5 min under Kaiser and 4.8 min under REPT (≈3.3 s vs ≈10.7 s each). The
  audit script's "1.02× median" line is meaningless and should be ignored —
  more than half the draws fail instantly, so the median is 0.00 s for both.
  Total time is the honest metric here.

Over the 27 shared samples REPT/Kaiser σ ratios are 1.56 (qiso), 1.29 (qap),
1.48 (fσ_r), 1.61 (m) — Kaiser under-reports across the box, not just at the
fiducial.

### What changed

`build_shapefit_likelihood(theory_cls=...)` now defaults to
`REPTVelocileptorsTracerPowerSpectrumMultipoles`. There is no `--theory` flag
anywhere, by design: the generators, the regression harness and the validators
all read this one default, so the switch is a single line and cannot drift
between them.

New `default_theory_kwargs(theory_cls, tracer_bin)` supplies
`prior_basis='physical'` plus the per-tracer preset, resolved from
`tracers.yaml` rather than passed in. The preset picks (fsat, sigv), which set
the SN2 prior width — DESI's f_sat σ_v²/n̄ normalisation. desilike accepts only
BGS/LRG/ELG/QSO, so our MIX bin maps to **LRG**, since DESI's own 0.8–1.1
full-shape bin is LRG-only and that is the sample the prior was tuned against.
Unknown tracer types raise rather than defaulting; a silently wrong preset
would be an invisibly wrong prior.

Verified end-to-end: `run_fisher` on LRG2 with no theory arguments returns
0.81 / 0.85 / 0.91 / 0.87 of DESI, matching the §19 REPT scorecard.

**§21's golden baseline is now invalid** — it was dumped under the Kaiser
default. Regenerate.

---

## §23 — First REPT training set, and the mean pipeline validated against Table 11

Generated `dr1/base/{covar,mean}/v1`, all six tracers, 512 accepted samples
each (409 train / 103 test), on the §22 REPT default.

**Correction to an earlier claim in this session.** It was stated that nothing
had been generated since the July rebuild. Wrong — a `find -maxdepth 4` missed
the tree, which sits at depth 6 under `bedcosmo/num_tracers/emulator/shapefit/`.
What was already there:

| path | n | date | what |
|---|---|---|---|
| `base/covar/v3` | 20000 | 2026-03-19 | legacy toy pipeline, pre-dataset-segment layout, has `Lya_QSO` |
| `dr1/base/{covar,mean}/v99` | 204 / 409 | 2026-07-27 | rebuilt pipeline, smoke test |
| `dr1/base/{covar,mean}/v100` | 800 / 1600 | 2026-07-29 | rebuilt pipeline, smoke test |

All superseded regardless (v3 predates even the de-wiggling fix), but the claim
was wrong as stated. The auto-versioner consequently landed production at
v101, next to those two smoke tests; it was renamed to **v1** afterwards.
Nothing inside the .npz records a version, and no model had been trained
against it, so the rename is a plain directory move with no stale references.
The v99/v100 smoke tests are left in place — superseded, not deleted.

### Validation

*covar*: all six clean — finite throughout, every ρ strictly inside (−1, 1),
every σ positive, target/param ordering identical across all 12 files. Median
σ(qiso) orders by effective volume (LRG3_ELG1 0.040 tightest → BGS 0.089).

Confirmed the REPT switch reached the **data**, not just a one-off call, by
comparing LRG2 v100 (Kaiser) against v101 (REPT) medians: σ(qiso) ×1.50,
σ(fσ_r) ×1.63, σ(m) ×1.90, and ρ(f_sigmar,m) −0.21 → −0.55, the §15 sign-
strength change. σ(qap) went ×0.90 against the audit's ×1.29 — different seeds,
409 vs 800 samples, medians over a box dominated by extreme cosmologies. Worth
a matched-cosmology look, not a red flag.

*mean*: **f_sigmar reproduces DESI's Table 11 fiducial.** This is the only
predictive panel — qiso and qap are 1 by construction at the fiducial — and it
lands:

| | ours | DESI T11 | ratio |
|---|---|---|---|
| BGS | 0.47206 | 0.4723 | 0.9995 |
| LRG1 | 0.47331 | 0.4733 | 1.0000 |
| LRG2 | 0.46084 | 0.4608 | 1.0001 |
| LRG3_ELG1 | 0.43684 | 0.4398 | 0.9933 |
| ELG2 | 0.39600 | 0.3944 | 1.0041 |
| QSO | 0.39146 | 0.3750 | 1.0439 |

Four tracers to ≤0.4%. The two that miss are the two with known offsets:
LRG3_ELG1 is a different galaxy sample (LRG+ELG1 vs DESI's LRG-only) at
z_eff 0.945 vs 0.919, and QSO's z_eff is 1.343 vs DESI's 1.491 by design
(ours Fisher-weighted, theirs volume-weighted). fσ8 falls with z, so a lower
z_eff must give a higher fσ8 — both deviate in the right direction and by
about the right size.

### The m convention, settled for bedcosmo

At the fiducial input cosmology our mean pipeline returns **m = −0.5776**
(LRG2), i.e. the **absolute** slope. DESI's Eq. (4.9) m is a **deviation** —
their measured LRG2 value is +0.0467, ~0 at the fiducial. The conversion is
`dm = m − m_fid`, and m_fid is near-constant across tracers because m is a
*shape* parameter and linear shape does not evolve with z:

    BGS −0.577872  LRG1 −0.577702  LRG2 −0.577576
    LRG3_ELG1 −0.577402  ELG2 −0.577211  QSO −0.577194

That near-constancy also explains the flat median m (≈ −0.125) across tracers
in the generated data — expected, not a bug.

### Bug fixed: DR2 area in the mean eval path

`util.py:411` passed a hardcoded `14000.0` to `_fiducial_z_eff` while
`generate_mean_data.py` resolves `dataset_area(dataset)` = 7500 for DR1 — the
same footprint bug §18 fixed in `core.py`, surviving at another call site. Had
it mattered, eval would have scored the emulator against a z_eff it was never
trained on.

**It does not matter numerically: the area cancels exactly.** z_eff is a
volume-weighted average over redshift slices and the area multiplies every
slice identically, so it drops out of the normalised weight — all six tracers
agree to 5 decimal places between 7500 and 14000. Fixed anyway, since the
constant was wrong and inconsistent with the generator.

---

## §24 — REPT's k-range shifts the template's `m_fid` by 0.092 (no target affected)

Regenerating the golden baseline under REPT (§22) changed 816/1158 arrays.
Structurally correct: all 198 `mean/` arrays are **bit-identical** — the mean
pipeline uses `ShapeFitPowerSpectrumExtractor`, not `theory_cls`, so a theory
switch must not touch it — as are `obs_k`, `obs_ells` and `z_eff`. Everything
downstream of the theory moved.

One entry in that list needed explaining: `m_fid` shifted by **−0.0921,
identically on all six tracers** (−0.5775 → −0.6699), and `f_sigmar_fid` by a
uniform +0.031%.

### Cause, confirmed

`power_template.py:_set_base` computes

```python
state['m'] = np.diff(np.log(pknow_dd / pk_prim))[0] / np.diff(np.log(k))[0]
```

at `k = kp/s * [1∓0.01]`, kp = 0.03 — the log-slope of the **no-wiggle**
spectrum. `pknow_dd_interpolator` is built over whatever k grid the attached
theory requests, and `with_now='wallish2018'` fits its smooth component across
that range, so the range changes the no-wiggle spectrum and hence the slope.

A/B, LRG2 z_eff, same fiducial:

| | m_fid | template k grid |
|---|---|---|
| template standalone | −0.577531 | — |
| template + Kaiser | −0.577531 | 1.00e−3 … 1.00e0 (n=500) |
| template + REPT | **−0.669599** | **2.50e−4** … 1.01e0 (n=500) |
| `ShapeFitPowerSpectrumExtractor` | −0.577530 | — (mean pipeline) |

REPT asks for k down to 2.5e−4 rather than 1e−3 — 4× wider at the bottom, same
500 points. Three of the four agree to 1e−6; REPT is the outlier. This is the
same family as the known wallish2018 k-sensitivity, an order of magnitude
larger than the ~0.15% sawtooth already on record.

### Impact: none on any emulator target

An earlier note in this session called it a 1.3σ systematic for bedcosmo. **That
was wrong.**

- σ(m) is offset-invariant — `J = diag(1, 1, f_sigmar_fid, 1)`, the m entry is
  1, so σ(m) = σ(dm) wherever m_fid sits. Covar targets untouched.
- Absolute `m` is produced by the **mean** emulator, which uses the extractor
  and is theory-independent (hence the bit-identical `mean/` arrays).
- bedcosmo takes `m` from mean and σ(m) from covar, and never adds the two
  zero-points.

What is true is narrower: `dm` is defined relative to the attached template, so
`m = m_fid + dm` is internally self-consistent *within* the covar pipeline —
but the covar path's recorded `m_fid` now sits on a different zero-point from
the mean path's.

**Rule: do not use the covar path's `m_fid` as the absolute zero-point.** It is
theory-dependent bookkeeping. The extractor's −0.5775 (per-tracer values in
§23) is the reference, and it is the one the mean emulator is trained on.

`f_sigmar_fid` moves by the same mechanism, but there the attached template's
value is the *correct* one — `df` is defined relative to it, so using it is
required for self-consistency — and at +0.031% it is negligible either way.

---

## §25 — The other three mean targets, validated

§23 validated `f_sigmar` against DESI's Table 11 fiducial. The other three need
a different approach, because **at the fiducial cosmology qiso and qap are
exactly 1 and m is exactly m_fid, by construction** — the template's fiducial
*is* the input cosmology, so those panels test nothing there. They are only
testable off-fiducial.

### qiso, qap — against independently computed distances

Ours vs `D_V/r_d` and `D_H/D_M` built straight from cosmoprimo, both ratioed to
the fiducial, LRG2 z_eff:

| case | qiso ours/indep | qap ours/indep |
|---|---|---|
| fiducial | 1.00000 | 1.00000 |
| ω_cdm 0.09 | 0.99993 | 0.99995 |
| ω_cdm 0.16 | 1.00007 | 1.00004 |
| h 0.75 | 0.99994 | 0.99996 |
| h 0.60 | 1.00007 | 1.00004 |

**7e−5 across ±30% cosmology swings.** The AP mapping is right.

### m — against a reimplementation of the slope formula

Agreement to **6e−5** in dm and 5e−5 in the absolute value (−0.57758 vs
−0.57753), for ω_cdm, n_s and h perturbations.

Getting a *fair* reference took three attempts, and the two failures are worth
recording because both are easy to repeat:

1. **Dividing out the primordial spectrum.** `_set_base` does
   `if self.n_varied: pk_prim = ... else: pk_prim = np.ones_like(k)`. Our `dn`
   is fixed, so `n_varied=False` and m is the slope of the no-wiggle P
   **itself**, not of P/P_prim. The n_s rows exposed this exactly: ours tracks
   Δn_s (−0.04452 for n_s 0.9649→0.92, i.e. Δn_s = −0.0449) while the bad
   reference returned 0.000.
2. **Using the full linear spectrum.** kp = 0.03 sits below the first BAO peak
   at phase k·r_d ≈ 3, and perturbing ω_cdm moves r_d and hence the wiggle
   contribution to the slope. That left a spurious 0.08–0.11 gap on the Ω_m
   rows only — large enough to look like a real pipeline error. Applying
   `PowerSpectrumBAOFilter(engine='wallish2018')` to the reference collapsed it
   to 6e−5.

**Caveat on what this proves.** The reference shares the underlying wallish2018
engine — it is an independent *code path* (cosmoprimo's filter on a 1-d P(k,z)
vs desilike's template internals), so it validates the formula and the plumbing
but **cannot catch a flaw in the de-wiggling algorithm itself**. The known
~0.15% Ω_m-locked sawtooth would pass this test unnoticed.

### Status of the mean pipeline

| target | reference | agreement |
|---|---|---|
| qiso, qap | cosmoprimo distances, independent | 7e−5 |
| f_sigmar | DESI Table 11 fiducial (§23) | ≤0.4%, 4 tracers |
| m | wallish2018 slope, independent code path | 6e−5 |

### §25a — How much of the above is a real test (asked, and worth answering)

Grading the three checks by how much they could actually have failed:

**`f_sigmar` vs Table 11 — a real external test.** DESI computed 0.4608 for
LRG2 with their own pipeline and published it; nothing of ours entered that
number. It exercises the cosmology mapping, z_eff, f(z) and — the part most
likely to be silently wrong — the σ_s8 convention (cold+baryon vs total, r = 8
Mpc/h, no-wiggle vs full). A convention mismatch would show at 0.1–0.5%; we see
0.01%.

**`qiso`/`qap` — a wiring test, not a validation of the AP mapping.** It was
described that way above and that was too strong. The reference uses the same
cosmoprimo object *and* the D_V = D_H^⅓ D_M^⅔ z^⅓ formula copied from
`BAOExtractor._set_base` — the code under test. It catches inversions, wrong z,
wrong parameter mapping and unit slips, which is not nothing (exactly those
bugs — inverted qap, a spurious ×h — were caught this way earlier), but it
cannot catch a shared convention error. The external anchor for the distance
side is §20's `fiducial_dv_dhdm` vs Table 11 at ≤0.13%, not this.

**`m` vs the reimplemented slope — near-tautological.** Same engine, and the
formula was reimplemented from the source lines being checked. 6e−5 largely
means "the source was read correctly".

#### One genuinely non-circular test of m

With `n_varied=False`, m is the log-slope of P_nw ∝ k^{n_s} T²(k). Changing n_s
leaves T(k) untouched, so **dm must equal Δn_s exactly** — from the definition
of the primordial spectrum, not from any code.

| n_s | Δn_s | dm ours | dm/Δn_s |
|---|---|---|---|
| 0.8800 | −0.08490 | −0.08419 | 0.99160 |
| 0.9200 | −0.04490 | −0.04452 | 0.99160 |
| 1.0100 | +0.04510 | +0.04472 | 0.99160 |
| 1.0500 | +0.08510 | +0.08439 | 0.99160 |

Fit: `dm = 0.99160 Δn_s + 3.5e−10`. Perfectly linear, 0.84% shy of unity.

The **constancy** is the informative part: a purely multiplicative 0.84% with no
nonlinearity is the de-wiggling filter partially absorbing the primordial tilt
when it re-fits the smooth component — not noise, not plumbing. DESI's fiducial
de-wiggling is also Wallisch2018, so the bias is shared and largely cancels
against them.

This would catch a tilt applied twice, a wrong pivot, a parameter-mapping swap
or a sign error — none of which the reimplementation check could.

#### The gap that remains

n_s does not move T(k), so this says nothing about **m's Ω_m response** — and
Ω_m is the direction the emulator is mostly used in. There is currently no
non-circular check on it: any reference must reproduce the de-wiggling, which
is the k-range-sensitive component (§24) and the source of the known ~0.15%
Ω_m-locked sawtooth. Recorded as open, not solved.

---

## §26 — The 0.815 covariance ratio is mostly the shot-noise floor, not non-Gaussianity

Prompted by asking which of this session's checks are circular. The LRG2 checks
are the least circular work here — they compare against DESI **data products**
(EZmock covariance, measured spectra, randoms-derived window, measured shot
noise, published MCMC constraints), not against reimplementations of our own
code. Running the most model-independent of them turned up a problem.

### `--check shot` fails at 30%

| | n_eff | P_shot |
|---|---|---|
| ours (FKP V_eff → n_eff) | 2.7342e−04 | 3657.3 |
| DESI measured (`num_shotnoise/norm`) | 1.9122e−04 | 5229.5 |
| ratio | 1.430 | **0.699** |

### It explains the covariance deficit

If the shot-noise floor were the only error, the Gaussian covariance ratio would
be ((P0 + P_shot_ours)/(P0 + P_shot_DESI))². On our band:

| k | P0 | predicted cov ratio |
|---|---|---|
| 0.022 | 67094 | 0.957 |
| 0.063 | 30562 | 0.914 |
| 0.108 | 15521 | 0.854 |
| 0.153 | 10593 | 0.811 |
| 0.198 | 7925 | 0.775 |

**Band-median 0.852, against the measured 0.815.** So the shot-noise floor
accounts for most of the 18.5% deficit, leaving ~4% for genuine non-Gaussian
effects.

**This corrects §19**, which called 0.815 "the same modest non-Gaussian deficit
the BAO side documents (config/bundle 0.66–0.88)". That attribution is not
supported. The k-dependence is the giveaway: the predicted ratio steepens from
0.957 to 0.775 exactly as P0 falls toward the shot-noise floor, and a
non-Gaussian deficit has no reason to take that shape. (Caveat: the prediction
uses windowless P0 while 0.815 is the rotated comparison — fine for the
argument, not for a precise budget.)

### What is NOT established

That our n_eff is *wrong*. The two numbers are different definitions: ours is
the single effective density reproducing the FKP V_eff, DESI's is I₁₂/I₂₂, the
estimator's actual shot noise. With a varying n(z) these need not agree, which
is why an earlier lead here was retracted on the grounds that
`bao/config_space.py` carries the identical offset. But that establishes
common-mode, not correctness — and the quantitative link to the covariance is a
much sharper handle than existed then.

This also plausibly feeds the σ scorecard: a covariance low by ~0.85 is ~0.92
in σ, and we sit at 0.81–0.85 against DESI. Between this and Fisher-vs-MCMC the
residual is accounted for without invoking a non-Gaussian deficit at all.

**Open, and now the top item**: decide whether the V_eff-matched n_eff is the
right quantity to set the covariance shot-noise floor, or whether the floor
should be set from I₁₂/I₂₂ directly. It affects every σ in both pipelines.

### §26a — It is not the V_eff-vs-I₁₂/I₂₂ definition (hypothesis killed)

§26 proposed that the shot-noise gap came from using the V_eff-matched n_eff
where the covariance wants the estimator's own I₁₂/I₂₂, and flagged switching
them as the top open item. **Tested and false.** Computed first-principles from
our own n(z) slices (no DESI data — n̄ = N·frac/V_bin, w = 1/(1+n̄P_FKP)):

| | P_shot |
|---|---|
| ours, V_eff-matched n_eff | 3657.3 |
| ours, I₁₂/I₂₂ first-principles | **3597.7** |
| DESI measured | 5229.5 |

The two of ours agree to 2%; the choice between them is irrelevant. Insensitive
to P_FKP too (3597.7 / 3597.5 / 3590.6 at P_FKP = 1e4 / 8.9e3 / 0), because our
constructed n̄ is nearly flat across the bin (2.66–3.01e−4), so the w² weighting
has almost nothing to bite on.

**Do not spend effort switching the covariance floor to I₁₂/I₂₂.** §26's
numerical finding stands — the shot-noise offset does explain most of the 0.815
covariance ratio, so §19's non-Gaussian attribution is still wrong — but the
*cause* is a ~1.45× error in the effective density itself, not the definition.

#### Two things that turned up

**`nbar_file` is on a different area normalisation.** `nbar_ours/nbar_file` is
**0.524 in every slice** — a pure constant — and integrating `nbar_file` gives
N = 1,473,377 against Table 1's 771,875 (ratio 0.5239). This is the "computed
at the file's effective area" caveat in `_load_nz_slice_fractions`'s docstring.
Harmless as used, since the pipeline rebuilds n̄ = N·frac/V and never reads
`nbar_file` for density — but it must not be used directly, and anything that
does is wrong by 1.9×.

**Leading candidate for the residual 1.45×: weights.** DESI's `num_shotnoise`
is Σw²_tot over the real catalogue — completeness and redshift-failure weights
included, plus the α²·randoms term. Our calculation assumes ideal Poisson
sampling with no completeness weighting whatsoever. 5229.5/3597.7 = **1.454**,
the right ballpark for ⟨w²⟩/⟨w⟩² with DESI LRG completeness weights. Stated as
a plausible size, **not** a measurement — it has not been checked against the
actual weight distribution.

If it holds, the forecast is optimistic for a concrete physical reason:
weighted galaxies inflate shot noise without adding proportionate information.
Whether a survey-design forecast *should* carry that is a separate question —
it is a property of the observing strategy, and folding in a measured weight
variance would be exactly the kind of data-derived calibration this project
rejects. Computing it from an assumed completeness model would not be.

### §26b — The interface should carry N_eff, not N_success

bedcosmo already applies completeness upstream: its `N_tracers` is the number
of **successful observations**, not targets. That is the right convention, and
it matches — our `ntracers('LRG2','dr1')` = 771,894 against DESI Table 1's
N_tracer = 771,875. It also retires a concern raised earlier in this session
that a perfect-completeness forecast would bias the design optimum toward too
many tracers: it does not, because the fibres are charged for upstream.

**But the shot noise responds to a different count.** Completeness weights
restore the *target* density, so Σw ≈ N_target, while the estimator
normalisation goes as (Σw)² and the shot-noise term as Σw². Hence

    P_shot ∝ V·Σw² / (Σw)² = V / N_eff,     N_eff ≡ (Σw)² / Σw²

and by Cauchy–Schwarz **N_eff ≤ N_success**, with equality only if every weight
is identical. Neither targets nor successes: the shot noise sees the effective
number of *independent* observations, which depends on how completeness varies
across the footprint, not on its mean.

LRG2:

| | |
|---|---|
| N_success | 771,894 |
| V/N_success | 3595.5 |
| DESI measured P_shot | 5229.5 |
| implied N_eff | 530,705 |
| N_eff/N_success | 0.6875 |
| implied 1 + Var(w)/⟨w⟩² | 1.4545 |
| implied σ_w/⟨w⟩ | 0.674 |

Feeding N_eff reproduces DESI's measured shot noise exactly, and it is coherent
throughout: n_eff flows into the FKP weights and V_eff as well, and the
sample-variance term (P²/N_modes) has no n̄ dependence, so nothing
double-counts. **No pipeline change is required — this is a definitional change
to what the design variable means.**

Open question for bedcosmo: does its completeness model produce the weight
*distribution* (→ N_eff is computable) or only the mean (→ only N_success is)?
Pass-coverage inhomogeneity is what drives the dispersion.

**Caveat.** The whole 1.45× was *assumed* to be weight dispersion and then
solved for; σ_w/⟨w⟩ = 0.674 is what would be required, not what has been
observed. Part of the gap could be our shell volume, the n(z) shape, the
α·randoms term, or a convention in DESI's `norm`. Checking it means looking at
the actual DR1 LRG weight distribution — not done.

### §26c — bedcosmo's completeness model is mean-only, so N_eff is not available

Read `~/bedcosmo/src/bedcosmo/num_tracers/experiment.py` and
`data/desi/bao_dr1/desi_tracers.csv`. Completeness is two **per-tracer scalars**:

    N_tracers = targets x comp x efficiency

`comp` is mean fibre-assignment completeness (LRG 0.693, ELG 0.352, QSO 0.874,
BGS 0.636), `efficiency` the mean redshift success rate (LRG 0.991, ELG/QSO
0.727/0.668). The chain reproduces DESI Table 1's N_tracer essentially exactly:

| | bedcosmo | Table 1 |
|---|---|---|
| BGS | 300,017 | 300,017 |
| LRG1 | 506,911 | 506,905 |
| LRG2 | 771,894 | 771,875 |
| LRG3 | 859,822 | 859,824 |
| ELG2 | 1,415,707 | 1,415,687 |
| QSO | 856,652 | 856,652 |

**The interface convention is confirmed correct.** `N_tracers` is the
redshift-confirmed count and that is what this pipeline wants.

**But N_eff cannot come from this model**, and no refinement of a scalar would
help. Under uniform completeness C every weight is 1/C, so Σw = N_target,
Σw² = N_success/C², and

    N_eff = (Σw)²/Σw² = N_success   exactly.

Mean completeness cancels out of the shot-noise inflation entirely. Only the
**spatial dispersion** of completeness contributes, and nothing in the model
represents it. Carrying §26b's N_eff would require a pass-coverage /
fibre-assignment dispersion model that does not currently exist on either side.

**Numerology to check, not an explanation:** 1/comp(LRG) = 1.443 sits 0.8% from
the 1.4545 the shot noise requires. Suggestive, but working through the FKP
normalisation the mean completeness cancels (both S and A carry it), so there is
no derivation behind it. Recorded so it is not mistaken for a result.

### Interim position

The forecast assumes every redshift-confirmed object is an independent Poisson
sample. Real weighted galaxies are not, so our shot-noise floor is ~1.45× low on
LRG2, the Gaussian covariance ~0.85× low, and σ correspondingly tight — which
together with Fisher-vs-MCMC accounts for the 0.81–0.85 scorecard without
invoking a non-Gaussian deficit (§26, correcting §19).

This is a **stated limitation**, not something to patch with a constant: a
hardcoded 1.45 does not vary with the design variable, which is the one thing a
design forecast needs it to do.

## §27 — Verifying the shot-noise attribution (option 3)

Before building any dispersion model, check that dispersion is actually the
cause. Three results.

### 1. The 1/comp numerology is dead, analytically

§26c flagged that 1/comp(LRG) = 1.443 sits 0.8% from the required 1.4545.
Worked through: completeness weights restore the target density, so
Σ_gal w_comp·X → ∫ n̄_target X dV, giving

    P_shot = ∫ n̄_t w_comp w_F² dV / ∫ n̄_t² w_F² dV  →  1/(C n̄_target) = 1/n̄_obs

The mean completeness **cancels identically**. Under uniform completeness
P_shot = V/N_success exactly, whatever C is. Coincidence, confirmed by
derivation rather than by inspection. This also rules out any mean-completeness
scalar as the fix — consistent with §26c's Cauchy–Schwarz argument
(N_eff = N_success under uniform weights).

### 2. Our volume and n̄ are right — the cross-check is DESI's own V_eff

Table 1 publishes V_eff per tracer. Ours against theirs (converted to (Gpc/h)³
with h = 0.6736):

| tracer | ours | Table 1 | ratio |
|---|---|---|---|
| BGS | 0.523 | 0.520 | **1.007** |
| LRG1 | 0.918 | 0.795 | 1.155 |
| LRG2 | 1.407 | 1.223 | 1.151 |
| LRG3_ELG1 | 3.043 | 1.528 | 1.991 |
| ELG2 | 0.778 | 0.825 | **0.943** |
| QSO | 0.446 | 0.458 | **0.972** |

BGS, ELG2 and QSO land within 6%; LRG is 15% high; LRG3_ELG1's factor 2 is the
known sample mismatch (ours is LRG+ELG1, DESI's LRG-only). **Nothing is wrong
by 1.45×** — the volume/n̄ construction is sound, so the shot-noise gap is not a
volume error.

### 3. DESI's own two numbers are mutually inconsistent under a uniform n̄ — which is the evidence

For LRG2, solving for the n̄ scaling each DESI quantity implies:

| from | implied n̄ / ours |
|---|---|
| their V_eff | 0.80 |
| their P_shot | 0.6875 |

A **uniform** survey admits one n̄ satisfying both V_eff = V(n̄P/(1+n̄P))² and
P_shot = 1/n̄. These differ by 16%, and in the direction dispersion predicts:
V_eff is information-weighted and favours dense regions, P_shot is a w²-weighted
mean of 1/n̄ and favours sparse ones, so n̄(V_eff) > n̄(P_shot) necessarily.

That ordering is a **signature of spatial dispersion**, obtained from two
independent published DESI quantities with no per-galaxy weights required.

### Verdict, and what is still open

Dispersion is **supported, not proven**. Supported: the ordering above, and the
elimination of mean-completeness and volume as causes. Not proven: the size.
Something must also account for why V_eff and P_shot imply 0.80 and 0.6875
rather than a single number — and 15% of the LRG V_eff gap is unexplained on
its own.

The decisive test is the cross-tracer one: `comp` spans 0.352–0.874, so the
required inflation factor either tracks the completeness pattern or it does not.
**Blocked** — only the LRG2 full-shape bundle is local, the plain covariance
files store `num_shotnoise = 0, norm = 1` placeholders, and data.desi.lbl.gov is
still serving the NERSC-outage page (`server: GitHub.com` on every path). The
other five bundles are the unblocking item.

### §27a — Half the cross-tracer test needs no bundles: it is NOT completeness

§27 called the cross-tracer test blocked. Half of it is not: **Table 1 publishes
V_eff for all six tracers**, and only P_shot is bundle-gated. Solving for the n̄
scale `s` that reconciles our V_eff with theirs:

| tracer | s | comp |
|---|---|---|
| BGS | 0.987 | 0.636 |
| LRG1 | 0.792 | 0.693 |
| LRG2 | 0.798 | 0.693 |
| LRG3_ELG1 | 0.392 | 0.693 (sample mismatch — excluded) |
| ELG2 | **1.041** | **0.352** |
| QSO | 1.016 | 0.874 |

**`s` does not track completeness.** `comp` spans a factor 2.5; `s` spans
0.79–1.04. ELG2 decides it: worst completeness by far, yet needs no n̄
correction at all — the reverse of what a completeness-driven deficit requires.
(The correlation, −0.283 on five points, is consistent with zero and should not
be leaned on; the **range mismatch** is the argument.)

Third independent strike against completeness, after §26c's Cauchy–Schwarz
(N_eff = N_success under uniform weights) and §27's derivation (mean
completeness cancels from S and A alike).

#### What LRG2 + Table 1 establish, and what still needs the bundles

Established:
1. The survey window is a no-op for the compressed parameters (§19).
2. The analytic Gaussian covariance machinery is validated against 1000 EZmocks
   — 0.815 with the right correlation structure. Two methodologically disjoint
   calculations at 18%.
3. That 18% is the shot-noise floor, not non-Gaussianity (§26).
4. The n̄ deficit is **LRG-specific and ~20%** (all three LRG bins low; BGS,
   ELG2, QSO within 4%). A tracer-specific signal, not a global calibration.
5. It is not mean completeness — three ways.

Still open: for LRG2, V_eff implies s = 0.792 but P_shot implies 0.6875. Real
inconsistency, in the direction dispersion predicts. But **V_eff is the weak
probe** — it saturates where n̄P ≫ 1, so it is blind to the sparse tail that
sets shot noise. A null result there cannot refute dispersion, only fail to
confirm it.

The bundles now answer a sharper question than "is it completeness" (answered,
no): **why is LRG specifically ~20% off, and does the P_shot deficit track that
pattern or a different one?**

## §28 — n(z) slice audit: a harmless parser bug, and a RETRACTION of §27's dispersion evidence

### The slice files are sound where it matters

Audited all six. `slice_fraction` sums to **exactly 1.000000** for every tracer,
and slice coverage matches `tracers.yaml` zrange exactly (BGS 0.10–0.40,
LRG1 0.40–0.60, LRG2 0.60–0.80, LRG3_ELG1 0.80–1.10, ELG2 1.10–1.60,
QSO 0.80–2.10).

**One real bug, harmless.** LRG2 is the only tracer with a non-uniform
`file_area_deg2`: its first slice (0.60–0.62) carries 30092.2 against 20061.5
everywhere else — **exactly 1.5×**, i.e. the area accumulator summed 3 file-rows
instead of 2. That slice is the trimmed bin boundary (`Nbin_file` 241310.6 vs
`shape_weight` 162885.2, trim 0.674, which matches the comoving-volume fraction
of trimming a native ~0.59–0.62 bin to 0.60–0.62 — the trimming itself is
correct). It is confined to the `file_area_deg2` /
`file_effective_area_deg2` **metadata** columns.

Two claims made about this in passing were both wrong, and they cancelled:
`nbar_file` is **not** corrupted — it is `shape_weight / volume_file_trimmed`
exactly on every slice, built from counts and volumes and never from the area,
and the density sequence is smooth across the bad row (5.4535, 5.4011, 5.2802,
… 5.7359 e−4; a 1.5× error would be glaring). And the pipeline **does** read
`nbar_file` — `shapefit/core.py:386` and `bao/core.py:221` use it for the
per-slice FKP weight that sets `z_eff`, and `bao/fkp_analytic_cov.py:235` reads
it directly. So it is harmless for the opposite reason to the one first given:
the pipeline consumes `nbar_file`, and `nbar_file` is clean.

Not the cause of anything, and LRG1 shows the same anomaly-free but equally low
V_eff, so it could not have been.

### RETRACTION: the V_eff/P_shot "inconsistency" was a P₀ artifact

§27 argued that DESI's V_eff and P_shot imply different n̄ (0.80 vs 0.6875) in
the direction dispersion predicts, and called that the positive evidence for
weight dispersion. **That is wrong.**

V_eff constrains only the **product** n̄·P₀ — s is exactly ∝ 1/P₀ — while
P_shot = I₁₂/I₂₂ is P₀-independent. §27 used Table 1's "∼8.9×10³", but DESI's
standard LRG FKP weight is P₀ = 10⁴:

| assumed P₀ | s from V_eff | s from P_shot | ratio |
|---|---|---|---|
| 8900 (Table 1) | 0.798 | 0.6875 | 1.160 |
| 10000 (standard FKP) | **0.710** | 0.6875 | **1.033** |

At a consistent P₀ the two **agree to 3.3%**. There is no inconsistency, and
therefore no dispersion signal. §27's positive evidence is withdrawn; §26c's and
§27's *negative* results (mean completeness cancels; the deficit does not track
`comp`) are unaffected, as is §27a's finding that the anomaly is LRG-specific —
but §27a should read "LRG-specific in the product n̄·P₀", not "in n̄".

Sanity check that the method is sound rather than broken: the P₀ that would make
s = 1 is 9083 for BGS (Table 1: 9200), 3018 for ELG2 (2900) and 5081 for QSO
(5000) — all within a few percent. Only LRG wants 7050 against 8900.

### The corrected picture

Both DESI probes agree, once P₀ is consistent, that **our LRG2 n̄ is too
high** — by **1.454×** (P_shot = 5229.5 / (V/N) = 3595.47; direct and
P₀-independent) and **1.408×** (V_eff at P₀ = 10⁴; P₀-dependent). They agree to
3%. *(An earlier draft quoted "~1.42×", which is neither figure — it was a gloss
splitting the two and should not be cited as a result.)* A plain normalisation
error, not a weighting subtlety — which is a
better problem to have, being checkable rather than requiring a fibre-assignment
model.

N is not the culprit: 771,894 against DESI Table 1's 771,875. So the excess is
in **V** — the shell volume or the 7500 deg² area. Noted for the chase:
`file_effective_area_deg2` in the LRG slice files is **19,100 deg²** against the
7,500 the pipeline uses. That is a factor 2.5, not 1.42, so it is not a direct
explanation, but the area bookkeeping clearly needs understanding before this
closes.

## §29 — The same n̄ excess is in the BAO pipeline, and it is larger there

`bao/config_space.py` builds n̄ identically — `config_space.py:656` says so
outright ("n̄ is exactly linear in N_design … nbar = N·frac/V_shell at the fixed
frame") — and `XiSigmaGenerator('LRG2').slices.nbar` returns exactly the
shapefit values (2.8571e−4, 2.8297e−4, 2.7663e−4 …) at the same N_fid = 771894.
So the §26/§28 shot-noise excess is **not a shapefit bug**; it is shared.

Because n̄ is linear in N, feeding `N/1.4545` *is* the corrected n̄. Effect on
the BAO σ triplet:

| tracer | σ(D_H) | σ(D_M) | σ(D_V) |
|---|---|---|---|
| BGS | ×1.112 | ×1.112 | ×1.112 |
| LRG1 | ×1.089 | ×1.107 | ×1.101 |
| LRG2 | ×1.096 | ×1.113 | ×1.108 |
| LRG3_ELG1 | ×1.141 | ×1.169 | ×1.162 |
| ELG2 | ×1.241 | ×1.241 | ×1.241 |
| QSO | ×1.264 | ×1.264 | ×1.264 |

Sparse tracers are most sensitive, as expected for a shot-noise-dominated
regime.

### Consequences for two recorded BAO conclusions

- Production under-predicts DESI by a **uniform ~23%** (P/D ≈ 0.72–0.80).
  Applying these factors: LRG 0.76 → ~0.84, ELG2/QSO 0.76 → ~0.95. Roughly half
  the gap on LRG, nearly all of it on the sparse tracers.
- The ξ-covariance deficit (config/bundle 0.66–0.88) is attributed to physical
  non-Gaussian / α_SN effects. As in §26 on the shapefit side, that attribution
  would then be substantially wrong.

### Caveats — do not act on this yet

1. **1.4545 is measured on LRG2 alone.** It is the only tracer with a bundle
   carrying `num_shotnoise`/`norm`. Extending it to all six is an assumption,
   and the sparse tracers — where the effect is largest — are the least
   constrained. This is precisely what the five missing bundles would settle.
2. It would **destroy the uniformity** of the 23% under-prediction, currently
   one of the more striking features of that comparison. Either a clue that the
   real factor is tracer-dependent, or a warning against the assumption.
3. `bao/` is regression-frozen and was **not modified** — this is a read-only
   measurement through the public `XiSigmaGenerator` API.

### §29a — RETRACTION: "correcting n̄ by 1.4545" is not a correction, and the test was malformed

§29 framed `sigma_triplet(N_tracers=N/1.4545)` as *correcting* n̄ and concluded
the BAO 23% gap is roughly half explained. **Both the framing and the test are
wrong.**

**1. It is a fudge factor.** 1.4545 is an unexplained *discrepancy*, not a known
correction — §26a ruled out the V_eff/I₁₂/I₂₂ definition, §26c/§27 ruled out
completeness, §28 ruled out the slice bug, and the cause remains open (three
mutually inconsistent area normalisations, §28). Applying it would be
data-derived, constant in `N_tracers`, and would fit our pipeline to DESI's
number — destroying the independence that makes the comparison meaningful. It is
precisely what this project rejects, and what §26b/§26c argued against twice
before §29 went and did it anyway. Since `N_tracers` is the **design variable**,
rescaling n̄ by it means "asked for N, return σ for N/1.4545", which corrupts
the design mapping itself.

**2. The test does not correspond to the plausible physical fix.** n̄ = N·frac/V.
The leading candidate cause is the volume/area, so a real fix changes n̄ **and**
V_survey together: larger V lowers sample variance (more modes) while lower n̄
raises shot noise, and the two partially cancel. Scaling N alone captures only
the σ-inflating half. The ×1.09–1.26 factors therefore **overstate the effect by
an unknown amount**, and the "closes half the gap on LRG, nearly all on
ELG2/QSO" conclusion is withdrawn.

**What survives from §29:**
- `bao/config_space.py` builds n̄ identically to shapefit — verified, and the
  numbers match exactly. The excess is shared, not a shapefit bug. *(fact)*
- BAO σ are sensitive to n̄ at the ~10–26% level, sparse tracers most.
  *(a sensitivity, not a correction — and an upper bound, per point 2)*

**What does not survive:** any statement about how much of the documented BAO
23% under-prediction, or the 0.66–0.88 ξ-cov deficit, this accounts for. That
needs the *cause* identified, not a ratio applied.

## §30 — Two §22 fallout bugs: changing a default broke callers of the old one

Both found while regenerating the comparison plots, both the same failure mode —
§22 changed a shared name and a default without checking who depended on the old
behaviour.

**1. Duplicate `_REPT_TRACER_PRESET` (loud).** §22 added a second dict of that
name keyed by tracer TYPE, shadowing a pre-existing one keyed by tracer BIN.
`compare_to_desi.py` indexed it directly with bin names → `KeyError: 'LRG1'` for
every tracer but LRG2. Fixed by deleting the stale bin-keyed duplicate and
routing the call site through `default_theory_kwargs`, so there is one source of
truth. **Production unaffected** — the generators always went through
`default_theory_kwargs`, so `v1` training data and the REPT golden baseline are
correct.

**2. `theory="kaiser"` silently meant REPT (SILENT).** `our_forecast` set
`kw = {}` for Kaiser and let `build_shapefit_likelihood` supply the default —
which §22 flipped to REPT. So **both series in every comparison plot were REPT**,
which is exactly what they looked like: two identical curves. Caught only
because the plots were eyeballed and the overlap looked wrong.

Fixed by passing `theory_cls` explicitly for **both** theories. Verified: LRG2
σ(qiso) = 0.00787 (Kaiser) vs 0.01359 (REPT), a factor 1.73, consistent with
§15's documented 1.6–2.1× Kaiser under-reporting.

### Sweep of the other implicit call sites

19 call sites build a likelihood without an explicit `theory_cls`. All the rest
are *intended* to follow the production default (generators, `run_fisher`, the
regress harness, `validate_forecast`, `util.get_pipeline`) — that is the design,
and it is why §22 was a one-line switch. `compare_to_desi` was the only place
that needed a *specific* theory and relied on the default to get it.

### Left as a cleanup item

Stale text, not stale behaviour: `README.md:127–130` ("Fiducial anchor
(2026-07-27, Kaiser…)", "un-windowed Kaiser Gaussian-cov Fisher is expected to
sit on the tight side") and `validate_forecast.py:91` still describe the
forecast as Kaiser in printed output. They compute the right thing and say the
wrong one.

### Lesson

A loud failure (`KeyError`) announced itself; a silent one produced a plausible
plot and survived until a human looked at it. When flipping a default, grep the
callers — the ones that break are less dangerous than the ones that don't.

## §31 — ELG1 must be EXCLUDED from the full-shape 0.8–1.1 bin (defect, not caveat)

DESI 2024 V §2 is explicit:

> for the Full-Shape type of analyses presented in this paper, the ELG bin
> between 0.8 < z < 1.1 (ELG1) **was not included as it failed to pass the
> required tests before unblinding** … ELG1 showed uncorrected systematic
> effects related to fibre collisions … **However, the impact … on the BAO
> measurements was negligible, and for this reason, this bin was included in the
> BAO analysis** … in combination with the high-z LRG bin.

**ELG1 is IN for BAO and OUT for full-shape.** The bin definition is
analysis-dependent, and `tracers.yaml` has only one entry.

### What the pipeline does

`ntracers('LRG3_ELG1','dr1')` = **1,876,187** = LRG3 (859,822) + ELG1
(1,016,365), against DESI full-shape's 859,824 — a factor **2.18**. The n(z)
slices combine both too (`source_file` lists all four ELG and LRG `nz.txt`).

### It fully explains the LRG3_ELG1 scorecard outlier

| N used | σ(qiso)/DESI | σ(qap)/DESI | σ(m)/DESI |
|---|---|---|---|
| combined 1,876,187 | 0.69 | 0.66 | 0.95 |
| **LRG3-only 859,822** | **0.86** | **0.80** | **0.98** |

√(2.18) = 1.477 predicts a naive σ ratio of 0.677 — against the observed
0.69/0.66. Essentially exact. Corrected, the bin sits with LRG1 (0.99/1.01) and
LRG2 (0.81/0.85) instead of being an outlier.

**Knock-on for §16:** "agreement degrades with redshift" rested partly on
LRG3_ELG1 being anomalous. It is not. That leaves ELG2 (0.65/0.58) as the
genuine high-z problem, with QSO middling — a weaker and less monotonic trend
than recorded.

### This was flagged and then reasoned around

`desi_reference.SAMPLE_MISMATCH` and `compare_to_desi._SAMPLE_MISMATCH` both
carried the warning, and §27a excluded the bin from the V_eff fit on those
grounds. Flagging it was not enough: the forecast is still generated, and the
mismatch is a factor 2.18 in N, not a footnote.

### Consequences

1. **`v1` LRG3_ELG1 training data is wrong for full-shape.** It forecasts a
   combined sample DESI cannot measure. If bedcosmo consumes it, the design
   credits ELG1 with growth-rate constraining power DESI explicitly found it
   does not have.
2. **The fix cannot be a `tracers.yaml` edit.** That file is shared with `bao/`,
   which is frozen and where the combined bin is *correct*. The bin definition
   has to become analysis-aware.
3. **A full fix needs CFS.** LRG-only n(z) slices for 0.8–1.1 require re-running
   `parse_desi_nz.py` with `DEFAULT_TARGETS_BY_BIN['LRG3_ELG1'] = ['LRG']`
   (one line) against the raw `*_nz.txt`. Blocked with everything else.

Not fixed here — the design decision (how to express an analysis-dependent bin
without touching frozen `bao/`) is the user's.

## §32 — tracers.yaml is analysis- and release-scoped; LRG3 replaces LRG3_ELG1 for full shape

Implements §31. The raw `*_nz.txt` turned out to be **local** after all
(`~/data/desi/nz_data/`), so the LRG-only slices were never blocked on CFS.

### Schema

`tracers.yaml` gains four optional keys, all documented in its header:

- `analyses` — which analyses a bin belongs to; absent means all
  (back-compatible). `LRG3_ELG1` is now `[bao]`, the new `LRG3` is `[shapefit]`,
  `Lya_QSO` is `[bao]`.
- `components` — sub-samples in `desi_tracers.csv`. `util.ntracers` sums
  `targets × comp × efficiency` over them, because `desi_data.csv` has only the
  BAO combinations (no LRG3 row, only LRG3+ELG1).
- `nz_slices` — slice-file basename when it differs from the bin key.
- `overrides` — per-analysis / per-release deltas, merged as `<analysis>` then
  `<analysis>/<dataset>`. **Wired but unpopulated**: the mechanism exists so a
  DR2 sample change drops in without another refactor, per the DR1-first rule.

`util.get_tracer_config(bin, analysis=None, dataset=None)` and
`util.ntracers(bin, dataset, analysis=None)` take the new arguments; plus
`util.tracers_for(analysis)`.

### Backward compatibility (bao/ is frozen)

With `analysis` omitted the accessors return exactly what they did before.
Verified: all six BAO tracers return identical N, tracer_type, zrange and
f_interloper. Both drivers list tracers explicitly, so the new key cannot leak
into a BAO run, and requesting `get_tracer_config('LRG3','bao')` raises. **No
file under `bao/` was modified** — including `parse_desi_nz.py`, which the new
`shapefit/make_lrg3_nz_slices.py` drives by injecting the bin at runtime rather
than forking the slicing code.

`LRG3_nz_slices.csv`: 15 slices, `slice_fraction` sums to 1.000000, coverage
0.800–1.100, uniform `file_area_deg2` (no §28 boundary anomaly).

### RESULT — and a correction to §31

§31 claimed that correcting the bin would move it to ~0.86/0.80, "in line with
LRG1 and LRG2". **That was from the partial fix** (LRG3-only N while keeping the
MIX HOD and the combined n(z)). The full fix gives:

| | b1p | n_eff | z_eff | σ(qiso)/DESI |
|---|---|---|---|---|
| old LRG3_ELG1 bin | 0.906 | 1.50e−4 | 0.9453 | 0.69 |
| **new LRG3 bin** | **1.344** | 1.44e−4 | **0.9122** | **0.68** |

The driver is **b1, not N or n(z)**: the HOD switch MIX→LRG raises the bias 48%,
and more signal against the same shot noise tightens σ. Both are self-consistent
— b1p/σ8(0.92) gives b_E ≈ 1.77 for the MIX (matching a number-weighted
LRG+ELG1 blend) and ≈ 2.63 for LRG-only. z_eff also moves to 0.9122 against
DESI's 0.919, much closer than the old 0.9453.

Full LRG3 scorecard: **0.68 / 0.66 / 0.73 / 0.73**.

**So §31's knock-on was also wrong.** It said the redshift trend was weaker than
recorded. With the properly-corrected bin the trend is *restored*:

| | LRG1 | LRG2 | LRG3 | ELG2 | QSO |
|---|---|---|---|---|---|
| σ(qiso)/DESI | 0.99 | 0.81 | 0.68 | 0.65 | 0.79 |

§16's "agreement degrades with redshift" stands, with QSO the exception.

### Follow-ups — both done

- **Training data.** Regenerated `LRG3` for both quantities into `v1` (512
  accepted each, 409/103 split) and deleted the four orphaned
  `LRG3_ELG1_{train,test}.npz`. Only that bin was touched — the driver takes
  `--tracers`, so the other 20 files are the originals. Validated: finite, σ>0,
  ρ inside (−1,1), correct target counts. The N_tracers box confirms the sample
  really changed — [4.31e5, 1.29e6] against the old [9.41e5, 2.81e6], i.e.
  0.5–1.5 × 859,822 rather than × 1,876,187.
- **Golden baseline.** Regenerated; its grid now enumerates `LRG3`. 1158 arrays,
  and **two independent dumps compare bit-identical**, same as the previous two
  baselines. (It was briefly installed *before* that check was run — the check
  is the harness's only purpose, so skipping it would have left every later
  dataset's reproducibility assumed rather than established.)
- **Comparison plots** regenerated on the corrected pipeline.

## §33 — RETRACTION: there is no redshift trend, and never was

§16 recorded "agreement degrades with redshift", and §31/§32 argued about
whether the ELG1 fix weakened or sharpened it. **The trend does not exist.**

All six tracers, current pipeline (REPT, LRG3, corrected fiducials):

| tracer | z_eff | n̄P₀ | σ(qiso)/DESI |
|---|---|---|---|
| **BGS** | **0.292** | 2.92 | **0.61** |
| LRG1 | 0.508 | 2.52 | 0.99 |
| LRG2 | 0.704 | 2.48 | 0.81 |
| LRG3 | 0.912 | 1.26 | 0.68 |
| ELG2 | 1.303 | 0.33 | 0.65 |
| QSO | 1.343 | 0.13 | 0.80 |

    corr(ratio, z_eff)   = -0.128
    corr(ratio, nbar*P0) = +0.199

Both consistent with zero. **BGS is a direct counterexample** — the worst
agreement of all six, at the lowest redshift.

### How it survived

§32's table listed LRG1 / LRG2 / LRG3 / ELG2 / QSO — five tracers, omitting the
one that breaks the pattern. Read that way the numbers do fall monotonically
(0.99 → 0.81 → 0.68 → 0.65) and it looks convincing. It is a cherry-pick.

And it was never supported: §16's own scorecard had BGS at 0.62, already the
lowest. The counterexample was in the data from the first time the claim was
made, and it was then repeated through §31 and §32 — including an argument about
whether fixing ELG1 *strengthened* a trend that was not there.

### What is actually true

The ratios scatter 0.61–0.99 with no monotonic dependence on redshift or on
density. Whatever drives the spread is tracer-specific. LRG1 (0.99) and BGS
(0.61) are the extremes and sit adjacent in redshift with similar n̄P₀, so no
smooth function of either variable can separate them.

This does not touch §26–§28: the ~1.45× shot-noise discrepancy is measured on
LRG2 against a DESI data product and stands on its own. What is withdrawn is the
claim that the *residual scatter across tracers* has a redshift explanation.

### Note to self

Four asserted mechanisms failed checks today (V_eff-vs-I₁₂/I₂₂, 1/comp
numerology, weight dispersion, this). Each time the arithmetic was right and the
pattern was imposed. A five-of-six table is a warning sign, not a result.

## §34 — BAO vs shapefit: shared level, unshared scatter (shelved)

Prompted by the observation that the shapefit σ ratios look like the BAO ones.
Recorded and shelved, not pursued.

**Supported — the level is shared.** Mean P/D (our σ ÷ DESI's published σ):

| | mean P/D | range |
|---|---|---|
| BAO σ(D_H/r_s) | 0.803 | 0.722–1.001 |
| BAO σ(D_M/r_s) | 0.756 | 0.678–0.788 |
| BAO σ(D_V/r_s) | 0.781 | 0.708–0.884 |
| shapefit σ(qiso) | 0.757 | 0.610–0.990 |

Two analyses with different estimators, data products and theory landing within
~3% on how much they under-predict DESI. They share n̄, V, the n(z) slices and
the HOD, and little else — so this points upstream, and it is roughly what a
1.45× n̄ error predicts (§26–§28). That lead now has to explain two pipelines.

**Not supported — the per-tracer pattern.** corr(BAO P/D, shapefit qiso P/D) is
+0.727 / +0.063 / +0.562 depending purely on whether DH, DM or DV is paired
against qiso. On n=6 that is noise; the +0.562 quoted earlier carries no weight.
BGS and LRG2 actively disagree between pipelines.

**Caveat that matters for any retry: BGS and QSO are isotropic-only** in DR1
BAO. `_triplet_from_recon_cov` fills a 1×1 covariance by scaling DH/DM/DV
identically ("σ scales DH/DM/DV identically"), which is why both read 0.779
three times. So the DH and DM columns are effectively n=4, not 6 — do not
average or correlate over them without dropping BGS and QSO.

DV is the only column where all six are genuine, and it is also the correct
pairing for qiso (both are the isotropic dilation).

## §35 — The mean emulator emits DESI's `m` (the deviation), not the absolute slope

`MEAN_TARGET_NAMES` keeps the name `m`, but the value is now DESI's Eq. (4.9)
shape parameter:

    P'_lin(k) = P^fid_lin(k) exp[ (m/a) tanh(a ln(k/kp)) + n ln(k/kp) ]

which multiplies the **fiducial** template, so m = 0 is no shape change. The
generator returns m = −4.5e−05 at the DESI fiducial cosmology, as it must.

### Why

- **It is what DESI publishes.** Their Appendix A central values are in this
  convention, so the mean comparison needs no offset. (Their "m + n" label is
  an interpretive relabel of a single varied parameter — §4.9, *"we will vary m
  keeping n fixed. Later, in the interpretation step m can be seen as if it
  were m + n"* — and our `dn` is fixed at 0, verified.)
- **It removes §24 from the interface.** The theory-dependent `m_fid` (REPT's
  attached template reports −0.6699 where the extractor says −0.5775) only
  mattered because something downstream had to subtract a fiducial. Nothing
  does now.
- **It is what desilike varies.** `dm` is the free template parameter.
- **Zero cost to covar.** σ is offset-invariant: σ(m) = σ(dm) exactly.

### The naming trap

An intermediate version renamed the target to `dm`. **That was wrong** —
"dm" appears nowhere in DESI 2024 V, and it collided with our own `sigma_m` /
`rho_*_m`. The confusion is that desilike and DESI use the symbol `m` for
different things:

| | `m` means | value at fiducial |
|---|---|---|
| DESI 2024 V | Eq. (4.9) shape parameter (a deviation) | 0 |
| desilike | absolute log-slope of P_nw at k_p | −0.5775 |

DESI's m **==** desilike's `dm`. So the mean worker reads `extractor.dm` into a
target named `m`, which is documented at both ends rather than left as a trap.

Mean training data regenerated for all six tracers into `dr1/base/mean/v1`.
Note the box now spans m ∈ [−10.4, +3.7] (median +0.45) because the Ω_m prior
reaches shapes far from the fiducial — DESI's measured values are ~0.05, so the
emulator is trained well outside where any real measurement lands.

---

## §36 — The matrix plot's bottom row is a percentage of DESI, not a difference

`plot_covar_matrix`'s third row was `generator − DESI` in raw units. That row is
hard to read for two reasons: the covariance entries span three orders of
magnitude across tracers, so a shared color scale is set by whichever tracer has
the largest σ and everything else washes out; and an absolute ΔC of 3e−3 means
something completely different on the `fσ_r` diagonal (~1e−2) than on a
`q_iso`–`q_ap` off-diagonal (~2e−4).

Now:

    P_ij = 100 * (ours_ij − DESI_ij) / |DESI_ij|

`|DESI|` rather than `DESI` in the denominator, so the sign always reads
"generator above/below DESI" instead of flipping with the sign of the reference
entry. Linear norm ±100% (cov) / ±60% (corr), PuOr, `extend="both"`.

### Where the percentage is undefined

An off-diagonal entry of *either* matrix is proportional to ρ_DESI, so one cut —
`|ρ_DESI| < 0.05` — masks the ill-conditioned cells for both kinds. Those cells
render as a grey box with `--`; the empty lower triangle stays white, which is
a distinction the first attempt lost by setting the colormap's bad color to
grey (masked-outside-triangle and masked-undefined both route through
`set_bad`). Fixed by drawing the grey explicitly as a patch and keeping
`set_bad("white")`.

Only five cells are actually masked in DR1 — `q_iso`–`fσ_r` for LRG1/LRG2/LRG3,
`q_iso`–`q_ap` for QSO, and `fσ_r`–`m` for BGS — where DESI's ρ is 0.01–0.03.

### What it shows

The residual structure is not uniform, which the absolute version hid:

- **Diagonal** (cov): −63% to −3% on σ²(q_iso), i.e. the known σ deficit.
- **`fσ_r`–`m`**: −70% to −103% on all five tracers where it is defined (BGS is
  masked). Our correlation is ≈ 0 (−0.01 to +0.07) where DESI measures +0.23 to
  +0.37. This is the single most consistent
  disagreement in the matrix and it is a *shape* disagreement, not a level one,
  so the shot-noise discrepancy (§26) does not explain it.
- **`q_ap`–`m`**: +36% to +90%, same sign everywhere.
- **`q_ap`–`fσ_r`**: −9% to −30%; ours is the stronger anticorrelation
  (−0.65..−0.73 vs DESI's −0.53..−0.63).

The `q_iso` row percentages on BGS/ELG2/QSO are large (+282%, −148%, +175%) but
that is a sign flip on a small ρ, not a large absolute miss — read those off the
top two rows, not the percentage.

Diagonal cells of the **correlation** percentage are +0% by construction
(1 vs 1); they carry no information and are left in only so the triangle reads
as a matrix.

---

## §37 — The mean plot's AP panels compare against the FIDUCIAL, not the data

`q_iso` and `q_ap` are 1 on our side by construction at the fiducial cosmology
(§35, and the module docstring), so the old panels drew a flat blue line at 1
with DESI's measurement scattered around it. That plots the *ratio* while hiding
both numbers that go into it.

An intermediate version plotted `D_V/r_d` and `D_H/D_M` in absolute units with
three series — DESI measured, Table 11 fiducial, ours. **DESI's measurement does
not belong on this axis.** It is data; whether the universe matches the fiducial
is a statement about DESI, and drawing it beside our prediction invites reading
a cosmological result as a code error. The AP panels now show the generator
against DESI's published fiducial and nothing else. (`fσ_r` and `m` still carry
the DR1 measurement — there the comparison is meaningful, see the docstring.)

The y-axis is therefore the residual `generator / Table 11 − 1` in percent, with
the Table 11 value annotated along the top so the physical number is still
visible (it spans 3–7× across tracers, which is why it cannot be the axis).
Two generator markers separate the two things that can move us off the table:

| marker | evaluated at | isolates |
|---|---|---|
| open circle | DESI's z_eff | distance / r_d convention only |
| filled circle | our own z_eff | that, plus our Fisher-weighted z_eff |

### What the markers do and do not test

Our distances come from the same cosmoprimo `"DESI"` cosmology the pipeline
uses, so this is NOT an independent check of the distance calculation — same
code path. The open markers land on 0 (≤0.13%), which confirms the convention
(D_H, D_M, D_V, r_d definitions) rather than the arithmetic.

The filled-vs-open gap is entirely our **z_eff**, which is Fisher-weighted where
DESI's is volume-weighted:

| | ours | DESI | Δ |
|---|---|---|---|
| BGS | 0.29 | 0.295 | ~0 |
| LRG1 | 0.51 | 0.510 | ~0 |
| LRG2 | 0.70 | 0.706 | −0.9% |
| LRG3 | 0.91 | 0.919 | −1.0% |
| ELG2 | 1.30 | 1.317 | −1.3% |
| QSO | 1.34 | 1.491 | **−10%** |

QSO is the outlier, and the AP panels put a number on what that costs:
**−5.2% in D_V/r_d and +15.9% in D_H/D_M**. `D_H/D_M` falls steeply with z, so a
10% shift in z_eff is amplified. Every other tracer is under 1.6% in both.

This is a real difference in where the bin's constraining power sits — the QSO
n(z) is broad and flat, so Fisher weighting pulls low where n̄P is best while
volume weighting pushes high — not a plotting artifact. It means any QSO
comparison against DESI is being made at a different effective redshift, which
is worth remembering when reading QSO in the σ and covariance plots. Flagged
here; not chased.

## §38 — The AP panels never tested the pipeline; add a check that does

Follow-on from §37, prompted by a simple question: given fiducial cosmology
inputs, does the mean pipeline return exactly 1?

### What the comparison plot actually validates

`plot_mean`'s two AP panels draw

```python
at_z_desi = desi_ref.fiducial_dv_dhdm(z_desi)[idx]
at_z_ours = desi_ref.fiducial_dv_dhdm(z_ours)[idx]
```

Both markers are cosmoprimo's DESI *fiducial* evaluated at a redshift, compared
against Table 11 — which is also a fiducial. The mean pipeline's own output
never enters. So the panels test that our cosmoprimo fiducial reproduces DESI's
published fiducial (<=0.13%), plus the z_eff offset on top. That is a
reference-data check, not a pipeline check, and the panel title oversells it.

Nor can the panels be rewired to do better, because of how q is defined
(desilike `power_template.py`, `BAOExtractor.get`):

    qiso = DV_over_rd / DV_over_rd_fid
    qap  = DH_over_DM / DH_over_DM_fid          # DH over DM, not DM over DH

Numerator and denominator share `self.z`, so at fiducial input q == 1 for ANY
z_eff. q is dimensionless and normalised to the fiducial: the absolute
DV/rd is divided out by construction. Recovering it means multiplying by a
fiducial you fetched from cosmoprimo or Table 11 — circular. z_eff can
therefore never appear in q itself, only in the fiducial multiplied back in.

### But q == 1 is a real, falsifiable assertion

It needs no DESI reference data at all: it says the extractor's varying
cosmology and its fixed fiducial agree when handed the same cosmology. Nothing
in the suite was asserting it, and it is the ONLY test that exercises the mean
path's cosmology mapping. That matters because the two pipelines map cosmology
differently — `_to_mean_extractor_params` assembles `Omega_m` from the omega
basis (including `omega_ncdm`), while the covar path passes `omega_cdm`
straight through. If they diverge, the mean and covar emulators are trained on
different cosmologies and every other check still passes.

### Added: `validate_forecast.py --check mean-ap`

(a) fiducial input, q must be 1; (b) perturbed input, q must equal the same
ratio built directly from cosmoprimo — the AP machinery in the regime the
emulator is actually used in, which the plot cannot reach.

All 18 rows pass. Deviation 4.6e-8 (BGS) to 1.3e-7 (QSO), rising with z.

**What that floor is** (corrected — the first version of this entry blamed the
`omega_cdm -> Omega_m -> omega_cdm` round trip, which is wrong; that mapping is
exact). Interrogating the extractor at QSO:

    parameters   h, n_s, N_ncdm, m_ncdm      rel = 0.000e+00  (bit-identical)
    derived      rs_drag                     +1.05e-08
                 comoving_angular_distance   -9.13e-08
                 efunc                       +1.63e-07
    engine       cosmoprimo.classy.ClassEngine on BOTH sides

The parameters agree exactly, so the mapping is not the source. The floor is
two separately-initialised CLASS instances — `self.fiducial` built once at
`initialize`, `self.cosmo` run per call — requesting different outputs and
therefore interpolating the background off different grids. `efunc` (read off
the background table) is worst; `rs_drag` (a scalar thermo output) is best.
That also explains the rise with z: the grid is coarser there. A parameter
offset would not behave that way.

Corollary: there is no meaningful difference in HOW numerator and denominator
are computed. `BAOExtractor._set_base` is one function called twice, and line
316 (`cosmo = self.fiducial if fiducial else self.cosmo`) is the only branch.
Identical arithmetic, identical engine. The check is therefore an assertion
about the cosmology handed in, and its noise floor is CLASS's interpolation
reproducibility.

Tolerances 1e-5 on both. Placed by negative test, not by taste: injecting the
bug the check exists for (drop `omega_ncdm`, the legacy
`util.to_extractor_params` behaviour) gives 7.1e-4 and fails every row. So the
threshold sits ~70x above the numerical floor and ~70x below the target bug.

### Also: `fiducial="DESI"` is now explicit in `_get_mean_extractor`

It was relying on desilike's default. The default is correct today, but it is
the denominator of every mean label, and this is exactly the `with_now`
footgun again — where the `'peakaverage'` default was silently wrong and
mislabelled sigma ~2x. Made explicit; behaviour unchanged.

### Not changed

The AP panels themselves. They are still worth plotting — reference-data
agreement and the z_eff offset are both things we want to see — but they are
not evidence about the pipeline, and §37's framing of them should be read with
that in mind.

## §39 — Extend the fiducial check to the shape targets: q is blind to n_s and ln10A_s

§38's check asserts qiso == qap == 1 at fiducial input. Scoping that honestly:
it pins one claim, *fiducial in -> fiducial out*, and catches three ways to
break it (FID_SAMPLE drifting from cosmoprimo's DESI, a wrong
`_to_mean_extractor_params`, a changed `fiducial=` on the extractor).

But q is built from distances alone. **It cannot see `n_s` or `ln10A_s` at
all** — neither enters a background integral. So a bug mapping either through
the omega basis passed §38 clean, and two of the four mean targets had no
fiducial assertion on them.

`m` and `f_sigmar` have the identical self-consistent form: the extractor
stores `m_fid` and `f_sigmar_fid` at init, exactly parallel to
`DV_over_rd_fid`, so at fiducial input `dm` must be 0 and
`f_sigmar / f_sigmar_fid` must be 1.

### Floors, and why they are looser

    tracer      dm        f_sigmar/fid - 1
    BGS      -7.83e-05        -6.47e-05
    LRG1     -4.87e-05        -4.01e-05
    LRG2     -4.46e-05        -3.67e-05
    LRG3     -1.83e-05        -1.48e-05
    ELG2     -1.86e-05        -1.51e-05
    QSO      -2.34e-05        -1.91e-05

~1e-5 to 1e-4, three orders looser than q's 1e-7, worst at low z and NOT
monotonic — so this is the de-wiggled P(k) and the numerical log-slope at kp,
a different numerical path from §38's background-grid floor. Physically
irrelevant: DESI's sigma_m is 0.03-0.09, so 8e-5 is ~0.1% of an error bar.

`dn` is exactly 0 on every tracer (n_s passes straight through).

### Sensitivity, measured on LRG3

    ln10A_s +0.1%   dm unchanged        f_sigmar/fid - 1  +1.5e-03
    n_s     +0.1%   dm +9.4e-04         f_sigmar/fid - 1  +1.1e-03

`f_sigmar` is the ONLY one of the four that moves with `ln10A_s` — amplitude
does not change a slope, so `dm` is flat against it. Without the f_sigmar row
the ln10A_s mapping has no test anywhere in the suite.

Tolerance 5e-4: ~6x above the floor, and a 0.1% parameter error lands 2-3x
above it, so the check bites at roughly the 0.05% level. Looser than q's 1e-5
because the floor is, not because the requirement is weaker.

24/24 rows pass.

## §40 — Pin the fiducial in ABSOLUTE terms; ratios cannot see it move

Closes the gap §38/§39 left open, and ends the fiducial-check thread.

Every assertion in `check_mean_ap` is a ratio of the varying cosmology to the
fiducial. Anything that moves BOTH sides together cancels exactly and passes:

  - cosmoprimo repointing the `DESI` alias. It is literally an alias --
    `DESI = AbacusSummitBase` (fiducial.py:264) -> `AbacusSummit(name='000')`
    -- and the same module already ships `DESIDR2Flatw0waCDM` with materially
    different values (Omega_m 0.3192, w0 -0.754, wa -0.857);
  - a different Boltzmann engine;
  - a different precision setting. `AbacusSummitBase` takes `precision=None`
    and documents `precision='base'` as materially different.

Any of those silently redefines every mean training label, because the labels
ARE ratios to this object. This is not hypothetical: cosmoprimo was upgraded
in place (1b100803), and we have already been burned once by trusting an
upstream default, when `with_now='peakaverage'` mislabelled sigma ~2x.

### Added: `validate_forecast.py --check fiducial-id`

Recorded 2026-08-02 against cosmoprimo 1b100803:

    omega_cdm 0.1200000000    n_s       0.9649000000   rs_drag 99.0844267934
    omega_b   0.0223700000    m_ncdm    0.0599999193   sigma8   0.8076353990
    h         0.6736000000    N_ncdm    1.0            engine  classy.ClassEngine
    ln10A_s   3.0363942553    N_eff     3.0459982215
    tau_reio  0.0544000000    w0/wa     -1.0 / 0.0

Inputs at 1e-8 (definitional), derived scalars at 1e-6 (CLASS jitter, but
tight enough that a precision or engine change shows). 14/14 pass at 1e-11 or
better.

### Known, not fixed: FID_SAMPLE carries a rounded ln10A_s

`validate_forecast.FID_SAMPLE` has `ln10A_s = 3.036394`; cosmoprimo's DESI is
`3.0363942553`. 8.4e-8 relative — two orders below the §39 shape floor, so
nothing measurable, and it is NOT the source of any floor reported there.

Deliberately not corrected: FID_SAMPLE feeds the bit-exact
`np.array_equal` goldens, so a 1e-8 shift would break every one of them for no
physical gain. Align it at the next golden regeneration, or leave it.

### Where the fiducial actually comes from (the question that started this)

  - mean extractor: `fiducial="DESI"` — a bare string, fully built-in, we pass
    NOTHING. This is why q != 1 off-fiducial and the labels carry AP signal.
  - covar template: `fiducial=("DESI", dict(theta_cosmo))` — the tuple form,
    "(name, dict of parameters to update)". The built-in overridden by OUR
    sample, so q == 1 identically and only the Fisher curvature is meaningful.
  - `FID_SAMPLE`: a THIRD encoding, our own literal.

So "the DESI fiducial" is written down twice independently. §38's q == 1 is
best understood as the assertion that those two agree; §40 is the assertion
that cosmoprimo's copy has not moved.

## §41 — `fiducial-id` promoted to an actual unit test

§40 added the check but left it manual, which undercut its whole point: its
value is catching a change *nobody in this repo made*, and a check you have to
remember to type is the wrong shape for that.

Added `tests/test_fiducial_identity.py`, the repo's first real test. It calls
`validate_forecast.check_fiducial_identity` rather than restating the numbers,
so the CLI and the test cannot drift apart on what the fiducial is. A second
test asserts `FID_SAMPLE` and the recorded cosmoprimo values agree to 1e-6 —
admitting the ln10A_s rounding §40 documented, and nothing larger.

Only this check was promoted. The rest of `validate_forecast.py` needs CLASS
runs over six tracers; `fiducial-id` is 3.2 s of assertions on constants.

Verified by negative test: perturbing the recorded `h` to 0.68 fails with
`h 0.6800000000 0.6736000000 -9.4e-03 <-- FAIL`, naming the parameter.

### Repo notes

- **`test_cov_scaling.py` is not a test.** It is an argparse plotting CLI that
  happens to carry the `test_` prefix, and pytest fails collecting it. Hence
  `testpaths = ["tests"]` plus `norecursedirs` in pyproject.toml — `pytest`
  from the repo root now works.
- pytest was not in the emulator env. Installed after a dry run confirmed it
  adds only `iniconfig`, `pluggy`, `pytest` and upgrades nothing (the env is
  pinned: desilike 4cfd6bec, cosmoprimo 1b100803). Declared as the `dev`
  optional-dependency; no pipeline needs it.
- Still no CI. The test runs on demand.

## §42 — z_eff depends on N_tracers (and on cosmology): both pipelines fixed

Prompted by a direct question: should the mean values depend on N_tracers?
They should, and neither pipeline was doing it.

### Why N moves z_eff at all

The FKP weight is not linear in n̄. With `w_i = n̄_i V_i / (1 + n̄_i P₀)`,
scaling n̄ -> alpha·n̄ leaves alpha in place, because slices at different
densities respond differently. Both limits ARE alpha-independent:

    n̄P << 1  ->  w -> alpha·n̄V,  alpha cancels  ->  NUMBER-weighted mean
    n̄P >> 1  ->  w -> V/P₀,      n̄ drops out    ->  VOLUME-weighted mean

so z_eff slides between them. Adding galaxies saturates the dense slices first
(they cross into n̄P >> 1 and stop gaining weight -- extra galaxies where you
already have plenty add little information) while sparse slices keep gaining,
so weight migrates toward the sparse end.

The effect is therefore largest where the bin STRADDLES n̄P ~ 1, NOT where the
bin is widest:

    tracer   n̄P range      spread   dz over [0.5, 1.5] x N_dr1
    LRG3     0.70-5.66      8.0x       +1.22%   <- only one crossing n̄P = 1
    ELG2     0.89-2.15      2.4x       +0.67%
    BGS      4.02-5.18      1.3x       +0.36%
    QSO      0.15-0.22      1.5x       +0.10%   <- WIDEST bin (dz = 1.3), but
    LRG1     5.12-5.46      1.1x       -0.02%      uniformly shot-noise
    LRG2     5.08-5.74      1.1x       -0.01%      limited, so alpha cancels

QSO is the instructive case: Δz = 1.3 and almost no N-sensitivity, because
n̄P ≈ 0.2 everywhere. LRG3 has Δz = 0.3 and moves twelve times more.
LRG1/LRG2 move the other way because their n(z) RISES across the bin, putting
the number-weighted mean above the volume-weighted one.

### Three treatments, none of which agreed

    pipeline    cosmology dep.        N dep.
    mean        FROZEN at fiducial    none
    covar       per sample            none
    correct     yes                   yes

The covar path took `N_tracers`, did the V_eff -> n_eff root-find with it, and
then computed z_eff from the unscaled file density -- internally inconsistent.
The mean path froze z_eff entirely, so mu and C of the SAME per-tracer Gaussian
likelihood were evaluated at different redshifts. Cosmology dependence across
the prior box, which the mean was discarding:

    BGS  -0.74%..+0.65%   LRG1 -0.19%..+0.19%   LRG2 -0.11%..+0.12%
    LRG3 -0.13%..+0.16%   ELG2 -0.16%..+0.21%   QSO  -0.84%..+1.08%

Note these are COMPLEMENTARY to the N effect: N dominates for LRG3, cosmology
for QSO and BGS. Neither is negligible relative to the other.

### Changes

`bao/core.py`: `_nz_scale_factor(tracer_bin, n_tracers, dataset)` returning
alpha = N/N_dataset; `_desi_z_eff_from_nz` and the `fisher_veff` branch both
apply it; `n_tracers`/`dataset` threaded through `_compute_z_eff_from_nz` and
both call sites (`build_bao_likelihood`, `compute_pipeline_sigmas`).
`n_tracers=None` restores the old behaviour exactly.

`shapefit/core.py`: same through `_fs_compute_z_eff`.

`shapefit/fourier_space.py`: the mean worker derives z_eff per sample from
cosmology + N_tracers. Extractor cache quantized to 1e-4 in z with FIFO
eviction at 16 entries -- per-sample z_eff would otherwise never hit the cache
AND grow without bound, and the mean path already holds ~0.5 GB RSS/worker.
Quantization is 0.01% in z -> ~0.002% in f_sigmar, three orders under the
emulator's own median error.

`shapefit/generate_mean_data.py`: **N_tracers is now a mean-emulator input**,
bounds from `ntracers_range` (never hardcoded, bao S33n). `--z-eff` still pins.

### Measured impact

Mean labels, over [0.5, 1.5] x N_dr1 at the fiducial cosmology:

    LRG3  f_sigmar 0.438254 -> 0.436993   (-0.29%)
    QSO   f_sigmar 0.374710 -> 0.374546   (-0.04%)

qiso and qap stay EXACTLY 1 regardless of N, as the ratio structure requires
(same z top and bottom). So the whole N-dependence of the mean lands on
f_sigmar, and m at the 1e-7 level.

Covar sigmas move as expected with the corrected z_eff; LRG3 z_eff now tracks
N (0.9324 / 0.9399 / 0.9439 at 0.5/1.0/1.5 x N_dr1).

### Why this is worth the interface change

All of it is <=0.3% on f_sigmar against DESI's 4-10% sigma, so it invalidates
nothing. The reason to fix rather than document is that the N piece varies
ALONG THE DESIGN AXIS. A constant bias cancels when bedcosmo compares two N
values; an N-dependent one does not, and comparing N values is the entire
question the emulator exists to answer.

### Consequences

- Goldens invalidated (bit-exact) -- both bao and shapefit. Regenerate.
- shapefit mean training data must be regenerated with the new input vector;
  covar too (already pending from the §36 z_eff convention change, so this
  costs one regen rather than two).
- bedcosmo: the mean emulator's input vector gains N_tracers, so models.yaml
  and `_build_emulator_input`'s whitelist need it. NOT yet done.
- bao Fourier sigmas change; bao CONFIG SPACE IS UNAFFECTED, since it pins
  z_eff to the DESI bundle and never reaches this path. Production BAO for
  bedcosmo is config space, so it is insulated.

### Caveat on the model

"More tracers = uniform rescaling of n(z)" holds the SHAPE fixed. Realistically
deeper observation also shifts the shape (more high-z objects), which would
move z_eff further. The numbers above are the conservative version.

## §43 — All six full-shape bundles fetched; §27a's "LRG-specific" n̄ deficit is REFUTED

### The blocker is gone

All six `likelihood_spectrum-poles-rotated_syst-hod_*_thetacut0.05.h5` are now in
`~/data/desi/bao_dr1/likelihoods/`. The CFS path, previously only guessed:

```
/global/cfs/cdirs/desi/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/data/likelihood/
```

LRG2's fetched copy is **md5-identical** to the pre-existing local one, which
validates the transfer of the other five. `compare_to_desi.py:_FS_BUNDLES` was
hardcoded to LRG2 (written when that was all we had); it is now derived from
`_DESI_SAMPLE`, so the two cannot drift apart, and absent files drop out
silently as before.

Access notes for next time — `data.desi.lbl.gov` and `portal.nersc.gov` are both
Spin-backed and stay down for the whole outage; the NOIRLab mirror is up but
carries only `spectro/`, no `vac/`. The DTNs stay up when Perlmutter is down
(they front CFS). NERSC firewalls **outbound** ssh from the DTNs, so you must
PULL from entropy, not push. `dtn01` resolves IPv6-only.

### The measured shot noise

`observable/spectrum/0/{norm, num_shotnoise}` — the covariance files carry
`1`/`0` placeholders, only the bundles have the real values.

| bin | norm A | num_shotnoise S | P_shot = S/A |
|---|---|---|---|
| BGS | 6.974617 | 39916.47 | 5723.11 |
| LRG1 | 5.702207 | 28975.91 | 5081.53 |
| LRG2 | 8.900322 | 46544.19 | 5229.50 |
| LRG3 | 11.988125 | 114768.89 | 9573.55 |
| ELG2 | 62.459913 | 667820.14 | 10691.98 |
| QSO | 14.409514 | 682678.92 | 47376.96 |

### 🔴 §27a was wrong: the deficit is universal, not LRG-specific

`s ≡ Pshot_ours/Pshot_DESI` is the n̄ rescale that reconciles us with DESI.

| tracer | z_eff | s (P_shot) | s (V_eff, §27a) | comp |
|---|---|---|---|---|
| BGS | 0.296 | 0.549 | 0.987 | 0.636 |
| LRG1 | 0.510 | 0.699 | 0.792 | 0.693 |
| LRG2 | 0.706 | 0.699 | 0.798 | 0.693 |
| LRG3 | 0.922 | 0.752 | — | 0.693 |
| ELG2 | 1.326 | 0.800 | 1.041 | 0.352 |
| QSO | 1.484 | 0.773 | 1.016 | 0.874 |

**LRG mean 0.717, non-LRG mean 0.707 — a difference of 0.009.** Every tracer is
low, at s = 0.712 with 11.5% scatter. §27a's "LRG-specific and ~20%, with BGS /
ELG2 / QSO within 4%" was an artifact of using V_eff, which **saturates where
n̄P ≫ 1 and is therefore blind to the sparse tail that sets shot noise**. §27
anticipated this ("V_eff is the weak probe... a null result cannot refute
dispersion, only fail to confirm it") — it turns out V_eff did not merely fail
to confirm, it pointed the wrong way. The V_eff − P_shot gap is **positive for
all five comparable tracers** (mean +0.223, max +0.438 on BGS), exactly the
signature of that blindness.

### What it is, and is not

- **Not completeness**: corr(s, comp) = −0.127, consistent with zero. Fourth
  independent strike after §26c, §27 and §27a — and the first from the sharp
  probe rather than a plausibility argument.
- **Tracks z_eff**: corr = **+0.869**, monotone apart from QSO. Worst at low z
  (BGS 0.549), approaching 0.8 by ELG2. The defect is in the **redshift
  behaviour of the FKP V_eff→n_eff mapping**, not in any per-tracer catalog
  property.

⚠️ Do **not** close this with a fitted per-tracer or global n̄ rescale. That is
precisely the fudge factor policy forbids. The z-trend is the lead to chase.

Speculative and NOT established: the BAO forecast's uniform ~23% σ
under-prediction (P/D ≈ 0.72–0.80) spans nearly the same range, and both would
follow from n_eff being too high. Untested.

## §44 — `W C Wᵀ` is invalid for the Fourier window (and valid for config space)

### The observation

`--check window` on all six tracers. Engine control (analytic vs desilike, same
grid, no window) is 0.99–1.09 everywhere, so the window/no-window differences
are the window and not the engine swap.

| tracer | unwindowed | windowed | offdiag ours/DESI |
|---|---|---|---|
| BGS | 1.111 | 0.266 | 0.391 |
| LRG1 | 2.332 | 0.574 | 0.565 |
| LRG2 | 2.038 | 0.510 | 0.537 |
| LRG3 | 1.768 | 0.608 | 0.476 |
| ELG2 | 0.836 | 0.320 | 0.403 |
| QSO | 0.726 | 0.344 | 0.205 |

Windowing drops our covariance by 3.3× on average, landing at 0.44 of DESI's
with only 43% of its bin-to-bin correlation. Both symptoms at once.

Note §19's headline **0.815 for LRG2 is superseded** — the post-damping-fix
value is 2.05 → 0.512, and we reproduce 2.038 → 0.510.

### Ruled out, cheaply

- **Survey area**: `DATASET_AREAS["dr1"] = 7500`, and our geometric volumes
  reproduce by hand (LRG2 2.775 vs 2.85 Gpc³/h³). Not it.
- **Window normalization**: `W @ 1` = 0.9418 / 1.0345 / 1.0190 / 0.9688 on the
  monopole block. Properly normalized. Not it.
- **The 3 zeroed systematic columns**: fixed templates with free amplitudes
  carry no variance, so dropping them cannot bias the covariance. §19's "mildly
  optimistic" remark applies to the theory/derivative side at the Fisher stage,
  not here.
- **Sub-fundamental binning of `C_kin`**: `C_kin` IS exactly diagonal
  (max off-diag corr `0.000e+00`) on a Δk = 0.001 grid whose bins are 4.5×
  narrower than LRG2's fundamental 2π/L = 0.0045. Looks damning, **is not**:
  block-averaging the theory grid gives 0.5103 → 0.5069 (×2) → 0.4955 (×5), a
  no-op. `fkp_analytic_cov` sets `dk = mean(diff(k))`, so the 1/δ variance
  scaling exactly compensates the independence assumption. Algebra: 5 independent
  fine bins give `5w²·(5V_coarse)`, one correlated coarse bin gives
  `(5w)²·V_coarse` — identical.

### The actual defect

| | theory Δ | obs Δ | bin ratio | M_eff | suppression |
|---|---|---|---|---|---|
| **bao config-space** (LRG2, QSO) | 1.00 Mpc/h | 4.00 | 4.00 | **4.00** | **1.000** |
| **shapefit Fourier** BGS | 0.001 h/Mpc | 0.005 | 5 | 17.8 | 0.282 |
| LRG2 | | | 5 | 19.1 | 0.262 |
| ELG2 | | | 5 | 13.5 | 0.371 |
| QSO | | | 5 | 10.5 | 0.477 |

`M_eff ≡ (ΣW)²/ΣW²` is the effective number of theory bins each observable bin
draws on. In Fourier it is **2–4× the observable bin width**, so a diagonal
`C_theory` picks up `ΣW² = (ΣW)²/M_eff` and the variance is suppressed by
≈ 5/M_eff. That is an **artifact**: the window redistributes modes, it does not
create independent ones. The mode count in an observable bin is set by the
survey volume and the bin width, not by the kernel width.

It also explains the second symptom — a diagonal `C_theory` has no correlation
to propagate, hence the 0.43 off-diagonal.

Underlying reason: `fkp_analytic_cov` reduces, for uniform n̄, to
`2(P + 1/n̄)²/N_modes` (the `I` integrals supply the volume; `bin_factor/V =
1/N_modes`). So `C_box` is already the covariance of the FKP estimator `P̂` —
and `P̂` is itself window-convolved, normalized by the same `I22`. Applying `W`
convolves a **second** time.

### ✅ Production BAO is NOT affected

`bao/config_space.py:442` uses the same `W @ C_theory @ W.T` pattern, but its
window is an **exact 4:1 rebinning**: M_eff = 4.00 matching the bin ratio to the
decimal, row sums exactly 1.0000, suppression exactly 1.000. Block-averaging is
the correct operation for a diagonal covariance. The same code pattern is valid
there and invalid here because the correlation-function window is a local rebin
while the Fourier full-shape window is a genuine mode-mixing kernel — which is
why the analogy did not catch it.

### What this leaves

The correct Gaussian baseline for shapefit is the FKP covariance on the
**observable grid** — the "unwindowed" column above, which was never the naive
comparison it was first labelled. Against DESI, Gaussian-only should sit a
little below 1 (the non-Gaussian remainder is the rest). ELG2 (0.836) and QSO
(0.726) are already about right. **The three LRG bins sit at 1.77–2.33, a factor
~2.5 spread from the others**, and that is the open question.

⚠️ **See §45 — the above "1.77–2.33 for the LRG bins" is NOT established as a
physical defect.** DESI's covariance sits below the cosmic-variance floor on the
same tracers, so part or all of that ratio is an unresolved property of the
comparison, not of our covariance. Do not chase a physical LRG explanation
before reading §45.

## §45 — DESI's covariance sits BELOW the cosmic-variance floor; §44's LRG claim is withdrawn

> ⚠️ **SUPERSEDED IN PART BY §56.** The headline claim below — that DESI's
> covariance is anomalous — overstates what the evidence supports. The floor it
> is measured against rests on two assumptions this section never tested. Read
> §56 before acting on anything here.

### What was chased, and eliminated

§44 closed by calling the LRG bins' 1.77–2.33 "the open question". Four candidate
causes were tested and **all four are dead**:

1. **n(z) slice volumes.** `Σ Vᵢ / V_geom = 1.000` and `Σ n̄ᵢVᵢ / N_dr1 = 1.000`
   for all six tracers, exactly. The slices are self-consistent.
2. **SSC.** 0.0006 (QSO) to 0.098 (BGS) of our diagonal. Removing it entirely
   leaves 2.05–3.15 against the plain covariance. Not the driver.
3. **Rotation / θ-cut.** Against the PLAIN (unrotated, no θ-cut) covariance the
   ratios are *worse*, not better: 2.275 / 3.410 / 3.139 / 2.867 / 1.497 / 1.166.
4. **Normalization convention.** The plain covariance files carry `norm = 1`,
   `num_shotnoise = 0` — but these are "already applied" markers, NOT raw-unit
   flags. Bundle P0 / plain P0 on matched k is 0.921 / 1.072 / 0.952
   (BGS/LRG2/QSO), where raw units would demand 1/norm = 0.143 / 0.112 / 0.069.
   Both files are in the same normalized units.

### The actual finding

`N_from_cov ≡ 2(P₀+P_shot)²/Var` is the independent-mode count DESI's covariance
behaves as if it had. As a multiple of the survey's own `V k²Δk/2π²`:

| tracer | k=0.048 | k=0.098 | k=0.198 |
|---|---|---|---|
| BGS | 3.08 | 3.55 | 3.87 |
| LRG2 | 2.77 | 2.97 | 2.79 |
| QSO | 1.91 | 2.02 | 2.09 |

**Flat in k** across a 4× range (LRG2: 2.77 / 2.97 / 2.79). A physical effect
would carry k-structure; a flat offset is a normalization.

And the level is impossible: a covariance cannot behave as if it had 2–4× more
independent modes than the survey volume supports. That is below the
cosmic-variance floor. Worse, the floor quoted here is itself conservative —
with RSD, `Var(P₀) = 2/N ∫dμ/2 [P(k,μ)+1/n̄]² ≥ 2(P₀+1/n̄)²/N` by Jensen — so the
true floor is higher and the violation larger. Solving for an area that would
rescue it needs ≥ 22,000 deg², against DR1's 7500 and DESI's 14,000 final.

The violation is confined to the **cosmic-variance-dominated** tracers. It scales
with n̄P: QSO (n̄P = 0.21) sits at 1.9–2.1, ELG2 (0.63) between, the dense LRG/BGS
bins (2.7–4.8) at 2.8–3.9. In the shot-noise-dominated limit the discrepancy
nearly vanishes — which is also why the bundle covariance gives a *physical*
V_implied/V_geom of 0.99 / 1.06 for ELG2/QSO and 1.29–1.82 for BGS/LRG.

### Consequence

**§44's LRG claim is withdrawn.** Our covariance reproducing ~2–3× DESI's on the
dense bins cannot be attributed to a defect on our side while DESI's own
covariance is below the floor by a comparable factor on the same bins. The two
observations are the same unexplained number seen from opposite ends.

Working rule until this is resolved:
- The **bundle** covariance is the only internally consistent reference (real
  `norm`/`num_shotnoise`, physical V_implied for the sparse tracers). The plain
  covariance files are unusable for absolute levels.
- Do **not** tune anything against these ratios, and do not build a physical
  story for the LRG bins on them.

### The check that bypasses all of this

`--check compressed` compares against DESI 2024 V Appendix A's **published**
constraints on the compressed parameters. That needs no covariance surgery, no
window, no volume rescaling — and it is the number that actually matters for the
emulator. It should be the arbiter, not the element-by-element covariance ratio.

### The arbiter, run

`--check compressed` against DESI 2024 V Appendix A, ours/DESI:

| tracer | σ(qiso) | σ(qap) | σ(fσ_r)/fσ_r | σ(m) |
|---|---|---|---|---|
| BGS | 0.61 | 0.95* | 0.67 | 0.67 |
| LRG1 | 0.99 | 1.01 | 1.03 | 1.07 |
| LRG2 | 0.81 | 0.85 | 0.91 | 0.87 |
| LRG3 | 0.69 | 0.67 | 0.73 | 0.73 |
| ELG2 | 0.65 | 0.58 | 0.54 | 0.87 |
| QSO | 0.78 | 0.81 | 0.75 | 0.94 |

*DESI warns BGS α_AP is prior-dominated (flat 0.8–1.2), so its σ(qap) is a prior
width, not a measurement.

**0.54–1.07, mostly 0.65–0.95** — where a Fisher forecast at the truth belongs
against an MCMC posterior. LRG1 is 0.99–1.07.

🔴 **RETRACTED.** This does NOT settle §45. The argument assumed our σ would
otherwise equal DESI's published σ — but those are MCMC posterior widths with
nuisance marginalization, and a Fisher forecast at the truth is inherently
tighter. The two comparisons do not share a denominator.

Chaining them properly (`--check sigma` gives σ_ours(our C)/σ_ours(DESI C);
`--check compressed` gives σ_ours(our C)/σ_DESI_pub):

| tracer | cov ratio | √(cov ratio) | measured σ swap | ours/pub | with DESI's C |
|---|---|---|---|---|---|
| BGS | 1.111 | 1.054 | 1.114 | 0.725 | 0.651 |
| LRG1 | 2.332 | 1.527 | 1.300 | 1.025 | 0.788 |
| LRG2 | 2.038 | 1.428 | 1.279 | 0.860 | 0.673 |
| LRG3 | 1.768 | 1.330 | 1.267 | 0.705 | 0.556 |
| ELG2 | 0.836 | 0.914 | 0.946 | 0.660 | 0.698 |
| QSO | 0.726 | 0.852 | 0.952 | 0.820 | 0.861 |

Columns 3 and 4 agree: **the covariance ratio is real and propagates into σ as
expected**. Column 6 — our Fisher with DESI's own covariance against DESI's
published MCMC — is 0.56–0.86, a normal Fisher-vs-MCMC gap. So "our covariance
is 2× large" and "our σ are tighter than published" are fully consistent, and
the σ comparison never refuted anything.

Correlation structure is broadly right — `qap–f_sigmar` −0.65…−0.73 against
DESI's −0.53…−0.63 on every tracer, `qiso–qap` +0.22…+0.28 vs +0.21…+0.24 on the
LRG bins. Two genuine weaknesses, both small and both worth revisiting only after
the above is resolved: we under-predict `f_sigmar–m` (+0.01…+0.07 vs DESI's
+0.24…+0.37), and several near-zero correlations disagree in sign (|ρ| < 0.1,
where a Fisher-vs-MCMC sign flip is not meaningful on its own).

### §45a — the covariance ratio is fully accounted for by ONE anomaly

Chaining the implied-volume numbers removes the need for two separate stories.
For LRG2 against the bundle covariance:

- ours implies **0.917** of the survey's mode count (just below 1, as FKP
  weighting requires)
- DESI's implies **1.818**
- predicted ratio 1.818 / 0.917 = **1.98**; measured **2.038**

That reproduces the whole excess. There is **one** anomaly, not two: our
covariance sits where a Gaussian FKP calculation should, and DESI's sits below
the cosmic-variance floor. The "LRG excess" is not a defect on our side, and
§44's framing of it as one was wrong.

### Table 1's V_eff is in Gpc³, NOT (Gpc/h)³

A trap worth recording. DESI 2024 V Table 1 prints `Veff [Gpc3]` while printing
`P0 [(h⁻¹Mpc)³]` in the adjacent column. Read naively in (Gpc/h)³ it gives
V_eff/V_geom = 1.80 / 1.45 / 1.44 for BGS/LRG1/LRG2 — impossible, since
V_eff = ∫[n̄P/(1+n̄P)]²dV ≤ V, and alarming because those are exactly the three
bins with the covariance excess. Multiplying by h³ = 0.306:

| tracer | V_eff (Gpc/h)³ | V_geom | ratio |
|---|---|---|---|
| BGS | 0.52 | 0.944 | 0.55 |
| LRG1 | 0.79 | 1.791 | 0.44 |
| LRG2 | 1.22 | 2.775 | 0.44 |
| LRG3 | 1.53 | 5.733 | 0.27 |
| ELG2 | 0.83 | 12.392 | 0.067 |
| QSO | 0.46 | 32.294 | 0.014 |

All ≤ 1, monotonically ordered by sparsity. **Our geometric volumes are correct**
and the paper confirms the ~7,500 deg² footprint. Do not re-derive an area from
Table 1 without the h³.

### Still open

Why DESI's covariance sits below the Gaussian floor. Note the floor should
arguably be computed with V_eff (0.44 V_geom for LRG2), not V_geom, which makes
the expected variance *larger* and the violation *worse* — so the resolution is
not a volume choice. Unresolved; do not tune against these ratios meanwhile.

### Provenance of the published constraints — verified

`desi_reference._SF` and `_T11` were hand-transcribed and are now checked
against arXiv:2411.12021 directly (`pdftotext -layout`). All **84** `_SF` values
(6 tracers × 4 data-vector + 10 covariance entries) match verbatim AND
positionally — the stored sequence is an exact prefix of the numbers in each
tracer's own Appendix-A section, which rules out transposition, the failure mode
presence-only matching would miss. All six `_T11` rows confirmed with their
z_eff row headers. The paper PDF is not in the repo; re-fetch from arXiv if this
needs redoing.

## §46 — §43 RETRACTED: the shot-noise deficit is a NORMALIZATION convention, not physics

### What §43 claimed, and why it was wrong

§43 concluded the FKP `V_eff → n_eff` mapping mis-sets the shot-noise floor
(`s = 0.712`, tracking z_eff at r = +0.869) and called the z-trend "the lead to
chase". **There is no physical deficit.** The mapping is fine, our n(z) is fine,
and the z-trend was the normalization ratio's tracer ordering in disguise.

Established with the DR1 LSS catalogues (v1.5) read directly on CFS at
`/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/`.
Galaxy counts in our z-bins match `N_dr1` **exactly** (LRG1 506911, LRG2 771894,
LRG3 859822, ELG2 1415707; BGS +26, QSO +179), independently confirming our
sample definitions and the §31 LRG-only LRG3 bin.

### The authority: DESI 2024 II (arXiv:2411.12020) §10, Eqs. (10.5)–(10.6)

```
A  = (alpha_R / dV) * sum_cells n_D,i n_R,i      dV^(1/3) = 10 Mpc/h  (FIXED)
N0 = (1/A) [ sum_D w_D,i^2 + alpha_R^2 sum_R w_R,i^2 ]
alpha_R = sum w_d / sum w_r        (by WEIGHT, not by count)
```

So the bundle's `num_shotnoise` is the bracket and `norm` is A.

**Shot noise: CONFIRMED to 42 ppm.** Reconstructed from the catalogues for LRG2:
46546.14 vs the bundle's 46544.19. alpha-by-count gives 46519.7 (5.3e-4), an
order of magnitude worse — the paper's alpha-by-weight is right. We compute S
exactly as DESI does.

**Norm: a MESH quantity, not a particle sum.** No particle-sum variant could ever
have matched, which is why guessing variants was the wrong approach. Measured
`A_naive = sum_data NX w w_fkp^2` against the bundles:

| bin | A_naive | A_DESI | ratio | required inflation | residual |
|---|---|---|---|---|---|
| LRG1 | 8.3182 | 5.7022 | 1.4588 | 1.438 | **1.014** |
| LRG2 | 12.8853 | 8.9003 | 1.4477 | 1.454 | **0.996** |
| LRG3 | 16.9079 | 11.9881 | 1.4104 | 1.436 | **0.982** |
| ELG2 | 77.1786 | 62.4599 | 1.2356 | 1.221 | **1.012** |
| QSO | 17.3144 | 14.4095 | 1.2016 | 1.257 | **0.956** |
| BGS | 8.9518 | 6.9746 | 1.2835 | 1.819 | 0.706 |

**Five of six agree to ≤4.4%**, reproducing not just the level but the
tracer-to-tracer *variation* (LRG ~1.45, ELG2/QSO ~1.2) that defeated every
physical hypothesis tried.

Mechanism: the particle sum sees the true small-scale density; the mesh smooths
over 10 Mpc/h cells, which lowers `sum n_D n_R` wherever the selection function
is patchy below that scale (veto masks, fibre-collision holes, tile boundaries).
Suppression therefore varies per tracer — correlated with completeness without
equalling it (LRG 0.686-0.709 vs comp 0.693; but ELG2 0.809 vs comp 0.352).

### Hypotheses killed on the way (do not re-run these)

1. **n_eff definition.** FKP-consistent `A/S` gives 0.702 vs V_eff-match's 0.712.
   Identical. The functional was never the problem.
2. **Density dispersion at fixed z.** Needs implausible lognormal sigma = 1.26-1.85
   for BGS/LRG, and is *unreachable in the wrong direction* for LRG3/ELG2/QSO —
   adding scatter LOWERS P_shot (QSO 1.00 -> 0.23), because the FKP weight peaks
   at nbar*P ~ 1 and down-weights both ends. This is also why LRG3's CV(n) = 0.568
   still yields P_shot/uniform = 0.998.
3. **Weight dispersion.** Real and large (CV(w) = 0.31-0.61, measured), and it
   closes most of the *mean* offset (1.40 -> 1.08 residual) — but not the
   per-tracer pattern, and it over-shoots ELG2 (measured 1.45 vs required 1.22).
   A genuine survey property, but not the cause here.

### Consequence for the emulator

A splits as `A(N) = [patchiness] x integral(nbar^2 w^2 dV)`. The bracket is a
property of the MASK and RANDOMS — the same class as a randoms-derived window,
which policy permits (geometry, not clustering) — and to first order it is
N-independent, since n_D and n_R both scale linearly and the smoothing ratio
depends on the shape of the selection function, not its amplitude. Residual
N-dependence enters only weakly through `w_fkp = 1/(1+nbar*P)`, already modelled.
**So the N-scaling stays analytic and the DR1 value only calibrates geometry.**
This is NOT a fudge factor: it is a measured geometric normalization, not a
tuned parameter, and it does not absorb any cosmology or N dependence.

### Still open

**BGS residual 0.706** — an extra factor 1.42 beyond the norm ratio, unique to
BGS. Bright-time sample with its own footprint; suspicion is our V_survey/area
for that bin rather than the norm, but that is a guess and needs its own check.

Other facts from §10 worth keeping: DESI uses **all 18 randoms** ("more than 100x
the data density"), and per-tracer box sizes 4000/7000/9000/10000 Mpc/h for
BGS/LRG/ELG/QSO with 6 Mpc/h cells (k_Nyq ~ 0.5). The **10 Mpc/h norm cell is
separate** from the 6 Mpc/h measurement grid.

## §47 — DESI's norm reproduced from randoms alone: the normalization is GEOMETRY

§46 identified the mesh normalization as the cause but left it as a measured
ratio `A_naive/A_DESI`, which is a fudge factor by policy and — worse — a fixed
constant cannot respond to N_tracers. This section removes both objections.

### Reproducing Eq. (10.5) directly

Painted the DR1 v1.5 catalogues onto the 10 Mpc/h grid Eq. (10.5) specifies,
NGC+SGC, LRG 0.6–0.8, and evaluated two forms:

```
A_exact = (alpha_R/dV)   sum_cells n_D,i n_R,i      data x randoms
A_ran   = (alpha_R^2/dV) sum_cells n_R,i^2          randoms only (n_D ~ alpha_R n_R)
```

| NRAN | A_exact | A_ran raw | A_ran corr | corr/exact | exact/TRUTH |
|---|---|---|---|---|---|
| 4 | 9.03945 | 10.75558 | 9.04164 | **1.0002** | 1.0156 |
| 8 | 9.04465 | 9.89866 | 9.04158 | **0.9997** | 1.0162 |

Three results:

1. **Eq. (10.5) reproduces the bundle norm to 1.6%** (8.900322). The offset is
   stable across NRAN, so it is not statistical — almost certainly NGP vs DESI's
   TSC painting. Accept for now; revisit if 1.6% ever matters.
2. **Randoms-only equals data×randoms to 0.03%** once self-pairs are removed.
   `n_R,i^2 = sum_j w_j^2 + sum_{j!=k} w_j w_k`; the first term is pure shot
   noise, so the unbiased estimator is `sum_i n_R,i^2 - sum_all w_r^2`.
3. **The correction removes the NRAN dependence exactly**, not approximately:
   raw drifts 10.756 -> 9.899 while corrected moves 9.04164 -> 9.04158 (7e-6).

So the normalization is **pure survey geometry** — no data catalogue, no fitted
factor, computable from the random catalogues alone. That is the same footing as
a randoms-derived window, which policy permits.

### ⚠️ Why a single constant is NOT enough

`w_r = WEIGHT x WEIGHT_FKP` and `w_fkp = 1/(1 + nbar*P0)` depends on nbar —
which is exactly what N_tracers changes. Extracting one `G` at DR1 density would
silently freeze the FKP weighting and break the N-scaling, i.e. reintroduce the
§42 failure mode (a constant bias cancels between two N values; an N-dependent
one does not).

A 10 Mpc/h cell spans a narrow z range, so `w_fkp` is constant within a cell:

```
n_R,i        = w_fkp(z_i) * m_i          m_i = sum_j w_comp,j   (completeness only)
sum n_R,i^2  = sum_i w_fkp(z_i)^2 m_i^2
```

The extraction must therefore be a **per-z-slice geometry summary** — `sum m_i^2`,
`sum m_i`, cell count, and the self-pair `sum w_comp^2`, on the same z slices as
`~/data/desi/nz_slices/{tracer}_nz_slices.csv`. Then `A(N)` is analytic: reweight
each slice by `w_fkp(z; nbar(N))^2` and sum.

### Where this leaves the shot noise

Both halves of `P_shot = S/A` now come from DESI's own definitions:
- `S` verified to 42 ppm (§46)
- `A` verified to 1.6%, from geometry, with analytic N-dependence

No fitted parameter anywhere, and nothing that absorbs cosmology or N.

### Method notes

- Distance integrator (flat LCDM, Om = 0.31519 from AbacusSummit c000 incl. one
  0.06 eV neutrino) agrees with cosmoprimo to 0.007% over z = 0.6–0.8.
- `np.bincount` on flattened cell indices, not `np.add.at` — the latter is far
  slower at ~90M points.
- Grid from the data bounding box + 200 Mpc/h padding: 28.3M cells (NGC),
  15.8M (SGC) at 10 Mpc/h.
- Both A forms are invariant to NRAN in the mean (`n_R ~ N_ran`,
  `alpha_R ~ 1/N_ran`); only the Poisson bias in `sum n_R^2` is not, hence the
  self-pair subtraction rather than simply using all 18.

## §48 — the per-slice factorization works; the factor 2 was our w_fkp, and it exposes an n(z) discrepancy

### The factorization is sound (11%, not 2x)

Computing both forms in ONE script on ONE grid with ONE z-cut (LRG 0.6–0.8,
NRAN=2) removes the confounds that made the first comparison unreadable:

```
Q_exact = sum_i n_R,i^2 - sum_j (w w_fkp)^2                    = 1.52389e6
Q_recon = sum_s <w_fkp>_s^2 (sum_i m_i^2 - sum_j w_j^2)        = 1.69348e6
ratio 1.1113   (NGC 1.1215, SGC 1.0880)
```

`Q_exact` reproduces the direct mesh run (1.525e6), so §47's method is confirmed.
The per-slice reconstruction is good to **11%** — a real approximation error from
`w_fkp` varying within a slice, reducible with finer z bins, and worth reducing
since it is large next to §47's 1.6% painting offset.

### 🔴 The factor 2.096 was a self-inflicted error

The earlier reconstruction used `w_fkp` computed from **our** n(z) slices
(0.272–0.297). DESI ships it as a column, and their value is **0.19–0.21**:

```
(0.29/0.20)^2 = 2.10        observed discrepancy 2.096
```

Exactly the factor. Do not recompute a quantity DESI ships — `WEIGHT_FKP` is in
every clustering catalogue.

### 🔴 What that exposes: our n(z) is ~29% LOW

DESI's `NX` for LRG 0.6–0.8 is **3.47–3.98e-4** (`NX*P0 ~ 3.2`); ours is
**2.66–3.01e-4** (`nbar*P0 ~ 2.45`). Since `sum n_i V_i = N` holds exactly by
construction (§46), an n̄ that is 29% low means the slice **volumes are 29% too
large** — i.e. the effective area is nearer **5800 deg²** than the nominal 7500
(`V = N/nbar = 2.14e9` vs our `V_survey = 2.775e9`).

The nominal DR1 footprint is not the completeness-weighted effective area, and we
have been using the nominal one everywhere.

Worse, `~/data/desi/nz_slices/{tracer}_nz_slices.csv` carries **three** different
n̄ columns and none matches DESI:

| quantity | LRG2 |
|---|---|
| `nbar_design_file_volume` | 2.2e-4 |
| what `load_nz_slices` returns | 2.8e-4 |
| `nbar_file` = `nbar_shape_file_volume` | 5.3e-4 |
| **DESI `NX`** | **3.6e-4** |

And `file_area_deg2` is 30092.214 on row 0 but 20061.476 on rows 1+, with
`file_effective_area_deg2` 28651 / 19101. Neither is a plausible DESI area and it
should not vary by row. Row 0 also has `Nbin_file != shape_weight` while later
rows have them identical — consistent with the parser artifact §28 called
"harmless". It may not be.

### Why this matters beyond the norm

`V_survey` sets the covariance mode count, not just the shot noise. A 29% volume
error propagates into every sigma. This is also the leading candidate for the
**BGS residual** (§46, 0.706, needing an extra 1.42) — bright-time BGS would have
its own effective area, different again from the dark-time tracers.

### Open

1. Extract `<NX>(z)` per tracer from the catalogues (cheap — data files only) and
   compare against all three of our columns, for all six tracers.
2. Settle which of our n̄ columns is meant to be DESI's, and whether the
   `file_area_deg2` inconsistency is a parser bug or a misread of the source.
3. Only then implement §46/§47 — the norm fix depends on n̄ through `w_fkp`, so
   fixing the norm against a wrong n(z) would bake the error in.

## §49 — the FKP pivots in tracers.yaml are the wrong QUANTITY; §48's n(z) claim is NOT established

### Solid: our `fkp_p0` values are not DESI's FKP pivots

DESI 2024 II Eqs. (8.3)–(8.4), verbatim:

```
n_x(z, n_tile) = n(z) <C_assign>(n_tile)
w_FKP(z, n_tile) = 1 / [1 + n_x(z, n_tile) P0]
```

> "The value of P0 is chosen separately for each tracer, given an approximate
> nominal value of the power spectrum monopole P0(k = 0.15 h/Mpc). The values used
> in the DR1 analysis are P0,BGS = 7000, P0,LRG = 10,000, P0,ELG = 4000 and
> P0,QSO = 6000 (Mpc/h)^3. **These values are only roughly consistent with the
> actual clustering amplitude** of the respective DESI samples..."

So the pivots are a **convention**, coarse round numbers per tracer. Our
`tracers.yaml` carries DESI 2024 III **Table 2's P0(k=0.14)** — a *measured*
clustering amplitude, a different quantity:

| bin | ours (Table 2) | DESI Eq. (8.4) | ratio |
|---|---|---|---|
| BGS | 9200 | 7000 | 1.31 |
| LRG1 | 8900 | 10000 | 0.89 |
| LRG2 | 8900 | 10000 | 0.89 |
| LRG3 | 8400 | 10000 | 0.84 |
| ELG2 | 2900 | 4000 | 0.73 |
| QSO | 5000 | 6000 | 0.83 |

Wrong in both directions, by up to 31%. Confirmed empirically: back-solving
`P0 = (1/w_FKP - 1)/NX` on the LRG catalogue gives **10648 ± 17** (0.16% scatter)
against the stated 10000 — the 6.5% is Jensen bias from averaging `1/(1+nP)`
over a slice. The relation and the value both check out.

`NX` is DESI's own column: the local expected density `n(z)<C_assign>`, per
object. Do not recompute it (§48) — and note it is **not purely radial**, since
`<C_assign>` varies with n_tile.

### ⚠️ But applying the correct pivots makes z_eff WORSE

z_eff vs DESI's published values (2411.12021 Table 1), mean |error| over six bins:

| pivots | nbar scale | max |err| | mean |err| |
|---|---|---|---|
| Table-2 (current) | 1.00 | **0.650%** | **0.313%** |
| Table-2 | 1.29 | 0.945% | 0.457% |
| Eq-8.4 | 1.00 | 1.022% | 0.401% |
| Eq-8.4 | 1.29 | 1.311% | 0.545% |

The current configuration is the best of the four. Monotonic degradation in both
directions.

### 🔴 RETRACTION: §48's "our n(z) is ~29% LOW" is NOT established

§48 compared our slice n̄ (a **volume average**, `N x slice_fraction / V_slice`)
against `<NX>` averaged over **randoms weighted by WEIGHT** — an object-weighted
average of a spatially varying quantity. Those estimate different things, and a
22% gap between them is unremarkable. The comparison was set up wrong.

This also explains why the pivot fix alone degrades z_eff: only the product
`nbar*P0` enters `w_FKP`, so one side cannot be corrected while the other is
measured on a different footing.

### Do not change tracers.yaml yet

The pivot mismatch is real, but the one thing we can validate (z_eff against
published values) gets worse when we "fix" it. That is the signature of
compensating errors, and changing one of them alone is how you make a pipeline
agree with nothing.

Next, and it is a measurement not an inference: extract `<NX>` **volume-weighted**
on the same 10 Mpc/h mesh (cell-average NX, then average over cells, not over
objects). That is apples-to-apples with our slice n̄ and settles whether any
discrepancy exists. Only then revisit the pivots — together with n̄, not alone.

## §50 — the n(z) gap is REAL and converged at ~23%; §49's retraction was half right

Volume-weighted `<NX>` on the 10 Mpc/h mesh, LRG 0.6–0.8, NRAN 2/4/8.

### The estimator mattered, but only for 42% of it

| | ratio to our slice n̄ |
|---|---|
| `<NX>_obj` (object-weighted — what §48 used) | **1.394** |
| `<NX>_vol` (volume-weighted — correct) | **1.227** |

§49 retracted §48's "n(z) is ~29% low" on the grounds that the comparison was
object- vs volume-weighted. That was **half right**: switching estimators moves
1.394 → 1.227, closing 42% of the gap. The remaining **23% is real.**

`<NX>_obj` converges instantly (4.0083/4.0086/4.0084e-4 at z=0.60 across
NRAN 2/4/8) since object-weighting is occupancy-independent. `<NX>_vol` drifts
~0.6% per doubling and is still settling.

### The boundary systematic cancels in the ratio

Occupied-cell volume is badly behaved — 1.162 → 1.272 → 1.318 of our full-shell
volume across NRAN 2/4/8, **not converged**, because a 10 Mpc/h cell straddling
the patchy DESI mask counts as fully occupied and each extra random file finds
more edge cells. So `V_occ` is unusable on its own, and so is `int(NX dV)`
(1.453 → 1.572 → 1.619 × N).

But their **ratio converges**: boundary-corrected `int(NX dV)/N` = 1.250 →
1.235 → **1.229**, and it agrees with `<NX>_vol/ours` = 1.227 measured
independently. Two routes, same answer.

### What 1.23 means

`integral(NX dV) = 1.23 N` over our 7500 deg² shell. Either DESI's `NX`
overcounts by 23%, or **our V is 23% too large — an effective area near
6100 deg² rather than the nominal 7500.** The nominal DR1 footprint is not the
completeness-weighted effective area, and `V_survey` sets the covariance mode
count, not just the shot noise.

Still the leading candidate for the **BGS residual** (§46, 0.706): bright-time BGS
would have its own effective area again.

### Status of the three coupled quantities

- `fkp_p0`: wrong quantity, established (§49). DESI Eq. (8.4) pivots are
  7000/10000/4000/6000.
- n̄: ~23% low, established here.
- z_eff: currently agrees with DESI to 0.31% mean **with both errors in place**.

Only the product `nbar*P0` enters `w_FKP`, and z_eff degrades when either is
fixed alone (§49). With n̄ ×1.23 and the Eq-8.4 pivots, LRG2 gives
`nbar*P0 = 2.8e-4 × 1.23 × 10000 = 3.44` against DESI's measured `NX*P0 = 3.6` —
close, where the current config gives 2.49. **Fix them together and re-test
z_eff; do not fix either alone.**

## §51 — Eq. (2.1) reproduced from DESI's own randoms to 0.064%: a reference z_eff

Computing DESI 2024 III Eq. (2.1) directly from the v1.5 random catalogues —
`n_ran = sum(WEIGHT x WEIGHT_FKP)` per slice, weight `n_ran^2 r^2 dr`, mean z
within each slice from `sum(z w wf)/sum(w wf)`, dz = 0.01, NRAN = 2:

| bin | z_eff (DESI randoms) | published | err |
|---|---|---|---|
| BGS | 0.2954 | 0.295 | +0.138% |
| LRG1 | 0.5095 | 0.510 | −0.097% |
| LRG2 | 0.7058 | 0.706 | −0.028% |
| LRG3 | 0.9185 | 0.919 | −0.054% |
| ELG2 | 1.3169 | 1.317 | −0.010% |
| QSO | 1.4901 | 1.491 | −0.057% |

**mean 0.064%, max 0.138%** — against our pipeline's **0.313% / 0.650%**. Five
times better, with a residual consistent with the binning and NRAN=2 noise.

No n(z) model, no `fkp_p0`, no slice files, no effective-area assumption (the
area cancels in the ratio). Only DESI's catalogue and the fiducial distance
relation. This is a **reference implementation** of Eq. (2.1) and it validates
both the formula and the estimator.

### The missing ingredient, identified

`n_ran` uses **WEIGHT x WEIGHT_FKP**. `WEIGHT` carries completeness, imaging
systematics and redshift failures (DESI 2024 II Eq. 8.2). Our
`_desi_z_eff_from_nz` uses `nbar/(1 + nbar*P0)` — the FKP factor **only**.

That is the compensating error §49 was circling: with the completeness term
absent, the Table-2 pivots partially stand in for it, which is why "fixing" the
pivots alone made z_eff worse.

### It factorizes, so N-scaling stays analytic

```
n_ran,s = S1_s * w_fkp(z_s ; nbar(N))        S1_s = sum WEIGHT per slice
```

`S1_s` is randoms-derived geometry (already extracted, §47/§48) and every N and
cosmology dependence sits in `w_fkp` and `V_s(cosmo)`. So:

```
z_eff(N, cosmo) = SUM z_s [S1_s w_fkp,s]^2 / V_s   /   SUM [S1_s w_fkp,s]^2 / V_s
```

fully analytic, no fitted parameter, same footing as the §47 norm.

### Unblocks the §49/§50 tangle

We now have a target rather than a coincidence to preserve. The three coupled
quantities can be fixed **together** and checked against 0.064%:

1. add the `WEIGHT` (completeness) term via `S1_s`,
2. restore the Eq. (8.4) pivots (7000/10000/4000/6000) — note these are exactly
   what `_FKP_P0_BY_TYPE` held before commit 740402d, which replaced them with
   Table-2's `P0(k=0.14)`; `bao/fkp_analytic_cov.py:51` still has the correct
   `P_FKP_DEFAULT = 1.0e4`, so the two paths currently disagree for LRG,
3. re-examine n̄ (§50) once 1 and 2 are in.

If our z_eff does not approach 0.064% with all three, something else is wrong —
which is a far better test than any we have had.

## §52 — what DESI's `WEIGHT` column actually is (2411.12020, verified)

Needed before implementing §51, since `S1_s = sum WEIGHT` is the geometry term.

| symbol | definition | ref |
|---|---|---|
| `f_TLID` | inverse of the number of unique DESI targets competing for that tile+fiber | §5.1 |
| `w_comp = 1/f_TLID` | fiber-assignment completeness; "analogous to the SDSS close pair weight" | Eq. (5.2) |
| `f_tile` | per-`t_group` completeness for what `f_TLID` misses (fiber lost to a standard star/sky; ELG losing to an LRG). Analogous to SDSS sectors / `C_BOSS` | §5.1 |
| `w_zfail` | redshift-failure weight | §7 |
| `w_imsys` | imaging-systematics weight | §6 |
| `w'_tot = w_comp w_zfail w_imsys` | | Eq. (8.1) |
| **`w_tot = w'_tot / <w_comp>(n_tile)`** | **= the `WEIGHT` column** | Eq. (8.2) |

### Three consequences

1. **`WEIGHT` is NOT `1/completeness`.** Eq. (8.2) divides the mean completeness
   at each `n_tile` back out, so total data counts "sum to approximately the
   number of observed objects" while still tracking selection variation. What
   remains is the *variation about* the mean, not the mean. Consistent with §46:
   our per-bin counts match `N_dr1` **exactly**, rather than exceeding it.
2. **`f_tile` is applied to the RANDOMS, not the data** (§5.3, §8.1). So the data
   and random `WEIGHT` columns are different quantities — relevant because
   `S1_s` was extracted from randoms.
3. **Random weights are sampled FROM galaxies** (§8.1): randoms take redshifts
   and all weights "from randomly chosen galaxies", present "purely to make sure
   that their weighted (normalized) redshift distribution matches the data."

Point 3 explains §51's shape agreement: our `slice_fraction` matched the `S1`
shape to **0.77%** because the randoms' radial distribution is shuffled *from*
the data and must match by construction. It also confirms `S1_s` is a
**geometric** quantity — mask, tiling, per-region normalization — not an
independent re-measurement of n(z). That is the footing policy allows.

### ⚠️ Caveat for implementation

§5.3 states the fiducial `w_comp` "does not produce unbiased 2-point clustering
statistics" alone — the **θ-cut** removes the small-angle bias and is the DR1
default. Our pipeline has no θ-cut. Adopting DESI's weighting therefore matches
their *weights* but not their *pair cut*; say so explicitly rather than implying
full equivalence.

## §53 — z_eff computed DESI's way: 0.313% -> 0.062%

Three coupled corrections, landed together because each alone makes things worse
(§49). Benchmarked by `shapefit/benchmark_desi.py`.

### 1. Eq. (2.1) now uses the WEIGHTED RANDOM density

`n_ran,s = S1_s * w_fkp(z_s ; nx_s)`, so the Eq. (2.1) weight is
`(S1_s w_fkp)^2 / V_s`, with both inputs from DESI's own v1.5 randoms:

- `S1_s` = `sum WEIGHT` per slice — mask, tiling, per-region normalization.
  Survey GEOMETRY, fixed in N.
- `nx_s` = `NX` = `n(z) <C_assign>` (2411.12020 Eq. 8.3) — the density DESI's
  own `WEIGHT_FKP` is built from. SCALES with N.

Was `nbar_file`, which is neither (1.19-2.37x high per tracer). Shipped as
`~/data/desi/nz_slices/{tracer}_desi_nx.csv`; falls back with a warning where
absent (LRG3_ELG1, which needs the LRG+ELG_LOPnotqso catalogue).

### 2. FKP pivots restored to Eq. (8.4)

7000 / 10000 / 4000 / 6000. These are exactly what `_FKP_P0_BY_TYPE` held before
commit 740402d replaced them with Table-2's `P0(k=0.14)` — a measured clustering
amplitude, not the convention DESI weights with.

### 3. Footprint is PER TRACER (§54 below)

### Result

| | mean \|err\| | max \|err\| |
|---|---|---|
| before | 0.313% | 0.653% |
| **after** | **0.062%** | **0.121%** |
| reference (Eq. 2.1 on DESI's randoms, §51) | 0.064% | 0.138% |

At the reference. QSO −0.476% → −0.054%, LRG3 +0.370% → −0.055%, ELG2 +0.653%
→ −0.013%.

**N-dependence preserved and stronger**: LRG3 +2.10% across [0.5, 1.5] x N_dr1
(§42 measured +1.22%), ELG2 +0.99%, same sign pattern and ordering. N enters
only through `w_fkp = 1/(1 + nx*alpha(N)*P0)` — `S1` is geometry and a uniform
factor would cancel in a weighted mean regardless.

## §54 — the footprint is per TRACER, not per release

DESI 2024 II Table 2 gives area per tracer class, because priority vetoes remove
sky from lower-priority samples ("a QSO target can remove sky area from lower
priority samples") and imaging vetoes differ:

| | BGS | LRG | ELG | QSO |
|---|---|---|---|---|
| area [deg²] | 7473 | 5740 | 5924 | 7249 |
| ours was | 7500 | 7500 | 7500 | 7500 |
| n̄ correction | 1.004 | **1.307** | **1.266** | 1.035 |

Now `area_deg2` in tracers.yaml, read via `util.tracer_area()`.

**This is why BGS was the outlier in §46, §50 and §51**: BGS (0.996) and QSO
(0.967) barely move, so they never carried the error LRG (0.765) and ELG (0.790)
did. Three "anomalies" and one cause.

Validated against DESI's own `NX` — our `n_eff` over their FKP²-weighted ⟨NX⟩:

| | BGS | LRG1 | LRG2 | LRG3 | ELG2 | QSO | mean | scatter |
|---|---|---|---|---|---|---|---|---|
| before | 0.998 | 0.758 | 0.735 | 0.466 | 0.631 | 0.972 | 0.760 | 24.3% |
| after | 1.001 | 0.991 | 0.961 | 0.629 | 0.802 | 1.010 | **0.899** | **15.5%** |

Confirms §50's n(z) gap was the footprint. LRG3 (0.629) remains open — its n(z)
falls steeply across 0.8–1.1, so the FKP-weighted mean is sensitive to shape.

⚠️ Area is GEOMETRY and is held fixed. `N_tracers` is the design variable and
scales the DENSITY within the footprint; scaling area with N instead would leave
n̄ invariant and the design axis inert. Caveat: the areas are coupled across
tracers by the priority ordering, so a design that changed observing priorities
would move them, and the pipeline would not know.

### What did NOT improve, and why that is expected

`--check shot` ratios move 0.699 -> 0.535 (LRG2). That compares `1/n_eff` to
DESI's `S/A` with `A` the mesh norm we have NOT implemented (§46/§47). We have
corrected the density while the normalization is still ours, so the comparison
is mismatched in a new way. Not a regression: §46 already established the
shot-noise gap is the norm, not n̄.

### Still open

- The mesh norm in the covariance path (§46/§47).
- `fkp_analytic_cov.py:51` `P_FKP_DEFAULT = 1.0e4` is still uniform across
  tracers where Eq. (8.4) wants 7000/4000/6000 for BGS/ELG/QSO.
- The covariance still uses a THIRD n(z) (`load_nz_slices`, N*frac/V) — now
  improved by the area fix but not yet routed through `NX`.
- LRG3's residual density ratio, 0.629.

### Regeneration

z_eff is a per-sample emulator input (§42), so this invalidates the goldens and
BOTH v2 training sets. Do the remaining items above before regenerating, so the
cost is paid once.

## §55 — the FKP pivot and footprint were also uniform in the CONFIG-SPACE path

§53/§54 fixed the z_eff and shapefit paths. The production **BAO config-space**
path — `XiSigmaGenerator`, the emulator's σ-triplet driver — carried the same
two errors independently:

| | was | now |
|---|---|---|
| `_AREA` (config_space.py:56) | 7500 for every tracer | `util.tracer_area(tracer)` |
| `P_FKP` | `P_FKP_DEFAULT = 1.0e4` for every tracer | `_pivot(tracer)` from tracers.yaml |

`P_FKP_DEFAULT = 1e4` is **LRG's** Eq. (8.4) value, applied to all six. Wrong by
1.43x for BGS (7000), 2.5x for ELG (4000) and 1.67x for QSO (6000).

Threaded at four sites: `load_nz_slices` and both `gaussian_xi_multipole_cov`
calls in the bundle builder, plus `XiSigmaGenerator.windowed_cov` — the last of
which feeds production emulator training data, so it mattered most.

`_AREA` survives as a fallback for call sites with no tracer in hand, documented
as such. The synthetic self-test at config_space.py:479 keeps the default
deliberately: it uses a uniform one-slice n(z) with no tracer.

⚠️ This changes the production BAO covariance, and therefore every BAO σ. The
bao goldens must be regenerated and the change measured before the config-space
training data is trusted. Not yet done.

### Consistency check that motivated the area fix

`integral(NX dV)` against `N_tracers`, with DESI's density over DESI's footprint:

| | BGS | LRG1 | LRG2 | LRG3 | ELG2 | QSO |
|---|---|---|---|---|---|---|
| Table-2 areas | 1.014 | 1.003 | 1.020 | 1.017 | **1.173** | 1.003 |
| uniform 7500 | 1.018 | 1.310 | 1.333 | 1.329 | 1.485 | 1.038 |

Five of six within 2%. A triangular check between three independently sourced
quantities — DESI's measured `NX`, DESI's Table 2 area, and our
`targets x comp x efficiency` — so agreement at 2% is meaningful.

**ELG2 (1.173) is now the open one.** Its `analyses` split (ELG1 vs ELG2 within
the 0.8–1.6 catalogue) is the obvious suspect; Table 2's area and counts cover
the full ELG range while our bin is 1.1–1.6 only.

### Still open after this

- The **mesh norm** (§46/§47): validated to 0.03% from randoms alone, still not
  wired into the covariance. Until it is, `--check shot` compares `1/n_eff`
  against DESI's `S/A` and is not a fair test.
- The covariance's **third n(z)**: `load_nz_slices` still builds `N*frac/V`
  rather than using `NX` directly. The area fix moved it from 0.760 to 0.899 of
  DESI's density; routing through `NX` would close the rest.
- **LRG3's 0.629.** Note this compares `n_eff` (a V_eff-matched effective
  density from the Brent solve) against an FKP²-weighted mean of `NX`. For a
  steeply falling n(z) — which LRG3 has across 0.8–1.1 — those are not the same
  functional, so part of the gap may be the comparison rather than the pipeline.

## §56 — §45 overstated: the "floor violation" rests on two untested assumptions

§45 concluded that DESI's covariance "sits below the cosmic-variance floor",
treating that as a hard physical impossibility on their side. That claim is
stronger than the evidence supports, and the balance of evidence now points the
other way.

### The evidence against §45's framing

**DESI's covariance is externally validated; our floor is not.** §45's own chain
showed that our Fisher, using DESI's covariance, reproduces their published σ to
0.56–0.86 — a normal Fisher-at-truth vs MCMC-posterior gap. So their covariance
is self-consistent with constraints that appear in a published paper. Meanwhile
the "floor" is a formula we wrote down. When an externally validated measurement
and an in-house bound disagree, the bound is the more likely suspect.

### The two assumptions

1. **That the shipped covariance carries the same normalization as the shipped
   data vector.** §46 verified the *data vector* is in normalized units (bundle
   P0 / plain P0 = 0.92–1.07 where raw units would demand 0.07–0.14). It never
   verified the *covariance*. The files carry no normalization metadata at all —
   no attributes, just `value` and an observable grid labelled
   `mesh2_spectrum_poles`. §45's own finding that the discrepancy is **flat in
   k** is exactly a normalization signature.

2. **That `N_modes = V k²Δk / 2π²` bounds a cutsky FKP estimator.** That is the
   periodic-box mode count. DESI measures on a padded FFT box (7000 Mpc/h for
   LRG, §10) with FKP weights and a survey window; the mapping from box modes to
   independent modes for a weighted, windowed, cutsky estimator is not that
   formula. §45 applied a textbook periodic-box bound to a measurement it does
   not obviously govern.

### Corrected reading

The covariance ratios (ours 2.28–3.62 × DESI's, §55-era numbers) are **not**
evidence that DESI's covariance is wrong. They are evidence that
**our comparison of covariances rests on an unverified normalization and an
unjustified mode count.** Nothing about the ratio has ever been interpretable,
which is consistent with the fact that no correction has ever moved it: not the
window treatment (§44), the shot noise (§46), the mesh norm (§55, tested — moves
it the wrong way by 1.88x), n(z), the footprint, or the pivots.

### Consequence

The covariance ratio should not be used as a target, and the quarantine §45
imposed stands — but for a different and weaker reason than §45 gave. The
element-level comparison is unusable until (1) is settled, which needs
either normalization metadata we do not have or an independent handle on the
mock covariance's units.

**`--check compressed` remains the usable end-to-end test**, and note it uses
OUR covariance (`our_forecast` with no `cov_override`) against DESI's published
constraints — not DESI's covariance. It is the only comparison here that has
never depended on the disputed normalization.


## §57 — Projection effects do NOT explain the `m` correlation deficit (`mcmc.py`)

> **PARTLY SUPERSEDED by §63.** This section runs one tracer (LRG2), one seed,
> 780 iterations. The 6-tracer × 4-seed × 2500-iteration sweep reverses two of
> its three `m`-correlation results *on LRG2 itself*: ρ(qiso,m) 0.52 → **0.70**
> (here: 0.28) and ρ(qap,m) 0.27 → **0.26** (here: −0.63). The σ results below
> hold 6/6. Read §63 for the numbers; this section for the reasoning that
> motivated the test.

§45–§56 left one open discrepancy that no covariance-level fix has touched:
against DESI's published 4×4, **every ρ involving `m` is under-predicted**
(mean |ρ| ratio 0.356) while **every ρ not involving `m` is over-predicted**
(1.399) — and σ(m) itself is the best-matching of the four σ. Three
explanations were checked and eliminated before this section:

1. the velocileptors swap — already done (§22), we run REPT in production;
2. counterterm prior widths — ours are DESI's N(0, 12.5) exactly (2411.12021);
3. a free `dn` — already fixed, as DESI fix it.

That left **projection effects**. Our targets come from a Fisher matrix, the
Gaussian approximation of the likelihood *at the peak*. DESI's published 4×4
(2411.12021 App. A) is a Gaussian *fit to an MCMC marginal posterior*, and DESI
name both mechanisms that make those different objects (§4.5): the **prior
weight effect** and the **prior volume effect**, the latter explicitly able to
"shift the peak of the marginal posterior away from the most-likely value". A
Fisher matrix can produce neither. If the `m` row is where our likelihood is
most non-Gaussian, marginalisation was the obvious candidate.

### The test

`shapefit/mcmc.py` — emcee over the *same* likelihood and the *same* priors the
Fisher linearises (priors read off the parameter objects, so the comparison
measures the Gaussian approximation and not a prior change), compressed through
the same reduction as `_sf_fisher_reduction`: `f_sigmar = df·f_sigmar_fid`,
`m = dm` (§35), unit Jacobian.

LRG2, REPT, 32 walkers × 780 iterations, 40% burn-in, acceptance 0.24, ~1.8 h.

```
                       Fisher      MCMC      DESI     F/D     M/D
sigma_qiso             0.0150    0.0179    0.0168   0.893   1.067
sigma_qap              0.0497    0.0559    0.0529   0.939   1.056
sigma_f_sigmar_frac    0.1073    0.1078    0.1096   0.979   0.983
sigma_m                0.0668    0.0568    0.0690   0.968   0.823
rho_qiso_qap           0.2789    0.3484    0.2390   1.167   1.458
rho_qiso_f_sigmar      0.0568   -0.1675   -0.0129  -4.390  12.934
rho_qiso_m            -0.1705   -0.0930   -0.3296   0.517   0.282
rho_qap_f_sigmar      -0.6901   -0.7712   -0.5425   1.272   1.422
rho_qap_m             -0.0536    0.1251   -0.2003   0.267  -0.625
rho_f_sigmar_m         0.0746   -0.0297    0.2423   0.308  -0.123
```

### Result: the fourth explanation is eliminated too

**The σ behave as projection effects predict.** Marginalising widens the
posterior relative to the peak curvature, and σ(qiso) and σ(qap) move from
under-predicting DESI to slightly over-predicting (0.893 → 1.067, 0.939 →
1.056); σ(fσ_r) was already right and stays right (0.979 → 0.983). That is the
expected signature, and it confirms the machinery works.

**The `m` row does the opposite.** σ(m) moves *away* from DESI (0.968 → 0.823),
and every ρ involving `m` gets *worse*, two of the three flipping sign:

  - ρ(qiso, m)   −0.171 → −0.093   vs DESI −0.330   (0.517 → 0.282)
  - ρ(qap, m)    −0.054 → **+0.125**  vs DESI −0.200   (0.267 → −0.625)
  - ρ(fσ_r, m)   +0.075 → **−0.030**  vs DESI +0.242   (0.308 → −0.123)

So marginalisation is not the missing ingredient. Whatever couples `m` to the
other three in DESI's posterior is absent from our likelihood *before*
projection, and projection moves us further from it, not closer.

`ρ(qiso, fσ_r)` is worth flagging separately: DESI's is ≈0 (−0.013), so the
ratio column is meaningless there (−4.39, 12.93 are division by near-zero, not
a 13× error). The absolute values are 0.057 → −0.168 against −0.013; small
numbers, but the MCMC does move it well past DESI's.

### Caveats

**One seed.** `feedback_mcmc_chain_noise_sparse_tracers` and the bao CHANGELOG
both record that per-seed scatter is what bites, worst for the sparse tracers.
No seed sweep has been run, so the third decimal of every MCMC number above is
untrusted. The *sign flips* in ρ(qap, m) and ρ(fσ_r, m) are large enough that
seed noise is unlikely to be the whole story, but that has not been shown.

**One tracer.** LRG2 only, because the run is ~1.8 h and the log-prob closure is
not picklable, so there is no multiprocessing pool yet.

### Status

The `m` correlation deficit is now the one substantive open discrepancy in the
shapefit validation, with four explanations eliminated: theory (§22), priors,
`dn`, and projection (here). It does not block the emulator — `m` is a target we
predict and σ(m) matches DESI to 0.968 under the production Fisher path — but it
means our 4×4's `m` row should not be treated as validated at the ρ level.


## §58 — the per-tracer footprint never reached the generators

§54 replaced the uniform 7500 deg² footprint with DESI 2024 II Table 2's
per-tracer areas (BGS 7473, LRG 5740, ELG 5924, QSO 7249). The fix went into
`core.build_shapefit_likelihood`, which resolves

```python
area = float(area) if area is not None else tracer_area(tracer_bin, dataset)
```

— a **fallback**, applied only when the caller passes nothing. Six call sites
were passing something: `dataset_area("dr1")` = 7500. They therefore overrode
the correction and went on running the old geometry.

### The split

| path | area used | affected |
|---|---|---|
| `our_forecast` (`--check compressed`, every comparison plot) | `tracer_area` | no |
| `mcmc.py` (§57 and the seed sweep) | `tracer_area` | no |
| `benchmark_desi.py` | `tracer_area` | no |
| **`generate_covar_data.py`** | 7500 | **yes** |
| **`generate_mean_data.py`** | 7500 | **yes** |
| **`regress_sigmas.py`** (`_AREA`) | 7500 | **yes** |
| `validate_forecast.py` (3 sites) | 7500 | yes |
| `compare_to_desi.py` (analytic-cov sites) | 7500 | yes |

So **the pipeline we validate was not the pipeline we generate training data
with**. Every number quoted against DESI in §54–§57 came from the corrected
path; every training label came from the uncorrected one.

### Cost, measured

Kaiser Fisher at the DESI fiducial and the DR1 count, area 7500 vs per-tracer,
everything else fixed:

```
LRG2   7500 -> 5740     sigma_qiso   -8.4%   sigma_qap   -8.7%
                        sigma_f_sr   -6.2%   sigma_m    -10.3%
ELG2   7500 -> 5924     sigma_qiso   -1.2%   sigma_qap   -1.9%
                        sigma_f_sr   -0.4%   sigma_m     -4.5%
QSO    7500 -> 7249     all four     <0.5%
```

**The sign is the one that matters and it is not the obvious one.** A bigger
area dilutes n̄ = N/V at fixed N, which raises shot noise — but it also buys
volume, and the volume wins. So the 7500 made the forecast *tighter*: the
emulator was being taught that DESI is **more** constraining than it is, by
6–10% on the three LRG bins, ~2–5% on ELG2, and negligibly on BGS/QSO.

That per-tracer pattern is the same one that ran through §46–§51, where BGS and
QSO kept behaving differently from LRG and ELG. Their areas were already nearly
right (0.996 and 0.967 of 7500); LRG's and ELG's were not.

### z_eff is untouched, and that is why this survived §53

Under `Z_EFF_CONVENTION = "desi_eq21"` the area **cancels** out of z_eff: the
per-slice weight carries `1/V_bin` and `V_bin ∝ area`, so it divides out of the
ratio. Measured z_eff is identical to eight decimals at either area. §53's
0.062% agreement was therefore never evidence that the footprint was right
anywhere — it could not have been.

### The consistency check was blind by construction

`validate_forecast.py`'s z_eff covar-vs-mean check carried this rationale:

> Note `area` is deliberately NOT passed to run_fisher [...] Forcing them equal
> here would hide a drift between those two routes to the footprint.

The intent is right and it did not work. The two routes *were* drifted — 5740
on the covar side, 7500 on the mean side, a factor 1.31 — and the check passed,
because the only quantity it compares is the one the area cancels out of. A
test that watches for drift in a variable its observable is independent of
cannot fail. Both sides now resolve `tracer_area`, and the docstring says
plainly that this is not a footprint test.

### Fix

All six call sites resolve `util.tracer_area(tracer, dataset)`; `--area` still
overrides. `regress_sigmas._AREA` became `_area(tracer)` — a per-tracer
footprint cannot be a module constant, and freezing one was the same mistake as
the bao golden pinning z_eff (commit 19dc4b3): a harness that freezes an input
stops testing the code that derives it.

### Consequence

The golden baseline must be regenerated, and both v2 training sets were already
invalidated by §53–§55 — this adds a reason and does not change that verdict.
**Regenerate with the fix in, or the regeneration inherits the bug.**

Nothing in §54–§57 is retracted: those numbers all came from `our_forecast` or
`mcmc.py`, both of which were already correct.


## §59 — shapefit mean eval was raising on every sample

`util.get_pipeline(analysis="shapefit", quantity="mean")` builds the ground
truth `eval.py` scores the mean emulator against. It called

```python
sf_fs._worker_run_mean_targets((sample, _tracer, _z, None))
```

with a **4-tuple**, against a worker whose task contract is **6 fields**:

```python
(sample, tracer_bin, z_eff, param_defaults, area_deg2, dataset) = args_tuple
```

Verified by calling it:

```
target_names: ['qiso', 'qap', 'f_sigmar', 'm']
CRASH: ValueError not enough values to unpack (expected 6, got 4)
```

So `eval.py --analysis shapefit --quantity mean` could not have scored a single
sample. And it failed in the worst available way: the unpack sits *above* the
worker's `try`, so instead of the `(None, None, traceback)` its contract
promises — which the caller checks for and turns into a clean RuntimeError —
it raised a bare `ValueError` out of the worker.

### How it drifted

The worker grew `area_deg2` and `dataset` when §42 made z_eff depend on the
sampled cosmology and on N_tracers. Every other caller was updated;
`get_pipeline` was not. The plan's step-8 round trip (`train.py` → `eval.py`
for `--quantity mean`) predates §42, so nothing re-exercised it afterwards.
This is the third instance of the same failure mode, after §30 ("changing a
default broke callers of the old one") and §58 — a shared contract changing
under a caller that no test covers.

### The second bug underneath it

The call also pinned `z_eff` to one fiducial value, computed once per tracer:

```python
z_eff = gen_mean._fiducial_z_eff(tracer_bin, sf_core.dataset_area("dr1"))
```

That is the pre-§42 convention. `generate_mean_data.py` passes `z_eff=None` so
the worker derives it *per sample*. Fixing only the arity would have produced a
silently wrong answer instead of a crash: eval would score the emulator against
labels evaluated at a different redshift from the ones it was trained on. The
crash was, in this one respect, lucky.

Its area argument was `dataset_area("dr1")` = 7500, the §58 bug again.

### Fix

Pass the full 6-tuple with `z_eff=None` (derive per sample, as the generator
does) and `tracer_area(tracer_bin, "dr1")`. The `generate_mean_data` module
load exists only for `_fiducial_z_eff` and is dropped.

Verified after the fix: `qiso = qap = 1` and `m = 0` at the DESI fiducial (the
identity §38–§41 pin), and all four move under `omega_cdm +5%`.


## §60 — §17's correction never reached `f_sigmar`

§17 established that DESI's σ must be divided by the **fiducial**, not by the
measured central value, because DR1's measurement sits up to 5–6% off its
fiducial and using it "inflated DESI's σ and so deflated every ratio". It fixed
`sigma_qiso` and `sigma_qap`. It left the third parameter on the measurement:

```python
"sigma_f_sigmar_frac": float(sig[2] / vec[2]),   # vec[2] = MEASURED f sigma_s8
```

Our side has always divided by our own fiducial (`compare_to_desi.py`,
`f_sigmar_fid`), so the comparison carried a tracer-dependent skew of
measured/fiducial — and here that factor is **larger** than the one §17 fixed,
because fσ8 is measured far less precisely than D_V/r_d:

| tracer | measured fσ8 | fiducial (Table 11) | meas/fid |
|---|---|---|---|
| BGS | 0.3772 | 0.4723 | **0.799** |
| LRG1 | 0.5136 | 0.4733 | 1.085 |
| LRG2 | 0.4836 | 0.4608 | 1.049 |
| LRG3 | 0.4222 | 0.4398 | 0.960 |
| ELG2 | 0.3767 | 0.3944 | 0.955 |
| QSO | 0.4349 | 0.3750 | **1.160** |

−20% to +16%, against 0.94–1.04 for q.

### Why the fiducial is the right denominator here too

Not merely by analogy with §17. Our mean pipeline returns **f_sigmar = 0.460725
at the LRG2 fiducial** against Table 11's **0.4608** — the same number to 0.02%.
So "our fiducial" and "DESI's fiducial" are not two conventions, they are one
quantity, and dividing each side by it compares σ against σ directly. It also
absorbs the small z_eff offset between the two sides in a controlled way, which
dividing by the measurement does not: that mixes the z offset with DR1's
fluctuation.

### Effect on the reported agreement

`--check compressed`, REPT, ours/DESI on the fσ_r column:

| tracer | before | after |
|---|---|---|
| BGS | 0.67 | **0.84** |
| LRG1 | 1.11 | 1.02 |
| LRG2 | 0.98 | 0.93 |
| LRG3 | 0.77 | 0.80 |
| ELG2 | 0.54 | 0.57 |
| QSO | 0.74 | **0.64** |

Mixed, not uniformly favourable: BGS improves sharply, QSO degrades sharply.
That is the expected signature of removing a data fluctuation rather than of
tuning. The six ρ are untouched — a ratio to a constant leaves ρ invariant,
verified (`rho_qiso_m` LRG2 = −0.329629 before and after).

Note the MCMC sweep's JSON files carry a `"desi"` block computed before this
change, so their printed `M/DESI` for f_sigmar is stale by the factors above.
The plots read `desi_reference` live and are correct.


## §61 — the pipeline's two n(z) definitions, and ELG2's 15%

With §58's per-tracer footprint in, the two independent n(z) the pipeline
carries can finally be compared without the area confounding them:

* the **covariance** builds `nbar_i = N_tracers × frac_i / V_i` — the design
  count over slice fractions over comoving shell volume. This is the one
  `N_tracers` acts on, so it is the design axis.
* **z_eff** uses DESI's `NX` column from their randoms (§51–§53),
  `n(z)⟨C_assign⟩`, the density their own `WEIGHT_FKP` is built from.

Nothing in the code forces these to agree. Per-slice ratio of the first to the
second, and the same statement as an implied galaxy count (∫NX dV over the
slices, against the count we feed in from `desi_data.csv`):

| tracer | per-slice mean | N (desi_data.csv) | N implied by NX | ratio |
|---|---|---|---|---|
| BGS | 1.003 | 3.000e5 | 3.042e5 | 0.986 |
| LRG1 | 0.997 | 5.069e5 | 5.082e5 | 0.997 |
| LRG2 | 0.980 | 7.719e5 | 7.876e5 | 0.980 |
| LRG3 | 0.979 | 8.598e5 | 8.749e5 | 0.983 |
| **ELG2** | **0.850** | **1.416e6** | **1.660e6** | **0.853** |
| QSO | 0.998 | 8.567e5 | 8.591e5 | 0.997 |

Five of six agree to within 2%. ELG2 is 15% low: DESI's randoms imply 1.66M
ELG2 galaxies in 1.1 < z < 1.6, and we feed the pipeline 1.416M.

This is the same discrepancy as the old `∫NX dV/N` check's 1.173 (= 1/0.853),
now isolated to one tracer because the footprint is no longer confounding it.

Two candidate causes are eliminated:

* **not fraction normalisation** — every tracer's `frac` sums to exactly 1.0000.
* **not an ELG1/ELG2 slice mixup** — ELG2's slices span 1.100–1.600, the right
  bin (§31/§32).

What remains is on the `desi_data.csv` side: whether that ELG2 count is after a
cut `NX` does not carry. That needs the catalogue, not more arithmetic.

**It does not explain ELG2's tight σ, and points the wrong way.** ELG2 is our
worst σ agreement (fσ_r 0.57, qap 0.59). Raising N to the randoms-implied 1.66M
raises n̄ by 17%, which *lowers* our σ and makes the ratio worse, not better.
Whatever is behind ELG2's tightness, this is not it.

### §61a — where ELG2's 1.416M comes from, and what it is not

`ntracers("ELG2", "dr1")` is a literal table lookup. ELG2 declares no
`components` in tracers.yaml, so `util.ntracers` falls through to the `passed`
column of `~/data/desi/bao_dr1/desi_data.csv`:

```
tracer,passed,observed,z,efficiency,...
ELG2,1415707.0,1947327.372764787,1.317,0.727,...
```

and `1947327 × 0.727 = 1415707` exactly — `passed = observed × efficiency`,
with 0.727 the ELG redshift-success rate (DESI 2024 II Table 2).

**The obvious hypothesis is wrong.** If DESI's `NX` came from randoms matched to
the successful-redshift catalogue, applying the efficiency again would
under-count. QSO refutes it: its efficiency is 0.668, *lower* than ELG's, and

```
tracer   passed     observed    eff     NX implied   NX/passed   NX/observed
BGS      3.000e5    3.034e5    0.989    3.042e5        0.986        0.997
LRG1     5.069e5    5.115e5    0.991    5.082e5        0.997        1.007
LRG2     7.719e5    7.789e5    0.991    7.876e5        0.980        0.989
LRG3     8.598e5    (components)   -    8.749e5        0.983          -
ELG2     1.416e6    1.947e6    0.727    1.660e6        0.853        1.173
QSO      8.567e5    1.282e6    0.668    8.591e5        0.997        1.493
```

QSO's `NX/passed = 0.997` and `NX/observed = 1.493`: the efficiency is applied
exactly once, correctly, at an efficiency as low as ELG's. So this is not a
convention error in the pipeline — it is specific to ELG2, which matches
*neither* column (0.853 of `passed`, 1.173 of `observed`).

**The live hypothesis: a z-independent efficiency on a z-dependent quantity.**
Inverting, DESI's randoms imply an effective ELG2 success of 1.660/1.947 =
**85.3%** against the 72.7% applied. And `desi_tracers.csv` gives ELG1 and ELG2
the *same* efficiency and completeness:

```
ELG1,ELG,0.727,0.352,...
ELG2,ELG,0.727,0.352,...
```

0.727 is the ELG **sample average**, applied uniformly to both bins. ELG
redshift success is strongly z-dependent ([OII] moving through the sky-line
forest), so a shared average necessarily under-counts one bin and over-counts
the other. An ELG2 true success near 85% with ELG1 correspondingly worse
reproduces the measurement exactly.

**Not tested, and not testable locally.** Confirming it needs `∫NX dV` over
ELG1's 0.8–1.1 range, and no `ELG1_desi_nx.csv` exists — only ELG2's. The ELG
randoms over that range would have to come from CFS (§43's recipe). Until then
this is a hypothesis with one supporting coincidence, not a result.

**Why it matters beyond a bookkeeping error.** `N_tracers` is the emulator's
design variable, and `ntracers_range` centres the training box on `passed`
(`low 0.5`, `high 1.5`). If ELG2's DR1 anchor is 15% low, the entire ELG2 design
box is centred on the wrong point — not just one evaluation. That is a
different and worse failure than a mis-scaled forecast.

### §61b — RETRACTION: the ELG2 count is DESI's own, the NX integral is the suspect

§61 and §61a put the ELG2 discrepancy on the `desi_data.csv` side — "whether
that ELG2 count is after a cut NX does not carry", and a hypothesis that 0.727
under-counts ELG2. Both are wrong. Our counts are DESI's published values:

| tracer | ours | DESI 2024 V Table 1 | rel |
|---|---|---|---|
| BGS | 300017 | 300017 | 0 |
| LRG1 | 506911 | 506905 | 1.2e-05 |
| LRG2 | 771894 | 771875 | 2.5e-05 |
| LRG3 | 859822 | 859824 | −2.3e-06 |
| **ELG2** | **1415707** | **1415687** | **1.4e-05** |
| QSO | 856652 | 856652 | 0 |

(also in DESI 2024 II Table 2). Every tracer agrees to ≤2.5e-5. `passed =
targets × comp × efficiency` reproduces the published counts exactly, ELG2
included — so the 0.727 shared-efficiency hypothesis in §61a is dead: if it
mis-split ELG1/ELG2, the ELG2 total would not land on DESI's number to 20
galaxies.

**The suspect is `∫NX dV`, which over-predicts by 17%.**

### The live hypothesis, with the ordering that supports it

`NX` is `n(z)⟨C_assign⟩` (2411.12020 Eq. 8.3) — already completeness-weighted —
and `{tracer}_desi_nx.csv` stores a **randoms-weighted mean** of it per slice.
Recovering N by `Σ NX_i V_i` over the geometric footprint is only exact when
⟨C_assign⟩ is uniform across that footprint. Sorted by assignment completeness:

| tracer | comp | NX-implied / N |
|---|---|---|
| QSO | 0.874 | 0.997 |
| LRG | 0.693 | 0.980–0.997 |
| BGS | 0.636 | 0.986 |
| **ELG** | **0.352** | **0.853** |

Monotonic. ELG's fibre assignment is both the lowest (35.2%, it is the lowest-
priority target class) and the most spatially variable, so the randoms-weighted
mean departs furthest from the volume average. The other five sit at 64–87% and
land within 2%.

### What this changes

* **The training input is not wrong.** `N_tracers` for ELG2 is DESI's published
  count, so the design box is centred correctly. §61a's closing claim — that the
  entire ELG2 box is miscentred — is **withdrawn**.
* **z_eff is what consumes NX**, and it is a ratio, so a uniform 17% scale error
  in NX largely cancels there. That is consistent with ELG2's z_eff agreeing
  with DESI's published value to 0.062% (§53) despite this.
* What remains open is whether the per-slice SHAPE of NX is also affected, which
  would not cancel in z_eff. Testing that needs the raw randoms, not the
  aggregated table.

**Nothing about ELG2's tight σ is explained by this**, and the §61 note stands:
the covariance's n̄ comes from `N_tracers × frac / V`, which uses the correct
count, not from NX.

### §61c — RESOLVED: there is no ELG2 bug; two different densities were being compared

The DR1 LSS clustering catalogues are on the **public** server, so this needed
no NERSC access at all (see the note at the end). With them local, the ELG2
question closes completely — and not in favour of any of §61/§61a/§61b.

### Three measurements

**1. The counts are exact, for the third time.** Catalogue rows in range:
ELG2 **1415707** against `util.ntracers` 1415707; LRG2 **771894** against
771894. Paper (§61b), CFS read (§46), and now the local catalogue all agree.

**2. The per-slice SHAPE of NX is correct.** §61b left this as the open item.
Comparing NX against the galaxy counts slice by slice, on the same objects:

```
ELG2  NX/count shape ratio: min 0.9931  max 1.0099  spread 1.7%
LRG2                        min 0.9948  max 1.0022  spread 0.7%
```

`frac(csv)`, `frac(count)` and `frac(NX)` agree column for column. The ~9% tilt
seen in the aggregated table is not in NX itself. **z_eff is not distorted.**

**3. The footprint is right, measured without randoms.** Galaxies sample volume
with density NX, so `Σ_gal 1/NX = ∫(1/NX)·NX dV = V` — the catalogue measures
its own effective volume:

| tracer | area used | V_NX/V_geom | implied area |
|---|---|---|---|
| ELG2 | 5924 | 1.0163 | 6021 |
| LRG2 | 5740 | 0.9878 | 5670 |
| LRG1 | 5740 | 1.0082 | 5787 |
| LRG3 | 5740 | 1.0076 | 5784 |

All within 1.6%. §58's per-tracer areas are independently confirmed, ELG2's
included.

### The resolution

`∫NX dV = N_gal` **by definition** if NX is the galaxy density, so the CSV's
1.66M was never a physical prediction to be reconciled — it is what a
*randoms-weighted mean* of NX integrates to, which is not the volume average.
The two differ by the within-slice variance of NX, largest where completeness is
most variable:

```
csv/(N_slice/V_slice):   ELG2 mean 1.1759 (1.150-1.214)
                         LRG2 mean 1.0197 (1.010-1.034)
```

**But the randoms-weighted quantity is the correct one for z_eff.** DESI 2024 III
Eq. (2.1) weights by the *squared weighted random density*, not by a volume
average. Swapping in the volume average degrades z_eff against DESI's published
values:

```
mean |err| vs DESI:   NX (current) 0.062%     N*frac/V  0.273%
                      (BGS is the driver: +0.121% -> +1.346%)
```

So both quantities are right in their place, and the 0.850 ratio §61 opened with
was comparing two things that were never supposed to be equal. **No code change
is warranted.** §61's "ELG2 is 15% low", §61a's efficiency hypothesis and
§61b's "the NX integral is the suspect" are all withdrawn.

### What is actually open

The **covariance** takes its shot-noise n̄ from the volume average
(`N_tracers × frac / V`). For an FKP-weighted estimator on a variably-complete
sample, the density entering the shot-noise floor is plausibly the *weighted*
one, which for ELG2 is 17.6% higher. ELG2 is also our worst σ agreement (fσ_r
0.57, qap 0.59), and raising shot noise raises σ — the right direction. This is
a hypothesis with a suggestive direction, not a result; it needs the FKP
shot-noise algebra done properly (§46's Eq. 10.5/10.6 route), not another ratio.

### Operational note

`/global/cfs/cdirs/desi/public/dr1/...` is mirrored at
`https://data.desi.lbl.gov/public/dr1/...` and is directly `curl`-able. The four
clustering catalogues used here are 460 MB total and downloaded in under a
minute with no MFA, no tty and no pull-from-entropy. §43's ssh recipe is only
needed for non-public paths. Filenames use `NGC`/`SGC`, not `N`/`S`.


## §62 — `nz_slices` `frac` comes from non-DR1 n(z) tables

Downloading the remaining four catalogues (BGS, QSO — the two skipped when the
download was scoped to the ELG2 hypothesis) completed the checks and turned up
something the ELG/LRG pair could not have shown.

### First, two corrections to §61c's method

**NX is correct for all six tracers.** §61c compared NX against *unweighted*
galaxy counts. DESI's n(z) is weighted, and with `WEIGHT` applied:

| tracer | NX vs count | NX vs weighted count |
|---|---|---|
| BGS | 29.3% | **2.1%** |
| LRG1 | 1.5% | 0.5% |
| LRG2 | 2.6% | 0.7% |
| LRG3 | 5.7% | 0.7% |
| ELG2 | 3.1% | 1.7% |
| QSO | 5.4% | 0.4% |

(spread of the per-slice shape ratio). BGS's apparent 29% anomaly was an
artifact of the unweighted comparison — its weights are strongly z-dependent.
NX matches the weighted galaxy density to ≤2.1% everywhere.

**§58's per-tracer areas are now confirmed for all six**, via `Σ 1/NX = V`:

| tracer | area used | implied | ratio |
|---|---|---|---|
| BGS | 7473 | 7489 | 1.002 |
| LRG1/2/3 | 5740 | 5670–5787 | 0.988–1.008 |
| ELG2 | 5924 | 6021 | 1.016 |
| QSO | 7249 | 7303 | 1.008 |

Counts also reproduce §46 exactly, including its two known offsets (BGS +26,
QSO +179; the other four exact).

### The finding

Our stored per-slice `frac` — which the **covariance** uses to build
`nbar_i = N_tracers × frac_i / V_i` — does not come from the catalogues. The
`source_file` column names DESI's published `*_nz.txt` tables, and those
describe a substantially larger sample than DR1:

| tracer | Σ Nbin (file) | DR1 N | ratio | file_area_deg2 |
|---|---|---|---|---|
| BGS | 968857 | 300017 | 3.23 | 24710 |
| LRG1 | 1209217 | 506911 | 2.39 | 20062 |
| LRG2 | 1954607 | 771894 | 2.53 | 30092 |
| LRG3 | 2087501 | 859822 | 2.43 | 20062 |
| ELG2 | 5837681 | 1415707 | 4.12 | 20704 |
| QSO | 1548236 | 856652 | 1.81 | 11181 |

1.8–4.1× the DR1 counts over areas of 11k–30k deg², none matching DR1's
per-tracer footprints. These are final-survey/forecast n(z) tables, not DR1.

Only the *shape* is consumed (`slice_fraction`; the normalisation comes from the
DR1 `N_tracers`), so this is not a factor-of-3 error. But the shape is not DR1's:

| tracer | `frac` vs DR1 weighted n(z) |
|---|---|
| **BGS** | **13.6%** |
| QSO | 8.0% |
| ELG2 | 4.9% |
| LRG3 | 3.3% |
| LRG1 | 3.0% |
| LRG2 | 2.4% |

### Status and what it touches

`frac` feeds the covariance's per-slice n̄, hence the FKP weight per slice, hence
V_eff and σ. It is **not** the z_eff path, which uses NX (confirmed correct).

Worth flagging without asserting a link, given how many hypotheses this thread
has already burned: BGS has the worst `frac` shape error (13.6%) **and** our
worst σ agreement (qiso 0.61); QSO is second on both (8.0%, 0.77). LRG2, best on
shape (2.4%), is also our best-agreeing bin (0.89–0.97). That ordering is
suggestive and now **testable**, because the DR1 catalogues are local and `frac`
can be regenerated exactly rather than inherited from a forecast table.

Not done here: regenerating the tables. That changes every training label and
the golden, so it belongs with the §53–§58 regeneration, not bolted on at the
end of a session.

### §62a — measured: the `frac` shape error costs ≤0.15% in σ

§62 flagged the ordering "BGS worst on `frac` shape (13.6%) and worst on σ
(qiso 0.61)" as suggestive and testable. Tested, by swapping `frac` for the DR1
catalogue's weighted n(z) and rerunning the Kaiser Fisher:

```
BGS    z_eff 0.2954 -> 0.2954     sigma_qiso/qap/f_sigmar/m   all +0.01%
QSO    z_eff 1.4902 -> 1.4902                                 +0.12 .. +0.15%
LRG2   z_eff 0.7058 -> 0.7058                                 +0.04 .. +0.09%
```

**The correlation is a coincidence.** A 13.6% shape error on BGS moves its σ by
0.01% — three orders below the 0.39 it would need to explain qiso 0.61.

The reason is structural: the slice n̄ is not used directly. It drives per-slice
HOD b1 and the band-averaged FKP² weight, which the Brent root-find then maps
back onto a *single* `n_eff` at z_eff (core.py's V_eff block). A redistribution
across slices that preserves the total is exactly what that mapping averages
over. z_eff is untouched for the separate reason that it consumes NX, not
`frac`.

### Consequence

`nz_slices` regeneration is **not** required for accuracy. §62's provenance
finding stands as a description — the tables really do come from final-survey
n(z) — but it is a tidiness issue, not a correctness one, and it should not be
bundled into the §53–§58 regeneration as if it mattered. If it is ever done, the
recipe is now trivial: the DR1 catalogues are local and public.

This closes the third hypothesis this thread produced about ELG2/BGS n(z), all
three refuted by measurement (§61c, §61c's covariance note, §62). The open item
from §61c — whether FKP shot noise wants the weighted density rather than the
volume average — is *not* addressed by this test, which only redistributed
`frac` at fixed definition. That one is still open.

### §62b — tested in BAO and across N, then rebuilt the tables anyway

§62a's "≤0.15%" was measured on **shapefit Fisher, fiducial cosmology,
N = N_dr1** and stated as though it covered everything. It did not. Two gaps,
both fair:

* **BAO config-space uses the slices differently.** There the per-slice n(z)
  feeds the Gaussian ξ covariance directly (`bao/config_space.py:696`), with no
  `n_eff` root-find to average over it — so §62a's compression argument, which
  is the whole reason shapefit was insensitive, does not apply.
* **Only one N was tested.** `N_tracers` is the design axis and the FKP weight
  `1/(1+n̄P₀)` is non-linear in n̄, so the error need not wash out equally across
  the 0.5×–1.5× box.

### Both tested

Swapping `frac` for the DR1 catalogue's weighted n(z):

```
BAO config-space (sigma_triplet)
  LRG2  (2.4% shape change)    <0.005%   at N = 0.5, 1.0, 1.5x
  BGS   (13.6% shape change)   -0.07 .. -0.08%,  flat in N

shapefit Fisher
  BGS    +0.04% / +0.01% / -0.00%   at N = 0.5 / 1.0 / 1.5x
  QSO    +0.16% / +0.15% / +0.15%
```

The `*_rd_fid` columns move by exactly 0.00%, which is the control: they are
fiducial values, not σ, and confirm the comparison is well-formed rather than
the patch silently failing.

So the conclusion holds — **≤0.16% in both analyses, flat across the design
range** — but it needed both tests before it was worth asserting, and the
non-linearity concern is answered rather than assumed.

### Rebuilt regardless

An input being immaterial is not a reason to leave it wrong.
`shapefit/make_nz_slices.py` regenerates the tables from the DR1 catalogues.
Two details that are not free:

* **Slice edges are preserved exactly.** `bao/core._desi_nz_geometry`
  length-checks `{tracer}_desi_nx.csv` against the post-`>0` slice count and
  *silently* falls back to `nbar_file` on a mismatch — which would revert z_eff
  to the pre-§53 convention with nothing raising. Verified after the rebuild:
  15/10/10/15/25/65 rows, matching the NX tables exactly.
* **`slice_fraction` is WEIGHT-weighted**, not a raw count fraction. DESI's n(z)
  is weighted, and for BGS the two differ by 29% (§62).

Installed for the six full-shape bins (backups at `*.prefinal.bak`). Shape change
vs the shipped tables: BGS 13.6%, QSO 8.0%, ELG2 4.9%, LRG3 3.3%, LRG1 3.0%,
LRG2 2.4%.

**`LRG3_ELG1` deliberately NOT installed.** It rebuilds to a 69.7% shape change
with `N_cat = 2898381` against `N_dr1 = 1876187` (1.54×), so the combined
`LRG+ELG_LOPnotqso` catalogue over 0.8–1.1 is not the selection behind DESI's
BAO bin count. It is also the one bin with no NX table, making `nbar_file` live
rather than inert there. BAO-only, so it blocks nothing here; do not swap it in
without resolving the 1.54× first.

### Verification after install

* z_eff vs DESI 2024 V Table 1: mean |err| **0.062%**, per-tracer values
  unchanged to four decimals. Expected — z_eff consumes NX, not `frac`.
* `tests/test_fiducial_identity.py`: 2 passed.
* Slice row counts match the NX tables for all six.

Training labels move by ≤0.16%, far below the emulator's own error, so this does
not by itself force a regeneration — but it is in place for the §53–§58 one.

### §62c — OPEN: the n(z) layer is not release-scoped

Deferred deliberately, recorded so it is not lost.

Every other part of the geometry stack takes a `dataset`: `util.ntracers`,
`tracer_area`, `ntracers_range`, `get_default_save_path`, `tracers.yaml`'s
per-release overrides (§32), even `bao/core._nz_scale_factor`. The n(z) tables
do not:

```
_NZ_SLICES_DIR = ~/data/desi/nz_slices        # flat, no dataset segment
  {tracer}_nz_slices.csv   {tracer}_desi_nx.csv

_load_nz_slice_fractions(tracer_bin)          # no dataset arg
_desi_nz_geometry(tracer_bin, nbar_file)      # no dataset arg
load_nz_slices(..., nz_slices_dir=None)       # dir override only
```

`make_nz_slices.py` (§62b) does not fix this — it is hardcoded DR1 in the
catalogue dir, the areas, the counts and the download URL (`public/dr1/.../
iron/LSScats/v1.5/`).

**Risk.** With DR2 present, `ntracers` and `tracer_area` would switch release
while `_load_nz_slice_fractions` silently returned DR1 shapes — a mixed-release
covariance with nothing raising. Same class as §58 (a fallback that never fired)
and §59 (a caller never updated).

**Fix when picked up:** (1) path convention
`~/data/desi/nz_slices/{dataset}/{tracer}_*.csv`, flat layout as a deprecation
fallback; (2) thread `dataset` through the three functions above and their ~6
call sites in `bao/` and `shapefit/`; (3) `make_nz_slices.py --dataset` with a
release → (catalogue dir, LSS version) map, **stubbed to fail loudly** rather
than silently return DR1.

Not urgent while the pipeline is DR1-only (`--dataset` is `choices=["dr1"]`),
and (3) is DR2 work, which the DR1-first rule defers. (1)+(2) are defensive and
want a deliberate caller audit — signature changes on functions shared by both
analyses are exactly what produced §30, §58 and §59.


## §63 — the projection question, at 6 tracers × 4 seeds (supersedes §57)

§57 tested whether projection effects explain the `m` correlation deficit using
one tracer, one seed and 780 iterations, and concluded they do not — "the
fourth explanation is eliminated too". It flagged its own weakness ("the third
decimal of every MCMC number above is untrusted"). The full sweep shows the
*first* decimal was untrusted as well, and the real answer is a split rather
than an elimination.

### The run

`run_mcmc_sweep.sh`, 24 jobs (6 tracers × seeds 42–45), one process each with
BLAS pinned to 1 thread, 32 walkers × 2500 iterations, 40% burn-in →
48 000 samples per chain. 409.7 min wall, zero failures, 24/24 JSONs.

Three `mcmc.py` changes made it runnable and auditable:

  - **per-seed chain files.** The sweep loop wrote every seed to the same
    `--save` path, so a 4-seed run kept one chain. Now `…_seed{N}.npz`.
  - **progress ticks at iterations 10/25/50 then every 5%**, so a wrong
    iteration budget shows up in minutes instead of after the first 5% of a
    multi-hour run.
  - **`diag` block** (acceptance, `tau_max`, `n_samples`, `niter`, `nwalkers`)
    serialised to the JSON — which is what exposed the convergence problem
    below.

### Convergence: these chains are ~5× short

```
tracer   niter   tau_max      niter/tau     acceptance
BGS      2500    213-247      10.1-11.8     0.207
LRG1     2500    224-259       9.6-11.2     0.205
LRG2     2500    221-240      10.4-11.3     0.196
LRG3     2500    235-280       8.9-10.6     0.164
ELG2     2500    226-243      10.3-11.1     0.213
QSO      2500    234-266       9.4-10.7     0.192
```

emcee's guidance is `niter > 50τ`. Every chain is at **9–12 τ**. The seed rms
below (0.03–0.17 in absolute ρ, against ρ values of 0.1–0.4) is the visible
consequence: individual cells are not resolved. Statements that survive across
all 24 chains are usable; single cells are not. **A definitive `m`-row result
needs ≈12 500 iterations** — ~34 h at this throughput, or under 6 h by widening
to more walkers across the idle cores.

### Result: two of three `m` correlations improve, one gets worse

Ratio to DESI, averaged over the six tracers, signed (so a sign flip shows as a
negative, not as spurious agreement):

```
                  F/D   ->   M/D     sign matches DESI
rho_qiso_m       0.625 ->  0.712      6/6 -> 6/6      improves
rho_qap_m        0.300 ->  0.551      6/6 -> 6/6      improves (5/6 tracers)
rho_f_sigmar_m  -0.196 -> -0.115      4/6 -> 1/6      WORSE, sign flips
```

`ρ(qap,m)` is the clean one: DESI is negative on all six tracers, we are
negative on all six under both estimators, and marginalising nearly doubles the
magnitude toward DESI. `ρ(qiso,m)` improves on 4 of 6.

`ρ(fσ_r,m)` moves decisively the wrong way. DESI has it at **+0.23 to +0.37**
on five tracers; our Fisher has it at ≈0 (+0.00 to +0.07) and marginalising
drives it *negative* on five of six. Under an absolute-value convention that
reads as improvement (0.377 → 0.553) because |ρ| grows — but it grows with the
wrong sign, which is not agreement. **This is why §63 reports signed ratios.**

### A statistic to not use

A first pass at this section quoted "mean ratio to DESI, Fisher 0.243 → MCMC
0.382" as evidence that projection helps. That is the signed mean over all 18
`m`-involving ρ, **including BGS `ρ(fσ_r,m)` where DESI's value is −0.0144**.
Dividing by it gives −1.93 (Fisher) and +1.13 (MCMC): one entry contributing a
0.17 swing to an aggregate that moved 0.14. The entire "improvement" was that
denominator.

§57 already named this trap for `ρ(qiso,fσ_r)` ("the ratio column is
meaningless there ... division by near-zero, not a 13× error") and it was walked
into anyway one section later. **Any ρ aggregate in this pipeline must exclude
|ρ_DESI| < 0.05 and must be signed.** Under that rule the m-involving mean is
0.371 → 0.338 — i.e. flat, and the honest summary is the per-correlation split
above, not a single number.

### σ: §57's one-tracer result generalises cleanly

```
sigma_qiso            0.793 -> 0.868      toward DESI
sigma_qap             0.854 -> 0.911      toward DESI
sigma_f_sigmar_frac   0.801 -> 0.849      toward DESI
sigma_m               0.916 -> 0.759      AWAY, 6/6 tracers
```

Marginalising widens the posterior relative to the peak curvature, so three of
the four σ move from under-predicting DESI toward agreement — the expected
projection signature, now confirmed on six tracers rather than one. σ(m) does
the opposite on every tracer, which is §57's result standing up.

That combination is the sharpest form of the open discrepancy: our `m` is
**too well determined and too weakly coupled** to the other three, and
marginalisation makes the first worse while partly fixing the second.

#### Correction: the `sigma_f_sigmar_frac` row used a stale DESI denominator

The tables above were aggregated from the sweep JSONs, whose `"desi"` block was
written by worker processes that had imported `desi_reference` **before** §60
landed. Checked key by key, exactly one differs from the live module:

```
  sigma_f_sigmar_frac      max |JSON - live| / live = 25.22 %
  the other nine keys                                  0.00 %
```

— §60 changed only that denominator (measured -> fiducial). Per tracer:

```
tracer   DESI in JSON  DESI live  F/D old  F/D new  M/D old  M/D new
BGS            0.2494     0.1992    0.670    0.839    0.663    0.830
LRG1           0.1251     0.1358    1.105    1.018    1.100    1.014
LRG2           0.1096     0.1151    0.979    0.933    1.035    0.986
LRG3           0.1120     0.1075    0.769    0.801    0.911    0.949
ELG2           0.0993     0.0949    0.540    0.565    0.587    0.614
QSO            0.1023     0.1186    0.743    0.641    0.799    0.689
mean                                0.801    0.800    0.849    0.847
```

Individual cells move by up to 25% (BGS 0.670 -> 0.839, QSO 0.743 -> 0.641),
**the mean does not**: 0.801 -> 0.800 Fisher, 0.849 -> 0.847 MCMC. So the
`sigma_f_sigmar_frac` line in the σ summary above stands as written, and the
projection conclusion is untouched — but the per-tracer DESI column for that one
row is stale and the corrected values are the ones here.

The forecast PLOT was never affected: it reads DESI from the live module.
Only the JSON-derived tables in this section carried the old value.

### Status of the four explanations

theory (§22), priors, free `dn`, projection (§57/§63). The first three stay
eliminated. Projection is now **partial, not eliminated**: it accounts for a
meaningful share of the `ρ(qap,m)` and `ρ(qiso,m)` deficit and none of
`ρ(fσ_r,m)`, whose sign we do not reproduce under either estimator.

Unchanged from §57: this does not block the emulator. `m` is a target we
predict, and under the production Fisher path σ(m) is the best-matching of the
four σ (0.916). The `m` *row* of the 4×4 is not validated at the ρ level.

### Provenance

The sweep ran against the pre-§62b n(z) tables — all 24 builds finished ~22:19,
the rebuilt tables installed at 23:16. §62a measured that shape change at
≤0.16% on shapefit at fixed N, well under the seed rms here, so the numbers
stand; a rerun would not move them visibly. §58 (per-tracer area) does **not**
affect these: `mcmc.py:build()` never passes `area`, so it always took the
`tracer_area()` path, before and after. §60 changes only the `"desi"` reference
block echoed into the JSONs, not the chains.


## §64 — the covar and mean pipelines define `f_sigmar` at different radii

Found while tracing what `r` means in `f σ_r`. Not a bug in anything that
currently runs; a trap sitting precisely where the bedcosmo integration lands.

### The two frames

`ShapeFit` measures every scale in units of the standard ruler, via
`s = r_d(cosmo) / r_d(fid)` (`power_template.py:664`). The sphere radius is
`r·s`, the pivot is `kp/s`, `Ap` carries `1/s³`. The two pipelines set
`fiducial` differently, so `s` behaves differently in each:

| | `fiducial` | `s` | `f_sigmar` is |
|---|---|---|---|
| covar (`core.py:744`) | `("DESI", theta_cosmo)` — the sample | ≡ 1 | `f σ_r(8)` |
| mean (`fourier_space.py:337`) | `"DESI"` — fixed | varies | `f σ_r(8s)` · tilt |

The covar template's fiducial *is* the sample, so `cosmo.rs_drag /
fiducial.rs_drag` is a number over itself. That is deliberate — it is what makes
`q ≡ 1` there so the Fisher curvature is the only meaningful content — but it
also means the two sides label different quantities with the same name.

Measured at z = 0.706:

```
case                  s   covar f_sigmar_fid   mean f_sigmar    ratio
fiducial          1.000             0.460725        0.460725   1.0000
omega_cdm=0.08    1.082             0.334282        0.250719   0.7500
omega_cdm=0.20    0.884             0.632717        0.915489   1.4469
omega_cdm=0.50    0.666             0.845678        2.450082   2.8972
h=0.55            0.817             0.449451        0.520607   1.1583
h=0.85            1.262             0.451208        0.391600   0.8679
```

`s` spans 0.17–1.84 over the `base` box (200 draws), so the sphere the mean
pipeline integrates runs 1.4–14.7 Mpc/h while the covar pipeline is pinned at 8.

### What is unaffected

The Fisher determines the FRACTIONAL error. With `J = diag(1, 1,
f_sigmar_fid, 1)` (`fourier_space.py:130`),

```
sigma(f_sigmar) / f_sigmar_fid == sigma(df)      exactly, by construction
```

and `sigma(df)` does not depend on which fiducial the template references. So
every DESI comparison is clean: `fsr_frac` (`comparison_plots.py:121`,
`compare_to_desi.py:706`) divides by `f_sigmar_fid` and recovers `sigma(df)`,
cancelling the radius. §45–§57 and §63, and every forecast plot, stand.

### What is exposed

The absolute `sigma_f_sigmar` emulator target carries the `r=8` scale, so it is
NOT in the same units as the mean emulator's `f_sigmar` label. A Gaussian
likelihood built as `(d - mu)^T C^-1 (d - mu)` with `mu` from the mean emulator
and `C` from the covar emulator mis-sizes that entry by the ratio column above.

`qiso`/`qap` have the same structure for a different reason: they are identically
1 in the covar frame, so `sigma_qiso` is already fractional, while the mean
pipeline's `qiso` is a real AP ratio against the fixed DESI fiducial. The
conversion is `sigma_abs = sigma_qiso * qiso_mean(theta)`.

`m` is the one clean entry — an additive dimensionless offset in both frames.

### Fix at assembly time

Work in fractional units throughout and multiply by the mean emulator's own
prediction, i.e. per sample

```
sigma_abs(f_sigmar) = sigma_f_sigmar / f_sigmar_fid * f_sigmar_mean
sigma_abs(qiso)     = sigma_qiso  * qiso_mean
sigma_abs(qap)      = sigma_qap   * qap_mean
sigma_abs(m)        = sigma_m
```

This needs `f_sigmar_fid` per sample, which the covar generator already records
(`fourier_space.py:199`) but does not currently ship as a target. Decide at
integration time whether to emit the fractional targets directly instead —
cleaner, but it changes the trained target set, so not a unilateral change.

Same class as §58 (a fallback that never fired) and §59 (a caller never
updated): two paths, one definitional mismatch, nothing checking the seam.


## §65 — §60's caller, in the mean plot's `f_sigmar` error bar

Found while answering where the mean plot's error bars come from. Plot-only; no
target, dataset or forecast number is touched.

Both error bars on that plot are DESI's, not ours — the mean pipeline emits
point predictions with no uncertainty. `m` reads `sigma_m` straight off the
published 4×4 diagonal and is correct. `f_sigmar` de-normalised the fractional
target back to absolute units and used the wrong denominator:

```python
err = [data[t]["desi"]["sigma_f_sigmar_frac"] * m for m, t in zip(meas, tracers)]
#                                               ^ the MEASURED value
```

§60 changed `sigma_f_sigmar_frac` from `sig[2]/vec[2]` (measured) to
`sig[2]/fid["f_sigma_s8"]` (fiducial). Multiplying by `meas` puts the
measured/fiducial factor straight back:

```
tracer   sig[2] (pub)  plotted err   ratio  meas/fid
BGS           0.09408      0.07513  0.7986    0.7986
LRG1          0.06426      0.06974  1.0852    1.0852
LRG2          0.05303      0.05565  1.0495    1.0495
LRG3          0.04730      0.04540  0.9599    0.9599
ELG2          0.03741      0.03574  0.9552    0.9552
QSO           0.04448      0.05158  1.1596    1.1596
```

`ratio == meas/fid` to every digit, which is the signature. The bar was 20%
short on BGS and 16% long on QSO — enough to change whether a residual reads as
consistent. Fixed to multiply by `published_fiducial(t)["f_sigma_s8"]`.

Third instance of one pattern (§58 a fallback that never fired, §59 a caller
never updated, §64 a fiducial set two ways, this one a denominator changed in
one place). None was found by a test. The seam check floated after §64 would
have caught §60's callers too.

Same commit, on request: the mean plot's AP panels drop the "@ DESI z_eff"
open marker (the remaining point is at our z_eff, so its offset no longer
attributes between the r_d convention and z_eff — noted in the docstring), the
f_sigmar panel drops its per-tracer Delta z annotations, and every subplot
title is now just the parameter name.


## §66 — the mean plot was two different plots wearing one title

`plot_mean`'s own docstring states the rule: DR1 data does not go on the axis,
because "whether the universe matches the fiducial is a statement about DESI,
not about this pipeline, and putting it on the same axis invites reading a
cosmological result as a code error." The AP panels obeyed it. The `f_sigmar`
and `m` panels plotted the DR1 measurement with its error bars.

So the left half asked "do we reproduce DESI's fiducial?" and the right half
asked "does DR1 agree with DESI's fiducial?" — a null test of our conventions
next to a DESI result, under one title, with no marking of which was which.

The `f_sigmar` panel was the worse of the two, because it LOOKED predictive
while being nothing of the kind. Our value there agrees with Table 11 to
+0.02% uniformly across all six tracers, so the panel was, numerically,
Table 11 vs Appendix A. Our contribution was reproducing Table 11 to two parts
in ten thousand; the visible ~1 sigma scatter was DR1 against Planck-LCDM.

### Now

All four panels are the same null test: generator vs DESI's FIDUCIAL, DR1
absent. `f_sigmar` is a percent residual against Table 11's `f sigma_s8`; `m`
is the absolute residual against 0 (Eq. 4.9 is a definition, so there is no
table entry and no ratio to form).

The `f_sigmar` and `m` points now come from `_mean_targets`, which calls
`_worker_run_mean_targets` at `FID_SAMPLE` — the ACTUAL mean-pipeline output.
The panel previously read `f_sigmar_fid` off the covar path's info dict. Those
agree to ~0.02% at the fiducial but they are not the same object (§64: the
covar template's fiducial is the sample, so its `f_sigmar_fid` is
`f sigma_r(8)`, while the mean extractor returns `f sigma_r(8s)`), and this
plot is about the mean pipeline.

Result, all six tracers:

```
D_V/r_d      -0.028% .. -0.005%
D_H/D_M      -0.003% .. +0.033%
f sigma_r    -0.015% .. +0.004%
m            -7.8e-05 .. -2.3e-05   (DESI sigma(m) = 0.051-0.167)
```

The `m` panel is scaled to the residual, not to DESI's `sigma(m)`: a +-0.051
band fills the panel and hides the 1e-5 structure that is the actual content.
The comparison to `sigma(m)` is annotated instead.

### Why `m` is 1e-5 and not 0

At the fiducial the mean pipeline's `dm` should be exactly zero — the fiducial
is its own reference. Traced: the two pipelines reach the same cosmology by
different routes.

```
covar   _to_shapefit_cosmo_params  -> omega_cdm = 0.12          (direct)
mean    _to_mean_extractor_params  -> Omega_m   = 0.3151918493  (assembled)
```

CLASS shoots for `omega_cdm` from `Omega_m` and lands on `0.1200000570`,
4.8e-7 relative. Handed a cosmology OBJECT instead, the same extractor returns
`dm = 0.000e+00` at z = 0.2954 / 0.7058 / 1.4902; through the worker's
parameter-passing path it returns `-4.5e-05`. Removing `w0_fld`, `wa_fld` or
`logA` from the passed params changes nothing, so it is the `Omega_m` route.

`_to_mean_extractor_params`' docstring says the `Omega_m` assembly exists "to
keep the mean and covar pipelines on the identical cosmology", and it does — to
5.7e-8 in `omega_cdm`. What is left is CLASS's shooting tolerance, not a
mapping error.

Magnitude: 4.5e-05 against DESI's `sigma(m)` of 0.051-0.167 is 0.0003-0.0009
sigma. Recorded because it is a SYSTEMATIC offset in the `m` labels (all six
tracers negative, monotone in z) rather than scatter, so anyone who sees it on
an auto-scaled axis should know it is the shooting tolerance and not physics.

### What this plot can and cannot say

It is a convention check and nothing more. At the fiducial cosmology the mean
pipeline returns 1, 1, Table 11 and 0 BY CONSTRUCTION, so the only thing it can
detect is a convention or implementation error — a wrong `r_d`, a wrong z_eff,
a wrong de-wiggling engine. That is worth having (it is how §53/§54's z_eff
work was verified) but it tests none of the pipeline's actual content, which is
how the four outputs VARY with cosmology and N_tracers. Nothing in this repo
currently plots that.


## §67 — `--reference fiducial|dr1` on the mean plot

§66 removed DR1 from the mean plot because it was mixed with a null test under
one title. The DR1 comparison is still worth having; it just has to be its own
plot, labelled as what it is.

```
comparison_plots.py mean --reference fiducial   -> shapefit_mean_vs_fiducial.png
comparison_plots.py mean --reference dr1        -> shapefit_mean_vs_dr1.png
```

Distinct filenames on purpose: `plots/` has no versioning and a rerun
overwrites in place, so a single name would make the two modes destroy each
other. `shapefit_mean_vs_desi.png` is retired — it named neither reference.

`plot_mean` is now one loop over `_MEAN_PANELS` with the reference chosen up
front, instead of two hand-written AP panels plus two bespoke ones. `_mean_vectors`
returns (generator, fiducial, dr1, dr1_sigma) as 4-vectors in DESI's basis, so
adding a third reference later is a fourth return value, not a fourth code path.
That structure is what §66 was really fixing: the panels diverged because
nothing forced them through a common shape.

In `dr1` mode each panel draws DESI's 1-sigma band around zero, so consistency
reads off directly rather than by comparing marker positions.

### On "MAP"

Appendix A publishes a datavector PLUS a Gaussian covariance, i.e. a Gaussian
summary of the posterior, and for a Gaussian the mean and the MAP coincide. So
"DR1 MAP values" is a fair label for those centres. The two would separate only
where the posterior is non-Gaussian — which §63 measured, and found in the `m`
row. Worth remembering if a future comparison leans on the `m` central value
rather than its sigma.

### What `dr1` mode is NOT

The generator is still evaluated at the FIDUCIAL cosmology. So the residual is
"does DR1 agree with Planck-LCDM", a DESI result — the plot's suptitle says so.
The visible features are DESI's, not ours: LRG2's D_V/r_d sits ~+5% (the
well-known DR1 low point, `desi_reference` records measured/fiducial = 0.948
there) and BGS's f_sigmar ~+25% on a ~20% error bar.

A genuine prediction test would evaluate the generator at DR1's OWN best-fit
cosmology and compare there. That needs DESI's LCDM parameter posterior;
`desi_reference` carries compressed parameters only (no omega_cdm, h, ln10A_s),
so it would mean transcribing another table. Not built, and flagged in the
`plot_mean` docstring so the gap is visible at the call site.


## §68 — `--reference dr1_bestfit`: the first mode where the generator predicts

§66/§67 established that neither existing mean-plot mode tests the pipeline:
`fiducial` is a null test by construction, `dr1` evaluates the generator at the
fiducial and so asks whether DR1 agrees with Planck-LCDM. This adds the mode
that actually predicts — generator at DR1's OWN best-fit cosmology, against
DR1's compressed measurement.

### The cosmology

`desi_reference.dr1_bestfit_cosmology()`, from DESI 2024 VII (arXiv:2411.12022)
Eq. (3.1), dataset DESI (FS+BAO)+BBN+ns10:

```
Omega_m = 0.2962 +- 0.0095    sigma8 = 0.842 +- 0.034    H0 = 68.56 +- 0.75
```

`omega_b` and `n_s` are not measured by DESI full-shape — they are priors, so
the prior centres are used (BBN 0.02218, ns10 0.9649; 2024 VII Table 1). Those
are the same numbers as `core.DEFAULT_PRIORS` because our priors came from that
table.

`ln10A_s` is not published. DESI quote `sigma8`, the mean pipeline takes `A_s`,
and linear `sigma8` scales exactly as `sqrt(A_s)` — so one Boltzmann call sets
the normalisation and a second verifies it. Recovers `sigma8 = 0.842000` and
`Omega_m = 0.296200` to the printed digits.

```
omega_cdm = 0.116404   (-3.00% vs fiducial)
h         = 0.6856     (+1.78%)
ln10A_s   = 3.147949   (+0.1116)
```

`omega_cdm` is assembled as `Omega_m h^2 - omega_b - omega_ncdm` with
`omega_ncdm` from the fiducial's single 0.06 eV neutrino — the convention
`core._to_mean_extractor_params` uses, so this does not open a second cosmology
definition (§66).

### Result

Residual (generator - DR1) in units of DESI's sigma, and per-tracer chi2 on the
full 4x4:

```
                 at fiducial                    at DR1 best-fit
tracer   DV/rd  DH/DM   fsr     m   chi2 |  DV/rd  DH/DM   fsr     m   chi2
BGS      +0.77  +0.22 +1.01 +0.19   2.50 |  +0.32  +0.27 +0.95 +0.08   2.01
LRG1     +1.33  +0.50 -0.63 -0.40   2.14 |  +0.41  +0.61 -0.65 -0.66   0.77
LRG2     +2.84  -0.43 -0.43 -0.68  11.05 |  +2.00  -0.27 -0.41 -0.94   5.59
LRG3     +0.19  -0.71 +0.37 +0.42   0.81 |  -0.61  -0.51 +0.44 +0.11   0.62
ELG2     +1.17  +1.01 +0.47 -0.91   3.91 |  +0.73  +1.18 +0.62 -1.18   5.82
QSO      +0.58  -0.15 -1.35 -1.26   4.73 |  +0.21  +0.05 -1.21 -1.61   4.19
TOTAL                              25.12 |                            19.01
```

24 dof. Moving to DR1's cosmology pulls `D_V/r_d` toward zero on five of six
tracers — the direction BAO drives the fit — and drops the total from 25.1 to
19.0.

**LRG2 remains the outlier**, +2.84 -> +2.00 sigma, the largest single chi2
contributor in both columns. That is the documented DR1 low point at z = 0.706
(`desi_reference` records measured/fiducial = 0.948 there), not something this
pipeline introduces.

### "Closure" here means the loop closes, not DESI's mock-recovery sense

⚠ Terminology. In DESI's own papers a "closure test" is usually MOCK RECOVERY:
fit simulations with a known truth and check the truth comes back (their
Fig. 6, which `desi_reference` already cites for the m_fid != 0 caveat). The
sense used here is different — the same DATA appears on both sides:

```
DR1 spectra --[DESI]--> App. A vectors --[DESI LCDM fit + BAO]--> Eq. (3.1)
                             ^                                       |
                             +---------- compare ----[OUR model]-----+
```

Consequences, both of which bound what the number above can mean:

  - **The floor is not zero.** Even a byte-identical forward model would
    recover the residual DESI's own fit left behind — one cosmology cannot
    pass exactly through 6 tracers x 4 parameters. 19.0/24 is roughly that
    scatter, not 19 units of our error.
  - **Shared errors are invisible.** Anything we get wrong the same way DESI
    does — the same de-wiggling convention, the same z_eff definition —
    closes perfectly. The test has power against a forward model that DIFFERS
    from theirs, and none against one that agrees with theirs and is wrong.

A blind test needs a cosmology from outside the loop (Planck alone, or DR2)
predicting DR1's compressed values. Not built.

### What this chi2 is NOT

Not a goodness-of-fit. Eq. (3.1) was inferred from these same compressed
measurements plus BAO, so the effective dof is well below 24, and the six
tracers are treated as independent here. Read 25.1 -> 19.0 as a RELATIVE
improvement, not as a p-value.

Three caveats carried in `desi_reference` at the definition:

  - **Not a joint MAP.** Eq. (3.1) reports marginalised means one parameter at
    a time; a vector built from them is the posterior's centre of mass, which
    equals the best-fit POINT only for a Gaussian posterior. DESI publish no
    LCDM chain here.
  - **Not independent.** Closure, not validation — see above.
  - **FS+BAO, not FS-alone.** Our compressed comparison is ShapeFit-alone; this
    cosmology carries BAO information the compressed vectors do not.

A residual in this mode therefore has three possible owners — our pipeline,
the marginalised-mean stand-in, or the FS+BAO/FS-alone mismatch — and the plot
cannot separate them.

### Plumbing

`_mean_targets(tracer, sample=None)` and `_mean_vectors(..., sample=None)` now
take a cosmology, cached per (tracer, sample) rather than per tracer. New
`desi_reference.dv_dhdm_at(z, params)` generalises `fiducial_dv_dhdm`, which
stays as the no-params path. Output: `shapefit_mean_vs_dr1_bestfit.png`.


## §69 — one DR1 mode, not two

§67 added `--reference dr1` (generator at the FIDUCIAL cosmology, against DR1's
measurement) and §68 added `--reference dr1_bestfit` (generator at DR1's own
best-fit LCDM, against the same measurement). Only the second is about this
pipeline. The first asks whether DR1 agrees with Planck-LCDM — a DESI result
that happens to share a basis with our outputs, which is exactly the confusion
§66 was written to end.

Keeping it as a named mode invited someone to run it, see a 2.8-sigma LRG2
residual, and file a bug against this repo. Removed.

`dr1_bestfit` is now simply `dr1`:

```
comparison_plots.py mean --reference fiducial   -> shapefit_mean_vs_fiducial.png
comparison_plots.py mean --reference dr1        -> shapefit_mean_vs_dr1.png
```

`shapefit_mean_vs_dr1.png` is now the §68 plot; the old file of that name (the
fiducial-cosmology comparison) is deleted rather than left to be mistaken for
the new one, and `shapefit_mean_vs_dr1_bestfit.png` is gone with the mode name.

Nothing about the §68 result changes — same cosmology, same chi2 25.1 -> 19.0,
same caveats. This is a naming and surface change only. The fiducial-cosmology
numbers survive in the §68 table as the comparison column, which is where they
belong: context for the improvement, not a mode anyone has to select.


## §70 — DESI publish the MAP, and their own chi2 at it

§68 built the DR1 cosmology by hand from DESI 2024 VII Eq. (3.1) — `Omega_m`,
`sigma8`, `H0` as three separate 1D marginals — because the chains were assumed
unavailable. They are not. The DR1 full-shape cosmology VAC is public, over the
same HTTPS route as the LSS catalogues (§62b), and it ships not just chains but
`iminuit/` posterior maximisations.

```
https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-cosmo-params/v1.0/
  iminuit/base/desi-shapefit-all-nolya_schoneberg2024-bbn_planck2018-ns10/
    bestfit.minimum.txt
```

mirrored at `~/data/desi/dr1_fs_cosmo/shapefit_all_nolya.bestfit.minimum.txt`.

That dataset is the right one on three counts: **ShapeFit**, not the direct
velocileptors fit — the compression Appendix A is in; **full-shape alone**, no
BAO — the compressed vectors carry none either; **`-nolya`** — exactly the six
tracers this module transcribes.

Read straight off, nothing assembled or inverted:

```
omega_cdm = 0.12215781   omega_b = 0.021975539   h = 0.69875343
ln10A_s   = 3.0282331    n_s     = 0.97266507
```

Two of §68's three caveats die: it is a genuine joint MAP rather than a
centre-of-mass of 1D marginals, and `ln10A_s` is published (`logA`) rather than
inverted from `sigma8`. The FS+BAO-vs-FS-alone mismatch dies with the dataset
choice. **Closure remains** — the MAP was fitted to these same vectors.

`omch2` is CDM-only: `ombh2 + omch2 = 0.144133` against `omegamh2 = 0.144778`,
the 0.000645 difference being the single 0.06 eV species. So it maps directly
onto our `omega_cdm`, and none of §66's `Omega_m` round-trip applies here.

### Correction to §68

§68 said `omega_b` and `n_s` are "priors, so the prior centres are used" and
that their posteriors are essentially the priors. The MAP disagrees: `n_s`
0.9649 -> **0.97267**, `omega_b` 0.02218 -> **0.021976**. They are priored, not
frozen, and the fit pulls them. Do not substitute prior centres.

### The floor is now measured, not described

§68 argued the closure chi2 has a non-zero floor — even an identical forward
model recovers the residual DESI's own fit leaves. The file publishes it: the
per-tracer log-likelihoods at that MAP sum to
`chi2__..._shapefit_all_nolya = 15.224566` over the same 24 numbers.

So the excess over that floor is the part attributable to our forward model
differing from DESI's:

```
tracer   our chi2   DESI chi2   excess
BGS          2.39        1.84    +0.55
LRG1         0.54        0.64    -0.10
LRG2         3.03        3.42    -0.39
LRG3         2.20        2.11    +0.09
ELG2         5.02        4.24    +0.78
QSO          2.38        2.98    -0.60
TOTAL       15.57       15.22    +0.34
```

**+0.34 in chi2 over 24 numbers**, per-tracer excess scattering both signs
between -0.60 and +0.78. Our cosmology -> compressed-parameter map agrees with
DESI's to well inside the noise of their own fit. The progression across this
section and §68: 25.1 at the fiducial -> 19.0 at the stitched Eq. (3.1)
cosmology -> **15.57 at the published MAP, floor 15.22**.

This is the strongest validation the MEAN pipeline has had. Note what it does
NOT cover: the covar pipeline (§63's open `m`-row question is untouched), and
anything we get wrong the same way DESI does, which closes perfectly and stays
invisible.

`DR1_BESTFIT_CHI2` / `DR1_BESTFIT_CHI2_TOTAL` carry the floor so the excess is
computed rather than eyeballed; the plot's suptitle prints all three numbers.

### Not yet used

The VAC also holds `cobaya/` chains and `iminuit/` MAPs for `base_w`,
`base_w_wa`, `base_mnu`, `base_mu_sigma`, and per-tracer datasets
(`fs-bao-bgs`, `fs-bao-lrg-z0`, ...). The per-tracer MAPs would allow a
tracer-by-tracer closure test; `base_w_wa` would exercise the `w0`/`wa` cosmo
models the emulator supports but nothing has validated against data.


## §71 — should the covar-side comparison run at the fiducial or DR1's MAP?

`mcmc.py:build()` and `compare_to_desi.our_forecast` both build at the DESI
fiducial (`theta_cosmo=dict(FID)`). §70 made DESI's own ShapeFit-alone MAP
available, so the question is whether the sigma comparison should move there.

### Measured (REPT, DR1 N, all six tracers)

```
                       sigma change      mean F/D
sigma_qiso              +3.42 %      0.794 -> 0.821
sigma_qap               +3.33 %      0.855 -> 0.883
sigma_f_sigmar_frac     +2.08 %      0.800 -> 0.817
sigma_m                 +1.40 %      0.916 -> 0.929
```

Per tracer +0.66% (ELG2 sigma_m) to +4.93% (QSO sigma_qiso), all POSITIVE, so
all toward DESI — we under-predict. `z_eff` moves <=0.13%.

Context for the size: §63's Fisher-vs-MCMC gap is 5-20% and its per-seed rms on
individual sigma is ~2-3%. This shift is at or below the noise already in those
chains, and changes no conclusion in §57/§63.

### The test moved BOTH the covariance and the derivatives

That is NOT DESI's configuration. Their sigma come from fitting DR1 against an
EZmock covariance built at AbacusSummit c000 — the fiducial, which §4.7 item 10
notes is simultaneously the grid and the ShapeFit template cosmology. Only the
THEORY and its derivatives sit at the best fit.

`build_shapefit_likelihood` uses one `theta_cosmo` for both, so three configs
exist and only two have been measured:

| | covariance | derivatives | measured |
|---|---|---|---|
| current | fiducial | fiducial | yes |
| MAP everywhere | MAP | MAP | yes, +1.4 to +3.4% |
| DESI-matching | fiducial | MAP | **no** |

The sign of the third is NOT predictable from the second: `C` and `dP/dtheta`
both scale with amplitude and partially cancel, so moving the derivatives alone
could push sigma the other way. `core.build_shapefit_likelihood(cov_override=)`
is the existing hook if it is ever built — it was added to substitute DESI's
own covariance in.

### Scope

This is a question about the VALIDATION COMPARISON, not about the pipeline.
Training data is unaffected: every sample carries its own cosmology with its own
self-consistent covariance, and it is the mean/covar consistency WITHIN a sample
that matters there (§64, §66). No emulator output changes based on the answer.

### Decision

Do not rerun the sweep for this alone — 6.8 h to move numbers by less than their
own seed noise. If the converged m-row run happens (§63: ~12500 iterations),
switch to the MAP then; it is free at that point and better motivated, since the
data was taken in the real universe rather than the fiducial one. Left as-is
for now, deliberately, so §63's numbers stay comparable to §57's.


## §72 — the convergence rerun: stop on tau, not on a budget

§63's sweep ran a fixed 2500 iterations and landed at 9-12 tau against emcee's
`niter > 50*tau`. The obvious fix — pick 12500 instead — is the same mistake
with a different constant, because **tau's ESTIMATE grows with chain length**.
A Kaiser smoke run makes it visible:

```
    60/200   tau  6  (9.3x)
   110/200   tau 13  (8.4x)
   160/200   tau 19  (8.2x)
   200/200   tau 24  (8.3x)
```

`niter/tau` sat at 8-10 throughout, however far it ran. Any fixed budget chosen
against today's tau can still finish short.

### What changed in `mcmc.py`

`run_emcee` now checks tau every `max(50, niter//40)` iterations and stops on
emcee's two-part criterion — `niter > 50*tau` AND `|tau_prev - tau|/tau < 0.01`.
Both must hold: long enough, and the estimate itself has settled. `NITER` is
therefore a **ceiling**; a job that exits early converged and one that reaches
the ceiling did not, recorded per seed as `converged` / `converged_at`.

Burn-in is now a fraction of what ACTUALLY ran, not of the requested budget —
an early stop would otherwise discard the wrong slice.

`--cosmology dr1_map|fiducial`, defaulting to the MAP (§70/§71). `fiducial` is
kept so §57/§63 stay reproducible rather than becoming unrecoverable history:
`COSMO=fiducial NITER=2500 ./run_mcmc_sweep.sh`.

Chains checkpoint to `*_seed<N>_partial.npz` every 1000 iterations and are
deleted when the seed finishes. At a 20000 ceiling this run is up to ~55 h and
the JSON is only written at exit; losing 40 hours to a reboot was not an
acceptable failure mode.

### Two corrections to what was said before

**"~6 h by widening to more walkers" was wrong**, twice over. The 50-tau rule
constrains chain LENGTH IN ITERATIONS, and tau is measured in iterations too —
more walkers buys effective sample size, not convergence. And there are no idle
cores to widen into: the box is a Threadripper 3975WX, **32 physical** cores
(64 threads), and the sweep's 24 jobs already sit inside that. 48 jobs would
oversubscribe and slow every job ~1.5x.

**S63's outputs were archived, not left in place.** `comparison_plots._load_mcmc`
GLOBS the logs directory, so 24 fiducial-cosmology JSONs sitting beside the new
MAP-cosmology ones would have been unioned into single error bars mixing two
input cosmologies. Moved to `archive/s63_fiducial_2500/` with a README. The glob
is non-recursive, so the subdirectory is not picked up.

### Launched

2026-08-04 17:35:56, 24 jobs (6 tracers x seeds 42-45), REPT, `dr1_map`,
ceiling 20000, burn-in 0.4.
`logs/mcmc_sweep_20260804_173556/`.

Expect ~55 h if the ceiling binds. Given tau grew from 213-280 at 2500
iterations and keeps growing, **the ceiling binding is the likely outcome, not
the exception**. That is still worth running: at 20000 iterations with tau ~400,
effective sample size is ~32*20000/400 = 1600 per seed, ~6400 over four — ample
for sigma and rho at the 1-2% level — and 40% burn-in of 20000 is 20-27 tau,
which is ample burn-in even if the 50-tau reliability rule for the tau ESTIMATE
is not met. Report `converged` honestly either way; do not describe a
ceiling-limited run as converged.


## §73 — §64 resolved from DESI's source: the `Ap` convention, and our `J` is right

§64 flagged that desilike implements two ShapeFit amplitude conventions and we
did not know which DESI's published 4x4 is in:

```
'Ap'       df = f*sqrt(Ap) / (f*sqrt(Ap))_fid      <- what our J assumes
'fsigmar'  df = f_sigmar / f_sigmar_fid            <- adds a dm-dependent term
```

They differ by up to 22% at |dm| ~ 0.1, and the difference feeds exactly
`sigma(f_sigmar)` and `rho(f_sigmar, m)` — the two entries §57/§63 could not
reproduce. §64 could not settle it and warned that tuning the coefficient to
close the gap would be the §33r error-cancellation trap.

**Settled by reading DESI's likelihood**: `cosmodesi/desi-kp-cosmological-
likelihoods` @ `7d51f4f86dc3bee6bf10f1a684913c943a89a844`, file
`dr1/cobaya/desi_shapefit_bao_all.py` lines 168-178 (sha256
`f492d1fd...bb39431`). Found via the MAP file's own column header,
`chi2__desi_y1_cosmo_bindings.cobaya_bindings.desi_shapefit_all_nolya`.
Archived at `~/data/desi/dr1_fs_cosmo/desi_kp_likelihood_7d51f4f8/`:

```python
elif param == 'df':
    flattheory[iparam] = shapefit['f_sqrt_Ap'][idx] / self._template['f_sqrt_Ap_fid'][idx]
elif param == 'dm':
    flattheory[iparam] = shapefit['m'][idx] - self._template['m_fid'][idx]
```

```python
Ap = 1. / s**3 * pk_dd_interpolator(kp)
f  = pk_tt_interpolator.sigma8() / pk_dd_interpolator.sigma8()
f_sqrt_Ap = f * Ap**0.5
```

DESI's compressed parameters ARE `(qiso, qap, df, dm)` in the **`Ap`
convention**. No `sigma_r`, no `exp(dm/2a)` correction. So `f_sigmar = df *
f_sigmar_fid` is a pure rescaling by a constant, `J = diag(1, 1, f_sigmar_fid,
1)` is correct, and **there is no missing off-diagonal term**. §64's concern is
withdrawn.

Two consequences:

  - The §72 sweep now running is measuring the right quantity. Had this gone the
    other way, 54 h would have bought a more precise wrong number.
  - The published "f sigma_s8" column is that same `df` relabelled against the
    FIDUCIAL value, which independently confirms §60's fix (divide by Table 11,
    not by the measurement) and §65's follow-on.

Two provenance caveats, since §73 and §74 both rest on this file.

  - The file read is the BASE class. The likelihood behind our MAP,
    `desi_shapefit_all_nolya`, is GENERATED by `generate_files_shapefit_bao.py`
    as `class desi_shapefit_all_nolya(desi_shapefit_bao_all)` with a different
    `tracers` list and `observable_name`, overriding no methods — so
    `get_flattheory` and `_get_f_m` are inherited unchanged. There is no
    committed file of that name to read directly.
  - The repo is `init repo` 2026-01-08 with one edit since; DR1 published
    2024-11. So this is a LATER RELEASE of the DR1 likelihood, not provably
    byte-identical to the code that produced the published chains. The
    convention is structural and desilike's separate observables corroborate
    it, but the pin is not "the code that made these numbers".

Corroborating detail: desilike's own `ShapeFitCompressionObservable` accepts
`quantities` from `['m', 'n', 'f_sqrt_Ap', 'dm', 'dn', 'df', 'DM_over_rd', ...]`
— **no `fsigmar`**. That name belongs to `StandardCompressionObservable`
(`['fsigmar', 'df', ...]`), the fixed-template compression. The two conventions
are not interchangeable and desilike keeps them in separate observables.

### Opened while reading: DESI de-wiggle with `peakaverage`, not `wallish2018`

```python
filter = PowerSpectrumBAOFilter(pk_dd_interpolator, engine='peakaverage',
                                cosmo=self.fiducial, cosmo_fid=self.fiducial)
```

That filter is what defines `m` in their cosmology -> compressed map, which is
the same map our MEAN pipeline implements — and we pass `with_now="wallish2018"`
at every construction site, on the understanding that it is DESI's choice
(`project_bao_dewiggling_engine`). At the fiducial the engines differ by 0.092 in
`m_fid`, which is 1.4x LRG2's sigma(m).

`dm = m - m_fid` is a DIFFERENCE with both terms on the same engine, so much of
the offset should cancel; the §70 closure test passing at chi2 15.57 vs DESI's
15.22 is evidence that it largely does. Measured separately — see §74.

Two smaller differences in the same function, not yet chased:

  - DESI's `Ap` uses `pk_dd_interpolator(kp)` — WITH wiggles. desilike's
    extractor uses `pknow_dd_interpolator(kp)`. At kp = 0.03 the wiggle
    contribution is small but not zero.
  - DESI's `f` is `sigma8(theta-theta) / sigma8(delta-delta)`, a ratio of
    velocity to density amplitudes, not a growth-rate derivative.


## §74 — the `peakaverage` question: measured, negligible, keep `wallish2018`

§73 found DESI's compressed likelihood de-wiggling with `engine='peakaverage'`
while every construction site here passes `with_now="wallish2018"`. The engines
differ by 0.092 in `m_fid`, 1.4x LRG2's sigma(m), so this needed a number.

Measured at DESI's ShapeFit-alone MAP (§70), all six tracers:

```
tracer    z_eff |  dm wallish  dm peakavg      diff | sig(m)  diff/sig
BGS      0.2954 |    +0.02507    +0.02901  +0.00394 | 0.1672    +0.024
LRG1     0.5095 |    +0.02504    +0.02900  +0.00396 | 0.0699    +0.057
LRG2     0.7058 |    +0.02502    +0.02900  +0.00398 | 0.0690    +0.058
LRG3     0.9185 |    +0.02500    +0.02900  +0.00399 | 0.0591    +0.068
ELG2     1.3168 |    +0.02499    +0.02899  +0.00400 | 0.0660    +0.061
QSO      1.4902 |    +0.02500    +0.02899  +0.00400 | 0.0513    +0.078

f_sigmar: +0.36% uniformly, against sigma of 10-25% -> 0.02-0.05 sigma
```

**+0.004 in `dm`, 0.02-0.08 sigma.** The 0.092 `m_fid` offset cancels almost
entirely, which is what `dm = m - m_fid` with both terms on one engine predicts.
The residual is nearly z-independent (+0.00394 to +0.00400 across z = 0.30-1.49),
consistent with a fixed shape difference rather than anything evolving.

This also explains why §70's closure test passed at chi2 15.57 against DESI's
15.22 despite the engine mismatch: at 0.02-0.08 sigma per tracer it cannot move
a 24-number chi2 appreciably.

### Decision: keep `wallish2018`

Two reasons beyond the size of the effect.

  - `peakaverage` is NUMERICALLY UNSTABLE in this repo's BAO path — it crashed
    chaotically (w0 sensitivity at 1e-9) and mislabelled sigma by ~2x, which is
    what the 2026-07-16 fix was for (`project_bao_dewiggling_engine`). Adopting
    it in shapefit to chase a 0.05-sigma agreement would trade a negligible
    systematic for a known instability.
  - Switching would invalidate every existing mean label and force a full
    regeneration, for a shift far below the sigma on the quantity.

Recorded so this is not rediscovered as a suspected bug. If a future comparison
needs sub-0.05-sigma fidelity on `m`, the engine is the first thing to revisit.

### Caveat

Measured at ONE cosmology (the MAP). The offset is nearly constant across z,
which suggests stability, but nothing here shows it stays 0.004 across the full
prior box. The training data spans omega_cdm U[0.01, 0.99]; if the engines
diverge more at the extremes, the effect on training labels is unmeasured.


## §75 — correction to §73/§74: wallish2018 was never sourced for full-shape

§73 and §74 framed DESI's `peakaverage` as conflicting with "DESI's fiducial
choice". That framing is wrong, and the mistake is worth recording because it
came from applying a BAO-side source to the full-shape side.

### What the source actually says

`project_bao_dewiggling_engine` cites Chen et al. 2024, *BAO Theory and
Modelling Systematics for the DESI 2024 results* (arXiv:2402.14070) §8(v):

> "The **DESI BAO template** is constructed following the method of Wallisch
> (2018)"

and §7.4, which twice calls Wallisch 2018 "our fiducial". That is the **BAO
systematics paper**, about the **BAO template**. It says nothing about the
ShapeFit compression, and does not extend to it.

### Where shapefit's choice came from

The build plan specifies `with_now="wallish2018"` with the note *"explicit
with_now (spelling 'wallish2018')"*, and `core.py:10` records the reason as
**"`with_now` MUST be explicit — the desilike [default]"**. The justification on
record is *do not silently inherit the library default* — which is sound, and is
the §33-era lesson — NOT *this is the engine DESI's full-shape analysis uses*.
The engine itself was carried over from `bao/`.

So there is no contradiction between the notes and DESI's source. DESI use
different engines in different analyses, and this repo's shapefit choice was
inherited, never independently sourced for full shape.

### A distinction §73/§74 blurred

| ours | DESI's equivalent | their engine |
|---|---|---|
| `fourier_space.py:337` extractor -> MEAN path | `_get_f_m` in the compressed likelihood | **peakaverage** (read at 7d51f4f8) |
| `core.py:747` template -> COVAR path | their full-shape FITTING template | **unknown** |

§73's finding and §74's measurement both concern the MEAN path only. What DESI's
full-shape fit template used is a separate, unanswered question — and desilike's
default there is also `peakaverage`, so if they did not override it, the covar
path differs too. §74 measured `dm` and `f_sigmar` from the EXTRACTOR; it says
nothing about the covariance.

### What still stands

§74's decision to keep `wallish2018` is unaffected, but its reasons narrow to
the two that do not depend on the mis-framing:

  - the measured mean-path effect is 0.02-0.08 sigma, far below anything that
    matters;
  - `peakaverage` is the engine that crashed chaotically and mislabelled sigma
    ~2x in the BAO path, so adopting it repo-wide trades a negligible
    systematic for a known instability.

What is NOT still standing is any claim that wallish2018 is what DESI use for
ShapeFit. We do not know that, and for the compressed map we now know the
opposite.

### Open — RESOLVED in §76

Find what `with_now` DESI's full-shape FITTING pipeline used (their FS fit
configs, not the compressed-cosmology likelihood read in §73). If it is
`peakaverage` there too, the covar path's engine is unsourced in the same way,
and the §74 measurement does not cover it.

§76: it IS `peakaverage`, and not by configuration — desilike's REPT class
overrides `with_now` unconditionally, so both DESI's fit template and ours were
forced there regardless of what either passed. The `with_now="wallish2018"` in
the covar path never took effect. The table above resolves to: mean path
peakaverage (matched to §73 by the §76 switch), covar path peakaverage
(always was).


## §76 — §75's open question, answered from desilike's source: REPT forces `peakaverage`, and our `with_now` was never in effect

§75 left one question open: what `with_now` did DESI's full-shape FITTING
pipeline use? It is not answerable from their configs, because **the theory
class overrides the config.**

### The mechanism

`REPTVelocileptorsPowerSpectrumMultipoles.initialize` (desilike
`full_shape.py:1416`):

```python
self.template.init.update(with_now='peakaverage')
```

Unconditional `.update()`. Compare `bao.py:81`, which uses
`setdefault('with_now', 'peakaverage', if_none=True)` and therefore RESPECTS an
explicit caller choice. The same unconditional override appears in
`FOLPSAXPowerSpectrumMultipoles` (2310) and in PyBird when
`with_nnlo_counterterm` is on (1688, 1953).

`core.py` builds the template with `with_now=...` and then hands it to
`theory_cls`. So the argument is overwritten for every theory DESI or we would
plausibly use, EXCEPT Kaiser. Measured on the real production object
(`mcmc.build("LRG2", theory, cosmo="dr1_map")`):

```
rept    -> template.with_now='peakaverage'   PeakAveragePowerSpectrumBAOFilter
kaiser  -> template.with_now='wallish2018'   Wallish2018PowerSpectrumBAOFilter
```

### Three consequences

1. **The covar path has always been `peakaverage`.** Every covar training set,
   every Fisher sigma, and the §72 chains running now. `with_now="wallish2018"`
   at the template was inert in production from the day it was written.
2. **This answers §75 favourably.** DESI's full-shape baseline IS desilike REPT
   (2024 V §4.7 item 2, velocileptors/EPT). Their fit template was forced the
   same way ours was. The covar path already agrees with DESI — silently, and
   for a reason neither analysis chose.
3. **Kaiser-vs-REPT deltas were never clean.** Any sigma difference attributed
   to the theory model also contained a de-wiggling engine swap. This affects
   `validate_forecast.py`'s sensitivity checks and Kaiser smoke comparisons —
   not production numbers.

### Decision: `peakaverage` everywhere (mean path switched)

The mean path (`fourier_space.py` extractor, no theory class) was the only place
`wallish2018` was ever live — and it is the one place DESI are known to use
`peakaverage` (§73). Leaving it would make the two halves of this pipeline
disagree with each other AND with DESI. Switched.

§74 declined this switch for two reasons. Both are now void:

  - *"forces a full regeneration"* — the regeneration is already required for
    §58 (the per-tracer footprint never reached the generators), so the switch
    is free if it lands first.
  - *"`peakaverage` is numerically unstable"* — that was a BAO-path finding
    (`project_bao_dewiggling_engine`). It does not transfer. See below.

Note the covar-side edit is documentation, not a change: it is bit-identical
under REPT (verified above) and only makes Kaiser agree with REPT.

### The stability probe (`probe_dewiggle_engine.py`), LRG2 and QSO

§74's caveat — "measured at ONE cosmology" — became load-bearing once the switch
was made, so it was tested across the prior box: MAP + 8 corners of
(`omega_cdm`, `h`, `ln10A_s`) + 4 interior points, both engines.

**Robustness — identical, so the engine is not the variable:**

```
peakaverage : 8/13 ok   FAILED at oc0.99 (all 4 corners), oc0.6_mid
wallish2018 : 8/13 ok   FAILED at the SAME five points
```

Both tracers, same five. That is CLASS failing on absurd cosmologies
(`omega_cdm` >= 0.6), not de-wiggling.

**Smoothness — `wallish2018` is the chaotic one HERE.** Nudging `omega_cdm` by
1e-9 relative (the BAO path's crash signature was exactly this: labels jumping
under a 1e-9 nudge):

```
                          LRG2       QSO
MAP           peakaverage 5.57e-05   5.53e-05
              wallish2018 6.50e-04   6.30e-04    12x
oc0.01_h0.2   peakaverage 2.26e-06   2.28e-06
              wallish2018 5.67e-04   5.67e-04   250x
oc0.05_mid    peakaverage 2.03e-06   1.70e-06
              wallish2018 2.58e-04   3.24e-04   127-190x
```

The instability §74 feared does not transfer to the shapefit mean path; the
inequality runs the other way, by 12-250x. Two honest caveats: both engines sit
far above the ~1e-9 an exactly-smooth function would give, so both have a
numerical noise floor and only their RATIO is being compared; and the metric is
per-label relative, which inflates `m` because `m` is near zero at the MAP.

**Agreement — §74's "0.02-0.08 sigma" is a MAP-only number:**

```
            LRG2 absmax    QSO absmax
qiso        0.000000       0.000000     (AP geometry: engine-independent)
qap         0.000000       0.000000
f_sigmar    0.996264       1.361711
m          17.113941      17.114457
```

At the MAP the engines agree to +0.004 in `m` (§74). In the far corners they
differ by up to **17**. The engine choice is nearly irrelevant near the data and
enormous at the box edges — which is where emulator training samples mostly
live. This is the strongest argument for matching DESI rather than picking on
aesthetics: at the corners there is no small-difference excuse.

### A defect probed for and NOT found

At the exact corner `oc=0.01, h=0.2`, `wallish2018` returns `f_sigmar` =
**0.000000** and `m` = -19.08 while `peakaverage` returns finite values. Zero is
finite, so `_worker_run_mean_targets`'s `np.isfinite` guard passes it — the
label would enter training silently.

Scanned all existing mean training data for it: **0 occurrences in 9072 labels**
(v1 3072, v2 6000; minimum `f_sigmar` ~1e-5 in v1/v2, never 0). The pathology is
a measure-zero box corner that random sampling does not hit exactly. Recorded as
a latent edge case, not a live data defect, and moot after the switch.

### Follow-ups

  - The `mcmc.py --cosmology dr1_map` / `comparison_plots.py --reference dr1`
    naming mismatch (same cosmology, two spellings; §69 renamed one and not the
    other). Deliberately deferred: the 24 running jobs write `"dr1_map"` into
    every diag record. Add `dr1` as an alias in both after the sweep.
  - Regenerate golden + v2 with §58 AND this switch in, together.
  - Re-examine any Kaiser-vs-REPT sigma delta recorded before this entry.


## §77 — §62c done: the n(z) layer is release-scoped, and the last uniform FKP pivot is gone

Two of the long-standing open items, landed together because they touch the
same layer. Behaviour-preserving: `benchmark_desi.py` reproduces §53's numbers
exactly (mean |err| 0.062%, max 0.121%, all six pivots matching Eq. 8.4).

### §62c — `dataset` threaded through the n(z) layer

Every other release-scoped lookup (`ntracers`, `tracer_area`,
`get_default_save_path`, `tracers.yaml` overrides) took a `dataset`. The n(z)
tables did not, so with DR2 present `ntracers` would switch release while
`_load_nz_slice_fractions` silently returned DR1 shapes.

**(1) Release-scoped path.** New `util.nz_slices_path(filename, dataset)`:
`{base}/{dataset}/{filename}`, with the flat pre-§62c layout accepted ONLY for
dr1 and only with a `DeprecationWarning`. For any other release a missing
scoped directory RAISES rather than falling back — flat *is* dr1, so silently
serving it to DR2 is the bug being fixed.

It lives in `util.py`, beside the other release-scoped lookups, and NOT in
`bao/core.py`: `bao/fkp_analytic_cov.py` needs it too, and a bare
`import core` there resolves to `shapefit/core.py` whenever cwd is `shapefit/`
— the exact collision the build plan warns about.

**(2) `dataset` is keyword-only and REQUIRED** on `_load_nz_slice_fractions`,
`_desi_nz_geometry` and `fkp_analytic_cov.load_nz_slices`. A default of "dr1"
would have preserved the very failure mode being fixed (a DR2 caller that
forgets still gets DR1); with no default, a missed call site is a `TypeError`
at import-adjacent time instead of a subtly wrong covariance. All 14 call sites
audited and updated; `_compute_v_eff_fkp` gained the parameter, every other
caller already had a release in scope.

**Found while threading:** `_DESI_NX_CACHE` was keyed on `tracer_bin` alone, so
a DR2 run following a DR1 run *in the same process* would have been served DR1
rows from cache even with the paths fixed. Key is now `(dataset, tracer_bin)`.

**(3) `make_nz_slices.py --dataset`** with a `_RELEASES` table holding dr1
only. Anything else exits with a message naming what would have to be audited
first (catalogue stems, slice edges, areas, counts, download URL). Stubbed to
fail loudly, per §62c — the danger is emitting DR1 tables into a `{dataset}/`
directory that then looks populated. Output is now release-scoped too.

The 13 live tables were COPIED (not moved) into `nz_slices/dr1/`, verified
byte-identical. The flat originals still resolve via the dr1 fallback, so
nothing depends on the copy; they are now dead weight and can be deleted once
you are satisfied.

### The last uniform FKP pivot (§54's open item 2)

`fkp_analytic_cov.P_FKP_DEFAULT = 1.0e4` — LRG's pivot — was reaching BGS, ELG
and QSO through the two `fkp_analytic_cov()` call sites in `compare_to_desi.py`,
which passed no `P_FKP`. That is a 1.43x error in `n*P0` for BGS, 2.5x for ELG,
1.67x for QSO.

New `fkp_analytic_cov.fkp_p0_for(tracer)` reads `fkp_p0` from tracers.yaml, and
both call sites now use it. `config_space._pivot` (which §55 had already fixed
for the config-space path) now delegates to the same helper, so there is ONE
definition of the lookup rather than two copies drifting.

`P_FKP_DEFAULT` stays as the last-resort default for the one caller with no
tracer in hand (a synthetic single-slice self-test at `config_space.py:486`).

**A bug written and caught in the same session:** the first `fkp_p0_for` passed
`analysis="bao"` to `get_tracer_config` and wrapped it in `except Exception`.
But `LRG3` is shapefit-only — bao uses the combined `LRG3_ELG1` — so the lookup
raised, the broad `except` swallowed it, and LRG3 silently took the fallback
pivot. It happened to be the right number (10000), which is exactly why it
would have survived review. `analysis` is no longer passed, config errors
propagate, and only a genuinely missing `fkp_p0` warns.

### Also: §76 stragglers

Three `wallish2018` sites §76 missed, all live (Kaiser templates and a direct
`PowerSpectrumBAOFilter`): `compare_to_desi.py:519`, `:575`,
`validate_mean.py:240`. Now `peakaverage`. The `bao/core.py:1918` site is
deliberate and stays — the BAO template is where Wallisch 2018 IS sourced
(Chen et al. 2024 §8(v)) and where `peakaverage` was unstable.

### Status of the three coupled quantities (closes §50's table)

  - `fkp_p0`: Eq. (8.4) everywhere, including the analytic-cov path. Done.
  - n̄: the ~23% gap was the footprint (§54), fixed.
  - z_eff: 0.062% mean against DESI's published values, at the §51 reference.

### Not done here

  - The mesh norm in the covariance path (§46/§47).
  - The covariance's THIRD n(z) (`load_nz_slices`, N*frac/V) is still not
    routed through `NX`.
  - LRG3's residual density ratio, 0.629 (§54).


## §78 — the seam check: the mean/covar consistency test, built and passing

Four of the last five findings were the same shape — a quantity BOTH pipelines
consume, changed on one side — and every one was caught by hand, late, after it
had already contaminated data:

```
S42  z_eff frozen at the fiducial in the mean path, derived per sample in covar
S58  the per-tracer footprint reached the covar path and not the generators
S60  a correction applied to one path's f_sigmar and not the other's
S64  the two paths APPEARED to define f_sigmar at different radii
S76  an explicit with_now honoured by one path, silently overridden in the other
```

That is a mechanical class of bug, so `seam_check.py` is the mechanical check.

### What it asserts

Per tracer, at two cosmologies (DESI's MAP and a deliberately off-fiducial
point — a seam that agrees only at the fiducial, as §42's did, passes a
fiducial-only test):

| seam | mean side | covar side |
|---|---|---|
| tracer area | `tracer_area(t, dataset)` | same |
| n(z) table | resolved path | same |
| cosmology | `_to_mean_extractor_params` -> CLASS | `_to_shapefit_cosmo_params` -> CLASS |
| z_eff | `_mean_z_eff_for_sample`, as the worker calls it | `build_shapefit_likelihood`'s `info["z_eff"]` |
| de-wiggle engine | `extractor.with_now` | `template.with_now` AFTER `theory_cls` |
| f_sigmar radius | `extractor.r` | `template.r` |

The engine row reads the template *after* `theory_cls` has had it, which is the
only way §76's override is visible — reading the constructor argument would
have shown agreement while the objects disagreed.

Defaults to Kaiser: none of the checked quantities depend on the theory model,
and it is far cheaper. `--theory rept` for the production path.

### Result: 16/16, LRG2 and QSO, both cosmologies

### What it found on the first run

The cosmology seam FAILED at a 1e-10 tolerance: `omega_cdm` differed by
4.7e-7 relative, and `Omega_m`, `rs_drag`, `sigma8` with it.

This is not a bug, and the distinction matters. The covar path hands cosmoprimo
`omega_cdm` directly; the mean path cannot — the extractor's pipeline exposes no
`omega_cdm` — so it hands over `Omega_m` and CLASS SHOOTS to recover the rest.
One route goes through a nonlinear solve, so the two cannot agree bit-for-bit,
and the residual is the shooting tolerance. **It is the same residual that makes
`m` come out at 1e-5 rather than 0 in the mean plot's fiducial null test (§66)** —
two symptoms, one cause, now pinned in a test instead of rediscovered.

Tolerances therefore split by KIND rather than being loosened globally:

  - `_RTOL_PASSTHRU = 1e-12` for parameters passed straight through on both
    sides (`omega_b`, `h`, `n_s`) — these must be bit-equal and are;
  - `_RTOL_SHOOT = 5e-6` for CLASS-solved quantities: 10x the observed 5e-7
    residual, while a genuine mapping error is far larger — dropping
    `omega_ncdm` from the `Omega_m` assembly (the bug
    `_to_mean_extractor_params` exists to prevent) shifts `omega_cdm` by ~6e-4,
    i.e. 5e-3 relative, a thousand times the tolerance.

The report prints each relative gap against its tolerance, so a seam drifting
toward its limit is visible before it becomes a failure.

### What this does NOT do

It is a CONSISTENCY test, not a validation: both paths can be wrong together and
it stays silent. Correctness against DESI lives in `benchmark_desi.py` (published
z_eff/area/pivots) and `validate_forecast.py`.

It also only covers quantities both paths currently share. It would NOT have
caught §65 (a stale denominator in plotting code, which is neither path) or §63's
misuse of a signed mean. Run it before every regeneration, not instead of
thinking.


## §79 — the reference tables are 39 KB; vendor them and the forecast needs no downloads

Two things landed: `init_desi_data.py` (fetch every DESI input over public
HTTPS) and, more usefully, the realisation that most of what a fresh machine
needs is tiny and already derived.

### `init_desi_data.py`

Public and anonymous — no NERSC account, no MFA, no DTN pull. The recipe in
`project_shapefit_bundle_blocker` (scp from dtn01) was an outage workaround from
when `data.desi.lbl.gov` was Spin-backed and down; both endpoints return 200 now.

Groups, measured (`--dry-run` HEADs every URL): `lss` 1.12 GB, `randoms` 20.9 GB
at nran=2, `bao` 3 MB, `fs` 4 MB, `cov` 13 MB.

Validated by pointing it at this box: the manifest resolved all 62 non-random
files to the exact bytes on disk. That is what caught a wrong FS bundle name —
known-local files showing as missing. (`likelihood_shapefit_spectrum-poles-
rotated+bao-recon_syst-rotation-hod-photo_*` is a different, 35 KB product; the
code loads `likelihood_spectrum-poles-rotated_syst-hod_*`.)

Two traps handled: DESI serves `+` as `%2B` while every consumer here globs for
a literal `+`, so the encoding applies to the URL only; and `--force` unlinks
first, because `curl -C -` against a complete file resumes to a no-op.

### The randoms are NOT required — and the reason matters

The claim that the 20.9 GB was needed for the `_desi_nx` tables was wrong.
`NX` and `WEIGHT` are columns in the `clustering.dat.fits` catalogues too.
Measured, z_eff vs DESI 2024 V Table 1, six tracers:

```
S1 randoms + NX randoms   0.062% mean   0.121% max   <- the shipped tables
S1 data    + NX randoms   0.064%
S1 randoms + NX data      0.117%
S1 data    + NX data      0.111%
(pre-S53, for scale:      0.313% mean, 0.653% max)
```

`S1` from the data costs nothing; `NX` from the data costs everything. The
mechanism, and it is not noise:

  - `S1` is a SUM of weights. Data and randoms differ by a constant 0.0772
    (~1/13, the density ratio) which cancels identically in Eq. (2.1)'s
    normalised ratio. Its residual 0.30% z-scatter sits at the 0.36% Poisson
    floor (77k objects/slice) — unbiased, just noisier.
  - `NX` is a mean of a DENSITY weighted by the objects themselves. Galaxies
    preferentially occupy high-`NX` regions, so the data-weighted mean is
    <n^2>/<n> rather than <n> — biased 4.8% high. Randoms sample the SELECTION
    FUNCTION, which is exactly what Eq. (2.1) means by `n_ran`.
  - The bias is z-DEPENDENT (0.41% scatter), which is why it survives the
    ratio. The two tracers that degrade most, LRG3 (0.9185 -> 0.9199) and ELG2
    (1.3168 -> 1.3199), are the ones with the steepest n(z) evolution across
    their bin — the same LRG3 shape sensitivity §54 flagged.

Same object-weighted vs selection-weighted distinction as §48's retracted "29%
low" and §50's 1.394-vs-1.227. There is no reweighting that recovers the
selection function from the data alone, and inventing a correction factor is
exactly the kind of fudge this project rejects.

### Vendoring: 39 KB, and the forecast stops needing downloads

The `_desi_nx.csv` tables ARE the reduction of the randoms — 20.9 GB in, ~10 KB
out. They were unobtainable on a fresh machine only because they sat in
`~/data`. Committed to `data/dr1/`:

| | size |
|---|---|
| `{tracer}_nz_slices.csv` (7) | n(z) shape |
| `{tracer}_desi_nx.csv` (6) | the randoms reduction |
| `desi_data.csv`, `desi_tracers.csv` | the `N_tracers` design box |
| **total** | **39 KB** |

`~/data/desi` still WINS when present; the vendored copies are a fallback
(`util._repo_fallback`, and a branch in `util.nz_slices_path`), so
`make_nz_slices.py --install` keeps taking effect and this box is unchanged.

`data/dr1/PROVENANCE.md` records what produced each file, sha256s, and the table
above — so the `NX` bias is not rediscovered.

Verified two ways: with `Path.home()` pointed at a nonexistent directory, all
seven `N_tracers` values and all n(z)/`desi_nx` tables resolve from the repo;
and with `~/data` present, `benchmark_desi.py` is bit-identical (0.062% / 0.121%).

The `N_tracers` box mattered most here: `util.ntracers` reads `desi_data.csv` /
`desi_tracers.csv`, so before this the pipeline's whole input axis was
unobtainable on a new machine.

### Still open

`LRG3_ELG1_desi_nx.csv` does not exist, so that bin still falls back to
`nbar_file`. It needs the `LRG+ELG` randoms (~3.3 GB, `--what randoms --nran 1`),
once, after which it is vendored like the rest and nobody needs randoms again.
That is the last thing keeping `nbar_file` alive.


## §80 — the repo owns the reference tables, and the `_desi_nx` recipe is recovered

§79 vendored the tables but left `~/data/desi` winning. That is two sources of
truth, which is how a machine silently disagrees with what is version-controlled.
Now the repo is the ONLY source, and — the harder half — every file in it is
reproducible by a committed script.

### Single source

`util.REPO_DATA_DIR` (override: `DESI_REF_DATA_DIR`) is the only place looked
at. Removed: the `~/data/desi/nz_slices` constant in `bao/core.py`, the
`~/data/desi/bao_{dataset}` lookups in `util.ntracers`, the `_repo_fallback`
helper, and `nz_slices_path`'s flat-layout branch (that existed only for the old
external layout). `make_nz_slices.py` and `make_lrg3_nz_slices.py` now write
into `data/{dataset}/nz_slices`, so regenerating produces a reviewable git diff
instead of silently changing a file outside the tree.

### `nz_slices`: 6 of 7 reproduce BYTE-IDENTICALLY

`make_nz_slices.py --out-dir <scratch>` then `cmp` against the committed files:
BGS, ELG2, LRG1, LRG2, LRG3, QSO all identical.

`LRG3_ELG1` does NOT, and the generator says why: `N_cat = 2,898,381` against
`N_dr1 = 1,876,187`, a 69.7% shape change. That table came from
`make_lrg3_nz_slices.py` reading the published `*_nz.txt` (final-survey sample),
not from the catalogues. Left alone rather than overwritten — only the SHAPE is
consumed (normalisation comes from `util.ntracers`), and which shape is right
for the BAO combined bin is a question, not a typo. Flagged, not fixed.

### `_desi_nx`: the recipe was not what it looked like

`make_desi_nx.py` is new. The obvious implementation — `WEIGHT`-weighted mean of
`NX` over randoms — reproduced the committed tables to only 4.8%, suspiciously
equal to §79's data-vs-randoms bias. Testing estimators against the committed
LRG2 table settled it:

```
WEIGHT * WEIGHT_FKP   0.071%   <- the recipe
unweighted            1.322%
harmonic              1.374%
WEIGHT alone          4.765%
```

The FKP factor is not a detail: Eq. (2.1)'s random density is `n_ran = S1 *
w_fkp`, so <NX> must carry the same weighting that appears in the z_eff weight.
And `S1` is a SUM, so it scales linearly with the file count — one random file
gives exactly 0.50063x the committed value, identifying nran=2.

Reproducible, not bit-exact: DESI's random files are independent realisations
and ours need not be the pair originally used. `--check` gives max|dNX|
0.06-0.10%, max|dS1| ~0.3%, and the z_eff that results lands within 0.02% of the
committed tables (LRG1 -0.101% -> -0.084%, LRG2 -0.029% -> -0.020%, LRG3
-0.055% -> -0.060% vs DESI 2024 V Table 1). Recorded in PROVENANCE.md so the
next person does not re-derive it.

This was the reproducibility hole: before §80 the six tables could not be
rebuilt by anything in the repo, and the recipe existed only in a scratch script
from §51-53 that was never committed.

### Deferred

The physical `~/data/desi/nz_slices/*.csv` files are no longer read by anything,
but deleting them is held until the §72 sweep finishes — those 24 processes hold
the pre-§80 code in memory. `rm -rf ~/data/desi/nz_slices` after that.

`LRG3_ELG1_desi_nx.csv` still does not exist; `make_desi_nx.py --tracers
LRG3_ELG1` will build it once the `LRG+ELG` randoms are fetched (~3.3 GB at
nran=1, ~6.6 GB at the nran=2 the other tables use). That is the last thing
keeping `nbar_file` alive.
