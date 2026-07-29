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

Fixed: `core.DATASET_AREAS` + `_DEFAULT_AREA = DATASET_AREAS["dr1"]` drive the
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
