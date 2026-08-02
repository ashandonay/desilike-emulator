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
