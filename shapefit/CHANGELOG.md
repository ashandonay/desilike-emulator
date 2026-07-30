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
