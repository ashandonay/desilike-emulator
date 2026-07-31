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
generate_covar_data.py   covar training-data CLI (per tracer)
generate_mean_data.py    mean training-data CLI (per tracer)
generate_training_data.sh  all-tracer driver, --quantity covar|mean,
                           one shared auto-versioned v{N} folder
regress_sigmas.py        bit-exact dump/compare regression harness
validate_forecast.py     fiducial σ table, σ(N) scaling, damping, kmax
validate_mean.py         mean-pipeline checks (mapping, AP, shape, covar)
desi_reference.py        DESI 2024 V App. A datavectors + 4×4 covariances,
                         and App. C Table 11 fiducials (the comparison anchor)
compare_to_desi.py       forecast vs DESI data products (shot, pk, cov, sigma)
comparison_plots.py      sigma / rho / rhomat / mean plots vs DESI
make_lrg3_nz_slices.py   LRG-only n(z) slices for the full-shape 0.8–1.1 bin
model.py                 emulator architecture (registered for analysis=shapefit)
model_config.yaml        NN hyperparameters (key "default" = all cosmo models)
```

Dependency direction: `core.py` → (`bao/core.py`, `util.py`); `fourier_space.py`
→ `core.py`; the generators/harnesses sit on top.

---

## Analysis configuration (decisions)

| choice | value | why |
|---|---|---|
| Template | `ShapeFitPowerSpectrumTemplate(apmode="qisoqap", with_now="wallish2018")` | `with_now` MUST be explicit; desilike's `'peakaverage'` default mislabels σ ~2× and crashes chaotically across wide priors. `dn` stays fixed — un-fixing it changes the definition of `m`. |
| **Theory** | **`REPTVelocileptorsTracerPowerSpectrumMultipoles`** (pluggable `theory_cls`) | **This is DESI's own baseline** — 2024 V §4.7 item 2, *"We select velocileptors with its EPT option as our baseline choice"*. Kaiser remains available but is a different model, not an approximation: it under-reports σ by 1.6–2.1× and inverts two ρ signs on every tracer (§15/§16). Audited over the full prior box: REPT adds **zero** failures relative to Kaiser (§22). |
| Nuisances | `prior_basis='physical'` → b1p, b2p, bsp, alpha0p, alpha2p, sn0p, sn2p — all Schur-marginalised | Matches DESI Table 4 **exactly**, including prior widths. `prior_basis='physical'` is what puts SN0 in units of 1/n̄ and SN2 in f_sat σ_v²/n̄, which is DESI's normalisation. Per-tracer preset (fsat, sigv) resolved by `core.default_theory_kwargs` from `tracers.yaml`. |
| Fit range | ells (0, 2), klim `[0.02, 0.2, 0.005]` | Matches DESI: §4.7 item 3 drops the hexadecapole deliberately ("it causes stronger prior weight effects"), and item 4 sets 0.02 < k < 0.20. |
| Tracers | BGS, LRG1, LRG2, **LRG3**, ELG2, QSO | **LRG3 is LRG-only, not LRG3+ELG1.** ELG1 failed DESI's pre-unblinding fibre-collision tests for growth-rate measurements and is excluded from full shape, while remaining in the BAO analysis (§2). `tracers.yaml` is analysis-scoped for exactly this reason (§31/§32). No Lya. |
| Recon | none (pre-recon everywhere) | Full shape is a pre-recon analysis: no Σ_post, no smoothing_scale/bias_recon, no shifted-random shot term. |
| Damping | `theory_fiducial_params` per theory; under REPT only `b1p` is set | The BAO de-wiggling scales must NOT be fed into a full-spectrum Gaussian damping — doing so drove P2 negative above k≈0.155 (§14). `float_sigma_damp` is a no-op under REPT, which has no `sigmapar`. |
| Broadband | none polynomial; the EFT counterterms and stochastic terms above ARE the small-scale freedom | DESI full-shape has no polynomial broadband either. Do not graft the BAO `pcs` basis onto FS, and do not tune nuisance choices to close the ratio to DESI (bao §33r error-cancellation lesson). |
| Window | **none** (diagnostic hook only) | Measured, not assumed: applying DESI's window consistently (`C_obs = M C_kin Mᵀ`) changes LRG2's σ from 0.82/0.85/0.91/0.87 of DESI to 0.81/0.79/0.85/0.80, and every ρ by <0.03. Derivative smoothing and covariance correlation cancel. Worth ~36 s/cosmology to move σ ~5% in the wrong direction (§19). ⚠ `wmatrix=` alone convolves the derivatives but NOT the covariance — read the `core.py` comment before using it. |
| Cosmology basis | `omega_cdm, omega_b, h, ln10A_s, n_s` (+ `w0`, `wa`) | Full shape constrains the P(k) shape/amplitude — no (Om, hrdrag) compression. No Ω_k models yet. |
| Domain constraint | derived Ω_m ∈ [0.01, 0.99] (`core._check_omega_m`, fail-fast) | The raw omega box reaches Ω_m ~ 17; ~58% of the raw LHS box is out-of-domain and rejected before any CLASS init (measured acceptance 42.2%, §22). In-domain failure rate is 0 for **both** theories. **The bedcosmo prior must carry the same constraint.** |
| FS band kernel | `dF/dk ∝ k²` over [0.02, 0.2] | Replaces the BAO Silk kernel in the FKP V_eff → n_eff mapping and the z_eff weight. |
| Fiducial cosmology | cosmoprimo `("DESI", …)` = AbacusSummit c000 | DESI 2024 V Table 6 row 1; §4.7 item 10 uses this one cosmology as BOTH grid and template cosmology. Verified: our fiducial distances reproduce their Table 11 to ≤0.13%. |
| Cov | Gaussian (FKP-effective volume) + SSC | Shared `bao/core.py` machinery; recon-shot term dropped (pre-recon). |

**Target contract.** `TARGET_NAMES` = 4 `sigma_*` + 6 `rho_*`. The name
prefixes are load-bearing: `util.transform_emulator_targets_*` and the
bedcosmo decode guards dispatch on them. Pairwise ρ-clamps only guarantee PSD
per 2×2 block — the bedcosmo-side 4×4 assembly must add an eigenvalue-floor /
nearest-PSD projection (follow-up).

**Physical basis.** desilike varies `(qiso, qap, df, dm)`;
`f_sigmar = df · f_sigmar_fid` and `m = m_fid + dm`, so the Jacobian is
`diag(1, 1, f_sigmar_fid, 1)`.

**`m` follows DESI's convention — no conversion needed.** The mean emulator
emits DESI's Eq. (4.9) shape parameter, which multiplies the fiducial template
so `m = 0` means no shape change. Verified: the generator returns
m = −4.5e−05 at the DESI fiducial cosmology.

⚠ Watch the naming collision: **desilike's `m` is a different quantity** — the
absolute log-slope of the de-wiggled spectrum at k_p, ≈ −0.5775 — and desilike's
`dm = m − m_fid` is what equals DESI's m. The mean worker therefore reads
`extractor.dm` into a target named `m`. Emitting the deviation also keeps §24's
theory-dependent `m_fid` out of the interface entirely (REPT's attached template
reports −0.6699 where the extractor says −0.5775). σ is offset-invariant, so the
covar targets `sigma_m` / `rho_*_m` are unaffected and consistently named.

Note the training box reaches m ∈ [−10.4, +3.7] because the Ω_m prior spans
shapes far from the fiducial; DESI's measured values are ~0.05.

---

## Environment

```bash
cd ~/desilike-emulator/shapefit
LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
  ~/miniconda3/envs/emulator/bin/python <script.py> [args]
```

`SCRATCH` must be set. Pinned deps: desilike @ `4cfd6bec`, cosmoprimo @
`1b100803`, `lsstypes` (install from SHAs, never bare `main`). `velocileptors`
provides both the REPT theory and the 1-loop displacement variances.

---

## 1. Emulator training data

```bash
# all 6 tracers -> next free v{N}
./generate_training_data.sh --quantity covar --cosmo-model base --n-samples 512
./generate_training_data.sh --quantity mean  --cosmo-model base --n-samples 512

# one tracer into an existing version
./generate_training_data.sh --quantity covar --version 1 --tracers LRG3
```

- Output: `$SCRATCH/bedcosmo/num_tracers/emulator/shapefit/training_data/dr1/{cosmo_model}/{covar|mean}/v{N}/{tracer}_{train,test}.npz`.
  Note this is **6 levels deep** — a shallow `find` will miss it.
- `--n-samples` is *accepted* rows; the generator redraws against the ~42%
  Ω_m acceptance, so it costs attempts, not samples.
- `--maxtasksperchild` defaults to 50. **Do not disable it** — the covar
  workers leak and a wide pool has OOM-killed the box.
- The `N_tracers` box comes from `util.ntracers_range`. **Never hardcode N.**
- REPT costs ~3.2× Kaiser per accepted sample (~10.7 s vs ~3.3 s).

## 2. Validation against DESI

The external anchor is `desi_reference.py` — DESI 2024 V Appendix A
(ShapeFit-alone datavectors + full 4×4 covariances) and Appendix C Table 11
(fiducial values). Use the **ShapeFit-alone** fits, not the tighter
ShapeFit+BAO ones, since this forecast is power-only.

```bash
python compare_to_desi.py --check shot pk cov sigma --tracers LRG2
python comparison_plots.py all          # sigma / rho / rhomat / mean
python validate_forecast.py --check fiducial
```

**Current scorecard** (σ generator/DESI, REPT, DR1 counts):

| | σ(qiso) | σ(qap) | σ(fσ_r) | σ(m) |
|---|---|---|---|---|
| BGS | 0.61 | 0.95 | 0.67 | 0.67 |
| LRG1 | 0.99 | 1.01 | 1.03 | 1.07 |
| LRG2 | 0.81 | 0.85 | 0.91 | 0.87 |
| LRG3 | 0.68 | 0.66 | 0.73 | 0.73 |
| ELG2 | 0.65 | 0.58 | 0.54 | 0.87 |
| QSO | 0.80 | 0.83 | 0.74 | 0.98 |

The forecast is systematically tighter than DESI. There is **no trend with
redshift or density** — corr(ratio, z_eff) = −0.128 and corr(ratio, n̄P₀) =
+0.199, and BGS is the worst agreement at the *lowest* redshift (§33; an
earlier "degrades with redshift" claim was a five-of-six cherry-pick). LRG1 is
the only tracer that matches on all four.

Two ρ offsets are systematic across all six tracers: ρ(qap, fσ_r) too strong
by 0.09–0.16, ρ(fσ_r, m) too weak by 0.17–0.36.

The **mean** pipeline is separately validated and in good shape: it reproduces
DESI's Table 11 fiducial fσ_s8 to ≤0.4% on four tracers (LRG1 exact to 4
decimals), with LRG3 and QSO off by 0.7% and 4.4% for known sample and z_eff
reasons. qiso/qap reproduce independent distance ratios to 7e−5.

**Leading explanation for the σ deficit, unresolved.** Our shot-noise floor is
`V/N` = 3595 for LRG2 against DESI's measured `num_shotnoise/norm` = 5229.5 —
a factor **1.4545**. Propagating that alone predicts a covariance ratio of
0.852 against the 0.815 measured on the correctly-rotated comparison, so it
accounts for most of the gap; the residual is Fisher-vs-MCMC. What is *ruled
out*: the V_eff-vs-I₁₂/I₂₂ definition (§26a), mean completeness (§26c/§27,
both by Cauchy–Schwarz and empirically), and the slice parser bug (§28). What
is *not* established: the cause. Three area normalisations disagree, and the
required area (10,908 deg²) matches none of them. **The same n̄ construction is
shared with `bao/`** (§29). Read CHANGELOG §26–§29 before reopening — several
plausible mechanisms have already been proposed and killed.

## 3. Regression harness

```bash
python regress_sigmas.py dump --out before.npz    # then change the dep
python regress_sigmas.py dump --out after.npz
python regress_sigmas.py compare before.npz after.npz
```

Exact-equality compare over a fixed 6-tracer × 8-cosmology grid. Run
before/after any desilike/cosmoprimo/scipy/numpy change — these outputs are
emulator training labels. Baseline: `golden_4cfd6bec.npz` (gitignored).
**Always verify a new baseline is reproducible** by dumping twice and comparing
— that bit-exactness is the harness's entire purpose.

---

## Follow-ups

- **Fetch the 5 missing DR1 full-shape bundles.** Only LRG2 is local, and the
  covariance files carry placeholder shot-noise fields (`num_shotnoise = 0,
  norm = 1`). They unblock the cross-tracer shot-noise test, the window
  question beyond LRG2, and the ELG2 disagreement.
- Resolve the shot-noise / effective-area discrepancy (above). Not a fudge
  factor — a constant would not vary with the design variable.
- bedcosmo integration: `models.yaml` `shapefit:` block, extend
  `_build_emulator_input` to the omega basis, and 4×4 assembly with a PSD
  guard. No `m` conversion is needed — the emulator already emits DESI's
  convention.
- Ω_k cosmology models; DR2 anchoring (the `tracers.yaml` `overrides`
  mechanism is wired but unpopulated); a DESI FS systematic-error layer.
- MCMC cross-check of the Fisher σ before trusting labels tracer-by-tracer.
- Cosmetic: `plot_rho_matrix` / `rhomat` / `shapefit_rho_matrix_vs_desi.png`
  now plot covariance, not correlation — the names are misnomers.
