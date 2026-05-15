# V_eff fix for BAO Fisher forecast

## The problem

Pipeline σ on BAO distance parameters was optimistic vs DESI DR1 published
values — worst for QSO and ELG2 (factor of 2-3× tighter than DESI).

## Root cause

The pipeline was using the geometric shell volume

    V_shell = (4π/3) (χ_max³ − χ_min³) × f_sky

DESI quotes effective volume, which downweights regions where shot noise
dominates:

    V_eff = ∫ dV [ n̄(z) P_FKP / (1 + n̄(z) P_FKP) ]²

V_pipe / V_eff (DESI Y1 published) ratios:

| Tracer    | V_pipe (Gpc³/h³) | V_eff DESI | Ratio |
|-----------|------------------|------------|-------|
| BGS       | ~3               | 1.7        | ~1.8× |
| LRG1-3    | ~10-30           | 5-13       | ~2-3× |
| ELG2      | ~40              | 6.4        | ~6×   |
| QSO       | ~106             | 9          | ~12×  |

Fisher σ scales as 1/√V, so QSO was √12 ≈ 3.5× too tight, ELG2 √6 ≈ 2.5× too
tight. That matched the σ-gap pattern almost exactly.

## The fix

`_compute_v_eff_fkp` in `prep_covar.py` computes V_eff from first principles
per tracer:

1. Read published n(z) shape (fractional distribution `frac(z)`) from the
   `~/data/desi/nz_slices/<tracer>_nz_slices.csv` files.
2. Per z-bin: `N_per_bin = N_tracers × frac(z)`, `V_bin` from current
   cosmology, `n̄(z) = N_per_bin / V_bin`.
3. Compute the **signal-relevant fiducial power per redshift slice**:

       P_g(k_BAO, z) = b₁_eff² · ⟨Kaiser⟩(z) · P_lin(k_BAO, z=0)

   where:
   - `b₁_eff = bias_recon · σ₈(0)/σ₈(z_eff)` — effective Eulerian bias
     using the tracer's published linear bias and growth-factor scaling
   - `⟨Kaiser⟩(z) = 1 + 2β/3 + β²/5` with `β = f(z)/b₁_eff` — angle-averaged
     RSD enhancement
   - `P_lin(k=0.14, z=0)` — linear matter power at the BAO scale

4. FKP-weight per bin with the per-bin P_g:

       V_eff = Σ V_bin × [n̄(z) · P_g(k_BAO, z) / (1 + n̄(z) · P_g(k_BAO, z))]²

Then in `build_bao_likelihood` we scale **both** `area` and `N_tracers` by
`V_eff / V_shell` so n̄ stays at its actual peak value. (Scaling area only
inflates n̄, which partially cancels the V reduction.)

Always on: the V_eff FKP-effective covariance rescaling is unconditional
for standard galaxy tracers in `build_bao_likelihood` / `run_fisher`.
(Lya_QSO keeps V_shell since it has no n(z) FKP weighting.) The
`use_v_eff` flag was removed; V_eff is the only path.

## Why P_g(k_BAO) instead of fixed P_FKP

Standard FKP estimator uses a single fixed `P_FKP` value (DESI Y1 publishes
tracer-specific values: ~7000 BGS, ~10000 LRG, ~4000 ELG, ~6000 QSO; see
Table 2 of arXiv:2404.03000). DESI's choices are **calibrated** to
optimize their actual BAO SNR (post-window, post-recon).

We replace this analyst choice with the **theoretical signal-relevant
power** `P_g(k_BAO) = b₁²·Kaiser·P_lin`. This is:

- **Tracer-aware**: high-bias tracers (LRGs, QSOs) get larger effective P,
  giving them more V_eff weight where their BAO peak is most detectable.
- **z-dependent**: per-slice P captures growth factor and RSD evolution
  across broad-n(z) tracers (especially QSO).
- **No calibration**: uses only the tracer's published linear bias and
  the standard cosmology. No per-tracer tuning to match DESI's σ.

This replaces the FKP "fiducial power" hyperparameter with a theoretical
quantity, but otherwise follows the same FKP optimal-weighting machinery.

## Inputs the V_eff calculation uses

All standard forecast survey specs. None of these are DESI published σ or
V_eff numbers:

1. `N_tracers` — objects passing cuts (CSV input, a forecast input)
2. `area_deg2` — survey footprint
3. n(z) **shape** (fractional `frac(z)` per z-bin) from published n(z)
   slices — needed to know where tracers sit in redshift, same kind of
   input as N_tracers
4. Cosmology — from the current sample (Om, hrdrag), not fixed
5. `bias_recon` per tracer — from `tracers.yaml` (DESI Y1 published linear
   bias, e.g. b₁=2.0 LRG, b₁=1.2 ELG, b₁=2.1 QSO)

## What is NOT used

- No DESI published V_eff values
- No DESI published σ values
- No DESI published P_FKP values (tested, see "Trade-off" section below)
- No covariance_scale, α_stoch, or other calibration knobs

## Trade-off vs DESI's published P_FKP

We tested overriding the theoretical P_g(k_BAO) with DESI Y1 Table 2
values. Net result:

| Effect on σ residual | DESI P_FKP | Theoretical P_g(k_BAO) |
|---|---|---|
| LRG σ(DH) | better (~−8%) | over (~+1 to −29%) |
| LRG σ(DM) | better (−8% to −26%) | worse (−28 to −43%) |
| ELG2 σ | catastrophic (+108 to +156%) | mid (+26 to +65%) |
| QSO σ | catastrophic (+212 to +343%) | best (+11 to +57%) |

DESI's P_FKP is optimal for **their** Fisher (signal-dominated tracers
where their full window/recon machinery extracts all info). Our P_g(k_BAO)
is optimal for **our** simpler Fisher (which doesn't replicate DESI's
multi-tracer cross-correlations or window function effects).

Neither convention closes all residuals. We keep the theoretical P_g(k_BAO)
because (a) it's no-fudge, (b) it produces the smallest residuals on the
most catastrophic tracer (QSO), and (c) the LRG residuals are interpretable
as physics gaps (anisotropic Σ damping) rather than V_eff-convention
artifacts.

## Result (DR1, P_g(k_BAO), with V_eff + Lorentzian FoG + n_iter=3)

| Tracer       | DH F/D | DM F/D | DV F/D |
|--------------|--------|--------|--------|
| BGS          | –      | –      | -1%    |
| LRG1         | +1%    | -30%   | –      |
| LRG2         | -29%   | -43%   | –      |
| LRG3+ELG1    | -21%   | -34%   | –      |
| ELG2         | +65%   | +34%   | –      |
| QSO          | –      | –      | +11%   |

Residual physics gaps (>10%):

- **σ(DM) systematically low for LRGs** (-30 to -43%): pipeline analytic
  Σ_⊥ from 1-loop residual integral is too small at z>0.7. DESI's empirical
  Σ_⊥ doesn't drop as fast with z — likely needs higher-order PT contributions
  (2-loop sv2) or non-linear-bias amplification of BAO peak.

- **ELG2 σ(DH) too loose** (+65% DR1): low-bias tracer (b₁=1.2) gets less
  V_eff weight in our P_g scheme than DESI's analysis effectively achieves.
  DESI extracts more BAO info from ELGs than linear-bias signal-to-noise
  predicts (likely from non-linear bias for star-forming galaxies).

- **QSO σ DR2 too loose** (+38-57%): single-tracer Fisher convention can't
  fully replicate DESI's multi-tracer + windowed analysis for sparse
  high-z tracers.

## n(z) parsing fixes (2026-05)

`parse_desi_nz.py` `combine_nz_files` had two issues exposed by plotting
the unnormalized n(z) curves and comparing to DESI Y1 paper Fig 1:

1. **Multi-target combination (LRG3_ELG1)**: combining LRG+ELG via
   `Nbin_combined / Vol_combined` (with both summed across non-overlapping
   sky areas) gave a sky-area-weighted average instead of the additive
   joint density. Fix: bucket files by target name (NGC+SGC have same
   target), combine NGC+SGC for each target the old way (sum Nbin/Vol),
   then sum the per-target nbar values across targets (3D density is
   additive for populations on overlapping sky). Result: LRG3_ELG1 now
   smoothly connects to ELG2 at z=1.1.

2. **BGS sample**: pipeline default was `BGS_ANY` which includes far more
   low-mass low-z galaxies than DESI Y1 BAO actually uses (`BGS_BRIGHT-21.5`).
   Fix: pass `--bgs-target BGS_BRIGHT-21.5` to parse_desi_nz. Result: BGS
   peak n(z) drops from ~350×10⁻⁴ to ~7×10⁻⁴, matching DESI Fig 1.

Note: these fixes correct the n(z) **plot** to match DESI Y1 Fig 1, but
they don't materially change the Fisher σ output because the pipeline
uses `slice_fraction × N_BAO_passed / V_bin` for per-bin n̄, which is
shape-normalized and anchored to the BAO-sample passed count from
`desi_data.csv`. The absolute `nbar_file` column is unused in the Fisher
unless we explicitly switch to weighted-density mode.
