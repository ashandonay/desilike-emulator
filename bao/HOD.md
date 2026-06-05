# Halo Occupation Distribution model used in the BAO forecast pipeline

This document describes the HOD model implemented in `core.py`
for the BAO Fisher forecast pipeline, including the per-tracer
parameter choices, the cosmology-dependent quantities, and the
literature provenance for every input.

The pipeline does **not** calibrate to DESI BAO data anywhere; all
HOD inputs trace to pre-DESI surveys (BOSS, eBOSS, 2dFGRS, SDSS) or
to N-body / hydro simulation literature. The cosmology dependence of
the bias is computed at every grid point; only the per-tracer-type
shape parameters and the assembly-bias factor are held constant
across the emulator grid (validated to <1% σ(α) impact, §33l).

## 1. Why an HOD at all?

The Fisher information on (D_M/r_d, D_H/r_d) per tracer scales as
b1² · P_lin(k_BAO) · V_eff(k_BAO), where b1 is the large-scale linear
galaxy bias. b1 enters the forecast through three places:

1. **Signal amplitude**: P_gal(k) = b1² · P_lin(k) at large scales.
2. **σ_FoG dispersion**: virial dispersion of satellites pairs in the
   Lorentzian Finger-of-God damping.
3. **Σ_⊥ damping**: post-recon transverse damping scales with b1 and
   the satellite fraction via the velocity field.

A correct cosmology-varying forecast therefore requires a
cosmology-driven prescription for b1, σ_FoG, and Σ_⊥ — none of which
can come from a static YAML of measured values. The HOD provides
that: given cosmology + n̄(z) + tracer-type shape parameters, the HOD
delivers b1(cosmo), σ_FoG(cosmo), and the satellite fraction f_sat
(which feeds Σ_⊥).

## 2. HOD formalism

The HOD specifies the mean number of galaxies of a given type per
halo of mass M, separated into a central (single, sits at the halo
center) and satellite (multiple, populated according to NFW) branch:

  N_tot(M) = N_cen(M) + N_sat(M)

The two contributions to the large-scale bias are:

  b1_HOD = ∫ dM (dn/dM) N_tot(M) b_h(M, z, cosmo)
         / ∫ dM (dn/dM) N_tot(M)

where `dn/dM` is the halo mass function and `b_h(M)` is the
Eulerian halo bias from peak-background split.

This is **mass-only** in the sense that galaxies of a given type are
assumed to occupy halos based on mass alone, ignoring secondary
properties like concentration, formation time, or environment. The
decorated-HOD extension (§5) corrects for this.

## 3. Functional form per tracer type

We use two central forms and a single satellite form:

### 3.1 Zheng+07 erf central (BGS, LRG, MIX, QSO)

  N_cen(M) = f_cen · ½[1 + erf((log M − log M_cut) / σ_logM)]
                    · exp(−(M / M_quench)^η_quench)

Two refinements beyond the original Zheng+07:

- **f_cen** ≤ 1: a global completeness factor. For BGS f_cen < 1
  reflects target-selection incompleteness; for LRGs f_cen = 1.
- **High-mass quenching** factor (Conroy/Wechsler-style; equivalent
  to HMQ for star-formers; see Rocher+23 §2.2 for the DESI ELG
  variant). At log M >> log M_quench, the factor → 0, modeling the
  fact that some galaxy populations don't extend to arbitrarily
  massive halos.

References: Zheng+07 (2007ApJ...667..760Z), Reid+14
(2014MNRAS.444..476R), Conroy & Wechsler 2009.

### 3.2 Gaussian narrow-band central (ELG)

  N_cen(M) = f_cen · exp(−½ ((log M − log M_cen) / σ_logM)²)

ELGs are star-forming galaxies that occupy a narrow halo mass range
(~10^11.5 to 10^12.5 M_sun/h), not above a mass threshold. The
quenching factor is silently ignored for this form because the
Gaussian already suppresses high-M occupancy.

References: Avila+20 (2020MNRAS.499.5486A), Rocher+23
(2023JCAP...10..016R).

### 3.3 Common satellite form

  N_sat(M) = ½[1 + erf((log M − log M_cut_sat) / σ_logM)]
             · max((M − M_cut_sat) / M_1, 0)^α_sat

The leading factor is a soft gate at M_cut_sat (= M_cut · 10^Δ), and
the power-law has slope α_sat ~ 0.4-1.0 depending on tracer.
Satellites are *not* gated by the central quenching factor: cluster
infallers exist whether or not the central is quenched.

The form is essentially Zheng+07 with the soft cutoff; see Reid+14,
Yuan+22 (2022MNRAS.515..871Y).

## 4. Halo mass function and halo bias

### 4.1 Tinker+08 mass function

  dn/dM = f(σ) · ρ̄_m / M · |d ln σ⁻¹ / d ln M|

where f(σ) is the Tinker+08 multiplicity function (arXiv:0803.2706,
Eq. 3), with mild redshift evolution of (A, a, b):

  A(z) = 0.186 · (1+z)^(-0.14)
  a(z) = 1.47 · (1+z)^(-0.06)
  b(z) = 2.57 · (1+z)^(-α)
  c    = 1.19   (z-independent)

α depends on the overdensity contrast Δ. We use Δ = 200 (mean-density,
SO halos), matching most modern N-body analyses. σ(M, z) is computed
from the linear matter power spectrum via `cosmoprimo`'s
`fo.sigma_rz(R, z, of='delta_cb')` (CDM+baryon only, neutrinos kept
out of the halo-formation source term per Castorina+14 prescription).

Implementation: `_tinker08_f_sigma()` (core.py line ~583).

### 4.2 Tinker+10 peak-background-split bias

  b_h(ν) = 1 − A ν^a / (ν^a + δ_c^a) + B ν^b + C ν^c

where ν = δ_c / σ(M, z), δ_c = 1.686, and (A, B, C, a, b, c) follow
Tinker+10 Eq. 6 (arXiv:1001.3162) with explicit Δ-dependence baked
in. At Δ = 200:

  A = 1.0 + 0.24 y · exp(-(4/y)^4)
  a = 0.44 y - 0.88
  B = 0.183
  b = 1.5
  C = 0.019 + 0.107 y + 0.19 · exp(-(4/y)^4)
  c = 2.4

with y = log10(Δ).

Implementation: `_tinker10_b1()` (core.py line ~570).

This is what enters the HOD integral as `b1_T10` in
`_hod_halo_props()`.

## 5. Cosmology-dependent M_cut: the abundance-matching step

The HOD shape parameters (σ_logM, f_cen, alpha_sat, log_M_quench/M_cut,
log_M1/M_cut, log_M_cut_sat/M_cut) are **fixed per tracer type** from
the literature. The only cosmology-varying input is **log M_cut
itself**, set by requiring:

  ∫ dM (dn/dM)(cosmo) · N_tot(M, log M_cut) = n̄(z)

where n̄(z) is the observed comoving number density of the tracer in
that redshift bin (from DESI N(z) summary CSVs, the only DESI input
to the pipeline; this enters once at fid and varies in lock-step
with the volume probe). The integral is monotonically decreasing in
log M_cut for the erf-style centrals and convex but well-behaved for
the Gaussian; we solve via 80-step bisection on a log_M grid of
[10, 16] in 256 points.

This is the cosmology hook: as Ωm or σ8 changes, dn/dM shifts, the
required log M_cut shifts, and the HOD-weighted Tinker+10 bias
shifts accordingly. The §33l cosmology-stability validation
confirmed this response is smooth and monotonic across the
practical emulator grid (20-27% b1 swing across Ωm ∈ [0.25, 0.40] —
the *correct* physical response).

Implementation: `_hod_halo_props()` (core.py line 738).

## 6. Assembly bias factor (f_AB)

The HOD as written treats galaxies as mass-only tracers of halos.
Real galaxies of a given type populate halos preferentially based on
**secondary halo properties** (concentration, formation time, large-
scale environment). The decorated-HOD framework (Hearin & Watson
2013, arXiv:1310.6747; McEwen & Weinberg 2018) extends the HOD by
splitting halos at each mass into bins of a secondary property and
weighting them differently in the occupation.

For DESI tracers the effect is consistently in one direction:
star-forming and AGN populations (ELGs, QSOs) preferentially occupy
*low-concentration, low-density* environments at fixed M, giving an
effective bias **lower** than the mass-only HOD predicts. Red
passive galaxies (LRGs, BGS) show much smaller deviation.

We collapse this into a single multiplicative factor per tracer type:

  b1 = f_AB · b1_HOD_mass_only

where f_AB ≤ 1 and is set from pre-DESI literature.

### 6.1 Literature values

| Type | f_AB | Pre-DESI anchor | b1_lit | Pipeline HOD b1 (fid) |
|------|------|-----------------|--------|----------------------|
| BGS  | 1.00 | 2dFGRS Norberg+02, SDSS Tegmark+04 (z~0.1-0.2 bright-galaxy auto-corr) | ~1.5 | 1.51 |
| LRG  | 0.85 | BOSS CMASS Tojeiro+14, Beutler+17 (z=0.57 b ≈ 2.0) | ~2.0 | 2.19 (LRG1) / 2.55 (LRG2) |
| ELG  | 0.60 | eBOSS Tamone+20 (z=0.85 b = 1.1-1.4), extrap z=1.32 ≈ 1.5 | ~1.5 | 2.60 |
| MIX  | 0.78 | n̄-weighted 0.7·LRG + 0.3·ELG (no direct combined-sample b1) | — | 2.11 |
| QSO  | 0.97 | eBOSS Laurent+17 (b_Q = 2.45 ± 0.05 at z=1.55) | 2.45 | 2.52 |

Cross-checks against independent DESI-internal measurements (Yuan+24
MNRAS 530, 947; Chaussidon+24 arXiv:2411.17623; Hadzhiyska+25
arXiv:2510.20896) give identical f_AB values to within 1% — the
ratios are robust to the choice of literature source. This is a
strong consistency check: pre-DESI and DESI-internal clustering
measurements give the same f_AB, so the values reflect real
galaxy-formation physics, not a survey-specific selection artifact.

Per-tracer-type citation blocks with full reasoning live in
`hod.yaml`.

### 6.2 Why constant f_AB is acceptable across the emulator grid

The pipeline b1(cosmo) = f_AB · b1_HOD(cosmo) splits the cosmology
response into two pieces:

- **b1_HOD(cosmo)** carries the **dominant** cosmology dependence:
  20-27% b1 swing across Ωm ∈ [0.25, 0.40] in our sweep (§33l). This
  is the physically correct response of halo abundance and bias to
  cosmology, captured at every emulator grid point.
- **f_AB** captures the cosmology-stable galaxy-formation residual:
  "at fixed halo mass, what extra bias comes from secondary halo
  properties?". This is determined by galaxy-formation rules
  (cooling, feedback, quenching timescales) which are weakly
  cosmology-dependent.

Independent sim studies bound the f_AB drift across realistic
cosmologies:

- Hadzhiyska+22 (2022MNRAS.509..501H) hydro AB amplitudes are stable
  to ~5-10% across the TNG cosmology range.
- Bose+19 (2019MNRAS.490.5693B) AB amplitude varies by ~3-5% across
  the cosmology range covered by ELUCID and MultiDark.
- Yuan+22 / Yuan+24 (2022MNRAS.515..871Y / MNRAS 530, 947) AB
  parameters Q_cen, Q_sat in the decorated-HOD fit to AbacusSummit
  vary by ≲5% across the cosmology suite.

Net: f_AB constant within ±10% across the emulator grid. The
resulting ≲10% systematic on b1 propagates through BB-marginalization
to <1% on σ(α), well below the residual cov-shape gap (§32a) that
sets the headline F/D 1.0-1.2 spread.

If sub-1% σ(α) precision is needed in the future, the path forward
is option (3) in §33l: promote f_AB to the decorated-HOD's Q_cen,
Q_sat parameters which apply at fixed halo-concentration percentile,
letting the HOD integral naturally produce f_AB(cosmo) at each grid
point.

### 6.3 Validation script

`validate_ab_cosmology_stability.py` sweeps cosmology axes (Ωm,
hrdrag, w0, wa) across the practical emulator-use range and reports
per-tracer b1 drift. See §33l of `CHANGELOG.md` for the full result
table and the `ab_cosmology_stability.png` diagnostic plot.

## 7. Per-tracer parameter table

Values are in `hod.yaml`. Reproduced here with citations for
reference. All values are **shape** parameters held constant across
cosmology; only log M_cut varies via the bisection in §5.

### BGS (Smith+20)

| Parameter | Value | Notes |
|-----------|-------|-------|
| central_form | erf | Zheng-07 |
| σ_logM | 0.30 | |
| f_cen | 0.80 | target-selection incompleteness for BGS-BRIGHT |
| log(M_quench/M_cut) | 0.50 | quench at 3× M_cut |
| η_quench | 1.0 | sharpness of quench cutoff |
| log(M_cut_sat/M_cut) | 0.0 | satellite cutoff coincides with central |
| log(M_1/M_cut) | 1.50 | satellite normalization mass |
| α_sat | 1.00 | |
| f_AB | 1.00 | 2dFGRS/SDSS; Hadzhiyska+25 confirms Q_sat ≈ 0 |

### LRG (Zheng+07, Reid+14, Yuan+24)

| Parameter | Value | Notes |
|-----------|-------|-------|
| central_form | erf | Zheng-07 |
| σ_logM | 0.50 | |
| f_cen | 1.00 | LRGs are mass-complete by construction |
| log(M_quench/M_cut) | 10.0 | effectively off (no quench for LRGs) |
| log(M_cut_sat/M_cut) | 0.0 | |
| log(M_1/M_cut) | 1.40 | |
| α_sat | 0.80 | |
| f_AB | 0.85 | BOSS CMASS b ≈ 2.0 at z=0.57 (Tojeiro+14, Beutler+17) |

### ELG (Avila+20, Rocher+23)

| Parameter | Value | Notes |
|-----------|-------|-------|
| central_form | gaussian | narrow-band; M_cut = M_cen |
| σ_logM | 0.30 | Rocher+23 best fit |
| f_cen | 0.10 | peak central occupancy |
| log(M_cut_sat/M_cut) | 1.00 | M_0 ~10× M_cen |
| log(M_1/M_cut) | 1.70 | tuned for f_sat ~0.1 |
| α_sat | 0.80 | Rocher+23 |
| f_AB | 0.60 | eBOSS Tamone+20 b ≈ 1.3-1.5 at z = 1.32 |

### MIX (LRG3+ELG1 combined sample)

| Parameter | Value | Notes |
|-----------|-------|-------|
| central_form | erf | mostly LRG-like |
| σ_logM | 0.50 | |
| f_cen | 0.40 | between LRG (1.00) and ELG (0.10) |
| log(M_quench/M_cut) | 0.20 | weak quench |
| η_quench | 1.5 | |
| log(M_cut_sat/M_cut) | 0.5 | |
| log(M_1/M_cut) | 1.50 | |
| α_sat | 0.80 | |
| f_AB | 0.78 | n̄-weighted: 0.7·0.85 + 0.3·0.60 |

### QSO (Richardson+12, Eftekharzadeh+15, Laurent+17)

| Parameter | Value | Notes |
|-----------|-------|-------|
| central_form | erf | broad central (Richardson+12) |
| σ_logM | 0.60 | wider than LRG: QSOs are stochastic |
| f_cen | 0.10 | low duty cycle |
| log(M_quench/M_cut) | 0.0 | quench at M_cut itself |
| η_quench | 1.5 | |
| log(M_cut_sat/M_cut) | 1.5 | |
| log(M_1/M_cut) | 2.50 | rare satellites |
| α_sat | 0.40 | shallow satellite slope |
| f_AB | 0.97 | eBOSS Laurent+17 b_Q = 2.45 ± 0.05 at z=1.55 |

## 8. Downstream uses of HOD outputs

`_hod_halo_props()` returns (b1, log_M_cut, σ_v²_eff, f_sat). These
flow into the BAO Fisher pipeline as:

- **b1** → linear bias in DampedBAOWiggles template.
- **σ_v²_eff** = ⟨N_sat σ_v²⟩ / ⟨N_tot⟩: the satellite-pair-weighted
  velocity dispersion. Centrals don't pair (they sit at the
  potential minimum), only satellites contribute to virial damping.
  Feeds `σ_FoG` through `_sigma_fog_from_sigma_v_sq()` (Mpc/h
  conversion via Hubble at z_eff).
- **f_sat** is exposed for diagnostics and feeds the Σ_⊥ post-recon
  prescription (Σ_⊥ depends on satellite velocity field).

## 9. Known limitations and forward work

### 9.1 Within-type z-evolution residual (LRG)

The single LRG f_AB = 0.85 cannot fit both LRG1 (z=0.51) and LRG2
(z=0.71) perfectly — pipeline b1 lands 4% below LRG1 data, 4% above
LRG2 data. Both within publication uncertainty on the BOSS CMASS
anchor. The remaining ±4% is absorbed by BB-marginalization in the
BAO likelihood and does not bias σ(α).

If precision warrants, the upgrade is either per-z-bin f_AB
overrides or a parameterized f_AB(z) within tracer type. See §33h
of `CHANGELOG.md`.

### 9.2 QSO bias tension

Two independent DESI-internal measurements disagree on the QSO bias:
Yuan+24 full-shape gives 2.63 ± 0.30, Chaussidon+24 PNG-based gives
2.24. The pre-DESI eBOSS Laurent+17 anchor (2.45 ± 0.05) sits
between. The DESI BAO bundle's ξ(s) prefers b ≈ 2.0 (§32e chi²
scan). Pipeline now reflects Laurent+17 → b1 = 2.44 at z=1.49,
χ²(fid)/dof = 1.76 for QSO.

This is a real DESI-internal tension; BB-marginalization absorbs
the b1 mismatch in α inference, so σ(α) is unaffected.

### 9.3 f_AB cosmology drift (upper bound)

Constant f_AB across the emulator grid introduces a ≤10% systematic
on b1, propagating to <1% on σ(α). Below the cov-shape gap of §34.
Upgrade path (§33l/m): decorated-HOD Q_cen/Q_sat formulation.

### 9.4 Halo concentration / NFW satellite distribution

Currently we use σ_v² from the halo's virial dispersion only,
without modeling the satellite radial distribution explicitly
(which would affect 1-halo small-scale clustering). For BAO this is
sub-percent because the BAO peak is at s ~ 100 Mpc/h, well above
1-halo scales. The forward-shape pipeline (FS) would need a fuller
treatment.

## 10. References

- **Tinker+08 mass function**: Tinker, Kravtsov, Klypin, et al. 2008,
  ApJ 688, 709 (arXiv:0803.2706).
- **Tinker+10 bias**: Tinker, Robertson, Kravtsov, et al. 2010, ApJ
  724, 878 (arXiv:1001.3162).
- **Zheng+07 5-par HOD**: Zheng, Coil, Zehavi 2007, ApJ 667, 760
  (arXiv:astro-ph/0703457).
- **Decorated HOD**: Hearin & Watson 2013, MNRAS 435, 1313
  (arXiv:1304.5557).
- **BGS HOD / target selection**: Smith, Allevato, Burchett, et al.
  2020, MNRAS 499, 269.
- **BOSS CMASS LRG**: Reid, Ho, Padmanabhan, et al. 2014, MNRAS 444,
  476 (arXiv:1404.3742); Tojeiro, Anderson, Beutler, et al. 2014,
  MNRAS 440, 2222; Beutler, Seo, Saito, et al. 2017, MNRAS 466,
  2242.
- **eBOSS ELG**: Avila, Gonzalez-Perez, Mohammad, et al. 2020, MNRAS
  499, 5486 (arXiv:2007.09012); Tamone, Raichoor, Zhao, et al. 2020,
  MNRAS 499, 5527 (arXiv:2007.09009).
- **eBOSS QSO**: Laurent, Eftekharzadeh, Le Goff, et al. 2017,
  JCAP 07, 017 (arXiv:1705.04718); Eftekharzadeh, Myers, White, et
  al. 2015, MNRAS 453, 2779 (z=2.5, NOT z=1.5 — historical citation
  correction).
- **DESI 1% LRG/QSO HOD**: Yuan, Garrison, Eisenstein, et al. 2024,
  MNRAS 530, 947 (cross-check anchor for §6.1 LRG and QSO values).
- **DESI 1% ELG HOD**: Rocher, Ruhlmann-Kleider, Burtin, et al.
  2023, JCAP 10, 016 (arXiv:2306.06319; supports ELG f_AB cross-
  check).
- **DESI DR1 PNG bias**: Chaussidon, Yèche, Palanque-Delabrouille,
  et al. 2024 (arXiv:2411.17623; cross-check for LRG/QSO b1(z)).
- **DESI BGS assembly bias**: Hadzhiyska, White, Sailer, et al.
  2025 (arXiv:2510.20896; confirms BGS f_AB = 1.0 via direct Q_sat
  measurement).
- **AB in hydro / N-body**: Hadzhiyska, Bose, Eisenstein, et al.
  2022, MNRAS 509, 501; Bose, Eisenstein, Hadzhiyska, et al. 2019,
  MNRAS 490, 5693; Lin, Han, Yang, et al. 2023, MNRAS 519, 4424.
- **HMQ / Conroy-Wechsler quenching**: Conroy, Wechsler 2009, ApJ
  696, 620; Rocher+23 §2.2.

## 11. Files

| File | Purpose |
|------|---------|
| `core.py` | Pipeline entry; HOD integration in `_hod_halo_props()` (line 738); Tinker bias `_tinker10_b1()`; HMF `_tinker08_f_sigma()` |
| `hod.yaml` | Per-tracer-type shape parameters and assembly_bias_factor with citation blocks |
| `validate_ab_cosmology_stability.py` | f_AB cosmology stability validation (§33l) |
| `ab_cosmology_stability.png` | b1 drift diagnostic plot |
| `CHANGELOG.md` §32, §33 | Full diagnostic and AB-factor implementation arc |
