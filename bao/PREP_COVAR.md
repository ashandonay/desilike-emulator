# `prep_covar.py` — BAO Fisher pipeline reference

Given a cosmology + tracer count + tracer key, `prep_covar.py` produces a desilike Gaussian likelihood for P_ℓ(k), runs `desilike.Fisher` to get the 2×2 marginalized Fisher on (qpar, qper), and converts to σ(DH/rd), σ(DM/rd). For DV-only tracers (BGS, QSO) the same path returns a 1×1 Fisher on the AP-isotropic combination.

This doc walks the **physics pipeline in data-flow order**, naming the specific external-package calls at each stage. CHANGELOG cross-refs (§N) point to design / audit decisions.

---

## 1. Cosmology in

The driver passes `{N_tracers, Om, hrdrag, [Ok, w0, wa]}`. `_to_bao_cosmo_params` converts it to:

- a desilike `theta_cosmo` dict — `Omega_m, Omega_k, w0_fld, wa_fld, h, omega_b, n_s, logA`
- a separate `hrdrag = h × rd` in Mpc

**Fixed at fiducial values** (set the linear-P(k) shape but not varied in the BAO prior): `omega_b = 0.02237`, `m_ν = 0.06 eV` → `omega_ν`, `n_s = 0.9649`, `ln(10¹⁰ A_s) = 3.044`, `h = 0.6736`. Only `hrdrag` floats — `r_d` is recovered downstream as `hrdrag / h_FID`.

`ω_cdm = Om × h_FID² − ω_ν − ω_b` (raises if Om is too small to give positive ω_cdm).

---

## 2. Cosmoprimo background + power spectrum

```python
cosmo = get_cosmo(**theta_cosmo)        # cosmoprimo.Cosmology
fo    = cosmo.get_fourier()             # fourier-space accessor
```

From this point the pipeline pulls:

- **Distances**: `cosmo.comoving_radial_distance(z)` (returns Mpc/h directly — do not divide by h; §8)
- **σ₈(z)**: `fo.sigma8_z(z, of="delta_cb")` and `of="theta_cb"`; `f(z) = σ₈,θ / σ₈,δ`
- **Linear P(k, z)**: `fo.pk_interpolator(of="delta_cb").to_1d(z=z)(k)` via `_linear_pk_1d` — Mpc³/h³, k in h/Mpc

---

## 3. Effective redshift

`_compute_z_eff_from_nz` returns the Fisher-info-weighted z_eff over the n(z) slices:

```
z_eff = Σᵢ Vᵢ (n̄ᵢ P_g(k_BAO, zᵢ) / (1 + n̄ᵢ P_g))² zᵢ  /  Σᵢ [same weight]
```

`P_g = b1² P_lin(k_BAO, zᵢ)`. Volumes `Vᵢ = (4π/3)(χ_hi³ − χ_lo³) × f_sky` come from the cosmoprimo χ(z). Matches Adame+24 Sec 2.5 (§18). No DESI σ values used — the calculation is cosmology-clean.

---

## 4. HOD → galaxy bias, virial dispersion, satellite fraction

`_hod_halo_props(n̄_target, z, cosmo, fo, tracer_type)` does the cosmology-dependent half of the HOD chain:

1. Build the HMF on a log-M grid via **Tinker+08** `f(σ)` (`_tinker08_f_sigma`) and the per-mass linear bias via **Tinker+10** (`_tinker10_b1`).
2. Build `N_cen(M)`, `N_sat(M)` from the per-tracer-type Zheng+07-style HOD shape parameters (`hod.yaml`). LRG/BGS/QSO use erf centrals + quench; ELG uses the Gaussian-peak central form (Avila+20 / Rocher+23; §24c).
3. Bisect `log_M_cut` until `∫ dn/dlnM × (N_cen + N_sat) dM = n̄_target`.
4. Return `b1 = ⟨N_tot × b1_T10⟩ / ⟨N_tot⟩`, `σ_v²_eff = f_sat × ⟨σ_v²_virial⟩_sat` (centrals sit at the potential minimum and contribute zero to FoG), `f_sat`.

HOD shape parameters are cosmology-independent literature inputs (§2, §22b).

---

## 5. Galaxy bias adjustments + LOS smearing

Inside `_get_standard_tracer_bao_params`:

- **Interloper suppression**: `b1 → b1 × (1 − f_interloper)` (§25). Reads `tracers.yaml` per tracer.
- **Redshift-error inflation**: `σ_v²_total = σ_v²_eff + (z_err × c)²` (§10b). Non-zero only for QSO by default.
- **FoG damping scale**: `σ_FoG = √σ_v²_total × (1+z) / (100 E(z))` (Mpc/h, comoving LOS). Enters the BAO observable as the Lorentzian `sigmas` parameter — **not** added in quadrature to Σ_∥ (§5).

---

## 6. Reconstruction damping Σ_⊥, Σ_∥

`_sigma_nl_post_from_recon` does an anisotropic (k, μ) integration with the iterative Wiener filter:

```
P_g(k, μ)   = (b1 + f μ²)² P_lin(k)
W(k, μ)     = P_g / (P_g + 1/n̄)             # Wiener filter
S(k)        = exp(−k² R² / 2)                # smoothing kernel
T_res(k, μ) = (1 − S W)^n_iter               # residual transfer
```

then

- `Σ_⊥² = (1/2π²) ∫ dk P_lin ∫₀¹ dμ ½(1−μ²) T_res²`  + 1-loop sv²
- `Σ_∥² = (1+f)² × ⟨⟩_∥` + 1-loop velocity-divergence corrections + 1-loop sv²
- Plus isotropic random-catalog shot displacement `σ_shot² = (1/(6π² b1² n̄)) ∫ S²(k) dk`

The 1-loop LPT pieces `(sv², sv²_dot, sv²_ddot)` come from `velocileptors.LPT.velocity_moments_fftw.VelocityMoments` via `_sigma_v_sq_1loop_rsd` (§14). The naive `(1+f)²` boost on Σ_∥ assumes `sv²_dot = sv²_ddot = sv²`; the 1-loop result gives typically `sv²_dot ≈ 2 sv²`, `sv²_ddot ≈ 3 sv²`.

**Key bookkeeping**: Σ_post uses `b1 = cfg["bias_recon"]` — DESI's analysis-stage value — not the HOD `b1`. The HOD `b1` describes the underlying tracer-halo relation; `bias_recon` is the analysis choice that drove the actual Wiener filter, and Σ_post should reflect that specific filter (§22a). `n_iter = 1` per §10 (DESI documents N=3 but the effective behaviour is N≈1).

---

## 7. FKP-effective volume and effective density

The full survey volume `V_shell = (4π/3) f_sky × Σᵢ (χ_hi,i³ − χ_lo,i³)` overstates the modes the BAO Fisher can resolve when n(z) is sparse. Two outputs from this stage:

- **Signal V_eff** via `_compute_v_eff_fkp`:  
  `V_eff = Σᵢ Vᵢ × ⟨(n̄ᵢ P_g / (1 + n̄ᵢ P_g))²⟩_band` with the BAO Fisher band kernel `W²(k) = k⁴ exp(−k² Σ_silk²)` (`_fkp_band_weight_sq`).
- **Effective 3D density n_eff** via Brent root-find:  
  Solve `⟨(n_eff P_g(k, z_eff) / (1 + n_eff P_g))²⟩_band = V_eff / V_shell` for a single n_eff that reproduces the band-integrated V_eff at the slice's effective redshift. More accurate than a single-k_peak mapping because FKP²(k) has finite width in k.

---

## 8. Survey footprint into desilike

```python
footprint = CutskyFootprint(
    area  = area,                                # deg², physical
    zrange= zrange,
    nbar  = n_eff × V_shell / area,              # angular: deg^-2
    cosmo = cosmo,
)
```

`CutskyFootprint` stores a scalar `nbar` as angular density and recovers `nbar_3d = size / volume = (nbar × area) / V_shell`. With the rescaling above, the internal `nbar_3d` lands at exactly `n_eff`, while `footprint.volume = V_shell` (physical geometry). desilike's single FKP² weight applied at `n_eff` then reproduces V_eff by construction — no double-counting, no separate covariance-volume rescaling (§6, §31).

Lya_QSO bypasses this block; its `nbar_for_footprint` is the angular density of sightlines (§13).

---

## 9. BAO theory + observable

```python
template = BAOPowerSpectrumTemplate(z=z_eff, fiducial=..., apmode='qparqper')
theory   = DampedBAOWigglesTracerPowerSpectrumMultipoles(
    mode='recsym',                       # 'recsym' for galaxies, '' (pre-recon) for Lya
    smoothing_radius=cfg['smoothing_scale'],
    model='fog-damping',
    template=template,
)
observable = TracerPowerSpectrumMultipolesObservable(
    klim={0: [kmin_eff, kmax], 2: [kmin_eff, kmax]},
    theory=theory,
)
```

- `kmin_eff = max(0.02, 2π / V_shell^(1/3))` — simple survey-window cutoff. The full window projection adds < 0.2% (§26).
- `kmax = kmax_cov or 0.30` h/Mpc.
- Lya_QSO uses `mode=''` (pre-recon, no Wiener) and a flux-bias template; see §13 below.
- AP parameters: qpar, qper are the only varied template parameters; b1, Σ_⊥, Σ_∥, σ_s float with Gaussian priors centered on the analytical values, scale 2.0 Mpc/h (§21).

---

## 10. Covariance assembly

```python
C_gauss = ObservablesCovarianceMatrix(
    observable,
    footprints=footprint,
    resolution=resolution,        # default 3
)(**fid_params)
```

This is the Grieb+16 Gaussian-disconnected cov_P, diagonal in k, dense in ℓ, with the `(2ℓ+1)(2ℓ'+1)` ℓ-coupling factors and a `2 / N_modes(V_shell)` overall normalization.

Two non-Gaussian additions:

- **Super-sample covariance** (`_ssc_cov`): rank-1 outer product `σ_b² R Rᵀ` where `σ_b²` is the variance of the background mode in a top-hat sphere of radius `R_eff = (3 V_shell / 4π)^(1/3)` and `R` is the linear response of P_ℓ(k) to a long-wavelength mode (§16c).
- **Shifted-random shot inflation** (`_shifted_random_noise_cov`): post-recon shot-noise boost from the finite-α random catalog. Computes the analytic difference `[P + (1 + 1/α) N]² − [P + N]²` so the desilike baseline isn't modified in place. With `α = 50`, ~2% inflation on the shot piece (§9, §28c).

Total:

```python
C_total = C_gauss + C_SSC + C_recon_shot
likelihood = ObservablesGaussianLikelihood(observables, covariance=C_total)
```

---

## 11. Fisher and conversion to (DH/rd, DM/rd)

```python
F_full = desilike.Fisher(likelihood)()    # Hessian in all free params
```

`_q_fisher_from_bao_likelihood_info` picks out the qpar/qper rows and Schur-complement-marginalizes over everything else (b1, Σ_⊥, Σ_∥, σ_s):

```
cov_q = (F_qq − F_qθ F_θθ⁻¹ F_θq)⁻¹     # 2×2 on (qpar, qper)
```

Cholesky-then-solve for stability, falls back to pinv if `F_θθ` is singular. A small ridge `1e-12 × tr(F)` is added before factorisation (`_regularize_fisher_matrix`).

Converted to physical units via the Jacobian `diag(DH_fid/rd, DM_fid/rd)`:

```
cov(DH/rd, DM/rd) = diag(DH_fid/rd, DM_fid/rd) × cov_q × diagᵀ
```

Returned as the upper triangle keyed by `cov_DH_DH, cov_DH_DM, cov_DM_DM`. The DV-only entry points (BGS, QSO) project onto the isotropic combination before inverting.

---

## 12. Entry points

| Function | Use |
|---|---|
| `get_bao_fisher_covariance(sample, tracer_bin, ...)` | Single-z_eff path. Builds one likelihood at `z_eff` and runs the Fisher. |
| `get_bao_fisher_covariance_sliced_nz(sample, tracer_bin, nz_slices_path, ...)` | Per-slice path (§30a). Loops over the n(z) CSV rows, builds a separate likelihood at each `z_mid` with `N_i = N × slice_fraction_i`, and **sums the per-slice 2×2 (qpar, qper) Fisher matrices**. Inverts at the bin-level effective redshift. More correct for wide-bin tracers (ELG2, QSO); ~30–45× slower. Opt-in. |
| `compute_pipeline_sigmas(sample, tracer_bin, ...)` | Returns the analytical `(Σ_⊥_post, Σ_∥_post)` without building the full likelihood. Used by the override path. |

---

## 13. Lyα QSO path

Lyα QSO bypasses HOD, Σ_post-from-recon, and the V_eff/n_eff machinery — it is pre-reconstruction with an effective-white-noise template.

- **Bias**: `b_F(z) = b_F0 × ((1+z)/(1+z_ref))^γ` (Arinyo-i-Prats+15, `_lya_flux_bias`).
- **Σ_⊥, Σ_∥**: fiducial values rescaled by growth — `Σ × (σ₈(z) / σ₈_fid)`, with an additional `f^0.5` factor on Σ_∥.
- **Noise**: effective white-noise floor `P_N_eff = σ_pix² × Δr_pix / n_sightline_3D` — replaces `1/n̄` in the cov.
- **Footprint**: angular sightline density only; no FKP rescaling.
- **Theory mode**: `DampedBAOWigglesTracerPowerSpectrumMultipoles(mode='')` (pre-recon).

---

## 14. Cross-reference to CHANGELOG

| Stage | CHANGELOG sections |
|---|---|
| Cosmology + fiducial constants | §1, §2 |
| HOD + HMF + bias | §2, §3, §4, §22b, §24c |
| 1-loop sv² (velocileptors) | §14 |
| V_eff / n_eff (FKP) | §5, §6, §31 |
| Pre/post Σ_NL damping | §5, §10, §17a, §20a, §22a |
| Σ_post prior widths | §21 |
| `n_iter` choice | §10 |
| Shifted-random shot | §9, §28c |
| SSC | §16c |
| z_eff from n(z) | §18, §19 |
| f_interloper | §25 |
| z_error_kms | §10b, §28b |
| Broadband nuisance priors | §27 |
| Window-function approximation (kmin) | §26 |
| Sliced-nz Fisher | §30a |
| Cosmoprimo χ unit fix | §8 |
| Residual-σ closure | §28d, §29, §30 |

---

## 15. Pipeline diagram

```
sample {N_tracers, Om, hrdrag, ...}
      │
      ▼
 _to_bao_cosmo_params      ──►  theta_cosmo + hrdrag
      │
      ▼
 cosmoprimo: get_cosmo, fo, σ8(z), P_lin(k,z), χ(z)
      │
      ▼
 _compute_z_eff_from_nz    ──►  z_eff   (Fisher-info-weighted)
      │
      ▼
 _hod_halo_props           ──►  b1, σ_v², f_sat   (Tinker+08/+10 HMF/bias)
      │       │
      │       ▼
      │  _sigma_fog_from_σ_v²    ──►  σ_FoG (Lorentzian sigmas)
      ▼
 _sigma_nl_post_from_recon ──►  Σ_⊥, Σ_∥   (uses bias_recon, not HOD b1)
      │                          (+ velocileptors 1-loop sv²)
      ▼
 _compute_v_eff_fkp        ──►  V_eff, V_shell
      │
      ▼
 Brent root-find           ──►  n_eff  s.t.  ⟨FKP²(n_eff, P_g)⟩_band = V_eff/V_shell
      │
      ▼
 CutskyFootprint(area, zrange, nbar = n_eff × V_shell / area, cosmo)
      │
      ▼
 BAOPowerSpectrumTemplate(z_eff, apmode='qparqper')
 DampedBAOWigglesTracerPowerSpectrumMultipoles(mode='recsym', sigmas, smoothing_radius)
 TracerPowerSpectrumMultipolesObservable(klim, theory)
      │
      ▼
 ObservablesCovarianceMatrix(observable, footprint, resolution)   ──►  C_gauss
      │
      ▼
 + _ssc_cov          (rank-1 σ_b² R Rᵀ)
 + _shifted_random_noise_cov   (α=50 random-catalog shot inflation)
      │
      ▼
 ObservablesGaussianLikelihood(observables, covariance=C_total)
      │
      ▼
 desilike.Fisher(likelihood)         ──►  F_full
      │
      ▼
 _q_fisher_from_bao_likelihood_info  ──►  cov_q (2×2 on qpar, qper)
      │
      ▼
 × diag(DH_fid/rd, DM_fid/rd)        ──►  cov(DH/rd, DM/rd)
```
