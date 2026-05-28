# AGENTS.md

Guide to the primary BAO Fisher-forecast testing scripts and plots in `bao/`.
This repo is a working tree for the DESI BAO Fisher/emulator forecast; the
day-to-day work is comparing the production pipeline's forecast errors against
DESI DR1 measurements.

## Environment

All `bao/` scripts must run with the `emulator` conda env, from inside `bao/`:

```bash
cd ~/desilike-emulator/bao
LD_LIBRARY_PATH=~/miniconda3/envs/emulator/lib:$LD_LIBRARY_PATH \
    ~/miniconda3/envs/emulator/bin/python <script>.py
```

The `cwd` must be `bao/` (scripts use relative imports of sibling modules such
as `prep_covar`, `fisher_sigmas`, `util`).

## DESI reference

The forecast is compared against DESI DR1 per-tracer **`bao-recon` stat-only**
likelihood files (`~/data/desi/bao_dr1/likelihoods/likelihood_bao-recon_*.h5`),
which hold DESI's post-marginalization α-covariance. This is the authoritative
reference for σ(D_H/r_d), σ(D_M/r_d), σ(D_V/r_d) — **not** `desi_data.csv`
(which carries a fiducial-conversion offset). DESI publishes D_V for the sparse
tracers (qiso: BGS, QSO) and (D_H, D_M) for the multipole tracers (qparqper:
LRG/ELG); the unused channels are masked in plots.

The `correlation-recon-poles` files are a *different* product (raw ξ + cov +
window matrix W) used as the "bundle" — DESI's RascalC covariance — for cov
substitution tests.

## 1. Forecast σ vs DESI bao-recon (the money plot)

**`plot_fisher_vs_desi.py`** — writes a σ-comparison PNG into `bao/` (the
filename embeds the fiducial Om/hrdrag, so it varies).

Three stacked panels (D_H/r_d, D_M/r_d, D_V/r_d) over the 6 DR1 tracers, with
five markers per (tracer, quantity):

| marker | source |
|---|---|
| black ✕ | DESI bao-recon σ (the reference) |
| blue ● | production Fisher, analytic Gaussian cov |
| blue ▲ | production MCMC (analytic cov posterior σ) |
| orange ■ | Fisher with DESI bundle (RascalC) cov substituted |
| orange ◆ | MCMC with bundle cov |

```bash
~/miniconda3/envs/emulator/bin/python plot_fisher_vs_desi.py
```

- The **Fisher** markers (●/■) recompute fresh on every run.
- The **MCMC** markers (▲/◆) load cached chains from `mcmc_results/`,
  `mcmc_results_bundle_native/`, and seed-sweep means in
  `seed_sweep/sigma_of_sigma.json` — they do **not** re-run MCMC. Re-running the
  chains (expensive; sparse-tracer chains need a seed sweep, see below) is a
  separate step before the MCMC markers reflect pipeline changes.

**`fisher_sigmas.py`** — library (no `__main__`) backing the plot. Compute σ
directly without the plot via:
- `_prod_fisher_sigmas(tracer)` — production Fisher (analytic cov).
- `_bundle_fisher_sigmas(tracer, cov_override=None)` — native ξ-theory Fisher
  with the bundle cov (or any cov passed via `cov_override`).
- `_recon_sigmas(tracer, fid)` — DESI bao-recon σ converted to σ(D/r_d).

> **Pitfall:** do not read σ values off a saved PNG — regenerate or compute via
> `fisher_sigmas`. Saved plots can lag committed pipeline changes (the blue
> Fisher markers in particular).

Current state: production analytic-cov Fisher under-predicts DESI bao-recon by
a fairly uniform ~23% (the missing non-Gaussian/covariance-structure content);
the bundle-cov cases bracket DESI well.

## 2. Covariance correlation-matrix comparison

**`compare_cov_matrices.py`** — writes the correlation-matrix comparison PNG(s)
into `bao/` (one for the full matrices, one for the diagonals).

Side-by-side correlation matrices r[i,j] = C[i,j]/√(C[i,i]C[j,j]) for the
analytic ξ-cov (left) vs the bundle RascalC ξ-cov (right), shared colorbar, per
tracer — plus a diagonal-σ²(s) overlay. Shows *where* in the (s,s′) cov the
analytic Gaussian approximation diverges from the bundle.

```bash
~/miniconda3/envs/emulator/bin/python compare_cov_matrices.py
```

## 3. Pipeline theory vs bundle data at fiducial

**`compare_pipeline_theory_to_bundle.py`** — writes the pipeline-theory-vs-bundle
comparison PNG into `bao/`.

Checks whether the pipeline's theory ξ at the fiducial cosmology matches the
DESI bundle data (per tracer: P-space theory → FFTLog to ξ on the bundle theory
s-grid → apply bundle W → compare to bundle data with bundle cov, Schur-marg
over DESI broadband {s⁰, s²}). Low χ²(fid) after BB-marg means the theory is
fine and only the covariance differs from the bundle. Provides `load_bundle`
and `bb_basis`, reused by `fisher_sigmas`.

```bash
~/miniconda3/envs/emulator/bin/python compare_pipeline_theory_to_bundle.py
```

## Supporting / diagnostic scripts in `bao/`

- `prep_covar.py` — core pipeline: `build_bao_likelihood`, `run_fisher`,
  `get_bao_fisher_covariance`. Has a `cov_engine` switch
  (`desilike` default, `fkp_analytic`, `thecov_cached`, `thecov_hybrid`).
- `gaussian_xi_cov.py` — analytic Gaussian config-space (ξ) multipole cov
  (Grieb 2016); validated correct but Gaussian-only.
- `fkp_analytic_cov.py` — FKP-1994 Gaussian P-space cov (matches desilike).
- `analytic_xi_cov_diagnostic.py`, `compare_cov_matrices.py`,
  `diagnose_*` / `validate_*` — covariance diagnostics.
- `seed_sweep/` — per-seed MCMC σ for noise estimation (sparse tracers QSO/BGS
  require this; single 2500-iter chains are not reproducible).
