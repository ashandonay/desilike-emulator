# dr1/base_omegak_w_wa/config — extended-cosmo architecture optimization (LRG2 dev tracer)

Search for the best base-cosmo→**extended-cosmo** NN architecture for the
`base_omegak_w_wa` model in **config space**, dataset
`training_data/bao/dr1/base_omegak_w_wa/config/v1` (40k train / 10k test per
tracer). Branch `autoresearch/dr1-base_omegak_w_wa-config`.

## Problem setup

- **Inputs (6)**: `N_tracers, Om, Ok, w0, wa, hrdrag` (vs 3 for the `base` model:
  N_tracers, Om, hrdrag). Curvature + dynamical dark energy (w0, wa) add a much
  larger, more structured input manifold.
- **Outputs (3)**: `sigma_DH_over_rd, sigma_DM_over_rd, sigma_DV_over_rd`.
- Metric: standardized test MSE (`prepare.evaluate_test_mse`), seed-swept on the
  faithful full schedule (`validate.py`: 10000 epochs, lambda sched, 10 restarts,
  gamma 0.85). Screens (`screen.py`) only prune capacity — they mis-rank.
- Prior reference point: the `base` (3-input) model settled at **24/6/4** at a
  ~1e-7 floor. With 6 inputs and DE/curvature freedom the function is harder, so
  the capacity knee is expected to sit higher (the legacy `base_omegak_w_wa` YAML
  key used 48/6/4). Note: test-target std (0.59) < train (1.0) — heavier-tailed
  train samples; watch generalization gap.

## 1. Capacity screen (fast pre-filter, NOT decisive)

Wide width sweep at depth 6 / expand 4, plus depth-8 and expand-6 probes.
Compressed 1200-epoch single-cosine schedule (relative ranking only).

| dim | blocks | expand | params  | screen best_mse |
|-----|--------|--------|---------|-----------------|
| 24  | 6      | 4      | 28,611  | 3.32e-2 |
| 32  | 6      | 4      | 50,435  | 1.16e-1 |
| 48  | 6      | 4      | 112,515 | 2.69e-2 |
| 64  | 6      | 4      | 199,171 | 8.46e-2 |
| 96  | 6      | 4      | 446,211 | **2.13e-2** |
| 128 | 6      | 4      | 791,555 | 2.68e-2 |
| 48  | 8      | 4      | 149,859 | 1.49e-1 |
| 48  | 6      | 6      | 168,387 | 7.55e-2 |

Screen is single-seed (42) + compressed (1200 ep, 1 cosine) → very noisy and it
mis-ranks (32/64 and the depth-8/expand-6 probes screened far worse than their
d6e4 neighbours — init noise, not signal). The **trustworthy** signal is the clean
monotone width ladder at d6/e4: 24→48→96 = 3.32→2.69→2.13e-2, still falling at 96
(no knee). Conclusion: the 6-input extended-cosmo function needs **substantially
more capacity than base's 24/6/4**; floor location must come from `validate.py`.

## 2. Faithful seed sweep (`validate.py`, authoritative)

Phase 1 — locate the capacity knee with a single faithful run (seed 0) per width
along d6/e4: 48, 96, 144. Then seed-sweep (3 seeds) the 1–2 finalists.

| dim | blocks | expand | params | seed | best_mse | best_epoch | schedule |
|-----|--------|--------|--------|------|----------|------------|----------|
| 48  | 6 | 4 | 112,515 | 0 | 3.356e-2 | **500**  | 10-restart γ0.85 lr1e-3 |
| 96  | 6 | 4 | 446,211 | 0 | 7.054e-2 | **1000** | 10-restart γ0.85 lr1e-3 |
| 144 | 6 | 4 | ~1.0M   | 0 | 3.011e-2 | 2500     | 10-restart γ0.85 lr1e-3 |

**Phase-1 finding (decisive): the base-cosmo 10-restart schedule is wrong for
this config.** Every run hit its best right after warmup (best_epoch 500–2500) and
then *never improved* — the final low-LR cycle (epoch 9050→10000) gave 0.30–0.70,
far worse, not a deep minimum. On the rougher 6-input (w0/wa/Ok) landscape the
warm-restart LR peaks (1e-3) repeatedly destabilize the fit and it never re-settles.
Capacity is NOT the lever (48≈144≫better than 96 here is just init noise at a
non-converged point). The floor is also intrinsically far above base's 1e-7.
→ Pivot to **schedule search**: single cosine decay (no restarts), and lower peak LR.

## 3. Schedule search (single cosine, no restarts)

`validate.py` parametrized with `--lr/--restarts/--gamma/--eval-every`. Single
cosine = anneal once to a low LR and settle (minimum at epoch ~10000).

| dim | expand | seed | lr | restarts | best_mse (raw-σ) | note |
|-----|--------|------|-----|----------|----------|------|
| 96 | 4 | 0 | 1e-3 | 1 | 4.51e-2 | best behaved; still annealing when stopped |
| 96 | 4 | 0 | 3e-4 | 1 | 8.68e-2 | stuck at epoch-500 value |
| 48 | 4 | 0 | 1e-3 | 1 | 3.36e-2 | divergence spike to 1.3e+2 mid-run |

Single cosine is more stable than restarts but still caps ~1e-2 and still spikes —
because the metric is broken (see below). Runs stopped early once root cause found.

## 4. ROOT CAUSE — target distribution, not architecture/schedule

Inspected the raw LRG2 targets (`config/v1`). They span **~10 orders of magnitude**
with a catastrophic tail from the over-wide extended prior box:

| target | median | p99 | p99.9 | max |
|--------|--------|-----|-------|-----|
| σ_DH/rd | 1.8 | 32 | 4.5e4 | 5.4e6 |
| σ_DM/rd | 0.73 | 7.5 | 1.9e5 | 2.7e7 |
| σ_DV/rd | 0.51 | 8.1 | 8.8e4 | 1.3e7 |

- **99.95% of the standardized-target variance is in the top 0.1% of samples.**
  `prepare.py` z-scores raw σ, so σ-std is set by the ~1e6 outliers; the test-MSE
  metric measures fitting ~40 degenerate-cosmology points and ignores the other
  ~39,960. Those outlier batches are also what blow training up (loss → 130).
- Source: the prior box is enormous and includes pathological cosmologies —
  Om 0.055–0.99, hrdrag 10–1000, w0 −3→1, wa −3→2, Ok ±0.3. At near-degenerate
  Fisher points σ explodes. 99.78% of samples have all σ<1e3; 0.21% have σ>1e4.
- This is NOT an architecture or LR-schedule problem. No net can regress a target
  whose loss is 99.95% one outlier cluster.

**Fix is target representation (standard σ/covariance-emulator practice):**
emulate **log σ** (all targets >0; natural-log dynamic range collapses 10 orders →
~22, std ~1.8 — well-conditioned), exponentiate at inference. Optionally also
tighten/filter the prior box (the degenerate tail is non-physical and never visited
by a real DESI MCMC). Both are pipeline-level changes (data regen or a transform
layer + `exp` in the live ground truth / consumer), beyond pure architecture search
in `autoresearch/` — escalated to the user before proceeding. PAUSED here.

**The fix is already built in.** `train.py --log-normalize` symlog-transforms targets
(`sign·log1p(|y|/linthresh)`, per-col linthresh = min abs nonzero) before z-score and
stores it in the checkpoint; `eval.py` inverts it. Base never needed it (tame targets,
1e-7). Decision (user): **emulate log σ over the full prior box, no samples discarded.**

To make the sandbox faithful to that objective, `prepare.py` gained an opt-in
`EMULATOR_LOG_NORMALIZE=1` path (default OFF → base campaign unchanged) that applies the
identical symlog before its z-score. Verified: top-0.1% variance share drops 0.9995 →
0.04–0.08, std=1.0, targets bounded ~[−3.4, +10.5]. The raw-σ results in §2–3 are VOID
for architecture choice — redoing the search in log space below.

## 5. Architecture search in log-σ space (EMULATOR_LOG_NORMALIZE=1)

Re-running screen → faithful validate seed sweep in the production (symlog+zscore)
objective. Promotion will require training with `--log-normalize`.

### 5a. Capacity screen (log space)

| dim | blocks | expand | params  | screen best_mse |
|-----|--------|--------|---------|-----------------|
| 24  | 6 | 4 | 28,611  | 3.84e-2 |
| 32  | 6 | 4 | 50,435  | 3.81e-2 |
| 48  | 6 | 4 | 112,515 | **2.68e-2** |
| 64  | 6 | 4 | 199,171 | 3.23e-2 |
| 96  | 6 | 4 | 446,211 | **2.63e-2** |
| 128 | 6 | 4 | 791,555 | 3.44e-2 |

Far more consistent than raw-σ (no 1e-1 spikes). Knee ~48–96; 24/32 clearly worse
→ undercapacity at 24 (unlike base). Faithful seed sweep below decides 48 vs 96 and
the schedule (10-restart vs single cosine), now that the metric isn't outlier-broken.

### 5b. Faithful seed sweep (log space)

| dim | blocks | expand | seed | schedule | best_mse | best_epoch |
|-----|--------|--------|------|----------|----------|------------|
| 32 | 6 | 4 | 0 | 10-restart    | **1.79e-2** | 2250 |
| 48 | 6 | 4 | 0 | 10-restart    | 3.38e-2 | 4500 |
| 96 | 6 | 4 | 0 | 10-restart    | 2.45e-2 | ~3500 |
| 48 | 6 | 4 | 0 | single cosine | 2.46e-2 | 4000 |

Seed-0 ordering is non-monotonic (32<96<48) with scattered best-epochs (2250–4500,
not the late ~9000 of base) → **seed/cycle-variance dominated**; can't pick from one
seed. 32 and 96 are the two strong points. Seed-sweeping both (seeds 1,2) for 3-seed
triplets to decide capacity + reproducibility.

| dim | seed | schedule | best_mse | best_epoch |
|-----|------|----------|----------|------------|
| 32 | 0 | 10-restart | 1.794e-2 | 2250 |
| 32 | 1 | 10-restart | 2.344e-2 | 8000 |
| 32 | 2 | 10-restart | 2.097e-2 | 3250 |
| 96 | 0 | 10-restart | 2.446e-2 | ~3500 |
| 96 | 1 | 10-restart | 2.066e-2 | 4250 |
| 96 | 2 | 10-restart | 3.072e-2 | 1500 |

**3-seed means: 32/6/4 = 2.08e-2 (±0.27), 96/6/4 = 2.53e-2 (±0.50).** 32 has lower
mean error AND lower seed variance at **1/9 the parameters** (50k vs 446k). Best-epochs
scatter across cycles (high cycle-to-cycle variance) but the per-seed best_mse is
stable. 48 (seed-0 3.38e-2) is dominated. The base 10-restart schedule transfers as-is
in log space — single cosine was no better (48: single 2.46 vs r10 3.38 is within
seed noise), so no schedule change is warranted.

## 6. DECISION & promotion

**Winner: `ResNetRegressor` 32 dim / 6 blocks / expand 4, base schedule (lr 1e-3,
λ-sched, 10 restarts γ0.85, batch 256, wd 1e-5), trained with `log_normalize`.**
Only deltas from the `base` config: width 24→32 and mandatory log-normalization.

Promoted (config key `<release>_<cosmo_model>_<space>`):
- `bao/model_config.yaml` → new key **`dr1_base_omegak_w_wa_config`** (32/6/4 +
  `log_normalize: true`).
- `bao/model.py` UNCHANGED — 32/6/4 is the existing `ResNetRegressor` structure.
- `train.py` wired so `model_config.yaml: log_normalize` is honored as a default
  (CLI `--log-normalize` still overrides); fixed a duplicate-MLflow-param crash on
  collision between `vars(args)` and `model_cfg`.
- `eval.py` device made robust (works under `CUDA_VISIBLE_DEVICES` masking).
- Sandbox: `prepare.py` gained opt-in `EMULATOR_LOG_NORMALIZE=1` (mirrors train.py's
  symlog) so the search measured the production objective.

Real train (production path): MLflow run `9e8055979fee4df2b40c031da7166638`
(seed 0, 10000 epochs, LRG2) → **best standardized-log test_mse 6.92e-3** (train.py's
50-epoch eval grid catches a deeper minimum than validate's 250). Auto-eval over 1000
accepted prior samples (416 rejected as constraint-violating), live config-space ground
truth, %-errors after inverse-symlog:

| target | mean %-err | std %-err |
|--------|-----------|-----------|
| σ_DH/rd | +0.22% | 9.07% |
| σ_DM/rd | +0.40% | 8.81% |
| σ_DV/rd | +0.26% | 9.24% |

**Unbiased (<0.5%), ~9% scatter** over the full extended box (incl. degenerate w0/wa/Ok
points in the thin tail). Solid for a 6-param σ-emulator (base 3-param ≈ 3–4%). The raw-σ
per-target MAE/RMSE that train.py prints are tail-dominated and not meaningful for a
log-emulator — judge by these %-errors. Plots: `eval_*.png` in the run artifacts.

## Status: COMPLETE

Search → root-cause (target tail) → fix (log-σ) → seed-swept decision (32/6/4) →
promotion → production train+eval all done on branch
`autoresearch/dr1-base_omegak_w_wa-config`. Production deltas vs `main` to merge:
`bao/model_config.yaml` (new key), `train.py` (yaml log_normalize + MLflow dedup fix),
`eval.py` (device robustness), `autoresearch/prepare.py` (opt-in EMULATOR_LOG_NORMALIZE),
`autoresearch/program.md`. `experiments.md` stays branch-only (per-campaign artifact).
Only LRG2 trained; the other 5 tracers reuse this arch+config (retrain per tracer).
