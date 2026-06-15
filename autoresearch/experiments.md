# dr1/base/config — LRG2 architecture optimization

Data: `bao/dr1/base/config`, tracer LRG2.
Task: base cosmo model, 3 inputs (N_tracers, Om, hrdrag) -> 3 outputs
(sigma_DH/rd, sigma_DM/rd, sigma_DV/rd).
Metric: standardized test MSE (lower is better).

**2026-06-14: switched v1 -> v2** (40k train / 10k test, +20k/bin over v1's
24k/6k). Exp1/Screen1/Validation1 below were on v1; Validation2 onward use v2.

Infra notes:
- DataLoader-over-GPU-tensors replaced by per-epoch GPU index permutation in
  train.py (math-neutral; cuts host<->device overhead). Runs are launch-bound
  (~0.2 s/epoch) for this tiny model, not compute-bound.
- `torch.compile` is unusable in the `emulator` env (Triton launcher fails to
  build against system CUDA headers: CUlaunchConfig undefined). Eager only.
- results.tsv logging switched to scientific notation (`%.6e`): test_mse drops
  below 1e-6, where `%.6f` floored to 0.000000 and broke global-best compares.

## Exp1: baseline — 24dim, 6blocks, expand=4, 10 restarts, gamma=0.85, batch=256, LR=1e-3
Known-good base config from program.md; global-best reference for the campaign.
**Result**: best test_mse ~7e-7 (epoch ~3000+; sub-1e-6 by epoch 5000).
The metric saturates at a ~7e-7 noise floor — capacity is not the bottleneck.

## Screen 1: architecture grid (single seed=42, compressed 1200-epoch / 1-restart schedule)
Fast relative-ranking sweep (dev/screen.py) over dim x blocks x expand to find
the SMALLEST model that still reaches the floor (program.md simplicity criterion).

| dim/blk/exp | params | best_mse |
|-------------|-------:|---------:|
| 8/4/4       |   2267 | 1.09e-6  |
| 16/4/2      |   4403 | 9.20e-7  |
| 12/4/4      |   4935 | 9.50e-7  |
| 12/6/4      |   7359 | 7.37e-7  |
| 16/4/4      |   8627 | 1.00e-6  |
| 16/6/4      |  12883 | 8.24e-7  |
| 24/6/2      |  14427 | 7.96e-7  |
| 24/4/4      |  19083 | 7.72e-7  |
| 24/6/4 (base)|  28539 | 7.37e-7 |
| 24/8/4      |  37995 | 7.27e-7  |
| 32/6/4      |  50339 | 7.99e-7  |
| 48/6/4      | 112371 | 8.44e-7  |

Finding: whole 2k–112k-param range spans only 7.3e-7–1.1e-6 (all ~floor).
Bigger-than-baseline does NOT help. **12/6/4 (7359 params) matches the 28539-param
baseline exactly (7.37e-7) at 3.9x fewer params** — the simplicity winner.
Single-seed deltas are within plausible noise -> validate faithfully + seed sweep.

## Validation 1: 12/6/4 vs 24/6/4 baseline, faithful full schedule, 3 seeds
Faithful run (dev/validate.py): full 10000-epoch / 10-restart / gamma=0.85 /
batch=256 schedule (mirrors train.py), best test_mse over the whole schedule.

| seed | 12/6/4 (7359p) | 24/6/4 (28539p) | best_epoch |
|-----:|---------------:|----------------:|-----------:|
| 42   | 6.241e-07      | 2.468e-07       | 9000       |
| 123  | 6.217e-07      | 6.245e-07       | 6000/9000  |
| 7    | 6.348e-07      | 2.683e-07       | 7000/9000  |
| mean | 6.27e-07       | 3.80e-07        |            |

**The screen mis-ranked.** Its compressed 1200-epoch schedule could not reach the
deeper minimum that 24/6/4 finds only after many restart cycles (best_epoch 9000).
On the full faithful schedule the extra capacity DOES help: 24/6/4 reaches ~2.5e-7
on 2/3 seeds vs 12/6/4's rock-steady ~6.3e-7 (12/6/4 plateaus, can't go lower).

Lesson: short screens rank capacity-to-floor, NOT the late deep minimum from LR
restarts. Faithful seed-swept runs are the only trustworthy comparison here.

**Both are far past practical sufficiency** (standardized outputs): 6.3e-7 ~ 7.9e-4 σ
RMS, 2.5e-7 ~ 5.0e-4 σ RMS — both <0.1% of the output spread. Choice is a
simplicity (12/6/4, 3.9x fewer params) vs absolute-floor (24/6/4, ~2.4x lower MSE)
trade-off, not an accuracy-limited one.

## Validation 2: same two candidates on v2 (40k/10k), faithful full schedule, 3 seeds
Re-run of Validation 1 on the larger v2 dataset (dev/val_v2_*).

| seed | 12/6/4 (7359p) | 24/6/4 (28539p) | best_epoch |
|-----:|---------------:|----------------:|-----------:|
| 42   | 6.036e-07      | 1.144e-07       | 9000/9000  |
| 123  | 2.175e-07      | 3.059e-07       | 9000/10000 |
| 7    | 6.021e-07      | 1.662e-07       | 9000/10000 |
| mean | 4.75e-07       | 1.96e-07        |            |

More data helped both vs v1, but helped the BASELINE more. Verdict is cleaner than
v1: **24/6/4 reaches a lower AND more consistent floor (mean 1.96e-7, best 1.14e-7)
vs 12/6/4's noisier 4.75e-7** (small net plateaus on 2/3 seeds). The larger model
exploits the richer data; capacity now matters slightly more. Both still <0.1% σ RMS
(4.4e-4 vs 6.9e-4). Open: with more data, does 32/48-dim beat baseline's 1.96e-7?

## Validation 3: upward capacity sweep on v2 (32/6/4, 48/6/4), faithful, 3 seeds
Answering the open question — does more capacity beat baseline on the richer data.

| config  | params  | mean best_mse | best seed | seeds (42/123/7)            |
|---------|--------:|--------------:|----------:|-----------------------------|
| 12/6/4  |   7359  | 4.75e-7       | 2.18e-7   | 6.04 / 2.18 / 6.02 e-7      |
| 24/6/4  |  28539  | 1.96e-7       | 1.14e-7   | 1.14 / 3.06 / 1.66 e-7      |
| 32/6/4  |  50339  | 1.89e-7       | 1.62e-7   | 1.87 / 2.18 / 1.62 e-7      |
| 48/6/4  | 112371  | 1.32e-7       | 6.89e-8   | 0.69 / 1.06 / 2.19 e-7      |

Capacity curve on v2: big jump 12->24 (4.75->1.96e-7), FLAT 24->32 (tied, 1.89 vs
1.96), modest 32->48 (1.89->1.32e-7). **24/6/4 sits at the knee.** 48/6/4 is the
genuine MSE winner (~1.5x below baseline, best seed 6.9e-8) but costs 4x the params
for a ~1.5x gain on already-negligible error (3.6e-4 vs 4.4e-4 σ RMS).

Decision = knee/simplicity (24/6/4) vs absolute floor (48/6/4); 32/6/4 ruled out
(no gain over baseline).

## FINAL (2026-06-14): adopt 24dim / 6blocks / expand4 (28,539 params)
Chosen as the production base-cosmo architecture for dr1/base/config/v2 (LRG2).
Rationale: sits at the capacity/accuracy knee — 32-dim gives no gain, and the
48-dim improvement (1.96e-7 -> 1.32e-7) is negligible in practice (both <0.1% σ
RMS) at 4x the params. Matches program.md's simplicity criterion. This is the
config already in train.py (HIDDEN_DIM=24, N_HIDDEN=6, EXPAND=4); LR schedule,
batch, optimizer unchanged from the known-good baseline.

Campaign takeaways:
- The base model is data/floor-limited, not capacity-limited; metric ~1e-7 (<0.1% σ).
- Compressed screens rank capacity-to-floor but MISS the late LR-restart deep
  minimum — only faithful, seed-swept, full-schedule runs are trustworthy here.
- More data (v1->v2) lowered the floor and mildly rewarded capacity, but not
  enough to move off the 24-dim knee.
