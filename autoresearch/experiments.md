# dr1/base/config/v1 — LRG2 architecture optimization

Data: `bao/dr1/base/config/v1`, tracer LRG2.
Task: base cosmo model, 3 inputs (N_tracers, Om, hrdrag) -> 3 outputs
(sigma_DH/rd, sigma_DM/rd, sigma_DV/rd). 24k train / 6k test.
Metric: standardized test MSE (lower is better).

## Exp1: baseline — 24dim, 6blocks, expand=4, 10 restarts, gamma=0.85, batch=256, LR=1e-3
Known-good base config from program.md. Establishes the global-best reference
for this campaign before any architecture changes.
**Result**: (running)
