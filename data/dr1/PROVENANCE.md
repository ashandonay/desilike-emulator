# DR1 reference tables (vendored)

Small DESI-derived tables committed to the repo so a fresh checkout runs the
forecast with **no downloads**. 39 KB total. These are the SINGLE source of
truth, not a fallback — there is deliberately no `~/data/desi` path (util.py
:214); point `DESI_REF_DATA_DIR` elsewhere to override.

Source: DESI DR1 (public). Cite DESI Collaboration 2024 (arXiv:2503.14745)
if these are used. Regenerate with `init_desi_data.py` + the scripts below;
nothing here is hand-edited.

## Files

| file | what | produced by |
|---|---|---|
| \`{tracer}_nz_slices.csv\` | n(z) shape: zmid/zlow/zhigh/slice_fraction/nbar_file | \`shapefit/make_nz_slices.py\` from \`*_clustering.dat.fits\` |
| \`{tracer}_desi_nx.csv\` | \`nbar_desi_nx\` = selection-weighted <NX>, \`S1_weight\` = sum WEIGHT, per slice (2411.12020 Eqs. 8.1-8.3) | reduced from DESI v1.5 **randoms** |
| \`desi_data.csv\` | per-tracer \`passed\` counts + published BAO sigma | DESI 2024 III/V tables |
| \`desi_tracers.csv\` | per-component targets/comp/efficiency, for bins declaring \`components\` | DESI 2024 V Table 1 |

## Why `_desi_nx.csv` is vendored rather than regenerated

These ~10 KB are the whole reason the 20.9 GB of randoms exist. They are the
reduction, so shipping them means no user needs the randoms.

They CANNOT be rebuilt from the data catalogues without loss. `NX` must be
averaged over the **selection function**, which is what randoms sample;
averaging over galaxies instead weights by the density itself and biases the
result 4.8% high with 0.41% z-dependent structure. Measured effect on z_eff
vs DESI 2024 V Table 1, six tracers:

```
S1 randoms + NX randoms   0.062% mean   <- these files
S1 data    + NX randoms   0.064%
S1 randoms + NX data      0.117%
S1 data    + NX data      0.111%
```

`S1` is a sum, so data/randoms differ by a constant (0.0772) that cancels in
Eq. (2.1)'s ratio; its residual 0.30% scatter is at the 0.36% Poisson floor.
`NX` is a density-weighted mean of a density and does not cancel. See
shapefit/CHANGELOG.md S79.

## The `_desi_nx` recipe (recovered in S80)

`nbar_desi_nx` is `NX` averaged with weights **`WEIGHT * WEIGHT_FKP`**, over
**2** random files per cap; `S1_weight` is the plain `sum(WEIGHT)` over the same.
The FKP factor is not optional -- Eq. (2.1)'s random density is `n_ran = S1 *
w_fkp`, so <NX> must carry the same weighting that appears in the z_eff weight.
Measured against the committed LRG2 table, per estimator:

```
WEIGHT * WEIGHT_FKP   0.071%   <- this one
unweighted            1.322%
harmonic              1.374%
WEIGHT alone          4.765%
```

`S1` scales linearly with the file count (it is a sum), which is how nran=2 was
identified: one file gives exactly 0.50063x the committed value.

Regeneration IS bit-exact. An earlier note here recorded max|dNX| 0.06-0.10%;
that was measured before the full random set was local, against tables built
from a different pair of realisations. With all 20 files present
`make_desi_nx.py --check` now reports **0.000%** on every column of all eight
tables -- including the four originally reduced on NERSC by
`nersc_make_desi_nx.py`, so the two generators are identical, not merely
consistent.

## Parent decompositions (S88)

`ELG1_desi_nx.csv` is not an analysis bin — DR1 fits 0.8<z<1.1 as the combined
`LRG3_ELG1`, and `tracers.yaml` has no `ELG1` entry, so nothing in the pipeline
consumes it. It is vendored because it is the ELG half of that combined bin,
selected out of the same `LRG+ELG_LOPnotqso` randoms by `WEIGHT_RF` (finite for
ELG rows, NaN for LRG rows), and it makes `nbar_total` auditable: the two
parents sum to it exactly. Rebuild with `make_desi_nx.py --tracers ELG1`; the
LRG half is `--tracers LRG3p` (not vendored, it exists to be diffed).

Both parents were validated against the standalone catalogues they also appear
in — the LRG half against `LRG`, the ELG half against `ELG_LOPnotqso` over
0.8<z<1.1. Unweighted `<NX>` and row counts match at **0.00000000**, i.e. the
`WEIGHT_RF` split recovers the populations exactly. Two things do differ, both
expected and neither a defect:

- `WEIGHT` in the combined file carries Eq. (4.14a)'s per-tracer bias `b_t` —
  measured as a constant 1.9900 (LRG) and 1.6176 (ELG). z_eff inherits it, which
  is why LRG3_ELG1 agrees with DESI to -0.077%.
- `WEIGHT_FKP` uses the *combined* n̄_eff (Eq. 4.13, P0=6000), so the LRG rows'
  back-solved pivot runs 11298→51652 across the bin instead of the flat 10000
  they carry in the LRG file. Since `nbar_desi_nx` is an FKP-weighted mean, the
  LRG half sits up to 2.1% below `LRG3_desi_nx.csv`. Each table is internally
  consistent with the catalogue it came from; they are not interchangeable.

## Checksums

```
ebde2d1d8edcf822bac4148610b5ae367f441da0e6ed7a0aaef0fc6dbdeba210  desi_data.csv
4ea277c800b428fdf0f91feb888641fab23d71a3e8a649c59bfcb783e976300b  desi_tracers.csv
52b677d47c0684cbf4bd223e77fc9a42330576e3283d41a1c71dcb7b7211cac4  BGS_desi_nx.csv
0fd47ea9569ac65a8f7723e545a3b36d69f8f7359ebd04582315936cf9d98929  BGS_nz_slices.csv
c0c226ffb131b2a7273ba8cf27fce75c958224a15fe9e04a369349f5e35f7127  ELG1_desi_nx.csv
4038ec3323a8b8b4497baad331e7dcf677c40c6a6ca308da51e35d5d29142f0e  ELG2_desi_nx.csv
a9870e474066721948a5a57086b91a3eb7a845c313ee576ca41e8bd8b63cd3e2  ELG2_nz_slices.csv
b09324c2afa0ce57163d4478abb5604942df8119792e89de6381fa8a22b3ca96  LRG1_desi_nx.csv
edd104836ae922782025abb96031bc0b915d6eae1191f4b4d0d75f2470bdad4e  LRG1_nz_slices.csv
659152bce91cb6ca8474a45754e3075bc29803184b82d78400a5a728e73ece2d  LRG2_desi_nx.csv
c12da3a007eed2d291a645bce5aebdf84a7c5418f220eb23f1c2103e963543f6  LRG2_nz_slices.csv
64fd70df5f935f7409ce163197b389d0ad4db3be9046690292ddaba3c04a10a1  LRG3_desi_nx.csv
f91816d771b94848f0c2c14d5c372ec43bf895d00128b4a77e98ac1d391339f1  LRG3_ELG1_desi_nx.csv
0a29951a097b01ef2923df859811f2923708f557b93e86fd57d9670b226c0101  LRG3_ELG1_nz_slices.csv
cd995a4b331adfec66ccca2a48d75008ceec9897529375b2b278c9c031d7c5ec  LRG3_nz_slices.csv
9e6da0497f1702b656ab7f62005d689899ecf2fcfe9283120c61c5a57a859e5b  QSO_desi_nx.csv
1dc194270b410516e4476d17437a5bfef4d8f14ca95a7aa1ede366adebf1f411  QSO_nz_slices.csv
```
