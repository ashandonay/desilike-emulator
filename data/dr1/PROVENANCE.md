# DR1 reference tables (vendored)

Small DESI-derived tables committed to the repo so a fresh checkout runs the
forecast with **no downloads**. 39 KB total. `~/data/desi` still wins when
present; these are a fallback (util._repo_fallback / util.nz_slices_path).

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

Regeneration is reproducible but not bit-exact -- DESI's random files are
independent realisations and ours are not necessarily the pair originally used.
`make_desi_nx.py --check` gives max|dNX| 0.06-0.10%, max|dS1| ~0.3%, and the
resulting z_eff lands within 0.02% of the committed tables (LRG1 -0.101% ->
-0.084%, LRG2 -0.029% -> -0.020%, LRG3 -0.055% -> -0.060% vs DESI).

## Known gap

`LRG3_ELG1_desi_nx.csv` does not exist — the BAO combined bin. Building it
needs the `LRG+ELG_LOPnotqso` randoms (~3.3 GB, `init_desi_data.py --what
randoms --nran 1'). Until then `bao/core._desi_nz_geometry` falls back to
`nbar_file` for that bin, with a warning.

## Checksums

```
ebde2d1d8edcf822bac4148610b5ae367f441da0e6ed7a0aaef0fc6dbdeba210  desi_data.csv
4ea277c800b428fdf0f91feb888641fab23d71a3e8a649c59bfcb783e976300b  desi_tracers.csv
9164a906cdb577abf1d9f7002418e09164e13cd70ecb1d01336f86bc2299987d  BGS_desi_nx.csv
0fd47ea9569ac65a8f7723e545a3b36d69f8f7359ebd04582315936cf9d98929  BGS_nz_slices.csv
00de4dfca7d0415839c0c7b362e0b5b95730d113b1ff9274d8d56e3631a92fe1  ELG2_desi_nx.csv
a9870e474066721948a5a57086b91a3eb7a845c313ee576ca41e8bd8b63cd3e2  ELG2_nz_slices.csv
7319fc6e62966c052f05a4322a137e3b57e19fe79f6a58488f1be787c39fcfee  LRG1_desi_nx.csv
edd104836ae922782025abb96031bc0b915d6eae1191f4b4d0d75f2470bdad4e  LRG1_nz_slices.csv
d854ff5f851ca24b113ac6f1d450ebdeb8c5a769b287975e9b672ebae629a608  LRG2_desi_nx.csv
c12da3a007eed2d291a645bce5aebdf84a7c5418f220eb23f1c2103e963543f6  LRG2_nz_slices.csv
457738276a2107120089224b7a5825ad8b8e0d7540cff74bd1e08628ad632b28  LRG3_desi_nx.csv
8e2596e6edee947dbd559246c30d4826fa3509e43af8c0b34b0966bbf51d6873  LRG3_ELG1_nz_slices.csv
cd995a4b331adfec66ccca2a48d75008ceec9897529375b2b278c9c031d7c5ec  LRG3_nz_slices.csv
6b8825bd647954548dc453e2e56fa89c065bfe44191b08dfae6cd40268af9468  QSO_desi_nx.csv
1dc194270b410516e4476d17437a5bfef4d8f14ca95a7aa1ede366adebf1f411  QSO_nz_slices.csv
```
