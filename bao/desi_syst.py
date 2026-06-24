"""DESI systematic-error inflation layer — dependency-free (numpy only).

Split out of ``desi_reference.py`` so the post-emulator systematic σ inflation
(the frozen ``DESI_SYST_INFLATION`` table + :func:`apply_desi_syst`) can be
imported without pulling in ``desilike`` (via ``core``/``fourier_space``). This
lets the downstream ``bedcosmo`` runtime (which has no ``desilike``) apply DESI's
systematic budget on top of the emulator's statistical σ. ``desi_reference.py``
re-exports these names, so it remains the canonical reference module.
"""
from __future__ import annotations

import numpy as np

# DESI DR1 systematic-error inflation σ_tot/σ_stat per tracer per distance,
# measured as σ(bao-recon_syst)/σ(bao-recon_stat-only) (the syst file IS σ_stat⊕sys,
# so this ratio already folds in the quadrature). Clamped to ≥1 (systematics only
# inflate; sub-noise negatives → 1.0, e.g. ELG2 D_H). Isotropic tracers (BGS, QSO)
# carry the single qiso factor on all three (DH/DM/DV are degenerate projections).
# Regenerate with `python desi_reference.py --regen-syst` (needs the _syst .h5).
DESI_SYST_INFLATION = {
    "BGS":       {"DH_over_rs": 1.0071, "DM_over_rs": 1.0071, "DV_over_rs": 1.0071},
    "LRG1":      {"DH_over_rs": 1.0428, "DM_over_rs": 1.0020, "DV_over_rs": 1.0241},
    "LRG2":      {"DH_over_rs": 1.0063, "DM_over_rs": 1.0122, "DV_over_rs": 1.0243},
    "LRG3_ELG1": {"DH_over_rs": 1.0538, "DM_over_rs": 1.0306, "DV_over_rs": 1.0598},
    "ELG2":      {"DH_over_rs": 1.0000, "DM_over_rs": 1.0653, "DV_over_rs": 1.0623},
    "QSO":       {"DH_over_rs": 1.0056, "DM_over_rs": 1.0056, "DV_over_rs": 1.0056},
}
_SIGMA_KEYS = ("DH_over_rs", "DM_over_rs", "DV_over_rs")  # SIGMA_TARGET_NAMES order


def apply_desi_syst(sigma_triplet, tracer, ratios=None):
    """Inflate emulator σ_stat → σ_tot with DESI's measured per-tracer systematic
    budget (`DESI_SYST_INFLATION`): σ_tot[q] = R[tracer][q] · σ_stat[q].

    This is a fixed, fiducial-calibrated diagonal scaling applied AFTER the emulator;
    it does not vary with cosmology/N_tracers and does not touch the training data.
    The clean cosmology-varying quantity (σ_stat) is what the emulator predicts.

    `sigma_triplet` is either a dict keyed by {DH,DM,DV}_over_rs, or an array-like of
    shape (..., 3) in SIGMA_TARGET_NAMES order [DH, DM, DV]. Returns the same type.
    Unknown tracers / missing quantities scale by 1.0 (no-op).
    """
    R = (ratios or DESI_SYST_INFLATION).get(tracer, {})
    if isinstance(sigma_triplet, dict):
        return {k: sigma_triplet[k] * float(R.get(k, 1.0)) for k in sigma_triplet}
    factors = np.array([float(R.get(k, 1.0)) for k in _SIGMA_KEYS])
    return np.asarray(sigma_triplet, dtype=float) * factors
