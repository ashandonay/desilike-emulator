"""Analytical FKP-1994 Gaussian P-space covariance from n(z) slices.

Periodic-box / Padmanabhan limit: diagonal in k, full angular coupling via
numerical mu-quadrature. No window mode-coupling. Calibrated against thecov
at fid (see validate_thecov_lrg2.py).

Formula (FKP-1994 + Yamamoto-style multipoles; per-bin variance):

    cov[P_la(k), P_lb(k)]
        = (2la+1)(2lb+1) * 2 * (2 pi)^3 / (4 pi k^2 dk) * (1 / I_22^2) * (
            <L_la L_lb P(k,mu)^2>_mu        * I_44
          + 2 <L_la L_lb P(k,mu)>_mu        * I_34
          +   <L_la L_lb>_mu                * I_24
        )

with FKP integrals over n(z) slices (V_i = comoving shell volume of slice i,
w_i = 1 / (1 + n_i P_FKP) — FKP weight, k-independent at the pivot):

    I_22 = sum_i V_i * n_i^2 * w_i^2      (FKP normalization)
    I_44 = sum_i V_i * n_i^4 * w_i^4      (cosmic-variance term)
    I_34 = sum_i V_i * n_i^3 * w_i^4      (mixed P x shot-noise)
    I_24 = sum_i V_i * n_i^2 * w_i^4      (pure shot-noise term)

Uniform-box limit reproduces textbook 2(2 pi)^3 (P+1/n)^2 / (V * V_k):
    I_22 = Vn^2w^2, I_44 = Vn^4w^4, I_34 = Vn^3w^4, I_24 = Vn^2w^4
    -> sum = w^4 * V * (nP+1)^2 * n^2
    -> 2(2pi)^3/V_k * w^4 V n^2 (nP+1)^2 / (V^2 n^4 w^4)
    -> 2(2pi)^3 (nP+1)^2 / (V_k V n^2) = 2(2pi)^3 (P+1/n)^2 / (V * V_k)  ✓

The angular averages <...>_mu are computed numerically by Legendre-Gauss
quadrature in mu in [-1, 1]. P(k, mu) = sum_ell L_ell(mu) P_ell(k) with the
sum over the input multipoles (typically ell in {0, 2, 4}).

The output is a dict keyed by (la, lb) pairs with each value an (Nk, Nk)
matrix that is diagonal (off-diagonal = 0 in this approximation).

Validation against thecov sets the overall normalization NORM_CALIB.
Initial value 1.0 = pure literature formula; will be tuned to <5% match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss


# Pivot used for FKP weights w = 1 / (1 + n P_FKP). DESI/BOSS standard for
# LRG is P_FKP = 1e4 (Mpc/h)^3. Re-verify per-tracer at calibration time.
P_FKP_DEFAULT = 1.0e4

# Calibration knob — initialized to 1.0 (pure FKP-1994 formula). After
# comparing per-k diagonal to thecov at fid, set this to the geometric-mean
# ratio so the formula reproduces thecov to ~1% before angular shape is
# checked. Stays as a single scalar constant — not per-tracer or per-k.
NORM_CALIB = 1.0


@dataclass(frozen=True)
class NZSlices:
    """Piecewise-constant n(z) profile used by fkp_analytic_cov.

    Attributes
    ----------
    z_mid : (Nz,) array of slice midpoints
    nbar  : (Nz,) array of comoving number density per slice [(Mpc/h)^-3]
    V     : (Nz,) array of comoving shell volume per slice  [(Mpc/h)^3]
    """
    z_mid: np.ndarray
    nbar:  np.ndarray
    V:     np.ndarray

    def __post_init__(self):
        for name in ("z_mid", "nbar", "V"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, arr)
        assert self.z_mid.shape == self.nbar.shape == self.V.shape, \
            "z_mid, nbar, V must share shape"


def _fkp_integrals(slices: NZSlices, P_FKP: float = P_FKP_DEFAULT
                   ) -> Tuple[float, float, float, float]:
    """Return scalar FKP integrals (I_22, I_44, I_34, I_24).

    w_i = 1 / (1 + n_i P_FKP) is k-independent at the FKP pivot, so all
    integrals are scalars (no per-k variation in the periodic-box limit).
    """
    n = slices.nbar
    V = slices.V
    w2 = 1.0 / (1.0 + n * P_FKP) ** 2
    w4 = w2 ** 2
    I22 = float(np.sum(V * n ** 2 * w2))
    I44 = float(np.sum(V * n ** 4 * w4))
    I34 = float(np.sum(V * n ** 3 * w4))
    I24 = float(np.sum(V * n ** 2 * w4))
    return I22, I44, I34, I24


def fkp_analytic_cov(
    k: np.ndarray,
    P_ells_in: np.ndarray,
    ells_in: Sequence[int],
    ells_obs: Sequence[int],
    slices: NZSlices,
    P_FKP: float = P_FKP_DEFAULT,
    n_mu: int = 64,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Compute FKP-1994 Gaussian cov in periodic-box limit.

    Parameters
    ----------
    k          : (Nk,) array, k-bin centers in h/Mpc. Assumed uniform binning;
                 dk is taken as mean(diff(k)).
    P_ells_in  : (N_in, Nk), values of P_{ell_in}(k) for ell_in in ells_in.
                 SHOT NOISE INCLUDED? No — pass shot-noise-removed P_0.
                 The 1/n term is added per-slice inside the formula.
    ells_in    : list of ells appearing in the model power, e.g. (0, 2, 4).
    ells_obs   : list of ells appearing in the observable (typically (0, 2)).
                 Output is a dict over (la, lb) pairs with la, lb in ells_obs.
    slices     : NZSlices instance (n(z) and V(z)).
    P_FKP      : FKP pivot power. 1e4 (Mpc/h)^3 standard for LRG.
    n_mu       : Number of Legendre-Gauss quadrature points in mu.

    Returns
    -------
    dict mapping (la, lb) -> (Nk, Nk) diagonal matrix.
    """
    k = np.asarray(k, dtype=np.float64)
    P_ells_in = np.asarray(P_ells_in, dtype=np.float64)
    Nk = k.size
    if k.size > 1:
        dk = float(np.mean(np.diff(k)))
    else:
        dk = 0.005

    # FKP integrals over n(z) slices (scalars at the FKP pivot).
    I22, I44, I34, I24 = _fkp_integrals(slices, P_FKP=P_FKP)

    # mu-quadrature.
    mu, mu_w = leggauss(n_mu)                       # nodes in [-1, 1], weights sum to 2
    # Legendre L_ell(mu) tables.
    L = {}
    for ell in set(list(ells_in) + list(ells_obs) + [0]):
        coeff = np.zeros(ell + 1)
        coeff[ell] = 1.0
        L[ell] = np.polynomial.legendre.legval(mu, coeff)  # (n_mu,)

    # Build P(k, mu) on the quadrature grid.
    # P_kmu has shape (n_mu, Nk).
    P_kmu = np.zeros((n_mu, Nk), dtype=np.float64)
    for i_ell, ell in enumerate(ells_in):
        P_kmu += L[ell][:, None] * P_ells_in[i_ell][None, :]

    # Prefactor: 2 * (2 pi)^3 / (4 pi k^2 dk), shape (Nk,).
    bin_factor = 2.0 * (2.0 * np.pi) ** 3 / (4.0 * np.pi * k ** 2 * dk)

    # Output blocks.
    blocks: Dict[Tuple[int, int], np.ndarray] = {}
    for la in ells_obs:
        for lb in ells_obs:
            LaLb = L[la] * L[lb]                                  # (n_mu,)
            # 0.5 * integral over [-1, 1] of L_a L_b · f(mu) dmu
            #     = 0.5 * sum_q mu_w[q] * L_a(mu_q) L_b(mu_q) f(mu_q)
            mean_LLP2 = 0.5 * np.sum((LaLb * mu_w)[:, None] * P_kmu ** 2, axis=0)  # (Nk,)
            mean_LLP  = 0.5 * np.sum((LaLb * mu_w)[:, None] * P_kmu,     axis=0)
            mean_LL   = 0.5 * float(np.sum(LaLb * mu_w))                            # scalar

            cov_diag = (2 * la + 1) * (2 * lb + 1) * bin_factor / I22 ** 2 * (
                mean_LLP2 * I44
                + 2.0 * mean_LLP * I34
                + mean_LL * I24
            ) * NORM_CALIB
            blocks[(la, lb)] = np.diag(cov_diag)

    return blocks


def assemble_full_cov(
    blocks: Dict[Tuple[int, int], np.ndarray],
    ells_obs: Sequence[int],
) -> np.ndarray:
    """Stack per-(la, lb) blocks into a single (Nells*Nk, Nells*Nk) matrix.

    Ordering matches desilike's TracerPowerSpectrumMultipolesObservable:
    block (la=0, lb=0) in top-left, then (la=0, lb=2), (la=2, lb=0),
    (la=2, lb=2), etc., per ells_obs order.
    """
    Nell = len(ells_obs)
    Nk = next(iter(blocks.values())).shape[0]
    C = np.zeros((Nell * Nk, Nell * Nk), dtype=np.float64)
    for ia, la in enumerate(ells_obs):
        for ib, lb in enumerate(ells_obs):
            C[ia * Nk:(ia + 1) * Nk, ib * Nk:(ib + 1) * Nk] = blocks[(la, lb)]
    return C


def load_nz_slices(
    tracer_bin: str,
    cosmo,
    area_deg2: float,
    N_design: float | None = None,
    nz_slices_dir: str | None = None,
) -> NZSlices:
    """Build NZSlices for a tracer at a given cosmology and total tracer count.

    The slice fractions are cosmology-INDEPENDENT (file-based). V_shell varies
    with cosmology through chi(z). n(z) = N_design * frac_i / V_i.
    """
    import pandas as pd
    from pathlib import Path

    if nz_slices_dir is None:
        nz_slices_dir = str(Path.home() / "data" / "desi" / "nz_slices")
    path = Path(nz_slices_dir) / f"{tracer_bin}_nz_slices.csv"
    if not path.exists():
        raise FileNotFoundError(f"No n(z) slices file: {path}")

    df = pd.read_csv(path)
    df = df[df["slice_fraction"] > 0.0].reset_index(drop=True)
    z_mid = df["zmid"].to_numpy(dtype=np.float64)
    z_lo  = df["zlow"].to_numpy(dtype=np.float64)
    z_hi  = df["zhigh"].to_numpy(dtype=np.float64)
    frac  = df["slice_fraction"].to_numpy(dtype=np.float64)
    frac  = frac / frac.sum()

    # Recompute V_shell at the supplied cosmology and area.
    sky_frac = float(area_deg2) / 41252.96
    chi_lo = np.asarray(cosmo.comoving_radial_distance(z_lo), dtype=np.float64)
    chi_hi = np.asarray(cosmo.comoving_radial_distance(z_hi), dtype=np.float64)
    V_shell = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3) * sky_frac

    if N_design is None:
        # Use file's Nbin as the absolute count (DESI's published per-bin counts).
        nbar = df["nbar_file"].to_numpy(dtype=np.float64)
    else:
        nbar = N_design * frac / np.maximum(V_shell, 1.0)

    return NZSlices(z_mid=z_mid, nbar=nbar, V=V_shell)
