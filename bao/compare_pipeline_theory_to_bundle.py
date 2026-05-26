"""Diagnostic: does pipeline theory ξ at fid match bundle data at fid?

If theory ≠ data at fid (beyond what BB can absorb), the pipeline theory is
biased relative to DESI's measurements — could indicate different fid
cosmology, different recon convention, or other theory-side mismatch.

If theory matches data at fid (low χ²(fid) after BB-marg), the theory is
fine and only the cov differs from bundle.

For each tracer:
  pipeline P-space theory at fid → FFTLog to ξ at bundle theory s grid →
  apply bundle W → compare to bundle data (with bundle cov for χ²,
  Schur-marg over DESI BB {s⁰, s²}).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import prep_covar
from util import TRACER_CONFIGS, dr1_ntracers
from cosmoprimo.fftlog import PowerToCorrelation
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood

_LIK = Path.home() / 'data/desi/bao_dr1/likelihoods'
_BUNDLES = {
    'BGS':       ('likelihood_correlation-recon-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5', 0.295),
    'LRG1':      ('likelihood_correlation-recon-poles_LRG_GCcomb_z0.4-0.6.h5', 0.51),
    'LRG2':      ('likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5', 0.706),
    'LRG3_ELG1': ('likelihood_correlation-recon-poles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1.h5', 0.93),
    'ELG2':      ('likelihood_correlation-recon-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5', 1.317),
    'QSO':       ('likelihood_correlation-recon-poles_QSO_GCcomb_z0.8-2.1.h5', 1.491),
}


def load_bundle(tracer):
    fname, z = _BUNDLES[tracer]
    with h5py.File(_LIK / fname, 'r') as f:
        cov = np.asarray(f['covariance/value'][...])
        W = np.asarray(f['window/value'][...])
        obs_ells, obs_s, obs_data = [], [], []
        for k in sorted(f['observable'].keys()):
            if k.isdigit():
                obs_ells.append(int(f[f'observable/{k}/meta/ell'][()]))
                obs_s.append(np.asarray(f[f'observable/{k}/s'][...]))
                obs_data.append(np.asarray(f[f'observable/{k}/value'][...]))
        th_ells, th_s = [], []
        for k in sorted(f['window/theory'].keys()):
            if k.isdigit():
                th_ells.append(int(f[f'window/theory/{k}/meta/ell'][()]))
                th_s.append(np.asarray(f[f'window/theory/{k}/s'][...]))
    return {'cov': cov, 'W': W, 'z': z,
            'obs_ells': obs_ells, 'obs_s': obs_s, 'obs_data': obs_data,
            'th_ells': th_ells, 'th_s': th_s}


def build_pipe_theory(tracer):
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, 'Om': 0.3153, 'hrdrag': 99.53})
    apmode = 'qiso' if tracer in ('BGS', 'QSO') else 'qparqper'
    info = prep_covar.build_bao_likelihood(
        N_tracers=dr1_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg['zrange'], z_eff=float(cfg['z_eff']),
        area=7500., apmode=apmode)
    info['likelihood'](**info['params'])
    pipe_theory = None
    for c in info['observable'].runtime_info.pipeline.calculators:
        if type(c).__name__.startswith('DampedBAOWigglesTracerPowerSpectrumMultipoles'):
            pipe_theory = c; break
    k_min, k_max, nb = 0.0014, 0.5, 600
    dk = (k_max - k_min) / nb
    wide_obs = TracerPowerSpectrumMultipolesObservable(
        data=info['params'],
        klim={0: [k_min, k_max, dk], 2: [k_min, k_max, dk]},
        theory=pipe_theory)
    wide_lik = ObservablesGaussianLikelihood(observables=[wide_obs], covariance=np.eye(2 * nb))
    # Explicitly pin all BB params to zero to avoid FFTLog blow-up at default values.
    bb_zero = {p.name: 0.0 for p in wide_lik.all_params
               if (p.name.startswith('al') or p.name.startswith('bl'))}
    wide_lik(**{**info['params'], **bb_zero})
    return info, wide_obs, np.asarray(wide_obs.k[0]), bb_zero


def xi_at_bundle_grid(wide_obs, k_lin, bundle, ells=(0, 2), use_ells=(0, 2)):
    P_lk = np.asarray(wide_obs.theory)
    k_log = np.logspace(np.log10(k_lin[0]), np.log10(k_lin[-1]), 1024)
    P_log = np.empty((len(ells), len(k_log)))
    for ie in range(len(ells)):
        P_log[ie] = np.interp(k_log, k_lin, P_lk[ie])
    fft = PowerToCorrelation(k_log, ell=list(ells))
    s_out, xi_out = fft(P_log, extrap='log')
    s_out = np.asarray(s_out); xi_out = np.asarray(xi_out)
    target_s = bundle['th_s'][0]
    blocks = []
    for ell in [0, 2, 4]:
        if ell in ells and ell in use_ells:
            ie = list(ells).index(ell)
            s_e = s_out[ie] if s_out.ndim == 2 else s_out
            order = np.argsort(s_e)
            blocks.append(np.interp(target_s, s_e[order], xi_out[ie][order]))
        else:
            blocks.append(np.zeros_like(target_s))
    by_ell = {0: blocks[0], 2: blocks[1], 4: blocks[2]}
    return np.concatenate([by_ell[e] for e in bundle['th_ells']])


def bb_basis(obs_ells, obs_s, n_total, powers=(0, 2)):
    cols = []
    cur = 0
    for _, s in zip(obs_ells, obs_s):
        n = len(s)
        for p in powers:
            col = np.zeros(n_total)
            col[cur:cur + n] = s ** p
            cols.append(col)
        cur += n
    return np.array(cols).T


def chi2_marg(residual, C_inv, BB):
    C_inv_r = C_inv @ residual
    C_inv_BB = C_inv @ BB
    A = BB.T @ C_inv_BB
    b = BB.T @ C_inv_r
    chi2_no_bb = float(residual @ C_inv_r)
    chi2_marg = float(chi2_no_bb - b @ np.linalg.solve(A, b))
    return chi2_no_bb, chi2_marg


def main():
    print(f'{"Tracer":<11} {"n_data":>6} {"χ²(fid,noBB)":>13} {"χ²(fid,BBmarg)":>15} {"χ²/dof":>8}')
    print('-' * 60)
    for tracer in ['BGS', 'LRG1', 'LRG2', 'LRG3_ELG1', 'ELG2', 'QSO']:
        try:
            bundle = load_bundle(tracer)
            info, wide_obs, k_lin, _bb_zero = build_pipe_theory(tracer)
            # For DV-only tracers (BGS, QSO), the data is ℓ=0 only and ℓ=2 FFTLog
            # can be unstable; W's ℓ=2 columns are 5e-5 (negligible) for these.
            use = (0,) if tracer in ('BGS', 'QSO') else (0, 2)
            xi_th = xi_at_bundle_grid(wide_obs, k_lin, bundle, use_ells=use)
            pred = bundle['W'] @ xi_th
            data = np.concatenate(bundle['obs_data'])
            n_data = data.size
            cov = 0.5 * (bundle['cov'] + bundle['cov'].T)
            C_inv = np.linalg.inv(cov)
            residual = data - pred
            BB = bb_basis(bundle['obs_ells'], bundle['obs_s'], n_data, powers=(0, 2))
            n_bb = BB.shape[1]
            chi2_no_bb, chi2_m = chi2_marg(residual, C_inv, BB)
            dof = n_data - n_bb
            print(f'{tracer:<11} {n_data:>6d} {chi2_no_bb:>13.2f} {chi2_m:>15.2f} {chi2_m/dof:>8.3f}')
        except Exception as e:
            print(f'{tracer:<11}  FAILED: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
