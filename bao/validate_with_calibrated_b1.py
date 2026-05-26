"""Validation test: with b1 calibrated to DESI data, do we match DESI σ?

Computes Fisher σ(α) for each tracer using:
- Pipeline theory with b1 overridden to the chi²-minimizing value
- Bundle data + cov + W (substituted)
- DESI Adame+24 {s⁰, s²} BB basis (Schur-marginalized)

If F/D ≈ 1 across tracers, the methodology (cov + BB + b1 calibration) is
validated. The next step is to get analytic b1(cosmology) for emulator use.

This script uses calibrated b1 — explicitly NOT for production emulator
use (violates "no DESI data dependence" policy). It's a one-off
validation that proves: matching DESI σ requires getting b1 right.
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
from desilike.theories.galaxy_clustering import (
    DampedBAOWigglesTracerCorrelationFunctionMultipoles,
)
from compare_pipeline_theory_to_bundle import load_bundle, bb_basis

# Calibrated b1 from chi² scan against bundle data (chi²-minimizing values)
B1_CALIBRATED = {
    'BGS':       1.51,   # pipeline already 1.51, no change needed
    'LRG1':      2.00,   # close to pipeline; refine via local scan if needed
    'LRG2':      2.10,
    'LRG3_ELG1': 1.70,
    'ELG2':      1.50,
    'QSO':       2.00,
}

# DESI bao-recon stat-only σ_q reference (post-marg α-cov from DESI VAC files)
DESI_BAO_RECON_SIGMA_Q = {
    'BGS':       {'qiso': 0.0205},
    'LRG2':      {'qpar': 0.0281, 'qper': 0.0169},
    'QSO':       {'qiso': 0.0230},
    # For LRG1, LRG3+ELG1, ELG2 we don't have bao-recon files downloaded;
    # use desi_data.csv reference as fallback
}


def fisher_sigma(tracer, b1_override=None):
    cfg = TRACER_CONFIGS[tracer]
    theta, hrdrag = prep_covar._to_bao_cosmo_params(
        {**prep_covar.PARAM_DEFAULTS, 'Om': 0.3153, 'hrdrag': 99.53})
    apmode = 'qiso' if tracer in ('BGS', 'QSO') else 'qparqper'
    info = prep_covar.build_bao_likelihood(
        N_tracers=dr1_ntracers(tracer), theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg['zrange'], z_eff=float(cfg['z_eff']),
        area=7500., apmode=apmode)
    info['likelihood'](**info['params'])
    template = info['template']
    bundle = load_bundle(tracer)
    smoothing = float(cfg['smoothing_scale'])

    native = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        s=bundle['th_s'][0], ells=tuple(bundle['th_ells']),
        template=template, mode='recsym', smoothing_radius=smoothing,
        model='fog-damping')

    # Build fid params (with optional b1 override)
    base = {}
    for k in ('b1', 'sigmapar', 'sigmaper', 'sigmas'):
        if k in info['params']:
            base[k] = float(info['params'][k])
    if b1_override is not None:
        base['b1'] = float(b1_override)
    # Pin BB to zero
    bb_zero = {p.name: 0.0 for p in native.all_params
               if (p.name.startswith('al') or p.name.startswith('bl'))}
    base.update(bb_zero)
    native_pnames = {p.name for p in native.all_params}

    def predict(overrides=None):
        params = dict(base)
        if overrides: params.update(overrides)
        p_filt = {k: v for k, v in params.items() if k in native_pnames}
        native(**p_filt)
        xi = np.asarray(native.corr)
        xi_flat = np.concatenate([xi[i] for i in range(xi.shape[0])])
        return bundle['W'] @ xi_flat

    cov = 0.5 * (bundle['cov'] + bundle['cov'].T)
    C_inv = np.linalg.inv(cov)

    # Build Fisher: marginalize over {b1, dbeta, σ_s, Σ_par, Σ_per} + BB (DESI {0,2})
    nl_params = (('qiso', 'b1', 'dbeta', 'sigmas', 'sigmapar', 'sigmaper')
                 if apmode == 'qiso'
                 else ('qpar', 'qper', 'b1', 'dbeta', 'sigmas', 'sigmapar', 'sigmaper'))
    steps = {'qpar': 0.01, 'qper': 0.01, 'qiso': 0.01, 'b1': 0.05, 'dbeta': 0.05,
             'sigmas': 0.05, 'sigmapar': 0.1, 'sigmaper': 0.1}
    fid_vals = {}
    for p in nl_params:
        if p in base:
            fid_vals[p] = base[p]
        else:
            for q in native.all_params:
                if q.name == p:
                    fid_vals[p] = float(q.value); break
            else:
                fid_vals[p] = 1.0  # for qpar/qper/qiso default

    data = np.concatenate(bundle['obs_data'])
    n_data = data.size

    pred_fid = predict(fid_vals)
    J = np.zeros((n_data, len(nl_params)), dtype=np.float64)
    for i, p in enumerate(nl_params):
        dp = steps[p]
        plus = predict({**fid_vals, p: fid_vals[p] + dp})
        minus = predict({**fid_vals, p: fid_vals[p] - dp})
        J[:, i] = (plus - minus) / (2.0 * dp)

    BB = bb_basis(bundle['obs_ells'], bundle['obs_s'], n_data, powers=(0, 2))
    J_full = np.concatenate([J, BB], axis=1)
    F = J_full.T @ C_inv @ J_full
    # Schur-marginalize down to q's
    if apmode == 'qiso':
        idx_keep = [0]
    else:
        idx_keep = [0, 1]
    idx_marg = [i for i in range(F.shape[0]) if i not in idx_keep]
    F_kk = F[np.ix_(idx_keep, idx_keep)]
    F_km = F[np.ix_(idx_keep, idx_marg)]
    F_mm = F[np.ix_(idx_marg, idx_marg)]
    F_marg = F_kk - F_km @ np.linalg.solve(F_mm, F_km.T)
    cov_marg = np.linalg.inv(F_marg)
    sigmas = np.sqrt(np.diag(cov_marg))
    return {'apmode': apmode, 'sigmas': sigmas, 'b1_used': base['b1']}


def main():
    print(f'{"Tracer":<11} {"b1 pipe":>8} {"b1 cal":>7} {"σ(q1)":>9} {"σ(q2)":>9}   '
          f'{"DESI σ":>9}   {"F/D":>6}')
    print('-' * 80)
    for tracer in ['BGS', 'LRG1', 'LRG2', 'LRG3_ELG1', 'ELG2', 'QSO']:
        try:
            res_pipe = fisher_sigma(tracer, b1_override=None)
            res_cal = fisher_sigma(tracer, b1_override=B1_CALIBRATED[tracer])
            # Compare to DESI σ (bao-recon if available; else use desi_data.csv)
            apmode = res_pipe['apmode']
            ref_DESI = DESI_BAO_RECON_SIGMA_Q.get(tracer)
            if apmode == 'qiso':
                s_pipe = res_pipe['sigmas'][0]
                s_cal = res_cal['sigmas'][0]
                desi_s = ref_DESI['qiso'] if ref_DESI else float('nan')
                desi_str = f'{desi_s:.4f}' if np.isfinite(desi_s) else '—'
                fd_str = f'{s_cal/desi_s:.3f}' if np.isfinite(desi_s) else '—'
                print(f'{tracer:<11} {res_pipe["b1_used"]:>8.2f} {res_cal["b1_used"]:>7.2f} '
                      f'{s_pipe:>9.4f} {"":>9}   {desi_str:>9}   {fd_str:>6}')
            else:
                s_pipe = res_pipe['sigmas']
                s_cal = res_cal['sigmas']
                if ref_DESI:
                    fd_par = s_cal[0] / ref_DESI['qpar']
                    fd_per = s_cal[1] / ref_DESI['qper']
                    desi_str = f'{ref_DESI["qpar"]:.4f}/{ref_DESI["qper"]:.4f}'
                    fd_str = f'{fd_par:.3f}/{fd_per:.3f}'
                else:
                    desi_str = '—'; fd_str = '—'
                print(f'{tracer:<11} {res_pipe["b1_used"]:>8.2f} {res_cal["b1_used"]:>7.2f} '
                      f'{s_cal[0]:>9.4f} {s_cal[1]:>9.4f}   {desi_str:>9}   {fd_str}')
        except Exception as e:
            print(f'{tracer:<11}  FAILED: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
