"""Per-ell χ² and pull stats: HOD b1 vs calibrated b1.

Uses native ξ-theory (no FFTLog). Reports for each tracer & ell:
- χ²/dof of (data - pred - BB) on the diagonal cov sub-block
- max |pull| and rms |pull| at fid (after best-fit BB Schur-marg)

This complements the full-cov χ²/dof in the plot titles by showing
which ell contributes most of the remaining residual.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
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

B1_CALIBRATED = {
    'BGS':       1.51,
    'LRG1':      2.00,
    'LRG2':      2.10,
    'LRG3_ELG1': 1.70,
    'ELG2':      1.50,
    'QSO':       2.00,
}


def predict_native(tracer, b1_override=None):
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
    base = {}
    for k in ('b1', 'sigmapar', 'sigmaper', 'sigmas'):
        if k in info['params']:
            base[k] = float(info['params'][k])
    b1_hod = base['b1']
    if b1_override is not None:
        base['b1'] = float(b1_override)
    bb_zero = {p.name: 0.0 for p in native.all_params
               if (p.name.startswith('al') or p.name.startswith('bl'))}
    base.update(bb_zero)
    pnames = {p.name for p in native.all_params}
    native(**{k: v for k, v in base.items() if k in pnames})
    xi = np.asarray(native.corr)
    xi_flat = np.concatenate([xi[i] for i in range(xi.shape[0])])
    pred = bundle['W'] @ xi_flat
    return bundle, pred, b1_hod, base['b1']


def stats_one(bundle, pred):
    """Return per-ell {chi2_marg/dof, max|pull|, rms|pull|} after full-cov BB Schur-marg."""
    data = np.concatenate(bundle['obs_data'])
    cov = 0.5 * (bundle['cov'] + bundle['cov'].T)
    err = np.sqrt(np.diag(cov))
    C_inv = np.linalg.inv(cov)
    BB = bb_basis(bundle['obs_ells'], bundle['obs_s'], data.size, powers=(0, 2))
    # Best-fit BB under full cov; residual is data - pred - BB c
    A = BB.T @ C_inv @ BB
    b = BB.T @ C_inv @ (data - pred)
    c_bb = np.linalg.solve(A, b)
    r = data - pred - BB @ c_bb
    pulls = r / err

    # Per-ell stats using each ell's diagonal sub-cov for the χ²
    sizes = [len(s) for s in bundle['obs_s']]
    offs = np.insert(np.cumsum(sizes), 0, 0)
    out = {}
    for je, ell in enumerate(bundle['obs_ells']):
        lo, hi = offs[je], offs[je+1]
        r_l = r[lo:hi]
        cov_l = cov[lo:hi, lo:hi]
        C_inv_l = np.linalg.inv(cov_l)
        chi2_l = float(r_l @ C_inv_l @ r_l)
        # dof: take per-ell as n - 2 (2 BB columns per ell, but Schur-marg eats them via full cov)
        dof_l = len(r_l) - 2
        p_l = pulls[lo:hi]
        out[int(ell)] = {
            'chi2_dof': chi2_l / dof_l, 'n': len(r_l),
            'max': float(np.max(np.abs(p_l))),
            'rms': float(np.sqrt(np.mean(p_l**2))),
        }
    # Full-cov χ²/dof
    chi2_full = float(r @ C_inv @ r)
    dof_full = data.size - BB.shape[1]
    out['all'] = {'chi2_dof': chi2_full / dof_full, 'n': data.size}
    return out


def main():
    print(f'{"Tracer":<11} {"b1":<6} {"ℓ":>3} {"n":>4} {"χ²/dof":>8} '
          f'{"rms|pull|":>10} {"max|pull|":>10}')
    print('-' * 70)
    for tracer in ['BGS', 'LRG1', 'LRG2', 'LRG3_ELG1', 'ELG2', 'QSO']:
        for label, b1_ov in [('HOD', None), ('cal', B1_CALIBRATED[tracer])]:
            bundle, pred, b1_hod, b1_used = predict_native(tracer, b1_override=b1_ov)
            s = stats_one(bundle, pred)
            ells = [e for e in s if e != 'all']
            for ell in sorted(ells):
                e = s[ell]
                print(f'{tracer:<11} {label:<6} {ell:>3} {e["n"]:>4} '
                      f'{e["chi2_dof"]:>8.2f} {e["rms"]:>10.2f} {e["max"]:>10.2f}')
            e = s['all']
            print(f'{tracer:<11} {label:<6} {"all":>3} {e["n"]:>4} '
                  f'{e["chi2_dof"]:>8.2f}     —          —')
        print()


if __name__ == '__main__':
    main()
