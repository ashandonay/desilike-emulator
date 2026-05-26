"""Plot pipeline theory at fid vs bundle data — PRODUCTION pipeline
(literature-grounded AB factors in hod.yaml, no per-tracer DESI calibration).

Same layout/style as plot_pipeline_theory_vs_bundle.py but uses whatever
b1 the pipeline produces (with current hod.yaml). After the §33
literature-grounded AB factors are in place, this should look very
similar to plot_pipeline_theory_vs_bundle_calibrated.png — confirming
that the AB factors close the χ²(fid) gap without DESI-data calibration.

Output: pipeline_theory_vs_bundle_production.png
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_TRACERS = ['BGS', 'LRG1', 'LRG2', 'LRG3_ELG1', 'ELG2', 'QSO']


def predict_native(tracer):
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
    b1_pipe = base['b1']
    bb_zero = {p.name: 0.0 for p in native.all_params
               if (p.name.startswith('al') or p.name.startswith('bl'))}
    base.update(bb_zero)
    pnames = {p.name for p in native.all_params}
    native(**{k: v for k, v in base.items() if k in pnames})
    xi = np.asarray(native.corr)
    xi_flat = np.concatenate([xi[i] for i in range(xi.shape[0])])
    pred = bundle['W'] @ xi_flat
    return bundle, pred, b1_pipe


def best_fit_bb(data, pred, C_inv, BB):
    A = BB.T @ C_inv @ BB
    b = BB.T @ C_inv @ (data - pred)
    c = np.linalg.solve(A, b)
    return BB @ c


def chi2_marg_after_bb(data, pred, C_inv, BB):
    A = BB.T @ C_inv @ BB
    b = BB.T @ C_inv @ (data - pred)
    chi2_full = float((data - pred) @ C_inv @ (data - pred))
    return chi2_full - float(b @ np.linalg.solve(A, b))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    axes = axes.flatten()

    for i, tracer in enumerate(_TRACERS):
        ax = axes[i]
        try:
            bundle, pred, b1_pipe = predict_native(tracer)
        except Exception as e:
            ax.text(0.5, 0.5, f'{tracer}\nFAILED: {e}', ha='center', va='center',
                    transform=ax.transAxes); ax.set_title(tracer); continue

        data = np.concatenate(bundle['obs_data'])
        cov = 0.5 * (bundle['cov'] + bundle['cov'].T)
        err = np.sqrt(np.diag(cov))
        C_inv = np.linalg.inv(cov)
        BB = bb_basis(bundle['obs_ells'], bundle['obs_s'], data.size, powers=(0, 2))
        chi2_m = chi2_marg_after_bb(data, pred, C_inv, BB)
        dof = data.size - BB.shape[1]
        bb_corr = best_fit_bb(data, pred, C_inv, BB)
        pred_with_bb = pred + bb_corr

        obs_ells = bundle['obs_ells']
        obs_s_list = bundle['obs_s']
        idx = 0
        colors = {'0': 'C0', '2': 'C3', '4': 'C2'}
        for j, ell in enumerate(obs_ells):
            n = len(obs_s_list[j])
            s = obs_s_list[j]; s2 = s ** 2
            d_block = data[idx:idx+n]
            e_block = err[idx:idx+n]
            p_block = pred[idx:idx+n]
            p_bb = pred_with_bb[idx:idx+n]
            ax.errorbar(s, s2 * d_block, yerr=s2 * e_block, fmt='o', ms=3,
                        capsize=2, color=colors[str(ell)],
                        label=f'data ℓ={ell}', alpha=0.8, zorder=3)
            ax.plot(s, s2 * p_block, '--', color=colors[str(ell)],
                    label=f'pipe theory ℓ={ell}', alpha=0.7, lw=1.5)
            ax.plot(s, s2 * p_bb, '-', color=colors[str(ell)],
                    label=f'pipe+best-fit BB ℓ={ell}', alpha=0.9, lw=2)
            idx += n

        ax.set_xlabel(r'$s$ [$h^{-1}$ Mpc]')
        ax.set_ylabel(r'$s^2 \xi(s)$ [$h^{-2}$ Mpc$^2$]')
        ax.set_title(f"{tracer}  b1_pipe = {b1_pipe:.2f}\n"
                     f"χ²(BB-marg)/dof = {chi2_m:.1f}/{dof} = {chi2_m/dof:.2f}")
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)

    fig.suptitle('Production pipeline (hod.yaml literature-grounded AB) vs DESI DR1 bundle data\n'
                 '(dashed = theory only; solid = theory + best-fit DESI BB)', fontsize=12)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / 'pipeline_theory_vs_bundle_production.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
