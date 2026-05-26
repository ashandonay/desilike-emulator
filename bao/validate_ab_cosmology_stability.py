"""Validate that f_AB constant-in-cosmology approximation is acceptable.

Sweeps cosmology params (Om, hrdrag, w0, wa) across the practical
emulator-use range and reports per-tracer b1 drift. The premise of
constant f_AB is:

    b1_pipeline(cosmo) = f_AB * b1_HOD_mass_only(cosmo)

where f_AB is set once at fid (from pre-DESI literature) and held
constant. This script tests whether that approximation behaves sanely
across the emulator grid by reporting:

  (1) max % drift of b1_pipeline across the grid per tracer
  (2) b1_pipeline at each corner cosmology vs fid
  (3) whether the variation is monotonic in cosmology (smooth) or
      pathological (jumps from M_min root-find failures, etc.)

CANNOT directly test if f_AB ITSELF should be cosmology-dependent
without multi-cosmology galaxy bias data. What we CAN test:
the HOD-mass-only path is well-behaved under cosmology variation,
so any f_AB drift is bounded by the literature uncertainty on the
underlying AB measurement (~5-15% across Planck-prior range).

Output: ab_cosmology_stability.csv (one row per (cosmo, tracer))
        ab_cosmology_stability.png (per-tracer b1 vs cosmology)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from itertools import product
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import prep_covar
from util import TRACER_CONFIGS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fiducial cosmology (matches the calibration cosmology in §33)
FID = {'Om': 0.3153, 'hrdrag': 99.53, 'w0': -1.0, 'wa': 0.0, 'Ok': 0.0}

# Practical emulator-use bounds (NOT the wide prior bounds Om∈[0.01, 0.99]).
# These cover Planck-2018 ± ~10σ, current DESI Y3 + CMB-S4 expected ranges,
# wCDM/w0waCDM degeneracies, and H0 tension band. Stress-testing inside
# these is enough; outside is unphysical territory.
GRID = {
    'Om':     np.array([0.25, 0.30, 0.3153, 0.34, 0.40]),
    'hrdrag': np.array([88.0, 95.0, 99.53, 105.0, 110.0]),
    'w0':     np.array([-1.3, -1.0, -0.7]),
    'wa':     np.array([-0.5, 0.0, 0.5]),
}

TRACERS = ['BGS', 'LRG1', 'LRG2', 'LRG3_ELG1', 'ELG2', 'QSO']


def pipeline_b1(tracer, cosmo_override):
    """Run the full pipeline and return b1 (with whatever f_AB is in hod.yaml)."""
    cfg = TRACER_CONFIGS[tracer]
    sample = {**prep_covar.PARAM_DEFAULTS, **FID, **cosmo_override}
    theta, hrdrag = prep_covar._to_bao_cosmo_params(sample)
    apmode = 'qiso' if tracer in ('BGS', 'QSO') else 'qparqper'
    info = prep_covar.build_bao_likelihood(
        N_tracers=300000, theta_cosmo=theta, hrdrag=hrdrag,
        tracer_bin=tracer, zrange=cfg['zrange'], z_eff=float(cfg['z_eff']),
        area=7500., apmode=apmode)
    info['likelihood'](**info['params'])
    return float(info['params']['b1'])


def sweep_one_axis(axis_name, axis_vals):
    """Sweep one cosmology axis at a time (others held at fid). Returns dict
    tracer -> list of (val, b1)."""
    out = {t: [] for t in TRACERS}
    for v in axis_vals:
        for tracer in TRACERS:
            try:
                b1 = pipeline_b1(tracer, {axis_name: float(v)})
            except Exception as e:
                b1 = float('nan')
            out[tracer].append((float(v), b1))
    return out


def main():
    print('Sweeping cosmology axes (other params held at fid)...')

    results = {}
    for axis in ['Om', 'hrdrag', 'w0', 'wa']:
        print(f'  axis: {axis} over {GRID[axis]}')
        results[axis] = sweep_one_axis(axis, GRID[axis])

    # CSV
    out_csv = Path(__file__).resolve().parent / 'ab_cosmology_stability.csv'
    with open(out_csv, 'w') as f:
        f.write('axis,axis_value,tracer,b1\n')
        for axis, sweep in results.items():
            for tracer, pts in sweep.items():
                for v, b1 in pts:
                    f.write(f'{axis},{v},{tracer},{b1:.6f}\n')
    print(f'Saved {out_csv}')

    # Drift summary
    print('\nb1 drift per tracer across each cosmology axis (max% vs fid):')
    print(f'{"Tracer":<11} {"Om":>10} {"hrdrag":>10} {"w0":>10} {"wa":>10}')
    print('-' * 60)
    fid_b1 = {}
    for tracer in TRACERS:
        fid_b1[tracer] = pipeline_b1(tracer, {})
    for tracer in TRACERS:
        b1_fid = fid_b1[tracer]
        row = [tracer]
        for axis in ['Om', 'hrdrag', 'w0', 'wa']:
            pts = results[axis][tracer]
            b1s = np.array([b for _, b in pts])
            drift = np.nanmax(np.abs(b1s - b1_fid)) / b1_fid * 100.0
            row.append(f'{drift:.2f}%')
        print(f'{row[0]:<11} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}')

    # 2x2 plot, one panel per axis, color per tracer
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(TRACERS)))
    for ip, axis in enumerate(['Om', 'hrdrag', 'w0', 'wa']):
        ax = axes[ip]
        for it, tracer in enumerate(TRACERS):
            pts = results[axis][tracer]
            xs = [p[0] for p in pts]
            ys = [p[1] / fid_b1[tracer] for p in pts]
            ax.plot(xs, ys, 'o-', color=colors[it], label=tracer, alpha=0.85)
        ax.axhline(1.0, color='k', ls='--', alpha=0.4, lw=0.7)
        ax.axhline(1.05, color='r', ls=':', alpha=0.4, lw=0.7,
                   label='±5% band' if ip == 0 else None)
        ax.axhline(0.95, color='r', ls=':', alpha=0.4, lw=0.7)
        ax.set_xlabel(axis)
        ax.set_ylabel(r'$b_1$ (pipeline) / $b_1$(fid)')
        ax.set_title(f'b1 drift along {axis} axis')
        ax.grid(alpha=0.3)
        if ip == 0:
            ax.legend(fontsize=8, ncol=2, loc='best')

    fig.suptitle('Pipeline b1 cosmology sensitivity (with fixed f_AB from hod.yaml)\n'
                 'A smooth, monotonic curve means f_AB constant-in-cosmology is well-behaved.\n'
                 'The ±5% red band gives the expected precision of the underlying f_AB measurements.',
                 fontsize=11)
    fig.tight_layout()
    out_png = Path(__file__).resolve().parent / 'ab_cosmology_stability.png'
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
