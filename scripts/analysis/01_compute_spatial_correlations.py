#!/usr/bin/env python3
"""
01_compute_spatial_correlations.py

Compute rho and tau correlations with spatial coordinates (x, y, z).
Includes spin permutation tests for spatial autocorrelation correction.

Input: data/mous/parcel_group_maps.csv
Output: results/spatial_correlations.csv

Key results:
- rho vs z (DV): r = -0.72, p_spin = 0.002
- tau vs y (AP): r = +0.45
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "mous"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_PERM = 5000


def run_spin_test(x, y, hemi, n_perm=N_PERM):
    """Hemisphere-preserving spin permutation test."""
    r_obs, p_param = stats.pearsonr(x, y)

    lh_idx = np.where(hemi == 'lh')[0]
    rh_idx = np.where(hemi == 'rh')[0]

    null_r = np.zeros(n_perm)
    np.random.seed(42)
    for i in range(n_perm):
        perm = np.zeros(len(x), dtype=int)
        perm[lh_idx] = np.random.permutation(lh_idx)
        perm[rh_idx] = np.random.permutation(rh_idx)
        null_r[i] = stats.pearsonr(x[perm], y)[0]

    p_spin = np.mean(np.abs(null_r) >= np.abs(r_obs))
    return r_obs, p_param, p_spin


def main():
    print("=" * 70)
    print("Spatial Correlations Analysis")
    print("=" * 70)

    # Load data
    df = pd.read_csv(DATA_DIR / "parcel_group_maps.csv")
    print(f"Loaded {len(df)} parcels")

    # Extract arrays
    rho = df['rho_mean'].values
    tau = df['tau_mean'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    hemi = df['hemi'].values

    # Compute correlations
    results = []

    tests = [
        ('rho', 'x', rho, x, 'ML'),
        ('rho', 'y', rho, y, 'AP'),
        ('rho', 'z', rho, z, 'DV'),
        ('tau', 'x', tau, x, 'ML'),
        ('tau', 'y', tau, y, 'AP'),
        ('tau', 'z', tau, z, 'DV'),
    ]

    print(f"\n{'Metric':<8} {'Axis':<8} {'r':>10} {'p_param':>12} {'p_spin':>10}")
    print("-" * 50)

    for metric, coord, metric_vals, coord_vals, axis in tests:
        r, p_param, p_spin = run_spin_test(coord_vals, metric_vals, hemi)
        sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
        print(f"{metric:<8} {axis:<8} {r:>10.4f} {p_param:>12.2e} {p_spin:>10.4f} {sig}")

        results.append({
            'metric': metric,
            'coordinate': coord,
            'axis': axis,
            'r': r,
            'p_param': p_param,
            'p_spin': p_spin,
        })

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "spatial_correlations.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'spatial_correlations.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    rho_dv = df_results[(df_results['metric'] == 'rho') & (df_results['axis'] == 'DV')].iloc[0]
    tau_ap = df_results[(df_results['metric'] == 'tau') & (df_results['axis'] == 'AP')].iloc[0]
    print(f"rho vs z (DV): r = {rho_dv['r']:.2f}, p_spin = {rho_dv['p_spin']:.3f}")
    print(f"tau vs y (AP): r = {tau_ap['r']:.2f}, p_spin = {tau_ap['p_spin']:.3f}")


if __name__ == '__main__':
    main()
