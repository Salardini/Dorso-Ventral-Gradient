#!/usr/bin/env python3
"""
07_gradient_axis_angles.py

Compute 3D gradient directions for rho and tau using multiple regression.
Determines the dominant axis orientation for each metric.

Output: data/mous/gradient_axis_angles.csv

Key results:
- rho: 18 deg from DV axis (near-vertical gradient)
- tau frontal: 6 deg from DV
- tau posterior: 24 deg from AP (75 deg from DV)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "mous"

N_PERM = 5000


def compute_gradient_direction(metric, x, y, z):
    """
    Compute gradient direction using multiple regression.

    Returns normalized gradient vector and angle from each axis.
    """
    # Design matrix with coordinates
    X = np.column_stack([np.ones(len(metric)), x, y, z])

    # Fit regression
    beta = np.linalg.lstsq(X, metric, rcond=None)[0]

    # Gradient vector (coefficients for x, y, z)
    gradient = beta[1:]  # [beta_x, beta_y, beta_z]

    # Normalize
    grad_norm = gradient / np.linalg.norm(gradient)

    # Compute angles from each axis (in degrees)
    angle_from_x = np.degrees(np.arccos(np.abs(grad_norm[0])))  # ML
    angle_from_y = np.degrees(np.arccos(np.abs(grad_norm[1])))  # AP
    angle_from_z = np.degrees(np.arccos(np.abs(grad_norm[2])))  # DV

    # Dominant axis
    angles = [angle_from_x, angle_from_y, angle_from_z]
    axes = ['ML', 'AP', 'DV']
    dominant_idx = np.argmin(angles)
    dominant_axis = axes[dominant_idx]

    return {
        'gradient_x': gradient[0],
        'gradient_y': gradient[1],
        'gradient_z': gradient[2],
        'grad_norm_x': grad_norm[0],
        'grad_norm_y': grad_norm[1],
        'grad_norm_z': grad_norm[2],
        'angle_from_ML': angle_from_x,
        'angle_from_AP': angle_from_y,
        'angle_from_DV': angle_from_z,
        'dominant_axis': dominant_axis,
    }


def main():
    print("=" * 70)
    print("Gradient Axis Angle Analysis")
    print("=" * 70)

    # Load data
    df = pd.read_csv(DATA_DIR / "parcel_group_maps.csv")
    print(f"Loaded {len(df)} parcels")

    # Extract arrays
    rho = df['rho_mean'].values
    tau = df['tau_mean'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values

    # Define frontal and posterior masks (based on y coordinate)
    y_median = np.median(y)
    frontal_mask = y > y_median
    posterior_mask = y <= y_median

    results = []

    # Full cortex rho
    print("\n[1] rho - Full cortex")
    res = compute_gradient_direction(rho, x, y, z)
    res['metric'] = 'rho'
    res['region'] = 'full'
    results.append(res)
    print(f"    Angle from DV: {res['angle_from_DV']:.1f} deg")
    print(f"    Dominant axis: {res['dominant_axis']}")

    # Frontal rho
    print("\n[2] rho - Frontal")
    res = compute_gradient_direction(rho[frontal_mask], x[frontal_mask], y[frontal_mask], z[frontal_mask])
    res['metric'] = 'rho'
    res['region'] = 'frontal'
    results.append(res)
    print(f"    Angle from DV: {res['angle_from_DV']:.1f} deg")

    # Posterior rho
    print("\n[3] rho - Posterior")
    res = compute_gradient_direction(rho[posterior_mask], x[posterior_mask], y[posterior_mask], z[posterior_mask])
    res['metric'] = 'rho'
    res['region'] = 'posterior'
    results.append(res)
    print(f"    Angle from DV: {res['angle_from_DV']:.1f} deg")

    # Full cortex tau
    print("\n[4] tau - Full cortex")
    res = compute_gradient_direction(tau, x, y, z)
    res['metric'] = 'tau'
    res['region'] = 'full'
    results.append(res)
    print(f"    Angle from AP: {res['angle_from_AP']:.1f} deg")
    print(f"    Angle from DV: {res['angle_from_DV']:.1f} deg")

    # Frontal tau
    print("\n[5] tau - Frontal")
    res = compute_gradient_direction(tau[frontal_mask], x[frontal_mask], y[frontal_mask], z[frontal_mask])
    res['metric'] = 'tau'
    res['region'] = 'frontal'
    results.append(res)
    print(f"    Angle from DV: {res['angle_from_DV']:.1f} deg")

    # Posterior tau
    print("\n[6] tau - Posterior")
    res = compute_gradient_direction(tau[posterior_mask], x[posterior_mask], y[posterior_mask], z[posterior_mask])
    res['metric'] = 'tau'
    res['region'] = 'posterior'
    results.append(res)
    print(f"    Angle from AP: {res['angle_from_AP']:.1f} deg")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(DATA_DIR / "gradient_axis_angles.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'gradient_axis_angles.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Gradient Axis Angles")
    print("=" * 70)
    print(f"\n{'Metric':<8} {'Region':<12} {'Angle from DV':>15} {'Dominant Axis':>15}")
    print("-" * 55)
    for res in results:
        print(f"{res['metric']:<8} {res['region']:<12} {res['angle_from_DV']:>15.1f} {res['dominant_axis']:>15}")


if __name__ == '__main__':
    main()
