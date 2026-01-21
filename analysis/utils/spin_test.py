#!/usr/bin/env python3
"""
Spin permutation test utilities.

Hemisphere-preserving permutation tests for cortical maps to account for
spatial autocorrelation.
"""

import numpy as np
from scipy import stats


def run_spin_test(x, y, hemi, n_perm=5000, seed=42):
    """
    Hemisphere-preserving spin permutation test.

    Permutes parcel indices within each hemisphere separately to preserve
    spatial structure while breaking the relationship between variables.

    Parameters
    ----------
    x : array-like
        First variable (e.g., spatial coordinate)
    y : array-like
        Second variable (e.g., rho values)
    hemi : array-like
        Hemisphere labels ('lh' or 'rh') for each parcel
    n_perm : int
        Number of permutations (default: 5000)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    r_obs : float
        Observed Pearson correlation
    p_param : float
        Parametric p-value (assumes normality)
    p_spin : float
        Spin permutation p-value (two-tailed)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    hemi = np.asarray(hemi)

    r_obs, p_param = stats.pearsonr(x, y)

    lh_idx = np.where(hemi == 'lh')[0]
    rh_idx = np.where(hemi == 'rh')[0]

    null_r = np.zeros(n_perm)
    np.random.seed(seed)

    for i in range(n_perm):
        perm = np.zeros(len(x), dtype=int)
        perm[lh_idx] = np.random.permutation(lh_idx)
        perm[rh_idx] = np.random.permutation(rh_idx)
        null_r[i] = stats.pearsonr(x[perm], y)[0]

    # Two-tailed p-value
    p_spin = np.mean(np.abs(null_r) >= np.abs(r_obs))

    return r_obs, p_param, p_spin


def residualize(y, X):
    """
    Residualize y with respect to X using OLS regression.

    Parameters
    ----------
    y : array-like
        Variable to residualize
    X : array-like
        Confound variable(s). Can be 1D or 2D.

    Returns
    -------
    residuals : ndarray
        Residualized values
    """
    y = np.asarray(y)
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add intercept
    X_design = np.column_stack([np.ones(len(X)), X])

    # Fit and compute residuals
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    residuals = y - X_design @ beta

    return residuals
