#!/usr/bin/env python3
"""
06_spectral_tau_controls.py
Test whether SCov G2 predicts ρ after controlling for spectral exponent
and intrinsic timescale (τ). Progressive partial correlations.

Requires:
    pip install numpy scipy pandas scikit-learn

Input:
    - rho_with_valk.csv: Merged dataset with Valk gradients
    - parcel_spectral_features.csv: MEG spectral exponent per parcel
    - schaefer400_parcel_means.csv: τ per parcel

Output:
    - progressive_controls_table.csv
    - incremental_r2_table.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq
from sklearn.linear_model import LinearRegression

# ============================================================
# CONFIGURATION
# ============================================================
RHO_FILE = 'rho_with_valk.csv'
SPECTRAL_FILE = 'parcel_spectral_features.csv'
TAU_FILE = 'schaefer400_parcel_means.csv'

# ============================================================
# FUNCTIONS
# ============================================================
def partial_spearman_multi(x, y, covariates):
    """Partial Spearman correlation controlling for multiple covariates."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rcov = np.column_stack([stats.rankdata(c) for c in covariates])
    A = np.column_stack([rcov, np.ones(len(rx))])
    res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(res_x, res_y)

def partial_spearman(x, y, z):
    return partial_spearman_multi(x, y, [z])

def sig(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

# ============================================================
# LOAD AND MERGE
# ============================================================
df = pd.read_csv(RHO_FILE)
spec = pd.read_csv(SPECTRAL_FILE)
tau_df = pd.read_csv(TAU_FILE)

df['spectral_exponent'] = spec['spectral_exponent'].values
df['tau'] = tau_df['tau'].values

rho = df['rho'].values
z = df['z'].values
se = df['spectral_exponent'].values
tau = df['tau'].values
g2 = df['scov_G2'].values
gen_g2 = df['coher_G2'].values

# ============================================================
# PROGRESSIVE CONTROLS FOR G2
# ============================================================
print("=" * 70)
print("SCov G2 → ρ: PROGRESSIVE CONTROLS")
print("=" * 70)

tests = [
    ('Zero-order', []),
    ('| z', [z]),
    ('| SE', [se]),
    ('| τ', [tau]),
    ('| z + SE', [z, se]),
    ('| z + τ', [z, tau]),
    ('| SE + τ', [se, tau]),
    ('| z + SE + τ', [z, se, tau]),
]

rows = []
for label, covs in tests:
    if len(covs) == 0:
        r, p = stats.spearmanr(rho, g2)
    else:
        r, p = partial_spearman_multi(rho, g2, covs)
    print(f"  G2 {label:<20s}: r = {r:+.4f}, p = {p:.2e} ({sig(p)})")
    rows.append({'control': label, 'predictor': 'SCov G2', 'r': r, 'p': p})

# Repeat for Genetic G2
for label, covs in tests:
    if len(covs) == 0:
        r, p = stats.spearmanr(rho, gen_g2)
    else:
        r, p = partial_spearman_multi(rho, gen_g2, covs)
    rows.append({'control': label, 'predictor': 'Genetic G2', 'r': r, 'p': p})

pd.DataFrame(rows).to_csv('progressive_controls_table.csv', index=False)

# ============================================================
# INCREMENTAL R²
# ============================================================
print(f"\n{'='*70}")
print("INCREMENTAL VARIANCE EXPLAINED")
print(f"{'='*70}")

models = [
    ('z', np.column_stack([z])),
    ('SE', np.column_stack([se])),
    ('tau', np.column_stack([tau])),
    ('z + SE', np.column_stack([z, se])),
    ('z + SE + tau', np.column_stack([z, se, tau])),
    ('z + SE + G2', np.column_stack([z, se, g2])),
    ('z + SE + tau + G2', np.column_stack([z, se, tau, g2])),
]

r2_rows = []
for name, X in models:
    r2 = LinearRegression().fit(X, rho).score(X, rho)
    print(f"  {name:<30s}: R² = {r2:.4f}")
    r2_rows.append({'model': name, 'R2': r2})

pd.DataFrame(r2_rows).to_csv('incremental_r2_table.csv', index=False)
print("\nSaved progressive_controls_table.csv and incremental_r2_table.csv")
