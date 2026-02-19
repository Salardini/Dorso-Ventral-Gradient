#!/usr/bin/env python3
"""
03_valk_gradients.py
Correlate Valk et al. (2020) structural covariance and genetic correlation
gradients with ρ. This is the primary analysis testing the dual-origin hypothesis.

Requires:
    pip install numpy scipy pandas

Input:
    - rho_schaefer400.csv: CSV with 'rho', 'x', 'y', 'z' columns (400 parcels)
    - strcov_gradient.csv: 400 x 10 structural covariance gradients (from Valk GitHub)
    - coher_gradient.csv: 400 x 10 genetic correlation gradients (from Valk GitHub)

Data source:
    https://github.com/sofievalk/projects/tree/master/Structure_of_Structure

Output:
    - rho_valk_results.csv: Full data with gradient values
    - valk_correlation_table.txt: Summary statistics
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq

# ============================================================
# CONFIGURATION
# ============================================================
RHO_FILE = 'rho_schaefer400.csv'
SCOV_GRADIENT_FILE = 'strcov_gradient.csv'
COHER_GRADIENT_FILE = 'coher_gradient.csv'

# ============================================================
# FUNCTIONS
# ============================================================
def partial_spearman(x, y, z):
    """Partial Spearman correlation: r(x,y | z).
    Rank-transform all variables, regress out z from both x and y ranks,
    then compute Pearson r on residuals.
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    A = np.column_stack([rz, np.ones(len(rz))])
    res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(res_x, res_y)

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(RHO_FILE)
scov_grad = np.loadtxt(SCOV_GRADIENT_FILE, delimiter=',')  # 400 x 10
coher_grad = np.loadtxt(COHER_GRADIENT_FILE, delimiter=',')  # 400 x 10

assert scov_grad.shape == (400, 10), f"Expected 400x10, got {scov_grad.shape}"
assert coher_grad.shape == (400, 10), f"Expected 400x10, got {coher_grad.shape}"

# Add gradient values to dataframe
for i in range(10):
    df[f'scov_G{i+1}'] = scov_grad[:, i]
    df[f'coher_G{i+1}'] = coher_grad[:, i]

# ============================================================
# PRIMARY ANALYSIS: Correlations with ρ
# ============================================================
z = df['z'].values
rho = df['rho'].values

print("=" * 80)
print("STRUCTURAL COVARIANCE GRADIENTS vs ρ")
print("=" * 80)
print(f"{'Gradient':<15s} {'ρₛ':>7s} {'p':>10s} {'sig':>5s} {'r|z':>7s} {'p|z':>10s} {'sig':>5s}")
print("-" * 65)

for i in range(5):
    r, p = stats.spearmanr(rho, df[f'scov_G{i+1}'])
    rp, pp = partial_spearman(rho, df[f'scov_G{i+1}'].values, z)
    print(f"  SCov G{i+1:<8d} {r:+.4f} {p:>9.2e} {sig_stars(p):>5s} {rp:+.4f} {pp:>9.2e} {sig_stars(pp):>5s}")

print()
print("=" * 80)
print("GENETIC CORRELATION GRADIENTS vs ρ")
print("=" * 80)
print(f"{'Gradient':<15s} {'ρₛ':>7s} {'p':>10s} {'sig':>5s} {'r|z':>7s} {'p|z':>10s} {'sig':>5s}")
print("-" * 65)

for i in range(5):
    r, p = stats.spearmanr(rho, df[f'coher_G{i+1}'])
    rp, pp = partial_spearman(rho, df[f'coher_G{i+1}'].values, z)
    print(f"  Genet G{i+1:<7d} {r:+.4f} {p:>9.2e} {sig_stars(p):>5s} {rp:+.4f} {pp:>9.2e} {sig_stars(pp):>5s}")

# ============================================================
# VALIDATION: G2 orthogonality to z
# ============================================================
print()
print("=" * 80)
print("VALIDATION")
print("=" * 80)

r_g2z, p_g2z = stats.spearmanr(df['scov_G2'], z)
print(f"SCov G2 vs z-coordinate: ρₛ = {r_g2z:.4f}, p = {p_g2z:.2e}")
print(f"  → G2 is {'orthogonal to' if abs(r_g2z) < 0.1 else 'correlated with'} z")

r_g1z, p_g1z = stats.spearmanr(df['scov_G1'], z)
print(f"SCov G1 vs z-coordinate: ρₛ = {r_g1z:.4f}, p = {p_g1z:.2e}")

# ============================================================
# REGRESSION: Incremental R²
# ============================================================
print()
print("=" * 80)
print("INCREMENTAL VARIANCE EXPLAINED (OLS R²)")
print("=" * 80)

from sklearn.linear_model import LinearRegression

y = rho
X_z = z.reshape(-1, 1)
X_g2 = df[['z', 'scov_G2']].values
X_gen = df[['z', 'coher_G2']].values
X_both = df[['z', 'scov_G2', 'coher_G2']].values

r2_z = LinearRegression().fit(X_z, y).score(X_z, y)
r2_g2 = LinearRegression().fit(X_g2, y).score(X_g2, y)
r2_gen = LinearRegression().fit(X_gen, y).score(X_gen, y)
r2_both = LinearRegression().fit(X_both, y).score(X_both, y)

print(f"  z alone:                   R² = {r2_z:.4f}")
print(f"  z + SCov G2:               R² = {r2_g2:.4f}  (ΔR² = {r2_g2 - r2_z:+.4f})")
print(f"  z + Genetic G2:            R² = {r2_gen:.4f}  (ΔR² = {r2_gen - r2_z:+.4f})")
print(f"  z + SCov G2 + Genetic G2:  R² = {r2_both:.4f}  (ΔR² = {r2_both - r2_z:+.4f})")

# ============================================================
# SAVE
# ============================================================
df.to_csv('rho_valk_results.csv', index=False)
print(f"\nSaved rho_valk_results.csv")
