#!/usr/bin/env python3
"""
CRITICAL TEST: Does rho-DV survive after controlling for spectral exponent?

The reviewer's main concern:
  "rho might just track 1/f slope across cortex"

We found: spectral_exponent vs rho: r = -0.89 (VERY HIGH)

This script tests if rho-DV is just a proxy for spectral slope-DV.

RUN THIS - IT'S THE MOST IMPORTANT ANALYSIS.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
PARCEL_FILE = DATA_DIR / 'group' / 'parcel_group_maps.csv'
SPECTRAL_FILE = DATA_DIR / 'spectral_confound_controls' / 'parcel_spectral_features.csv'

# ============================================
# LOAD AND MERGE DATA
# ============================================
print("="*60)
print("CRITICAL TEST: rho-DV controlling for SPECTRAL EXPONENT")
print("="*60)

print("\n1. Loading data...")
df_parcel = pd.read_csv(PARCEL_FILE)
df_spectral = pd.read_csv(SPECTRAL_FILE)

# Merge on parcel_idx
df = df_parcel.merge(df_spectral, on='parcel_idx', how='left')
print(f"   Merged {len(df)} parcels")

# Extract variables
rho = df['rho_mean'].values
z = df['z'].values
hemi = df['hemi'].values
spec_exp = df['spectral_exponent'].values
total_power = df['total_power'].values
gamma_delta = df['gamma_delta_ratio'].values

# Remove NaN
valid = ~(np.isnan(rho) | np.isnan(z) | np.isnan(spec_exp))
print(f"   Valid parcels: {np.sum(valid)}")

rho_v = rho[valid]
z_v = z[valid]
hemi_v = hemi[valid]
spec_exp_v = spec_exp[valid]
total_power_v = total_power[valid]
gamma_delta_v = gamma_delta[valid]

# ============================================
# BASELINE
# ============================================
print("\n2. Baseline correlations...")
r_base, _ = stats.pearsonr(rho_v, z_v)
r_spec_rho, _ = stats.pearsonr(spec_exp_v, rho_v)
r_spec_z, _ = stats.pearsonr(spec_exp_v, z_v)

print(f"   rho vs DV:               r = {r_base:.4f}")
print(f"   spectral_exp vs rho:     r = {r_spec_rho:.4f}  <-- HIGH!")
print(f"   spectral_exp vs DV:      r = {r_spec_z:.4f}")

# ============================================
# PARTIAL CORRELATION FUNCTIONS
# ============================================
def partial_corr(x, y, covar):
    """Partial correlation controlling for covariate(s)."""
    if covar.ndim == 1:
        covar = covar.reshape(-1, 1)
    X = np.column_stack([np.ones(len(x)), covar])
    x_resid = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    y_resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(x_resid, y_resid)

def spin_test_partial(x, y, covar, hemi, n_perm=10000, seed=42):
    """Spin permutation test for partial correlation."""
    r_obs, _ = partial_corr(x, y, covar)
    
    lh_idx = np.where(hemi == 'lh')[0]
    rh_idx = np.where(hemi == 'rh')[0]
    
    np.random.seed(seed)
    null_r = np.zeros(n_perm)
    
    for i in range(n_perm):
        perm = np.zeros(len(x), dtype=int)
        perm[lh_idx] = np.random.permutation(lh_idx)
        perm[rh_idx] = np.random.permutation(rh_idx)
        null_r[i], _ = partial_corr(x[perm], y, covar)
    
    p_spin = (np.sum(np.abs(null_r) >= np.abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, p_spin

# ============================================
# CRITICAL TEST: CONTROL FOR SPECTRAL EXPONENT
# ============================================
print("\n" + "="*60)
print("3. CRITICAL TEST: rho-DV controlling for spectral exponent")
print("="*60)

print("\n   Running 10,000 spin permutations...")
r_partial_spec, p_spin_spec = spin_test_partial(rho_v, z_v, spec_exp_v, hemi_v, n_perm=10000)

print(f"\n   RESULTS:")
print(f"   Baseline rho-DV:                    r = {r_base:.4f}")
print(f"   rho-DV controlling for spec_exp:    r = {r_partial_spec:.4f}")
print(f"   p_spin:                             {p_spin_spec:.4f}")
print(f"   Change from baseline:               {r_partial_spec - r_base:+.4f}")

if p_spin_spec < 0.05:
    print(f"\n   ✅ GRADIENT SURVIVES spectral exponent control!")
    print(f"   rho is NOT just a proxy for 1/f slope")
else:
    print(f"\n   ❌ GRADIENT DOES NOT SURVIVE")
    print(f"   rho may be confounded by spectral slope")

# ============================================
# ADDITIONAL CONTROLS
# ============================================
print("\n" + "="*60)
print("4. Additional controls")
print("="*60)

# Control for total power
print("\n   Controlling for TOTAL POWER...")
r_partial_pow, p_spin_pow = spin_test_partial(rho_v, z_v, total_power_v, hemi_v, n_perm=10000)
print(f"   r_partial = {r_partial_pow:.4f}, p_spin = {p_spin_pow:.4f}")
print(f"   {'SURVIVES' if p_spin_pow < 0.05 else 'DOES NOT SURVIVE'}")

# Control for gamma/delta ratio
print("\n   Controlling for GAMMA/DELTA RATIO...")
r_partial_gd, p_spin_gd = spin_test_partial(rho_v, z_v, gamma_delta_v, hemi_v, n_perm=10000)
print(f"   r_partial = {r_partial_gd:.4f}, p_spin = {p_spin_gd:.4f}")
print(f"   {'SURVIVES' if p_spin_gd < 0.05 else 'DOES NOT SURVIVE'}")

# Control for ALL confounds together
print("\n   Controlling for ALL CONFOUNDS (spec_exp + power + gamma/delta)...")
all_confounds = np.column_stack([spec_exp_v, total_power_v, gamma_delta_v])
r_partial_all, p_spin_all = spin_test_partial(rho_v, z_v, all_confounds, hemi_v, n_perm=10000)
print(f"   r_partial = {r_partial_all:.4f}, p_spin = {p_spin_all:.4f}")
print(f"   {'SURVIVES' if p_spin_all < 0.05 else 'DOES NOT SURVIVE'}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"""
BASELINE:
  rho-DV: r = {r_base:.4f}

CONFOUND CORRELATIONS:
  spectral_exponent vs rho: r = {r_spec_rho:.4f} (HIGH - potential confound)
  spectral_exponent vs DV:  r = {r_spec_z:.4f}

PARTIAL CORRELATIONS (rho-DV controlling for...):
  Spectral exponent:  r = {r_partial_spec:.4f}, p_spin = {p_spin_spec:.4f} {'✅' if p_spin_spec < 0.05 else '❌'}
  Total power:        r = {r_partial_pow:.4f}, p_spin = {p_spin_pow:.4f} {'✅' if p_spin_pow < 0.05 else '❌'}
  Gamma/delta ratio:  r = {r_partial_gd:.4f}, p_spin = {p_spin_gd:.4f} {'✅' if p_spin_gd < 0.05 else '❌'}
  ALL confounds:      r = {r_partial_all:.4f}, p_spin = {p_spin_all:.4f} {'✅' if p_spin_all < 0.05 else '❌'}

CONCLUSION:
""")

if p_spin_all < 0.05:
    print("  The rho-DV gradient SURVIVES all spectral confound controls.")
    print("  rho captures genuine dynamical structure beyond spectral composition.")
else:
    print("  ⚠️ The gradient is weakened or eliminated by spectral controls.")
    print("  Consider whether rho adds information beyond spectral features.")

print("\n" + "="*60)
print("COPY THIS OUTPUT AND SEND IT BACK")
print("="*60)
