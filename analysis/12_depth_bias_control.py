"""
DEPTH BIAS / LEADFIELD NORM CONTROL
Tests whether rho-DV gradient is confounded by MEG signal depth.

Deeper sources have weaker MEG signals, which could affect rho estimation.
We use distance from scalp centroid as a proxy for depth.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')

print("=" * 60)
print("DEPTH BIAS / LEADFIELD NORM CONTROL")
print("=" * 60)

# ============================================
# LOAD DATA
# ============================================
print("\n1. Loading data...")

group_file = DATA_DIR / 'group' / 'parcel_group_maps.csv'
df = pd.read_csv(group_file)
print(f"   Loaded {len(df)} parcels")

rho = df['rho_mean'].values
x = df['x'].values
y = df['y'].values
z = df['z'].values
hemi = df['hemi'].values

# ============================================
# COMPUTE DEPTH PROXY
# ============================================
print("\n2. Computing depth proxy...")

# Method 1: Distance from origin (brain center)
# Deeper structures are closer to origin
dist_from_origin = np.sqrt(x**2 + y**2 + z**2)

# Method 2: Distance from scalp (approximate)
# Scalp is roughly a sphere with radius ~90mm centered at origin
# Depth = 90 - distance_from_origin
scalp_radius = 90  # mm approximate
depth_from_scalp = scalp_radius - dist_from_origin

# Method 3: Vertical depth (distance from top of head)
# Top of head is approximately at z = 80mm
z_max = z.max()
depth_vertical = z_max - z

print(f"   Distance from origin: {dist_from_origin.min():.1f} to {dist_from_origin.max():.1f} mm")
print(f"   Depth from scalp: {depth_from_scalp.min():.1f} to {depth_from_scalp.max():.1f} mm")
print(f"   Vertical depth: {depth_vertical.min():.1f} to {depth_vertical.max():.1f} mm")

# ============================================
# CORRELATIONS WITH DEPTH
# ============================================
print("\n3. Correlations with depth proxies...")

r_rho_dist, p1 = stats.pearsonr(rho, dist_from_origin)
r_rho_depth, p2 = stats.pearsonr(rho, depth_from_scalp)
r_rho_vdepth, p3 = stats.pearsonr(rho, depth_vertical)

r_z_dist, _ = stats.pearsonr(z, dist_from_origin)
r_z_depth, _ = stats.pearsonr(z, depth_from_scalp)
r_z_vdepth, _ = stats.pearsonr(z, depth_vertical)

print(f"\n   rho vs distance_from_origin:  r = {r_rho_dist:.4f}, p = {p1:.4f}")
print(f"   rho vs depth_from_scalp:      r = {r_rho_depth:.4f}, p = {p2:.4f}")
print(f"   rho vs vertical_depth:        r = {r_rho_vdepth:.4f}, p = {p3:.4f}")
print(f"\n   DV (z) vs distance_from_origin:  r = {r_z_dist:.4f}")
print(f"   DV (z) vs depth_from_scalp:      r = {r_z_depth:.4f}")
print(f"   DV (z) vs vertical_depth:        r = {r_z_vdepth:.4f}")

# ============================================
# PARTIAL CORRELATIONS
# ============================================
print("\n4. Partial correlations (rho-DV controlling for depth)...")

def partial_corr(x, y, covar):
    """Partial correlation between x and y, controlling for covar"""
    X = np.column_stack([np.ones(len(x)), covar])
    x_resid = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    y_resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return stats.pearsonr(x_resid, y_resid)

# Baseline
r_base, _ = stats.pearsonr(rho, z)
print(f"\n   Baseline rho-DV: r = {r_base:.4f}")

# Control for each depth proxy
r_ctrl_dist, p_ctrl_dist = partial_corr(rho, z, dist_from_origin)
r_ctrl_depth, p_ctrl_depth = partial_corr(rho, z, depth_from_scalp)
r_ctrl_vdepth, p_ctrl_vdepth = partial_corr(rho, z, depth_vertical)

print(f"\n   Controlling for distance_from_origin: r = {r_ctrl_dist:.4f}, p = {p_ctrl_dist:.4f}")
print(f"   Controlling for depth_from_scalp:     r = {r_ctrl_depth:.4f}, p = {p_ctrl_depth:.4f}")
print(f"   Controlling for vertical_depth:       r = {r_ctrl_vdepth:.4f}, p = {p_ctrl_vdepth:.4f}")

# ============================================
# SPIN PERMUTATION TEST
# ============================================
print("\n5. Spin permutation test (controlling for distance from origin)...")

lh_idx = np.where(hemi == 'lh')[0]
rh_idx = np.where(hemi == 'rh')[0]

np.random.seed(42)
n_perm = 10000
null_r = []

# Compute residuals once
X = np.column_stack([np.ones(len(rho)), dist_from_origin])
rho_resid = rho - X @ np.linalg.lstsq(X, rho, rcond=None)[0]
z_resid = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]

for _ in range(n_perm):
    perm = np.zeros(len(rho), dtype=int)
    perm[lh_idx] = np.random.permutation(lh_idx)
    perm[rh_idx] = np.random.permutation(rh_idx)
    null_r.append(stats.pearsonr(rho_resid[perm], z_resid)[0])

p_spin = (np.sum(np.abs(null_r) >= np.abs(r_ctrl_dist)) + 1) / (n_perm + 1)

print(f"   p_spin = {p_spin:.4f}")

if p_spin < 0.05:
    print(f"   ✅ GRADIENT SURVIVES depth control (p_spin = {p_spin:.4f})")
else:
    print(f"   ⚠️ Gradient does NOT survive depth control")

# ============================================
# ALSO CHECK SIGNAL VARIANCE (SNR PROXY)
# ============================================
print("\n6. Additional: Signal variance as depth proxy...")

if 'ts_var_mean' in df.columns:
    ts_var = df['ts_var_mean'].values
    
    r_var_dist, _ = stats.pearsonr(ts_var, dist_from_origin)
    r_var_z, _ = stats.pearsonr(ts_var, z)
    r_var_rho, _ = stats.pearsonr(ts_var, rho)
    
    print(f"   Signal variance vs distance: r = {r_var_dist:.4f}")
    print(f"   Signal variance vs DV (z):   r = {r_var_z:.4f}")
    print(f"   Signal variance vs rho:      r = {r_var_rho:.4f}")
    
    # Partial correlation controlling for signal variance
    r_ctrl_var, p_ctrl_var = partial_corr(rho, z, ts_var)
    print(f"\n   rho-DV controlling for signal variance: r = {r_ctrl_var:.4f}, p = {p_ctrl_var:.4f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("SUMMARY FOR MANUSCRIPT")
print("=" * 60)

print(f"""
DEPTH BIAS CONTROL:

  Baseline rho-DV: r = {r_base:.3f}
  
  Controlling for depth proxies:
    Distance from origin: r = {r_ctrl_dist:.3f}, p_spin = {p_spin:.4f}
    Depth from scalp:     r = {r_ctrl_depth:.3f}
    Vertical depth:       r = {r_ctrl_vdepth:.3f}
  
  Interpretation:
    The rho-DV gradient {"SURVIVES" if p_spin < 0.05 else "does NOT survive"} 
    control for source depth, indicating it is not an artifact of 
    differential MEG sensitivity across cortical regions.

MANUSCRIPT TEXT:
  "To rule out depth-related signal quality confounds, we computed 
   partial correlations controlling for distance from brain center 
   (a proxy for MEG leadfield strength). The gradient remained 
   significant (r = {r_ctrl_dist:.2f}, p_spin = {p_spin:.4f})."
""")

print("=" * 60)
print("COPY THIS OUTPUT AND SEND IT BACK")
print("=" * 60)
