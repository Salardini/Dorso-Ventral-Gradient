#!/usr/bin/env python3
"""
04_bigbrain_mpc.py
Analyze BigBrain microstructure profile covariance (MPC) data.

NOTE: The BigBrain MPC data from MICA-MNI are in Glasser 360 parcellation,
while ρ is in Schaefer 400. This script computes BigBrain laminar features
but the Glasser→Schaefer mapping requires vertex-level Glasser labels on
conte69 for accurate spatial correspondence. Without these labels, results
should be interpreted cautiously.

Requires:
    pip install brainspace numpy scipy pandas scikit-learn

Input:
    - rho_schaefer400.csv
    - BigBrain_MPC_glasser.txt: 361 x 361 MPC matrix
    - BigBrain_intensity_profiles_glasser.txt: 15 x 361 intensity profiles
    
Data source:
    https://github.com/MICA-MNI/micaopen/tree/master/MPC

Output:
    - bigbrain_features_glasser.csv: Laminar features per Glasser parcel
    - rho_bigbrain_mapped.csv: Mapped BigBrain features + ρ (approximate)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from brainspace.gradient import GradientMaps

# ============================================================
# CONFIGURATION
# ============================================================
RHO_FILE = 'rho_schaefer400.csv'
BB_MPC_FILE = 'BigBrain_MPC_glasser.txt'
BB_PROFILES_FILE = 'BigBrain_intensity_profiles_glasser.txt'

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(RHO_FILE)
bb_mpc = np.loadtxt(BB_MPC_FILE, delimiter=',')
bb_profiles = np.loadtxt(BB_PROFILES_FILE, delimiter=',')

print(f"BigBrain MPC: {bb_mpc.shape}")
print(f"BigBrain profiles: {bb_profiles.shape}")

n_parcels_glasser = 360  # exclude potential medial wall at index 360

# ============================================================
# COMPUTE BIGBRAIN LAMINAR FEATURES
# ============================================================
print("\nComputing laminar features per Glasser parcel...")

# Profile statistics (computed across 15 equivolumetric surfaces)
bb_mean = np.mean(bb_profiles[:, :n_parcels_glasser], axis=0)
bb_std = np.std(bb_profiles[:, :n_parcels_glasser], axis=0)
bb_cv = bb_std / bb_mean
bb_skew = stats.skew(bb_profiles[:, :n_parcels_glasser], axis=0)
bb_kurt = stats.kurtosis(bb_profiles[:, :n_parcels_glasser], axis=0)

# Profile gradient (mean absolute depth-wise change)
bb_grad = np.mean(np.abs(np.diff(bb_profiles[:, :n_parcels_glasser], axis=0)), axis=0)

# MPC gradient (diffusion map embedding)
mpc_pos = np.maximum(bb_mpc[:n_parcels_glasser, :n_parcels_glasser], 0)
np.fill_diagonal(mpc_pos, 0)

gm = GradientMaps(n_components=3, approach='dm', kernel='normalized_angle', random_state=42)
gm.fit(mpc_pos)

# MPC node strength
mpc_strength = np.mean(mpc_pos, axis=1)

# Save Glasser-level features
glasser_df = pd.DataFrame({
    'glasser_idx': np.arange(n_parcels_glasser),
    'hemisphere': ['LH'] * 180 + ['RH'] * 180,
    'bb_mean_intensity': bb_mean,
    'bb_sd_intensity': bb_std,
    'bb_cv': bb_cv,
    'bb_skewness': bb_skew,
    'bb_kurtosis': bb_kurt,
    'bb_profile_gradient': bb_grad,
    'bb_mpc_G1': gm.gradients_[:, 0],
    'bb_mpc_G2': gm.gradients_[:, 1],
    'bb_mpc_strength': mpc_strength,
})

glasser_df.to_csv('bigbrain_features_glasser.csv', index=False)
print(f"Saved bigbrain_features_glasser.csv ({n_parcels_glasser} parcels)")

print(f"\nFeature ranges:")
for col in ['bb_sd_intensity', 'bb_cv', 'bb_skewness', 'bb_kurtosis', 'bb_mpc_G1']:
    print(f"  {col}: [{glasser_df[col].min():.3f}, {glasser_df[col].max():.3f}]")

# ============================================================
# APPROXIMATE MAPPING: Glasser → Schaefer (via KMeans)
# ============================================================
print("\n--- Approximate Glasser → Schaefer mapping ---")
print("WARNING: This mapping is approximate. For publication-quality results,")
print("use vertex-level Glasser labels on conte69 surface.")

try:
    from brainspace.datasets import load_conte69
    from sklearn.cluster import KMeans
    
    surf_lh, surf_rh = load_conte69()
    pts_all = np.vstack([surf_lh.Points, surf_rh.Points])
    n_lh = surf_lh.Points.shape[0]
    
    # Load Schaefer vertex labels
    import os
    sch_parc = np.loadtxt(
        '/usr/local/lib/python3.12/dist-packages/brainspace/datasets/parcellations/schaefer_400_conte69.csv'
    )
    sch_labels = np.unique(sch_parc)
    sch_labels = sch_labels[sch_labels != 0]
    
    # Compute Schaefer centroids on conte69
    sch_centroids = np.zeros((400, 3))
    sch_hemi = np.zeros(400, dtype=int)
    for i, lab in enumerate(sch_labels):
        verts = np.where(sch_parc == lab)[0]
        sch_centroids[i] = pts_all[verts].mean(axis=0)
        sch_hemi[i] = 0 if np.mean(verts) < n_lh else 1
    
    # Approximate Glasser centroids via KMeans
    km_lh = KMeans(n_clusters=180, random_state=42, n_init=5).fit(pts_all[:n_lh])
    km_rh = KMeans(n_clusters=180, random_state=42, n_init=5).fit(pts_all[n_lh:])
    gl_centroids = np.vstack([km_lh.cluster_centers_, km_rh.cluster_centers_])
    
    # Nearest-neighbor matching (hemisphere-aware)
    for feat in ['bb_sd_intensity', 'bb_cv', 'bb_skewness', 'bb_kurtosis', 
                 'bb_mpc_G1', 'bb_mpc_strength', 'bb_profile_gradient']:
        mapped = np.zeros(400)
        feat_vals = glasser_df[feat].values
        for i in range(400):
            if sch_hemi[i] == 0:
                dists = cdist(sch_centroids[i:i+1], gl_centroids[:180])[0]
                mapped[i] = feat_vals[np.argmin(dists)]
            else:
                dists = cdist(sch_centroids[i:i+1], gl_centroids[180:])[0]
                mapped[i] = feat_vals[180 + np.argmin(dists)]
        df[feat] = mapped
    
    # Validate mapping
    r_val, p_val = stats.spearmanr(df['bb_sd_intensity'], df['z'])
    print(f"\nMapping validation: BigBrain SD vs z: ρₛ = {r_val:.4f}, p = {p_val:.2e}")
    if abs(r_val) < 0.1:
        print("  ⚠ Mapping appears noisy (expected moderate SD-z correlation)")
    
    # Correlate with rho
    print(f"\nBigBrain features vs ρ (APPROXIMATE mapping):")
    for feat in ['bb_sd_intensity', 'bb_cv', 'bb_skewness', 'bb_mpc_G1']:
        r, p = stats.spearmanr(df['rho'], df[feat])
        print(f"  {feat:<25s}: ρₛ = {r:+.4f}, p = {p:.2e}")
    
    df.to_csv('rho_bigbrain_mapped.csv', index=False)
    print(f"\nSaved rho_bigbrain_mapped.csv")

except Exception as e:
    print(f"Mapping failed: {e}")
    print("Run with vertex-level Glasser labels for accurate mapping.")
