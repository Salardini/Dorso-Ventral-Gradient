#!/usr/bin/env python3
"""
02_fc_metrics.py
Compute functional connectivity metrics and gradients from HCP data.

Requires:
    pip install brainspace numpy scipy pandas

Input:
    - rho_schaefer400.csv: CSV with columns 'label', 'rho', 'x', 'y', 'z', 'hemi', 'network'

Output:
    - rho_fc_metrics.csv: Rho + all FC metrics per parcel
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from brainspace.datasets import load_group_fc
from brainspace.gradient import GradientMaps

# ============================================================
# CONFIGURATION
# ============================================================
RHO_FILE = 'rho_schaefer400.csv'
FC_THRESHOLD_PERCENTILE = 90  # for binarization
N_GRADIENT_COMPONENTS = 10

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(RHO_FILE)
coords = df[['x', 'y', 'z']].values  # MNI centroids

# Load HCP group-average FC (Schaefer 400)
fc = load_group_fc('schaefer', scale=400)
print(f"FC matrix: {fc.shape}")

# ============================================================
# FC GRADIENTS (diffusion map embedding)
# ============================================================
print("Computing FC gradients...")
fc_pos = np.maximum(fc, 0)
np.fill_diagonal(fc_pos, 0)

gm = GradientMaps(
    n_components=N_GRADIENT_COMPONENTS,
    approach='dm',
    kernel='cosine',
    random_state=42
)
gm.fit(fc_pos)

for i in range(min(5, N_GRADIENT_COMPONENTS)):
    df[f'fc_G{i+1}'] = gm.gradients_[:, i]
    r, p = stats.spearmanr(df['rho'], gm.gradients_[:, i])
    print(f"  FC G{i+1} vs ρ: ρₛ = {r:+.4f}, p = {p:.2e}")

# ============================================================
# GRAPH-THEORETIC METRICS
# ============================================================
print("Computing graph metrics...")

# Binarize FC at threshold
thresh = np.percentile(fc_pos[np.triu_indices(400, k=1)], FC_THRESHOLD_PERCENTILE)
fc_bin = (fc_pos > thresh).astype(float)
np.fill_diagonal(fc_bin, 0)

# Degree
degree = fc_bin.sum(axis=1)

# Clustering coefficient
clustering = np.zeros(400)
for i in range(400):
    neighbors = np.where(fc_bin[i] > 0)[0]
    k = len(neighbors)
    if k < 2:
        clustering[i] = 0
        continue
    subgraph = fc_bin[np.ix_(neighbors, neighbors)]
    triangles = subgraph.sum() / 2
    clustering[i] = 2 * triangles / (k * (k - 1))

df['fc_clustering'] = clustering

# Participation coefficient
# Requires network assignments (Yeo 7)
networks = df['network'].values
unique_nets = np.unique(networks)

participation = np.zeros(400)
for i in range(400):
    ki = degree[i]
    if ki == 0:
        participation[i] = 0
        continue
    neighbors = np.where(fc_bin[i] > 0)[0]
    p_sum = 0
    for net in unique_nets:
        kis = np.sum(networks[neighbors] == net)
        p_sum += (kis / ki) ** 2
    participation[i] = 1 - p_sum

df['fc_participation'] = participation

# ============================================================
# FC DISTANCE METRICS
# ============================================================
print("Computing FC distance metrics...")

# Euclidean distance matrix
dist_matrix = cdist(coords, coords)

# Mean geometric distance (to all parcels, weighted by binary FC)
mean_geom_dist = np.zeros(400)
mean_fc_dist = np.zeros(400)

for i in range(400):
    connected = np.where(fc_bin[i] > 0)[0]
    if len(connected) > 0:
        mean_geom_dist[i] = dist_matrix[i, connected].mean()
        # FC-weighted distance
        weights = fc_pos[i, connected]
        mean_fc_dist[i] = np.average(dist_matrix[i, connected], weights=weights)

df['mean_geom_dist'] = mean_geom_dist
df['mean_fc_dist'] = mean_fc_dist

# FC distance bias: residual of fc_dist after regressing out geom_dist
from numpy.linalg import lstsq
A = np.column_stack([mean_geom_dist, np.ones(400)])
beta = lstsq(A, mean_fc_dist, rcond=None)[0]
df['fc_dist_bias'] = mean_fc_dist - A @ beta

# Short/long FC ratio
median_dist = np.median(dist_matrix[np.triu_indices(400, k=1)])
short_long_ratio = np.zeros(400)
for i in range(400):
    connected = np.where(fc_bin[i] > 0)[0]
    if len(connected) > 0:
        n_short = np.sum(dist_matrix[i, connected] < median_dist)
        n_long = np.sum(dist_matrix[i, connected] >= median_dist)
        short_long_ratio[i] = n_short / max(n_long, 1)

df['short_long_fc_ratio'] = short_long_ratio

# ============================================================
# MESULAM HIERARCHY
# ============================================================
print("Assigning Mesulam cytoarchitectural types...")

try:
    from brainspace.datasets import load_conte69
    pts_lh, pts_rh = load_conte69()
    pts_all = np.vstack([pts_lh.Points, pts_rh.Points])
    
    import os
    parc_dir = os.path.dirname(os.path.abspath(__file__))
    # Try BrainSpace default location
    brainspace_parc = '/usr/local/lib/python3.12/dist-packages/brainspace/datasets/parcellations/'
    
    mes_parc = np.loadtxt(os.path.join(brainspace_parc, 'mesulam_conte69.csv'))
    sch_parc = np.loadtxt(os.path.join(brainspace_parc, 'schaefer_400_conte69.csv'))
    
    sch_labels = np.unique(sch_parc)
    sch_labels = sch_labels[sch_labels != 0]
    
    mesulam = np.zeros(400)
    for i, lab in enumerate(sch_labels):
        verts = np.where(sch_parc == lab)[0]
        mlabels = mes_parc[verts]
        mlabels = mlabels[mlabels != 0]
        if len(mlabels) > 0:
            mesulam[i] = stats.mode(mlabels, keepdims=True).mode[0]
    
    df['mesulam'] = mesulam
    r, p = stats.spearmanr(df['rho'], mesulam)
    print(f"  Mesulam vs ρ: ρₛ = {r:+.4f}, p = {p:.2e}")
except Exception as e:
    print(f"  Mesulam assignment failed: {e}")

# ============================================================
# SAVE
# ============================================================
df.to_csv('rho_fc_metrics.csv', index=False)
print(f"\nSaved rho_fc_metrics.csv with {len(df.columns)} columns")
