"""
08_structural_connectivity.py
Structural connectivity analysis using HCP tractography data from ENIGMA Toolbox.

Tests whether white matter connectivity range predicts ρ and mediates the G2 effect.

Requirements:
    pip install enigmatoolbox  # or: pip install git+https://github.com/MICA-MNI/ENIGMA.git
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq
from enigmatoolbox.datasets import load_sc

def partial_spearman_multi(x, y, covariates):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rcov = np.column_stack([stats.rankdata(c) for c in covariates])
    A = np.column_stack([rcov, np.ones(len(rx))])
    res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(res_x, res_y)

# ── Load data ────────────────────────────────────────────────
sc_ctx, sc_ctx_labels, _, _ = load_sc(parcellation='schaefer_400')
np.fill_diagonal(sc_ctx, 0)

df = pd.read_csv('data/rho_master.csv')

# Align ENIGMA label order to our order
our_labels_clean = df['label'].str.replace('-lh', '').str.replace('-rh', '').values
enigma_to_our = {}
for e_idx, e_label in enumerate(sc_ctx_labels):
    for o_idx, o_label in enumerate(our_labels_clean):
        if e_label == o_label:
            enigma_to_our[e_idx] = o_idx
            break

inv_perm = np.argsort([enigma_to_our[i] for i in range(400)])
sc_aligned = sc_ctx[np.ix_(inv_perm, inv_perm)]
sc_pos = np.maximum(sc_aligned, 0)

# ── Compute SC metrics ───────────────────────────────────────
coords = df[['x', 'y', 'z']].values
dist_matrix = cdist(coords, coords, 'euclidean')
networks = df['network'].values

sc_strength = sc_pos.sum(axis=1)
sc_degree = (sc_pos > 0).sum(axis=1).astype(float)

sc_mean_dist = np.zeros(400)
sc_short_range = np.zeros(400)
sc_long_range = np.zeros(400)
sc_short_long_ratio = np.zeros(400)

for i in range(400):
    mask = sc_pos[i] > 0
    if mask.sum() > 0:
        weights = sc_pos[i, mask]
        dists = dist_matrix[i, mask]
        sc_mean_dist[i] = np.average(dists, weights=weights)
        short = dists < 50
        long_m = dists >= 50
        sc_short_range[i] = weights[short].sum() if short.any() else 0
        sc_long_range[i] = weights[long_m].sum() if long_m.any() else 0
        sc_short_long_ratio[i] = sc_short_range[i] / (sc_long_range[i] + 1e-10)

# Weighted clustering (Onnela et al. 2005)
sc_w_third = np.cbrt(sc_pos)
sc_clustering = np.zeros(400)
for i in range(400):
    neighbors = np.where(sc_pos[i] > 0)[0]
    k = len(neighbors)
    if k < 2:
        continue
    triangles = 0
    for j_idx, j in enumerate(neighbors):
        for k_idx in range(j_idx + 1, len(neighbors)):
            kn = neighbors[k_idx]
            if sc_pos[j, kn] > 0:
                triangles += (sc_w_third[i,j] * sc_w_third[i,kn] * sc_w_third[j,kn])
    sc_clustering[i] = 2 * triangles / (k * (k - 1))

# ── Test vs rho ──────────────────────────────────────────────
rho = df['rho'].values
z = df['z'].values
se = df['spectral_exponent'].values
tau = df['tau'].values
g2 = df['scov_G2'].values

metrics = {
    'sc_strength': sc_strength, 'sc_degree': sc_degree,
    'sc_mean_dist': sc_mean_dist, 'sc_short_range': sc_short_range,
    'sc_long_range': sc_long_range, 'sc_short_long_ratio': sc_short_long_ratio,
    'sc_clustering': sc_clustering,
}

print("=" * 70)
print("STRUCTURAL CONNECTIVITY vs ρ")
print("=" * 70)

for name, vals in metrics.items():
    r0, p0 = stats.spearmanr(rho, vals)
    r1, p1 = partial_spearman_multi(rho, vals, [z])
    r3, p3 = partial_spearman_multi(rho, vals, [z, se, tau])
    s3 = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else 'n.s.'
    print(f"  {name:30s}: ρₛ={r0:+.3f} | r|z={r1:+.3f} | r|z+SE+τ={r3:+.3f} {s3}")

# Mediation
print("\nMEDIATION")
r_g2_sc, p_g2_sc = partial_spearman_multi(rho, g2, [z, se, tau, sc_short_long_ratio])
r_sc_g2, p_sc_g2 = partial_spearman_multi(rho, sc_short_long_ratio, [z, se, tau, g2])
print(f"  G2 | z+SE+τ+SC_sl: r = {r_g2_sc:+.4f}, p = {p_g2_sc:.2e}")
print(f"  SC | z+SE+τ+G2:    r = {r_sc_g2:+.4f}, p = {p_sc_g2:.2e}")
