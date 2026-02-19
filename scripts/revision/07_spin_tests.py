#!/usr/bin/env python3
"""
07_spin_tests.py
Spatial autocorrelation-preserving significance tests for key results.
Alexander-Bloch et al. (2018) spin permutation framework.

Requires:
    pip install numpy scipy brainspace matplotlib

Input:
    - rho_master.csv: Full dataset with all predictors

Output:
    - spin_test_results.csv: p_spin for all key tests
    - spin_test_nulls.npz: Null distributions (5000 permutations)
    - fig_spin_tests.png/pdf: Null distribution figure
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq
from brainspace.datasets import load_conte69
import time

# ============================================================
# CONFIGURATION
# ============================================================
RHO_FILE = 'rho_master.csv'
N_SPINS = 5000
SEED = 42

# ============================================================
# FUNCTIONS
# ============================================================
def partial_spearman_multi(x, y, covariates):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rcov = np.column_stack([stats.rankdata(c) for c in covariates])
    A = np.column_stack([rcov, np.ones(len(rx))])
    res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(res_x, res_y)[0]

def random_rotation_matrix():
    """Uniform random rotation on SO(3) via QR decomposition."""
    H = np.random.randn(3, 3)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def spin_permutation(values, centroids_lh, centroids_rh, lh_mask, rh_mask):
    """Spin permutation: rotate parcel centroids, nearest-neighbor reassignment."""
    rotated = np.empty(len(values))
    
    R_lh = random_rotation_matrix()
    rot_lh = centroids_lh @ R_lh.T
    dists = cdist(rot_lh, centroids_lh)
    nearest = np.argmin(dists, axis=1)
    lh_idx = np.where(lh_mask)[0]
    rotated[lh_idx] = values[lh_idx[nearest]]
    
    R_rh = random_rotation_matrix()
    rot_rh = centroids_rh @ R_rh.T
    dists = cdist(rot_rh, centroids_rh)
    nearest = np.argmin(dists, axis=1)
    rh_idx = np.where(rh_mask)[0]
    rotated[rh_idx] = values[rh_idx[nearest]]
    
    return rotated

def p_spin(obs, null):
    """Two-tailed p-value with continuity correction."""
    return (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)

# ============================================================
# SETUP
# ============================================================
df = pd.read_csv(RHO_FILE)
rho = df['rho'].values
z = df['z'].values
se = df['spectral_exponent'].values
tau = df['tau'].values
g2 = df['scov_G2'].values
gen_g2 = df['coher_G2'].values

# Parcel centroids on conte69
surf_lh, surf_rh = load_conte69()
pts_all = np.vstack([surf_lh.Points, surf_rh.Points])
n_lh = surf_lh.Points.shape[0]

sch_parc = np.loadtxt(
    '/usr/local/lib/python3.12/dist-packages/brainspace/datasets/parcellations/schaefer_400_conte69.csv'
)
sch_labels = np.unique(sch_parc)
sch_labels = sch_labels[sch_labels != 0]

centroids = np.zeros((400, 3))
hemi_idx = np.zeros(400, dtype=int)
for i, lab in enumerate(sch_labels):
    verts = np.where(sch_parc == lab)[0]
    centroids[i] = pts_all[verts].mean(axis=0)
    hemi_idx[i] = 0 if np.mean(verts) < n_lh else 1

lh_mask = hemi_idx == 0
rh_mask = hemi_idx == 1
lh_centroids = centroids[lh_mask]
rh_centroids = centroids[rh_mask]

# ============================================================
# OBSERVED VALUES
# ============================================================
obs = {
    'g2_raw': stats.spearmanr(rho, g2)[0],
    'g2_z': partial_spearman_multi(rho, g2, [z]),
    'g2_zst': partial_spearman_multi(rho, g2, [z, se, tau]),
    'gen_z': partial_spearman_multi(rho, gen_g2, [z]),
    'gen_zst': partial_spearman_multi(rho, gen_g2, [z, se, tau]),
}

# ============================================================
# SPIN PERMUTATIONS
# ============================================================
np.random.seed(SEED)
nulls = {k: np.zeros(N_SPINS) for k in obs}

print(f"Running {N_SPINS} spin permutations...")
t0 = time.time()

for i in range(N_SPINS):
    g2_spin = spin_permutation(g2, lh_centroids, rh_centroids, lh_mask, rh_mask)
    gen_spin = spin_permutation(gen_g2, lh_centroids, rh_centroids, lh_mask, rh_mask)
    
    nulls['g2_raw'][i] = stats.spearmanr(rho, g2_spin)[0]
    nulls['g2_z'][i] = partial_spearman_multi(rho, g2_spin, [z])
    nulls['g2_zst'][i] = partial_spearman_multi(rho, g2_spin, [z, se, tau])
    nulls['gen_z'][i] = partial_spearman_multi(rho, gen_spin, [z])
    nulls['gen_zst'][i] = partial_spearman_multi(rho, gen_spin, [z, se, tau])
    
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N_SPINS} ({time.time()-t0:.0f}s)")

print(f"Done in {time.time()-t0:.0f}s")

# ============================================================
# RESULTS
# ============================================================
results = []
labels = {
    'g2_raw': 'SCov G2, zero-order',
    'g2_z': 'SCov G2 | z',
    'g2_zst': 'SCov G2 | z + SE + τ',
    'gen_z': 'Genetic G2 | z',
    'gen_zst': 'Genetic G2 | z + SE + τ',
}

print(f"\n{'='*70}")
print(f"{'Test':<30s} {'r_obs':>8s} {'p_spin':>10s}")
print(f"{'-'*50}")

for k in obs:
    ps = p_spin(obs[k], nulls[k])
    print(f"  {labels[k]:<28s} {obs[k]:>+8.4f} {ps:>10.4f}")
    results.append({'test': labels[k], 'r_observed': obs[k], 'p_spin': ps, 'n_perms': N_SPINS})

pd.DataFrame(results).to_csv('spin_test_results.csv', index=False)
np.savez('spin_test_nulls.npz', **nulls)
print("\nSaved spin_test_results.csv and spin_test_nulls.npz")

# ============================================================
# FIGURE
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
plt.subplots_adjust(wspace=0.35)

panels = [
    ('SCov G2, zero-order', obs['g2_raw'], nulls['g2_raw']),
    ('SCov G2 | z', obs['g2_z'], nulls['g2_z']),
    ('SCov G2 | z + SE + τ', obs['g2_zst'], nulls['g2_zst']),
]

for ax, (title, ob, null) in zip(axes, panels):
    ps = p_spin(ob, null)
    ax.hist(null, bins=50, color='#BBBBBB', edgecolor='white', linewidth=0.5, density=True)
    ax.axvline(ob, color='#C0392B', linewidth=2, label=f'Observed: {ob:.3f}')
    ax.set_xlabel('Null correlation')
    ax.set_ylabel('Density')
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    ax.text(0.03, 0.95, f'p_spin = {ps:.4f}', transform=ax.transAxes,
            va='top', fontsize=9, fontweight='bold',
            color='#C0392B' if ps < 0.05 else '#666666')
    ax.legend(fontsize=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.savefig('fig_spin_tests.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_spin_tests.pdf', bbox_inches='tight')
print("Saved fig_spin_tests.png/pdf")
