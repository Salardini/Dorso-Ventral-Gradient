"""
09_bigbrain_proper.py
BigBrain cytoarchitectural analysis with proper Glasser→Schaefer mapping.

Uses vertex-level Glasser 360 labels on conte69 surface to map BigBrain
staining intensity profiles and MPC to Schaefer 400 parcels.

Requires:
    - data/glasser_360_conte69.csv (vertex-level Glasser labels)
    - data/BigBrain_intensity_profiles_glasser.txt
    - data/BigBrain_MPC_glasser.txt
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq
from brainspace.gradient import GradientMaps

def partial_spearman_multi(x, y, covariates):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rcov = np.column_stack([stats.rankdata(c) for c in covariates])
    A = np.column_stack([rcov, np.ones(len(rx))])
    res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(res_x, res_y)

# ── Load parcellations ───────────────────────────────────────
glasser = np.loadtxt('data/glasser_360_conte69.csv')
schaefer = np.loadtxt('data/schaefer_400_conte69.csv')  # from BrainSpace

sch_labels = np.unique(schaefer)
sch_labels = sch_labels[sch_labels != 0]

# Build vertex-level Glasser→Schaefer mapping
sch_to_glasser = {}
for i, s_lab in enumerate(sch_labels):
    verts = np.where(schaefer == s_lab)[0]
    g_labels = glasser[verts]
    g_labels = g_labels[g_labels != 0]
    if len(g_labels) > 0:
        vals, counts = np.unique(g_labels, return_counts=True)
        sch_to_glasser[i] = int(vals[np.argmax(counts)])
    else:
        sch_to_glasser[i] = None

# ── Load BigBrain data ───────────────────────────────────────
bb_profiles = np.loadtxt('data/BigBrain_intensity_profiles_glasser.txt', delimiter=',')
bb_mpc = np.loadtxt('data/BigBrain_MPC_glasser.txt', delimiter=',')

# Compute features per Glasser parcel
bb_std = np.std(bb_profiles, axis=0)[:360]
bb_mean = np.mean(bb_profiles, axis=0)[:360]
bb_cv = bb_std / bb_mean
bb_skew = stats.skew(bb_profiles[:, :360], axis=0)
bb_kurt = stats.kurtosis(bb_profiles[:, :360], axis=0)
bb_grad = np.mean(np.abs(np.diff(bb_profiles[:, :360], axis=0)), axis=0)

# MPC gradient
gm = GradientMaps(n_components=3, approach='dm', kernel='normalized_angle', random_state=42)
mpc_pos = np.maximum(bb_mpc[:360, :360], 0)
np.fill_diagonal(mpc_pos, 0)
gm.fit(mpc_pos)
bb_g1 = gm.gradients_[:, 0]
bb_strength = np.mean(mpc_pos, axis=1)

# ── Map to Schaefer 400 ─────────────────────────────────────
features = {'mpc_g1': bb_g1, 'mpc_strength': bb_strength,
            'sd': bb_std, 'cv': bb_cv, 'skew': bb_skew,
            'kurt': bb_kurt, 'gradient': bb_grad}

df = pd.read_csv('data/rho_master.csv')

for name, vals in features.items():
    mapped = np.full(400, np.nan)
    for i in range(400):
        g_lab = sch_to_glasser.get(i)
        if g_lab is not None:
            g_idx = g_lab - 1
            if 0 <= g_idx < len(vals):
                mapped[i] = vals[g_idx]
    df[f'bb_{name}_proper'] = mapped

# ── Test vs rho ──────────────────────────────────────────────
rho, z = df['rho'].values, df['z'].values
se, tau = df['spectral_exponent'].values, df['tau'].values

print("=" * 70)
print("BigBrain LAMINAR FEATURES vs ρ (proper Glasser→Schaefer mapping)")
print("=" * 70)

for col in [c for c in df.columns if c.startswith('bb_') and c.endswith('_proper')]:
    mask = df[col].notna()
    r0, p0 = stats.spearmanr(rho[mask], df.loc[mask, col])
    r3, p3 = partial_spearman_multi(rho[mask], df.loc[mask, col].values, 
                                     [z[mask], se[mask], tau[mask]])
    s3 = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else 'n.s.'
    print(f"  {col:30s}: ρₛ={r0:+.3f} | r|z+SE+τ={r3:+.3f} {s3}")
