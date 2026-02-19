#!/usr/bin/env python3
"""
05_comprehensive_analysis.py
Comprehensive correlation and partial correlation analysis of all predictors
against ρ. Generates the main summary figure.

Requires:
    pip install numpy scipy pandas matplotlib scikit-learn

Input:
    - rho_with_valk.csv: Merged dataset with all predictors
      (output from scripts 01-03)

Output:
    - fig_comprehensive.png/pdf: Main 6-panel figure
    - predictor_table.csv: Full correlation table
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression

# ============================================================
# FUNCTIONS
# ============================================================
def partial_spearman(x, y, z):
    """Partial Spearman correlation controlling for z."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    A = np.column_stack([rz, np.ones(len(rz))])
    res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(res_x, res_y)

def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('rho_with_valk.csv')
print(f"Loaded {len(df)} parcels, {len(df.columns)} columns")

# Network colors
NET_COLORS = {
    'Vis': '#9B59B6', 'SomMot': '#3498DB', 'DorsAttn': '#27AE60',
    'SalVentAttn': '#E74C3C', 'Cont': '#F39C12', 'Default': '#E91E63',
    'Limbic': '#795548'
}

# ============================================================
# FIGURE: 6 panels
# ============================================================
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

# --- Panel A: ρ vs z ---
ax = fig.add_subplot(gs[0, 0])
for net, color in NET_COLORS.items():
    sub = df[df['network'] == net]
    ax.scatter(sub['z'], sub['rho'], c=color, s=10, alpha=0.5, label=net)
r, p = stats.spearmanr(df['rho'], df['z'])
ax.set_xlabel('MNI z-coordinate', fontsize=9)
ax.set_ylabel('ρ (rotational dynamics)', fontsize=9)
ax.set_title(f'A  ρ vs Dorsoventral Axis\nρₛ = {r:.3f}***', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=5.5, loc='upper right', ncol=2, framealpha=0.7)

# --- Panel B: ρ vs SCov G2 ---
ax = fig.add_subplot(gs[0, 1])
for net, color in NET_COLORS.items():
    sub = df[df['network'] == net]
    ax.scatter(sub['scov_G2'], sub['rho'], c=color, s=10, alpha=0.5)
r, p = stats.spearmanr(df['rho'], df['scov_G2'])
rp, pp = partial_spearman(df['rho'].values, df['scov_G2'].values, df['z'].values)
ax.set_xlabel('Structural Covariance G2 (Valk et al.)', fontsize=9)
ax.set_ylabel('ρ', fontsize=9)
ax.set_title(f'B  ρ vs SCov G2 (dual origin)\nρₛ = {r:.3f}**, r|z = {rp:.3f}***', fontsize=10, fontweight='bold', loc='left')

# --- Panel C: ρ vs Genetic G2 ---
ax = fig.add_subplot(gs[0, 2])
for net, color in NET_COLORS.items():
    sub = df[df['network'] == net]
    ax.scatter(sub['coher_G2'], sub['rho'], c=color, s=10, alpha=0.5)
r, p = stats.spearmanr(df['rho'], df['coher_G2'])
rp, pp = partial_spearman(df['rho'].values, df['coher_G2'].values, df['z'].values)
ax.set_xlabel('Genetic Correlation G2 (Valk et al.)', fontsize=9)
ax.set_ylabel('ρ', fontsize=9)
ax.set_title(f'C  ρ vs Genetic G2\nρₛ = {r:.3f}**, r|z = {rp:.3f}***', fontsize=10, fontweight='bold', loc='left')

# --- Panel D: ρ vs PVALB (null) ---
ax = fig.add_subplot(gs[0, 3])
mask_pv = df['PVALB'].notna()
for net, color in NET_COLORS.items():
    sub = df[(df['network'] == net) & mask_pv]
    ax.scatter(sub['PVALB'], sub['rho'], c=color, s=10, alpha=0.5)
r, p = stats.spearmanr(df.loc[mask_pv, 'rho'], df.loc[mask_pv, 'PVALB'])
ax.set_xlabel('PVALB expression (z-scored)', fontsize=9)
ax.set_ylabel('ρ', fontsize=9)
ax.set_title(f'D  ρ vs PVALB Expression\nρₛ = {r:.3f}, n.s.', fontsize=10, fontweight='bold', loc='left')

# --- Panel E: Zero-order effect sizes ---
ax = fig.add_subplot(gs[1, 0:2])

predictors_zero = [
    ('Z-coordinate', 'z', None, 'arch'),
    ('SCov G2', 'scov_G2', None, 'arch'),
    ('SCov G1', 'scov_G1', None, 'arch'),
    ('Genetic G2', 'coher_G2', None, 'arch'),
    ('FC Gradient 1', 'fc_G1', None, 'fc'),
    ('FC Gradient 2', 'fc_G2', None, 'fc'),
    ('FC Clustering', 'fc_clustering', None, 'fc'),
    ('FC Participation', 'fc_participation', None, 'fc'),
    ('PVALB', 'PVALB', mask_pv, 'intern'),
    ('SST', 'SST', df['SST'].notna(), 'intern'),
]

effects = []
for label, col, mask, cat in predictors_zero:
    if mask is not None:
        r, p = stats.spearmanr(df.loc[mask, 'rho'], df.loc[mask, col])
    else:
        r, p = stats.spearmanr(df['rho'], df[col])
    effects.append((label, r, p, cat))

effects.sort(key=lambda x: abs(x[1]))
cat_colors = {'arch': '#2980B9', 'fc': '#F39C12', 'intern': '#E74C3C'}
bar_colors = [cat_colors[e[3]] for e in effects]

ax.barh(range(len(effects)), [e[1] for e in effects], color=bar_colors, alpha=0.8, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(effects)))
ax.set_yticklabels([e[0] for e in effects], fontsize=8)
ax.set_xlabel('Spearman ρₛ with ρ', fontsize=9)
ax.set_title('E  Zero-order Correlations', fontsize=10, fontweight='bold', loc='left')
ax.axvline(0, color='gray', linewidth=0.5)
for i, (_, r, p, _) in enumerate(effects):
    ax.text(r + 0.01 if r >= 0 else r - 0.01, i, f'{r:.3f} {sig_label(p)}',
            va='center', ha='left' if r >= 0 else 'right', fontsize=7)
legend_handles = [
    Patch(facecolor='#2980B9', label='Structural/Developmental'),
    Patch(facecolor='#F39C12', label='Functional connectivity'),
    Patch(facecolor='#E74C3C', label='Interneuron expression'),
]
ax.legend(handles=legend_handles, fontsize=7, loc='lower right')

# --- Panel F: Partial correlations (z-controlled) ---
ax = fig.add_subplot(gs[1, 2:4])

partial_preds = [
    ('SCov G2', 'scov_G2', 'arch'),
    ('Genetic G2', 'coher_G2', 'arch'),
    ('SCov G1', 'scov_G1', 'arch'),
    ('SCov G3', 'scov_G3', 'arch'),
    ('FC Gradient 1', 'fc_G1', 'fc'),
    ('FC Gradient 2', 'fc_G2', 'fc'),
    ('FC Clustering', 'fc_clustering', 'fc'),
    ('FC Participation', 'fc_participation', 'fc'),
]

partials = []
for label, col, cat in partial_preds:
    rp, pp = partial_spearman(df['rho'].values, df[col].values, df['z'].values)
    partials.append((label, rp, pp, cat))

partials.sort(key=lambda x: abs(x[1]))
bar_colors_p = [cat_colors[e[3]] for e in partials]

ax.barh(range(len(partials)), [e[1] for e in partials], color=bar_colors_p, alpha=0.8, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(partials)))
ax.set_yticklabels([e[0] for e in partials], fontsize=8)
ax.set_xlabel('Partial r with ρ (controlling for z)', fontsize=9)
ax.set_title('F  Partial Correlations (z-controlled)', fontsize=10, fontweight='bold', loc='left')
ax.axvline(0, color='gray', linewidth=0.5)
for i, (_, r, p, _) in enumerate(partials):
    ax.text(r + 0.01 if r >= 0 else r - 0.01, i, f'{r:.3f} {sig_label(p)}',
            va='center', ha='left' if r >= 0 else 'right', fontsize=7)
ax.legend(handles=legend_handles[:2], fontsize=7, loc='lower right')

fig.suptitle('Dual-Origin Structural Covariance Predicts Rotational Dynamics Beyond the Dorsoventral Axis',
             fontsize=13, fontweight='bold', y=0.98)

plt.savefig('fig_comprehensive.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_comprehensive.pdf', bbox_inches='tight')
print("Saved fig_comprehensive.png and fig_comprehensive.pdf")

# ============================================================
# PREDICTOR TABLE
# ============================================================
table_rows = []
for label, col, mask, cat in predictors_zero:
    if mask is not None:
        r, p = stats.spearmanr(df.loc[mask, 'rho'], df.loc[mask, col])
        rp, pp = partial_spearman(df.loc[mask, 'rho'].values, df.loc[mask, col].values, df.loc[mask, 'z'].values)
        n = mask.sum()
    else:
        r, p = stats.spearmanr(df['rho'], df[col])
        if col == 'z':
            rp, pp = np.nan, np.nan
        else:
            rp, pp = partial_spearman(df['rho'].values, df[col].values, df['z'].values)
        n = 400
    table_rows.append({
        'predictor': label, 'category': cat, 'n': n,
        'rho_s': r, 'p_zero': p,
        'partial_r_z': rp, 'p_partial_z': pp,
    })

table_df = pd.DataFrame(table_rows)
table_df.to_csv('predictor_table.csv', index=False)
print("Saved predictor_table.csv")
