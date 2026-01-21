#!/usr/bin/env python3
"""
Create comparison figure: Resting-state vs Task (Visual/Auditory)
Clearly labeled panels showing rho-DV gradient replication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "group"

# Load data
rest_file = DATA_DIR.parent / "paper_submission" / "A_MOUS" / "parcel_group_maps.csv"
visual_file = OUTPUT_DIR / "mous_visual_task_group.csv"
auditory_file = OUTPUT_DIR / "mous_auditory_task_group.csv"

def residualize(y, X):
    """Residualize y with respect to X."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    return y - X_design @ beta

# Load all data
df_rest = pd.read_csv(rest_file)
df_visual = pd.read_csv(visual_file)
df_auditory = pd.read_csv(auditory_file)

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Color scheme
lh_color = '#E24A33'
rh_color = '#348ABD'

def plot_scatter(ax, df, x_col, y_col, title, xlabel, ylabel):
    """Plot scatter with regression line."""
    colors = [lh_color if h == 'lh' else rh_color for h in df['hemi']]
    ax.scatter(df[x_col], df[y_col], c=colors, alpha=0.5, s=15, edgecolors='none')

    # Regression line
    x, y = df[x_col].values, df[y_col].values
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, 'k-', lw=2)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{title}\nr = {r:.3f}', fontsize=12, fontweight='bold')

    return r

# Row 1: rho vs z (DV axis)
r_rest = plot_scatter(axes[0, 0], df_rest, 'z', 'rho_mean',
                      'RESTING STATE\n(N=208 subjects)',
                      'z (Dorsal-Ventral axis)', 'rho (rotational index)')

r_visual = plot_scatter(axes[0, 1], df_visual, 'z', 'rho_mean',
                        'VISUAL TASK\n(N=69 subjects)',
                        'z (Dorsal-Ventral axis)', 'rho (rotational index)')

r_auditory = plot_scatter(axes[0, 2], df_auditory, 'z', 'rho_mean',
                          'AUDITORY TASK\n(N=95 subjects)',
                          'z (Dorsal-Ventral axis)', 'rho (rotational index)')

# Row 2: tau vs rho (geometry-residualized)
# Compute residualized values for rest
coords_rest = np.column_stack([df_rest['x'], df_rest['y'], df_rest['z']])
df_rest['tau_resid'] = residualize(df_rest['tau_mean'].values, coords_rest)
df_rest['rho_resid'] = residualize(df_rest['rho_mean'].values, coords_rest)

plot_scatter(axes[1, 0], df_rest, 'rho_resid', 'tau_resid',
             'RESTING STATE\ntau vs rho (geometry-residualized)',
             'rho (residualized)', 'tau (residualized)')

plot_scatter(axes[1, 1], df_visual, 'rho_resid', 'tau_resid',
             'VISUAL TASK\ntau vs rho (geometry-residualized)',
             'rho (residualized)', 'tau (residualized)')

plot_scatter(axes[1, 2], df_auditory, 'rho_resid', 'tau_resid',
             'AUDITORY TASK\ntau vs rho (geometry-residualized)',
             'rho (residualized)', 'tau (residualized)')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=lh_color, markersize=10, label='Left Hemisphere'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=rh_color, markersize=10, label='Right Hemisphere')
]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=11)

# Row labels
fig.text(0.02, 0.75, 'rho-DV\nGradient', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
fig.text(0.02, 0.25, 'tau-rho\nDissociation', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)

plt.tight_layout(rect=[0.03, 0, 1, 1])

# Save
fig.savefig(OUTPUT_DIR / "mous_rest_vs_task_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTPUT_DIR / 'mous_rest_vs_task_comparison.png'}")

plt.close()
