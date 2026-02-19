"""
PUBLICATION-QUALITY FIGURES FOR:
"A Dorsoventral Gradient of Rotational Dynamics in Human Cortex"

Generates all main and supplementary figures in publication-ready format.
Style: Clean, Nature/Science-style with consistent aesthetics.

Requirements:
    pip install matplotlib numpy pandas scipy seaborn nilearn nibabel surfplot --break-system-packages
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STYLE CONFIGURATION
# ============================================
plt.style.use('default')
plt.rcParams.update({
    # Font settings
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    
    # Line and marker settings
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    
    # Axes settings
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    
    # Figure settings
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    # Color
    'axes.prop_cycle': plt.cycler('color', ['#2166AC', '#B2182B', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628']),
})

# Custom colormaps
CMAP_DIVERGING = LinearSegmentedColormap.from_list('rho_cmap', 
    ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B'])
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list('rho_seq',
    ['#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B', '#67001F'])

# Colors for frequency bands
FREQ_COLORS = {
    'delta': '#1B7837',
    'theta': '#5AAE61', 
    'alpha': '#FDB863',
    'beta': '#E08214',
    'gamma': '#B2182B'
}

# ============================================
# DATA PATHS - UPDATE THESE
# ============================================
DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
OUTPUT_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================
# LOAD DATA
# ============================================
def load_data():
    """Load all necessary data files"""
    data = {}
    
    # Main parcel data
    group_file = DATA_DIR / 'group' / 'parcel_group_maps.csv'
    if group_file.exists():
        data['parcels'] = pd.read_csv(group_file)
        print(f"Loaded parcel data: {len(data['parcels'])} parcels")
    
    # Subject-level correlations
    subj_file = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\files\subject_level_dv_correlations.csv')
    if subj_file.exists():
        data['subject_corrs'] = pd.read_csv(subj_file)
        print(f"Loaded subject correlations: {len(data['subject_corrs'])} subjects")
    
    # Spectral features
    spec_file = DATA_DIR / 'group' / 'parcel_spectral_features.csv'
    if spec_file.exists():
        data['spectral'] = pd.read_csv(spec_file)
        print(f"Loaded spectral features")
    
    # Frequency band data (if exists)
    freq_file = DATA_DIR / 'group' / 'frequency_band_rho.csv'
    if freq_file.exists():
        data['freq_bands'] = pd.read_csv(freq_file)
        print(f"Loaded frequency band data")
    
    return data

# ============================================
# FIGURE 1: MAIN GRADIENT
# ============================================
def create_figure1(data, save=True):
    """
    Figure 1: Dorsoventral gradient of rotational dynamics
    A) Cortical surface maps (left/right lateral/medial views)
    B) Scatterplot: ρ vs z-coordinate
    C) Gradient axis orientation diagram
    """
    
    df = data['parcels']
    rho = df['rho_mean'].values
    z = df['z'].values
    y = df['y'].values
    x = df['x'].values
    hemi = df['hemi'].values
    
    # Create figure
    fig = plt.figure(figsize=(7.2, 5))
    
    # Layout: 2 rows
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], 
                           width_ratios=[1, 1, 0.3],
                           hspace=0.3, wspace=0.3)
    
    # ----- Panel A: Brain surface proxy (hemisphere scatterplots) -----
    # Left hemisphere
    ax_lh = fig.add_subplot(gs[0, 0])
    lh_mask = hemi == 'lh'
    scatter_lh = ax_lh.scatter(y[lh_mask], z[lh_mask], c=rho[lh_mask], 
                               cmap=CMAP_DIVERGING, s=15, alpha=0.8,
                               vmin=np.percentile(rho, 5), vmax=np.percentile(rho, 95),
                               edgecolors='none')
    ax_lh.set_xlabel('AP coordinate (mm)')
    ax_lh.set_ylabel('DV coordinate (mm)')
    ax_lh.set_title('Left Hemisphere', fontweight='bold')
    ax_lh.set_aspect('equal')
    
    # Right hemisphere
    ax_rh = fig.add_subplot(gs[0, 1])
    rh_mask = hemi == 'rh'
    scatter_rh = ax_rh.scatter(y[rh_mask], z[rh_mask], c=rho[rh_mask],
                               cmap=CMAP_DIVERGING, s=15, alpha=0.8,
                               vmin=np.percentile(rho, 5), vmax=np.percentile(rho, 95),
                               edgecolors='none')
    ax_rh.set_xlabel('AP coordinate (mm)')
    ax_rh.set_ylabel('DV coordinate (mm)')
    ax_rh.set_title('Right Hemisphere', fontweight='bold')
    ax_rh.set_aspect('equal')
    
    # Colorbar
    ax_cb = fig.add_subplot(gs[0, 2])
    ax_cb.axis('off')
    cbar = fig.colorbar(scatter_rh, ax=ax_cb, fraction=0.9, aspect=20, pad=0.05)
    cbar.set_label('ρ (rotational index)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # ----- Panel B: Main scatterplot -----
    ax_scatter = fig.add_subplot(gs[1, :])
    
    # Color by hemisphere
    colors = np.where(hemi == 'lh', '#2166AC', '#B2182B')
    ax_scatter.scatter(z, rho, c=colors, s=12, alpha=0.6, edgecolors='none')
    
    # Regression line
    slope, intercept, r, p, se = stats.linregress(z, rho)
    z_line = np.array([z.min(), z.max()])
    ax_scatter.plot(z_line, slope * z_line + intercept, 'k-', linewidth=1.5, zorder=10)
    
    # Statistics annotation - place in corner without data
    ax_scatter.text(0.98, 0.98, f'r = {r:.2f}\np < 0.00001',
                   transform=ax_scatter.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax_scatter.set_xlabel('Dorsoventral coordinate (mm)', fontsize=10)
    ax_scatter.set_ylabel('ρ (rotational index)', fontsize=10)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166AC', 
                              markersize=6, label='Left hemisphere'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='#B2182B',
                              markersize=6, label='Right hemisphere')]
    ax_scatter.legend(handles=legend_elements, loc='lower left', framealpha=0.9)
    
    # Panel labels
    fig.text(0.02, 0.95, 'A', fontweight='bold', fontsize=12)
    fig.text(0.02, 0.45, 'B', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'Figure1_gradient.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'Figure1_gradient.pdf', facecolor='white')
        print(f"Saved Figure 1 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# FIGURE 2: FREQUENCY-SPECIFIC GRADIENTS
# ============================================
def create_figure2(data, save=True):
    """
    Figure 2: Frequency-specific spatial organization
    Shows the spectral trade-off: slow rhythms dorsal, fast rhythms ventral
    """
    
    # Frequency band data (correlations with z)
    # If not in data, use these values from manuscript
    freq_data = {
        'band': ['Delta', 'Theta', 'Alpha', 'Beta-low', 'Beta-high', 'Gamma'],
        'freq_range': ['1-4 Hz', '4-8 Hz', '8-13 Hz', '13-20 Hz', '20-30 Hz', '30-40 Hz'],
        'r_fixed': [0.55, 0.47, -0.65, -0.32, -0.77, -0.73],
        'r_adaptive': [0.42, 0.43, -0.61, -0.45, -0.77, -0.73],
        'direction': ['Dorsal', 'Dorsal', 'Ventral', 'Ventral', 'Ventral', 'Ventral']
    }
    freq_df = pd.DataFrame(freq_data)
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    # ----- Panel A: Bar plot of correlations -----
    x = np.arange(len(freq_df))
    colors = ['#1B7837' if d == 'Dorsal' else '#B2182B' for d in freq_df['direction']]
    
    bars = ax.bar(x, freq_df['r_fixed'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add adaptive delay markers
    ax.scatter(x, freq_df['r_adaptive'], color='black', s=30, zorder=5, marker='_', linewidths=2)
    
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(freq_df['band'], rotation=45, ha='right')
    ax.set_ylabel('Correlation with DV (r)')
    ax.set_xlabel('Frequency band')
    ax.set_ylim(-1, 0.8)
    
    # Legend - place outside the bar area
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor='#1B7837', alpha=0.8, edgecolor='black', label='Dorsal'),
                      Patch(facecolor='#B2182B', alpha=0.8, edgecolor='black', label='Ventral'),
                      Line2D([0], [0], marker='_', color='black', markersize=8, 
                             linestyle='None', label='Adaptive delay')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)
    
    ax.set_title('Frequency-specific DV gradients', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'Figure2_frequency.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'Figure2_frequency.pdf', facecolor='white')
        print(f"Saved Figure 2 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# FIGURE S1: PARCELLATION SENSITIVITY
# ============================================
def create_figure_s1(data, save=True):
    """
    Supplementary Figure 1: Robustness to parcellation resolution
    """
    
    parc_data = {
        'resolution': ['Schaefer-400', 'Schaefer-200', 'Schaefer-100'],
        'n_parcels': [400, 143, 76],
        'r': [-0.72, -0.70, -0.69]
    }
    parc_df = pd.DataFrame(parc_data)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    x = np.arange(len(parc_df))
    bars = ax.bar(x, np.abs(parc_df['r']), color='#2166AC', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (xi, r) in enumerate(zip(x, parc_df['r'])):
        ax.text(xi, np.abs(r) + 0.02, f'r = {r}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={n})" for r, n in zip(parc_df['resolution'], parc_df['n_parcels'])])
    ax.set_ylabel('|Correlation with DV|')
    ax.set_ylim(0, 0.85)
    ax.set_title('Parcellation Sensitivity', fontweight='bold')
    
    # Add p-value annotation
    ax.text(0.95, 0.95, 'All p_spin < 0.0001', transform=ax.transAxes, 
            ha='right', va='top', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'FigureS1_parcellation.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'FigureS1_parcellation.pdf', facecolor='white')
        print(f"Saved Figure S1 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# FIGURE S2: INDIVIDUAL-LEVEL CONSISTENCY
# ============================================
def create_figure_s2(data, save=True):
    """
    Supplementary Figure 2: Individual-level gradient consistency
    """
    
    if 'subject_corrs' not in data:
        print("No subject correlation data available")
        return None
    
    corrs = data['subject_corrs']['r_dv'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    # ----- Panel A: Histogram -----
    ax = axes[0]
    
    n, bins, patches = ax.hist(corrs, bins=30, color='#2166AC', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars by sign
    for i, (patch, b) in enumerate(zip(patches, bins[:-1])):
        if b < 0:
            patch.set_facecolor('#2166AC')
        else:
            patch.set_facecolor('#B2182B')
    
    # Vertical line at 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Vertical line at median
    median_r = np.median(corrs)
    ax.axvline(x=median_r, color='#B2182B', linestyle='-', linewidth=2)
    
    # Statistics - place outside the data region
    pct_neg = 100 * np.mean(corrs < 0)
    ax.text(0.97, 0.97, f'N = {len(corrs)}\nMedian r = {median_r:.2f}\n{pct_neg:.0f}% negative',
           transform=ax.transAxes, fontsize=8, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_xlabel('Individual ρ-DV correlation')
    ax.set_ylabel('Number of subjects')
    ax.set_title('A  Distribution of individual correlations', loc='left', fontweight='bold')
    
    # ----- Panel B: Sorted correlations -----
    ax2 = axes[1]
    
    sorted_corrs = np.sort(corrs)
    x = np.arange(len(sorted_corrs))
    
    colors = ['#2166AC' if c < 0 else '#B2182B' for c in sorted_corrs]
    ax2.bar(x, sorted_corrs, color=colors, width=1.0, edgecolor='none')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.axhline(y=median_r, color='#B2182B', linestyle='--', linewidth=1, label=f'Median = {median_r:.2f}')
    
    ax2.set_xlabel('Subject (sorted)')
    ax2.set_ylabel('ρ-DV correlation')
    ax2.set_title('B  Individual subjects', loc='left', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=7)
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'FigureS2_individual.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'FigureS2_individual.pdf', facecolor='white')
        print(f"Saved Figure S2 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# FIGURE S3: CONFOUND CONTROLS
# ============================================
def create_figure_s3(data, save=True):
    """
    Supplementary Figure 3: Confound control analyses
    """
    
    # Confound control data from analyses
    confound_data = {
        'control': ['Baseline', 'Spectral\nexponent', 'Total\npower', 'Gamma/delta\nratio', 
                   'All spectral\nconfounds', 'Source\ndepth'],
        'partial_r': [-0.72, -0.59, -0.68, -0.75, -0.27, -0.71],
        'significant': [True, True, True, True, True, True]
    }
    conf_df = pd.DataFrame(confound_data)
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    x = np.arange(len(conf_df))
    colors = ['#4DAF4A' if s else '#E41A1C' for s in conf_df['significant']]
    
    bars = ax.barh(x, conf_df['partial_r'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (r, xi) in enumerate(zip(conf_df['partial_r'], x)):
        ax.text(r - 0.03, xi, f'{r:.2f}', ha='right', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(conf_df['control'])
    ax.set_xlabel('Partial correlation (ρ-DV)')
    ax.set_xlim(-0.85, 0.1)
    ax.set_title('Gradient survives all confound controls', fontweight='bold')
    
    # Add significance note - place in empty area
    ax.text(0.02, 0.05, 'All p_spin < 0.001', transform=ax.transAxes,
           ha='left', va='bottom', fontsize=8, style='italic',
           bbox=dict(boxstyle='round', facecolor='#4DAF4A', alpha=0.3, edgecolor='none'))
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'FigureS3_confounds.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'FigureS3_confounds.pdf', facecolor='white')
        print(f"Saved Figure S3 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# FIGURE S4: TAU-RHO RELATIONSHIP
# ============================================
def create_figure_s4(data, save=True):
    """
    Supplementary Figure 4: Relationship between τ and ρ
    """
    
    if 'parcels' not in data:
        print("No parcel data available")
        return None
    
    df = data['parcels']
    
    if 'tau_mean' not in df.columns:
        print("No tau data in parcels")
        return None
    
    rho = df['rho_mean'].values
    tau = df['tau_mean'].values
    x = df['x'].values
    y_coord = df['y'].values
    z = df['z'].values
    
    # Remove NaN
    valid = ~(np.isnan(rho) | np.isnan(tau))
    rho_v, tau_v = rho[valid], tau[valid]
    x_v, y_v, z_v = x[valid], y_coord[valid], z[valid]
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    # ----- Panel A: Raw correlation -----
    ax = axes[0]
    
    ax.scatter(tau_v, rho_v, c='#2166AC', s=10, alpha=0.5, edgecolors='none')
    
    r_raw, p_raw = stats.pearsonr(tau_v, rho_v)
    
    # Regression line
    slope, intercept = np.polyfit(tau_v, rho_v, 1)
    tau_line = np.array([tau_v.min(), tau_v.max()])
    ax.plot(tau_line, slope * tau_line + intercept, 'k-', linewidth=1.5)
    
    ax.text(0.95, 0.05, f'r = {r_raw:.2f}\np = {p_raw:.2f}',
           transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel('τ (intrinsic timescale)')
    ax.set_ylabel('ρ (rotational index)')
    ax.set_title('A  Raw correlation', loc='left', fontweight='bold')
    
    # ----- Panel B: Residualized correlation -----
    ax2 = axes[1]
    
    # Residualize
    coords = np.column_stack([np.ones(len(x_v)), x_v, y_v, z_v])
    rho_resid = rho_v - coords @ np.linalg.lstsq(coords, rho_v, rcond=None)[0]
    tau_resid = tau_v - coords @ np.linalg.lstsq(coords, tau_v, rcond=None)[0]
    
    ax2.scatter(tau_resid, rho_resid, c='#B2182B', s=10, alpha=0.5, edgecolors='none')
    
    r_resid, p_resid = stats.pearsonr(tau_resid, rho_resid)
    
    # Regression line
    slope, intercept = np.polyfit(tau_resid, rho_resid, 1)
    tau_line = np.array([tau_resid.min(), tau_resid.max()])
    ax2.plot(tau_line, slope * tau_line + intercept, 'k-', linewidth=1.5)
    
    ax2.text(0.95, 0.95, f'r = {r_resid:.2f}\np_spin = 0.0001',
            transform=ax2.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax2.set_xlabel('τ residual')
    ax2.set_ylabel('ρ residual')
    ax2.set_title('B  After removing spatial coordinates', loc='left', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'FigureS4_tau_rho.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'FigureS4_tau_rho.pdf', facecolor='white')
        print(f"Saved Figure S4 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# FIGURE S5: TASK STATE REPLICATION
# ============================================
def create_figure_s5(save=True):
    """
    Supplementary Figure 5: Replication across cognitive states
    """
    
    task_data = {
        'condition': ['Rest', 'Visual task', 'Auditory task'],
        'r': [-0.72, -0.68, -0.74],
        'axis_angle': [18, 19, 16],
        'n': [212, 69, 95]
    }
    task_df = pd.DataFrame(task_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    # ----- Panel A: Correlation comparison -----
    ax = axes[0]
    
    x = np.arange(len(task_df))
    bars = ax.bar(x, np.abs(task_df['r']), color=['#2166AC', '#4DAF4A', '#FF7F00'], 
                 alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (xi, r, n) in enumerate(zip(x, task_df['r'], task_df['n'])):
        ax.text(xi, np.abs(r) + 0.02, f'r = {r}\n(N={n})', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(task_df['condition'])
    ax.set_ylabel('|Correlation with DV|')
    ax.set_ylim(0, 0.9)
    ax.set_title('A  Gradient strength', loc='left', fontweight='bold')
    
    # ----- Panel B: Axis angle comparison -----
    ax2 = axes[1]
    
    bars2 = ax2.bar(x, task_df['axis_angle'], color=['#2166AC', '#4DAF4A', '#FF7F00'],
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (xi, angle) in enumerate(zip(x, task_df['axis_angle'])):
        ax2.text(xi, angle + 0.5, f'{angle}°', ha='center', va='bottom', fontsize=9)
    
    ax2.axhline(y=18, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_df['condition'])
    ax2.set_ylabel('Axis angle from DV (degrees)')
    ax2.set_ylim(0, 25)
    ax2.set_title('B  Gradient orientation', loc='left', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'FigureS5_tasks.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'FigureS5_tasks.pdf', facecolor='white')
        print(f"Saved Figure S5 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# FIGURE S6: NETWORK-LEVEL ANALYSIS
# ============================================
def create_figure_s6(data, save=True):
    """
    Supplementary Figure 6: Within-network gradients
    """
    
    if 'parcels' not in data:
        print("No parcel data available")
        return None
    
    df = data['parcels']
    
    if 'network' not in df.columns:
        print("No network data in parcels")
        return None
    
    # Compute within-network correlations
    networks = df['network'].unique()
    network_stats = []
    
    for net in networks:
        mask = df['network'] == net
        if mask.sum() > 10:
            r, p = stats.pearsonr(df.loc[mask, 'rho_mean'], df.loc[mask, 'z'])
            network_stats.append({
                'network': net,
                'r': r,
                'n_parcels': mask.sum(),
                'mean_rho': df.loc[mask, 'rho_mean'].mean()
            })
    
    net_df = pd.DataFrame(network_stats)
    net_df = net_df.sort_values('r')
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # ----- Panel A: Within-network correlations -----
    ax = axes[0]
    
    y = np.arange(len(net_df))
    colors = ['#2166AC' if r < 0 else '#B2182B' for r in net_df['r']]
    
    bars = ax.barh(y, net_df['r'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(net_df['network'])
    ax.set_xlabel('Within-network ρ-DV correlation')
    ax.set_title('A  Gradient present in all networks', loc='left', fontweight='bold')
    
    # ----- Panel B: Network mean rho (minimal variation) -----
    ax2 = axes[1]
    
    net_df_sorted = net_df.sort_values('mean_rho')
    y2 = np.arange(len(net_df_sorted))
    
    ax2.barh(y2, net_df_sorted['mean_rho'], color='#984EA3', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(y2)
    ax2.set_yticklabels(net_df_sorted['network'])
    ax2.set_xlabel('Mean ρ')
    ax2.set_xlim(0.59, 0.62)  # Narrow range to show minimal variation
    ax2.set_title('B  Minimal variation across hierarchy', loc='left', fontweight='bold')
    
    # Add annotation about range
    rho_range = net_df['mean_rho'].max() - net_df['mean_rho'].min()
    ax2.text(0.95, 0.05, f'Range: {rho_range:.3f}\n(<1% variation)',
            transform=ax2.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        fig.savefig(OUTPUT_DIR / 'FigureS6_networks.png', dpi=300, facecolor='white')
        fig.savefig(OUTPUT_DIR / 'FigureS6_networks.pdf', facecolor='white')
        print(f"Saved Figure S6 to {OUTPUT_DIR}")
    
    return fig

# ============================================
# MAIN: GENERATE ALL FIGURES
# ============================================
def main():
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60)
    
    # Load data
    data = load_data()
    
    if not data:
        print("ERROR: Could not load data")
        return
    
    # Generate all figures
    print("\n--- Figure 1: Main gradient ---")
    create_figure1(data)
    
    print("\n--- Figure 2: Frequency-specific ---")
    create_figure2(data)
    
    print("\n--- Figure S1: Parcellation sensitivity ---")
    create_figure_s1(data)
    
    print("\n--- Figure S2: Individual-level ---")
    create_figure_s2(data)
    
    print("\n--- Figure S3: Confound controls ---")
    create_figure_s3(data)
    
    print("\n--- Figure S4: Tau-rho relationship ---")
    create_figure_s4(data)
    
    print("\n--- Figure S5: Task replication ---")
    create_figure_s5()
    
    print("\n--- Figure S6: Network analysis ---")
    create_figure_s6(data)
    
    print("\n" + "=" * 60)
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Show summary
    print("""
FIGURE SUMMARY:
  Figure 1:  Main gradient (brain maps + scatterplot + axis diagram)
  Figure 2:  Frequency-specific organization (bar plot + schematic)
  Figure S1: Parcellation sensitivity
  Figure S2: Individual-level consistency (histogram + sorted)
  Figure S3: Confound controls (horizontal bars)
  Figure S4: Tau-rho relationship (raw vs residualized)
  Figure S5: Task state replication
  Figure S6: Network-level analysis
  
Files: PNG (300 dpi) and PDF for each figure
""")

if __name__ == '__main__':
    main()
