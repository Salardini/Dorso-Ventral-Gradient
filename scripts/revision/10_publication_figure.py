#!/usr/bin/env python3
"""
10_publication_figure.py — 4-panel publication-ready figure

Panel a: ρ vs z-coordinate scatter (network colored, quadratic fit)
Panel b: ρ vs SCov G2 scatter (zero-order and partial annotations)
Panel c: Progressive controls on G2 (horizontal bars)
Panel d: All predictors under toughest test (| z + SE + τ)

Requires: data/rho_master.csv
Outputs: figures/fig_publication.pdf, figures/fig_publication.png
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATA_DIR = 'data/'
FIG_DIR = 'figures/'

# Yeo network colors
NET_COLORS = {
    'Vis': '#781286', 'SomMot': '#4682B4', 'DorsAttn': '#00760E',
    'SalVentAttn': '#C43AFA', 'Limbic': '#DCF8A4',
    'Cont': '#E69422', 'Default': '#CD3E4E'
}

def partial_spearman_multi(x, y, covariates):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rcov = np.column_stack([stats.rankdata(c) for c in covariates])
    A = np.column_stack([rcov, np.ones(len(rx))])
    res_x = rx - A @ lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(res_x, res_y)


def main():
    df = pd.read_csv(DATA_DIR + 'rho_master.csv')
    rho, z, se, tau = df['rho'].values, df['z'].values, df['spectral_exponent'].values, df['tau'].values
    g2 = df['scov_G2'].values

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.32,
                  bottom=0.12, top=0.94, left=0.08, right=0.96)

    # ── Panel a: ρ vs z ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for net in NET_COLORS:
        mask = df['network'] == net
        ax_a.scatter(z[mask], rho[mask], c=NET_COLORS[net], s=12, alpha=0.7,
                     label=net, edgecolors='none')
    z_fit = np.linspace(z.min(), z.max(), 200)
    coeffs = np.polyfit(z, rho, 2)
    ax_a.plot(z_fit, np.polyval(coeffs, z_fit), 'k-', lw=2, zorder=5)
    r_s, p_s = stats.spearmanr(z, rho)
    ax_a.set_xlabel('z-coordinate (mm)', fontsize=11)
    ax_a.set_ylabel('ρ (rotational dynamics)', fontsize=11)
    ax_a.set_title(f'a   ρₛ = {r_s:.2f}, p < 10⁻⁶⁰', fontsize=12, fontweight='bold', loc='left')

    # ── Panel b: ρ vs G2 ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for net in NET_COLORS:
        mask = df['network'] == net
        ax_b.scatter(g2[mask], rho[mask], c=NET_COLORS[net], s=12, alpha=0.7, edgecolors='none')
    z_fit2 = np.linspace(g2.min(), g2.max(), 200)
    m, b = np.polyfit(g2, rho, 1)
    ax_b.plot(z_fit2, m * z_fit2 + b, 'k-', lw=2)
    r0, p0 = stats.spearmanr(g2, rho)
    r_full, p_full = partial_spearman_multi(rho, g2, [z, se, tau])
    ax_b.set_xlabel('SCov G2 (dual-origin gradient)', fontsize=11)
    ax_b.set_ylabel('ρ (rotational dynamics)', fontsize=11)
    ax_b.set_title('b   ρ vs SCov G2', fontsize=12, fontweight='bold', loc='left')
    ax_b.text(0.95, 0.95, f'ρₛ = {r0:.3f} (p = .002)\nr|z,SE,τ = {r_full:.3f}***',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=9,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # ── Panel c: Progressive controls ───────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    controls = [
        ('Zero-order', stats.spearmanr(rho, g2)[0]),
        ('| z', partial_spearman_multi(rho, g2, [z])[0]),
        ('| z + SE', partial_spearman_multi(rho, g2, [z, se])[0]),
        ('| z + SE + τ', partial_spearman_multi(rho, g2, [z, se, tau])[0]),
    ]
    labels_c, vals_c = zip(*controls)
    y_pos = np.arange(len(labels_c))
    colors_c = ['#999999', '#6699CC', '#336699', '#003366']
    bars = ax_c.barh(y_pos, vals_c, color=colors_c, height=0.6, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals_c):
        x_pos = v - 0.008 if v < -0.2 else v + 0.008
        ha = 'right' if v < -0.2 else 'left'
        ax_c.text(x_pos, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                  ha=ha, va='center', fontsize=9, fontweight='bold', color='white' if v < -0.2 else 'black')
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels_c, fontsize=10)
    ax_c.set_xlabel('Partial Spearman r', fontsize=11)
    ax_c.set_title('c   G2 strengthens with progressive controls', fontsize=12, fontweight='bold', loc='left')
    ax_c.axvline(0, color='black', lw=0.5)
    ax_c.invert_yaxis()

    # ── Panel d: All predictors under toughest test ─────────────
    ax_d = fig.add_subplot(gs[1, 1])
    predictors = [
        ('SCov G2', 'scov_G2', 'Developmental'),
        ('Genetic G2', 'coher_G2', 'Developmental'),
        ('FC G1', 'fc_G1', 'Functional'),
        ('FC clustering', 'fc_clustering', 'Functional'),
        ('SCov G1', 'scov_G1', 'Developmental'),
        ('FC participation', 'fc_participation', 'Functional'),
        ('FC G2', 'fc_G2', 'Functional'),
        ('Mesulam', 'mesulam_vertex', 'Cytoarchitect.'),
        ('SST', 'SST', 'Cell type'),
        ('PVALB', 'PVALB', 'Cell type'),
    ]
    cat_colors = {'Developmental': '#003366', 'Functional': '#CC6600',
                  'Cell type': '#669933', 'Cytoarchitect.': '#996699'}

    labels_d, vals_d, colors_d = [], [], []
    for label, col, cat in predictors:
        vals = df[col].values
        mask = ~np.isnan(vals)
        r, p = partial_spearman_multi(rho[mask], vals[mask], [z[mask], se[mask], tau[mask]])
        labels_d.append(label)
        vals_d.append(r)
        colors_d.append(cat_colors[cat])

    y_pos_d = np.arange(len(labels_d))
    bars_d = ax_d.barh(y_pos_d, vals_d, color=colors_d, height=0.6, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars_d, vals_d):
        offset = -0.008 if v < 0 else 0.008
        ha = 'right' if v < 0 else 'left'
        c = 'white' if abs(v) > 0.15 else 'black'
        ax_d.text(v + offset, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                  ha=ha, va='center', fontsize=8, fontweight='bold', color=c)
    ax_d.set_yticks(y_pos_d)
    ax_d.set_yticklabels(labels_d, fontsize=9)
    ax_d.set_xlabel('Partial r (| z + SE + τ)', fontsize=11)
    ax_d.set_title('d   All predictors — toughest test', fontsize=12, fontweight='bold', loc='left')
    ax_d.axvline(0, color='black', lw=0.5)
    ax_d.invert_yaxis()

    # Legend for categories at bottom
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in cat_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    # Network legend for panel a
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=6, label=n)
               for n, c in NET_COLORS.items()]
    ax_a.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.8,
                handletextpad=0.3, borderpad=0.3)

    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(FIG_DIR + 'fig_publication.png', dpi=300, facecolor='white')
    plt.savefig(FIG_DIR + 'fig_publication.pdf', facecolor='white')
    plt.close()
    print(f"Saved {FIG_DIR}fig_publication.pdf/png")


if __name__ == '__main__':
    main()
