#!/usr/bin/env python
"""
Figure 3 composite: The dual-origin structural covariance gradient predicts rotational dynamics.

8 panels (a-h):
  a: Heatmap of partial correlations (12 predictors × 4 control levels)
  b: Brain surface SCov G2 map (4 views: LH/RH lateral/medial)
  c: ρ vs SCov G2 scatter (network colored, ρₛ = −0.155; r|z,SE,τ = −0.249)
  d: Progressive strengthening bar chart (zero-order → full controls)
  e: Ranked predictors by |partial r| under full control (z + SE + τ)
  f: ρ vs SCov G2 within Mesulam cytoarchitectural classes
  g: ρ vs SC short/long range ratio (ρₛ = −0.273; r|z,SE,τ = −0.115)
  h: Mediation analysis diagram (G2 survives SC control; SC reduced to n.s.)

Layout: 4 rows × 2 columns
  Row 1: a (heatmap, wide) | b (brain surface, wide)
  Row 2: c (G2 scatter) | d (progressive strengthening)
  Row 3: e (ranked predictors, wide across both columns)
  Row 4: f (cytoarchitectural) | g (SC scatter) — h (mediation inset)

Target: 180 mm wide (Nature Neuroscience full-width), ~240 mm tall
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# TODO: Import actual data and panel-generation functions from revision scripts
# Key data source: data/revision/rho_master.csv
# Panel generation: scripts/revision/05_comprehensive_figure.py
#                   scripts/revision/10_publication_figure.py
#                   scripts/revision/08_structural_connectivity.py


def create_fig3_composite():
    """Create the Figure 3 composite layout."""
    fig = plt.figure(figsize=(7.08, 9.45))  # 180mm × 240mm

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.35,
                           height_ratios=[1, 1, 0.8, 1])

    # Panel a: Heatmap of partial correlations
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title('a', fontsize=10, fontweight='bold', loc='left')
    ax_a.text(0.5, 0.5, 'Predictor heatmap\n(12 predictors × 4 levels)',
              ha='center', va='center', transform=ax_a.transAxes, fontsize=8)

    # Panel b: Brain surface SCov G2
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title('b', fontsize=10, fontweight='bold', loc='left')
    ax_b.text(0.5, 0.5, 'SCov G2 brain map\n(Valk et al. 2020)',
              ha='center', va='center', transform=ax_b.transAxes, fontsize=8)

    # Panel c: ρ vs SCov G2 scatter
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title('c', fontsize=10, fontweight='bold', loc='left')
    ax_c.text(0.5, 0.5, 'ρ vs SCov G2\nρₛ = −0.155\nr|z,SE,τ = −0.249',
              ha='center', va='center', transform=ax_c.transAxes, fontsize=8)

    # Panel d: Progressive strengthening
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title('d', fontsize=10, fontweight='bold', loc='left')
    ax_d.text(0.5, 0.5, 'G2–ρ strengthening\n−0.155 → −0.249',
              ha='center', va='center', transform=ax_d.transAxes, fontsize=8)

    # Panel e: Ranked predictors (wide)
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.set_title('e', fontsize=10, fontweight='bold', loc='left')
    ax_e.text(0.5, 0.5, 'Ranked predictors by |partial r| under full control (z + SE + τ)',
              ha='center', va='center', transform=ax_e.transAxes, fontsize=8)

    # Panel f: Within cytoarchitectural classes
    ax_f = fig.add_subplot(gs[3, 0])
    ax_f.set_title('f', fontsize=10, fontweight='bold', loc='left')
    ax_f.text(0.5, 0.5, 'ρ vs G2 within\nMesulam classes\n(paralimbic: r = −0.34)',
              ha='center', va='center', transform=ax_f.transAxes, fontsize=8)

    # Panel g+h: SC scatter and mediation
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, 1],
                                                wspace=0.4)
    ax_g = fig.add_subplot(gs_inner[0, 0])
    ax_g.set_title('g', fontsize=10, fontweight='bold', loc='left')
    ax_g.text(0.5, 0.5, 'ρ vs SC\nshort/long',
              ha='center', va='center', transform=ax_g.transAxes, fontsize=7)

    ax_h = fig.add_subplot(gs_inner[0, 1])
    ax_h.set_title('h', fontsize=10, fontweight='bold', loc='left')
    ax_h.text(0.5, 0.5, 'Mediation\nG2 → SC → ρ',
              ha='center', va='center', transform=ax_h.transAxes, fontsize=7)

    return fig


if __name__ == '__main__':
    fig = create_fig3_composite()
    fig.savefig('figures/main/Figure3_composite.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('figures/main/Figure3_composite.png', dpi=300, bbox_inches='tight')
    print('Saved figures/main/Figure3_composite.pdf and .png')
    plt.show()
