#!/usr/bin/env python
"""
Figure 1 composite: A dorsoventral gradient of rotational dynamics in human cortex.

8 panels (a-h):
  a: Schematic (time series → state space → eigenvalues → high/low ρ)
  b: Brain surface ρ map (4 views: LH/RH lateral/medial)
  c: Cortical flatmap (Voronoi, hemispheres separated)
  d: ρ vs z scatter (network colored, quadratic fit, ρₛ = −0.73)
  e: Individual r(ρ,z) histogram (n=25, mean = −0.43, 100% < 0)
  f: Frequency band bars (δ through broadband)
  g: τ–ρ residual scatter (residualized for xyz, ρₛ = −0.34)
  h: fMRI ρ vs z scatter (ρₛ = 0.13)

Layout: 4 rows × 2 columns
  Row 1: a (schematic, wide) | b (brain surface, wide)
  Row 2: c (flatmap) | d (scatter)
  Row 3: e (histogram) | f (frequency bars)
  Row 4: g (τ–ρ scatter) | h (fMRI scatter)

Target: 180 mm wide (Nature Neuroscience full-width), ~220 mm tall
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# TODO: Import actual data and panel-generation functions
# from fig1a_schematic import plot_schematic
# from render_brain_rho import plot_brain_surface
# from fig1c_flatmap import plot_flatmap
# etc.


def create_fig1_composite():
    """Create the Figure 1 composite layout."""
    fig = plt.figure(figsize=(7.08, 8.66))  # 180mm × 220mm at 1 inch = 25.4 mm

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.30)

    # Panel a: Schematic
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title('a', fontsize=10, fontweight='bold', loc='left')
    ax_a.text(0.5, 0.5, 'Schematic\n(time series → state space\n→ eigenvalues → ρ)',
              ha='center', va='center', transform=ax_a.transAxes)

    # Panel b: Brain surface
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title('b', fontsize=10, fontweight='bold', loc='left')
    ax_b.text(0.5, 0.5, 'Brain surface ρ map\n(4 views)', ha='center', va='center',
              transform=ax_b.transAxes)

    # Panel c: Flatmap
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title('c', fontsize=10, fontweight='bold', loc='left')
    ax_c.text(0.5, 0.5, 'Cortical flatmap\n(Voronoi)', ha='center', va='center',
              transform=ax_c.transAxes)

    # Panel d: ρ vs z scatter
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title('d', fontsize=10, fontweight='bold', loc='left')
    ax_d.text(0.5, 0.5, 'ρ vs z scatter\n(ρₛ = −0.73)', ha='center', va='center',
              transform=ax_d.transAxes)

    # Panel e: Individual histogram
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.set_title('e', fontsize=10, fontweight='bold', loc='left')
    ax_e.text(0.5, 0.5, 'Individual r(ρ,z)\nhistogram', ha='center', va='center',
              transform=ax_e.transAxes)

    # Panel f: Frequency bands
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.set_title('f', fontsize=10, fontweight='bold', loc='left')
    ax_f.text(0.5, 0.5, 'Frequency band bars\n(δ through broadband)', ha='center', va='center',
              transform=ax_f.transAxes)

    # Panel g: τ–ρ residual scatter
    ax_g = fig.add_subplot(gs[3, 0])
    ax_g.set_title('g', fontsize=10, fontweight='bold', loc='left')
    ax_g.text(0.5, 0.5, 'τ–ρ residual scatter\n(ρₛ = −0.34)', ha='center', va='center',
              transform=ax_g.transAxes)

    # Panel h: fMRI scatter
    ax_h = fig.add_subplot(gs[3, 1])
    ax_h.set_title('h', fontsize=10, fontweight='bold', loc='left')
    ax_h.text(0.5, 0.5, 'fMRI ρ vs z scatter\n(ρₛ = 0.13)', ha='center', va='center',
              transform=ax_h.transAxes)

    return fig


if __name__ == '__main__':
    fig = create_fig1_composite()
    fig.savefig('Figure1_composite.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('Figure1_composite.png', dpi=300, bbox_inches='tight')
    print('Saved Figure1_composite.pdf and .png')
    plt.show()
