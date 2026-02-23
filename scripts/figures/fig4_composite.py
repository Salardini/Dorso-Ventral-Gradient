#!/usr/bin/env python
"""
Figure 4 composite: Excitatory-inhibitory balance model and interneuron gene expression.

6 panels (a-f):
  a: E-I computational model (inhibitory gain sweep)
  b: Model τ-ρ tradeoff scatter (r = −0.84)
  c: Gene expression correlation bars (PVALB, SST, VIP, GAD1, GAD2, PV−SST)
  d: PVALB vs ρ scatter (r = −0.07, n.s.)
  e: SST vs ρ scatter (r = 0.12*)
  f: PV−SST vs ρ scatter (r = −0.12*)

Layout: 3 rows × 2 columns
  Row 1: a (E-I model) | b (model scatter)
  Row 2: c (gene bars, wide) spanning both columns
  Row 3: d (PVALB) | e (SST) | f (PV−SST) — 3 columns in bottom row

Target: 180 mm wide (Nature Neuroscience full-width), ~165 mm tall
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def create_fig4_composite():
    """Create the Figure 4 composite layout."""
    fig = plt.figure(figsize=(7.08, 6.50))  # 180mm × 165mm

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.30,
                           height_ratios=[1, 0.8, 1])

    # Panel a: E-I model
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.set_title('a', fontsize=10, fontweight='bold', loc='left')
    ax_a.text(0.5, 0.5, 'E-I network model\n(inhibitory gain sweep)',
              ha='center', va='center', transform=ax_a.transAxes)

    # Panel b: Model scatter
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_title('b', fontsize=10, fontweight='bold', loc='left')
    ax_b.text(0.5, 0.5, 'Model τ vs ρ\n(r = −0.84)',
              ha='center', va='center', transform=ax_b.transAxes)

    # Panel c: Gene bars (spanning full width)
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_title('c', fontsize=10, fontweight='bold', loc='left')
    ax_c.text(0.5, 0.5, 'Gene expression correlation bars\n(PVALB, SST, VIP, GAD1, GAD2, PV−SST)',
              ha='center', va='center', transform=ax_c.transAxes)

    # Panel d: PVALB scatter
    ax_d = fig.add_subplot(gs[2, 0])
    ax_d.set_title('d', fontsize=10, fontweight='bold', loc='left')
    ax_d.text(0.5, 0.5, 'PVALB vs ρ\n(r = −0.07, n.s.)',
              ha='center', va='center', transform=ax_d.transAxes)

    # Panel e: SST scatter
    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.set_title('e', fontsize=10, fontweight='bold', loc='left')
    ax_e.text(0.5, 0.5, 'SST vs ρ\n(r = 0.12*)',
              ha='center', va='center', transform=ax_e.transAxes)

    # Panel f: PV−SST scatter
    ax_f = fig.add_subplot(gs[2, 2])
    ax_f.set_title('f', fontsize=10, fontweight='bold', loc='left')
    ax_f.text(0.5, 0.5, 'PV−SST vs ρ\n(r = −0.12*)',
              ha='center', va='center', transform=ax_f.transAxes)

    return fig


if __name__ == '__main__':
    fig = create_fig4_composite()
    fig.savefig('Figure4_composite.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('Figure4_composite.png', dpi=300, bbox_inches='tight')
    print('Saved Figure4_composite.pdf and .png')
    plt.show()
