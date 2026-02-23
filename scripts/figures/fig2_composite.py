#!/usr/bin/env python
"""
Figure 2 composite: Robustness and cross-modal replication.

6 panels (a-f):
  a: Task replication scatter (rest + auditory + visual overlaid)
  b: HCP fMRI replication scatter (n=1,096, r ≈ 0)
  c: Atlas sensitivity bars (Schaefer 100/200/400)
  d: Adaptive delay control (fixed vs adaptive per band)
  e: fMRI complementary measures bars (ρ, τ, SE vs z)
  f: Summary statistics table (all datasets/modalities)

Layout: 3 rows × 2 columns
  Row 1: a (task replication) | b (HCP replication)
  Row 2: c (atlas sensitivity) | d (adaptive delay)
  Row 3: e (fMRI measures) | f (summary table)

Target: 180 mm wide (Nature Neuroscience full-width), ~165 mm tall
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def create_fig2_composite():
    """Create the Figure 2 composite layout."""
    fig = plt.figure(figsize=(7.08, 6.50))  # 180mm × 165mm

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    # Panel a: Task replication
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title('a', fontsize=10, fontweight='bold', loc='left')
    ax_a.text(0.5, 0.5, 'Task replication\n(rest + auditory + visual)',
              ha='center', va='center', transform=ax_a.transAxes)

    # Panel b: HCP fMRI
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title('b', fontsize=10, fontweight='bold', loc='left')
    ax_b.text(0.5, 0.5, 'HCP fMRI replication\n(n=1,096, r ≈ 0)',
              ha='center', va='center', transform=ax_b.transAxes)

    # Panel c: Atlas sensitivity
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title('c', fontsize=10, fontweight='bold', loc='left')
    ax_c.text(0.5, 0.5, 'Atlas sensitivity\n(Schaefer 100/200/400)',
              ha='center', va='center', transform=ax_c.transAxes)

    # Panel d: Adaptive delay
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title('d', fontsize=10, fontweight='bold', loc='left')
    ax_d.text(0.5, 0.5, 'Adaptive delay\n(fixed vs adaptive)',
              ha='center', va='center', transform=ax_d.transAxes)

    # Panel e: fMRI measures
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.set_title('e', fontsize=10, fontweight='bold', loc='left')
    ax_e.text(0.5, 0.5, 'fMRI measures bars\n(ρ, τ, SE vs z)',
              ha='center', va='center', transform=ax_e.transAxes)

    # Panel f: Summary table
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.set_title('f', fontsize=10, fontweight='bold', loc='left')
    ax_f.text(0.5, 0.5, 'Summary statistics\ntable', ha='center', va='center',
              transform=ax_f.transAxes)
    ax_f.axis('off')

    return fig


if __name__ == '__main__':
    fig = create_fig2_composite()
    fig.savefig('Figure2_composite.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('Figure2_composite.png', dpi=300, bbox_inches='tight')
    print('Saved Figure2_composite.pdf and .png')
    plt.show()
