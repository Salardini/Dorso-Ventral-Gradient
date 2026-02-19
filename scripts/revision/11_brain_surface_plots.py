#!/usr/bin/env python3
"""
11_brain_surface_plots.py — Render ρ and G2 on conte69 brain surfaces

Uses matplotlib 3D rendering (no VTK display required).
For publication-quality renderings, consider Connectome Workbench or pysurfer.

Requires: data/rho_master.csv, BrainSpace
Outputs: figures/brain_rho.png, figures/brain_g2.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels

DATA_DIR = 'data/'
FIG_DIR = 'figures/'


def render_brain_4view(surf_lh, surf_rh, data, cmap_name, title, fname, vmin=None, vmax=None):
    """Standard 4-view brain rendering: LH/RH lateral/medial."""
    n_lh = surf_lh.n_points
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), subplot_kw={'projection': '3d'})

    data_lh = data[:n_lh]
    data_rh = data[n_lh:]

    if vmin is None: vmin = np.nanpercentile(data, 2)
    if vmax is None: vmax = np.nanpercentile(data, 98)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    configs = [
        (surf_lh, data_lh, 10, 180, 'LH lateral'),
        (surf_lh, data_lh, 10, 0,   'LH medial'),
        (surf_rh, data_rh, 10, 0,   'RH lateral'),
        (surf_rh, data_rh, 10, 180, 'RH medial'),
    ]

    for ax, (surf, d, elev, azim, lab) in zip(axes.flat, configs):
        pts = surf.Points
        faces = surf.GetCells2D()
        face_vals = np.nanmean(d[faces], axis=1)
        face_colors = cmap(norm(face_vals))
        face_colors[np.isnan(face_vals)] = [0.85, 0.85, 0.85, 1.0]

        poly = Poly3DCollection(pts[faces], facecolors=face_colors,
                                 edgecolors='none', linewidths=0)
        ax.add_collection3d(poly)
        mid = pts.mean(axis=0)
        r = np.max(np.abs(pts - mid)) * 0.65
        ax.set_xlim(mid[0]-r, mid[0]+r)
        ax.set_ylim(mid[1]-r, mid[1]+r)
        ax.set_zlim(mid[2]-r, mid[2]+r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(lab, fontsize=10, pad=-10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, aspect=25, pad=0.01, location='bottom')
    cbar.set_label(title, fontsize=12)
    plt.savefig(fname, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {fname}")


def main():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(DATA_DIR + 'rho_master.csv')
    sch_parc = np.loadtxt(DATA_DIR + 'schaefer_400_conte69.csv')
    surf_lh, surf_rh = load_conte69()

    rho_vertex = map_to_labels(df['rho'].values, sch_parc, mask=sch_parc != 0, fill=np.nan)
    g2_vertex = map_to_labels(df['scov_G2'].values, sch_parc, mask=sch_parc != 0, fill=np.nan)

    render_brain_4view(surf_lh, surf_rh, rho_vertex, 'RdBu_r',
                       'ρ (rotational dynamics)', FIG_DIR + 'brain_rho.png')
    render_brain_4view(surf_lh, surf_rh, g2_vertex, 'RdBu_r',
                       'SCov G2 (dual-origin gradient)', FIG_DIR + 'brain_g2.png')


if __name__ == '__main__':
    main()
