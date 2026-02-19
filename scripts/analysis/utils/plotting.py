#!/usr/bin/env python3
"""
Plotting utilities for Paper 1 figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Color scheme
LH_COLOR = '#E24A33'  # Red for left hemisphere
RH_COLOR = '#348ABD'  # Blue for right hemisphere


def plot_scatter_regression(ax, x, y, hemi, xlabel, ylabel, title=None,
                            show_stats=True, alpha=0.5, s=15):
    """
    Plot scatter with regression line and hemisphere coloring.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : array-like
        X values
    y : array-like
        Y values
    hemi : array-like
        Hemisphere labels ('lh' or 'rh')
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    show_stats : bool
        Whether to show correlation statistics
    alpha : float
        Point transparency
    s : int
        Point size

    Returns
    -------
    r : float
        Pearson correlation
    """
    x = np.asarray(x)
    y = np.asarray(y)
    hemi = np.asarray(hemi)

    # Color by hemisphere
    colors = [LH_COLOR if h == 'lh' else RH_COLOR for h in hemi]

    # Scatter
    ax.scatter(x, y, c=colors, alpha=alpha, s=s, edgecolors='none')

    # Regression line
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, 'k-', lw=2)

    # Labels
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if title:
        if show_stats:
            title = f'{title}\nr = {r:.3f}'
        ax.set_title(title, fontsize=12, fontweight='bold')
    elif show_stats:
        ax.set_title(f'r = {r:.3f}', fontsize=12)

    return r


def add_hemisphere_legend(fig, loc='upper right', bbox_to_anchor=(0.99, 0.99)):
    """Add hemisphere legend to figure."""
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LH_COLOR,
               markersize=10, label='Left Hemisphere'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RH_COLOR,
               markersize=10, label='Right Hemisphere')
    ]
    fig.legend(handles=legend_elements, loc=loc, bbox_to_anchor=bbox_to_anchor,
               fontsize=11)


def setup_figure_style():
    """Set up consistent figure style."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False
