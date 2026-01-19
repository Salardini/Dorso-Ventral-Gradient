"""
atlas/ - Parcellation atlas utilities.

Provides deterministic loading of Schaefer parcellation labels and
fsaverage-space centroid coordinates.
"""

from .schaefer import (
    get_schaefer_labels,
    get_schaefer_centroids,
    ensure_schaefer_centroids_csv,
    SCHAEFER_CENTROIDS_FILE,
)
