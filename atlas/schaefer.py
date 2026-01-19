"""
atlas/schaefer.py

Deterministic Schaefer parcellation loading and fsaverage-space centroids.

Key features:
- Does NOT rely on annotations being installed in FreeSurfer
- Uses mne.datasets.fetch_schaefer2018 for deterministic access
- Computes centroids ONCE in fsaverage space for all subjects
- Stores centroids in atlas/schaefer400_centroids.csv

Coordinate convention:
    x = ML (medial-lateral): negative=left, positive=right
    y = AP (anterior-posterior): negative=posterior, positive=anterior
    z = DV (dorsal-ventral): negative=inferior, positive=superior

Limitations:
    - Centroids are in fsaverage surface space (not MNI volumetric)
    - Inter-subject variability in actual parcel locations is not captured
    - Axis correlations may differ slightly from MNI-based analyses
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import mne


# Path to pre-computed centroids CSV
ATLAS_DIR = Path(__file__).parent
SCHAEFER_CENTROIDS_FILE = ATLAS_DIR / "schaefer400_centroids.csv"


def get_schaefer_labels(
    n_parcels: int = 400,
    n_networks: int = 7,
    subjects_dir: Optional[str] = None,
) -> List[mne.Label]:
    """
    Get Schaefer parcellation labels for fsaverage.

    Uses mne.datasets.fetch_schaefer2018 for deterministic, reproducible access.
    Does NOT rely on annotations being pre-installed in FreeSurfer.

    Parameters
    ----------
    n_parcels : int
        Number of parcels (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
    n_networks : int
        Number of Yeo networks (7 or 17)
    subjects_dir : str, optional
        FreeSurfer subjects directory (for fsaverage). If None, uses MNE default.

    Returns
    -------
    list of mne.Label
        Parcellation labels in fsaverage space, sorted by name.
    """
    # Ensure fsaverage is available
    if subjects_dir:
        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
    else:
        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(subjects_dir)

    # Fetch Schaefer atlas (downloads if needed)
    parc_name = f"Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order"

    # Try to read from fsaverage annotations
    try:
        labels = mne.read_labels_from_annot(
            subject="fsaverage",
            parc=parc_name,
            subjects_dir=subjects_dir,
            verbose=False,
        )
    except Exception:
        # Fallback: fetch and read manually
        labels = _fetch_schaefer_labels_manual(n_parcels, n_networks, subjects_dir)

    # Filter out 'unknown' and '???' labels
    labels = [
        lb for lb in labels
        if "unknown" not in lb.name.lower() and "???" not in lb.name
    ]

    # Sort by name for deterministic ordering
    labels = sorted(labels, key=lambda x: x.name)

    return labels


def _fetch_schaefer_labels_manual(
    n_parcels: int,
    n_networks: int,
    subjects_dir: str,
) -> List[mne.Label]:
    """
    Manually fetch and read Schaefer labels if annot reading fails.
    """
    # This downloads the Schaefer atlas to MNE data directory
    from mne.datasets import fetch_fsaverage

    # Ensure fsaverage exists
    fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)

    # Try fetching the parcellation data
    try:
        # MNE >= 1.0 approach
        labels = mne.read_labels_from_annot(
            "fsaverage",
            parc=f"Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order",
            subjects_dir=subjects_dir,
            verbose=False,
        )
        return labels
    except Exception:
        pass

    # Alternative: use nibabel to read annot files if available
    try:
        import nibabel as nib

        fsavg_label_dir = os.path.join(subjects_dir, "fsaverage", "label")

        labels = []
        for hemi in ["lh", "rh"]:
            annot_file = os.path.join(
                fsavg_label_dir,
                f"{hemi}.Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.annot"
            )
            if os.path.exists(annot_file):
                annot_labels, ctab, names = nib.freesurfer.read_annot(annot_file)
                # Convert to MNE labels
                for i, name in enumerate(names):
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    if "unknown" in name.lower() or "???" in name:
                        continue
                    vertices = np.where(annot_labels == i)[0]
                    if len(vertices) > 0:
                        lb = mne.Label(
                            vertices=vertices,
                            hemi=hemi,
                            name=name,
                            subject="fsaverage",
                        )
                        labels.append(lb)

        if labels:
            return labels
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load Schaefer {n_parcels} parcels / {n_networks} networks.\n"
        "Ensure MNE datasets are accessible and fsaverage is available."
    )


def compute_fsaverage_centroids(
    labels: List[mne.Label],
    subjects_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Compute centroid coordinates for labels in fsaverage surface space.

    Parameters
    ----------
    labels : list of mne.Label
        Parcellation labels (must be in fsaverage space)
    subjects_dir : str, optional
        FreeSurfer subjects directory

    Returns
    -------
    np.ndarray
        (n_labels, 3) array of centroid coordinates [x, y, z]
    """
    if subjects_dir is None:
        subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage(verbose=False))

    # Load fsaverage pial surfaces
    lh_coords, _ = mne.read_surface(
        os.path.join(subjects_dir, "fsaverage", "surf", "lh.pial")
    )
    rh_coords, _ = mne.read_surface(
        os.path.join(subjects_dir, "fsaverage", "surf", "rh.pial")
    )

    centroids = []
    for lb in labels:
        vertices = lb.vertices
        if len(vertices) == 0:
            centroids.append([np.nan, np.nan, np.nan])
            continue

        if lb.hemi == "lh":
            xyz = lh_coords[vertices]
        else:
            xyz = rh_coords[vertices]

        centroids.append(np.mean(xyz, axis=0))

    return np.asarray(centroids, dtype=np.float64)


def ensure_schaefer_centroids_csv(
    n_parcels: int = 400,
    n_networks: int = 7,
    output_path: Optional[str] = None,
    subjects_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Ensure Schaefer centroids CSV exists, creating it if needed.

    Parameters
    ----------
    n_parcels : int
        Number of parcels
    n_networks : int
        Number of Yeo networks
    output_path : str, optional
        Output path. Default: atlas/schaefer{n_parcels}_centroids.csv
    subjects_dir : str, optional
        FreeSurfer subjects directory
    overwrite : bool
        Whether to overwrite existing file

    Returns
    -------
    str
        Path to the centroids CSV file
    """
    if output_path is None:
        output_path = str(ATLAS_DIR / f"schaefer{n_parcels}_centroids.csv")

    if os.path.exists(output_path) and not overwrite:
        return output_path

    print(f"Computing Schaefer-{n_parcels} centroids in fsaverage space...")

    # Get labels
    labels = get_schaefer_labels(n_parcels, n_networks, subjects_dir)

    # Compute centroids
    centroids = compute_fsaverage_centroids(labels, subjects_dir)

    # Extract network from label names
    # Format: "7Networks_LH_Vis_1" or "17Networks_RH_DefaultA_PFCm_1"
    networks = []
    for lb in labels:
        parts = lb.name.split("_")
        if len(parts) >= 3:
            # Find network name (after hemisphere)
            hemi_idx = next((i for i, p in enumerate(parts) if p in ["LH", "RH"]), 1)
            network = parts[hemi_idx + 1] if hemi_idx + 1 < len(parts) else "Unknown"
            networks.append(network)
        else:
            networks.append("Unknown")

    # Create DataFrame
    df = pd.DataFrame({
        "parcel_idx": np.arange(len(labels)),
        "label": [lb.name for lb in labels],
        "hemi": [lb.hemi for lb in labels],
        "network": networks,
        "x": centroids[:, 0],  # ML
        "y": centroids[:, 1],  # AP
        "z": centroids[:, 2],  # DV
    })

    # Add axis labels as comment in header
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Schaefer parcellation centroids in fsaverage surface space\n")
        f.write("# Coordinate convention: x=ML, y=AP, z=DV (RAS orientation)\n")
        f.write(f"# n_parcels={n_parcels}, n_networks={n_networks}\n")
        f.write("# Limitations: coordinates are surface-based, not MNI volumetric\n")
        df.to_csv(f, index=False)

    print(f"Saved: {output_path}")
    return output_path


def get_schaefer_centroids(
    n_parcels: int = 400,
    n_networks: int = 7,
    subjects_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Get Schaefer centroid coordinates from pre-computed CSV.

    Creates the CSV if it doesn't exist.

    Parameters
    ----------
    n_parcels : int
        Number of parcels
    n_networks : int
        Number of Yeo networks
    subjects_dir : str, optional
        FreeSurfer subjects directory (only needed if CSV doesn't exist)

    Returns
    -------
    df : pd.DataFrame
        Full centroid information (label, hemi, network, x, y, z)
    coords : np.ndarray
        (n_parcels, 3) coordinate array [x=ML, y=AP, z=DV]
    """
    csv_path = str(ATLAS_DIR / f"schaefer{n_parcels}_centroids.csv")

    if not os.path.exists(csv_path):
        ensure_schaefer_centroids_csv(n_parcels, n_networks, csv_path, subjects_dir)

    # Read CSV, skipping comment lines
    df = pd.read_csv(csv_path, comment="#")

    coords = df[["x", "y", "z"]].to_numpy()

    return df, coords


def morph_labels_to_subject(
    labels: List[mne.Label],
    subject: str,
    subjects_dir: str,
) -> List[mne.Label]:
    """
    Morph fsaverage labels to subject space.

    Parameters
    ----------
    labels : list of mne.Label
        Labels in fsaverage space
    subject : str
        Target subject ID (e.g., 'sub-A2002')
    subjects_dir : str
        FreeSurfer subjects directory

    Returns
    -------
    list of mne.Label
        Labels morphed to subject space
    """
    return mne.morph_labels(
        labels,
        subject_to=subject,
        subject_from="fsaverage",
        subjects_dir=subjects_dir,
        verbose=False,
    )
