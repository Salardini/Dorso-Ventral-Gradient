#!/usr/bin/env python3
"""
Compute Yeo17 split-label centroids for HCP MEG data and run rho vs DV correlation.

The HCP MEG data uses a split Yeo17 parcellation (107 parcels = subregions within 17 networks).
MNE provides the coarse Yeo17 (17 networks per hemisphere). We reconstruct subparcel centroids
by using k-means clustering within each network to match the HCP subparcel count.
"""

import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.cluster.vq import kmeans2
import re
from pathlib import Path

# Paths
HCP_DIR = Path(__file__).parent
METRICS_FILE = HCP_DIR / "hcp_subject_metrics.csv"
CENTROIDS_OUT = HCP_DIR / "yeo17_split_centroids.csv"
CORR_STATS_OUT = HCP_DIR / "hcp_correlation_stats.csv"

N_PERM = 5000


def get_fsaverage_and_labels():
    """Fetch fsaverage and read Yeo17 parcellation labels."""
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = Path(fs_dir).parent

    labels = mne.read_labels_from_annot(
        'fsaverage', parc='Yeo2011_17Networks_N1000',
        subjects_dir=subjects_dir, verbose=False
    )

    lh_pial = mne.read_surface(Path(fs_dir) / 'surf' / 'lh.pial')
    rh_pial = mne.read_surface(Path(fs_dir) / 'surf' / 'rh.pial')

    return labels, lh_pial, rh_pial


def parse_mne_label_name(label_name):
    """Parse MNE label name like '17Networks_12-lh'."""
    if 'Medial_Wall' in label_name:
        return None, None, True
    match = re.match(r'17Networks_(\d+)-(\w+)', label_name)
    if match:
        return int(match.group(1)), match.group(2), False
    return None, None, True


def parse_hcp_parcel_name(name):
    """Parse HCP parcel name like 'lh.17Networks_12-3'."""
    if name.startswith('lh.'):
        hemi = 'lh'
        rest = name[3:]
    elif name.startswith('rh.'):
        hemi = 'rh'
        rest = name[3:]
    else:
        return None, None, None, True

    if 'Medial_Wall' in rest:
        return hemi, None, None, True

    match = re.match(r'17Networks_(\d+)-(\d+)', rest)
    if match:
        return hemi, int(match.group(1)), int(match.group(2)), False
    return hemi, None, None, True


def cluster_network_into_subparcels(label, pial_coords, n_subparcels, seed=42):
    """
    Cluster a network's vertices into n_subparcels using k-means.
    Returns list of (centroid_x, centroid_y, centroid_z) for each cluster.
    Clusters are sorted by their centroid position for consistent ordering.
    """
    vertices = label.vertices
    coords = pial_coords[vertices]

    if len(vertices) < n_subparcels:
        # Not enough vertices - return the overall centroid for all
        centroid = coords.mean(axis=0)
        return [centroid] * n_subparcels

    # K-means clustering
    np.random.seed(seed)
    centroids, cluster_labels = kmeans2(coords, n_subparcels, minit='++')

    # Sort clusters by a consistent criterion (e.g., by x then y then z)
    # This helps with reproducibility
    sort_idx = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
    centroids = centroids[sort_idx]

    return [centroids[i] for i in range(n_subparcels)]


def main():
    print("=" * 60)
    print("Computing Yeo17 split-label centroids for HCP MEG data")
    print("=" * 60)

    # Step 1: Read HCP parcel names and count subparcels per network
    print("\n[1/7] Reading HCP parcel names...")
    df_hcp = pd.read_csv(METRICS_FILE)
    parcels = df_hcp[['parcel_idx', 'parcel_name']].drop_duplicates().sort_values('parcel_idx')
    print(f"    Found {len(parcels)} unique parcels")

    # Count subparcels per (hemi, network)
    subparcel_counts = {}
    parcel_info = []
    for _, row in parcels.iterrows():
        hemi, network, subparcel, is_medial = parse_hcp_parcel_name(row['parcel_name'])
        parcel_info.append({
            'parcel_idx': row['parcel_idx'],
            'parcel_name': row['parcel_name'],
            'hemi': hemi,
            'network': network,
            'subparcel': subparcel,
            'is_medial_wall': is_medial
        })
        if not is_medial:
            key = (hemi, network)
            if key not in subparcel_counts:
                subparcel_counts[key] = 0
            subparcel_counts[key] = max(subparcel_counts[key], subparcel)

    print(f"    Subparcels per network:")
    for key in sorted(subparcel_counts.keys()):
        print(f"      {key[0]}.Network{key[1]:2d}: {subparcel_counts[key]} subparcels")

    # Step 2: Fetch fsaverage and labels
    print("\n[2/7] Fetching fsaverage and Yeo17 parcellation...")
    labels, lh_pial, rh_pial = get_fsaverage_and_labels()
    lh_coords, _ = lh_pial
    rh_coords, _ = rh_pial
    print(f"    Found {len(labels)} MNE labels")

    # Build label lookup
    label_lookup = {}
    for label in labels:
        network, hemi, is_medial = parse_mne_label_name(label.name)
        if not is_medial and network is not None:
            label_lookup[(hemi, network)] = label

    # Step 3: Cluster each network into subparcels and compute centroids
    print("\n[3/7] Clustering networks into subparcels...")
    subparcel_centroids = {}  # (hemi, network, subparcel) -> (x, y, z)

    for (hemi, network), n_subparcels in subparcel_counts.items():
        label = label_lookup.get((hemi, network))
        if label is None:
            print(f"    WARNING: No label for {hemi}.Network{network}")
            continue

        coords = lh_coords if hemi == 'lh' else rh_coords
        centroids = cluster_network_into_subparcels(label, coords, n_subparcels)

        for i, centroid in enumerate(centroids):
            subparcel_centroids[(hemi, network, i + 1)] = centroid

        print(f"    {hemi}.Network{network:2d}: {n_subparcels} subparcels clustered")

    print(f"    Total {len(subparcel_centroids)} subparcel centroids computed")

    # Step 4: Assign centroids to HCP parcels
    print("\n[4/7] Assigning centroids to HCP parcels...")
    centroid_data = []

    for info in parcel_info:
        if info['is_medial_wall']:
            centroid_data.append({
                'parcel_idx': info['parcel_idx'],
                'parcel_name': info['parcel_name'],
                'hemi': info['hemi'],
                'network': 0,
                'x': np.nan,
                'y': np.nan,
                'z': np.nan,
                'is_medial_wall': True
            })
            continue

        key = (info['hemi'], info['network'], info['subparcel'])
        if key in subparcel_centroids:
            centroid = subparcel_centroids[key]
            centroid_data.append({
                'parcel_idx': info['parcel_idx'],
                'parcel_name': info['parcel_name'],
                'hemi': info['hemi'],
                'network': info['network'],
                'x': centroid[0],
                'y': centroid[1],
                'z': centroid[2],
                'is_medial_wall': False
            })
        else:
            print(f"    WARNING: No centroid for {info['parcel_name']}")
            centroid_data.append({
                'parcel_idx': info['parcel_idx'],
                'parcel_name': info['parcel_name'],
                'hemi': info['hemi'],
                'network': info['network'],
                'x': np.nan,
                'y': np.nan,
                'z': np.nan,
                'is_medial_wall': False
            })

    df_centroids = pd.DataFrame(centroid_data)

    # Step 5: Save centroids
    print("\n[5/7] Saving centroids...")
    df_centroids.to_csv(CENTROIDS_OUT, index=False)
    print(f"    Saved to {CENTROIDS_OUT}")

    n_valid = df_centroids[~df_centroids['is_medial_wall']]['x'].notna().sum()
    n_medial = df_centroids['is_medial_wall'].sum()
    print(f"    {n_valid} parcels with valid centroids, {n_medial} medial wall parcels")

    # Check z-coordinate distribution
    valid_z = df_centroids[~df_centroids['is_medial_wall']]['z'].dropna()
    print(f"    z-coordinate range: [{valid_z.min():.1f}, {valid_z.max():.1f}]")
    print(f"    z-coordinate unique values: {len(valid_z.unique())}")

    # Step 6: Aggregate HCP metrics to parcel-level means (alpha band)
    print("\n[6/7] Aggregating HCP metrics (alpha band)...")
    df_alpha = df_hcp[df_hcp['band'] == 'alpha'].copy()

    parcel_means = df_alpha.groupby(['parcel_idx', 'parcel_name']).agg({
        'rho': 'mean',
        'tau': 'mean',
        'rho_r2': 'mean',
        'tau_exp_r2': 'mean'
    }).reset_index()
    parcel_means.columns = ['parcel_idx', 'parcel_name', 'rho_mean', 'tau_mean', 'rho_r2_mean', 'tau_r2_mean']

    n_subjects = df_alpha['subject'].nunique()
    n_runs = df_alpha['run'].nunique()
    print(f"    Aggregated {len(parcel_means)} parcels from {n_subjects} subjects x {n_runs} runs")

    # Merge with centroids
    df_merged = parcel_means.merge(df_centroids, on=['parcel_idx', 'parcel_name'])

    # Exclude medial wall for correlation
    df_valid = df_merged[~df_merged['is_medial_wall'] & df_merged['z'].notna()].copy()
    print(f"    {len(df_valid)} parcels for correlation analysis (excluding medial wall)")

    # Step 7: Compute correlations with spin tests
    print("\n[7/7] Computing spatial correlations with spin tests...")

    rho_vals = df_valid['rho_mean'].values
    tau_vals = df_valid['tau_mean'].values
    x_vals = df_valid['x'].values  # LR axis (not typically used)
    y_vals = df_valid['y'].values  # AP axis (anterior-posterior)
    z_vals = df_valid['z'].values  # DV axis (dorsal-ventral)

    # Hemisphere indices for spin test
    lh_mask = df_valid['hemi'] == 'lh'
    rh_mask = df_valid['hemi'] == 'rh'
    lh_indices = np.where(lh_mask)[0]
    rh_indices = np.where(rh_mask)[0]

    def run_spin_test(metric_vals, coord_vals, name):
        """Run correlation with spin permutation test."""
        r_obs, p_param = stats.pearsonr(metric_vals, coord_vals)
        print(f"    {name}: r = {r_obs:.4f}, p_param = {p_param:.4e}")

        null_r = np.zeros(N_PERM)
        np.random.seed(42)
        for i in range(N_PERM):
            perm_indices = np.zeros(len(df_valid), dtype=int)
            perm_indices[lh_indices] = np.random.permutation(lh_indices)
            perm_indices[rh_indices] = np.random.permutation(rh_indices)
            metric_perm = metric_vals[perm_indices]
            null_r[i] = stats.pearsonr(metric_perm, coord_vals)[0]

        p_spin = np.mean(np.abs(null_r) >= np.abs(r_obs))
        print(f"             p_spin = {p_spin:.4f}")
        return r_obs, p_param, p_spin

    # Run all correlations
    print(f"    Running spin permutation tests (N={N_PERM})...\n")

    results = []

    # rho vs DV (z)
    r, p_param, p_spin = run_spin_test(rho_vals, z_vals, "rho vs z (DV)")
    results.append({'var1': 'rho_mean', 'var2': 'z', 'r': r, 'p_param': p_param, 'p_spin': p_spin, 'n_parcels': len(df_valid)})

    # tau vs AP (y)
    r, p_param, p_spin = run_spin_test(tau_vals, y_vals, "tau vs y (AP)")
    results.append({'var1': 'tau_mean', 'var2': 'y', 'r': r, 'p_param': p_param, 'p_spin': p_spin, 'n_parcels': len(df_valid)})

    # tau vs DV (z)
    r, p_param, p_spin = run_spin_test(tau_vals, z_vals, "tau vs z (DV)")
    results.append({'var1': 'tau_mean', 'var2': 'z', 'r': r, 'p_param': p_param, 'p_spin': p_spin, 'n_parcels': len(df_valid)})

    # rho vs AP (y)
    r, p_param, p_spin = run_spin_test(rho_vals, y_vals, "rho vs y (AP)")
    results.append({'var1': 'rho_mean', 'var2': 'y', 'r': r, 'p_param': p_param, 'p_spin': p_spin, 'n_parcels': len(df_valid)})

    # tau vs rho
    r, p_param, p_spin = run_spin_test(tau_vals, rho_vals, "tau vs rho")
    results.append({'var1': 'tau_mean', 'var2': 'rho_mean', 'r': r, 'p_param': p_param, 'p_spin': p_spin, 'n_parcels': len(df_valid)})

    # Save correlation stats
    corr_stats = pd.DataFrame(results)
    corr_stats.to_csv(CORR_STATS_OUT, index=False)
    print(f"\n    Saved to {CORR_STATS_OUT}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Correlation':<20} {'r':>8} {'p_param':>12} {'p_spin':>10}")
    print("-" * 52)
    for res in results:
        label = f"{res['var1'].replace('_mean','')} vs {res['var2']}"
        print(f"{label:<20} {res['r']:>8.4f} {res['p_param']:>12.4e} {res['p_spin']:>10.4f}")
    print("=" * 60)
    print(f"n_parcels = {len(df_valid)}, N_perm = {N_PERM}")


if __name__ == '__main__':
    main()
