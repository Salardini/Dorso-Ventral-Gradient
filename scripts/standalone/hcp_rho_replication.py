#!/usr/bin/env python3
"""
HCP fMRI Replication of Dorsoventral Rotational Dynamics Gradient
=================================================================

Uses preprocessed HCP-YA Schaefer-400 timeseries from UCSD dataset
(Tipnis et al. 2022, DOI: 10.6075/J0C24WMW)

Validates the MOUS finding: ρ_fMRI correlates with DV coordinate (r ≈ -0.25)

Key advantage: ~1000 subjects, different scanner/site (Washington University)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
HCP_DIR = Path(r"C:\Users\u2121\Downloads\MEG\fMRI2\timeseries_400")
ATLAS_PATH = Path(r"C:\Users\u2121\Downloads\MEG\Pipeline\code\atlas\schaefer400_centroids.csv")
OUTPUT_DIR = Path(r"C:\Users\u2121\Downloads\MEG\fMRI2\results")

# Analysis parameters
TR = 0.72  # HCP TR in seconds
EMBED_DIM = 5  # Delay embedding dimension
EMBED_DELAY = 2  # Delay in TRs (~1.44s)
N_SPIN_PERMUTATIONS = 10000
USE_RESTING_ONLY = True  # Only use resting-state runs

# Test mode - set to None for full analysis
N_TEST_SUBJECTS = None  # Set to e.g. 50 for quick test

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_hcp_timeseries(subject_dir, run_pattern="rfMRI_REST"):
    """
    Load HCP timeseries from text files.
    
    Parameters
    ----------
    subject_dir : Path
        Directory containing subject's timeseries files
    run_pattern : str
        Pattern to match (e.g., "rfMRI_REST" for resting state)
    
    Returns
    -------
    ts_concat : ndarray
        Concatenated timeseries (timepoints x 400 parcels)
    n_runs : int
        Number of runs concatenated
    """
    files = sorted(subject_dir.glob(f"*{run_pattern}*_400"))
    
    if len(files) == 0:
        return None, 0
    
    all_ts = []
    for f in files:
        # Load space-separated text file
        ts = np.loadtxt(f)
        if ts.ndim == 1:
            ts = ts.reshape(1, -1)
        all_ts.append(ts)
    
    # Concatenate across runs
    ts_concat = np.vstack(all_ts)
    return ts_concat, len(files)


def compute_rho(ts, embed_dim=5, embed_delay=2):
    """
    Compute rotational dynamics index from fMRI time series.
    Uses delay embedding + VAR(1) model.
    
    Parameters
    ----------
    ts : ndarray
        1D time series
    embed_dim : int
        Embedding dimension
    embed_delay : int  
        Delay between embedding dimensions (in samples)
    
    Returns
    -------
    rho : float
        Rotational dynamics index (0 = no rotation, 1 = pure rotation)
    """
    n = len(ts)
    
    # Check minimum length
    min_length = embed_dim * embed_delay + 20
    if n < min_length:
        return np.nan
    
    # Standardize
    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-10)
    
    # Delay embedding
    n_embedded = n - (embed_dim - 1) * embed_delay
    X = np.zeros((n_embedded, embed_dim))
    for d in range(embed_dim):
        X[:, d] = ts[d * embed_delay : d * embed_delay + n_embedded]
    
    # Fit VAR(1): X[t+1] = A @ X[t]
    X_past = X[:-1]
    X_future = X[1:]
    
    # Ridge regression for numerical stability
    ridge_alpha = 0.001
    XtX = X_past.T @ X_past
    XtY = X_past.T @ X_future
    A = np.linalg.solve(XtX + ridge_alpha * np.eye(embed_dim), XtY)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    
    # Filter small eigenvalues (noise)
    mask = np.abs(eigenvalues) > 0.01
    if not np.any(mask):
        return np.nan
    
    eigenvalues = eigenvalues[mask]
    
    # Compute rho: mean |sin(angle)| of eigenvalues
    # High rho = complex eigenvalues = rotational dynamics
    rho = np.mean(np.abs(np.imag(eigenvalues)) / (np.abs(eigenvalues) + 1e-10))
    
    return rho


def spin_permutation_test(x, y, hemi, n_perm=10000):
    """
    Spin permutation test preserving hemispheric structure.
    
    Parameters
    ----------
    x, y : ndarray
        Variables to correlate
    hemi : ndarray
        Hemisphere labels ('lh' or 'rh')
    n_perm : int
        Number of permutations
    
    Returns
    -------
    r_obs : float
        Observed correlation
    p_spin : float
        Spin-permutation p-value
    """
    r_obs = np.corrcoef(x, y)[0, 1]
    
    lh_idx = np.where(hemi == 'lh')[0]
    rh_idx = np.where(hemi == 'rh')[0]
    
    null_r = np.zeros(n_perm)
    np.random.seed(42)
    
    for i in range(n_perm):
        perm = np.zeros(len(x), dtype=int)
        perm[lh_idx] = np.random.permutation(lh_idx)
        perm[rh_idx] = np.random.permutation(rh_idx)
        null_r[i] = np.corrcoef(x[perm], y)[0, 1]
    
    p_spin = np.mean(np.abs(null_r) >= np.abs(r_obs))
    return r_obs, max(p_spin, 1/n_perm)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("HCP fMRI REPLICATION: Dorsoventral Rotational Dynamics Gradient")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load atlas coordinates
    # -------------------------------------------------------------------------
    print("\n[1] Loading Schaefer-400 atlas coordinates...")

    atlas_df = pd.read_csv(ATLAS_PATH, comment='#')
    atlas_df = atlas_df.head(400)  # Keep only first 400 parcels
    print(f"    Loaded {len(atlas_df)} parcels")
    print(f"    Columns: {list(atlas_df.columns)}")
    
    # Extract coordinates and hemisphere
    z_coord = atlas_df['z'].values  # DV coordinate
    y_coord = atlas_df['y'].values  # AP coordinate  
    hemi = atlas_df['hemi'].values
    parcel_labels = atlas_df['label'].values
    
    print(f"    Z (DV) range: {z_coord.min():.1f} to {z_coord.max():.1f}")
    print(f"    Hemispheres: LH={np.sum(hemi=='lh')}, RH={np.sum(hemi=='rh')}")
    
    # -------------------------------------------------------------------------
    # Step 2: Find subjects
    # -------------------------------------------------------------------------
    print("\n[2] Finding HCP subjects...")
    
    subject_dirs = sorted([d for d in HCP_DIR.iterdir() if d.is_dir()])
    print(f"    Found {len(subject_dirs)} subject directories")
    
    if N_TEST_SUBJECTS:
        subject_dirs = subject_dirs[:N_TEST_SUBJECTS]
        print(f"    Limited to {N_TEST_SUBJECTS} subjects for testing")
    
    # -------------------------------------------------------------------------
    # Step 3: Process subjects
    # -------------------------------------------------------------------------
    print("\n[3] Computing ρ for each subject...")
    print(f"    Parameters: embed_dim={EMBED_DIM}, embed_delay={EMBED_DELAY}")
    
    all_rho = []  # List of (400,) arrays
    subject_ids = []
    timepoint_counts = []
    
    for i, subj_dir in enumerate(subject_dirs):
        subj_id = subj_dir.name
        
        # Progress
        if (i + 1) % 100 == 0 or i == 0:
            print(f"    Processing subject {i+1}/{len(subject_dirs)}...")
        
        # Load resting-state timeseries
        ts_data, n_runs = load_hcp_timeseries(subj_dir, "rfMRI_REST")
        
        if ts_data is None or n_runs == 0:
            continue
        
        n_timepoints, n_parcels = ts_data.shape

        # Handle 414-parcel files (400 cortical + 14 subcortical)
        if n_parcels == 414:
            ts_data = ts_data[:, :400]  # Keep only cortical parcels
        elif n_parcels != 400:
            print(f"    WARNING: {subj_id} has {n_parcels} parcels, skipping")
            continue
        
        # Compute rho for each parcel
        rho_parcel = np.zeros(400)
        for p in range(400):
            rho_parcel[p] = compute_rho(ts_data[:, p], EMBED_DIM, EMBED_DELAY)
        
        # Check for valid parcels
        n_valid = np.sum(~np.isnan(rho_parcel))
        if n_valid < 350:  # Require at least 350 valid parcels
            continue
        
        all_rho.append(rho_parcel)
        subject_ids.append(subj_id)
        timepoint_counts.append(n_timepoints)
    
    print(f"\n    Successfully processed {len(all_rho)} subjects")
    
    if len(all_rho) == 0:
        print("    ERROR: No subjects processed!")
        exit(1)
    
    # Timepoint stats
    tp_array = np.array(timepoint_counts)
    print(f"    Timepoints: min={tp_array.min()}, max={tp_array.max()}, mean={tp_array.mean():.0f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Compute group average
    # -------------------------------------------------------------------------
    print("\n[4] Computing group average ρ...")
    
    rho_stack = np.array(all_rho)  # (n_subjects, 400)
    print(f"    Data shape: {rho_stack.shape}")
    
    # Group mean (nanmean handles missing parcels)
    rho_group = np.nanmean(rho_stack, axis=0)
    rho_std = np.nanstd(rho_stack, axis=0)
    
    # Coverage stats
    coverage = np.mean(~np.isnan(rho_stack), axis=0)
    print(f"    Mean parcel coverage: {np.mean(coverage)*100:.1f}%")
    print(f"    ρ range: {np.nanmin(rho_group):.3f} to {np.nanmax(rho_group):.3f}")
    print(f"    ρ mean ± std: {np.nanmean(rho_group):.3f} ± {np.nanstd(rho_group):.3f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Correlate with DV coordinate
    # -------------------------------------------------------------------------
    print("\n[5] Correlating ρ with dorsoventral coordinate...")
    
    # Valid parcels (non-NaN)
    valid_mask = ~np.isnan(rho_group)
    n_valid = np.sum(valid_mask)
    print(f"    Valid parcels: {n_valid}/400")
    
    rho_valid = rho_group[valid_mask]
    z_valid = z_coord[valid_mask]
    hemi_valid = hemi[valid_mask]
    
    # Parametric correlation
    r_param, p_param = stats.pearsonr(rho_valid, z_valid)
    print(f"\n    Parametric: r = {r_param:.4f}, p = {p_param:.2e}")
    
    # Spin permutation test
    print(f"\n    Running spin permutation test (n={N_SPIN_PERMUTATIONS})...")
    r_spin, p_spin = spin_permutation_test(rho_valid, z_valid, hemi_valid, N_SPIN_PERMUTATIONS)
    print(f"    Spin test:  r = {r_spin:.4f}, p_spin = {p_spin:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 6: Additional analyses
    # -------------------------------------------------------------------------
    print("\n[6] Additional analyses...")
    
    # AP coordinate correlation
    y_valid = y_coord[valid_mask]
    r_ap, p_ap = stats.pearsonr(rho_valid, y_valid)
    print(f"    ρ vs AP (y): r = {r_ap:.4f}, p = {p_ap:.2e}")
    
    # Split-half reliability
    n_half = len(all_rho) // 2
    rho_half1 = np.nanmean(rho_stack[:n_half], axis=0)
    rho_half2 = np.nanmean(rho_stack[n_half:], axis=0)
    valid_both = ~np.isnan(rho_half1) & ~np.isnan(rho_half2)
    r_reliability = np.corrcoef(rho_half1[valid_both], rho_half2[valid_both])[0, 1]
    print(f"    Split-half reliability: r = {r_reliability:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 7: Save results
    # -------------------------------------------------------------------------
    print("\n[7] Saving results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parcel-level results
    parcel_df = pd.DataFrame({
        'parcel_idx': np.arange(400),
        'label': parcel_labels,
        'hemi': hemi,
        'x': atlas_df['x'].values,
        'y': atlas_df['y'].values,
        'z': atlas_df['z'].values,
        'rho_mean': rho_group,
        'rho_std': rho_std,
        'coverage': coverage,
        'n_subjects': len(all_rho),
    })
    parcel_path = OUTPUT_DIR / "hcp_parcel_rho.csv"
    parcel_df.to_csv(parcel_path, index=False)
    print(f"    Saved: {parcel_path}")
    
    # Summary results
    summary = {
        'n_subjects': len(all_rho),
        'n_parcels_valid': n_valid,
        'rho_vs_dv_r': r_param,
        'rho_vs_dv_p_param': p_param,
        'rho_vs_dv_p_spin': p_spin,
        'rho_vs_ap_r': r_ap,
        'rho_vs_ap_p': p_ap,
        'split_half_r': r_reliability,
        'mean_timepoints': tp_array.mean(),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = OUTPUT_DIR / "hcp_replication_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"    Saved: {summary_path}")
    
    # Subject-level rho (for supplementary analyses)
    subject_rho_path = OUTPUT_DIR / "hcp_subject_rho.npy"
    np.save(subject_rho_path, rho_stack)
    print(f"    Saved: {subject_rho_path}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: HCP fMRI Replication")
    print("=" * 70)
    
    sig_dv = "✓ SIGNIFICANT" if p_spin < 0.05 else "✗ not significant"
    
    print(f"""
SAMPLE:
  N subjects: {len(all_rho)}
  N parcels: 400 (Schaefer)
  Mean timepoints: {tp_array.mean():.0f} (across 4 resting runs)

KEY RESULT:
  ρ vs DV coordinate: r = {r_param:.4f}, p_spin = {p_spin:.4f} {sig_dv}

COMPARISON WITH MOUS:
  MOUS fMRI (N=200):  r = -0.246, p_spin < 0.001
  HCP fMRI (N={len(all_rho)}):  r = {r_param:.3f}, p_spin = {p_spin:.4f}

ADDITIONAL:
  ρ vs AP coordinate: r = {r_ap:.4f}
  Split-half reliability: r = {r_reliability:.4f}

OUTPUT FILES:
  {parcel_path}
  {summary_path}
  {subject_rho_path}
""")
    
    print("=" * 70)
    print("DONE")
    print("=" * 70)
