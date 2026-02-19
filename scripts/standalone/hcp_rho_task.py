#!/usr/bin/env python3
"""
HCP fMRI Replication - TASK DATA (broader bandpass filter)
==========================================================

Uses TASK timeseries which have broader filtering (0.009-0.25 Hz)
compared to resting state (0.009-0.08 Hz).

The broader filter may preserve more rotational dynamics signal.
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

HCP_DIR = Path(r"C:\Users\u2121\Downloads\MEG\fMRI2\timeseries_400")
ATLAS_PATH = Path(r"C:\Users\u2121\Downloads\MEG\Pipeline\code\atlas\schaefer400_centroids.csv")
OUTPUT_DIR = Path(r"C:\Users\u2121\Downloads\MEG\fMRI2\results")

TR = 0.72
EMBED_DIM = 5
EMBED_DELAY = 2
N_SPIN_PERMUTATIONS = 10000
N_TEST_SUBJECTS = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_hcp_timeseries(subject_dir, run_pattern="tfMRI"):
    """Load and concatenate HCP timeseries matching pattern."""
    files = sorted(subject_dir.glob(f"*{run_pattern}*_400"))
    
    if len(files) == 0:
        return None, 0
    
    all_ts = []
    for f in files:
        ts = np.loadtxt(f)
        if ts.ndim == 1:
            ts = ts.reshape(1, -1)
        all_ts.append(ts)
    
    ts_concat = np.vstack(all_ts)
    return ts_concat, len(files)


def compute_rho(ts, embed_dim=5, embed_delay=2):
    """Compute rotational dynamics index via delay embedding + VAR(1)."""
    n = len(ts)
    min_length = embed_dim * embed_delay + 20
    if n < min_length:
        return np.nan
    
    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-10)
    
    n_embedded = n - (embed_dim - 1) * embed_delay
    X = np.zeros((n_embedded, embed_dim))
    for d in range(embed_dim):
        X[:, d] = ts[d * embed_delay : d * embed_delay + n_embedded]
    
    X_past = X[:-1]
    X_future = X[1:]
    
    ridge_alpha = 0.001
    XtX = X_past.T @ X_past
    XtY = X_past.T @ X_future
    A = np.linalg.solve(XtX + ridge_alpha * np.eye(embed_dim), XtY)
    
    eigenvalues = np.linalg.eigvals(A)
    mask = np.abs(eigenvalues) > 0.01
    if not np.any(mask):
        return np.nan
    
    eigenvalues = eigenvalues[mask]
    rho = np.mean(np.abs(np.imag(eigenvalues)) / (np.abs(eigenvalues) + 1e-10))
    return rho


def spin_permutation_test(x, y, hemi, n_perm=10000):
    """Spin permutation test preserving hemispheric structure."""
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("HCP fMRI TASK DATA: Dorsoventral Rotational Dynamics")
    print("(Broader bandpass: 0.009-0.25 Hz vs rest 0.009-0.08 Hz)")
    print("=" * 70)
    
    # Load atlas
    print("\n[1] Loading atlas...")
    atlas_df = pd.read_csv(ATLAS_PATH, comment='#').head(400)
    z_coord = atlas_df['z'].values
    y_coord = atlas_df['y'].values
    hemi = atlas_df['hemi'].values
    parcel_labels = atlas_df['label'].values
    print(f"    {len(atlas_df)} parcels")
    
    # Find subjects
    print("\n[2] Finding subjects...")
    subject_dirs = sorted([d for d in HCP_DIR.iterdir() if d.is_dir()])
    print(f"    Found {len(subject_dirs)} subjects")
    
    if N_TEST_SUBJECTS:
        subject_dirs = subject_dirs[:N_TEST_SUBJECTS]
        print(f"    Limited to {N_TEST_SUBJECTS}")
    
    # Process
    print("\n[3] Computing ρ from TASK data...")
    all_rho = []
    subject_ids = []
    timepoint_counts = []
    
    for i, subj_dir in enumerate(subject_dirs):
        subj_id = subj_dir.name
        
        if (i + 1) % 100 == 0 or i == 0:
            print(f"    Processing {i+1}/{len(subject_dirs)}...")
        
        # Load ALL task runs (tfMRI_*)
        ts_data, n_runs = load_hcp_timeseries(subj_dir, "tfMRI")
        
        if ts_data is None or n_runs == 0:
            continue
        
        n_timepoints, n_parcels = ts_data.shape
        
        if n_parcels == 414:
            ts_data = ts_data[:, :400]
        elif n_parcels != 400:
            continue
        
        # Compute rho
        rho_parcel = np.zeros(400)
        for p in range(400):
            rho_parcel[p] = compute_rho(ts_data[:, p], EMBED_DIM, EMBED_DELAY)
        
        n_valid = np.sum(~np.isnan(rho_parcel))
        if n_valid < 350:
            continue
        
        all_rho.append(rho_parcel)
        subject_ids.append(subj_id)
        timepoint_counts.append(n_timepoints)
    
    print(f"\n    Processed {len(all_rho)} subjects")
    
    tp_array = np.array(timepoint_counts)
    print(f"    Timepoints: min={tp_array.min()}, max={tp_array.max()}, mean={tp_array.mean():.0f}")
    
    # Group average
    print("\n[4] Group average...")
    rho_stack = np.array(all_rho)
    rho_group = np.nanmean(rho_stack, axis=0)
    rho_std = np.nanstd(rho_stack, axis=0)
    
    print(f"    ρ range: {np.nanmin(rho_group):.3f} to {np.nanmax(rho_group):.3f}")
    print(f"    ρ mean ± std: {np.nanmean(rho_group):.3f} ± {np.nanstd(rho_group):.3f}")
    
    # Correlations
    print("\n[5] Correlating with DV coordinate...")
    valid_mask = ~np.isnan(rho_group)
    rho_valid = rho_group[valid_mask]
    z_valid = z_coord[valid_mask]
    hemi_valid = hemi[valid_mask]
    
    r_param, p_param = stats.pearsonr(rho_valid, z_valid)
    print(f"    Parametric: r = {r_param:.4f}, p = {p_param:.2e}")
    
    print(f"    Running spin test (n={N_SPIN_PERMUTATIONS})...")
    r_spin, p_spin = spin_permutation_test(rho_valid, z_valid, hemi_valid, N_SPIN_PERMUTATIONS)
    print(f"    Spin test:  r = {r_spin:.4f}, p_spin = {p_spin:.4f}")
    
    # Additional
    y_valid = y_coord[valid_mask]
    r_ap, p_ap = stats.pearsonr(rho_valid, y_valid)
    print(f"    ρ vs AP: r = {r_ap:.4f}")
    
    n_half = len(all_rho) // 2
    rho_half1 = np.nanmean(rho_stack[:n_half], axis=0)
    rho_half2 = np.nanmean(rho_stack[n_half:], axis=0)
    valid_both = ~np.isnan(rho_half1) & ~np.isnan(rho_half2)
    r_reliability = np.corrcoef(rho_half1[valid_both], rho_half2[valid_both])[0, 1]
    print(f"    Split-half reliability: r = {r_reliability:.4f}")
    
    # Save
    print("\n[6] Saving...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    parcel_df = pd.DataFrame({
        'parcel_idx': np.arange(400),
        'label': parcel_labels,
        'hemi': hemi,
        'x': atlas_df['x'].values,
        'y': atlas_df['y'].values,
        'z': atlas_df['z'].values,
        'rho_mean': rho_group,
        'rho_std': rho_std,
        'n_subjects': len(all_rho),
    })
    parcel_df.to_csv(OUTPUT_DIR / "hcp_task_parcel_rho.csv", index=False)
    
    summary = {
        'data_type': 'task',
        'bandpass': '0.009-0.25 Hz',
        'n_subjects': len(all_rho),
        'rho_vs_dv_r': r_param,
        'rho_vs_dv_p_spin': p_spin,
        'rho_vs_ap_r': r_ap,
        'rho_range': f"{np.nanmin(rho_group):.3f}-{np.nanmax(rho_group):.3f}",
        'split_half_r': r_reliability,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "hcp_task_summary.csv", index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: HCP TASK fMRI")
    print("=" * 70)
    sig = "✓ SIGNIFICANT" if p_spin < 0.05 else "✗ not significant"
    print(f"""
SAMPLE: N={len(all_rho)}, mean {tp_array.mean():.0f} timepoints

KEY RESULT:
  ρ vs DV: r = {r_param:.4f}, p_spin = {p_spin:.4f} {sig}
  ρ range: {np.nanmin(rho_group):.3f} to {np.nanmax(rho_group):.3f}

COMPARISON:
  REST (0.009-0.08 Hz): r = 0.0003, range 0.136-0.157
  TASK (0.009-0.25 Hz): r = {r_param:.4f}, range {np.nanmin(rho_group):.3f}-{np.nanmax(rho_group):.3f}
""")
    print("=" * 70)
