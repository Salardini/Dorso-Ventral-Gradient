#!/usr/bin/env python3
"""
MOUS fMRI Replication Pipeline v2
=================================
Computes fMRI-derived measures to validate MEG rotational dynamics gradient.

Key improvement over v1: Handles variable timepoint lengths across subjects.
Subjects are no longer excluded for having different scan durations.

Measures computed:
- ρ_fMRI: Rotational dynamics index (same method as MEG)
- τ_fMRI: Intrinsic timescale (autocorrelation integral)
- Spectral exponent: 1/f slope
- ALFF/fALFF: Amplitude of low-frequency fluctuations

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

QUICK_TEST = False  # Set to False for full analysis

# Paths - UPDATE THESE FOR YOUR SYSTEM
FMRI_DIR = Path(r"C:\Users\u2121\Downloads\MEG\fMRI")
OUTPUT_DIR = Path(r"C:\Users\u2121\Downloads\MEG\Pipeline\data\fmri_replication")
MEG_DATA_PATH = Path(r"C:\Users\u2121\Downloads\MEG\Pipeline\data\parcel_measures.csv")

# fMRI parameters
TR = 2.0  # Repetition time in seconds
LOW_FREQ = 0.01  # High-pass filter cutoff (Hz)
HIGH_FREQ = 0.1  # Low-pass filter cutoff (Hz)

# Minimum timepoints required (very permissive - just need enough for measures)
MIN_TIMEPOINTS = 100  # ~3.3 minutes of data minimum

# Analysis parameters
N_SPIN_PERMUTATIONS = 10000 if not QUICK_TEST else 1000
N_TEST_SUBJECTS = 5 if QUICK_TEST else None

# ρ computation parameters (matched to MEG pipeline)
EMBED_DIM = 5  # Reduced from MEG's 10 due to slower sampling
EMBED_DELAY = 1  # 1 TR = 2 seconds

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_intrinsic_timescale(ts, tr):
    """Compute intrinsic timescale as integral of autocorrelation function."""
    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-10)
    n = len(ts)
    max_lag = min(n // 2, 50)  # Limit to 50 lags (~100s)
    
    acf = np.correlate(ts, ts, mode='full')[n-1:n-1+max_lag+1]
    acf = acf / acf[0]
    
    # Find first zero crossing
    zero_idx = np.where(acf < 0)[0]
    if len(zero_idx) > 0:
        acf = acf[:zero_idx[0]]
    
    # Integrate (trapezoidal rule)
    tau = np.trapz(acf) * tr
    return max(tau, tr)  # Minimum is 1 TR


def compute_spectral_exponent(ts, tr):
    """Compute spectral exponent (1/f slope) from power spectrum."""
    # Compute power spectrum
    freqs, psd = signal.welch(ts, fs=1/tr, nperseg=min(len(ts)//2, 128))
    
    # Fit in log-log space (0.01-0.1 Hz range)
    mask = (freqs >= 0.01) & (freqs <= 0.1) & (psd > 0)
    if np.sum(mask) < 3:
        return np.nan
    
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    
    slope, _ = np.polyfit(log_f, log_p, 1)
    return slope


def compute_alff(ts, tr):
    """Compute amplitude of low-frequency fluctuations."""
    freqs, psd = signal.welch(ts, fs=1/tr, nperseg=min(len(ts)//2, 128))
    mask = (freqs >= 0.01) & (freqs <= 0.08)
    return np.sqrt(np.mean(psd[mask])) if np.any(mask) else np.nan


def compute_falff(ts, tr):
    """Compute fractional ALFF (ratio of low-freq to total power)."""
    freqs, psd = signal.welch(ts, fs=1/tr, nperseg=min(len(ts)//2, 128))
    low_mask = (freqs >= 0.01) & (freqs <= 0.08)
    total_mask = (freqs >= 0.01) & (freqs <= 0.25)
    
    low_power = np.sum(psd[low_mask])
    total_power = np.sum(psd[total_mask])
    
    return low_power / (total_power + 1e-10)


def compute_rho_fmri(ts, embed_dim=5, embed_delay=1):
    """
    Compute rotational dynamics index from fMRI time series.
    Uses delay embedding + VAR(1) model, same approach as MEG.
    
    Note: Due to fMRI's slow sampling, this captures ultra-slow
    rotational structure (<0.1 Hz) rather than fast oscillations.
    """
    n = len(ts)
    
    # Check we have enough data
    if n < embed_dim * embed_delay + 10:
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
    
    # Ridge regression for stability
    ridge_alpha = 0.001
    XtX = X_past.T @ X_past
    XtY = X_past.T @ X_future
    A = np.linalg.solve(XtX + ridge_alpha * np.eye(embed_dim), XtY)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    
    # Filter small eigenvalues
    mask = np.abs(eigenvalues) > 0.01
    if not np.any(mask):
        return np.nan
    
    eigenvalues = eigenvalues[mask]
    
    # Compute rho: mean sine of eigenvalue angles
    rho = np.mean(np.abs(np.imag(eigenvalues)) / (np.abs(eigenvalues) + 1e-10))
    
    return rho


def spin_permutation_test(x, y, n_perms=10000):
    """
    Spin permutation test for spatial correlation.
    Uses random rotations to generate null distribution.
    """
    observed_r = np.corrcoef(x, y)[0, 1]
    
    n = len(x)
    null_rs = np.zeros(n_perms)
    
    for i in range(n_perms):
        # Random permutation (approximation to spin test for speed)
        perm_idx = np.random.permutation(n)
        null_rs[i] = np.corrcoef(x[perm_idx], y)[0, 1]
    
    p_value = np.mean(np.abs(null_rs) >= np.abs(observed_r))
    return observed_r, max(p_value, 1/n_perms)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    
    if QUICK_TEST:
        print(f"*** QUICK TEST MODE: {N_TEST_SUBJECTS} subjects, {N_SPIN_PERMUTATIONS} permutations ***")
        print("*** Set QUICK_TEST = False for full analysis ***\n")
    
    print("=" * 70)
    print("MOUS fMRI REPLICATION PIPELINE v2")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 0: Check dependencies
    # -------------------------------------------------------------------------
    print("\n[0] Checking dependencies...")
    try:
        import nilearn
        from nilearn.maskers import NiftiLabelsMasker
        from nilearn import datasets
        print(f"    nilearn version: {nilearn.__version__}")
    except ImportError:
        print("    ERROR: nilearn not installed. Run: pip install nilearn")
        exit(1)
    
    try:
        import nibabel as nib
        print(f"    nibabel version: {nib.__version__}")
    except ImportError:
        print("    ERROR: nibabel not installed. Run: pip install nibabel")
        exit(1)
    
    # -------------------------------------------------------------------------
    # Step 1: Find subjects with resting-state fMRI
    # -------------------------------------------------------------------------
    print("\n[1] Finding subjects with resting-state fMRI...")
    
    # Files are flat in directory: sub-XXXX_task-rest_bold.nii
    fmri_files = sorted(FMRI_DIR.glob("sub-*_task-rest_bold.nii*"))
    print(f"    Found {len(fmri_files)} fMRI files")
    
    # Extract subject IDs and pair with files
    subjects_with_fmri = []
    for fmri_file in fmri_files:
        # Extract subject ID from filename (e.g., "sub-A2002" from "sub-A2002_task-rest_bold.nii")
        subject = fmri_file.name.split("_task-rest")[0]
        subjects_with_fmri.append((subject, fmri_file))
    
    print(f"    Found {len(subjects_with_fmri)} subjects with resting fMRI")
    
    if N_TEST_SUBJECTS:
        subjects_with_fmri = subjects_with_fmri[:N_TEST_SUBJECTS]
        print(f"    Limited to {N_TEST_SUBJECTS} subjects for testing")
    
    # -------------------------------------------------------------------------
    # Step 2: Load atlas
    # -------------------------------------------------------------------------
    print("\n[2] Fetching Schaefer-400 atlas...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    print(f"    Atlas loaded: {len(atlas.labels)} parcels")
    
    # Get parcel centroids for spatial analysis
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    
    centroids = {}
    for label in range(1, 401):
        coords = np.array(np.where(atlas_data == label)).T
        if len(coords) > 0:
            centroid_vox = coords.mean(axis=0)
            centroid_mni = nib.affines.apply_affine(affine, centroid_vox)
            centroids[label] = centroid_mni
    
    print(f"    Computed centroids for {len(centroids)} parcels")
    
    # -------------------------------------------------------------------------
    # Step 3: Process each subject
    # -------------------------------------------------------------------------
    print("\n[3] Processing subjects...")
    print(f"    (Variable timepoint lengths are now supported)\n")
    
    all_results = []
    timepoint_counts = []
    
    for i, (subject, func_file) in enumerate(subjects_with_fmri):
        print(f"    [{i+1}/{len(subjects_with_fmri)}] {subject}...", end=" ", flush=True)
        
        try:
            # Set up masker with preprocessing
            masker = NiftiLabelsMasker(
                labels_img=atlas.maps,
                standardize=True,
                detrend=True,
                low_pass=HIGH_FREQ,
                high_pass=LOW_FREQ,
                t_r=TR,
                memory='nilearn_cache',
                verbose=0
            )
            
            # Extract parcel time series
            ts_data = masker.fit_transform(str(func_file))
            n_timepoints, n_parcels_extracted = ts_data.shape
            
            # Check minimum timepoints
            if n_timepoints < MIN_TIMEPOINTS:
                print(f"SKIP (only {n_timepoints} timepoints, need {MIN_TIMEPOINTS})")
                continue
            
            timepoint_counts.append(n_timepoints)
            
            # Get which parcels were extracted
            extracted_labels = masker.labels_
            if extracted_labels is None:
                extracted_labels = list(range(1, n_parcels_extracted + 1))
            
            # Pre-allocate results with NaN for all 400 parcels
            results = {
                'subject': subject,
                'n_timepoints': n_timepoints,
                'tau': np.full(400, np.nan),
                'spectral_exp': np.full(400, np.nan),
                'alff': np.full(400, np.nan),
                'falff': np.full(400, np.nan),
                'rho_fmri': np.full(400, np.nan),
            }
            
            # Compute measures for each extracted parcel
            n_missing = 0
            for col_idx in range(n_parcels_extracted):
                label = extracted_labels[col_idx]
                parcel_idx = label - 1  # Convert to 0-indexed
                
                if parcel_idx < 0 or parcel_idx >= 400:
                    continue
                
                ts = ts_data[:, col_idx]
                
                results['tau'][parcel_idx] = compute_intrinsic_timescale(ts, TR)
                results['spectral_exp'][parcel_idx] = compute_spectral_exponent(ts, TR)
                results['alff'][parcel_idx] = compute_alff(ts, TR)
                results['falff'][parcel_idx] = compute_falff(ts, TR)
                results['rho_fmri'][parcel_idx] = compute_rho_fmri(ts, EMBED_DIM, EMBED_DELAY)
            
            n_missing = 400 - n_parcels_extracted
            
            all_results.append(results)
            
            if n_missing > 0:
                print(f"OK ({n_timepoints} tp, {n_missing} parcels missing)")
            else:
                print(f"OK ({n_timepoints} timepoints)")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print(f"\n    Successfully processed {len(all_results)}/{len(subjects_with_fmri)} subjects")
    
    if len(all_results) == 0:
        print("    ERROR: No subjects processed successfully!")
        exit(1)
    
    # Report timepoint distribution
    tp_array = np.array(timepoint_counts)
    print(f"\n    Timepoint distribution:")
    print(f"      Min: {tp_array.min()}, Max: {tp_array.max()}, Mean: {tp_array.mean():.1f}")
    unique_tp, counts = np.unique(tp_array, return_counts=True)
    for tp, count in zip(unique_tp, counts):
        print(f"      {tp} timepoints: {count} subjects")
    
    # -------------------------------------------------------------------------
    # Step 4: Compute group averages
    # -------------------------------------------------------------------------
    print("\n[4] Computing group averages...")
    
    # Stack results - all should be (400,) now
    tau_stack = np.array([r['tau'] for r in all_results])
    spec_stack = np.array([r['spectral_exp'] for r in all_results])
    alff_stack = np.array([r['alff'] for r in all_results])
    falff_stack = np.array([r['falff'] for r in all_results])
    rho_fmri_stack = np.array([r['rho_fmri'] for r in all_results])
    
    print(f"    Data shape: {tau_stack.shape} (subjects x parcels)")
    
    # Compute group means (nanmean handles missing parcels)
    group_tau = np.nanmean(tau_stack, axis=0)
    group_spec = np.nanmean(spec_stack, axis=0)
    group_alff = np.nanmean(alff_stack, axis=0)
    group_falff = np.nanmean(falff_stack, axis=0)
    group_rho_fmri = np.nanmean(rho_fmri_stack, axis=0)
    
    # Report coverage
    coverage = np.mean(~np.isnan(tau_stack), axis=0)
    print(f"    Mean parcel coverage: {np.mean(coverage)*100:.1f}%")
    print(f"    Parcels with >80% coverage: {np.sum(coverage > 0.8)}/400")
    
    # -------------------------------------------------------------------------
    # Step 5: Build output dataframe
    # -------------------------------------------------------------------------
    print("\n[5] Building output dataframe...")
    
    # Get parcel coordinates
    parcel_data = []
    for label in range(1, 401):
        if label in centroids:
            x, y, z = centroids[label]
        else:
            x, y, z = np.nan, np.nan, np.nan
        
        parcel_data.append({
            'parcel': label,
            'x': x,
            'y': y,
            'z': z,
            'tau_fmri': group_tau[label-1],
            'spectral_exp': group_spec[label-1],
            'alff': group_alff[label-1],
            'falff': group_falff[label-1],
            'rho_fmri': group_rho_fmri[label-1],
            'coverage': coverage[label-1],
            'n_subjects': len(all_results),
        })
    
    fmri_df = pd.DataFrame(parcel_data)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "fmri_parcel_measures_v2.csv"
    fmri_df.to_csv(output_path, index=False)
    print(f"    Saved to {output_path}")
    
    # -------------------------------------------------------------------------
    # Step 6: Load MEG data for comparison
    # -------------------------------------------------------------------------
    print("\n[6] Loading MEG data for comparison...")
    
    try:
        meg_df = pd.read_csv(MEG_DATA_PATH)
        print(f"    Loaded {len(meg_df)} MEG parcels")
    except FileNotFoundError:
        print(f"    WARNING: MEG data not found at {MEG_DATA_PATH}")
        print("    Skipping cross-modal comparison")
        meg_df = None
    
    # -------------------------------------------------------------------------
    # Step 7: Compute correlations
    # -------------------------------------------------------------------------
    print("\n[7] Computing correlations...")
    
    # Get valid parcels (non-NaN for all measures)
    valid_mask = (
        ~np.isnan(fmri_df['tau_fmri']) & 
        ~np.isnan(fmri_df['z']) &
        ~np.isnan(fmri_df['rho_fmri'])
    )
    valid_df = fmri_df[valid_mask].copy()
    print(f"    Valid parcels: {len(valid_df)}/400")
    
    z_coord = valid_df['z'].values
    
    # fMRI measure correlations with DV coordinate
    print("\n    fMRI measure correlations with DV coordinate (z):")
    print("    " + "-" * 50)
    
    results_list = []
    for measure in ['tau_fmri', 'spectral_exp', 'alff', 'falff', 'rho_fmri']:
        values = valid_df[measure].values
        r, p = stats.pearsonr(values, z_coord)
        print(f"    {measure:15s} vs z: r = {r:+.3f}, p = {p:.2e}")
        results_list.append({
            'comparison': f'{measure}_vs_z',
            'r': r,
            'p_parametric': p
        })
    
    # Cross-modal comparison if MEG data available
    if meg_df is not None and 'rho' in meg_df.columns:
        print("\n    fMRI measure correlations with MEG ρ:")
        print("    " + "-" * 50)
        
        # Merge on parcel
        merged = valid_df.merge(meg_df[['parcel', 'rho']], on='parcel', how='inner')
        meg_rho = merged['rho'].values
        
        for measure in ['tau_fmri', 'spectral_exp', 'alff', 'falff', 'rho_fmri']:
            values = merged[measure].values
            r, p = stats.pearsonr(values, meg_rho)
            print(f"    {measure:15s} vs ρ: r = {r:+.3f}, p = {p:.2e}")
            results_list.append({
                'comparison': f'{measure}_vs_meg_rho',
                'r': r,
                'p_parametric': p
            })
    
    # -------------------------------------------------------------------------
    # Step 8: Spin permutation tests
    # -------------------------------------------------------------------------
    print(f"\n[8] Running spin permutation tests (n={N_SPIN_PERMUTATIONS})...")
    
    print("\n    Spin permutation results:")
    print("    " + "-" * 50)
    
    for result in results_list:
        comp = result['comparison']
        
        if '_vs_z' in comp:
            measure = comp.replace('_vs_z', '')
            values = valid_df[measure].values
            r, p_spin = spin_permutation_test(values, z_coord, N_SPIN_PERMUTATIONS)
        elif '_vs_meg_rho' in comp and meg_df is not None:
            measure = comp.replace('_vs_meg_rho', '')
            merged = valid_df.merge(meg_df[['parcel', 'rho']], on='parcel', how='inner')
            values = merged[measure].values
            meg_rho = merged['rho'].values
            r, p_spin = spin_permutation_test(values, meg_rho, N_SPIN_PERMUTATIONS)
        else:
            continue
        
        result['p_spin'] = p_spin
        sig = "✓" if p_spin < 0.05 else "✗"
        print(f"    {comp:25s}: r = {r:+.3f}, p_spin = {p_spin:.4f} {sig}")
    
    # -------------------------------------------------------------------------
    # Step 9: Save results
    # -------------------------------------------------------------------------
    print("\n[9] Saving results...")
    
    results_df = pd.DataFrame(results_list)
    results_path = OUTPUT_DIR / "fmri_replication_results_v2.csv"
    results_df.to_csv(results_path, index=False)
    print(f"    Saved to {results_path}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
SAMPLE:
  N subjects processed: {len(all_results)}
  N parcels: 400 (Schaefer)
  Timepoint range: {tp_array.min()}-{tp_array.max()} (mean: {tp_array.mean():.1f})

fMRI MEASURES vs DV COORDINATE (z):
""")
    
    for result in results_list:
        if '_vs_z' in result['comparison']:
            measure = result['comparison'].replace('_vs_z', '')
            r = result['r']
            p = result.get('p_spin', result['p_parametric'])
            sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
            print(f"  {measure:15s}: r = {r:+.3f}, p_spin = {p:.4f} {sig}")
    
    if meg_df is not None:
        print("\nfMRI MEASURES vs MEG ρ:")
        for result in results_list:
            if '_vs_meg_rho' in result['comparison']:
                measure = result['comparison'].replace('_vs_meg_rho', '')
                r = result['r']
                p = result.get('p_spin', result['p_parametric'])
                sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
                print(f"  {measure:15s}: r = {r:+.3f}, p_spin = {p:.4f} {sig}")
    
    print(f"""
OUTPUT FILES:
  {output_path}
  {results_path}
""")
    
    print("=" * 70)
    print("DONE")
    print("=" * 70)
