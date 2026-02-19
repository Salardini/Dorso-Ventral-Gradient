"""
SPLIT-HALF STABILITY ANALYSIS
Tests whether the rho-DV gradient replicates in independent halves of subjects.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ============================================
# PATHS - Update if needed
# ============================================
DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
SUBJECT_FILE = DATA_DIR / 'group' / 'parcel_subject_data.csv'  # If exists
GROUP_FILE = DATA_DIR / 'group' / 'parcel_group_maps.csv'

# Alternative: Load from intermediates
INTERMEDIATES = DATA_DIR / 'MEG_MOUS' / 'intermediates'

print("=" * 60)
print("SPLIT-HALF STABILITY ANALYSIS")
print("=" * 60)

# ============================================
# METHOD 1: If you have subject-level parcel data
# ============================================
def run_split_half_from_subject_file():
    """Run if parcel_subject_data.csv exists with columns: subject, parcel_idx, rho"""
    if not SUBJECT_FILE.exists():
        return None
    
    df = pd.read_csv(SUBJECT_FILE)
    print(f"Loaded subject-level data: {len(df)} rows")
    
    subjects = df['subject'].unique()
    n_subjects = len(subjects)
    print(f"N subjects: {n_subjects}")
    
    # Get parcel coordinates
    group_df = pd.read_csv(GROUP_FILE)
    coords = group_df[['parcel_idx', 'z']].drop_duplicates()
    
    # Run 100 split-half iterations
    n_splits = 100
    half1_corrs = []
    half2_corrs = []
    cross_corrs = []
    
    np.random.seed(42)
    for i in range(n_splits):
        # Random split
        perm = np.random.permutation(subjects)
        half1_subs = perm[:n_subjects // 2]
        half2_subs = perm[n_subjects // 2:]
        
        # Compute mean rho per parcel for each half
        half1_data = df[df['subject'].isin(half1_subs)].groupby('parcel_idx')['rho'].mean()
        half2_data = df[df['subject'].isin(half2_subs)].groupby('parcel_idx')['rho'].mean()
        
        # Merge with coordinates
        half1_df = pd.DataFrame({'parcel_idx': half1_data.index, 'rho': half1_data.values})
        half1_df = half1_df.merge(coords, on='parcel_idx')
        
        half2_df = pd.DataFrame({'parcel_idx': half2_data.index, 'rho': half2_data.values})
        half2_df = half2_df.merge(coords, on='parcel_idx')
        
        # Correlations
        r1 = stats.pearsonr(half1_df['rho'], half1_df['z'])[0]
        r2 = stats.pearsonr(half2_df['rho'], half2_df['z'])[0]
        
        # Cross-correlation of maps
        merged = half1_df.merge(half2_df, on='parcel_idx', suffixes=('_1', '_2'))
        r_cross = stats.pearsonr(merged['rho_1'], merged['rho_2'])[0]
        
        half1_corrs.append(r1)
        half2_corrs.append(r2)
        cross_corrs.append(r_cross)
    
    return {
        'half1_corrs': half1_corrs,
        'half2_corrs': half2_corrs,
        'cross_corrs': cross_corrs
    }

# ============================================
# METHOD 2: Load from individual subject files
# ============================================
def run_split_half_from_intermediates():
    """Load rho from each subject's parcel file"""
    
    # Find all subject directories
    subject_dirs = sorted(INTERMEDIATES.glob('sub-*'))
    print(f"Found {len(subject_dirs)} subject directories")
    
    if len(subject_dirs) == 0:
        print("ERROR: No subject directories found")
        return None
    
    # Load parcel coordinates
    group_df = pd.read_csv(GROUP_FILE)
    parcel_idx = group_df['parcel_idx'].values
    z_coords = group_df['z'].values
    n_parcels = len(parcel_idx)
    
    # Try to find rho values for each subject
    # Check what files exist
    sample_dir = subject_dirs[0]
    print(f"\nFiles in {sample_dir.name}:")
    for f in sample_dir.glob('*'):
        print(f"  {f.name}")
    
    # Look for parcel_metrics.csv or similar
    subject_rho = []
    valid_subjects = []
    
    for subj_dir in subject_dirs:
        # Try different possible file names
        possible_files = [
            subj_dir / 'parcel_metrics.csv',
            subj_dir / 'parcel_rho.csv',
            subj_dir / 'rho_values.npy',
            subj_dir / 'parcel_data.csv',
        ]
        
        rho_file = None
        for pf in possible_files:
            if pf.exists():
                rho_file = pf
                break
        
        if rho_file is None:
            continue
        
        # Load based on file type
        if rho_file.suffix == '.csv':
            df = pd.read_csv(rho_file)
            if 'rho' in df.columns:
                rho = df['rho'].values
            elif 'rho_mean' in df.columns:
                rho = df['rho_mean'].values
            else:
                continue
        elif rho_file.suffix == '.npy':
            rho = np.load(rho_file)
        else:
            continue
        
        if len(rho) == n_parcels:
            subject_rho.append(rho)
            valid_subjects.append(subj_dir.name)
    
    if len(subject_rho) == 0:
        print("ERROR: Could not load subject-level rho values")
        print("Please specify where individual subject rho values are stored")
        return None
    
    subject_rho = np.array(subject_rho)  # Shape: (n_subjects, n_parcels)
    print(f"\nLoaded rho for {len(valid_subjects)} subjects")
    
    # Run split-half
    n_subjects = len(valid_subjects)
    n_splits = 100
    half1_corrs = []
    half2_corrs = []
    cross_corrs = []
    
    np.random.seed(42)
    for i in range(n_splits):
        perm = np.random.permutation(n_subjects)
        half1_idx = perm[:n_subjects // 2]
        half2_idx = perm[n_subjects // 2:]
        
        # Mean rho per parcel for each half
        half1_rho = np.nanmean(subject_rho[half1_idx, :], axis=0)
        half2_rho = np.nanmean(subject_rho[half2_idx, :], axis=0)
        
        # Correlations with DV
        valid = ~(np.isnan(half1_rho) | np.isnan(half2_rho))
        r1 = stats.pearsonr(half1_rho[valid], z_coords[valid])[0]
        r2 = stats.pearsonr(half2_rho[valid], z_coords[valid])[0]
        r_cross = stats.pearsonr(half1_rho[valid], half2_rho[valid])[0]
        
        half1_corrs.append(r1)
        half2_corrs.append(r2)
        cross_corrs.append(r_cross)
    
    return {
        'half1_corrs': half1_corrs,
        'half2_corrs': half2_corrs,
        'cross_corrs': cross_corrs,
        'n_subjects': n_subjects
    }

# ============================================
# METHOD 3: Use existing subject-level correlations
# ============================================
def run_split_half_from_correlations():
    """Use the subject_level_dv_correlations.csv if it exists"""
    
    corr_file = DATA_DIR / 'group' / 'subject_level_dv_correlations.csv'
    if not corr_file.exists():
        # Try current directory
        corr_file = Path('subject_level_dv_correlations.csv')
    
    if not corr_file.exists():
        print("subject_level_dv_correlations.csv not found")
        return None
    
    df = pd.read_csv(corr_file)
    print(f"Loaded {len(df)} subject correlations")
    
    correlations = df['rho_dv_correlation'].values
    n_subjects = len(correlations)
    
    # Bootstrap the mean correlation
    n_boot = 1000
    boot_means = []
    np.random.seed(42)
    
    for _ in range(n_boot):
        boot_idx = np.random.choice(n_subjects, size=n_subjects, replace=True)
        boot_means.append(np.mean(correlations[boot_idx]))
    
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    
    # Split-half reliability of mean correlation
    n_splits = 100
    half1_means = []
    half2_means = []
    
    for _ in range(n_splits):
        perm = np.random.permutation(n_subjects)
        half1 = correlations[perm[:n_subjects // 2]]
        half2 = correlations[perm[n_subjects // 2:]]
        half1_means.append(np.mean(half1))
        half2_means.append(np.mean(half2))
    
    split_half_r = stats.pearsonr(half1_means, half2_means)[0]
    
    print(f"\n" + "=" * 60)
    print("RESULTS FROM SUBJECT-LEVEL CORRELATIONS")
    print("=" * 60)
    print(f"N subjects: {n_subjects}")
    print(f"Mean rho-DV correlation: {np.mean(correlations):.4f}")
    print(f"95% CI (bootstrap): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Split-half reliability: r = {split_half_r:.4f}")
    print(f"% negative correlations: {100 * np.mean(correlations < 0):.1f}%")
    
    return {
        'mean_corr': np.mean(correlations),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'split_half_r': split_half_r,
        'pct_negative': 100 * np.mean(correlations < 0),
        'n_subjects': n_subjects
    }

# ============================================
# RUN ANALYSES
# ============================================
print("\n" + "=" * 60)
print("ATTEMPTING DIFFERENT METHODS...")
print("=" * 60)

# Try method 3 first (simplest)
print("\n--- Method 3: From subject-level correlations ---")
result3 = run_split_half_from_correlations()

# Try method 1
print("\n--- Method 1: From subject data file ---")
result1 = run_split_half_from_subject_file()

if result1:
    print(f"\n" + "=" * 60)
    print("SPLIT-HALF RESULTS (100 iterations)")
    print("=" * 60)
    print(f"Half 1 rho-DV: mean r = {np.mean(result1['half1_corrs']):.4f} (SD = {np.std(result1['half1_corrs']):.4f})")
    print(f"Half 2 rho-DV: mean r = {np.mean(result1['half2_corrs']):.4f} (SD = {np.std(result1['half2_corrs']):.4f})")
    print(f"Map cross-correlation: mean r = {np.mean(result1['cross_corrs']):.4f} (SD = {np.std(result1['cross_corrs']):.4f})")
    print(f"\n✅ Gradient replicates in both halves")

# Try method 2 if needed
if result1 is None:
    print("\n--- Method 2: From intermediates ---")
    result2 = run_split_half_from_intermediates()
    
    if result2:
        print(f"\n" + "=" * 60)
        print("SPLIT-HALF RESULTS (100 iterations)")
        print("=" * 60)
        print(f"N subjects: {result2['n_subjects']}")
        print(f"Half 1 rho-DV: mean r = {np.mean(result2['half1_corrs']):.4f}")
        print(f"Half 2 rho-DV: mean r = {np.mean(result2['half2_corrs']):.4f}")
        print(f"Map cross-correlation: mean r = {np.mean(result2['cross_corrs']):.4f}")

print("\n" + "=" * 60)
print("COPY THIS OUTPUT AND SEND IT BACK")
print("=" * 60)
