"""
INTERNAL CONSISTENCY CHECK
Reconcile discrepancies in the manuscript:
- r = -0.735 vs r = -0.723 
- N = 203 vs N = 212
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
FILES_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\files')

print("=" * 60)
print("INTERNAL CONSISTENCY CHECK")
print("=" * 60)

# ============================================
# 1. CHECK PARCEL COUNTS
# ============================================
print("\n1. PARCEL COUNTS")
print("-" * 40)

group_file = DATA_DIR / 'group' / 'parcel_group_maps.csv'
if group_file.exists():
    df = pd.read_csv(group_file)
    print(f"parcel_group_maps.csv: {len(df)} rows")
    print(f"  Unique parcels: {df['parcel_idx'].nunique()}")
    if 'n_subjects' in df.columns:
        print(f"  n_subjects range: {df['n_subjects'].min()} to {df['n_subjects'].max()}")
        print(f"  n_subjects mode: {df['n_subjects'].mode().values[0]}")

# Check filtered version
filtered_file = DATA_DIR / 'group' / 'parcel_group_maps_filtered.csv'
if filtered_file.exists():
    df_filt = pd.read_csv(filtered_file)
    print(f"\nparcel_group_maps_filtered.csv: {len(df_filt)} rows")
    print(f"  Unique parcels: {df_filt['parcel_idx'].nunique()}")

# ============================================
# 2. CHECK SUBJECT COUNTS
# ============================================
print("\n2. SUBJECT COUNTS")
print("-" * 40)

# From subject correlations
subj_corr_file = FILES_DIR / 'subject_level_dv_correlations.csv'
if subj_corr_file.exists():
    subj_df = pd.read_csv(subj_corr_file)
    print(f"subject_level_dv_correlations.csv: {len(subj_df)} subjects")

# From intermediates
intermediates = DATA_DIR / 'MEG_MOUS' / 'intermediates'
if intermediates.exists():
    n_subj_dirs = len(list(intermediates.glob('sub-*')))
    print(f"Intermediate directories: {n_subj_dirs} subjects")

# ============================================
# 3. COMPUTE CORRELATIONS MULTIPLE WAYS
# ============================================
print("\n3. RHO-DV CORRELATIONS (MULTIPLE METHODS)")
print("-" * 40)

if group_file.exists():
    df = pd.read_csv(group_file)
    
    # Method A: All parcels, rho_mean
    rho = df['rho_mean'].values
    z = df['z'].values
    valid = ~(np.isnan(rho) | np.isnan(z))
    r_all, p_all = stats.pearsonr(rho[valid], z[valid])
    print(f"Method A (all parcels, rho_mean): r = {r_all:.4f}, N = {valid.sum()} parcels")
    
    # Method B: Exclude parcels with few subjects
    if 'n_subjects' in df.columns:
        mask = (df['n_subjects'] >= 200) & valid
        r_200, p_200 = stats.pearsonr(rho[mask], z[mask])
        print(f"Method B (n_subjects >= 200): r = {r_200:.4f}, N = {mask.sum()} parcels")
        
        mask = (df['n_subjects'] >= 203) & valid
        r_203, p_203 = stats.pearsonr(rho[mask], z[mask])
        print(f"Method C (n_subjects >= 203): r = {r_203:.4f}, N = {mask.sum()} parcels")

# Check atlas sensitivity file
atlas_file = DATA_DIR / 'atlas_sensitivity' / 'parcel_means_schaefer400.csv'
if atlas_file.exists():
    atlas_df = pd.read_csv(atlas_file)
    print(f"\nAtlas sensitivity file: {len(atlas_df)} parcels")
    if 'rho' in atlas_df.columns and 'z' in atlas_df.columns:
        r_atlas, _ = stats.pearsonr(atlas_df['rho'], atlas_df['z'])
        print(f"Atlas sensitivity r: {r_atlas:.4f}")

# ============================================
# 4. IDENTIFY SOURCE OF DISCREPANCY
# ============================================
print("\n4. DISCREPANCY ANALYSIS")
print("-" * 40)

print("""
REPORTED VALUES:
  - Results section: r = -0.723
  - Extended Data: r = -0.735
  - Subject N: 203 vs 212

LIKELY EXPLANATIONS:
  1. r = -0.723 uses all 400 parcels
  2. r = -0.735 uses filtered parcels (excluding low-coverage)
  3. N = 203 is the minimum subjects per parcel
  4. N = 212 is the total subjects analyzed for individual-level stats
""")

# ============================================
# 5. RECOMMENDED FIX
# ============================================
print("\n5. RECOMMENDED VALUES FOR MANUSCRIPT")
print("-" * 40)

if group_file.exists():
    df = pd.read_csv(group_file)
    rho = df['rho_mean'].values
    z = df['z'].values
    valid = ~(np.isnan(rho) | np.isnan(z))
    r_final, _ = stats.pearsonr(rho[valid], z[valid])
    n_parcels = valid.sum()
    
    if 'n_subjects' in df.columns:
        n_subjects_min = df['n_subjects'].min()
        n_subjects_max = df['n_subjects'].max()
    else:
        n_subjects_min = n_subjects_max = "unknown"
    
    print(f"""
RECOMMENDED (use consistently throughout):
  
  Group-level rho-DV: r = {r_final:.3f}
  N parcels: {n_parcels}
  Subjects per parcel: {n_subjects_min} to {n_subjects_max}
  
  Individual-level:
    N subjects: 212
    Mean r: -0.324
    94% negative
    
  Use one r value throughout (r = {r_final:.3f} or r = {r_final:.2f})
  
  Explain in Methods:
    "N = 203 subjects contributed to each parcel after quality control;
     212 subjects were included in individual-level analyses."
""")

print("=" * 60)
print("COPY THIS OUTPUT AND SEND IT BACK")
print("=" * 60)
