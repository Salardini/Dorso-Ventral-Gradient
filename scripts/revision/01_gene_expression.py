#!/usr/bin/env python3
"""
01_gene_expression.py
Correlate Allen Human Brain Atlas gene expression (PVALB, SST) with ρ.

Requires:
    pip install abagen pandas numpy scipy
    
Input:
    - rho_schaefer400.csv: CSV with columns 'label', 'rho', 'x', 'y', 'z', 'hemi', 'network'
    
Output:
    - ahba_rho_correlations.csv: Correlation results
    - ahba_rho_merged.csv: Merged data (rho + gene expression per parcel)
"""

import numpy as np
import pandas as pd
from scipy import stats
import abagen

# ============================================================
# CONFIGURATION
# ============================================================
RHO_FILE = 'rho_schaefer400.csv'
GENES = ['PVALB', 'SST', 'VIP', 'GAD1', 'GAD2', 'SLC17A7']
ATLAS = 'schaefer_400'  # or path to Schaefer 400 atlas files

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(RHO_FILE)
print(f"Loaded {len(df)} parcels from {RHO_FILE}")

# ============================================================
# GET GENE EXPRESSION
# ============================================================
print("Fetching AHBA gene expression (this may take a few minutes)...")

# Using abagen with Schaefer 400 atlas
# For custom atlas, provide paths to .nii.gz annotation files
expression = abagen.get_expression_data(
    ATLAS,
    lr_mirror='bidirectional',
    missing='interpolate',
    return_donors=False,
    n_proc=4
)

print(f"Expression matrix: {expression.shape}")
print(f"Available genes: {expression.columns.tolist()[:20]}...")

# ============================================================
# MERGE AND CORRELATE
# ============================================================
results = []

for gene in GENES:
    if gene not in expression.columns:
        print(f"  {gene}: not found in expression data")
        continue
    
    gene_vals = expression[gene].values
    
    # Z-score
    gene_z = (gene_vals - np.nanmean(gene_vals)) / np.nanstd(gene_vals)
    df[gene] = gene_z
    
    # Correlation with rho
    mask = ~np.isnan(gene_z)
    r, p = stats.spearmanr(df.loc[mask, 'rho'], gene_z[mask])
    
    results.append({
        'gene': gene,
        'spearman_rho': r,
        'p_value': p,
        'n_parcels': mask.sum(),
        'significant_p05': p < 0.05,
    })
    
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f"  {gene}: ρₛ = {r:+.4f}, p = {p:.2e} ({sig}), n = {mask.sum()}")

# ============================================================
# SAVE
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv('ahba_rho_correlations.csv', index=False)
df.to_csv('ahba_rho_merged.csv', index=False)
print(f"\nSaved ahba_rho_correlations.csv and ahba_rho_merged.csv")
