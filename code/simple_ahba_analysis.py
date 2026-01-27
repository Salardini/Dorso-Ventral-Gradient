#!/usr/bin/env python3
"""
SIMPLE AHBA Analysis Script
===========================

Run this locally to download AHBA data and correlate with ρ.

INSTRUCTIONS:
1. Install: pip install abagen nilearn
2. Run: python simple_ahba_analysis.py
3. Share the output CSV and figure with Claude

This will take ~10 minutes on first run (downloads 4GB).
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION - Edit these paths!
# =============================================================================

# Your ρ data file
RHO_DATA_PATH = r"C:\Users\u2121\Downloads\MEG\Pipeline\data\atlas_sensitivity\parcel_means_schaefer400.csv"

# Output directory
OUTPUT_DIR = r"C:\Users\u2121\Downloads\MEG\Pipeline\Dorso-Ventral-Gradient\data"

# =============================================================================
# MAIN
# =============================================================================

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("AHBA Gene Expression Analysis")
    print("=" * 60)
    
    # Load ρ data
    print("\n[1] Loading ρ data...")
    rho_df = pd.read_csv(RHO_DATA_PATH)
    print(f"    Loaded {len(rho_df)} parcels")
    
    # Get atlas
    print("\n[2] Fetching Schaefer 400 atlas...")
    from nilearn import datasets
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    
    # Get AHBA expression
    print("\n[3] Fetching AHBA expression (this takes a while first time)...")
    import abagen
    expression = abagen.get_expression_data(schaefer['maps'], verbose=1)
    print(f"    Got {expression.shape[0]} parcels × {expression.shape[1]} genes")
    
    # Extract key genes
    print("\n[4] Extracting interneuron markers...")
    genes = ['PVALB', 'SST', 'VIP', 'GAD1', 'GAD2']
    gene_df = expression[genes].copy()
    gene_df['parcel_idx'] = range(len(gene_df))
    
    # Compute PV-SST
    pv_z = stats.zscore(gene_df['PVALB'], nan_policy='omit')
    sst_z = stats.zscore(gene_df['SST'], nan_policy='omit')
    gene_df['PV_minus_SST'] = pv_z - sst_z
    
    # Merge
    print("\n[5] Merging with ρ data...")
    merged = pd.merge(rho_df, gene_df, on='parcel_idx')
    
    # Correlations
    print("\n[6] Computing correlations...")
    print("\n    Gene vs ρ:")
    print("    " + "-" * 50)
    
    results = []
    for gene in ['PVALB', 'SST', 'VIP', 'GAD1', 'PV_minus_SST']:
        valid = ~np.isnan(merged[gene]) & ~np.isnan(merged['rho'])
        r, p = stats.pearsonr(merged.loc[valid, 'rho'], merged.loc[valid, gene])
        results.append({'gene': gene, 'r_vs_rho': r, 'p_vs_rho': p})
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"    {gene:15s}: r = {r:+.3f}, p = {p:.1e} {sig}")
    
    print("\n    Gene vs DV (z-coordinate):")
    print("    " + "-" * 50)
    
    for i, gene in enumerate(['PVALB', 'SST', 'VIP', 'GAD1', 'PV_minus_SST']):
        valid = ~np.isnan(merged[gene]) & ~np.isnan(merged['z'])
        r, p = stats.pearsonr(merged.loc[valid, 'z'], merged.loc[valid, gene])
        results[i]['r_vs_DV'] = r
        results[i]['p_vs_DV'] = p
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        loc = 'dorsal' if r > 0 else 'ventral'
        print(f"    {gene:15s}: r = {r:+.3f}, p = {p:.1e} {sig} ({loc})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/ahba_rho_correlations.csv", index=False)
    merged.to_csv(f"{OUTPUT_DIR}/ahba_rho_merged.csv", index=False)
    print(f"\n    Saved: ahba_rho_correlations.csv")
    print(f"    Saved: ahba_rho_merged.csv")
    
    # Figure
    print("\n[7] Creating figure...")
    create_figure(merged, results_df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("DONE! Share the results CSV with Claude.")
    print("=" * 60)


def create_figure(merged, results_df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    
    # A: PVALB vs rho
    ax = axes[0, 0]
    valid = ~np.isnan(merged['PVALB']) & ~np.isnan(merged['rho'])
    ax.scatter(merged.loc[valid, 'rho'], merged.loc[valid, 'PVALB'], 
               c='#e74c3c', s=15, alpha=0.5)
    r = results_df.loc[results_df['gene']=='PVALB', 'r_vs_rho'].values[0]
    ax.set_xlabel('ρ'); ax.set_ylabel('PVALB')
    ax.set_title(f'A. PVALB vs ρ (r={r:.2f})')
    
    # B: SST vs rho
    ax = axes[0, 1]
    valid = ~np.isnan(merged['SST']) & ~np.isnan(merged['rho'])
    ax.scatter(merged.loc[valid, 'rho'], merged.loc[valid, 'SST'],
               c='#3498db', s=15, alpha=0.5)
    r = results_df.loc[results_df['gene']=='SST', 'r_vs_rho'].values[0]
    ax.set_xlabel('ρ'); ax.set_ylabel('SST')
    ax.set_title(f'B. SST vs ρ (r={r:.2f})')
    
    # C: PV-SST vs rho
    ax = axes[1, 0]
    valid = ~np.isnan(merged['PV_minus_SST']) & ~np.isnan(merged['rho'])
    ax.scatter(merged.loc[valid, 'rho'], merged.loc[valid, 'PV_minus_SST'],
               c='#9b59b6', s=15, alpha=0.5)
    r = results_df.loc[results_df['gene']=='PV_minus_SST', 'r_vs_rho'].values[0]
    ax.set_xlabel('ρ'); ax.set_ylabel('PV - SST')
    ax.set_title(f'C. PV/SST Balance vs ρ (r={r:.2f})')
    ax.axhline(0, color='gray', ls='--')
    
    # D: Summary
    ax = axes[1, 1]
    genes = ['PVALB', 'SST', 'VIP', 'GAD1']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    r_vals = [results_df.loc[results_df['gene']==g, 'r_vs_rho'].values[0] for g in genes]
    ax.bar(genes, r_vals, color=colors, edgecolor='black')
    ax.axhline(0, color='black')
    ax.set_ylabel('r vs ρ')
    ax.set_title('D. Gene-ρ Correlations')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_ahba_pv_sst.png", dpi=300)
    plt.savefig(f"{output_dir}/fig_ahba_pv_sst.pdf")
    print(f"    Saved: fig_ahba_pv_sst.png")


if __name__ == "__main__":
    main()
