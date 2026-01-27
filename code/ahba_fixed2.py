#!/usr/bin/env python3
"""
AHBA Analysis - Full pandas 2.x compatibility fix
==================================================

abagen has multiple deprecated pandas calls. This patches all of them.

Run: python ahba_fixed2.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
RHO_DATA_PATH = r"C:\Users\u2121\Downloads\MEG\Pipeline\data\atlas_sensitivity\parcel_means_schaefer400.csv"
OUTPUT_DIR = r"C:\Users\u2121\Downloads\MEG\Pipeline\Dorso-Ventral-Gradient\data"

def apply_pandas2_patches():
    """Patch pandas to restore deprecated methods that abagen uses."""
    
    # 1. Restore DataFrame.append
    if not hasattr(pd.DataFrame, 'append'):
        def df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
            if isinstance(other, pd.DataFrame):
                return pd.concat([self, other], ignore_index=ignore_index, 
                               verify_integrity=verify_integrity, sort=sort)
            elif isinstance(other, pd.Series):
                return pd.concat([self, other.to_frame().T], ignore_index=ignore_index,
                               verify_integrity=verify_integrity, sort=sort)
            else:
                return pd.concat([self, pd.DataFrame([other])], ignore_index=ignore_index,
                               verify_integrity=verify_integrity, sort=sort)
        pd.DataFrame.append = df_append
        print("    ✓ Patched DataFrame.append")
    
    # 2. Restore Series.append
    if not hasattr(pd.Series, 'append'):
        def series_append(self, to_append, ignore_index=False, verify_integrity=False):
            return pd.concat([self, to_append], ignore_index=ignore_index,
                           verify_integrity=verify_integrity)
        pd.Series.append = series_append
        print("    ✓ Patched Series.append")
    
    # 3. Fix set_axis inplace argument
    _original_set_axis = pd.DataFrame.set_axis
    def fixed_set_axis(self, labels, axis=0, inplace=False, copy=None):
        if inplace:
            # Just ignore inplace and return new object
            pass
        return _original_set_axis(self, labels, axis=axis, copy=copy)
    pd.DataFrame.set_axis = fixed_set_axis
    print("    ✓ Patched DataFrame.set_axis")

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("AHBA Gene Expression Analysis")
    print("=" * 60)
    
    # Load rho data
    print("\n[1] Loading ρ data...")
    rho_df = pd.read_csv(RHO_DATA_PATH)
    print(f"    Loaded {len(rho_df)} parcels")
    
    # Get atlas
    print("\n[2] Fetching Schaefer 400 atlas...")
    from nilearn import datasets
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    atlas_path = schaefer['maps']
    
    # Apply patches BEFORE importing abagen
    print("\n[3] Applying pandas 2.x compatibility patches...")
    apply_pandas2_patches()
    
    # Now import and use abagen
    print("\n[4] Fetching AHBA expression...")
    import abagen
    expression = abagen.get_expression_data(atlas_path, verbose=1)
    print(f"    Got {expression.shape[0]} parcels × {expression.shape[1]} genes")
    
    # Extract genes
    print("\n[5] Extracting interneuron markers...")
    target_genes = ['PVALB', 'SST', 'VIP', 'GAD1', 'GAD2']
    available = [g for g in target_genes if g in expression.columns]
    print(f"    Found: {available}")
    
    gene_df = expression[available].copy()
    gene_df['parcel_idx'] = range(len(gene_df))
    
    # PV-SST
    if 'PVALB' in available and 'SST' in available:
        pv_z = stats.zscore(gene_df['PVALB'], nan_policy='omit')
        sst_z = stats.zscore(gene_df['SST'], nan_policy='omit')
        gene_df['PV_minus_SST'] = pv_z - sst_z
    
    # Merge
    print("\n[6] Merging with ρ data...")
    merged = pd.merge(rho_df, gene_df, on='parcel_idx')
    print(f"    Merged: {len(merged)} parcels")
    
    # Correlations
    print("\n[7] Computing correlations...")
    print("\n    Gene vs ρ:")
    print("    " + "-" * 50)
    
    results = []
    test_genes = available + (['PV_minus_SST'] if 'PV_minus_SST' in gene_df.columns else [])
    
    for gene in test_genes:
        valid = ~np.isnan(merged[gene]) & ~np.isnan(merged['rho'])
        if valid.sum() > 10:
            r, p = stats.pearsonr(merged.loc[valid, 'rho'], merged.loc[valid, gene])
            results.append({'gene': gene, 'r_vs_rho': r, 'p_vs_rho': p})
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {gene:15s}: r = {r:+.3f}, p = {p:.1e} {sig}")
    
    print("\n    Gene vs DV (z-coordinate):")
    print("    " + "-" * 50)
    
    for i, gene in enumerate(test_genes):
        if i < len(results):
            valid = ~np.isnan(merged[gene]) & ~np.isnan(merged['z'])
            if valid.sum() > 10:
                r, p = stats.pearsonr(merged.loc[valid, 'z'], merged.loc[valid, gene])
                results[i]['r_vs_DV'] = r
                results[i]['p_vs_DV'] = p
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                loc = 'dorsal' if r > 0 else 'ventral'
                print(f"    {gene:15s}: r = {r:+.3f}, p = {p:.1e} {sig} ({loc})")
    
    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/ahba_rho_correlations.csv", index=False)
    merged.to_csv(f"{OUTPUT_DIR}/ahba_rho_merged.csv", index=False)
    print(f"\n    Saved: ahba_rho_correlations.csv")
    
    # Figure
    print("\n[8] Creating figure...")
    create_figure(merged, results_df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)

def create_figure(merged, results_df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    
    # A: PVALB
    ax = axes[0,0]
    if 'PVALB' in merged.columns:
        valid = ~np.isnan(merged['PVALB']) & ~np.isnan(merged['rho'])
        x, y = merged.loc[valid, 'rho'], merged.loc[valid, 'PVALB']
        ax.scatter(x, y, c='#e74c3c', s=15, alpha=0.5)
        m, b = np.polyfit(x, y, 1)
        ax.plot([x.min(), x.max()], [m*x.min()+b, m*x.max()+b], 'k-', lw=2)
        row = results_df[results_df['gene']=='PVALB']
        if len(row):
            r, p = row['r_vs_rho'].values[0], row['p_vs_rho'].values[0]
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
            ax.text(0.95, 0.95, f'r = {r:.2f}{sig}', transform=ax.transAxes, ha='right', va='top', bbox=props)
    ax.set_xlabel('ρ'); ax.set_ylabel('PVALB'); ax.set_title('A. PVALB vs ρ', fontweight='bold', loc='left')
    
    # B: SST
    ax = axes[0,1]
    if 'SST' in merged.columns:
        valid = ~np.isnan(merged['SST']) & ~np.isnan(merged['rho'])
        x, y = merged.loc[valid, 'rho'], merged.loc[valid, 'SST']
        ax.scatter(x, y, c='#3498db', s=15, alpha=0.5)
        m, b = np.polyfit(x, y, 1)
        ax.plot([x.min(), x.max()], [m*x.min()+b, m*x.max()+b], 'k-', lw=2)
        row = results_df[results_df['gene']=='SST']
        if len(row):
            r, p = row['r_vs_rho'].values[0], row['p_vs_rho'].values[0]
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
            ax.text(0.95, 0.95, f'r = {r:.2f}{sig}', transform=ax.transAxes, ha='right', va='top', bbox=props)
    ax.set_xlabel('ρ'); ax.set_ylabel('SST'); ax.set_title('B. SST vs ρ', fontweight='bold', loc='left')
    
    # C: PV-SST
    ax = axes[1,0]
    if 'PV_minus_SST' in merged.columns:
        valid = ~np.isnan(merged['PV_minus_SST']) & ~np.isnan(merged['rho'])
        x, y = merged.loc[valid, 'rho'], merged.loc[valid, 'PV_minus_SST']
        ax.scatter(x, y, c='#9b59b6', s=15, alpha=0.5)
        m, b = np.polyfit(x, y, 1)
        ax.plot([x.min(), x.max()], [m*x.min()+b, m*x.max()+b], 'k-', lw=2)
        row = results_df[results_df['gene']=='PV_minus_SST']
        if len(row):
            r, p = row['r_vs_rho'].values[0], row['p_vs_rho'].values[0]
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
            ax.text(0.95, 0.95, f'r = {r:.2f}{sig}', transform=ax.transAxes, ha='right', va='top', bbox=props)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('ρ'); ax.set_ylabel('PV - SST'); ax.set_title('C. PV/SST Balance vs ρ', fontweight='bold', loc='left')
    
    # D: Summary
    ax = axes[1,1]
    genes = ['PVALB', 'SST', 'VIP', 'GAD1']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    r_vals, p_vals, valid_genes, valid_colors = [], [], [], []
    for g, c in zip(genes, colors):
        row = results_df[results_df['gene']==g]
        if len(row):
            r_vals.append(row['r_vs_rho'].values[0])
            p_vals.append(row['p_vs_rho'].values[0])
            valid_genes.append(g)
            valid_colors.append(c)
    
    bars = ax.bar(valid_genes, r_vals, color=valid_colors, edgecolor='black')
    for bar, p in zip(bars, p_vals):
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        y = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, y+0.02*np.sign(y), sig, ha='center', fontweight='bold')
    ax.axhline(0, color='black')
    ax.set_ylabel('r vs ρ'); ax.set_title('D. Summary', fontweight='bold', loc='left')
    
    for a in axes.flat:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_ahba_pv_sst.png", dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig_ahba_pv_sst.pdf", facecolor='white', bbox_inches='tight')
    print(f"    Saved figures")

if __name__ == "__main__":
    main()
