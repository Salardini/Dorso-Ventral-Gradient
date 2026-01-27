#!/usr/bin/env python3
"""
AHBA Gene Expression Analysis: PV/SST vs ρ
==========================================

This script correlates the ρ gradient with interneuron marker genes from
the Allen Human Brain Atlas to test the mechanistic hypothesis that PV/SST
balance underlies the τ-ρ trade-off.

HYPOTHESIS:
- PV (PVALB) interneurons → fast inhibition → high ρ → ventral
- SST interneurons → slow inhibition → low ρ → dorsal
- Therefore: ρ should correlate positively with PVALB and negatively with SST

REQUIREMENTS:
    pip install abagen nilearn nibabel pandas scipy matplotlib

USAGE:
    python ahba_gene_expression_analysis.py

OUTPUT:
    - ahba_interneuron_genes_schaefer400.csv: Gene expression data
    - ahba_rho_gene_correlations.csv: Correlation results
    - fig_ahba_pv_sst.png/pdf: Publication figure

Author: Generated for Salardini et al. "Dorsoventral Gradient of Rotational Dynamics"
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Path to your ρ data
RHO_DATA_PATH = r"C:\Users\u2121\Downloads\MEG\Pipeline\data\atlas_sensitivity\parcel_means_schaefer400.csv"

# Output directory
OUTPUT_DIR = r"C:\Users\u2121\Downloads\MEG\Pipeline\Dorso-Ventral-Gradient\data"

# Target genes (interneuron markers)
TARGET_GENES = {
    'PVALB': 'PV+ fast-spiking interneurons (expect: ventral enrichment, + with ρ)',
    'SST': 'SST+ interneurons (expect: dorsal enrichment, - with ρ)',
    'VIP': 'VIP+ interneurons',
    'LAMP5': 'LAMP5+ interneurons',
    'GAD1': 'GABAergic marker (glutamate decarboxylase)',
    'GAD2': 'GABAergic marker',
    'SLC17A7': 'Excitatory marker (VGLUT1)',
    'SLC17A6': 'Excitatory marker (VGLUT2)',
}


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 70)
    print("AHBA GENE EXPRESSION ANALYSIS: PV/SST vs ρ")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load ρ data
    # -------------------------------------------------------------------------
    print("\n[1] Loading ρ data...")
    
    rho_df = pd.read_csv(RHO_DATA_PATH)
    print(f"    Loaded {len(rho_df)} parcels")
    print(f"    Columns: {rho_df.columns.tolist()}")
    
    # -------------------------------------------------------------------------
    # Step 2: Get Schaefer 400 atlas
    # -------------------------------------------------------------------------
    print("\n[2] Fetching Schaefer 400 parcellation...")
    
    try:
        from nilearn import datasets as nilearn_datasets
        schaefer = nilearn_datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
        atlas_path = schaefer['maps']
        print(f"    Atlas: {atlas_path}")
    except Exception as e:
        print(f"    Error: {e}")
        print("    Please ensure nilearn is installed: pip install nilearn")
        return
    
    # -------------------------------------------------------------------------
    # Step 3: Get AHBA expression data
    # -------------------------------------------------------------------------
    print("\n[3] Fetching AHBA gene expression...")
    print("    (First run downloads ~4GB, subsequent runs use cache)")
    
    try:
        import abagen
        expression = abagen.get_expression_data(atlas_path, verbose=1)
        print(f"\n    ✓ Expression matrix: {expression.shape[0]} parcels × {expression.shape[1]} genes")
    except Exception as e:
        print(f"    Error: {e}")
        print("    Please ensure abagen is installed: pip install abagen")
        return
    
    # -------------------------------------------------------------------------
    # Step 4: Extract target genes
    # -------------------------------------------------------------------------
    print("\n[4] Extracting interneuron marker genes...")
    
    gene_data = {}
    for gene, description in TARGET_GENES.items():
        if gene in expression.columns:
            gene_data[gene] = expression[gene].values
            print(f"    ✓ {gene}: {description.split('(')[0].strip()}")
        else:
            print(f"    ✗ {gene}: not found")
    
    # Create gene expression dataframe
    gene_df = pd.DataFrame(gene_data, index=expression.index)
    
    # -------------------------------------------------------------------------
    # Step 5: Compute derived metrics
    # -------------------------------------------------------------------------
    print("\n[5] Computing derived metrics...")
    
    if 'PVALB' in gene_data and 'SST' in gene_data:
        # Z-score normalize
        pvalb_z = stats.zscore(gene_data['PVALB'], nan_policy='omit')
        sst_z = stats.zscore(gene_data['SST'], nan_policy='omit')
        
        # PV-SST difference (more stable than ratio)
        gene_df['PV_minus_SST'] = pvalb_z - sst_z
        print("    ✓ PV-SST difference computed")
        
        # PV/(PV+SST) ratio
        pv_raw = gene_data['PVALB']
        sst_raw = gene_data['SST']
        gene_df['PV_ratio'] = pv_raw / (pv_raw + sst_raw + 1e-10)
        print("    ✓ PV/(PV+SST) ratio computed")
    
    # -------------------------------------------------------------------------
    # Step 6: Merge with ρ data
    # -------------------------------------------------------------------------
    print("\n[6] Merging with ρ data...")
    
    # Ensure same order (abagen returns parcels 1-400, need 0-indexed)
    gene_df['parcel_idx'] = range(len(gene_df))
    
    merged = pd.merge(rho_df, gene_df, on='parcel_idx', how='inner')
    print(f"    Merged: {len(merged)} parcels")
    
    # -------------------------------------------------------------------------
    # Step 7: Correlate genes with ρ and DV axis
    # -------------------------------------------------------------------------
    print("\n[7] Computing correlations...")
    print("\n    Gene expression vs ρ:")
    print("    " + "-" * 50)
    
    results_rho = {}
    genes_to_test = ['PVALB', 'SST', 'VIP', 'GAD1', 'PV_minus_SST', 'PV_ratio']
    
    for gene in genes_to_test:
        if gene in merged.columns:
            valid = ~np.isnan(merged[gene]) & ~np.isnan(merged['rho'])
            r, p = stats.pearsonr(merged.loc[valid, 'rho'], merged.loc[valid, gene])
            results_rho[gene] = {'r': r, 'p': p, 'n': valid.sum()}
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            direction = '↑' if r > 0 else '↓'
            print(f"    {gene:15s} vs ρ: r = {r:+.3f}, p = {p:.1e} {sig} {direction}")
    
    print("\n    Gene expression vs DV axis (z coordinate):")
    print("    " + "-" * 50)
    
    results_dv = {}
    for gene in genes_to_test:
        if gene in merged.columns:
            valid = ~np.isnan(merged[gene]) & ~np.isnan(merged['z'])
            r, p = stats.pearsonr(merged.loc[valid, 'z'], merged.loc[valid, gene])
            results_dv[gene] = {'r': r, 'p': p, 'n': valid.sum()}
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            loc = 'dorsal' if r > 0 else 'ventral'
            print(f"    {gene:15s} vs DV: r = {r:+.3f}, p = {p:.1e} {sig} ({loc})")
    
    # -------------------------------------------------------------------------
    # Step 8: Save results
    # -------------------------------------------------------------------------
    print("\n[8] Saving results...")
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save merged data
    merged.to_csv(os.path.join(OUTPUT_DIR, 'ahba_rho_merged_schaefer400.csv'), index=False)
    print(f"    ✓ Saved merged data")
    
    # Save correlation results
    results_df = pd.DataFrame({
        'gene': list(results_rho.keys()),
        'r_vs_rho': [results_rho[g]['r'] for g in results_rho],
        'p_vs_rho': [results_rho[g]['p'] for g in results_rho],
        'r_vs_DV': [results_dv[g]['r'] for g in results_dv],
        'p_vs_DV': [results_dv[g]['p'] for g in results_dv],
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'ahba_rho_gene_correlations.csv'), index=False)
    print(f"    ✓ Saved correlation results")
    
    # -------------------------------------------------------------------------
    # Step 9: Create figure
    # -------------------------------------------------------------------------
    print("\n[9] Creating figure...")
    
    fig = create_figure(merged, results_rho, results_dv)
    
    fig_path = os.path.join(OUTPUT_DIR, 'fig_ahba_pv_sst')
    fig.savefig(fig_path + '.png', dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(fig_path + '.pdf', facecolor='white', bbox_inches='tight')
    print(f"    ✓ Saved figure")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("""
HYPOTHESIS TEST:
    If PV interneurons drive high ρ (fast inhibition → oscillations):
        → PVALB should correlate POSITIVELY with ρ
        → PVALB should be enriched VENTRALLY (negative r with z)
    
    If SST interneurons drive low ρ (slow inhibition → stable dynamics):
        → SST should correlate NEGATIVELY with ρ
        → SST should be enriched DORSALLY (positive r with z)

RESULTS:
""")
    
    for gene in ['PVALB', 'SST', 'PV_minus_SST']:
        if gene in results_rho:
            r_rho = results_rho[gene]['r']
            p_rho = results_rho[gene]['p']
            r_dv = results_dv[gene]['r']
            p_dv = results_dv[gene]['p']
            
            rho_dir = "+" if r_rho > 0 else "-"
            dv_dir = "dorsal" if r_dv > 0 else "ventral"
            
            print(f"    {gene}:")
            print(f"        vs ρ:  r = {r_rho:+.3f} (p = {p_rho:.1e}) → {rho_dir} correlation")
            print(f"        vs DV: r = {r_dv:+.3f} (p = {p_dv:.1e}) → enriched {dv_dir}ly")
            print()
    
    print("=" * 70)
    

def create_figure(merged, results_rho, results_dv):
    """Create publication figure for AHBA analysis."""
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
    
    # Panel A: PVALB vs ρ
    ax = axes[0, 0]
    valid = ~np.isnan(merged['PVALB']) & ~np.isnan(merged['rho'])
    x, y = merged.loc[valid, 'rho'], merged.loc[valid, 'PVALB']
    ax.scatter(x, y, c='#e74c3c', s=20, alpha=0.5, edgecolor='none')
    
    m, b = np.polyfit(x, y, 1)
    xf = np.linspace(x.min(), x.max(), 100)
    ax.plot(xf, m*xf + b, 'k-', lw=2)
    
    r, p = results_rho['PVALB']['r'], results_rho['PVALB']['p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax.text(0.95, 0.95, f'r = {r:.2f}{sig}', transform=ax.transAxes,
            va='top', ha='right', bbox=props, fontsize=11)
    
    ax.set_xlabel(r'Rotational Dynamics ($\rho$)', fontsize=12)
    ax.set_ylabel('PVALB Expression (z)', fontsize=12)
    ax.set_title('A. PVALB (PV+) vs ρ', fontsize=13, fontweight='bold', loc='left')
    
    # Panel B: SST vs ρ
    ax = axes[0, 1]
    valid = ~np.isnan(merged['SST']) & ~np.isnan(merged['rho'])
    x, y = merged.loc[valid, 'rho'], merged.loc[valid, 'SST']
    ax.scatter(x, y, c='#3498db', s=20, alpha=0.5, edgecolor='none')
    
    m, b = np.polyfit(x, y, 1)
    xf = np.linspace(x.min(), x.max(), 100)
    ax.plot(xf, m*xf + b, 'k-', lw=2)
    
    r, p = results_rho['SST']['r'], results_rho['SST']['p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax.text(0.95, 0.95, f'r = {r:.2f}{sig}', transform=ax.transAxes,
            va='top', ha='right', bbox=props, fontsize=11)
    
    ax.set_xlabel(r'Rotational Dynamics ($\rho$)', fontsize=12)
    ax.set_ylabel('SST Expression (z)', fontsize=12)
    ax.set_title('B. SST vs ρ', fontsize=13, fontweight='bold', loc='left')
    
    # Panel C: PV-SST difference vs ρ
    ax = axes[1, 0]
    if 'PV_minus_SST' in merged.columns:
        valid = ~np.isnan(merged['PV_minus_SST']) & ~np.isnan(merged['rho'])
        x, y = merged.loc[valid, 'rho'], merged.loc[valid, 'PV_minus_SST']
        ax.scatter(x, y, c='#9b59b6', s=20, alpha=0.5, edgecolor='none')
        
        m, b = np.polyfit(x, y, 1)
        xf = np.linspace(x.min(), x.max(), 100)
        ax.plot(xf, m*xf + b, 'k-', lw=2)
        
        r, p = results_rho['PV_minus_SST']['r'], results_rho['PV_minus_SST']['p']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(0.95, 0.95, f'r = {r:.2f}{sig}', transform=ax.transAxes,
                va='top', ha='right', bbox=props, fontsize=11)
    
    ax.set_xlabel(r'Rotational Dynamics ($\rho$)', fontsize=12)
    ax.set_ylabel('PV − SST (z-scored)', fontsize=12)
    ax.set_title('C. PV/SST Balance vs ρ', fontsize=13, fontweight='bold', loc='left')
    ax.axhline(y=0, color='gray', linestyle='--', lw=1)
    
    # Panel D: Summary bar chart
    ax = axes[1, 1]
    genes = ['PVALB', 'SST', 'VIP', 'GAD1']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    r_vals = [results_rho[g]['r'] for g in genes if g in results_rho]
    p_vals = [results_rho[g]['p'] for g in genes if g in results_rho]
    
    bars = ax.bar(range(len(genes)), r_vals, color=colors, edgecolor='black', lw=1.5)
    
    for i, (bar, p) in enumerate(zip(bars, p_vals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.02 * np.sign(y), sig,
                ha='center', va='bottom' if y > 0 else 'top', fontsize=12, fontweight='bold')
    
    ax.axhline(y=0, color='black', lw=1)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, fontsize=11)
    ax.set_ylabel(r'Correlation with $\rho$', fontsize=12)
    ax.set_title('D. Gene-ρ Correlations', fontsize=13, fontweight='bold', loc='left')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
