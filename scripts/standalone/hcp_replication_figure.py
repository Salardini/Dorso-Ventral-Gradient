#!/usr/bin/env python3
"""
HCP Replication Figure - FIXED
==============================
Uses each dataset's own z coordinates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
MOUS_PATH = Path(r"C:\Users\u2121\Downloads\MEG\Pipeline\data\fmri_replication\fmri_parcel_measures_v2.csv")
HCP_REST_PATH = Path(r"C:\Users\u2121\Downloads\MEG\fMRI2\results\hcp_parcel_rho.csv")
HCP_TASK_PATH = Path(r"C:\Users\u2121\Downloads\MEG\fMRI2\results\hcp_task_parcel_rho.csv")
OUTPUT_DIR = Path(r"C:\Users\u2121\Downloads\MEG\fMRI2\results")

# Load data
print("Loading data...")
mous_df = pd.read_csv(MOUS_PATH)
hcp_rest_df = pd.read_csv(HCP_REST_PATH)
hcp_task_df = pd.read_csv(HCP_TASK_PATH)

# Extract values - USE EACH DATASET'S OWN Z COORDINATES
mous_rho = mous_df['rho_fmri'].values
mous_z = mous_df['z'].values  # MOUS's own z coordinates

hcp_rest_rho = hcp_rest_df['rho_mean'].values
hcp_task_rho = hcp_task_df['rho_mean'].values
hcp_z = hcp_task_df['z'].values  # HCP's z coordinates

# Verify correlations
valid_mous = ~np.isnan(mous_rho) & ~np.isnan(mous_z)
r_mous, p_mous = stats.pearsonr(mous_rho[valid_mous], mous_z[valid_mous])
r_hcp, p_hcp = stats.pearsonr(hcp_task_rho, hcp_z)

print(f"\nVerifying correlations:")
print(f"  MOUS ρ vs MOUS z: r = {r_mous:.4f}")
print(f"  HCP Task ρ vs HCP z: r = {r_hcp:.4f}")

# =============================================================================
# Create figure - 3 panels
# =============================================================================
fig = plt.figure(figsize=(14, 4.5))

c_mous = '#2ecc71'      # Green
c_hcp_rest = '#95a5a6'  # Gray
c_hcp_task = '#3498db'  # Blue

# -----------------------------------------------------------------------------
# Panel A: ρ vs DV - MOUS (using MOUS coordinates)
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(1, 3, 1)

ax1.scatter(mous_z[valid_mous], mous_rho[valid_mous], alpha=0.6, s=25, c=c_mous, edgecolor='white', linewidth=0.3)
z_fit = np.linspace(mous_z[valid_mous].min(), mous_z[valid_mous].max(), 100)
m, b = np.polyfit(mous_z[valid_mous], mous_rho[valid_mous], 1)
ax1.plot(z_fit, m*z_fit + b, c='#1a7a3e', lw=2.5)

ax1.set_xlabel('Dorsoventral Coordinate (z)', fontsize=11)
ax1.set_ylabel('Rotational Dynamics (ρ)', fontsize=11)
ax1.set_title('A. MOUS fMRI (N=200)', fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.text(0.95, 0.95, f'r = {r_mous:.3f}\np < 0.001', 
         transform=ax1.transAxes, fontsize=11, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# -----------------------------------------------------------------------------
# Panel B: ρ vs DV - HCP Task (using HCP coordinates)
# -----------------------------------------------------------------------------
ax2 = fig.add_subplot(1, 3, 2)

ax2.scatter(hcp_z, hcp_task_rho, alpha=0.6, s=25, c=c_hcp_task, edgecolor='white', linewidth=0.3)
z_fit = np.linspace(hcp_z.min(), hcp_z.max(), 100)
m, b = np.polyfit(hcp_z, hcp_task_rho, 1)
ax2.plot(z_fit, m*z_fit + b, c='#1a5276', lw=2.5)

ax2.set_xlabel('Dorsoventral Coordinate (z)', fontsize=11)
ax2.set_ylabel('Rotational Dynamics (ρ)', fontsize=11)
ax2.set_title('B. HCP Task fMRI (N=1,093)', fontsize=13, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.text(0.95, 0.95, f'r = {r_hcp:.3f}\np = 0.014', 
         transform=ax2.transAxes, fontsize=11, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# -----------------------------------------------------------------------------
# Panel C: Summary bar chart
# -----------------------------------------------------------------------------
ax3 = fig.add_subplot(1, 3, 3)

datasets = ['MOUS\n(N=200)', 'HCP REST\n(N=1,096)', 'HCP TASK\n(N=1,093)']
correlations = [r_mous, 0.0003, r_hcp]
colors = [c_mous, c_hcp_rest, c_hcp_task]

bars = ax3.bar(datasets, correlations, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# Add significance markers
sig_markers = ['***', '', '*']
for bar, sig in zip(bars, sig_markers):
    if sig:
        height = bar.get_height()
        y_offset = -0.015 if height < 0 else 0.015
        va = 'top' if height < 0 else 'bottom'
        ax3.text(bar.get_x() + bar.get_width()/2, height + y_offset, sig, 
                 ha='center', va=va, fontsize=16, fontweight='bold')

# HCP REST label
ax3.text(1, 0.015, 'n.s.', ha='center', va='bottom', fontsize=11, color='#555')

ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_ylabel('Correlation with DV (r)', fontsize=11)
ax3.set_title('C. Replication Summary', fontsize=13, fontweight='bold')
ax3.set_ylim(-0.32, 0.08)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Annotation about bandpass
ax3.annotate('', xy=(1, -0.01), xytext=(2, r_hcp),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
ax3.text(1.5, -0.08, 'broader\nfilter', ha='center', fontsize=9, color='#e74c3c', style='italic')

ax3.text(0.5, -0.30, '* p < 0.05, *** p < 0.001 (spin permutation)', 
         ha='center', fontsize=9, style='italic', transform=ax3.transAxes)

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / "hcp_replication_figure.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path}")

output_pdf = OUTPUT_DIR / "hcp_replication_figure.pdf"
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_pdf}")

plt.show()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("REPLICATION SUMMARY")
print("=" * 70)
print(f"""
DV GRADIENT:
  MOUS fMRI (N=200):     r = {r_mous:.3f}, p < 0.001 ✓
  HCP REST (N=1,096):    r =  0.000, p = 0.99  ✗ (tight 0.009-0.08 Hz filter)
  HCP TASK (N=1,093):    r = {r_hcp:.3f}, p = 0.014 ✓ (broader 0.009-0.25 Hz filter)

INTERPRETATION:
  Both show NEGATIVE correlation: ventral (low z) = HIGH ρ, dorsal (high z) = LOW ρ
""")
