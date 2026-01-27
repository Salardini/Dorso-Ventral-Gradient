# GitHub Update - Final Steps

## Step 1: Download these figures from Claude's response above and save to:
```
C:\Users\u2121\Downloads\MEG\Pipeline\Dorso-Ventral-Gradient\figures\
```

Files to download and save:
- `fig_computational_model.png`
- `fig_computational_model.pdf`
- `fig_model_ahba_ABC.png`
- `fig_model_ahba_ABC.pdf`
- `fig_ahba_results.png`
- `fig_ahba_results.pdf`

## Step 2: Copy the AHBA figure from data to figures folder
```powershell
cd C:\Users\u2121\Downloads\MEG\Pipeline\Dorso-Ventral-Gradient
copy data\fig_ahba_pv_sst.png figures\
copy data\fig_ahba_pv_sst.pdf figures\
```

## Step 3: Git commands to push
```powershell
cd C:\Users\u2121\Downloads\MEG\Pipeline\Dorso-Ventral-Gradient

# Stage all new and modified files
git add manuscript/Paper1_v7_manuscript.md
git add code/computational_model_tau_rho.py
git add code/computational_model_description.md
git add code/ahba_fixed2.py
git add code/ahba_gene_expression_analysis.py
git add data/ahba_rho_correlations.csv
git add data/ahba_rho_merged.csv
git add figures/fig_computational_model.png
git add figures/fig_computational_model.pdf
git add figures/fig_model_ahba_ABC.png
git add figures/fig_model_ahba_ABC.pdf
git add figures/fig_ahba_results.png
git add figures/fig_ahba_results.pdf
git add figures/fig_ahba_pv_sst.png
git add figures/fig_ahba_pv_sst.pdf
git add GITHUB_UPDATE.md

# Commit
git commit -m "Add computational model and AHBA gene expression analysis

NEW RESULTS:
- Computational model: tau-rho trade-off emerges from E-I balance (r=-0.84)
- AHBA analysis: SST (not PV) correlates with high rho (r=+0.12, p=0.02)
- SST enriched ventrally (r=-0.17, p=0.002)
- Revised interpretation: dendritic (SST) vs somatic (PV) inhibition

FILES ADDED:
- manuscript/Paper1_v7_manuscript.md - Updated with model + AHBA
- code/computational_model_tau_rho.py - Full model implementation
- code/ahba_fixed2.py - AHBA analysis (pandas 2.x compatible)
- data/ahba_rho_*.csv - Gene expression results
- figures/fig_computational_model.* - Model figure
- figures/fig_model_ahba_ABC.* - Combined figure panels A-C
- figures/fig_ahba_*.* - Gene expression figures"

# Push to GitHub
git push origin main
```

## Summary of what's being pushed:

| Category | Files | Description |
|----------|-------|-------------|
| Manuscript | Paper1_v7_manuscript.md | Full updated paper |
| Code | computational_model_tau_rho.py | E-I network model |
| Code | ahba_fixed2.py | Gene expression analysis |
| Data | ahba_rho_correlations.csv | Results table |
| Data | ahba_rho_merged.csv | Full merged data |
| Figures | fig_computational_model.* | 4-panel model figure |
| Figures | fig_model_ahba_ABC.* | Panels A, B, C only |
| Figures | fig_ahba_*.* | Gene expression figures |
