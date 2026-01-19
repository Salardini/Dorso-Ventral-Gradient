# MEG Axes Pipeline

Subject-specific MEG source reconstruction with Schaefer-400 parcellation, computing parcel-wise temporal dynamics metrics (tau, rho) and group-level spatial statistics.

## Overview

This pipeline processes BIDS-formatted CTF MEG resting-state data through:

0. **Data extraction** - Extract tar.gz archives to BIDS structure
1. **FreeSurfer reconstruction** - Cortical surface reconstruction from T1w MRI
2. **BEM creation** - Boundary element model for forward modeling
3. **Coregistration** - MEG-MRI alignment via MNE coreg GUI
4. **Source reconstruction** - dSPM inverse with Schaefer-400 parcellation
5. **Group statistics** - Spatial correlations with spin-test inference

## Directory Structure

```
/mnt/work/
├── bids/
│   └── MEG_MOUS/              # BIDS MEG dataset
│       └── sub-AXXXX/
│           └── meg/
│               └── sub-AXXXX_task-rest_meg.ds
├── derivatives/
│   ├── freesurfer/            # FreeSurfer SUBJECTS_DIR
│   │   └── sub-AXXXX/
│   ├── bem/
│   │   └── sub-AXXXX/
│   │       ├── bem-sol.fif
│   │       └── DONE
│   ├── coreg/
│   │   └── sub-AXXXX/
│   │       ├── trans.fif
│   │       └── coreg_screenshot.png
│   ├── axes/
│   │   └── sub-AXXXX/
│   │       ├── parcel_ts.npy
│   │       ├── parcel_metrics.csv
│   │       └── DONE
│   ├── group/
│   │   ├── parcel_group_maps.csv
│   │   ├── map_stats.csv
│   │   └── figures/
│   └── logs/
│       └── reconall/
│           └── sub-AXXXX.log
```

## Environment Setup

```bash
# Activate conda environment
conda activate meg

# Set FreeSurfer
export SUBJECTS_DIR=/mnt/work/derivatives/freesurfer

# Verify installations
python -c "import mne; print(mne.__version__)"
freesurfer --version
```

## Workflow

### Stage 0: Extract Tarballs

The raw data arrives as tar.gz archives (e.g., `sub-A2002.tar.gz`). Extract to BIDS structure before processing.

```bash
# Extract MEG data
python scripts/00_extract_tarballs.py \
    --src /mnt/data/MEG_MOUS \
    --dest /mnt/work/bids/MEG_MOUS

# Extract anatomy data
python scripts/00_extract_tarballs.py \
    --src /mnt/data/anat \
    --dest /mnt/work/bids/anat

# Extract specific subjects only
python scripts/00_extract_tarballs.py \
    --src /mnt/data/MEG_MOUS \
    --dest /mnt/work/bids/MEG_MOUS \
    --subjects A2002,A2003,A2004
```

**Input:** `sub-XXXX.tar.gz` archives
**Output:** Extracted BIDS directories + `.extracted` marker

### Stage 1: FreeSurfer Reconstruction

Runs `recon-all` on T1w anatomical images.

```bash
# Single subject
./scripts/01_reconall.sh sub-A2002
```

**Input:** `/mnt/work/bids/<anat_dataset>/sub-XXX/anat/sub-XXX_T1w.nii`
**Output:** `$SUBJECTS_DIR/sub-XXX/` + `DONE` marker
**Time:** ~6-12 hours per subject

### Stage 2: BEM Solution

Creates single-shell BEM for MEG forward modeling.

```bash
# Single subject
python scripts/02_make_bem.py sub-A2002

# Batch
python scripts/run_batch.py --stage 2
```

**Input:** FreeSurfer reconstruction
**Output:** `/mnt/work/derivatives/bem/sub-XXX/bem-sol.fif` + `DONE` marker
**Time:** ~5 minutes per subject

### Stage 3: Coregistration (Manual)

Interactive MEG-MRI alignment using MNE coreg GUI.

See `scripts/03_make_trans.md` for detailed instructions.

```bash
# Launch coreg GUI
mne coreg --subject sub-A2002 --subjects-dir $SUBJECTS_DIR
```

**Output:** `/mnt/work/derivatives/coreg/sub-XXX/trans.fif`
**Time:** ~5-10 minutes per subject (manual)

### Stage 4: Source Reconstruction & Metrics

Main analysis: inverse solution, parcellation, tau/rho computation.

```bash
# Single subject
python scripts/04_extract_parcels_and_metrics.py sub-A2002

# Batch
python scripts/run_batch.py --stage 4
```

**Outputs:**
- `/mnt/work/derivatives/axes/sub-XXX/parcel_ts.npy` - Parcel time series (400 x T)
- `/mnt/work/derivatives/axes/sub-XXX/parcel_metrics.csv` - Tau, rho per parcel
- `DONE` marker

**Time:** ~10-30 minutes per subject

### Stage 5: Group Statistics

Aggregates subjects and computes spatial statistics.

```bash
python scripts/05_group_stats.py
```

**Outputs:**
- `/mnt/work/derivatives/group/parcel_group_maps.csv` - Mean/median tau, rho
- `/mnt/work/derivatives/group/map_stats.csv` - AP/DV/ML correlations, spin p-values
- Figures in `/mnt/work/derivatives/group/figures/`

## Metrics

### Tau (Intrinsic Timescale)

Autocorrelation function (ACF) integral timescale:

```
tau = integral from lag_min to lag_max of ACF(lag) d(lag)
```

Measures how slowly neural activity decorrelates - higher tau indicates longer temporal integration windows.

### Rho (Rotational Index)

Delay-embedded VAR(1) rotational dynamics index:

1. Embed parcel time series in delay coordinates
2. Fit VAR(1) model: x(t+1) = A @ x(t)
3. Compute eigenvalues of A
4. Rho = mean imaginary/real ratio of complex eigenvalue pairs

Measures rotational vs. decaying dynamics - higher rho indicates more oscillatory activity patterns.

## Configuration

Edit `config.yaml` to modify:

- Preprocessing: bandpass, notch frequencies, resampling
- Source reconstruction: spacing, SNR, method
- Parcellation: Schaefer parameters
- Tau/rho computation parameters
- Group statistics: spin test permutations

## Resumability

All stages use `DONE` markers for idempotent execution:

- `run_batch.py` automatically skips completed subjects
- Re-running a failed stage continues from last checkpoint
- Delete `DONE` marker to force re-processing

## Quality Control

Each stage produces QC outputs:

1. **recon-all:** Check `recon-all.log` for errors
2. **BEM:** Visual inspection of BEM surfaces
3. **Coreg:** Screenshot saved with trans.fif
4. **Metrics:** `parcel_metrics.csv` includes QC fields (variance, n_samples)
5. **Group:** Summary statistics and outlier flags

## Troubleshooting

**FreeSurfer fails:**
- Check log in `/mnt/work/derivatives/logs/reconall/sub-XXX.log`
- Common issues: poor T1 quality, incorrect orientation

**BEM fails:**
- Usually indicates FreeSurfer watershed issues
- Try: `mne watershed_bem --subject sub-XXX --overwrite`

**Coreg issues:**
- Ensure fiducials are correctly placed
- Use ICP refinement in coreg GUI

**Memory errors in Stage 4:**
- Reduce `resample_freq` in config
- Process fewer parcels at once

## References

- Schaefer et al. (2018) - Schaefer parcellation
- Honey et al. (2012) - Intrinsic timescales
- MNE-Python documentation: https://mne.tools
