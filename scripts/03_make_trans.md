# MEG-MRI Coregistration Guide

This document provides step-by-step instructions for creating the MEG-MRI transformation file (`trans.fif`) using MNE's coregistration GUI.

## Prerequisites

1. FreeSurfer reconstruction completed for the subject
2. MEG data accessible (for fiducial/digitization points)
3. X11 display available (local or X-forwarding)

## CTF-Specific Note: Using space-CTF_T1w.nii

Many CTF MEG datasets include a `*space-CTF_T1w.nii.gz` file which is the T1 image
**already aligned to MEG coordinates** during acquisition. This can significantly
simplify coregistration:

- If `space-CTF_T1w.nii` exists, it means the MRI was acquired with localizer coils
- The transformation from this space to MEG space is essentially identity
- Use this file to verify your coregistration matches the expected alignment

**To use space-CTF_T1w.nii:**

1. Check if it exists: `ls /mnt/work/bids/MEG_MOUS/sub-XXXX/anat/*space-CTF*.nii*`
2. If present, your trans.fif should result in minimal movement of the MRI
3. Use `scripts/coreg_check.py` to validate alignment quality

## Output Path Convention

All transformation files must be saved to:
```
/mnt/work/derivatives/coreg/sub-XXXX/trans.fif
```

## Step-by-Step Instructions

### 1. Setup Environment

```bash
# Activate conda environment
conda activate meg

# Set FreeSurfer paths
export SUBJECTS_DIR=/mnt/work/derivatives/freesurfer

# For remote servers, enable X11 forwarding
# ssh -X user@server
```

### 2. Launch Coregistration GUI

```bash
# For a single subject
SUBJECT=sub-A2002
mne coreg --subject $SUBJECT --subjects-dir $SUBJECTS_DIR
```

Or from Python:
```python
import mne
mne.gui.coregistration(subject='sub-A2002', subjects_dir='/mnt/work/derivatives/freesurfer')
```

### 3. Load MEG Data (Optional)

If the GUI doesn't auto-detect digitization points:

1. Click **"Load digitizer data"** (or Ctrl+D)
2. Navigate to MEG dataset: `/mnt/work/bids/MEG_MOUS/sub-XXXX/meg/sub-XXXX_task-rest_meg.ds`
3. The head digitization points should appear as small dots

### 4. Initial Alignment via Fiducials

#### 4.1 MRI Fiducials (Red markers)

Set anatomical landmarks on the MRI:

1. **LPA (Left Pre-Auricular)**: Click the point just anterior to the left ear canal
2. **RPA (Right Pre-Auricular)**: Same position on the right side
3. **Nasion**: Bridge of the nose between the eyes

**Tip**: Use the orthogonal views to precisely place fiducials.

#### 4.2 Lock Fiducials

Once satisfied with MRI fiducial positions:
- Click **"Lock Fiducials"** to prevent accidental changes

### 5. Coarse Alignment

1. Click **"Fit Fiducials"** to perform initial fiducial-based alignment
2. The MEG sensor helmet and head digitization should roughly align with the scalp

### 6. Fine Alignment with ICP

1. Click **"Fit ICP"** (Iterative Closest Point)
2. Recommended iterations: 20-50
3. Watch the error metric decrease
4. The head points should closely match the scalp surface

**Parameters** (adjust in GUI if needed):
- **Omit distance**: 5mm (ignore points far from scalp)
- **Iterations**: 20-50

### 7. Quality Check

Verify alignment quality:

1. **Visual inspection**: Rotate the 3D view to check alignment from multiple angles
2. **Error metric**: Should be < 5mm mean distance
3. **Fiducial distance**: Individual fiducial errors shown in GUI

**Warning signs of poor alignment:**
- Head points floating above/below scalp
- Systematic offset in one direction
- Fiducial errors > 10mm

### 8. Save Transformation

1. Click **"Save As..."** (or Ctrl+S)
2. Navigate to: `/mnt/work/derivatives/coreg/sub-XXXX/`
3. Create directory if needed
4. Save as: `trans.fif`

**Full path**: `/mnt/work/derivatives/coreg/sub-XXXX/trans.fif`

### 9. Save QC Screenshot

Capture a screenshot for quality control documentation:

1. Rotate view to show clear alignment (slightly angled lateral view works well)
2. Take screenshot:
   - **Linux**: Use screenshot tool or `import -window root screenshot.png`
   - **macOS**: Cmd+Shift+4
   - **Windows**: Win+Shift+S

3. Save as: `/mnt/work/derivatives/coreg/sub-XXXX/coreg_screenshot.png`

## Batch Processing Tips

### Create Subject List
```bash
# List all subjects needing coreg
for sub in /mnt/work/derivatives/freesurfer/sub-*/; do
    subj=$(basename $sub)
    if [ ! -f "/mnt/work/derivatives/coreg/$subj/trans.fif" ]; then
        echo "$subj needs coreg"
    fi
done
```

### Verify All Coreg Files
```bash
# Check all trans.fif exist
for sub in /mnt/work/derivatives/freesurfer/sub-*/; do
    subj=$(basename $sub)
    trans="/mnt/work/derivatives/coreg/$subj/trans.fif"
    if [ -f "$trans" ]; then
        echo "[OK] $subj"
    else
        echo "[MISSING] $subj"
    fi
done
```

## Troubleshooting

### GUI Won't Launch

```bash
# Check X11
echo $DISPLAY

# Test X11
xeyes

# If using SSH, reconnect with X forwarding
ssh -X user@server
```

### No Head Digitization Points

- CTF datasets should have digitization in the .ds folder
- Check for `*.pos` or `*.hsp` files
- Some CTF systems store points differently; consult dataset documentation

### Poor ICP Convergence

1. Reset alignment: **Edit > Reset**
2. Manually adjust MRI fiducials to better match MEG fiducials
3. Increase ICP iterations
4. Exclude outlier points by increasing omit distance

### Subject Not Found

```bash
# Verify FreeSurfer subject exists
ls $SUBJECTS_DIR/sub-XXXX/

# Check that recon-all completed
cat $SUBJECTS_DIR/sub-XXXX/scripts/recon-all.done
```

## Alternative: Scripted Coregistration

For datasets with reliable digitization, automated coregistration may work:

```python
import mne

subject = 'sub-A2002'
subjects_dir = '/mnt/work/derivatives/freesurfer'
meg_path = '/mnt/work/bids/MEG_MOUS/sub-A2002/meg/sub-A2002_task-rest_meg.ds'

# Load MEG info
info = mne.io.read_info(meg_path)

# Automated coregistration (requires good fiducials in MEG)
coreg = mne.coreg.Coregistration(info, subject, subjects_dir)
coreg.fit_fiducials()
coreg.fit_icp(n_iterations=50)

# Save
trans_path = f'/mnt/work/derivatives/coreg/{subject}/trans.fif'
mne.write_trans(trans_path, coreg.trans)
```

**Note**: Always visually verify automated coregistration results!

## Summary Checklist

- [ ] FreeSurfer recon-all completed
- [ ] MNE coreg GUI launched successfully
- [ ] MEG digitization points loaded
- [ ] MRI fiducials (LPA, RPA, Nasion) placed accurately
- [ ] Initial fiducial fit performed
- [ ] ICP refinement completed (error < 5mm)
- [ ] trans.fif saved to correct path
- [ ] QC screenshot saved
- [ ] Visual inspection confirms good alignment
