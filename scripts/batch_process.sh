#!/bin/bash
# Batch process all subjects from GCS mount
# Extracts each subject, processes, then cleans up to save disk space

set -e
cd ~/meg_axes_pipeline
source /mnt/work/miniconda/etc/profile.d/conda.sh
conda activate meg

GCS_DIR="/mnt/gcs/raw_data/MEG_MOUS"
BIDS_DIR="/mnt/work/bids/MEG_MOUS"
MAX_JOBS=${1:-4}

# Get list of subjects
SUBJECTS=$(ls $GCS_DIR/*.tar.gz | xargs -n1 basename | sed "s/.tar.gz//" | sort)
TOTAL=$(echo "$SUBJECTS" | wc -l)

echo "========================================"
echo "MEG Axes Pipeline - Batch Processing"
echo "========================================"
echo "Total subjects: $TOTAL"
echo "Max parallel jobs: $MAX_JOBS"
echo "========================================"

process_subject() {
    local subj=$1
    local tar_file="$GCS_DIR/${subj}.tar.gz"
    
    # Skip if already done
    if [ -f "/mnt/work/derivatives/axes/${subj}/DONE" ]; then
        echo "[SKIP] $subj - already processed"
        return 0
    fi
    
    echo "[START] $subj - extracting..."
    
    # Extract to BIDS dir
    tar -xzf "$tar_file" -C "$BIDS_DIR/" 2>/dev/null || {
        echo "[ERROR] $subj - extraction failed"
        return 1
    }
    
    echo "[PROC] $subj - running pipeline..."
    
    # Run pipeline
    python scripts/04_extract_parcels_and_metrics.py \
        --config config.yaml \
        --subject "$subj" \
        --skip-existing 2>&1 | grep -E "^20|Completed|Error" || true
    
    # Check success
    if [ -f "/mnt/work/derivatives/axes/${subj}/DONE" ]; then
        echo "[DONE] $subj"
        # Clean up extracted data to save space
        rm -rf "$BIDS_DIR/$subj"
    else
        echo "[FAIL] $subj"
    fi
}

export -f process_subject
export GCS_DIR BIDS_DIR

# Run in parallel
echo "$SUBJECTS" | xargs -P $MAX_JOBS -I {} bash -c "process_subject {}"

echo "========================================"
echo "Batch processing complete!"
echo "========================================"
