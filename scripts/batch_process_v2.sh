#\!/bin/bash
# Batch processing with proper parallelism

MAX_JOBS=${1:-2}  # Default to 2 parallel jobs
GCS_DIR="/mnt/gcs/raw_data/MEG_MOUS"
BIDS_DIR="/mnt/work/bids/MEG_MOUS"
DONE_DIR="/mnt/work/derivatives/axes"

mkdir -p "$BIDS_DIR" "$DONE_DIR"

# Get list of subjects from GCS
SUBJECTS=$(ls "$GCS_DIR"/*.tar.gz 2>/dev/null | xargs -n1 basename | sed "s/.tar.gz//" | sort)
TOTAL=$(echo "$SUBJECTS" | wc -l)

echo "========================================"
echo "MEG Axes Pipeline - Batch Processing v2"
echo "========================================"
echo "Total subjects: $TOTAL"
echo "Max parallel jobs: $MAX_JOBS"
echo "========================================"

# Create wrapper script for parallel execution
cat > /tmp/process_one.sh << 'EOF'
#\!/bin/bash
subj=$1
GCS_DIR="/mnt/gcs/raw_data/MEG_MOUS"
BIDS_DIR="/mnt/work/bids/MEG_MOUS"
DONE_DIR="/mnt/work/derivatives/axes"

cd ~/meg_axes_pipeline

# Skip if already done
if [ -f "$DONE_DIR/$subj/DONE" ]; then
    echo "[SKIP] $subj - already processed"
    exit 0
fi

echo "[START] $subj"

# Extract if not exists
if [ \! -d "$BIDS_DIR/$subj" ]; then
    tar -xzf "$GCS_DIR/${subj}.tar.gz" -C "$BIDS_DIR/"
fi

# Run pipeline
python scripts/04_extract_parcels_and_metrics.py --config config.yaml --subject "$subj" --skip-existing 2>&1

# Check success and cleanup
if [ -f "$DONE_DIR/$subj/DONE" ]; then
    echo "[DONE] $subj"
    rm -rf "$BIDS_DIR/$subj"
else
    echo "[FAIL] $subj"
fi
EOF
chmod +x /tmp/process_one.sh

echo "$SUBJECTS" | xargs -P "$MAX_JOBS" -I {} /tmp/process_one.sh {}

echo "========================================"
echo "Batch complete\!"
echo "========================================"
