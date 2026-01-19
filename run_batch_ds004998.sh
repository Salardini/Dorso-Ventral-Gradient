#!/bin/bash
cd ~/meg_axes_pipeline
SUBJECTS=$(ls -d /mnt/work/bids/ds004998/sub-* | xargs -n1 basename)
echo "Found subjects: $SUBJECTS"
for sub in $SUBJECTS; do
    echo "========================================"
    echo "Processing $sub at $(date)"
    echo "========================================"
    python3 scripts/04_extract_parcels_and_metrics.py --config config_ds004998.yaml --subject $sub 2>&1 | tail -10
    echo ""
done
echo "Batch complete at $(date)"
