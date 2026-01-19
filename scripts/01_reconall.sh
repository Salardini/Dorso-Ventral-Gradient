#!/usr/bin/env bash
#
# 01_reconall.sh - Run FreeSurfer recon-all for a single subject
#
# Usage:
#   ./01_reconall.sh sub-A2002 [/path/to/bids/anat]
#
# Inputs:
#   - BIDS T1w: <bids_anat>/sub-XXX/anat/sub-XXX_T1w.nii or .nii.gz
#
# Outputs:
#   - FreeSurfer reconstruction: $SUBJECTS_DIR/sub-XXX/
#   - Log: /mnt/work/derivatives/logs/reconall/sub-XXX.log
#   - DONE marker: $SUBJECTS_DIR/sub-XXX/DONE
#
# Environment:
#   - SUBJECTS_DIR must be set (default: /mnt/work/derivatives/freesurfer)
#   - FreeSurfer must be sourced
#
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (can be overridden by environment variables)
# -----------------------------------------------------------------------------
SUBJECTS_DIR="${SUBJECTS_DIR:-/mnt/work/derivatives/freesurfer}"
LOGS_DIR="${LOGS_DIR:-/mnt/work/derivatives/logs/reconall}"
BIDS_ANAT="${2:-/mnt/work/bids}"
N_THREADS="${FREESURFER_THREADS:-4}"

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <subject_id> [bids_anat_root]"
    echo "Example: $0 sub-A2002 /mnt/work/bids/anat"
    exit 1
fi

SUBJECT="$1"
# Ensure subject has sub- prefix
if [[ ! "$SUBJECT" =~ ^sub- ]]; then
    SUBJECT="sub-${SUBJECT}"
fi

# -----------------------------------------------------------------------------
# Verify FreeSurfer is available
# -----------------------------------------------------------------------------
if ! command -v recon-all &> /dev/null; then
    echo "ERROR: recon-all not found. Please source FreeSurfer:"
    echo "  source \$FREESURFER_HOME/SetUpFreeSurfer.sh"
    exit 1
fi

# -----------------------------------------------------------------------------
# Setup directories
# -----------------------------------------------------------------------------
mkdir -p "$SUBJECTS_DIR"
mkdir -p "$LOGS_DIR"

LOG_FILE="${LOGS_DIR}/${SUBJECT}.log"
DONE_MARKER="${SUBJECTS_DIR}/${SUBJECT}/DONE"

# -----------------------------------------------------------------------------
# Check if already completed
# -----------------------------------------------------------------------------
if [[ -f "$DONE_MARKER" ]]; then
    echo "[SKIP] ${SUBJECT}: DONE marker exists at ${DONE_MARKER}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Find T1w input
# -----------------------------------------------------------------------------
T1W_INPUT=""

# Try various common BIDS layouts
for ext in ".nii.gz" ".nii"; do
    # Standard BIDS: <root>/sub-XXX/anat/sub-XXX_T1w.nii[.gz]
    candidate="${BIDS_ANAT}/${SUBJECT}/anat/${SUBJECT}_T1w${ext}"
    if [[ -f "$candidate" ]]; then
        T1W_INPUT="$candidate"
        break
    fi

    # Nested dataset: <root>/<dataset>/sub-XXX/anat/sub-XXX_T1w.nii[.gz]
    for dataset_dir in "${BIDS_ANAT}"/*; do
        if [[ -d "${dataset_dir}/${SUBJECT}/anat" ]]; then
            candidate="${dataset_dir}/${SUBJECT}/anat/${SUBJECT}_T1w${ext}"
            if [[ -f "$candidate" ]]; then
                T1W_INPUT="$candidate"
                break 2
            fi
        fi
    done
done

if [[ -z "$T1W_INPUT" ]]; then
    echo "ERROR: Could not find T1w for ${SUBJECT}"
    echo "Searched in: ${BIDS_ANAT}"
    exit 1
fi

echo "========================================"
echo "FreeSurfer recon-all: ${SUBJECT}"
echo "========================================"
echo "Input T1w:    ${T1W_INPUT}"
echo "SUBJECTS_DIR: ${SUBJECTS_DIR}"
echo "Log file:     ${LOG_FILE}"
echo "Threads:      ${N_THREADS}"
echo "========================================"

# -----------------------------------------------------------------------------
# Run recon-all
# -----------------------------------------------------------------------------
{
    echo "=== recon-all started: $(date) ==="
    echo "Subject: ${SUBJECT}"
    echo "Input: ${T1W_INPUT}"
    echo "SUBJECTS_DIR: ${SUBJECTS_DIR}"
    echo ""

    recon-all \
        -subjid "${SUBJECT}" \
        -i "${T1W_INPUT}" \
        -all \
        -threads "${N_THREADS}" \
        -sd "${SUBJECTS_DIR}"

    RECON_STATUS=$?

    echo ""
    echo "=== recon-all finished: $(date) ==="
    echo "Exit status: ${RECON_STATUS}"

    if [[ $RECON_STATUS -eq 0 ]]; then
        # Verify critical outputs exist
        SURF_LH="${SUBJECTS_DIR}/${SUBJECT}/surf/lh.pial"
        SURF_RH="${SUBJECTS_DIR}/${SUBJECT}/surf/rh.pial"

        if [[ -f "$SURF_LH" && -f "$SURF_RH" ]]; then
            echo "OK" > "${DONE_MARKER}"
            echo "SUCCESS: ${SUBJECT} completed"
        else
            echo "ERROR: recon-all finished but surfaces missing"
            exit 1
        fi
    else
        echo "ERROR: recon-all failed with status ${RECON_STATUS}"
        exit ${RECON_STATUS}
    fi

} 2>&1 | tee "${LOG_FILE}"

# Get the exit status from the subshell
PIPE_STATUS=${PIPESTATUS[0]}
if [[ $PIPE_STATUS -ne 0 ]]; then
    echo "ERROR: recon-all failed. Check log: ${LOG_FILE}"
    exit $PIPE_STATUS
fi

echo "[OK] ${SUBJECT} -> ${SUBJECTS_DIR}/${SUBJECT}"
