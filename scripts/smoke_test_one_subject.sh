#!/bin/bash
# =============================================================================
# smoke_test_one_subject.sh
#
# Quick validation of the MEG axes pipeline on a single subject.
#
# This script tests the full pipeline (excluding FreeSurfer recon-all and
# manual coregistration) on one subject to verify the environment is working.
#
# Prerequisites:
#   - Subject has completed FreeSurfer recon-all
#   - Subject has trans.fif coregistration
#   - BEM can be computed or already exists
#
# Usage:
#   ./scripts/smoke_test_one_subject.sh sub-A2002
#   ./scripts/smoke_test_one_subject.sh A2002 --skip-bem
#
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG="config.yaml"
SKIP_BEM=false
VERBOSE=false

# Parse arguments
SUBJECT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --skip-bem)
            SKIP_BEM=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 <subject> [options]"
            echo ""
            echo "Arguments:"
            echo "  subject         Subject ID (e.g., sub-A2002 or A2002)"
            echo ""
            echo "Options:"
            echo "  --config FILE   Config file (default: config.yaml)"
            echo "  --skip-bem      Skip BEM creation (assume it exists)"
            echo "  --verbose, -v   Verbose output"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            SUBJECT="$1"
            shift
            ;;
    esac
done

if [ -z "$SUBJECT" ]; then
    echo -e "${RED}Error: Subject ID required${NC}"
    echo "Usage: $0 <subject> [options]"
    exit 1
fi

# Normalize subject ID
if [[ ! "$SUBJECT" == sub-* ]]; then
    SUBJECT="sub-${SUBJECT}"
fi

# Find script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check config exists
if [ ! -f "$PROJECT_DIR/$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $PROJECT_DIR/$CONFIG${NC}"
    exit 1
fi

echo "=============================================================================="
echo " MEG Axes Pipeline - Smoke Test"
echo "=============================================================================="
echo " Subject:  $SUBJECT"
echo " Config:   $CONFIG"
echo " Project:  $PROJECT_DIR"
echo "=============================================================================="

# Activate conda environment if needed
if command -v conda &> /dev/null; then
    echo -e "\n${YELLOW}[1/6] Checking conda environment...${NC}"
    # Try to detect if meg environment exists
    if conda env list | grep -q "^meg "; then
        eval "$(conda shell.bash hook)"
        conda activate meg 2>/dev/null || true
        echo "Using conda env: $CONDA_DEFAULT_ENV"
    fi
fi

# Check Python imports
echo -e "\n${YELLOW}[2/6] Checking Python imports...${NC}"
cd "$PROJECT_DIR"
python -c "
import sys
sys.path.insert(0, '.')
print('  mne:', end=' ')
import mne; print(mne.__version__)
print('  mne_bids:', end=' ')
import mne_bids; print(mne_bids.__version__)
print('  numpy:', end=' ')
import numpy as np; print(np.__version__)
print('  scipy:', end=' ')
import scipy; print(scipy.__version__)
print('  pandas:', end=' ')
import pandas as pd; print(pd.__version__)
print('  sklearn:', end=' ')
import sklearn; print(sklearn.__version__)

# Test local imports
from meg_axes.config import load_config
from meg_axes.metrics import compute_tau, compute_rho
from meg_axes.preprocessing import preprocess_raw
from meg_axes.source import build_source_model
from atlas.schaefer import get_schaefer_labels
print('  Local modules: OK')
" || {
    echo -e "${RED}Error: Python import check failed${NC}"
    exit 1
}

# Load config to get paths
echo -e "\n${YELLOW}[3/6] Loading configuration...${NC}"
PATHS=$(python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['paths']['derivatives'])
print(cfg['paths']['subjects_dir'])
print(cfg['paths']['bids_root'] + '/' + cfg['meg_dataset'])
")
DERIVATIVES=$(echo "$PATHS" | sed -n '1p')
SUBJECTS_DIR=$(echo "$PATHS" | sed -n '2p')
BIDS_ROOT=$(echo "$PATHS" | sed -n '3p')

echo "  Derivatives: $DERIVATIVES"
echo "  SUBJECTS_DIR: $SUBJECTS_DIR"
echo "  BIDS root: $BIDS_ROOT"

# Check prerequisites
echo -e "\n${YELLOW}[4/6] Checking prerequisites for $SUBJECT...${NC}"

# FreeSurfer
FS_DIR="$SUBJECTS_DIR/$SUBJECT"
if [ ! -d "$FS_DIR" ]; then
    echo -e "${RED}  ERROR: FreeSurfer directory not found: $FS_DIR${NC}"
    echo "  Run: recon-all -subject $SUBJECT -i <T1w.nii> -all -sd $SUBJECTS_DIR"
    exit 1
fi

FS_DONE="$FS_DIR/scripts/recon-all.done"
if [ ! -f "$FS_DONE" ]; then
    echo -e "${YELLOW}  WARNING: recon-all.done not found (reconstruction may be incomplete)${NC}"
else
    echo -e "${GREEN}  FreeSurfer: OK${NC}"
fi

# Trans.fif
TRANS_PATH="$DERIVATIVES/coreg/$SUBJECT/trans.fif"
if [ ! -f "$TRANS_PATH" ]; then
    echo -e "${RED}  ERROR: trans.fif not found: $TRANS_PATH${NC}"
    echo "  Run: mne coreg --subject $SUBJECT --subjects-dir $SUBJECTS_DIR"
    echo "  See: scripts/03_make_trans.md for instructions"
    exit 1
fi
echo -e "${GREEN}  Coregistration: OK${NC}"

# BEM
BEM_PATH="$DERIVATIVES/bem/$SUBJECT/bem-sol.fif"
if [ ! -f "$BEM_PATH" ] && [ "$SKIP_BEM" = false ]; then
    echo "  BEM: not found, will create"
else
    echo -e "${GREEN}  BEM: OK${NC}"
fi

# MEG data
MEG_DIR="$BIDS_ROOT/$SUBJECT/meg"
if [ ! -d "$MEG_DIR" ]; then
    echo -e "${RED}  ERROR: MEG data not found: $MEG_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}  MEG data: OK${NC}"

# Create BEM if needed
if [ ! -f "$BEM_PATH" ] && [ "$SKIP_BEM" = false ]; then
    echo -e "\n${YELLOW}[5/6] Creating BEM solution...${NC}"

    if [ "$VERBOSE" = true ]; then
        python scripts/02_make_bem.py --config "$CONFIG" --subject "$SUBJECT"
    else
        python scripts/02_make_bem.py --config "$CONFIG" --subject "$SUBJECT" 2>&1 | tail -5
    fi

    if [ ! -f "$BEM_PATH" ]; then
        echo -e "${RED}  ERROR: BEM creation failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}  BEM created successfully${NC}"
else
    echo -e "\n${YELLOW}[5/6] Skipping BEM (already exists)${NC}"
fi

# Run extraction
echo -e "\n${YELLOW}[6/6] Running parcel extraction...${NC}"

# Remove existing output to force reprocessing
AXES_DIR="$DERIVATIVES/axes/$SUBJECT"
if [ -d "$AXES_DIR" ]; then
    echo "  Removing existing output: $AXES_DIR"
    rm -rf "$AXES_DIR"
fi

START_TIME=$(date +%s)

if [ "$VERBOSE" = true ]; then
    python scripts/04_extract_parcels_and_metrics.py \
        --config "$CONFIG" \
        --subject "$SUBJECT" \
        --verbose
else
    python scripts/04_extract_parcels_and_metrics.py \
        --config "$CONFIG" \
        --subject "$SUBJECT" \
        2>&1 | grep -E "^\[|Starting|Completed|Saved|ERROR|WARNING" || true
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Check outputs
echo -e "\n${YELLOW}Checking outputs...${NC}"

DONE_FILE="$AXES_DIR/DONE"
METRICS_FILE="$AXES_DIR/parcel_metrics.csv"
TS_FILE="$AXES_DIR/parcel_ts.npy"
META_FILE="$AXES_DIR/meta.json"

if [ ! -f "$DONE_FILE" ]; then
    echo -e "${RED}  ERROR: DONE marker not created${NC}"
    echo "  Check logs at: $AXES_DIR/log.txt"
    exit 1
fi

if [ ! -f "$METRICS_FILE" ]; then
    echo -e "${RED}  ERROR: parcel_metrics.csv not created${NC}"
    exit 1
fi

if [ ! -f "$TS_FILE" ]; then
    echo -e "${RED}  ERROR: parcel_ts.npy not created${NC}"
    exit 1
fi

if [ ! -f "$META_FILE" ]; then
    echo -e "${YELLOW}  WARNING: meta.json not created${NC}"
fi

# Validate outputs
echo -e "\n${YELLOW}Validating outputs...${NC}"
python -c "
import numpy as np
import pandas as pd
import json

# Check metrics CSV
df = pd.read_csv('$METRICS_FILE')
n_parcels = len(df)
n_nan_tau = df['tau'].isna().sum()
n_nan_rho = df['rho'].isna().sum()

print(f'  Parcels: {n_parcels}')
print(f'  Tau: mean={df[\"tau\"].mean():.4f}, NaN={n_nan_tau}')
print(f'  Rho: mean={df[\"rho\"].mean():.4f}, NaN={n_nan_rho}')

# Check required columns
required = ['tau', 'rho', 'x', 'y', 'z', 'label']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f'  WARNING: Missing columns: {missing}')

# Check time series
ts = np.load('$TS_FILE')
print(f'  Time series shape: {ts.shape}')

# Check meta
if '$META_FILE':
    with open('$META_FILE') as f:
        meta = json.load(f)
    print(f'  Duration: {meta.get(\"duration_s\", \"N/A\")}s')
    print(f'  Sfreq: {meta.get(\"sfreq\", \"N/A\")} Hz')

# Validation
assert n_parcels == 400, f'Expected 400 parcels, got {n_parcels}'
assert n_nan_tau < n_parcels * 0.1, f'Too many NaN tau values: {n_nan_tau}'
assert n_nan_rho < n_parcels * 0.1, f'Too many NaN rho values: {n_nan_rho}'
assert ts.shape[0] == n_parcels, 'Time series parcel count mismatch'
assert ts.shape[1] > 100, 'Time series too short'

print('  Validation: PASSED')
"

# Summary
echo ""
echo "=============================================================================="
echo -e "${GREEN} SMOKE TEST PASSED${NC}"
echo "=============================================================================="
echo " Subject:    $SUBJECT"
echo " Runtime:    ${ELAPSED}s"
echo " Output:     $AXES_DIR"
echo ""
echo " Files created:"
echo "   - parcel_ts.npy"
echo "   - parcel_metrics.csv"
echo "   - meta.json"
echo "   - log.txt"
echo "   - DONE"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Process remaining subjects:"
echo "     python scripts/run_batch.py --config $CONFIG --stages extract"
echo ""
echo "  2. Run group statistics (after processing multiple subjects):"
echo "     python scripts/05_group_stats.py --config $CONFIG"
echo ""
