"""
MEG Axes Pipeline - Publication-grade MEG source reconstruction and metrics.

Modules:
    config: Configuration loading and CLI override handling
    metrics: Tau and rho computation with multiple estimators
    preprocessing: MEG preprocessing with optional ICA
    source: Source modeling and parcellation
    utils: Logging, versioning, and file utilities
"""

__version__ = "1.0.0"

from .config import load_config, PipelineConfig
from .utils import get_version_info, setup_logging, write_done_marker, write_meta_json
