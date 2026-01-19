"""
meg_axes/config.py

Single source of truth configuration loader with CLI override support.

Usage:
    from meg_axes.config import load_config

    # Load from file with CLI overrides
    cfg = load_config("config.yaml", cli_overrides={"preprocessing.resample_fs": 300})

    # Access nested config
    print(cfg.preprocessing.resample_fs)
    print(cfg.paths.bids_root)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class PathsConfig:
    """Path configuration."""
    bids_root: str = "/mnt/work/bids"
    derivatives: str = "/mnt/work/derivatives"
    subjects_dir: str = "/mnt/work/derivatives/freesurfer"
    logs_dir: str = "/mnt/work/derivatives/logs"

    # Derived paths (computed)
    bem_dir: str = ""
    coreg_dir: str = ""
    axes_dir: str = ""
    group_dir: str = ""

    def __post_init__(self):
        if not self.bem_dir:
            self.bem_dir = os.path.join(self.derivatives, "bem")
        if not self.coreg_dir:
            self.coreg_dir = os.path.join(self.derivatives, "coreg")
        if not self.axes_dir:
            self.axes_dir = os.path.join(self.derivatives, "axes")
        if not self.group_dir:
            self.group_dir = os.path.join(self.derivatives, "group")


@dataclass
class PreprocessingConfig:
    """MEG preprocessing configuration."""
    task: str = "rest"
    session: Optional[str] = None
    acquisition: Optional[str] = None
    run: str = "auto"

    # Filtering
    notch_freqs: Tuple[float, ...] = (60.0, 120.0, 180.0)
    l_freq: float = 1.0
    h_freq: float = 40.0
    resample_fs: float = 200.0

    # Duration
    max_dur_s: Optional[float] = None

    # ICA (optional, off by default)
    ica_enabled: bool = False
    ica_n_components: int = 20
    ica_method: str = "fastica"
    ica_max_iter: int = 500


@dataclass
class SourceConfig:
    """Source reconstruction configuration."""
    # Template mode: None for subject-specific, "fsaverage" for template-based
    template: Optional[str] = None

    # Source space
    spacing: str = "oct6"

    # Inverse
    method: str = "dSPM"
    snr: float = 3.0
    loose: float = 0.2
    depth: float = 0.8
    pick_ori: str = "normal"

    # BEM
    conductivity: Tuple[float, ...] = (0.3,)


@dataclass
class ParcellationConfig:
    """Parcellation configuration."""
    atlas: str = "schaefer"
    n_parcels: int = 400
    n_networks: int = 7
    resolution_mm: int = 1
    extract_mode: str = "pca_flip"

    # Robustness check
    alt_n_parcels: int = 200  # For Schaefer200 vs 400 comparison


@dataclass
class TauConfig:
    """Intrinsic timescale (tau) configuration."""
    # Integration window
    lag_min_s: float = 0.005
    lag_max_s: float = 0.300

    # Primary estimator
    primary_method: str = "integral"  # "integral" or "exponential"

    # Exponential fit settings (secondary)
    exp_fit_max_lag_s: float = 0.300


@dataclass
class RhoConfig:
    """Rotational index (rho) configuration."""
    embed_dim: int = 10
    embed_delay: int = 1
    ridge_alpha: float = 1e-3
    mag_min: float = 1e-2

    # Number of eigenvalue pairs to average
    n_pairs: int = 1


@dataclass
class GroupStatsConfig:
    """Group statistics configuration."""
    n_permutations: int = 1000
    corr_method: str = "spearman"

    # Spin test fallback
    spin_fallback_to_perm: bool = True

    # Coordinate axes (MNI convention)
    # x = ML (medial-lateral), y = AP (anterior-posterior), z = DV (dorsal-ventral)
    axes_mapping: Dict[str, str] = field(default_factory=lambda: {
        "ML": "x",
        "AP": "y",
        "DV": "z"
    })


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    # Parallelization limits
    max_reconall_jobs: int = 2  # FreeSurfer is memory-intensive
    max_bem_jobs: int = 4
    max_extract_jobs: int = 4

    # Behavior
    skip_existing: bool = True
    fail_fast: bool = False


@dataclass
class QCConfig:
    """Quality control configuration."""
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    # Dataset
    meg_dataset: str = "MEG_MOUS"

    # Sub-configs
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    parcellation: ParcellationConfig = field(default_factory=ParcellationConfig)
    tau: TauConfig = field(default_factory=TauConfig)
    rho: RhoConfig = field(default_factory=RhoConfig)
    group_stats: GroupStatsConfig = field(default_factory=GroupStatsConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    def get_meg_bids_root(self) -> str:
        """Get full path to MEG BIDS dataset."""
        return os.path.join(self.paths.bids_root, self.meg_dataset)


# =============================================================================
# Configuration Loading
# =============================================================================

def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update nested dictionary."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _apply_cli_overrides(config_dict: dict, overrides: Dict[str, Any]) -> dict:
    """
    Apply CLI overrides to config dictionary.

    Overrides use dot notation: "preprocessing.resample_fs" -> 300
    """
    result = config_dict.copy()

    for key, value in overrides.items():
        if value is None:
            continue

        parts = key.split(".")
        current = result

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value
        current[parts[-1]] = value

    return result


def _dict_to_config(data: dict) -> PipelineConfig:
    """Convert dictionary to PipelineConfig dataclass."""
    return PipelineConfig(
        meg_dataset=data.get("meg_dataset", "MEG_MOUS"),
        paths=PathsConfig(**data.get("paths", {})),
        preprocessing=PreprocessingConfig(
            **{k: tuple(v) if isinstance(v, list) and k == "notch_freqs" else v
               for k, v in data.get("preprocessing", {}).items()}
        ),
        source=SourceConfig(
            **{k: tuple(v) if isinstance(v, list) and k == "conductivity" else v
               for k, v in data.get("source", {}).items()}
        ),
        parcellation=ParcellationConfig(**data.get("parcellation", {})),
        tau=TauConfig(**data.get("tau", {})),
        rho=RhoConfig(**data.get("rho", {})),
        group_stats=GroupStatsConfig(**data.get("group_stats", {})),
        batch=BatchConfig(**data.get("batch", {})),
        qc=QCConfig(**data.get("qc", {})),
    )


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file with optional CLI overrides.

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml. If None, searches standard locations.
    cli_overrides : dict, optional
        Dictionary of dot-notation overrides, e.g., {"preprocessing.resample_fs": 300}

    Returns
    -------
    PipelineConfig
        Loaded and validated configuration.
    """
    config_dict = {}

    # Find config file
    if config_path is None:
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / ".meg_axes" / "config.yaml",
        ]
        for p in search_paths:
            if p.exists():
                config_path = str(p)
                break

    # Load YAML
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if cli_overrides:
        config_dict = _apply_cli_overrides(config_dict, cli_overrides)

    # Convert to dataclass
    return _dict_to_config(config_dict)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add standard config arguments to an argument parser.

    This adds --config plus common override arguments.
    """
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml"
    )

    # Path overrides
    parser.add_argument("--bids_root", help="Override paths.bids_root")
    parser.add_argument("--derivatives", help="Override paths.derivatives")
    parser.add_argument("--subjects_dir", help="Override paths.subjects_dir")

    # MEG dataset
    parser.add_argument("--meg_dataset", help="Override meg_dataset")

    # Preprocessing overrides
    parser.add_argument("--task", help="Override preprocessing.task")
    parser.add_argument("--run", help="Override preprocessing.run")
    parser.add_argument("--band", nargs=2, type=float, metavar=("L", "H"),
                        help="Override preprocessing l_freq/h_freq")
    parser.add_argument("--resample", type=float, help="Override preprocessing.resample_fs")
    parser.add_argument("--max_dur", type=float, help="Override preprocessing.max_dur_s")

    # Source overrides
    parser.add_argument("--inv_method", help="Override source.method")
    parser.add_argument("--src_spacing", help="Override source.spacing")

    # Parcellation overrides
    parser.add_argument("--n_parcels", type=int, help="Override parcellation.n_parcels")

    # Metric overrides
    parser.add_argument("--embed_dim", type=int, help="Override rho.embed_dim")
    parser.add_argument("--embed_delay", type=int, help="Override rho.embed_delay")
    parser.add_argument("--ridge_alpha", type=float, help="Override rho.ridge_alpha")

    return parser


def args_to_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert parsed arguments to config override dictionary.
    """
    overrides = {}

    # Paths
    if getattr(args, "bids_root", None):
        overrides["paths.bids_root"] = args.bids_root
    if getattr(args, "derivatives", None):
        overrides["paths.derivatives"] = args.derivatives
    if getattr(args, "subjects_dir", None):
        overrides["paths.subjects_dir"] = args.subjects_dir

    # Dataset
    if getattr(args, "meg_dataset", None):
        overrides["meg_dataset"] = args.meg_dataset

    # Preprocessing
    if getattr(args, "task", None):
        overrides["preprocessing.task"] = args.task
    if getattr(args, "run", None):
        overrides["preprocessing.run"] = args.run
    if getattr(args, "band", None):
        overrides["preprocessing.l_freq"] = args.band[0]
        overrides["preprocessing.h_freq"] = args.band[1]
    if getattr(args, "resample", None):
        overrides["preprocessing.resample_fs"] = args.resample
    if getattr(args, "max_dur", None):
        overrides["preprocessing.max_dur_s"] = args.max_dur

    # Source
    if getattr(args, "inv_method", None):
        overrides["source.method"] = args.inv_method
    if getattr(args, "src_spacing", None):
        overrides["source.spacing"] = args.src_spacing

    # Parcellation
    if getattr(args, "n_parcels", None):
        overrides["parcellation.n_parcels"] = args.n_parcels

    # Rho
    if getattr(args, "embed_dim", None):
        overrides["rho.embed_dim"] = args.embed_dim
    if getattr(args, "embed_delay", None):
        overrides["rho.embed_delay"] = args.embed_delay
    if getattr(args, "ridge_alpha", None):
        overrides["rho.ridge_alpha"] = args.ridge_alpha

    return overrides
