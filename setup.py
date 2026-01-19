#!/usr/bin/env python3
"""Setup script for meg_axes package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="meg_axes",
    version="1.0.0",
    author="Arash Salardini",
    author_email="",
    description="MEG source reconstruction with parcel-wise temporal dynamics metrics (tau, rho)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asalardini/meg_axes_pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "mne>=1.0.0",
        "mne-bids>=0.10",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "spin": ["brainspace>=0.1.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "meg-axes-extract=scripts.04_extract_parcels_and_metrics:main",
            "meg-axes-group=scripts.05_group_stats:main",
        ],
    },
    include_package_data=True,
    package_data={
        "meg_axes": ["../atlas/*.csv", "../config*.yaml"],
    },
)
