"""Filesystem configuration shared across the pipeline.

Paths for data that lives inside this repository (``input/``, ``models/``,
``output/``) default to the repository root and can be overridden with
environment variables, which is useful for running the pipeline against a
different data location without touching the code.

The raw dispersion-curve and piezometer data consumed by
:mod:`seismic_hydro_ml.data_prep` are internal SNCF Reseau datasets that are
not distributed with this repository. Their locations have no sensible
in-repo default and must be provided explicitly, either via the
``SHML_RAW_VR_DIR`` / ``SHML_RAW_PIEZO_DIR`` environment variables or via the
``--raw-vr-dir`` / ``--raw-piezo-dir`` command line options.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var_name: str, default: Path) -> Path:
    value = os.environ.get(var_name)
    return Path(value).expanduser().resolve() if value else default


INPUT_DIR = _env_path("SHML_INPUT_DIR", REPO_ROOT / "input")
MODELS_DIR = _env_path("SHML_MODELS_DIR", REPO_ROOT / "models")
OUTPUT_DIR = _env_path("SHML_OUTPUT_DIR", REPO_ROOT / "output")

# Raw data directories used only by the data-preparation step (see
# `seismic_hydro_ml.data_prep`). `None` means "not configured".
RAW_VR_DIR = os.environ.get("SHML_RAW_VR_DIR")
RAW_PIEZO_DIR = os.environ.get("SHML_RAW_PIEZO_DIR")


def ensure_data_dirs() -> None:
    """Create the in-repo data directories if they do not already exist."""
    for directory in (INPUT_DIR, MODELS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
