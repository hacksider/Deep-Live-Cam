"""Shared path constants for the Deep-Live-Cam project."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LIBRARY_DIR = os.path.join(ROOT_DIR, "library")
LIBRARY_SOURCES_DIR = os.path.join(LIBRARY_DIR, "sources")
LIBRARY_BACKGROUNDS_DIR = os.path.join(LIBRARY_DIR, "backgrounds")


def ensure_library_dirs() -> None:
    """Ensure in-app image library directories exist."""
    os.makedirs(LIBRARY_SOURCES_DIR, exist_ok=True)
    os.makedirs(LIBRARY_BACKGROUNDS_DIR, exist_ok=True)
