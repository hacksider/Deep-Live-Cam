"""Shared path constants for the Deep-Live-Cam project.

Both model locations can be overridden with environment variables so that
cloud deployments (Modal, RunPod, Docker, ...) can point them at a
persistent volume instead of the ephemeral container filesystem:

    DLC_MODELS_DIR        swapper / enhancer ONNX models (default: <repo>/models)
    DLC_INSIGHTFACE_ROOT  insightface model root (default: ~/.insightface)
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(name: str, default: str) -> str:
    value = os.environ.get(name)
    return os.path.abspath(os.path.expanduser(value)) if value else default


MODELS_DIR = _env_path("DLC_MODELS_DIR", os.path.join(ROOT_DIR, "models"))
INSIGHTFACE_ROOT = _env_path(
    "DLC_INSIGHTFACE_ROOT", os.path.join(os.path.expanduser("~"), ".insightface")
)
