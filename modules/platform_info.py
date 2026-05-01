"""Centralized platform + accelerator detection.

Imported once at startup to expose typed flags the rest of the codebase
can branch on without re-querying `platform`, `torch.cuda`, or
`onnxruntime.get_available_providers()` repeatedly.

The banner printed by :func:`print_banner` is the single user-facing
report of which code path the app will take.
"""
from __future__ import annotations

import platform as _platform
import sys
from typing import List, Tuple

IS_WINDOWS: bool = _platform.system() == "Windows"
IS_MACOS: bool = _platform.system() == "Darwin"
IS_LINUX: bool = _platform.system() == "Linux"
IS_APPLE_SILICON: bool = IS_MACOS and _platform.machine() == "arm64"


def _detect_torch_cuda() -> bool:
    try:
        import torch  # noqa: WPS433 — local import, avoid hard dep at module load
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_onnx_providers() -> List[str]:
    try:
        import onnxruntime
        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


HAS_TORCH_CUDA: bool = _detect_torch_cuda()
ONNX_PROVIDERS: List[str] = _detect_onnx_providers()
HAS_CUDA_PROVIDER: bool = "CUDAExecutionProvider" in ONNX_PROVIDERS
HAS_COREML_PROVIDER: bool = "CoreMLExecutionProvider" in ONNX_PROVIDERS
HAS_DML_PROVIDER: bool = "DmlExecutionProvider" in ONNX_PROVIDERS


def camera_backends() -> List[Tuple[int, int]]:
    """Return an ordered list of ``(device_index, cv2_backend)`` attempts.

    Windows prefers MSMF (60fps capable) with DirectShow as fallback.
    macOS/Linux use the default backend (AVFoundation / V4L2).
    """
    import cv2
    if IS_WINDOWS:
        return [
            (0, cv2.CAP_MSMF),
            (0, cv2.CAP_DSHOW),
            (0, cv2.CAP_ANY),
        ]
    return [(0, cv2.CAP_ANY)]


def accelerator_label() -> str:
    if HAS_TORCH_CUDA and HAS_CUDA_PROVIDER:
        return "CUDA (NVIDIA)"
    if IS_APPLE_SILICON and HAS_COREML_PROVIDER:
        return "CoreML (Apple Neural Engine)"
    if HAS_COREML_PROVIDER:
        return "CoreML"
    if HAS_DML_PROVIDER:
        return "DirectML"
    return "CPU"


def print_banner() -> None:
    """Print a one-line summary of the platform + accelerator selection."""
    os_label = f"{_platform.system()} {_platform.machine()}"
    print(
        f"[platform] {os_label} | python {sys.version.split()[0]} | "
        f"accelerator: {accelerator_label()} | providers: {ONNX_PROVIDERS}",
        flush=True,
    )
